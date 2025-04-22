# Replace these imports at the top of the file
import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template
import mediapipe as mp # type: ignore
from tensorflow.keras.applications import MobileNetV2 # type: ignore
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore
from sklearn.cluster import KMeans
import json
import random
import os
from flask_cors import CORS  # Import Flask-CORS

app = Flask(__name__)

# Enable CORS for the Flask app
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create uploads directory if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Initialize MediaPipe solutions instead of dlib
mp_face_mesh = mp.solutions.face_mesh
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Use MediaPipe Face Mesh for better facial landmark detection
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    min_detection_confidence=0.5
)

# Use MediaPipe Face Detection for face detection
face_detector = mp_face_detection.FaceDetection(
    model_selection=1,  # 0 for short-range, 1 for full-range detection
    min_detection_confidence=0.5
)

# Load feature model (unchanged)
feature_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Load fashion database
with open('fashion_database.json', 'r') as f:
    fashion_db = json.load(f)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_dominant_colors(img, n_colors=3):
    """Extract dominant colors from an image"""
    # Reshape image to be a list of pixels
    pixels = img.reshape(-1, 3)
    
    # Cluster the pixel intensities
    clt = KMeans(n_clusters=n_colors)
    clt.fit(pixels)
    
    # Return cluster centers as colors
    return clt.cluster_centers_

def detect_face_shape(landmarks, img_height, img_width):
    """Determine face shape based on facial landmarks using MediaPipe"""
    # Convert relative coordinates to absolute pixel coordinates
    landmarks_points = []
    for landmark in landmarks:
        x = int(landmark.x * img_width)
        y = int(landmark.y * img_height)
        landmarks_points.append((x, y))
    
    # Define key landmark indices for MediaPipe Face Mesh
    # These indices correspond to relevant facial points in MediaPipe's 468 landmarks
    
    # Face outline points
    chin = landmarks_points[152]  # Bottom of chin
    left_temple = landmarks_points[162]  # Left temple
    right_temple = landmarks_points[389]  # Right temple
    
    # Forehead point
    forehead_mid = landmarks_points[10]  # Middle of forehead
    
    # Cheekbone points
    left_cheekbone = landmarks_points[123]  # Left cheekbone
    right_cheekbone = landmarks_points[352]  # Right cheekbone
    
    # Jawline points
    left_jaw = landmarks_points[206]  # Left jaw
    right_jaw = landmarks_points[426]  # Right jaw
    
    # Calculate key measurements
    face_width = right_temple[0] - left_temple[0]
    face_height = chin[1] - forehead_mid[1]
    cheekbone_width = right_cheekbone[0] - left_cheekbone[0]
    jawline_width = right_jaw[0] - left_jaw[0]
    
    # Determine face shape based on measurements
    if face_height > 1.5 * face_width:
        return "oblong"
    elif jawline_width > 0.78 * cheekbone_width:
        return "square"
    elif cheekbone_width > face_width and cheekbone_width > jawline_width:
        return "diamond"
    elif forehead_mid[1] < left_temple[1] and jawline_width < 0.8 * cheekbone_width:
        return "heart"
    elif face_width > 0.8 * face_height and jawline_width > 0.8 * cheekbone_width:
        return "round"
    else:
        return "oval"  # Most common face shape, used as default


def detect_body_shape(img):
    """Determine body shape from full body image using improved edge detection"""
    height, width = img.shape[:2]
    
    # Convert to grayscale for better edge detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Edge detection
    edges = cv2.Canny(blurred, 50, 150)
    
    # Define body regions with better proportions
    shoulder_region = edges[int(height*0.15):int(height*0.3), :]
    waist_region = edges[int(height*0.4):int(height*0.5), :]
    hip_region = edges[int(height*0.5):int(height*0.65), :]
    
    # Calculate region widths based on edge detection
    # This provides more accurate measurements than the simple pixel counting
    shoulder_profile = np.sum(shoulder_region, axis=0)
    waist_profile = np.sum(waist_region, axis=0)
    hip_profile = np.sum(hip_region, axis=0)
    
    # Find the widest parts using peak detection
    shoulder_width = get_region_width(shoulder_profile)
    waist_width = get_region_width(waist_profile)
    hip_width = get_region_width(hip_profile)
    
    # Determine body shape based on width ratios with improved thresholds
    if shoulder_width > hip_width * 1.05 and shoulder_width > waist_width * 1.25:
        return "inverted triangle"
    elif hip_width > shoulder_width * 1.05 and hip_width > waist_width * 1.25:
        return "pear"
    elif shoulder_width > waist_width * 1.15 and hip_width > waist_width * 1.15:
        if abs(shoulder_width - hip_width) < shoulder_width * 0.1:
            return "hourglass"
        else:
            return "pear" if hip_width > shoulder_width else "inverted triangle"
    elif abs(shoulder_width - hip_width) < shoulder_width * 0.1 and abs(shoulder_width - waist_width) < waist_width * 0.2:
        return "rectangle"
    else:
        return "oval"

def get_region_width(profile):
    """Calculate width of body region from edge profile"""
    # Filter small edges to remove noise
    threshold = np.max(profile) * 0.2 if np.max(profile) > 0 else 0
    significant_edges = profile > threshold
    
    # Find leftmost and rightmost significant edges
    if np.any(significant_edges):
        edge_indices = np.where(significant_edges)[0]
        left_edge = np.min(edge_indices)
        right_edge = np.max(edge_indices)
        return right_edge - left_edge
    else:
        return 0

def analyze_skin_tone(face_img):
    """Analyze skin tone from face image"""
    # Convert to YCrCb color space which better separates skin colors
    ycrcb = cv2.cvtColor(face_img, cv2.COLOR_BGR2YCrCb)
    
    # Create a binary mask of skin pixels
    lower = np.array([0, 133, 77], dtype=np.uint8)
    upper = np.array([255, 173, 127], dtype=np.uint8)
    mask = cv2.inRange(ycrcb, lower, upper)
    
    # Apply mask to get only skin pixels
    skin = cv2.bitwise_and(face_img, face_img, mask=mask)
    
    # Get dominant color in skin area
    skin_pixels = skin[mask > 0]
    if len(skin_pixels) > 0:
        avg_color = np.mean(skin_pixels, axis=0)
        
        # Classify skin tone based on RGB values
        if avg_color[2] > 200:
            return "very fair"
        elif avg_color[2] > 170:
            return "fair"
        elif avg_color[2] > 140:
            return "medium"
        elif avg_color[2] > 110:
            return "olive"
        elif avg_color[2] > 80:
            return "brown"
        else:
            return "dark"
    else:
        return "medium"  # Default if no skin detected

def process_prompt(prompt):
    """Extract style keywords from user prompt"""
    # In a real system, use NLP for better keyword extraction
    style_keywords = []
    occasion_keywords = []
    color_keywords = []
    
    # Define keyword lists
    style_list = ["casual", "formal", "business", "party", "vintage", "boho", "minimalist", 
                  "streetwear", "preppy", "athletic", "elegant", "trendy", "classic"]
    occasion_list = ["work", "date", "wedding", "interview", "gym", "everyday", "beach", 
                     "concert", "dinner", "office", "weekend", "night out"]
    color_list = ["red", "blue", "green", "black", "white", "pink", "purple", "yellow", 
                  "orange", "brown", "gray", "navy", "beige", "teal"]
    
    # Simple keyword extraction
    prompt_lower = prompt.lower()
    for word in style_list:
        if word in prompt_lower:
            style_keywords.append(word)
    
    for word in occasion_list:
        if word in prompt_lower:
            occasion_keywords.append(word)
    
    for word in color_list:
        if word in prompt_lower:
            color_keywords.append(word)
    
    # Default values if nothing detected
    if not style_keywords:
        style_keywords = ["casual"]
    if not occasion_keywords:
        occasion_keywords = ["everyday"]
    
    return {
        "style": style_keywords,
        "occasion": occasion_keywords,
        "colors": color_keywords
    }

def generate_recommendations(body_shape, face_shape, skin_tone, prompt_data):
    """Generate fashion recommendations based on analysis"""
    recommendations = {
        "tops": [],
        "bottoms": [],
        "dresses": [],
        "accessories": [],
        "complete_outfits": []
    }
    
    # Filter fashion database based on body shape
    body_shape_items = [item for item in fashion_db["items"] 
                      if "suitable_body_shapes" in item and body_shape in item["suitable_body_shapes"]]
    
    # Filter by style and occasion
    filtered_items = []
    for item in body_shape_items:
        style_match = any(style in item.get("style", []) for style in prompt_data["style"])
        occasion_match = any(occ in item.get("occasion", []) for occ in prompt_data["occasion"])
        if style_match and occasion_match:
            filtered_items.append(item)
    
    # If too few items, relax constraints
    if len(filtered_items) < 10:
        filtered_items = body_shape_items
    
    # Color considerations for face shape and skin tone
    color_recommendations = []
    if skin_tone in ["fair", "very fair"]:
        color_recommendations.extend(["navy", "burgundy", "forest green"])
    elif skin_tone in ["medium", "olive"]:
        color_recommendations.extend(["cobalt blue", "emerald", "purple"])
    else:  # brown or dark
        color_recommendations.extend(["orange", "golden yellow", "bright red"])
    
    # Add any user-requested colors
    color_recommendations.extend(prompt_data["colors"])
    
    # Categorize recommendations
    for item in filtered_items:
        if item["category"] == "top":
            recommendations["tops"].append(item)
        elif item["category"] == "bottom":
            recommendations["bottoms"].append(item)
        elif item["category"] == "dress":
            recommendations["dresses"].append(item)
        elif item["category"] == "accessory":
            recommendations["accessories"].append(item)
    
    # Generate complete outfits
    outfits_count = min(3, len(recommendations["tops"]), len(recommendations["bottoms"]))
    for i in range(outfits_count):
        top = random.choice(recommendations["tops"]) if recommendations["tops"] else None
        bottom = random.choice(recommendations["bottoms"]) if recommendations["bottoms"] else None
        accessory = random.choice(recommendations["accessories"]) if recommendations["accessories"] else None
        
        if top and bottom:
            outfit = {
                "top": top["name"],
                "bottom": bottom["name"],
                "accessory": accessory["name"] if accessory else "No accessory recommendation",
                "description": f"A {top['style'][0] if 'style' in top and top['style'] else 'versatile'} {top['name']} paired with {bottom['name']}."
            }
            recommendations["complete_outfits"].append(outfit)
    
    # Add dress outfit if available
    if recommendations["dresses"]:
        dress = random.choice(recommendations["dresses"])
        accessory = random.choice(recommendations["accessories"]) if recommendations["accessories"] else None
        outfit = {
            "dress": dress["name"],
            "accessory": accessory["name"] if accessory else "No accessory recommendation",
            "description": f"A {dress['style'][0] if 'style' in dress and dress['style'] else 'versatile'} {dress['name']} dress."
        }
        recommendations["complete_outfits"].append(outfit)
    
    # Add recommendations based on face shape
    face_recommendations = {
        "oval": "Your oval face shape is versatile for most styles.",
        "round": "Frame your round face with v-necks and angular accessories.",
        "square": "Soften your square jawline with round necklines and soft fabrics.",
        "heart": "Balance your heart-shaped face with wider bottoms and boat necks.",
        "oblong": "Choose horizontal lines and round necklines to balance your oblong face.",
        "diamond": "Highlight your cheekbones with boat necklines and statement earrings."
    }
    
    # Add face shape recommendation
    if face_shape in face_recommendations:
        recommendations["face_shape_advice"] = face_recommendations[face_shape]
    else:
        recommendations["face_shape_advice"] = "Your face shape complements many styles."
    
    # Add color recommendations
    recommendations["color_recommendations"] = color_recommendations
    
    return recommendations

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400
    
    if file and allowed_file(file.filename):
        # Save the uploaded image
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        
        # Get user prompt
        prompt = request.form.get('prompt', '')
        prompt_data = process_prompt(prompt)
        
        # Read image
        img = cv2.imread(filepath)
        if img is None:
            return jsonify({'error': 'Could not read image'}), 400
        
        # Convert to RGB for MediaPipe
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width = img_rgb.shape[:2]
        
        # Analyze image
        try:
            # Detect body shape (unchanged)
            body_shape = detect_body_shape(img)
            
            # Use MediaPipe Face Detection to detect faces
            face_detection_results = face_detector.process(img_rgb)
            face_shape = "oval"  # Default if no face detected
            skin_tone = "medium"  # Default if no face detected
            
            if face_detection_results.detections:
                # Get the first face detected
                detection = face_detection_results.detections[0]
                
                # Extract bounding box for the face
                bbox = detection.location_data.relative_bounding_box
                x_min = int(bbox.xmin * width)
                y_min = int(bbox.ymin * height)
                x_width = int(bbox.width * width)
                y_height = int(bbox.height * height)
                
                # Ensure coordinates are within image bounds
                x_min = max(0, x_min)
                y_min = max(0, y_min)
                x_max = min(width, x_min + x_width)
                y_max = min(height, y_min + y_height)
                
                # Extract face region for skin tone analysis
                face_roi = img[y_min:y_max, x_min:x_max]
                if face_roi.size > 0:  # Check if face ROI is not empty
                    skin_tone = analyze_skin_tone(face_roi)
                
                # Use MediaPipe Face Mesh for facial landmarks
                face_mesh_results = face_mesh.process(img_rgb)
                if face_mesh_results.multi_face_landmarks:
                    # Get landmarks for the first face
                    face_landmarks = face_mesh_results.multi_face_landmarks[0].landmark
                    face_shape = detect_face_shape(face_landmarks, height, width)
            
            # Generate recommendations (unchanged)
            recommendations = generate_recommendations(body_shape, face_shape, skin_tone, prompt_data)
            
            # Prepare response
            response = {
                'analysis': {
                    'body_shape': body_shape,
                    'face_shape': face_shape,
                    'skin_tone': skin_tone,
                    'style_preferences': prompt_data
                },
                'recommendations': recommendations
            }
            
            return jsonify(response)
            
        except Exception as e:
            return jsonify({'error': f'Analysis failed: {str(e)}'}), 500
    
    return jsonify({'error': 'Invalid file format'}), 400

# Create a dummy fashion database for testing
def create_dummy_database():
    fashion_db = {
        "items": [
            {
                "name": "V-neck T-shirt",
                "category": "top",
                "style": ["casual", "minimalist"],
                "occasion": ["everyday", "weekend"],
                "suitable_body_shapes": ["inverted triangle", "hourglass", "rectangle", "oval"]
            },
            {
                "name": "Boat neck blouse",
                "category": "top",
                "style": ["elegant", "formal"],
                "occasion": ["work", "dinner"],
                "suitable_body_shapes": ["pear", "hourglass", "oval"]
            },
            {
                "name": "Button-down shirt",
                "category": "top",
                "style": ["business", "classic"],
                "occasion": ["work", "interview"],
                "suitable_body_shapes": ["all"]
            },
            {
                "name": "Wrap top",
                "category": "top",
                "style": ["elegant", "formal"],
                "occasion": ["work", "dinner", "date"],
                "suitable_body_shapes": ["hourglass", "rectangle", "oval", "inverted triangle"]
            },
            {
                "name": "Straight leg jeans",
                "category": "bottom",
                "style": ["casual", "classic"],
                "occasion": ["everyday", "weekend"],
                "suitable_body_shapes": ["hourglass", "rectangle"]
            },
            {
                "name": "A-line skirt",
                "category": "bottom",
                "style": ["elegant", "classic"],
                "occasion": ["work", "dinner"],
                "suitable_body_shapes": ["pear", "hourglass", "oval"]
            },
            {
                "name": "Tailored trousers",
                "category": "bottom",
                "style": ["business", "formal"],
                "occasion": ["work", "interview"],
                "suitable_body_shapes": ["hourglass", "rectangle", "inverted triangle"]
            },
            {
                "name": "Wide-leg pants",
                "category": "bottom",
                "style": ["trendy", "boho"],
                "occasion": ["everyday", "weekend", "beach"],
                "suitable_body_shapes": ["inverted triangle", "rectangle", "hourglass"]
            },
            {
                "name": "Wrap dress",
                "category": "dress",
                "style": ["elegant", "classic"],
                "occasion": ["work", "dinner", "date"],
                "suitable_body_shapes": ["all"]
            },
            {
                "name": "A-line dress",
                "category": "dress",
                "style": ["elegant", "formal"],
                "occasion": ["wedding", "formal event"],
                "suitable_body_shapes": ["pear", "oval", "rectangle"]
            },
            {
                "name": "Shift dress",
                "category": "dress",
                "style": ["minimalist", "business"],
                "occasion": ["work", "interview"],
                "suitable_body_shapes": ["inverted triangle", "rectangle"]
            },
            {
                "name": "Statement necklace",
                "category": "accessory",
                "style": ["trendy", "elegant"],
                "occasion": ["dinner", "party", "date"],
                "suitable_body_shapes": ["all"]
            },
            {
                "name": "Structured handbag",
                "category": "accessory",
                "style": ["business", "classic"],
                "occasion": ["work", "interview"],
                "suitable_body_shapes": ["all"]
            },
            {
                "name": "Hoop earrings",
                "category": "accessory",
                "style": ["casual", "trendy"],
                "occasion": ["everyday", "weekend", "party"],
                "suitable_body_shapes": ["all"]
            }
        ]
    }
    
    # Save to file
    with open('fashion_database.json', 'w') as f:
        json.dump(fashion_db, f, indent=2)

# Create HTML template file
def create_html_template():
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Fashion Recommendation System</title>
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                margin: 0;
                padding: 20px;
                background-color: #f8f9fa;
                color: #333;
            }
            .container {
                max-width: 800px;
                margin: 0 auto;
                background-color: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }
            h1 {
                text-align: center;
                color: #6c63ff;
                margin-bottom: 30px;
            }
            .upload-section {
                margin-bottom: 30px;
                padding: 20px;
                border: 2px dashed #ccc;
                border-radius: 5px;
                text-align: center;
            }
            .form-group {
                margin-bottom: 20px;
            }
            label {
                display: block;
                margin-bottom: 8px;
                font-weight: 600;
            }
            input[type="text"], textarea {
                width: 100%;
                padding: 10px;
                border: 1px solid #ddd;
                border-radius: 4px;
                box-sizing: border-box;
            }
            input[type="file"] {
                margin-top: 10px;
            }
            button {
                background-color: #6c63ff;
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 4px;
                cursor: pointer;
                font-size: 16px;
                display: block;
                margin: 20px auto 0;
                transition: background-color 0.3s;
            }
            button:hover {
                background-color: #5a52d5;
            }
            #preview {
                max-width: 300px;
                max-height: 300px;
                margin: 20px auto;
                display: none;
            }
            #results {
                display: none;
                margin-top: 30px;
                padding: 20px;
                background-color: #f0f0f0;
                border-radius: 5px;
            }
            .outfit {
                background-color: white;
                padding: 15px;
                margin-bottom: 15px;
                border-radius: 5px;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
            }
            .loader {
                border: 5px solid #f3f3f3;
                border-top: 5px solid #6c63ff;
                border-radius: 50%;
                width: 50px;
                height: 50px;
                animation: spin 2s linear infinite;
                margin: 20px auto;
                display: none;
            }
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Personal Fashion Stylist</h1>
            
            <div class="upload-section">
                <form id="upload-form" enctype="multipart/form-data">
                    <div class="form-group">
                        <label for="image">Upload your photo:</label>
                        <input type="file" id="image" name="image" accept="image/*" onchange="previewImage(event)">
                        <img id="preview" src="#" alt="Preview">
                    </div>
                    
                    <div class="form-group">
                        <label for="prompt">What are you looking for? (e.g., "casual outfit for weekend" or "business attire for interview")</label>
                        <textarea id="prompt" name="prompt" rows="3" placeholder="Describe your style preferences, occasion, color preferences..."></textarea>
                    </div>
                    
                    <button type="submit">Get Recommendations</button>
                </form>
            </div>
            
            <div class="loader" id="loader"></div>
            
            <div id="results">
                <h2>Your Fashion Analysis</h2>
                <div id="analysis-results"></div>
                
                <h2>Outfit Recommendations</h2>
                <div id="outfit-results"></div>
                
                <h2>Style Tips</h2>
                <div id="style-tips"></div>
            </div>
        </div>
        
        <script>
            function previewImage(event) {
                const preview = document.getElementById('preview');
                preview.src = URL.createObjectURL(event.target.files[0]);
                preview.style.display = 'block';
            }
            
            document.getElementById('upload-form').addEventListener('submit', async function(e) {
                e.preventDefault();
                
                // Show loader
                document.getElementById('loader').style.display = 'block';
                document.getElementById('results').style.display = 'none';
                
                const formData = new FormData(this);
                
                try {
                    const response = await fetch('/analyze', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const data = await response.json();
                    
                    if (response.ok) {
                        displayResults(data);
                    } else {
                        alert('Error: ' + data.error);
                    }
                } catch (error) {
                    alert('An error occurred. Please try again.');
                    console.error(error);
                }
                
                // Hide loader
                document.getElementById('loader').style.display = 'none';
            });
            
            function displayResults(data) {
                const results = document.getElementById('results');
                const analysisResults = document.getElementById('analysis-results');
                const outfitResults = document.getElementById('outfit-results');
                const styleTips = document.getElementById('style-tips');
                
                // Display analysis results
                analysisResults.innerHTML = `
                    <p><strong>Body Shape:</strong> ${data.analysis.body_shape}</p>
                    <p><strong>Face Shape:</strong> ${data.analysis.face_shape}</p>
                    <p><strong>Skin Tone:</strong> ${data.analysis.skin_tone}</p>
                    <p><strong>Style Preferences:</strong> ${data.analysis.style_preferences.style.join(', ')}</p>
                    <p><strong>Occasion:</strong> ${data.analysis.style_preferences.occasion.join(', ')}</p>
                `;
                
                // Display outfit recommendations
                outfitResults.innerHTML = '';
                if (data.recommendations.complete_outfits.length > 0) {
                    data.recommendations.complete_outfits.forEach((outfit, index) => {
                        let outfitHtml = `<div class="outfit">
                            <h3>Outfit ${index + 1}</h3>
                            <p>${outfit.description}</p>
                            <ul>`;
                        
                        if (outfit.top) outfitHtml += `<li><strong>Top:</strong> ${outfit.top}</li>`;
                        if (outfit.bottom) outfitHtml += `<li><strong>Bottom:</strong> ${outfit.bottom}</li>`;
                        if (outfit.dress) outfitHtml += `<li><strong>Dress:</strong> ${outfit.dress}</li>`;
                        if (outfit.accessory) outfitHtml += `<li><strong>Accessory:</strong> ${outfit.accessory}</li>`;
                        
                        outfitHtml += `</ul></div>`;
                        outfitResults.innerHTML += outfitHtml;
                    });
                } else {
                    outfitResults.innerHTML = '<p>No complete outfits could be generated based on your preferences.</p>';
                }
                
                // Display style tips
                styleTips.innerHTML = `
                    <p>${data.recommendations.face_shape_advice}</p>
                    <p><strong>Recommended Colors:</strong> ${data.recommendations.color_recommendations.join(', ')}</p>
                `;
                
                // Show results section
                results.style.display = 'block';
            }
        </script>
    </body>
    </html>
    """
    
    with open('templates/index.html', 'w') as f:
        f.write(html_content)

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    if not os.path.exists('templates'):
        os.makedirs('templates')
    
    # Create dummy database and HTML template
    create_dummy_database()
    create_html_template()
    
    print("Fashion Recommendation System starting...")
    print("NOTE: This implementation requires the following packages:")
    print("  - flask, opencv-python, numpy, face_recognition, dlib, tensorflow, scikit-learn")
    print("  - You'll also need to download the shape_predictor_68_face_landmarks.dat file from dlib")
    
    app.run(host='0.0.0.0', port=5000, debug=True)
