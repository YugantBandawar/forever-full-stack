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
from flask_cors import CORS
import requests
import time
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv( dotenv_path="AI/.env" )

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Limit upload size to 16MB
app.config['JSON_SORT_KEYS'] = False  # Disable sorting of JSON keys

# Create upload folder if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Set up MediaPipe
mp_face_mesh = mp.solutions.face_mesh
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    min_detection_confidence=0.5
)

face_detector = mp_face_detection.FaceDetection(
    model_selection=1,  # 0 for short-range, 1 for full-range detection
    min_detection_confidence=0.5
)

feature_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Load fashion database
with open('AI/fashion_database.json', 'r') as f:
    fashion_db = json.load(f)

# Get API keys from environment variables
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

# Default to Groq if both are set
USE_GROQ = GROQ_API_KEY is not None
USE_GEMINI = GEMINI_API_KEY is not None and not USE_GROQ

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_dominant_colors(img, n_colors=3):
    """Extract dominant colors from an image"""
    pixels = img.reshape(-1, 3)
    
    clt = KMeans(n_clusters=n_colors)
    clt.fit(pixels)
    
    return clt.cluster_centers_

def detect_face_shape(landmarks, img_height, img_width):
    """Determine face shape based on facial landmarks using MediaPipe"""
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
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    
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
    threshold = np.max(profile) * 0.2 if np.max(profile) > 0 else 0
    significant_edges = profile > threshold
    
    if np.any(significant_edges):
        edge_indices = np.where(significant_edges)[0]
        left_edge = np.min(edge_indices)
        right_edge = np.max(edge_indices)
        return right_edge - left_edge
    else:
        return 0

def analyze_skin_tone(face_img):
    """Analyze skin tone from face image"""
    ycrcb = cv2.cvtColor(face_img, cv2.COLOR_BGR2YCrCb)
    
    lower = np.array([0, 133, 77], dtype=np.uint8)
    upper = np.array([255, 173, 127], dtype=np.uint8)
    mask = cv2.inRange(ycrcb, lower, upper)
    
    skin = cv2.bitwise_and(face_img, face_img, mask=mask)
    
    skin_pixels = skin[mask > 0]
    if len(skin_pixels) > 0:
        avg_color = np.mean(skin_pixels, axis=0)
        
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
    style_keywords = []
    occasion_keywords = []
    color_keywords = []
    
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

def call_groq_api(prompt):
    """Call Groq API for enhanced fashion recommendations"""
    if not GROQ_API_KEY:
        return {"error": "Groq API key not configured"}
    
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "llama3-8b-8192",  # Using Llama 3 model
        "messages": [
            {
                "role": "system", 
                "content": "You are a fashion expert providing detailed fashion recommendations. Respond with JSON format containing detailed outfit descriptions, styling tips, and fashion trends."
            },
            {
                "role": "user", 
                "content": prompt
            }
        ],
        "temperature": 0.7,
        "max_tokens": 1024
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        print(f"Groq API response: {response.json()}")
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return {"error": f"Groq API error: {str(e)}"}

def call_gemini_api(prompt):
    """Call Google Gemini API for enhanced fashion recommendations"""
    if not GEMINI_API_KEY:
        return {"error": "Gemini API key not configured"}
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={GEMINI_API_KEY}"
    headers = {
        "Content-Type": "application/json"
    }
    
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "text": prompt
                    }
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.7,
            "maxOutputTokens": 1024
        }
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        return {"error": f"Gemini API error: {str(e)}"}

def enhance_recommendations_with_ai(base_recommendations, analysis, prompt_data):
    """Use AI to enhance fashion recommendations"""
    ai_prompt = f"""
    Please provide enhanced fashion recommendations based on the following information:
    
    Body Shape: {analysis['body_shape']}
    Face Shape: {analysis['face_shape']}
    Skin Tone: {analysis['skin_tone']}
    Style Preferences: {', '.join(prompt_data['style'])}
    Occasion: {', '.join(prompt_data['occasion'])}
    Color Preferences: {', '.join(prompt_data['colors'] if prompt_data['colors'] else ['any'])}
    
    Based on this analysis, provide:
    1. 2-3 complete outfit recommendations with detailed descriptions
    2. Specific style tips for this body shape, face shape, and skin tone combination
    3. Current fashion trends that would suit this individual
    4. Accessory recommendations
    5. Shopping tips (what to look for when buying items)
    
    Format your response as a JSON object with these sections.
    """
    
    try:
        if USE_GROQ:
            ai_response = call_groq_api(ai_prompt)
        elif USE_GEMINI:
            ai_response = call_gemini_api(ai_prompt)
        else:
            return base_recommendations
        
        # Try to parse the response as JSON
        try:
            if isinstance(ai_response, str):
                # Extract JSON if the response is wrapped in markdown code blocks
                if "```json" in ai_response:
                    json_start = ai_response.find("```json") + 7
                    json_end = ai_response.find("```", json_start)
                    ai_response = ai_response[json_start:json_end].strip()
                ai_recommendations = json.loads(ai_response)
            else:
                ai_recommendations = ai_response
                
            # Merge AI recommendations with base recommendations
            enhanced_recommendations = base_recommendations.copy()
            
            # Add AI-enhanced sections
            if "outfit_recommendations" in ai_recommendations:
                enhanced_recommendations["ai_outfit_recommendations"] = ai_recommendations["outfit_recommendations"]
            
            if "style_tips" in ai_recommendations:
                enhanced_recommendations["ai_style_tips"] = ai_recommendations["style_tips"]
                
            if "fashion_trends" in ai_recommendations:
                enhanced_recommendations["fashion_trends"] = ai_recommendations["fashion_trends"]
                
            if "accessory_recommendations" in ai_recommendations:
                enhanced_recommendations["ai_accessory_recommendations"] = ai_recommendations["accessory_recommendations"]
                
            if "shopping_tips" in ai_recommendations:
                enhanced_recommendations["shopping_tips"] = ai_recommendations["shopping_tips"]
                
            return enhanced_recommendations
        except json.JSONDecodeError:
            # If JSON parsing fails, try to extract structured information from text
            enhanced_recommendations = base_recommendations.copy()
            enhanced_recommendations["ai_fashion_advice"] = ai_response
            return enhanced_recommendations
            
    except Exception as e:
        print(f"AI enhancement error: {str(e)}")
        return base_recommendations

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
            # Detect body shape
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
                
                face_roi = img[y_min:y_max, x_min:x_max]
                if face_roi.size > 0:  # Check if face ROI is not empty
                    skin_tone = analyze_skin_tone(face_roi)
                
                face_mesh_results = face_mesh.process(img_rgb)
                if face_mesh_results.multi_face_landmarks:
                    face_landmarks = face_mesh_results.multi_face_landmarks[0].landmark
                    face_shape = detect_face_shape(face_landmarks, height, width)
            
            # Get base recommendations
            base_recommendations = generate_recommendations(body_shape, face_shape, skin_tone, prompt_data)
            
            # Analysis data
            analysis_data = {
                'body_shape': body_shape,
                'face_shape': face_shape,
                'skin_tone': skin_tone,
                'style_preferences': prompt_data
            }
            
            # Enhance recommendations with AI if API keys are available
            if USE_GROQ or USE_GEMINI:
                enhanced_recommendations = enhance_recommendations_with_ai(
                    base_recommendations, analysis_data, prompt_data
                )
                ai_provider = "Groq" if USE_GROQ else "Gemini"
            else:
                enhanced_recommendations = base_recommendations
                ai_provider = None
            
            response = {
                'analysis': analysis_data,
                'recommendations': enhanced_recommendations,
                'ai_enhanced': ai_provider is not None,
                'ai_provider': ai_provider
            }
            
            return jsonify(response)
            
        except Exception as e:
            return jsonify({'error': f'Analysis failed: {str(e)}'}), 500
    
    return jsonify({'error': 'Invalid file format'}), 400

@app.route('/api-status', methods=['GET'])
def api_status():
    """Check the status of connected AI APIs"""
    status = {
        "groq": "available" if USE_GROQ else "not configured",
        "gemini": "available" if USE_GEMINI else "not configured"
    }
    return jsonify(status)

if __name__ == '__main__':
    print("Enhanced Fashion Recommendation System starting...")
    print("NOTE: This implementation requires the following packages:")
    print("  - flask, opencv-python, numpy, mediapipe, tensorflow, scikit-learn, requests, python-dotenv")
    
    if USE_GROQ:
        print("Using Groq AI API for enhanced recommendations")
    elif USE_GEMINI:
        print("Using Google Gemini AI API for enhanced recommendations")
    else:
        print("No AI API configured. Set GROQ_API_KEY or GEMINI_API_KEY in .env file for enhanced recommendations")
    
    app.run(host='0.0.0.0', port=5000, debug=True)