from flask import Flask, request, jsonify
from pymongo import MongoClient
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global variables to store the model data
df = None
features = None
similarity_matrix = None
encoder = None
scaler = None
tfidf = None

# Connect to MongoDB
def connect_to_mongodb():
    try:
        client = MongoClient("mongodb+srv://yabandawar:yugant000000@cluster0.nutzzob.mongodb.net/e-commerce")
        db = client.get_database('e-commerce')
        collection = db['products']
        return collection
    except Exception as e:
        print(f"Error connecting to MongoDB: {e}")
        return None

# Load data from MongoDB
def load_data_from_mongodb():
    collection = connect_to_mongodb()
    if collection is None:
        return None
    
    mongo_data = list(collection.find({}))
    
    if not mongo_data:
        print("No data found in MongoDB collection")
        return pd.DataFrame()
    
    processed_data = []
    for item in mongo_data:
        try:
            product = {
                'id': str(item['_id']),
                'name': item['name'],
                'description': item['description'],
                'price': item['price'],
                'category': item['category'],
                'subCategory': item['subCategory'],
                'sizes': ','.join(item['sizes']),
                'bestseller': item['bestseller'],
                'image_count': len(item['image']) if 'image' in item and item['image'] else 0,
                'image_url': item['image'][0] if 'image' in item and item['image'] else ''
            }
            processed_data.append(product)
        except KeyError as e:
            print(f"Skipping record due to missing field: {e}")
            continue
    
    return pd.DataFrame(processed_data)

# Feature Engineering
def engineer_features(df):
    tfidf = TfidfVectorizer(max_features=100, stop_words='english')
    description_features = tfidf.fit_transform(df['description'].fillna(''))
    
    categorical_features = ['category', 'subCategory', 'sizes']
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_features = encoder.fit_transform(df[categorical_features].fillna(''))
    
    numerical_features = ['price', 'image_count']
    scaler = StandardScaler()
    scaled_numerical = scaler.fit_transform(df[numerical_features])
    
    all_features = np.hstack((
        scaled_numerical,
        encoded_features,
        description_features.toarray(),
        df[['bestseller']].astype(int).values
    ))
    
    return all_features, encoder, scaler, tfidf

# Compute similarity matrix
def compute_similarity(features):
    similarity_matrix = cosine_similarity(features)
    return similarity_matrix

# Get similar products
def get_similar_products(product_id, df, similarity_matrix, top_n=5):
    if product_id not in df['id'].values:
        return {"error": "Product ID not found"}
    
    idx = df[df['id'] == product_id].index[0]
    similarity_scores = list(enumerate(similarity_matrix[idx]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    similarity_scores = similarity_scores[1:top_n+1]
    
    product_indices = [i[0] for i in similarity_scores]
    similarity_values = [i[1] for i in similarity_scores]
    
    recommendations = df.iloc[product_indices][['id', 'name', 'category', 'price', 'image_url']].copy()
    recommendations['similarity_score'] = similarity_values
    
    return recommendations.to_dict('records')

# Get personalized recommendations
def get_personalized_recommendations(user_history, df, similarity_matrix, user_preferences=None, top_n=5):
    if not user_history or not all(pid in df['id'].values for pid in user_history):
        return {"error": "Invalid user history"}
    
    indices = [df[df['id'] == pid].index[0] for pid in user_history]
    scores = np.zeros(len(df))
    
    for idx in indices:
        scores += similarity_matrix[idx]
    
    if user_preferences:
        preference_weights = {
            'category': 0.3,
            'price_range': 0.2,
            'size': 0.1
        }
        
        if 'category' in user_preferences:
            preferred_cats = user_preferences['category']
            for cat in preferred_cats:
                cat_indices = df[df['category'] == cat].index
                scores[cat_indices] *= (1 + preference_weights['category'])
        
        if 'price_range' in user_preferences:
            min_price, max_price = user_preferences['price_range']
            price_indices = df[(df['price'] >= min_price) & (df['price'] <= max_price)].index
            scores[price_indices] *= (1 + preference_weights['price_range'])
        
        if 'size' in user_preferences:
            preferred_sizes = user_preferences['size']
            for size in preferred_sizes:
                size_indices = df[df['sizes'].str.contains(size)].index
                scores[size_indices] *= (1 + preference_weights['size'])
    
    scored_indices = list(enumerate(scores))
    scored_indices = sorted(scored_indices, key=lambda x: x[1], reverse=True)
    scored_indices = [(i, s) for i, s in scored_indices if i not in indices]
    
    top_indices = [i[0] for i in scored_indices[:top_n]]
    top_scores = [i[1] for i in scored_indices[:top_n]]
    
    recommendations = df.iloc[top_indices][['id', 'name', 'category', 'price', 'image_url']].copy()
    recommendations['recommendation_score'] = top_scores
    
    return recommendations.to_dict('records')

# Get trending items
def get_trending_items(df, top_n=5):
    trending = df.sort_values(by=['bestseller', 'price'], ascending=[False, True])
    return trending.head(top_n)[['id', 'name', 'category', 'price', 'image_url']].to_dict('records')

# Complete the look
def complete_the_look(product_id, df, similarity_matrix, top_n=3):
    if product_id not in df['id'].values:
        return {"error": "Product ID not found"}
    
    product = df[df['id'] == product_id].iloc[0]
    idx = df[df['id'] == product_id].index[0]
    
    similarity_scores = list(enumerate(similarity_matrix[idx]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    
    filtered_scores = []
    count = 0
    
    for i, score in similarity_scores:
        if count >= top_n:
            break
            
        if df.iloc[i]['subCategory'] != product['subCategory']:
            filtered_scores.append((i, score))
            count += 1
    
    product_indices = [i[0] for i in filtered_scores]
    similarity_values = [i[1] for i in filtered_scores]
    
    recommendations = df.iloc[product_indices][['id', 'name', 'category', 'subCategory', 'price', 'image_url']].copy()
    recommendations['similarity_score'] = similarity_values
    
    return recommendations.to_dict('records')

# API endpoint to get similar products
@app.route('/api/recommendations/similar/<product_id>', methods=['GET'])
def api_similar_products(product_id):
    global df, similarity_matrix
    
    if df is None or similarity_matrix is None:
        return jsonify({"error": "Recommendation system not initialized"}), 500
    
    top_n = int(request.args.get('top_n', 5))
    similar = get_similar_products(product_id, df, similarity_matrix, top_n)
    
    if isinstance(similar, dict) and 'error' in similar:
        return jsonify(similar), 404
    
    return jsonify({"similar_products": similar})

# API endpoint to get trending items
@app.route('/api/recommendations/trending', methods=['GET'])
def api_trending_items():
    global df
    
    if df is None:
        return jsonify({"error": "Recommendation system not initialized"}), 500
    
    top_n = int(request.args.get('top_n', 5))
    trending = get_trending_items(df, top_n)
    return jsonify({"trending_items": trending})

# API endpoint to get complete the look recommendations
@app.route('/api/recommendations/complete-look/<product_id>', methods=['GET'])
def api_complete_look(product_id):
    global df, similarity_matrix
    
    if df is None or similarity_matrix is None:
        return jsonify({"error": "Recommendation system not initialized"}), 500
    
    top_n = int(request.args.get('top_n', 3))
    recommendations = complete_the_look(product_id, df, similarity_matrix, top_n)
    
    if isinstance(recommendations, dict) and 'error' in recommendations:
        return jsonify(recommendations), 404
    
    return jsonify({"complete_the_look": recommendations})

# API endpoint to get personalized recommendations
@app.route('/api/recommendations/personalized', methods=['POST'])
def api_personalized_recommendations():
    global df, similarity_matrix
    
    if df is None or similarity_matrix is None:
        return jsonify({"error": "Recommendation system not initialized"}), 500
    
    data = request.get_json()
    
    if not data or 'user_history' not in data:
        return jsonify({"error": "Missing user_history in request"}), 400
    
    user_history = data['user_history']
    user_preferences = data.get('user_preferences', None)
    top_n = data.get('top_n', 5)
    
    recommendations = get_personalized_recommendations(user_history, df, similarity_matrix, user_preferences, top_n)
    
    if isinstance(recommendations, dict) and 'error' in recommendations:
        return jsonify(recommendations), 400
    
    return jsonify({"personalized_recommendations": recommendations})

# API endpoint to get all recommendations for a user
@app.route('/api/recommendations/all', methods=['POST'])
def api_all_recommendations():
    global df, similarity_matrix
    
    if df is None or similarity_matrix is None:
        return jsonify({"error": "Recommendation system not initialized"}), 500
    
    data = request.get_json()
    user_history = data.get('user_history', [])
    user_preferences = data.get('user_preferences', None)
    
    response = {
        "trending_items": get_trending_items(df)
    }
    
    if user_history:
        valid_history = [pid for pid in user_history if pid in df['id'].values]
        
        if valid_history:
            personalized = get_personalized_recommendations(valid_history, df, similarity_matrix, user_preferences)
            if not isinstance(personalized, dict) or 'error' not in personalized:
                response["personalized"] = personalized
            
            last_viewed = valid_history[-1]
            similar = get_similar_products(last_viewed, df, similarity_matrix)
            if not isinstance(similar, dict) or 'error' not in similar:
                response["similar_to_last_viewed"] = similar
                
            complete_look = complete_the_look(last_viewed, df, similarity_matrix)
            if not isinstance(complete_look, dict) or 'error' not in complete_look:
                response["complete_the_look"] = complete_look
    
    return jsonify(response)

# Initialize the recommendation system
@app.route('/api/recommendations/initialize', methods=['GET'])
def initialize_recommendation_system():
    global df, features, similarity_matrix, encoder, scaler, tfidf
    
    try:
        df = load_data_from_mongodb()
        if df is None or df.empty:
            return jsonify({"error": "Failed to load data from MongoDB"}), 500
        
        features, encoder, scaler, tfidf = engineer_features(df)
        similarity_matrix = compute_similarity(features)
        
        return jsonify({"message": f"Recommendation system initialized with {len(df)} products"})
    except Exception as e:
        return jsonify({"error": f"Failed to initialize recommendation system: {str(e)}"}), 500

# Health check endpoint
@app.route('/api/health', methods=['GET'])
def health_check():
    global df
    status = "initialized" if df is not None else "not initialized"
    return jsonify({"status": "healthy", "recommendation_system": status})

# Initialize on startup
@app.before_first_request
def before_first_request():
    global df, features, similarity_matrix, encoder, scaler, tfidf
    
    try:
        print("Initializing recommendation system...")
        df = load_data_from_mongodb()
        if df is not None and not df.empty:
            print(f"Loaded {len(df)} products")
            features, encoder, scaler, tfidf = engineer_features(df)
            similarity_matrix = compute_similarity(features)
            print("Recommendation system initialized successfully")
        else:
            print("Failed to load data")
    except Exception as e:
        print(f"Error initializing recommendation system: {e}")

if __name__ == '__main__':
    # Get port from environment variable or use 5000 as default
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)