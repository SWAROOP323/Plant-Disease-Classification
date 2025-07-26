import os
import tensorflow as tf
import numpy as np
from flask import Flask, request, render_template_string, jsonify
from flask_cors import CORS
from PIL import Image
import requests
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH = "plant_disease_resnet.keras"
MODEL_URL = "https://huggingface.co/spaces/SWAROOP323/plant-disease-predictor/resolve/main/plant_disease_resnet.keras"

# Plant disease class labels (38 classes)
class_labels = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global model variable
model = None

def download_model():
    """Download the model if it doesn't exist"""
    if not os.path.exists(MODEL_PATH):
        logger.info("Downloading model...")
        try:
            response = requests.get(MODEL_URL, stream=True)
            response.raise_for_status()
            
            with open(MODEL_PATH, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            logger.info("Model downloaded successfully.")
        except Exception as e:
            logger.error(f"Error downloading model: {e}")
            raise

def load_model():
    """Load the TensorFlow model"""
    global model
    try:
        download_model()
        model = tf.keras.models.load_model(MODEL_PATH)
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

def preprocess_image(image):
    """Preprocess the image for model prediction"""
    try:
        # Resize image to 224x224 (ResNet input size)
        image = image.resize((224, 224))
        # Convert to RGB if not already
        if image.mode != 'RGB':
            image = image.convert('RGB')
        # Convert to numpy array and normalize
        image_array = np.array(image) / 255.0
        # Add batch dimension
        return np.expand_dims(image_array, axis=0)
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        raise

HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Plant Disease Classifier</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #2c3e50;
            text-align: center;
            margin-bottom: 30px;
        }
        .upload-form {
            text-align: center;
            margin-bottom: 30px;
        }
        .file-input {
            margin: 20px 0;
            padding: 10px;
            border: 2px dashed #3498db;
            border-radius: 5px;
            background-color: #ecf0f1;
        }
        .submit-btn {
            background-color: #27ae60;
            color: white;
            padding: 12px 30px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
        }
        .submit-btn:hover {
            background-color: #229954;
        }
        .result {
            margin-top: 30px;
            padding: 20px;
            background-color: #e8f5e8;
            border-radius: 5px;
            border-left: 4px solid #27ae60;
        }
        .error {
            background-color: #fdeaea;
            border-left: 4px solid #e74c3c;
        }
        .prediction {
            font-size: 18px;
            font-weight: bold;
            color: #2c3e50;
        }
        .confidence {
            font-size: 14px;
            color: #7f8c8d;
            margin-top: 5px;
        }
        .info {
            margin-top: 30px;
            padding: 15px;
            background-color: #ebf3fd;
            border-radius: 5px;
            border-left: 4px solid #3498db;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🌱 Plant Disease Classifier</h1>
        <p style="text-align: center; color: #7f8c8d;">
            Upload an image of a plant leaf to detect diseases using AI
        </p>
        
        <form method="post" enctype="multipart/form-data" class="upload-form">
            <div class="file-input">
                <input type="file" name="file" accept="image/*" required>
            </div>
            <button type="submit" class="submit-btn">🔍 Analyze Plant</button>
        </form>
        
        {% if prediction %}
            <div class="result">
                <h3>Analysis Result:</h3>
                <div class="prediction">{{ prediction }}</div>
                {% if confidence %}
                    <div class="confidence">Confidence: {{ confidence }}%</div>
                {% endif %}
            </div>
        {% endif %}
        
        {% if error %}
            <div class="result error">
                <h3>Error:</h3>
                <p>{{ error }}</p>
            </div>
        {% endif %}
        
        <div class="info">
            <h4>Supported Plants:</h4>
            <p>Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, Tomato</p>
            <p><strong>Note:</strong> For best results, upload clear images of individual leaves with good lighting.</p>
        </div>
    </div>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    """Main route for the web interface"""
    prediction = None
    confidence = None
    error = None
    
    if request.method == "POST":
        try:
            if 'file' not in request.files:
                error = "No file uploaded"
            else:
                file = request.files['file']
                if file.filename == '':
                    error = "No file selected"
                elif file:
                    # Process the uploaded image
                    img = Image.open(file.stream)
                    processed_img = preprocess_image(img)
                    
                    # Make prediction
                    predictions = model.predict(processed_img)
                    pred_class_idx = np.argmax(predictions, axis=1)[0]
                    confidence_score = float(np.max(predictions) * 100)
                    
                    prediction = class_labels[pred_class_idx]
                    confidence = f"{confidence_score:.1f}"
                    
                    logger.info(f"Prediction: {prediction}, Confidence: {confidence}%")
                    
        except Exception as e:
            logger.error(f"Error processing request: {e}")
            error = f"Error processing image: {str(e)}"
    
    return render_template_string(HTML_PAGE, 
                                prediction=prediction, 
                                confidence=confidence, 
                                error=error)

@app.route("/api/predict", methods=["POST"])
def api_predict():
    """API endpoint for predictions"""
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        # Process the uploaded image
        img = Image.open(file.stream)
        processed_img = preprocess_image(img)
        
        # Make prediction
        predictions = model.predict(processed_img)
        pred_class_idx = np.argmax(predictions, axis=1)[0]
        confidence_score = float(np.max(predictions) * 100)
        
        result = {
            "prediction": class_labels[pred_class_idx],
            "confidence": confidence_score,
            "class_index": int(pred_class_idx)
        }
        
        logger.info(f"API Prediction: {result}")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"API Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/health")
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "model_loaded": model is not None})

@app.route("/classes")
def get_classes():
    """Get all available class labels"""
    return jsonify({"classes": class_labels, "total": len(class_labels)})

# Initialize the model when the app starts
try:
    load_model()
except Exception as e:
    logger.error(f"Failed to load model on startup: {e}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)

