import os
import logging
import gc

# Configure TensorFlow to use minimal memory and CPU only
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress all TF logs except errors
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'false'
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

# Memory optimization settings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MAX_VLOG_LEVEL'] = '0'

import tensorflow as tf
import numpy as np
from flask import Flask, request, render_template_string, jsonify
from flask_cors import CORS
from PIL import Image
import requests

# Configure TensorFlow for minimal memory usage
tf.config.set_visible_devices([], 'GPU')
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

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
CORS(app)

# Global model variable - will be loaded on first request
model = None
model_loading = False

def download_model():
    """Download the model if it doesn't exist"""
    if not os.path.exists(MODEL_PATH):
        logger.info("Downloading model...")
        try:
            response = requests.get(MODEL_URL, stream=True, timeout=300)
            response.raise_for_status()
            
            with open(MODEL_PATH, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            logger.info("Model downloaded successfully.")
        except Exception as e:
            logger.error(f"Error downloading model: {e}")
            raise

def load_model_lazy():
    """Load the model only when needed (lazy loading)"""
    global model, model_loading
    
    if model is not None:
        return model
        
    if model_loading:
        return None
        
    try:
        model_loading = True
        logger.info("Starting lazy model loading...")
        
        download_model()
        
        # Force garbage collection before loading
        gc.collect()
        
        logger.info("Loading TensorFlow model...")
        model = tf.keras.models.load_model(MODEL_PATH)
        
        # Force garbage collection after loading
        gc.collect()
        
        logger.info("Model loaded successfully.")
        model_loading = False
        return model
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        model_loading = False
        raise

def preprocess_image(image):
    """Preprocess the image for model prediction"""
    try:
        # Resize image to 224x224
        image = image.resize((224, 224))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        # Convert to numpy array and normalize
        image_array = np.array(image, dtype=np.float32) / 255.0
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
        .status {
            text-align: center;
            padding: 10px;
            margin: 10px 0;
            border-radius: 5px;
        }
        .status.ready {
            background-color: #d4edda;
            border: 1px solid #c3e6cb;
            color: #155724;
        }
        .status.loading {
            background-color: #fff3cd;
            border: 1px solid #ffeaa7;
            color: #856404;
        }
        .status.not-loaded {
            background-color: #f8d7da;
            border: 1px solid #f5c6cb;
            color: #721c24;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🌱 Plant Disease Classifier</h1>
        <p style="text-align: center; color: #7f8c8d;">
            Upload an image of a plant leaf to detect diseases using AI
        </p>
        
        {% if model_status %}
            <div class="status {{ status_class }}">{{ model_status }}</div>
        {% endif %}
        
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
            <p><strong>Note:</strong> Model loads on first use. First prediction may take longer.</p>
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
    
    # Determine model status
    if model is not None:
        model_status = "✅ Model ready"
        status_class = "ready"
    elif model_loading:
        model_status = "⏳ Model loading..."
        status_class = "loading"
    else:
        model_status = "💤 Model will load on first use"
        status_class = "not-loaded"
    
    if request.method == "POST":
        try:
            if 'file' not in request.files:
                error = "No file uploaded"
            else:
                file = request.files['file']
                if file.filename == '':
                    error = "No file selected"
                elif file:
                    # Load model if not already loaded
                    current_model = load_model_lazy()
                    if current_model is None:
                        error = "Model is currently loading. Please try again in a moment."
                    else:
                        # Process the uploaded image
                        img = Image.open(file.stream)
                        processed_img = preprocess_image(img)
                        
                        # Make prediction
                        predictions = current_model.predict(processed_img, verbose=0)
                        pred_class_idx = np.argmax(predictions, axis=1)[0]
                        confidence_score = float(np.max(predictions) * 100)
                        
                        prediction = class_labels[pred_class_idx]
                        confidence = f"{confidence_score:.1f}"
                        
                        logger.info(f"Prediction: {prediction}, Confidence: {confidence}%")
                        
                        # Force garbage collection after prediction
                        gc.collect()
                    
        except Exception as e:
            logger.error(f"Error processing request: {e}")
            error = f"Error processing image: {str(e)}"
    
    return render_template_string(HTML_PAGE, 
                                prediction=prediction, 
                                confidence=confidence, 
                                error=error,
                                model_status=model_status,
                                status_class=status_class)

@app.route("/api/predict", methods=["POST"])
def api_predict():
    """API endpoint for predictions"""
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        # Load model if not already loaded
        current_model = load_model_lazy()
        if current_model is None:
            return jsonify({"error": "Model is loading, please try again"}), 503
        
        # Process the uploaded image
        img = Image.open(file.stream)
        processed_img = preprocess_image(img)
        
        # Make prediction
        predictions = current_model.predict(processed_img, verbose=0)
        pred_class_idx = np.argmax(predictions, axis=1)[0]
        confidence_score = float(np.max(predictions) * 100)
        
        result = {
            "prediction": class_labels[pred_class_idx],
            "confidence": confidence_score,
            "class_index": int(pred_class_idx)
        }
        
        logger.info(f"API Prediction: {result}")
        
        # Force garbage collection after prediction
        gc.collect()
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"API Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/health")
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy", 
        "model_loaded": model is not None,
        "model_loading": model_loading,
        "tensorflow_version": tf.__version__
    })

@app.route("/classes")
def get_classes():
    """Get all available class labels"""
    return jsonify({"classes": class_labels, "total": len(class_labels)})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"Starting Flask app on port {port} with lazy model loading")
    app.run(host="0.0.0.0", port=port, debug=False)
