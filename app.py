import gradio as gr
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os
import requests
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure TensorFlow for CPU-only deployment
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Model configuration
MODEL_PATH = "plant_disease_resnet.keras"
MODEL_URL = "https://huggingface.co/spaces/SWAROOP323/plant-disease-predictor/resolve/main/plant_disease_resnet.keras"

# Define class labels based on your confusion matrix (in the same order)
class_labels = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
    "Blueberry___healthy", "Cherry_(including_sour)___Powdery_mildew", "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)___Common_rust",
    "Corn_(maize)___Northern_Leaf_Blight", "Corn_(maize)___healthy", "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)", "Peach___Bacterial_spot", "Peach___healthy",
    "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy", "Potato___Early_blight",
    "Potato___Late_blight", "Potato___healthy", "Raspberry___healthy", "Soybean___healthy",
    "Squash___Powdery_mildew", "Strawberry___Leaf_scorch", "Strawberry___healthy",
    "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight", "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot", "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot", "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]

# Global model variable
model = None

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

def load_model():
    """Load the trained model"""
    global model
    try:
        download_model()
        logger.info("Loading model...")
        model = tf.keras.models.load_model(MODEL_PATH)
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

# Image preprocessing function
def predict_disease(img):
    """Predict plant disease from uploaded image"""
    try:
        if model is None:
            return "Error: Model not loaded. Please try again."
        
        # Preprocess the image
        img = img.convert("RGB")  # Ensure 3 channels
        img = img.resize((224, 224))  # Resize to match model input
        img_array = image.img_to_array(img)
        img_array = img_array / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=0)

        # Make prediction
        prediction = model.predict(img_array, verbose=0)
        predicted_class = class_labels[np.argmax(prediction)]
        confidence = round(np.max(prediction) * 100, 2)
        
        # Format the result
        result = f"🌿 **Prediction:** {predicted_class.replace('___', ' - ')}\n"
        result += f"📊 **Confidence:** {confidence}%\n\n"
        
        # Add interpretation
        if "healthy" in predicted_class.lower():
            result += "✅ **Status:** This plant appears to be healthy!"
        else:
            disease_name = predicted_class.split('___')[1] if '___' in predicted_class else predicted_class
            result += f"⚠️ **Status:** Disease detected - {disease_name}"
            result += f"\n💡 **Recommendation:** Consider consulting with an agricultural expert for treatment options."
        
        return result
        
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        return f"Error processing image: {str(e)}"

# Load model on startup
try:
    load_model()
    logger.info("Application startup complete.")
except Exception as e:
    logger.error(f"Failed to load model on startup: {e}")
    model = None

# Create Gradio interface
def create_interface():
    """Create and configure the Gradio interface"""
    
    # Custom CSS for better styling
    css = """
    .gradio-container {
        max-width: 800px !important;
        margin: auto !important;
    }
    .title {
        text-align: center;
        color: #2d5a27;
        font-size: 2.5em;
        margin-bottom: 0.5em;
    }
    .description {
        text-align: center;
        font-size: 1.2em;
        color: #666;
        margin-bottom: 1em;
    }
    """
    
    # Create the interface
    interface = gr.Interface(
        fn=predict_disease,
        inputs=gr.Image(type="pil", label="Upload Plant Leaf Image"),
        outputs=gr.Textbox(label="Disease Analysis Result", lines=6),
        title="🌿 Plant Disease Detector",
        description="Upload a clear image of a plant leaf to detect diseases using AI. Supports 14 plant types including Apple, Tomato, Potato, Corn, and more.",
        article="""
        ### How to use:
        1. **Upload a clear image** of a plant leaf
        2. **Wait for analysis** (may take a few seconds)
        3. **Review the results** including disease prediction and confidence level
        
        ### Supported Plants:
        Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, Tomato
        
        ### Tips for best results:
        - Use clear, well-lit images
        - Focus on individual leaves
        - Avoid blurry or dark images
        - Ensure the leaf fills most of the image frame
        """,
        css=css,
        theme=gr.themes.Soft(),
        examples=[
            # You can add example images here if you have them
        ]
    )
    
    return interface

if __name__ == "__main__":
    # Get port from environment variable (Render.com sets this)
    port = int(os.environ.get("PORT", 7860))
    
    # Create and launch the interface
    interface = create_interface()
    
    # Launch with server configuration for Render.com
    interface.launch(
        server_name="0.0.0.0",  # Listen on all interfaces
        server_port=port,       # Use the port provided by Render
        share=False,            # Don't create a public link
        debug=False,            # Disable debug mode for production
        show_error=True,        # Show errors to users
        quiet=False             # Show startup logs
    )

