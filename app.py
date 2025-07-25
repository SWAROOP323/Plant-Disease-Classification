import gradio as gr
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import cv2

# Load your trained model
model = tf.keras.models.load_model("plant_disease_resnet.keras")

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
    "Tomato___Septoria_leaf_spot"
]

# Image preprocessing function
def predict_disease(img):
    img = img.convert("RGB")  # Ensure 3 channels
    img = img.resize((224, 224))  # Resize to match model input
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = class_labels[np.argmax(prediction)]
    confidence = round(np.max(prediction) * 100, 2)
    
    return f"Prediction: {predicted_class}\nConfidence: {confidence}%"

# Launch the Gradio interface
gr.Interface(fn=predict_disease,
             inputs=gr.Image(type="pil"),
             outputs="text",
             title="🌿 Plant Disease Detector using ResNet",
             description="Upload a plant leaf image to detect disease using a fine-tuned ResNet model.").launch()
