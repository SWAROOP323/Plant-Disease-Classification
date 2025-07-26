import os
from flask import Flask, request, render_template
import tensorflow as tf
from huggingface_hub import hf_hub_download
from PIL import Image
import numpy as np

app = Flask(__name__)

# Hugging Face repo details
REPO_ID = "SWAROOP323/plant-disease-predictor"

# Download models dynamically
RESNET_MODEL_PATH = hf_hub_download(repo_id=REPO_ID, filename="plant_disease_resnet.keras")
VGG_FT_MODEL_PATH = hf_hub_download(repo_id=REPO_ID, filename="plant_disease_vgg_finetuned.keras")

# Load models
resnet_model = tf.keras.models.load_model(RESNET_MODEL_PATH)
vgg_model = tf.keras.models.load_model(VGG_FT_MODEL_PATH)

# Image preprocessing
def preprocess_image(image):
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

@app.route("/", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        file = request.files['file']
        if file:
            img = Image.open(file.stream)
            processed_img = preprocess_image(img)

            # Predict using ResNet model
            resnet_pred = resnet_model.predict(processed_img)
            resnet_class = np.argmax(resnet_pred, axis=1)[0]

            # Predict using Fine-tuned VGG model
            vgg_pred = vgg_model.predict(processed_img)
            vgg_class = np.argmax(vgg_pred, axis=1)[0]

            return f"""
                <h2>Prediction Results</h2>
                <p><b>ResNet Model:</b> Class {resnet_class}</p>
                <p><b>Fine-tuned VGG Model:</b> Class {vgg_class}</p>
                <a href="/">Go Back</a>
            """
    return '''
        <h1>Plant Disease Prediction</h1>
        <form method="POST" enctype="multipart/form-data">
            <input type="file" name="file">
            <input type="submit" value="Predict">
        </form>
    '''

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
