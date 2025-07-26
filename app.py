import os
import tensorflow as tf
import numpy as np
from flask import Flask, request, render_template_string
from PIL import Image
import requests

MODEL_PATH = "plant_disease_resnet.keras"
MODEL_URL = "https://huggingface.co/spaces/SWAROOP323/plant-disease-predictor/resolve/main/plant_disease_resnet.keras"

# Download model if not exists
if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    r = requests.get(MODEL_URL)
    with open(MODEL_PATH, "wb") as f:
        f.write(r.content)
    print("Model downloaded.")

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

# Class labels (replace with actual 38 classes)
class_labels = [f"Class {i}" for i in range(38)]

app = Flask(__name__)

def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    return np.expand_dims(image, axis=0)

HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>Plant Disease Predictor</title>
</head>
<body>
    <h1>Upload a plant leaf image</h1>
    <form method="post" enctype="multipart/form-data">
        <input type="file" name="file" accept="image/*" required>
        <input type="submit" value="Predict">
    </form>
    {% if prediction %}
        <h2>Prediction: {{ prediction }}</h2>
    {% endif %}
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        file = request.files["file"]
        if file:
            img = Image.open(file).convert("RGB")
            processed = preprocess_image(img)
            preds = model.predict(processed)
            pred_class = np.argmax(preds, axis=1)[0]
            prediction = class_labels[pred_class]
    return render_template_string(HTML_PAGE, prediction=prediction)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
