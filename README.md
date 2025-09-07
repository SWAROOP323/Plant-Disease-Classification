Plant Disease Classification

This repository contains a plant disease detection application built with TensorFlow and Gradio. The application uses a fine-tuned ResNet model to identify various plant diseases from leaf images.

Features

•
Disease Prediction: Upload a plant leaf image and get a prediction of the disease, along with a confidence score.

•
Web Interface: Easy-to-use web interface powered by Gradio for quick predictions.

•
Pre-trained Model: Utilizes a pre-trained ResNet model for accurate disease detection.

Installation

1.
Clone the repository:

2.
Create a virtual environment (recommended):

3.
Install the required dependencies:

Usage

1.
Run the application:

2.
Open your web browser and navigate to the URL provided by Gradio (usually http://127.0.0.1:7860).

3.
Upload a plant leaf image using the interface to get a disease prediction.

Model

The application uses a plant_disease_resnet.keras model. If the model file is not present locally, it will be automatically downloaded from the following Hugging Face Space:

https://huggingface.co/spaces/SWAROOP323/plant-disease-predictor/resolve/main/plant_disease_resnet.keras

Contributing

Contributions are welcome! Please feel free to open issues or submit pull requests.

Contact

For any questions or suggestions, please contact.
Manchala D V V S Swaroop
swaroopmanchala323@gmail.com

