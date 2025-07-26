# Plant Disease Classification - Render.com Deployment

This is a Flask web application that uses a deep learning model to classify plant diseases from leaf images. The application is configured for deployment on Render.com.

## Features

- **Web Interface**: User-friendly web interface for uploading plant leaf images
- **AI Classification**: Uses a ResNet-based deep learning model to classify 38 different plant diseases
- **REST API**: Provides API endpoints for programmatic access
- **Health Monitoring**: Includes health check endpoints for monitoring
- **Responsive Design**: Mobile-friendly interface

## Supported Plants and Diseases

The model can classify diseases in the following plants:
- Apple (scab, black rot, cedar apple rust, healthy)
- Blueberry (healthy)
- Cherry (powdery mildew, healthy)
- Corn/Maize (cercospora leaf spot, common rust, northern leaf blight, healthy)
- Grape (black rot, esca, leaf blight, healthy)
- Orange (citrus greening)
- Peach (bacterial spot, healthy)
- Pepper/Bell (bacterial spot, healthy)
- Potato (early blight, late blight, healthy)
- Raspberry (healthy)
- Soybean (healthy)
- Squash (powdery mildew)
- Strawberry (leaf scorch, healthy)
- Tomato (multiple diseases and healthy)

## Deployment on Render.com

### Prerequisites
- A Render.com account
- Git repository with the code

### Deployment Steps

1. **Create a new Web Service** on Render.com
2. **Connect your repository** containing these files
3. **Configure the service**:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn app:app --bind 0.0.0.0:$PORT`
   - Environment: Python 3.10.13

### Environment Variables
The application will automatically use the `PORT` environment variable provided by Render.com.

### Files Included

- `app.py` - Main Flask application with web interface and API endpoints
- `requirements.txt` - Python dependencies
- `render.yaml` - Render.com service configuration
- `runtime.txt` - Python runtime version specification
- `README.md` - This documentation file

## API Endpoints

### Web Interface
- `GET /` - Main web interface for uploading images

### API Endpoints
- `POST /api/predict` - Upload image and get prediction
- `GET /health` - Health check endpoint
- `GET /classes` - Get list of all supported classes

### API Usage Example

```bash
# Upload an image for prediction
curl -X POST -F "file=@plant_leaf.jpg" https://your-app.onrender.com/api/predict
```

Response:
```json
{
  "prediction": "Tomato___Late_blight",
  "confidence": 95.2,
  "class_index": 30
}
```

## Model Information

The application uses a pre-trained ResNet model that is automatically downloaded from Hugging Face on first startup. The model file is approximately 85MB and will be cached after the first download.

## Performance Considerations

- **Cold Start**: The first request may take longer due to model loading
- **Memory Usage**: The TensorFlow model requires significant memory
- **Free Tier**: On Render's free tier, the service may sleep after inactivity

## Troubleshooting

### Common Issues

1. **Model Download Fails**: Check internet connectivity and Hugging Face availability
2. **Memory Errors**: The application requires sufficient memory for TensorFlow
3. **Slow Predictions**: First prediction may be slower due to model initialization

### Logs
Check the Render.com logs for detailed error messages and debugging information.

## Local Development

To run locally:

```bash
pip install -r requirements.txt
python app.py
```

The application will be available at `http://localhost:5000`

## License

This project is provided as-is for educational and demonstration purposes.

