# 🌿 Plant Disease Classification with Gradio

An AI-powered web application that detects plant diseases from leaf images using deep learning. Built with Gradio for an intuitive user interface and optimized for deployment on Render.com.

## 🚀 Features

- **Easy-to-Use Interface**: Drag-and-drop image upload with Gradio
- **AI-Powered Detection**: Uses a fine-tuned ResNet model for accurate predictions
- **38 Disease Classes**: Supports detection across 14 plant types
- **Real-Time Results**: Instant disease prediction with confidence scores
- **Cloud Deployment**: Optimized for Render.com free tier
- **Mobile Friendly**: Responsive design works on all devices

## 🌱 Supported Plants and Diseases

The model can detect diseases in the following plants:

| Plant Type | Diseases Detected |
|------------|-------------------|
| **Apple** | Apple scab, Black rot, Cedar apple rust, Healthy |
| **Blueberry** | Healthy |
| **Cherry** | Powdery mildew, Healthy |
| **Corn (Maize)** | Cercospora leaf spot, Common rust, Northern Leaf Blight, Healthy |
| **Grape** | Black rot, Esca (Black Measles), Leaf blight, Healthy |
| **Orange** | Citrus greening (Huanglongbing) |
| **Peach** | Bacterial spot, Healthy |
| **Pepper (Bell)** | Bacterial spot, Healthy |
| **Potato** | Early blight, Late blight, Healthy |
| **Raspberry** | Healthy |
| **Soybean** | Healthy |
| **Squash** | Powdery mildew |
| **Strawberry** | Leaf scorch, Healthy |
| **Tomato** | Bacterial spot, Early blight, Late blight, Leaf Mold, Septoria leaf spot, Spider mites, Target Spot, Yellow Leaf Curl Virus, Mosaic virus, Healthy |

## 🛠️ Technology Stack

- **Frontend**: Gradio (Python web framework)
- **Backend**: TensorFlow/Keras for deep learning
- **Model**: Fine-tuned ResNet architecture
- **Deployment**: Render.com cloud platform
- **Language**: Python 3.12.8

## 📋 Project Structure

```
plant-disease-classification/
├── app.py                      # Main Gradio application
├── requirements.txt            # Python dependencies
├── runtime.txt                # Python version specification
├── render.yaml                # Render.com deployment config
├── .gitignore                 # Git ignore rules
├── README.md                  # This file
├── GITHUB_UPLOAD_GUIDE.md     # GitHub upload instructions
└── RENDER_DEPLOYMENT_GUIDE.md # Render.com deployment guide
```

## 🚀 Quick Start

### Local Development

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/plant-disease-classification.git
   cd plant-disease-classification
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   python app.py
   ```

4. **Open in browser**
   - The app will start on `http://localhost:7860`
   - Upload a plant leaf image to test

### Cloud Deployment

Follow the detailed guides included in this repository:

1. **GitHub Upload**: See `GITHUB_UPLOAD_GUIDE.md`
2. **Render.com Deployment**: See `RENDER_DEPLOYMENT_GUIDE.md`

## 📊 Model Performance

The ResNet-based model achieves:
- **High Accuracy**: Trained on thousands of plant disease images
- **Fast Inference**: Optimized for CPU deployment
- **Robust Detection**: Works with various image qualities and lighting conditions

## 🎯 How to Use

1. **Access the Application**
   - Visit the deployed URL or run locally
   - You'll see the Gradio interface

2. **Upload an Image**
   - Click the upload area or drag-and-drop an image
   - Supported formats: JPG, PNG, JPEG
   - Best results with clear, well-lit leaf images

3. **Get Results**
   - The AI will analyze the image
   - Results include:
     - Disease prediction
     - Confidence percentage
     - Health status
     - Recommendations (if disease detected)

## 🔧 Configuration

### Environment Variables

The application uses these optional environment variables:

- `PORT`: Server port (automatically set by Render.com)
- `PYTHON_VERSION`: Python version (set in runtime.txt)

### Model Configuration

- **Model Source**: Downloads automatically from Hugging Face
- **Input Size**: 224x224 pixels (automatically resized)
- **Preprocessing**: Normalization and RGB conversion
- **Output**: 38-class probability distribution

## 🐛 Troubleshooting

### Common Issues

1. **Model Loading Errors**
   - Ensure internet connectivity for model download
   - Check logs for specific error messages
   - Model downloads automatically on first run

2. **Memory Issues**
   - Use `tensorflow-cpu` instead of `tensorflow`
   - Consider upgrading to paid Render.com tier
   - Monitor memory usage in deployment logs

3. **Slow Performance**
   - First prediction may be slower (model loading)
   - Subsequent predictions should be faster
   - Consider upgrading instance type for better performance

### Getting Help

- Check the deployment guides for detailed troubleshooting
- Review Render.com logs for error messages
- Ensure all dependencies are correctly specified

## 📈 Performance Optimization

### For Free Tier Deployment

- Uses `tensorflow-cpu` for reduced memory usage
- Optimized Gradio configuration
- Automatic model downloading (no large files in repo)
- Single-threaded operation for memory efficiency

### For Production Use

- Upgrade to Render.com paid tier for better performance
- Consider using GPU instances for faster inference
- Implement caching for frequently used models
- Add monitoring and alerting

## 🔒 Security and Privacy

- **No Data Storage**: Images are processed in memory only
- **No Logging**: User images are not saved or logged
- **HTTPS**: All communications encrypted via Render.com
- **Open Source**: Code is transparent and auditable

## 📄 License

This project is provided for educational and demonstration purposes. The model and code are available for non-commercial use.

## 🤝 Contributing

Contributions are welcome! Please feel free to:
- Report bugs or issues
- Suggest new features
- Improve documentation
- Submit pull requests

## 📞 Support

For questions or support:
- Check the troubleshooting guides
- Review the deployment documentation
- Open an issue on GitHub

## 🙏 Acknowledgments

- **TensorFlow Team**: For the deep learning framework
- **Gradio Team**: For the excellent web interface framework
- **Render.com**: For providing accessible cloud deployment
- **Plant Disease Dataset**: Contributors to the training dataset

---

**Ready to deploy?** Follow the step-by-step guides included in this repository to get your plant disease detector live on the web!

