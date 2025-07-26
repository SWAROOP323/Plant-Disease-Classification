# Render.com Deployment Guide

## Quick Start

1. **Upload to GitHub**:
   - Create a new repository on GitHub
   - Upload all files from this directory to the repository

2. **Deploy on Render.com**:
   - Go to [render.com](https://render.com) and sign up/login
   - Click "New +" → "Web Service"
   - Connect your GitHub repository
   - Render will automatically detect the `render.yaml` configuration

3. **Automatic Configuration**:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn app:app --bind 0.0.0.0:$PORT`
   - Python Version: 3.10.13

## Manual Configuration (if needed)

If automatic detection doesn't work:

1. **Service Settings**:
   - Name: `plant-disease-classification`
   - Environment: `Python`
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn app:app --bind 0.0.0.0:$PORT`

2. **Environment Variables**:
   - `PYTHON_VERSION`: `3.10.13`

## Expected Behavior

- **First Deploy**: Takes 5-10 minutes due to model download
- **Subsequent Deploys**: Faster as dependencies are cached
- **Cold Starts**: May take 30-60 seconds on free tier

## Testing Your Deployment

Once deployed, test these endpoints:
- `https://your-app.onrender.com/` - Web interface
- `https://your-app.onrender.com/health` - Health check
- `https://your-app.onrender.com/classes` - Available classes

## Troubleshooting

### Build Fails
- Check that all files are uploaded correctly
- Verify `requirements.txt` format
- Check Render logs for specific errors

### App Won't Start
- Ensure `gunicorn` is in requirements.txt
- Check that `app.py` has no syntax errors
- Verify the start command format

### Model Download Issues
- The app downloads a ~85MB model on first start
- This may cause timeout on free tier
- Check logs for download progress

### Memory Issues
- TensorFlow requires significant memory
- Consider upgrading to a paid plan for better performance
- Monitor memory usage in Render dashboard

## Performance Tips

1. **Upgrade Plan**: For production use, consider a paid plan
2. **Keep Warm**: Use a service like UptimeRobot to prevent sleeping
3. **Monitor Logs**: Check Render logs for performance insights
4. **Optimize Images**: Resize images before upload for faster processing

## Support

For issues with:
- **Render.com**: Check their documentation and support
- **Application Code**: Review the logs and error messages
- **Model Performance**: Ensure images are clear plant leaf photos

