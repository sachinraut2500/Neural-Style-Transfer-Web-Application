# 🎨 Neural Style Transfer Web Application

A professional-grade web application that applies artistic styles to images using deep learning and TensorFlow. This project implements the famous Neural Style Transfer algorithm with a user-friendly web interface.
----
## 🌟 Features

- **Real-time Style Transfer**: Apply artistic styles to any image
- **Web Interface**: Easy-to-use upload and preview system  
- **Multiple Optimizations**: Content, style, and total variation losses
- **RESTful API**: Programmatic access for integration
- **Production Ready**: Error handling, logging, and health checks

## 🚀 Quick Start

### Prerequisites
```bash
pip install tensorflow>=2.8.0 flask pillow numpy matplotlib
```

### Installation
1. Download the `neural_style_transfer_webapp.py` file
2. Run the application:
```bash
python neural_style_transfer_webapp.py
```
3. Open your browser to `http://localhost:5000`

### Usage
1. Upload a content image (the image you want to stylize)
2. Upload a style image (the artistic style to apply)
3. Click "Generate Stylized Image"
4. Wait for processing (typically 2-5 minutes)
5. Download your stylized result!

## 🏗️ Project Structure

```
neural-style-transfer-webapp/
├── neural_style_transfer_webapp.py    # Main application file
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── Dockerfile                         # Docker configuration
├── examples/                          # Sample images
│   ├── content/
│   └── styles/
└── tests/
    └── test_style_transfer.py
```

## 🔧 Technical Details

### Algorithm
- **Base Model**: VGG19 pre-trained on ImageNet
- **Content Layers**: `block5_conv2`
- **Style Layers**: `block1_conv1` through `block5_conv1`
- **Optimization**: Adam optimizer with custom loss function

### Loss Function
```python
Total Loss = α × Content Loss + β × Style Loss + γ × Total Variation Loss
```

Where:
- **Content Loss**: Preserves the structure of the original image
- **Style Loss**: Captures the artistic style using Gram matrices
- **Total Variation Loss**: Ensures smoothness and reduces noise

### API Endpoints

#### POST `/transfer`
Transfer style between two images.

**Request**: Multipart form data with:
- `content`: Content image file
- `style`: Style image file

**Response**:
```json
{
  "success": true,
  "result": "base64_encoded_image_string"
}
```

#### GET `/health`
Health check endpoint.

**Response**:
```json
{
  "status": "healthy",
  "model": "neural_style_transfer"
}
```

## 🐳 Docker Deployment

### Dockerfile
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY neural_style_transfer_webapp.py .

EXPOSE 5000

CMD ["python", "neural_style_transfer_webapp.py"]
```

### Build and Run
```bash
docker build -t neural-style-transfer .
docker run -p 5000:5000 neural-style-transfer
```

## 📋 Requirements.txt

```txt
tensorflow>=2.8.0
flask>=2.0.0
pillow>=8.0.0
numpy>=1.21.0
matplotlib>=3.5.0
```

## 🧪 Testing

Create `test_style_transfer.py`:

```python
import unittest
import numpy as np
from neural_style_transfer_webapp import NeuralStyleTransfer

class TestNeuralStyleTransfer(unittest.TestCase):
    def setUp(self):
        self.nst = NeuralStyleTransfer()
    
    def test_gram_matrix(self):
        # Test Gram matrix calculation
        test_input = np.random.rand(1, 10, 10, 64)
        gram = self.nst.gram_matrix(test_input)
        self.assertEqual(gram.shape, (1, 64, 64))
    
    def test_model_initialization(self):
        # Test model loads correctly
        self.assertIsNotNone(self.nst.vgg)
        self.assertFalse(self.nst.vgg.trainable)

if __name__ == '__main__':
    unittest.main()
```

Run tests:
```bash
python -m pytest test_style_transfer.py -v
```

## 🎯 Performance Optimization

### GPU Acceleration
```python
# Enable GPU if available
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
```

### Batch Processing
```python
# Process multiple images simultaneously
def batch_transfer(content_images, style_images):
    # Implementation for batch processing
    pass
```

## 📊 Example Results

| Content Image | Style Image | Result |
|---------------|-------------|---------|
| ![Portrait](examples/content/portrait.jpg) | ![Starry Night](examples/styles/starry_night.jpg) | ![Result](examples/results/portrait_starry.jpg) |

## 🛠️ Customization

### Adjust Style Strength
```python
# Modify weights in NeuralStyleTransfer class
self.style_weight = 1e-2      # Higher = more style
self.content_weight = 1e4     # Higher = more content preservation
self.total_variation_weight = 30  # Higher = smoother result
```

### Custom Layers
```python
# Use different VGG layers
content_layers = ['block4_conv2', 'block5_conv2']
style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1']
```

## 🚀 Advanced Features

### Real-time Preview
```javascript
// Add real-time preview during processing
function showProgress(step, total) {
    const progress = (step / total) * 100;
    document.getElementById('progress').style.width = progress + '%';
}
```

### Style Interpolation
```python
def interpolate_styles(style1, style2, alpha=0.5):
    """Blend two styles with given weight"""
    return alpha * style1 + (1 - alpha) * style2
```

## 🐛 Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce image size or use CPU-only mode
```python
tf.config.set_visible_devices([], 'GPU')
```

2. **Slow Processing**: Reduce epochs and steps_per_epoch
```python
result = nst.transfer_style(content_path, style_path, epochs=3, steps_per_epoch=25)
```

3. **Poor Results**: Adjust loss weights or try different layer combinations

## 📈 Monitoring and Logging

### Add Detailed Logging
```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('style_transfer.log'),
        logging.StreamHandler()
    ]
)
```

### Performance Metrics
```python
import time

def measure_performance(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.info(f"Function {func.__name__} took {end_time - start_time:.2f} seconds")
        return result
    return wrapper
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -am 'Add feature'`
4. Push to branch: `git push origin feature-name`
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Original Neural Style Transfer paper by Gatys et al.
- TensorFlow team for the excellent deep learning framework
- VGG architecture by Visual Geometry Group, Oxford

## 📞 Support

For issues and questions:
- Create an issue on GitHub
- Check the troubleshooting section
- Review TensorFlow documentation

---

**⭐ Star this repository if you found it helpful!**
