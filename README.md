# XAI Medical Images - Chest X-ray Analysis with Grad-CAM

A Flask web application for analyzing chest X-rays using deep learning with explainable AI visualization through Grad-CAM (Gradient-weighted Class Activation Mapping).

## Features

- **Deep Learning Model**: ResNet-50 pretrained on ImageNet, fine-tuned for chest X-ray classification
- **Explainable AI**: Grad-CAM visualization showing model attention areas
- **DICOM Support**: Upload and process medical imaging files (.dcm format)
- **Web Interface**: Interactive web application for image upload and analysis
- **Containerized**: Docker support for easy deployment

## Quick Start

### Prerequisites

- Python 3.10+
- Docker (optional, for containerized deployment)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd XAI-Medical-Images
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python run.py
```

4. Open your browser and navigate to `http://localhost:5000`

### Docker Deployment

1. Build the Docker image:
```bash
docker build -t xai-medical-images .
```

2. Run the container:
```bash
docker run -p 5000:5000 xai-medical-images
```

## Usage

1. **Upload Image**: Click "Choose File" to upload a chest X-ray image (DICOM or standard image formats)
2. **Analyze**: Click "Analyze" to run the AI model on your image
3. **View Results**: See the classification result and Grad-CAM heatmap overlay
4. **Interpret**: The heatmap shows which areas the model focused on for its decision

## Technical Details

### Model Architecture
- **Base Model**: ResNet-50 from torchvision.models
- **Pretraining**: ImageNet weights
- **Fine-tuning**: Binary classification for chest X-ray abnormality detection
- **Input Size**: 224x224 pixels
- **Preprocessing**: Histogram equalization, resizing, ImageNet normalization

### Grad-CAM Implementation
- **Target Layer**: Final convolutional layer of ResNet-50
- **Method**: Forward and backward hooks for gradient extraction
- **Visualization**: ReLU applied to gradients for visual clarity
- **Overlay**: Heatmap superimposed on original image

### Data Processing
- **DICOM Support**: Uses pydicom for medical image parsing
- **Preprocessing Pipeline**: 
  - Histogram equalization (cv2.equalizeHist)
  - Resize to 224x224
  - Normalization with ImageNet statistics
- **Supported Formats**: DICOM (.dcm), JPEG, PNG

## Project Structure

```
XAI-Medical-Images/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── Dockerfile               # Container configuration
├── run.py                   # Flask application entry point
├── src/                     # Source code
│   ├── __init__.py
│   ├── routes.py            # Flask routes and endpoints
│   ├── preprocess.py        # Image preprocessing utilities
│   ├── gradcam.py           # Grad-CAM implementation
│   └── train.py             # Model training script
├── static/                  # Static web assets
│   ├── css/                 # Stylesheets
│   ├── js/                  # JavaScript files
│   └── uploads/             # Temporary file storage
├── templates/               # HTML templates
│   └── index.html           # Main web interface
└── models/                  # Model storage
    └── checkpoints/         # Trained model weights
```

## Training Your Own Model

To train the model on the NIH ChestXray14 dataset:

1. Download the NIH ChestXray14 dataset
2. Update the data path in `src/train.py`
3. Run training:
```bash
python src/train.py
```

The training script includes:
- BCEWithLogitsLoss for binary classification
- Adam optimizer
- Optional TensorBoard logging for monitoring

## API Endpoints

- `GET /` - Main web interface
- `POST /upload` - Upload and analyze image
- `GET /result/<filename>` - Get analysis results

## Dependencies

- **Core**: torch (2.2+), torchvision (0.17+), flask (3.0+)
- **Image Processing**: opencv-python (4.10+), pydicom (2.4+)
- **Visualization**: matplotlib, numpy
- **Web**: flask-cors
- **Deployment**: gunicorn (optional)

## Model Performance

The model is designed for educational and research purposes. For clinical use, additional validation and regulatory approval would be required.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is for educational and research purposes. Please ensure compliance with relevant medical data regulations when using with real patient data.

## Disclaimer

This tool is for educational and research purposes only. It should not be used for actual medical diagnosis. Always consult qualified healthcare professionals for medical decisions.