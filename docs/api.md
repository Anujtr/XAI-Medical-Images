# API Documentation

## Flask Endpoints

### Main Routes

#### `GET /`
**Description:** Main web interface for uploading and analyzing chest X-ray images.

**Response:** HTML page with upload interface

---

#### `POST /upload`
**Description:** Upload and analyze a chest X-ray image.

**Request:**
- Content-Type: `multipart/form-data`
- Body:
  - `file`: Image file (DICOM, PNG, JPEG)

**Response:**
```json
{
  "success": true,
  "prediction": "Normal" | "Abnormal",
  "confidence": 0.95,
  "predicted_class": 0 | 1,
  "original_image": "data:image/png;base64,...",
  "gradcam_image": "data:image/png;base64,..."
}
```

**Error Response:**
```json
{
  "success": false,
  "error": "Error message"
}
```

**Supported File Types:**
- DICOM (.dcm, .dicom)
- PNG (.png)
- JPEG (.jpg, .jpeg)

**File Size Limit:** 16MB

---

#### `GET /health`
**Description:** Health check endpoint for monitoring.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cpu" | "cuda"
}
```

---

## Python API

### Core Classes

#### `XRayAnalyzer`
Main analyzer class that coordinates preprocessing, model inference, and Grad-CAM visualization.

```python
from src.routes import XRayAnalyzer

analyzer = XRayAnalyzer(model_path="path/to/model.pth")
result = analyzer.analyze_image("path/to/image.dcm")
```

**Methods:**
- `analyze_image(image_path: str) -> dict`: Analyze uploaded image

---

#### `ChestXrayPreprocessor`
Image preprocessing pipeline for chest X-rays.

```python
from src.preprocess import ChestXrayPreprocessor

preprocessor = ChestXrayPreprocessor()
model_input, display_image = preprocessor.process_uploaded_file("image.dcm")
```

**Methods:**
- `load_dicom(dicom_path: str) -> np.ndarray`: Load DICOM file
- `load_image(image_path: str) -> np.ndarray`: Load standard image
- `preprocess_for_model(image: np.ndarray) -> torch.Tensor`: Preprocess for model
- `process_uploaded_file(file_path: str) -> Tuple[torch.Tensor, np.ndarray]`: Complete processing

---

#### `GradCAM`
Grad-CAM implementation for model interpretability.

```python
from src.gradcam import create_gradcam

gradcam = create_gradcam(model)
heatmap = gradcam.generate_cam(input_tensor)
overlayed = gradcam.overlay_heatmap(original_image, heatmap)
```

**Methods:**
- `generate_cam(input_tensor: torch.Tensor, class_idx: int = None) -> np.ndarray`: Generate heatmap
- `overlay_heatmap(original_image: np.ndarray, heatmap: np.ndarray) -> np.ndarray`: Create overlay
- `generate_gradcam_visualization(...) -> Tuple[np.ndarray, float, int]`: Complete visualization

---

#### `ChestXrayTrainer`
Training pipeline for chest X-ray classification.

```python
from src.train import ChestXrayTrainer

trainer = ChestXrayTrainer(data_dir="data/", output_dir="models/")
trainer.setup_data(batch_size=32)
trainer.setup_model(num_classes=2)
trainer.train(num_epochs=50)
```

**Methods:**
- `setup_data(batch_size: int, val_split: float)`: Setup data loaders
- `setup_model(num_classes: int, learning_rate: float)`: Setup model and optimizer
- `train(num_epochs: int)`: Train the model
- `validate() -> Tuple[float, float]`: Validate model performance

---

## Configuration API

### Config Management

```python
from config import load_config, get_config

# Load configuration
config = load_config('config/development.yaml')

# Access configuration sections
print(config.model.architecture)
print(config.training.batch_size)
print(config.flask.port)
```

### Environment Variables

Key environment variables for configuration:

- `ENVIRONMENT`: Environment name (development/production/testing)
- `FLASK_HOST`: Flask host address
- `FLASK_PORT`: Flask port number
- `MODEL_CHECKPOINT_PATH`: Path to model checkpoint
- `DATA_DIR`: Data directory path
- `LOG_LEVEL`: Logging level

---

## Error Codes

### HTTP Status Codes

- `200`: Success
- `400`: Bad Request (invalid file, missing parameters)
- `413`: Payload Too Large (file size exceeds limit)
- `500`: Internal Server Error

### Error Messages

Common error messages returned by the API:

- `"No file provided"`: No file was uploaded
- `"No file selected"`: Empty filename
- `"File type not allowed"`: Unsupported file format
- `"File too large (max 16MB)"`: File exceeds size limit
- `"Processing error: ..."`: Error during image analysis
- `"Network error"`: Connection or server error

---

## Rate Limiting

- Default: 60 requests per minute per IP
- Configurable via `security.max_requests_per_minute` in config
- Production environments may have stricter limits

---

## Security Considerations

### File Upload Security
- File type validation based on extension and content
- File size limits enforced
- Temporary files are automatically cleaned up
- No file execution or server-side processing of untrusted content

### CORS Policy
- Configurable CORS support
- Disabled by default in production
- Can be enabled for specific origins

### Input Validation
- All file uploads are validated
- Image integrity checks performed
- DICOM files are safely parsed with pydicom

---

## Usage Examples

### cURL Examples

Upload and analyze an image:
```bash
curl -X POST \
  -F "file=@chest_xray.png" \
  http://localhost:5000/upload
```

Health check:
```bash
curl http://localhost:5000/health
```

### Python Client Example

```python
import requests

# Upload image for analysis
with open('chest_xray.dcm', 'rb') as f:
    response = requests.post(
        'http://localhost:5000/upload',
        files={'file': f}
    )
    
result = response.json()
if result['success']:
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.2%}")
else:
    print(f"Error: {result['error']}")
```

### JavaScript Example

```javascript
const formData = new FormData();
formData.append('file', fileInput.files[0]);

fetch('/upload', {
    method: 'POST',
    body: formData
})
.then(response => response.json())
.then(data => {
    if (data.success) {
        console.log('Prediction:', data.prediction);
        console.log('Confidence:', data.confidence);
    } else {
        console.error('Error:', data.error);
    }
});
```