import os
import uuid
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
import torch
import torchvision.models as models
import numpy as np
from PIL import Image
import io
import base64

from .preprocess import create_preprocessor
from .gradcam import create_gradcam


class XRayAnalyzer:
    """Main analyzer class that coordinates preprocessing, model inference, and Grad-CAM."""
    
    def __init__(self, model_path: str = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.preprocessor = create_preprocessor()
        self.model = self._load_model(model_path)
        self.gradcam = create_gradcam(self.model)
        self.class_names = ['Normal', 'Abnormal']
    
    def _load_model(self, model_path: str = None) -> torch.nn.Module:
        """Load the trained ResNet-50 model."""
        model = models.resnet50(pretrained=True)
        model.fc = torch.nn.Linear(model.fc.in_features, 2)  # Binary classification
        
        if model_path and os.path.exists(model_path):
            try:
                checkpoint = torch.load(model_path, map_location=self.device)
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
                print(f"Loaded model from {model_path}")
            except Exception as e:
                print(f"Warning: Could not load model from {model_path}: {e}")
                print("Using pretrained ResNet-50 with random final layer")
        else:
            print("No model checkpoint found. Using pretrained ResNet-50 with random final layer")
        
        model.to(self.device)
        model.eval()
        return model
    
    def analyze_image(self, image_path: str) -> dict:
        """
        Analyze uploaded image and generate Grad-CAM visualization.
        
        Args:
            image_path: Path to uploaded image
            
        Returns:
            Dictionary containing analysis results
        """
        try:
            # Preprocess image
            model_input, display_image = self.preprocessor.process_uploaded_file(image_path)
            model_input = model_input.to(self.device)
            
            # Generate Grad-CAM visualization
            overlayed_image, confidence, predicted_class = self.gradcam.generate_gradcam_visualization(
                model_input, display_image
            )
            
            # Convert images to base64 for web display
            original_b64 = self._image_to_base64(display_image)
            gradcam_b64 = self._image_to_base64(overlayed_image)
            
            return {
                'success': True,
                'prediction': self.class_names[predicted_class],
                'confidence': round(confidence, 3),
                'predicted_class': int(predicted_class),
                'original_image': original_b64,
                'gradcam_image': gradcam_b64
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def _image_to_base64(self, image: np.ndarray) -> str:
        """Convert numpy image to base64 string."""
        if len(image.shape) == 2:  # Grayscale
            pil_image = Image.fromarray(image, mode='L')
        else:  # RGB
            pil_image = Image.fromarray(image, mode='RGB')
        
        buffer = io.BytesIO()
        pil_image.save(buffer, format='PNG')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        return f"data:image/png;base64,{img_str}"


def create_app(model_path: str = None) -> Flask:
    """Create and configure Flask application."""
    app = Flask(__name__, template_folder='../templates', static_folder='../static')
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
    app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), '..', 'static', 'uploads')
    
    # Ensure upload folder exists
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # Initialize analyzer
    analyzer = XRayAnalyzer(model_path)
    
    # Allowed file extensions
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'dcm', 'dicom'}
    
    def allowed_file(filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    
    @app.route('/')
    def index():
        """Main page."""
        return render_template('index.html')
    
    @app.route('/upload', methods=['POST'])
    def upload_file():
        """Handle file upload and analysis."""
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file provided'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'})
        
        if not allowed_file(file.filename):
            return jsonify({'success': False, 'error': 'File type not allowed'})
        
        try:
            # Generate unique filename
            filename = str(uuid.uuid4()) + '_' + secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            # Save uploaded file
            file.save(filepath)
            
            # Analyze image
            result = analyzer.analyze_image(filepath)
            
            # Clean up uploaded file
            try:
                os.remove(filepath)
            except:
                pass  # Ignore cleanup errors
            
            return jsonify(result)
            
        except Exception as e:
            return jsonify({'success': False, 'error': f'Processing error: {str(e)}'})
    
    @app.route('/health')
    def health_check():
        """Health check endpoint."""
        return jsonify({
            'status': 'healthy',
            'model_loaded': analyzer.model is not None,
            'device': str(analyzer.device)
        })
    
    @app.errorhandler(413)
    def too_large(e):
        return jsonify({'success': False, 'error': 'File too large (max 16MB)'}), 413
    
    @app.errorhandler(500)
    def internal_error(error):
        return jsonify({'success': False, 'error': 'Internal server error'}), 500
    
    return app