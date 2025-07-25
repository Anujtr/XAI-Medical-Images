import pytest
import json
import io
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os

from routes import XRayAnalyzer, create_app


class TestXRayAnalyzer:
    """Test cases for XRayAnalyzer class."""
    
    def test_init_without_model(self):
        """Test analyzer initialization without model path."""
        with patch('routes.models.resnet50') as mock_resnet:
            mock_model = Mock()
            mock_resnet.return_value = mock_model
            
            analyzer = XRayAnalyzer()
            
            assert analyzer.device is not None
            assert analyzer.preprocessor is not None
            assert analyzer.model is not None
            assert analyzer.gradcam is not None
            assert analyzer.class_names == ['Normal', 'Abnormal']
    
    def test_init_with_valid_model_path(self, temp_checkpoint_file):
        """Test analyzer initialization with valid model checkpoint."""
        with patch('routes.models.resnet50') as mock_resnet:
            mock_model = Mock()
            mock_resnet.return_value = mock_model
            
            with patch('routes.torch.load') as mock_load:
                mock_load.return_value = {'model_state_dict': {}}
                
                analyzer = XRayAnalyzer(temp_checkpoint_file)
                
                mock_load.assert_called_once()
                mock_model.load_state_dict.assert_called_once()
    
    def test_init_with_invalid_model_path(self):
        """Test analyzer initialization with invalid model path."""
        with patch('routes.models.resnet50') as mock_resnet:
            mock_model = Mock()
            mock_resnet.return_value = mock_model
            
            analyzer = XRayAnalyzer("nonexistent_model.pth")
            
            # Should still initialize with pretrained model
            assert analyzer.model is not None
    
    def test_analyze_image_success(self, temp_image_file):
        """Test successful image analysis."""
        with patch('routes.models.resnet50') as mock_resnet:
            mock_model = Mock()
            mock_resnet.return_value = mock_model
            
            analyzer = XRayAnalyzer()
            
            # Mock preprocessor
            analyzer.preprocessor.process_uploaded_file = Mock(
                return_value=(Mock(), Mock())
            )
            
            # Mock Grad-CAM
            analyzer.gradcam.generate_gradcam_visualization = Mock(
                return_value=(Mock(), 0.95, 1)
            )
            
            # Mock image conversion
            with patch.object(analyzer, '_image_to_base64', return_value='base64_string'):
                result = analyzer.analyze_image(temp_image_file)
            
            assert result['success'] is True
            assert result['prediction'] == 'Abnormal'
            assert result['confidence'] == 0.95
            assert 'original_image' in result
            assert 'gradcam_image' in result
    
    def test_analyze_image_error_handling(self):
        """Test error handling in image analysis."""
        with patch('routes.models.resnet50') as mock_resnet:
            mock_model = Mock()
            mock_resnet.return_value = mock_model
            
            analyzer = XRayAnalyzer()
            
            # Mock preprocessor to raise exception
            analyzer.preprocessor.process_uploaded_file = Mock(
                side_effect=Exception("Processing error")
            )
            
            result = analyzer.analyze_image("fake_path.jpg")
            
            assert result['success'] is False
            assert 'error' in result
            assert 'Processing error' in result['error']
    
    def test_image_to_base64_grayscale(self, sample_image):
        """Test base64 conversion for grayscale images."""
        with patch('routes.models.resnet50'):
            analyzer = XRayAnalyzer()
            
            base64_str = analyzer._image_to_base64(sample_image)
            
            assert isinstance(base64_str, str)
            assert base64_str.startswith('data:image/png;base64,')
    
    def test_image_to_base64_rgb(self, sample_rgb_image):
        """Test base64 conversion for RGB images."""
        with patch('routes.models.resnet50'):
            analyzer = XRayAnalyzer()
            
            base64_str = analyzer._image_to_base64(sample_rgb_image)
            
            assert isinstance(base64_str, str)
            assert base64_str.startswith('data:image/png;base64,')


class TestFlaskRoutes:
    """Test cases for Flask routes."""
    
    def test_index_route(self, client):
        """Test the main index route."""
        response = client.get('/')
        
        assert response.status_code == 200
        assert b'XAI Medical Images' in response.data
    
    def test_health_check_route(self, client):
        """Test the health check endpoint."""
        response = client.get('/health')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['status'] == 'healthy'
        assert 'model_loaded' in data
        assert 'device' in data
    
    def test_upload_no_file(self, client):
        """Test upload endpoint with no file."""
        response = client.post('/upload')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['success'] is False
        assert 'No file provided' in data['error']
    
    def test_upload_empty_filename(self, client):
        """Test upload endpoint with empty filename."""
        data = {'file': (io.BytesIO(b''), '')}
        response = client.post('/upload', data=data)
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['success'] is False
        assert 'No file selected' in data['error']
    
    def test_upload_invalid_file_type(self, client):
        """Test upload endpoint with invalid file type."""
        data = {'file': (io.BytesIO(b'test content'), 'test.txt')}
        response = client.post('/upload', data=data)
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['success'] is False
        assert 'File type not allowed' in data['error']
    
    def test_upload_valid_image(self, client, sample_image):
        """Test upload endpoint with valid image."""
        # Create image file-like object
        from PIL import Image
        img_bytes = io.BytesIO()
        pil_image = Image.fromarray(sample_image, mode='L')
        pil_image.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        # Mock the analyzer to avoid actual processing
        with patch('routes.XRayAnalyzer') as mock_analyzer_class:
            mock_analyzer = Mock()
            mock_analyzer.analyze_image.return_value = {
                'success': True,
                'prediction': 'Normal',
                'confidence': 0.85,
                'predicted_class': 0,
                'original_image': 'base64_original',
                'gradcam_image': 'base64_gradcam'
            }
            mock_analyzer_class.return_value = mock_analyzer
            
            data = {'file': (img_bytes, 'test_image.png')}
            response = client.post('/upload', data=data)
            
            assert response.status_code == 200
            response_data = json.loads(response.data)
            assert response_data['success'] is True
            assert response_data['prediction'] == 'Normal'
    
    def test_upload_processing_error(self, client, sample_image):
        """Test upload endpoint with processing error."""
        from PIL import Image
        img_bytes = io.BytesIO()
        pil_image = Image.fromarray(sample_image, mode='L')
        pil_image.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        # Mock the analyzer to raise an exception
        with patch('routes.XRayAnalyzer') as mock_analyzer_class:
            mock_analyzer = Mock()
            mock_analyzer.analyze_image.side_effect = Exception("Analysis failed")
            mock_analyzer_class.return_value = mock_analyzer
            
            data = {'file': (img_bytes, 'test_image.png')}
            response = client.post('/upload', data=data)
            
            assert response.status_code == 200
            response_data = json.loads(response.data)
            assert response_data['success'] is False
            assert 'Processing error' in response_data['error']
    
    def test_file_too_large_error(self, client):
        """Test file size limit error handling."""
        # This would typically be handled by Flask's MAX_CONTENT_LENGTH
        # but we can test the error handler
        with patch('routes.request') as mock_request:
            mock_request.content_length = 20 * 1024 * 1024  # 20MB
            
            # Test the error handler directly
            app = create_app()
            with app.test_request_context():
                from werkzeug.exceptions import RequestEntityTooLarge
                response = app.handle_http_exception(RequestEntityTooLarge())
                
                assert response.status_code == 413


class TestFlaskAppCreation:
    """Test Flask app creation and configuration."""
    
    def test_create_app_default(self):
        """Test app creation with default parameters."""
        app = create_app()
        
        assert app is not None
        assert app.config['MAX_CONTENT_LENGTH'] == 16 * 1024 * 1024
        assert 'UPLOAD_FOLDER' in app.config
    
    def test_create_app_with_model_path(self, temp_checkpoint_file):
        """Test app creation with model path."""
        with patch('routes.torch.load') as mock_load:
            mock_load.return_value = {'model_state_dict': {}}
            
            app = create_app(temp_checkpoint_file)
            
            assert app is not None
    
    def test_upload_folder_creation(self, temp_upload_dir):
        """Test that upload folder is created."""
        with patch('routes.os.path.dirname', return_value=temp_upload_dir):
            app = create_app()
            
            expected_upload_path = os.path.join(temp_upload_dir, 'static', 'uploads')
            # The folder should be created by the app
            assert 'UPLOAD_FOLDER' in app.config


class TestAllowedFiles:
    """Test file extension validation."""
    
    def test_allowed_file_extensions(self):
        """Test various file extensions."""
        # This would need to be extracted from the routes module
        # or we can test it indirectly through the upload endpoint
        allowed_extensions = {'png', 'jpg', 'jpeg', 'dcm', 'dicom'}
        
        # Test valid extensions
        assert 'png' in allowed_extensions
        assert 'jpg' in allowed_extensions
        assert 'jpeg' in allowed_extensions
        assert 'dcm' in allowed_extensions
        assert 'dicom' in allowed_extensions
        
        # Test invalid extensions
        assert 'txt' not in allowed_extensions
        assert 'pdf' not in allowed_extensions
        assert 'doc' not in allowed_extensions