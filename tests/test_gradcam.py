import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch

from gradcam import GradCAM, ResNetGradCAM, create_gradcam


class TestGradCAM:
    """Test cases for GradCAM class."""
    
    def test_init(self, mock_model):
        """Test GradCAM initialization."""
        gradcam = GradCAM(mock_model, target_layer_name='layer4')
        
        assert gradcam.model == mock_model
        assert gradcam.target_layer_name == 'layer4'
        assert gradcam.gradients is None
        assert gradcam.activations is None
    
    def test_generate_cam_shape(self, mock_model, sample_tensor):
        """Test Grad-CAM generation output shape."""
        # Mock the target layer to avoid actual forward pass issues
        with patch.object(mock_model, 'layer4') as mock_layer:
            mock_layer.register_forward_hook = Mock()
            mock_layer.register_backward_hook = Mock()
            
            gradcam = GradCAM(mock_model, target_layer_name='layer4')
            
            # Mock activations and gradients
            gradcam.activations = torch.randn(1, 2048, 7, 7)  # ResNet-50 layer4 output
            gradcam.gradients = torch.randn(1, 2048, 7, 7)
            
            # Mock model output
            with patch.object(mock_model, 'forward', return_value=torch.randn(1, 2)):
                with patch.object(mock_model, 'zero_grad'):
                    heatmap = gradcam.generate_cam(sample_tensor)
            
            assert isinstance(heatmap, np.ndarray)
            assert len(heatmap.shape) == 2  # 2D heatmap
            assert 0 <= heatmap.min() <= heatmap.max() <= 1  # Normalized
    
    def test_overlay_heatmap_grayscale(self, sample_image):
        """Test heatmap overlay on grayscale image."""
        mock_model = Mock()
        gradcam = GradCAM(mock_model)
        
        # Create dummy heatmap
        heatmap = np.random.rand(224, 224)
        
        overlayed = gradcam.overlay_heatmap(sample_image, heatmap)
        
        assert isinstance(overlayed, np.ndarray)
        assert len(overlayed.shape) == 3  # RGB output
        assert overlayed.shape == (224, 224, 3)
        assert overlayed.dtype == np.uint8
    
    def test_overlay_heatmap_rgb(self, sample_rgb_image):
        """Test heatmap overlay on RGB image."""
        mock_model = Mock()
        gradcam = GradCAM(mock_model)
        
        # Create dummy heatmap
        heatmap = np.random.rand(224, 224)
        
        overlayed = gradcam.overlay_heatmap(sample_rgb_image, heatmap)
        
        assert isinstance(overlayed, np.ndarray)
        assert overlayed.shape == (224, 224, 3)
        assert overlayed.dtype == np.uint8
    
    def test_overlay_heatmap_different_sizes(self, sample_image):
        """Test heatmap overlay with different image and heatmap sizes."""
        mock_model = Mock()
        gradcam = GradCAM(mock_model)
        
        # Create heatmap with different size
        heatmap = np.random.rand(56, 56)  # Different from image size
        
        overlayed = gradcam.overlay_heatmap(sample_image, heatmap)
        
        assert overlayed.shape == (224, 224, 3)  # Should match original image size
    
    def test_overlay_heatmap_alpha_parameter(self, sample_image):
        """Test heatmap overlay with different alpha values."""
        mock_model = Mock()
        gradcam = GradCAM(mock_model)
        
        heatmap = np.ones((224, 224)) * 0.5
        
        # Test different alpha values
        overlay1 = gradcam.overlay_heatmap(sample_image, heatmap, alpha=0.2)
        overlay2 = gradcam.overlay_heatmap(sample_image, heatmap, alpha=0.8)
        
        assert not np.array_equal(overlay1, overlay2)  # Should be different
    
    def test_overlay_heatmap_colormap(self, sample_image):
        """Test heatmap overlay with different colormaps."""
        mock_model = Mock()
        gradcam = GradCAM(mock_model)
        
        heatmap = np.random.rand(224, 224)
        
        # Test different colormaps
        overlay_jet = gradcam.overlay_heatmap(sample_image, heatmap, colormap='jet')
        overlay_hot = gradcam.overlay_heatmap(sample_image, heatmap, colormap='hot')
        
        assert overlay_jet.shape == overlay_hot.shape
        assert not np.array_equal(overlay_jet, overlay_hot)


class TestResNetGradCAM:
    """Test cases for ResNetGradCAM class."""
    
    def test_init(self, mock_model):
        """Test ResNetGradCAM initialization."""
        gradcam = ResNetGradCAM(mock_model)
        
        assert gradcam.model == mock_model
        assert gradcam.target_layer_name == 'layer4'
    
    def test_get_feature_maps(self, mock_model, sample_tensor):
        """Test feature maps extraction."""
        # Mock all the required ResNet layers
        mock_model.conv1 = Mock(return_value=sample_tensor)
        mock_model.bn1 = Mock(return_value=sample_tensor)
        mock_model.relu = Mock(return_value=sample_tensor)
        mock_model.maxpool = Mock(return_value=sample_tensor)
        mock_model.layer1 = Mock(return_value=sample_tensor)
        mock_model.layer2 = Mock(return_value=sample_tensor)
        mock_model.layer3 = Mock(return_value=sample_tensor)
        mock_model.layer4 = Mock(return_value=sample_tensor)
        
        gradcam = ResNetGradCAM(mock_model)
        
        feature_maps = gradcam.get_feature_maps(sample_tensor)
        
        assert isinstance(feature_maps, torch.Tensor)
        # Verify all layers were called in sequence
        mock_model.conv1.assert_called_once()
        mock_model.bn1.assert_called_once()
        mock_model.relu.assert_called_once()
        mock_model.maxpool.assert_called_once()
        mock_model.layer1.assert_called_once()
        mock_model.layer2.assert_called_once()
        mock_model.layer3.assert_called_once()
        mock_model.layer4.assert_called_once()
    
    def test_create_gradcam_factory(self, mock_model):
        """Test factory function."""
        gradcam = create_gradcam(mock_model)
        
        assert isinstance(gradcam, ResNetGradCAM)
        assert gradcam.model == mock_model


class TestGradCAMIntegration:
    """Integration tests for Grad-CAM functionality."""
    
    def test_end_to_end_gradcam_generation(self, mock_model, sample_tensor, sample_image):
        """Test complete Grad-CAM generation pipeline."""
        # Setup mock model behavior
        mock_model.eval()
        
        with patch.object(mock_model, 'layer4') as mock_layer:
            mock_layer.register_forward_hook = Mock()
            mock_layer.register_backward_hook = Mock()
            
            gradcam = ResNetGradCAM(mock_model)
            
            # Mock the necessary components
            gradcam.activations = torch.randn(1, 2048, 7, 7)
            gradcam.gradients = torch.randn(1, 2048, 7, 7)
            
            with patch.object(mock_model, 'forward', return_value=torch.randn(1, 2)):
                with patch.object(mock_model, 'zero_grad'):
                    overlayed, confidence, predicted_class = gradcam.generate_gradcam_visualization(
                        sample_tensor, sample_image
                    )
            
            assert isinstance(overlayed, np.ndarray)
            assert overlayed.shape == (224, 224, 3)
            assert isinstance(confidence, float)
            assert isinstance(predicted_class, int)
            assert 0 <= confidence <= 1
            assert predicted_class in [0, 1]
    
    def test_gradcam_with_different_class_indices(self, mock_model, sample_tensor):
        """Test Grad-CAM generation with specific class indices."""
        with patch.object(mock_model, 'layer4') as mock_layer:
            mock_layer.register_forward_hook = Mock()
            mock_layer.register_backward_hook = Mock()
            
            gradcam = GradCAM(mock_model)
            gradcam.activations = torch.randn(1, 2048, 7, 7)
            gradcam.gradients = torch.randn(1, 2048, 7, 7)
            
            with patch.object(mock_model, 'forward', return_value=torch.randn(1, 2)):
                with patch.object(mock_model, 'zero_grad'):
                    # Test with specific class index
                    heatmap_class_0 = gradcam.generate_cam(sample_tensor, class_idx=0)
                    heatmap_class_1 = gradcam.generate_cam(sample_tensor, class_idx=1)
            
            assert isinstance(heatmap_class_0, np.ndarray)
            assert isinstance(heatmap_class_1, np.ndarray)
            assert heatmap_class_0.shape == heatmap_class_1.shape
    
    def test_gradcam_error_handling(self, mock_model, sample_tensor):
        """Test error handling in Grad-CAM generation."""
        gradcam = GradCAM(mock_model)
        
        # Test with missing activations/gradients
        gradcam.activations = None
        gradcam.gradients = None
        
        with patch.object(mock_model, 'forward', return_value=torch.randn(1, 2)):
            with patch.object(mock_model, 'zero_grad'):
                with pytest.raises(AttributeError):
                    gradcam.generate_cam(sample_tensor)