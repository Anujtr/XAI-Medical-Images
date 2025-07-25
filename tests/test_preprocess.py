import pytest
import numpy as np
import torch
from PIL import Image
import tempfile
import os

from preprocess import ChestXrayPreprocessor, create_preprocessor


class TestChestXrayPreprocessor:
    """Test cases for ChestXrayPreprocessor class."""
    
    def test_init(self):
        """Test preprocessor initialization."""
        preprocessor = ChestXrayPreprocessor()
        
        assert preprocessor.target_size == (224, 224)
        assert preprocessor.imagenet_mean == [0.485, 0.456, 0.406]
        assert preprocessor.imagenet_std == [0.229, 0.224, 0.225]
        assert preprocessor.transform is not None
    
    def test_histogram_equalization(self, sample_image):
        """Test histogram equalization function."""
        preprocessor = ChestXrayPreprocessor()
        
        # Test with sample image
        equalized = preprocessor.histogram_equalization(sample_image)
        
        assert equalized.shape == sample_image.shape
        assert equalized.dtype == np.uint8
        assert 0 <= equalized.min() <= equalized.max() <= 255
    
    def test_load_image(self, temp_image_file):
        """Test loading standard image formats."""
        preprocessor = ChestXrayPreprocessor()
        
        # Test loading valid image
        image = preprocessor.load_image(temp_image_file)
        
        assert isinstance(image, np.ndarray)
        assert len(image.shape) == 2  # Grayscale
        assert image.dtype == np.uint8
    
    def test_load_image_invalid_path(self):
        """Test loading image with invalid path."""
        preprocessor = ChestXrayPreprocessor()
        
        with pytest.raises(ValueError, match="Error loading image"):
            preprocessor.load_image("nonexistent_file.jpg")
    
    def test_preprocess_for_model(self, sample_image):
        """Test preprocessing for model input."""
        preprocessor = ChestXrayPreprocessor()
        
        tensor = preprocessor.preprocess_for_model(sample_image)
        
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (1, 3, 224, 224)  # Batch size 1, RGB, 224x224
        assert tensor.dtype == torch.float32
    
    def test_preprocess_for_display(self, sample_image):
        """Test preprocessing for display."""
        preprocessor = ChestXrayPreprocessor()
        
        display_image = preprocessor.preprocess_for_display(sample_image)
        
        assert isinstance(display_image, np.ndarray)
        assert display_image.shape == (224, 224)
        assert display_image.dtype == np.uint8
    
    def test_process_uploaded_file_image(self, temp_image_file):
        """Test processing uploaded standard image file."""
        preprocessor = ChestXrayPreprocessor()
        
        model_input, display_image = preprocessor.process_uploaded_file(temp_image_file)
        
        # Check model input
        assert isinstance(model_input, torch.Tensor)
        assert model_input.shape == (1, 3, 224, 224)
        
        # Check display image
        assert isinstance(display_image, np.ndarray)
        assert display_image.shape == (224, 224)
    
    def test_denormalize_tensor(self, sample_tensor):
        """Test tensor denormalization."""
        preprocessor = ChestXrayPreprocessor()
        
        # Use first sample from batch
        tensor_sample = sample_tensor[0]  # Shape: (3, 224, 224)
        
        denormalized = preprocessor.denormalize_tensor(tensor_sample)
        
        assert isinstance(denormalized, np.ndarray)
        assert denormalized.shape == (224, 224, 3)  # HWC format
        assert denormalized.dtype == np.uint8
        assert 0 <= denormalized.min() <= denormalized.max() <= 255
    
    def test_create_preprocessor_factory(self):
        """Test factory function."""
        preprocessor = create_preprocessor()
        
        assert isinstance(preprocessor, ChestXrayPreprocessor)
    
    def test_preprocessing_consistency(self, sample_image):
        """Test that preprocessing is consistent across calls."""
        preprocessor = ChestXrayPreprocessor()
        
        # Process same image twice
        tensor1 = preprocessor.preprocess_for_model(sample_image.copy())
        tensor2 = preprocessor.preprocess_for_model(sample_image.copy())
        
        # Should be identical
        assert torch.allclose(tensor1, tensor2, atol=1e-6)
    
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        preprocessor = ChestXrayPreprocessor()
        
        # Test with very small image
        small_image = np.random.randint(0, 255, (10, 10), dtype=np.uint8)
        tensor = preprocessor.preprocess_for_model(small_image)
        assert tensor.shape == (1, 3, 224, 224)  # Should be resized
        
        # Test with very large values
        large_image = np.full((100, 100), 255, dtype=np.uint8)
        tensor = preprocessor.preprocess_for_model(large_image)
        assert tensor.shape == (1, 3, 224, 224)
        
        # Test with zero image
        zero_image = np.zeros((100, 100), dtype=np.uint8)
        tensor = preprocessor.preprocess_for_model(zero_image)
        assert tensor.shape == (1, 3, 224, 224)