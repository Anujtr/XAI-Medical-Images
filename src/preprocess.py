import cv2
import numpy as np
import pydicom
from PIL import Image
import torch
from torchvision import transforms
from typing import Union, Tuple


class ChestXrayPreprocessor:
    """Preprocessing pipeline for chest X-ray images."""
    
    def __init__(self):
        self.target_size = (224, 224)
        self.imagenet_mean = [0.485, 0.456, 0.406]
        self.imagenet_std = [0.229, 0.224, 0.225]
        
        self.transform = transforms.Compose([
            transforms.Resize(self.target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.imagenet_mean, std=self.imagenet_std)
        ])
    
    def load_dicom(self, dicom_path: str) -> np.ndarray:
        """Load DICOM file and extract pixel array."""
        try:
            ds = pydicom.dcmread(dicom_path)
            pixel_array = ds.pixel_array
            
            # Convert to uint8 if needed
            if pixel_array.dtype != np.uint8:
                # Normalize to 0-255 range
                pixel_array = pixel_array.astype(np.float32)
                pixel_array = (pixel_array - pixel_array.min()) / (pixel_array.max() - pixel_array.min())
                pixel_array = (pixel_array * 255).astype(np.uint8)
            
            return pixel_array
        except Exception as e:
            raise ValueError(f"Error loading DICOM file: {str(e)}")
    
    def load_image(self, image_path: str) -> np.ndarray:
        """Load standard image formats (JPEG, PNG, etc.)."""
        try:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError("Could not load image")
            return image
        except Exception as e:
            raise ValueError(f"Error loading image: {str(e)}")
    
    def histogram_equalization(self, image: np.ndarray) -> np.ndarray:
        """Apply histogram equalization to enhance contrast."""
        return cv2.equalizeHist(image)
    
    def preprocess_for_model(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for model inference.
        
        Args:
            image: Input grayscale image as numpy array
            
        Returns:
            Preprocessed tensor ready for model input
        """
        # Apply histogram equalization
        image = self.histogram_equalization(image)
        
        # Convert to PIL Image and make it 3-channel (RGB)
        pil_image = Image.fromarray(image).convert('RGB')
        
        # Apply transforms (resize, normalize, etc.)
        tensor = self.transform(pil_image)
        
        # Add batch dimension
        tensor = tensor.unsqueeze(0)
        
        return tensor
    
    def preprocess_for_display(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for display purposes.
        
        Args:
            image: Input grayscale image as numpy array
            
        Returns:
            Processed image suitable for display
        """
        # Apply histogram equalization
        image = self.histogram_equalization(image)
        
        # Resize to target size
        image = cv2.resize(image, self.target_size)
        
        return image
    
    def process_uploaded_file(self, file_path: str) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Process uploaded file (DICOM or standard image).
        
        Args:
            file_path: Path to uploaded file
            
        Returns:
            Tuple of (model_input_tensor, display_image)
        """
        # Determine file type and load accordingly
        if file_path.lower().endswith('.dcm'):
            image = self.load_dicom(file_path)
        else:
            image = self.load_image(file_path)
        
        # Prepare tensor for model
        model_input = self.preprocess_for_model(image)
        
        # Prepare image for display
        display_image = self.preprocess_for_display(image)
        
        return model_input, display_image
    
    def denormalize_tensor(self, tensor: torch.Tensor) -> np.ndarray:
        """
        Denormalize tensor back to displayable image.
        
        Args:
            tensor: Normalized tensor from model
            
        Returns:
            Denormalized numpy array
        """
        tensor = tensor.clone()
        
        # Denormalize
        for t, m, s in zip(tensor, self.imagenet_mean, self.imagenet_std):
            t.mul_(s).add_(m)
        
        # Convert to numpy and clip values
        image = tensor.numpy().transpose(1, 2, 0)
        image = np.clip(image, 0, 1)
        image = (image * 255).astype(np.uint8)
        
        return image


def create_preprocessor() -> ChestXrayPreprocessor:
    """Factory function to create preprocessor instance."""
    return ChestXrayPreprocessor()