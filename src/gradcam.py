import torch
import torch.nn.functional as F
import cv2
import numpy as np
from typing import List, Tuple
import matplotlib.pyplot as plt
import matplotlib.cm as cm


class GradCAM:
    """Grad-CAM implementation for ResNet-50 model."""
    
    def __init__(self, model: torch.nn.Module, target_layer_name: str = 'layer4'):
        """
        Initialize Grad-CAM.
        
        Args:
            model: PyTorch model (ResNet-50)
            target_layer_name: Name of target layer for Grad-CAM
        """
        self.model = model
        self.model.eval()
        self.target_layer_name = target_layer_name
        
        # Storage for gradients and activations
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks on target layer."""
        target_layer = self._get_target_layer()
        
        def forward_hook(module, input, output):
            self.activations = output
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)
    
    def _get_target_layer(self):
        """Get the target layer from the model."""
        return getattr(self.model, self.target_layer_name)
    
    def generate_cam(self, input_tensor: torch.Tensor, class_idx: int = None) -> np.ndarray:
        """
        Generate Grad-CAM heatmap.
        
        Args:
            input_tensor: Input tensor for the model
            class_idx: Target class index (if None, uses predicted class)
            
        Returns:
            Grad-CAM heatmap as numpy array
        """
        # Forward pass
        self.model.zero_grad()
        output = self.model(input_tensor)
        
        # Get target class
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()
        
        # Backward pass
        target_score = output[0, class_idx]
        target_score.backward(retain_graph=True)
        
        # Get gradients and activations
        gradients = self.gradients[0].cpu()  # [C, H, W]
        activations = self.activations[0].cpu()  # [C, H, W]
        
        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=(1, 2))  # [C]
        
        # Weighted combination of activation maps
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)  # [H, W]
        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]
        
        # Apply ReLU to the result
        cam = F.relu(cam)
        
        # Normalize to 0-1
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        
        return cam.numpy()
    
    def overlay_heatmap(self, original_image: np.ndarray, heatmap: np.ndarray, 
                       alpha: float = 0.4, colormap: str = 'jet') -> np.ndarray:
        """
        Overlay heatmap on original image.
        
        Args:
            original_image: Original grayscale image [H, W] or RGB image [H, W, 3]
            heatmap: Grad-CAM heatmap [H, W]
            alpha: Transparency factor for overlay
            colormap: Matplotlib colormap name
            
        Returns:
            Overlayed image
        """
        # Ensure original image is 3-channel
        if len(original_image.shape) == 2:
            original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
        elif len(original_image.shape) == 3 and original_image.shape[2] == 1:
            original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
        
        # Resize heatmap to match original image
        heatmap_resized = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
        
        # Apply colormap to heatmap
        colormap_func = cm.get_cmap(colormap)
        heatmap_colored = colormap_func(heatmap_resized)[:, :, :3]  # Remove alpha channel
        heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
        
        # Normalize original image to 0-255 if needed
        if original_image.max() <= 1.0:
            original_image = (original_image * 255).astype(np.uint8)
        
        # Overlay heatmap on original image
        overlayed = cv2.addWeighted(original_image, 1 - alpha, heatmap_colored, alpha, 0)
        
        return overlayed
    
    def generate_gradcam_visualization(self, input_tensor: torch.Tensor, 
                                     original_image: np.ndarray,
                                     class_idx: int = None) -> Tuple[np.ndarray, float, int]:
        """
        Generate complete Grad-CAM visualization.
        
        Args:
            input_tensor: Input tensor for the model
            original_image: Original image for overlay
            class_idx: Target class index
            
        Returns:
            Tuple of (overlayed_image, confidence_score, predicted_class)
        """
        # Get model prediction
        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            confidence = torch.max(probabilities).item()
            predicted_class = torch.argmax(probabilities, dim=1).item()
        
        # Generate Grad-CAM heatmap
        heatmap = self.generate_cam(input_tensor, class_idx)
        
        # Create overlay
        overlayed_image = self.overlay_heatmap(original_image, heatmap)
        
        return overlayed_image, confidence, predicted_class
    
    def save_visualization(self, overlayed_image: np.ndarray, 
                          confidence: float, predicted_class: int,
                          output_path: str, class_names: List[str] = None):
        """
        Save Grad-CAM visualization to file.
        
        Args:
            overlayed_image: Overlayed image with heatmap
            confidence: Model confidence score
            predicted_class: Predicted class index
            output_path: Path to save the visualization
            class_names: List of class names
        """
        plt.figure(figsize=(10, 8))
        plt.imshow(overlayed_image)
        plt.axis('off')
        
        # Add title with prediction info
        if class_names:
            class_name = class_names[predicted_class]
        else:
            class_name = f"Class {predicted_class}"
        
        title = f"Prediction: {class_name} (Confidence: {confidence:.2f})"
        plt.title(title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()


class ResNetGradCAM(GradCAM):
    """Specialized Grad-CAM for ResNet architectures."""
    
    def __init__(self, model: torch.nn.Module):
        """Initialize with ResNet-specific target layer."""
        super().__init__(model, target_layer_name='layer4')
    
    def get_feature_maps(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Get feature maps from the target layer.
        
        Args:
            input_tensor: Input tensor
            
        Returns:
            Feature maps from target layer
        """
        # Forward pass up to target layer
        x = input_tensor
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)  # Target layer output
        
        return x


def create_gradcam(model: torch.nn.Module) -> ResNetGradCAM:
    """
    Factory function to create Grad-CAM instance for ResNet.
    
    Args:
        model: Trained ResNet model
        
    Returns:
        Configured Grad-CAM instance
    """
    return ResNetGradCAM(model)