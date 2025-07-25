import pytest
import torch
import numpy as np
from PIL import Image
import tempfile
import os
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


@pytest.fixture
def sample_image():
    """Create a sample grayscale image for testing."""
    image = np.random.randint(0, 255, (224, 224), dtype=np.uint8)
    return image


@pytest.fixture
def sample_rgb_image():
    """Create a sample RGB image for testing."""
    image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    return image


@pytest.fixture
def sample_tensor():
    """Create a sample tensor for testing."""
    return torch.randn(1, 3, 224, 224)


@pytest.fixture
def temp_image_file(sample_image):
    """Create a temporary image file."""
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        pil_image = Image.fromarray(sample_image, mode='L')
        pil_image.save(f.name)
        yield f.name
    os.unlink(f.name)


@pytest.fixture
def temp_dicom_file():
    """Create a mock DICOM file for testing."""
    with tempfile.NamedTemporaryFile(suffix='.dcm', delete=False) as f:
        # Create minimal DICOM-like content for testing
        f.write(b'DICM')  # DICOM magic number
        yield f.name
    os.unlink(f.name)


@pytest.fixture
def mock_model():
    """Create a mock ResNet-50 model for testing."""
    import torchvision.models as models
    model = models.resnet50(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    model.eval()
    return model


@pytest.fixture
def temp_checkpoint_file():
    """Create a temporary model checkpoint file."""
    checkpoint = {
        'epoch': 1,
        'model_state_dict': {},
        'optimizer_state_dict': {},
        'val_acc': 0.85
    }
    
    with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
        torch.save(checkpoint, f.name)
        yield f.name
    os.unlink(f.name)


@pytest.fixture
def temp_upload_dir():
    """Create a temporary upload directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture(scope="session")
def test_data_dir():
    """Create a temporary test data directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create sample directory structure
        data_dir = Path(temp_dir)
        (data_dir / "normal").mkdir()
        (data_dir / "abnormal").mkdir()
        
        # Create sample images
        for i in range(3):
            sample_img = np.random.randint(0, 255, (224, 224), dtype=np.uint8)
            pil_img = Image.fromarray(sample_img, mode='L')
            pil_img.save(data_dir / "normal" / f"normal_{i}.png")
            pil_img.save(data_dir / "abnormal" / f"abnormal_{i}.png")
        
        yield str(data_dir)


@pytest.fixture
def flask_app():
    """Create a Flask app for testing."""
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
    
    from routes import create_app
    app = create_app()
    app.config['TESTING'] = True
    return app


@pytest.fixture
def client(flask_app):
    """Create a test client."""
    return flask_app.test_client()