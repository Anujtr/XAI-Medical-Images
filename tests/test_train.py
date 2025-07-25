import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
from pathlib import Path
import pandas as pd

from train import ChestXrayDataset, ChestXrayTrainer


class TestChestXrayDataset:
    """Test cases for ChestXrayDataset class."""
    
    def test_init_with_csv(self, test_data_dir):
        """Test dataset initialization with CSV file."""
        # Create mock CSV data
        csv_data = pd.DataFrame({
            'Image Index': ['normal_0.png', 'abnormal_0.png'],
            'Finding Labels': ['No Finding', 'Pneumonia']
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            csv_data.to_csv(f.name, index=False)
            csv_path = f.name
        
        try:
            dataset = ChestXrayDataset(test_data_dir, csv_path)
            
            assert len(dataset) == 2
            assert len(dataset.labels) == 2
            assert dataset.labels[0] == 0  # Normal
            assert dataset.labels[1] == 1  # Abnormal
        finally:
            os.unlink(csv_path)
    
    def test_init_without_csv(self, test_data_dir):
        """Test dataset initialization without CSV file."""
        dataset = ChestXrayDataset(test_data_dir)
        
        assert len(dataset) > 0
        assert len(dataset.labels) == len(dataset.image_paths)
    
    def test_process_labels_binary(self, test_data_dir):
        """Test binary label processing."""
        csv_data = pd.DataFrame({
            'Image Index': ['img1.png', 'img2.png', 'img3.png'],
            'Finding Labels': ['No Finding', 'Pneumonia', 'Cardiomegaly']
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            csv_data.to_csv(f.name, index=False)
            csv_path = f.name
        
        try:
            dataset = ChestXrayDataset(test_data_dir, csv_path, binary_classification=True)
            
            # No Finding -> 0, everything else -> 1
            assert dataset.labels[0] == 0
            assert dataset.labels[1] == 1
            assert dataset.labels[2] == 1
        finally:
            os.unlink(csv_path)
    
    def test_getitem_success(self, test_data_dir):
        """Test successful item retrieval."""
        dataset = ChestXrayDataset(test_data_dir)
        
        # Mock transform to avoid actual image processing
        dataset.transform = Mock(return_value=torch.randn(3, 224, 224))
        
        image, label = dataset[0]
        
        assert isinstance(image, torch.Tensor)
        assert isinstance(label, torch.Tensor)
        assert label.dtype == torch.long
    
    def test_getitem_image_loading_error(self, test_data_dir):
        """Test item retrieval with image loading error."""
        dataset = ChestXrayDataset(test_data_dir)
        
        # Corrupt the image path to cause loading error
        dataset.image_paths[0] = Path('nonexistent_image.png')
        dataset.transform = Mock(return_value=torch.randn(3, 224, 224))
        
        # Should not raise error, should return black image
        image, label = dataset[0]
        
        assert isinstance(image, torch.Tensor)
        assert isinstance(label, torch.Tensor)
    
    def test_len(self, test_data_dir):
        """Test dataset length."""
        dataset = ChestXrayDataset(test_data_dir)
        
        assert len(dataset) == len(dataset.image_paths)
        assert len(dataset) > 0


class TestChestXrayTrainer:
    """Test cases for ChestXrayTrainer class."""
    
    def test_init(self, test_data_dir):
        """Test trainer initialization."""
        trainer = ChestXrayTrainer(test_data_dir)
        
        assert trainer.data_dir == test_data_dir
        assert trainer.csv_file is None
        assert trainer.output_dir.exists()
        assert trainer.device is not None
        assert trainer.model is None
        assert trainer.train_loader is None
        assert trainer.val_loader is None
    
    def test_setup_data(self, test_data_dir):
        """Test data setup."""
        trainer = ChestXrayTrainer(test_data_dir)
        
        with patch('train.ChestXrayDataset') as mock_dataset_class:
            # Mock dataset
            mock_dataset = Mock()
            mock_dataset.__len__ = Mock(return_value=100)
            mock_dataset_class.return_value = mock_dataset
            
            # Mock random_split
            with patch('train.random_split') as mock_split:
                mock_train = Mock()
                mock_val = Mock()
                mock_train.dataset = Mock()
                mock_val.dataset = Mock()
                mock_split.return_value = (mock_train, mock_val)
                
                # Mock DataLoader
                with patch('train.DataLoader') as mock_dataloader:
                    trainer.setup_data(batch_size=16, val_split=0.2)
                    
                    assert trainer.train_loader is not None
                    assert trainer.val_loader is not None
                    assert mock_dataloader.call_count == 2
    
    def test_setup_model(self, test_data_dir):
        """Test model setup."""
        trainer = ChestXrayTrainer(test_data_dir)
        
        with patch('train.models.resnet50') as mock_resnet:
            mock_model = Mock()
            mock_model.fc = Mock()
            mock_model.fc.in_features = 2048
            mock_resnet.return_value = mock_model
            
            trainer.setup_model(num_classes=2, learning_rate=1e-4)
            
            assert trainer.model is not None
            assert trainer.criterion is not None
            assert trainer.optimizer is not None
    
    def test_train_epoch(self, test_data_dir):
        """Test single epoch training."""
        trainer = ChestXrayTrainer(test_data_dir)
        
        # Setup mock model and components
        trainer.model = Mock()
        trainer.model.train = Mock()
        trainer.optimizer = Mock()
        trainer.criterion = Mock(return_value=torch.tensor(0.5))
        
        # Mock data loader
        mock_data = [(torch.randn(2, 3, 224, 224), torch.tensor([0, 1]))]
        trainer.train_loader = mock_data
        
        # Mock model output
        trainer.model.return_value = torch.randn(2, 2)
        
        with patch('train.tqdm') as mock_tqdm:
            mock_tqdm.return_value = mock_data
            mock_progress = Mock()
            mock_progress.set_postfix = Mock()
            mock_tqdm.return_value = [(torch.randn(2, 3, 224, 224), torch.tensor([0, 1]))]
            mock_tqdm.return_value.__iter__ = Mock(return_value=iter(mock_data))
            
            loss, acc = trainer.train_epoch(0)
            
            assert isinstance(loss, float)
            assert isinstance(acc, float)
            assert 0 <= acc <= 100
    
    def test_validate(self, test_data_dir):
        """Test validation."""
        trainer = ChestXrayTrainer(test_data_dir)
        
        # Setup mock model and components
        trainer.model = Mock()
        trainer.model.eval = Mock()
        trainer.criterion = Mock(return_value=torch.tensor(0.3))
        
        # Mock data loader
        mock_data = [(torch.randn(2, 3, 224, 224), torch.tensor([0, 1]))]
        trainer.val_loader = mock_data
        
        # Mock model output
        trainer.model.return_value = torch.randn(2, 2)
        
        with patch('torch.no_grad'):
            loss, acc = trainer.validate()
            
            assert isinstance(loss, float)
            assert isinstance(acc, float)
            assert 0 <= acc <= 100
    
    def test_save_checkpoint(self, test_data_dir):
        """Test checkpoint saving."""
        trainer = ChestXrayTrainer(test_data_dir)
        
        # Setup mock model and optimizer
        trainer.model = Mock()
        trainer.model.state_dict = Mock(return_value={})
        trainer.optimizer = Mock()
        trainer.optimizer.state_dict = Mock(return_value={})
        
        with patch('torch.save') as mock_save:
            trainer.save_checkpoint(epoch=5, val_acc=0.85, is_best=True)
            
            mock_save.assert_called_once()
            # Check that the saved data includes required keys
            saved_data = mock_save.call_args[0][0]
            assert 'epoch' in saved_data
            assert 'model_state_dict' in saved_data
            assert 'optimizer_state_dict' in saved_data
            assert 'val_acc' in saved_data
    
    def test_plot_training_curves(self, test_data_dir):
        """Test training curves plotting."""
        trainer = ChestXrayTrainer(test_data_dir)
        
        # Mock data
        train_losses = [0.8, 0.6, 0.4, 0.3]
        val_losses = [0.7, 0.5, 0.4, 0.35]
        train_accs = [60, 70, 80, 85]
        val_accs = [65, 75, 78, 82]
        
        with patch('train.plt') as mock_plt:
            mock_plt.subplots.return_value = (Mock(), (Mock(), Mock()))
            
            trainer.plot_training_curves(train_losses, val_losses, train_accs, val_accs)
            
            mock_plt.subplots.assert_called_once()
            mock_plt.tight_layout.assert_called_once()
            mock_plt.savefig.assert_called_once()
            mock_plt.close.assert_called_once()
    
    def test_train_integration(self, test_data_dir):
        """Test integration of training components."""
        trainer = ChestXrayTrainer(test_data_dir)
        
        # Setup all required components
        trainer.model = Mock()
        trainer.model.train = Mock()
        trainer.model.eval = Mock()
        trainer.model.state_dict = Mock(return_value={})
        
        trainer.optimizer = Mock()
        trainer.optimizer.state_dict = Mock(return_value={})
        
        trainer.criterion = Mock(return_value=torch.tensor(0.5))
        
        trainer.train_loader = [(torch.randn(1, 3, 224, 224), torch.tensor([0]))]
        trainer.val_loader = [(torch.randn(1, 3, 224, 224), torch.tensor([0]))]
        
        trainer.model.return_value = torch.randn(1, 2)
        
        with patch('train.tqdm') as mock_tqdm:
            mock_tqdm.return_value = trainer.train_loader
            
            with patch('torch.no_grad'):
                with patch('train.SummaryWriter') as mock_writer:
                    with patch.object(trainer, 'save_checkpoint'):
                        with patch.object(trainer, 'plot_training_curves'):
                            
                            trainer.train(num_epochs=2)
                            
                            # Verify training was attempted
                            assert trainer.model.train.called
                            assert trainer.model.eval.called


class TestTrainingUtilities:
    """Test training utility functions and edge cases."""
    
    def test_dataset_with_empty_directory(self):
        """Test dataset behavior with empty directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = ChestXrayDataset(temp_dir)
            
            assert len(dataset) == 0
    
    def test_trainer_with_invalid_data_dir(self):
        """Test trainer initialization with invalid data directory."""
        invalid_dir = "/nonexistent/directory"
        
        # Should not raise error during initialization
        trainer = ChestXrayTrainer(invalid_dir)
        assert trainer.data_dir == invalid_dir
    
    def test_dataset_error_handling(self, test_data_dir):
        """Test dataset error handling for corrupted data."""
        dataset = ChestXrayDataset(test_data_dir)
        
        # Test with non-existent image
        dataset.image_paths = [Path("nonexistent.png")]
        dataset.labels = [0]
        
        # Should handle gracefully and return default image
        image, label = dataset[0]
        assert isinstance(image, torch.Tensor)
        assert isinstance(label, torch.Tensor)