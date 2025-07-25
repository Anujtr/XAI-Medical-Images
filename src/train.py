#!/usr/bin/env python3
"""
Training script for chest X-ray classification model.
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

from preprocess import create_preprocessor


class ChestXrayDataset(Dataset):
    """Dataset class for chest X-ray images."""
    
    def __init__(self, data_dir: str, csv_file: str = None, transform=None, binary_classification=True):
        """
        Initialize dataset.
        
        Args:
            data_dir: Directory containing images
            csv_file: CSV file with labels (optional)
            transform: Image transformations
            binary_classification: If True, convert to binary normal/abnormal
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.binary_classification = binary_classification
        
        # Load data
        if csv_file and os.path.exists(csv_file):
            self.data_df = pd.read_csv(csv_file)
            self.image_paths = [self.data_dir / img for img in self.data_df['Image Index']]
            self.labels = self._process_labels(self.data_df)
        else:
            # Fallback: assume directory structure organized by class
            self.image_paths, self.labels = self._load_from_directory()
    
    def _process_labels(self, df):
        """Process labels from CSV file."""
        if self.binary_classification:
            # Convert to binary: 'No Finding' -> 0, everything else -> 1
            labels = []
            for _, row in df.iterrows():
                finding = row.get('Finding Labels', 'No Finding')
                if finding == 'No Finding':
                    labels.append(0)  # Normal
                else:
                    labels.append(1)  # Abnormal
            return labels
        else:
            # Multi-class classification (implement as needed)
            raise NotImplementedError("Multi-class classification not implemented")
    
    def _load_from_directory(self):
        """Load images from directory structure (fallback method)."""
        image_paths = []
        labels = []
        
        # Look for images in subdirectories
        for class_dir in self.data_dir.iterdir():
            if class_dir.is_dir():
                class_label = 0 if class_dir.name.lower() in ['normal', 'no_finding'] else 1
                for img_file in class_dir.glob('*.png'):
                    image_paths.append(img_file)
                    labels.append(class_label)
        
        # If no subdirectories, assume all images are normal (for testing)
        if not image_paths:
            for img_file in self.data_dir.glob('*.png'):
                image_paths.append(img_file)
                labels.append(0)  # Default to normal
        
        return image_paths, labels
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a black image as fallback
            image = Image.new('RGB', (224, 224), color=0)
        
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label, dtype=torch.long)


class ChestXrayTrainer:
    """Training class for chest X-ray classification."""
    
    def __init__(self, data_dir: str, csv_file: str = None, output_dir: str = 'models/checkpoints'):
        self.data_dir = data_dir
        self.csv_file = csv_file
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize components
        self.model = None
        self.criterion = None
        self.optimizer = None
        self.train_loader = None
        self.val_loader = None
        self.writer = None
    
    def setup_data(self, batch_size: int = 32, val_split: float = 0.2):
        """Setup data loaders."""
        # Define transforms
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Create dataset
        full_dataset = ChestXrayDataset(self.data_dir, self.csv_file, train_transform)
        
        # Split into train and validation
        val_size = int(len(full_dataset) * val_split)
        train_size = len(full_dataset) - val_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
        
        # Update validation dataset transform
        val_dataset.dataset.transform = val_transform
        
        # Create data loaders
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
    
    def setup_model(self, num_classes: int = 2, learning_rate: float = 1e-4):
        """Setup model, loss function, and optimizer."""
        # Load pretrained ResNet-50
        self.model = models.resnet50(pretrained=True)
        
        # Modify final layer for chest X-ray classification
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        
        # Move to device
        self.model = self.model.to(self.device)
        
        # Loss function and optimizer
        self.criterion = nn.BCEWithLogitsLoss() if num_classes == 2 else nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        
        print(f"Model setup complete. Parameters: {sum(p.numel() for p in self.model.parameters())}")
    
    def train_epoch(self, epoch: int):
        """Train for one epoch."""
        self.model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}')
        for batch_idx, (data, target) in enumerate(progress_bar):
            data, target = data.to(self.device), target.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            output = self.model(data)
            
            # Calculate loss
            if output.shape[1] == 2:  # Binary classification
                target_one_hot = torch.zeros_like(output)
                target_one_hot.scatter_(1, target.unsqueeze(1), 1)
                loss = self.criterion(output, target_one_hot.float())
            else:
                loss = self.criterion(output, target)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total_samples += target.size(0)
            correct_predictions += (predicted == target).sum().item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100 * correct_predictions / total_samples:.2f}%'
            })
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100 * correct_predictions / total_samples
        
        return epoch_loss, epoch_acc
    
    def validate(self):
        """Validate the model."""
        self.model.eval()
        val_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                
                # Calculate loss
                if output.shape[1] == 2:  # Binary classification
                    target_one_hot = torch.zeros_like(output)
                    target_one_hot.scatter_(1, target.unsqueeze(1), 1)
                    loss = self.criterion(output, target_one_hot.float())
                else:
                    loss = self.criterion(output, target)
                
                val_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total_samples += target.size(0)
                correct_predictions += (predicted == target).sum().item()
        
        val_loss /= len(self.val_loader)
        val_acc = 100 * correct_predictions / total_samples
        
        return val_loss, val_acc
    
    def train(self, num_epochs: int = 50, save_every: int = 10):
        """Main training loop."""
        if not self.model or not self.train_loader:
            raise ValueError("Model and data must be setup before training")
        
        # Setup tensorboard logging
        self.writer = SummaryWriter(log_dir=f'runs/chest_xray_training')
        
        best_val_acc = 0.0
        train_losses, val_losses = [], []
        train_accs, val_accs = [], []
        
        print("Starting training...")
        
        for epoch in range(num_epochs):
            # Training
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Validation
            val_loss, val_acc = self.validate()
            
            # Store metrics
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)
            
            # Log to tensorboard
            if self.writer:
                self.writer.add_scalar('Loss/Train', train_loss, epoch)
                self.writer.add_scalar('Loss/Validation', val_loss, epoch)
                self.writer.add_scalar('Accuracy/Train', train_acc, epoch)
                self.writer.add_scalar('Accuracy/Validation', val_acc, epoch)
            
            print(f'Epoch {epoch+1}/{num_epochs}:')
            print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_checkpoint(epoch, val_acc, is_best=True)
                print(f'  New best model saved! Val Acc: {val_acc:.2f}%')
            
            # Save checkpoint periodically
            if (epoch + 1) % save_every == 0:
                self.save_checkpoint(epoch, val_acc, is_best=False)
        
        # Save final model
        self.save_checkpoint(num_epochs-1, val_acc, is_best=False, filename='final_model.pth')
        
        # Close tensorboard writer
        if self.writer:
            self.writer.close()
        
        # Plot training curves
        self.plot_training_curves(train_losses, val_losses, train_accs, val_accs)
        
        print(f"Training completed! Best validation accuracy: {best_val_acc:.2f}%")
    
    def save_checkpoint(self, epoch: int, val_acc: float, is_best: bool = False, filename: str = None):
        """Save model checkpoint."""
        if filename is None:
            filename = 'best_model.pth' if is_best else f'checkpoint_epoch_{epoch+1}.pth'
        
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_acc': val_acc,
        }
        
        filepath = self.output_dir / filename
        torch.save(checkpoint, filepath)
    
    def plot_training_curves(self, train_losses, val_losses, train_accs, val_accs):
        """Plot and save training curves."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss curves
        ax1.plot(train_losses, label='Training Loss')
        ax1.plot(val_losses, label='Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy curves
        ax2.plot(train_accs, label='Training Accuracy')
        ax2.plot(val_accs, label='Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'training_curves.png', dpi=150)
        plt.close()


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train chest X-ray classification model')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing training data')
    parser.add_argument('--csv_file', type=str, help='CSV file with labels')
    parser.add_argument('--output_dir', type=str, default='models/checkpoints', help='Output directory for models')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--val_split', type=float, default=0.2, help='Validation split ratio')
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = ChestXrayTrainer(args.data_dir, args.csv_file, args.output_dir)
    
    # Setup data and model
    trainer.setup_data(batch_size=args.batch_size, val_split=args.val_split)
    trainer.setup_model(num_classes=2, learning_rate=args.lr)
    
    # Train model
    trainer.train(num_epochs=args.epochs)


if __name__ == '__main__':
    main()