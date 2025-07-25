"""
Data utilities for loading, validating, and managing datasets.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pydicom
from PIL import Image
import logging

logger = logging.getLogger(__name__)


class DataValidator:
    """Validate data integrity and format."""
    
    @staticmethod
    def validate_image_file(file_path: Union[str, Path]) -> bool:
        """
        Validate if file is a valid image.
        
        Args:
            file_path: Path to image file
            
        Returns:
            True if valid image, False otherwise
        """
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                return False
            
            # Check file extension
            valid_extensions = {'.png', '.jpg', '.jpeg', '.dcm', '.dicom'}
            if file_path.suffix.lower() not in valid_extensions:
                return False
            
            # Try to load the image
            if file_path.suffix.lower() in ['.dcm', '.dicom']:
                # Validate DICOM file
                try:
                    ds = pydicom.dcmread(file_path)
                    if not hasattr(ds, 'pixel_array'):
                        return False
                    _ = ds.pixel_array  # Try to access pixel data
                except Exception:
                    return False
            else:
                # Validate standard image formats
                try:
                    with Image.open(file_path) as img:
                        img.verify()  # Verify image integrity
                except Exception:
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating image file {file_path}: {e}")
            return False
    
    @staticmethod
    def validate_csv_file(csv_path: Union[str, Path], 
                         required_columns: List[str] = None) -> Tuple[bool, str]:
        """
        Validate CSV file format and required columns.
        
        Args:
            csv_path: Path to CSV file
            required_columns: List of required column names
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            csv_path = Path(csv_path)
            
            if not csv_path.exists():
                return False, f"CSV file does not exist: {csv_path}"
            
            # Try to read CSV
            try:
                df = pd.read_csv(csv_path)
            except Exception as e:
                return False, f"Cannot read CSV file: {e}"
            
            if df.empty:
                return False, "CSV file is empty"
            
            # Check required columns
            if required_columns:
                missing_columns = set(required_columns) - set(df.columns)
                if missing_columns:
                    return False, f"Missing required columns: {missing_columns}"
            
            return True, "Valid CSV file"
            
        except Exception as e:
            return False, f"Error validating CSV file: {e}"
    
    @staticmethod
    def validate_dataset_structure(data_dir: Union[str, Path]) -> Dict[str, any]:
        """
        Validate dataset directory structure.
        
        Args:
            data_dir: Path to dataset directory
            
        Returns:
            Dictionary with validation results
        """
        data_dir = Path(data_dir)
        result = {
            'valid': True,
            'total_images': 0,
            'valid_images': 0,
            'invalid_images': [],
            'subdirectories': [],
            'errors': []
        }
        
        if not data_dir.exists():
            result['valid'] = False
            result['errors'].append(f"Data directory does not exist: {data_dir}")
            return result
        
        # Check for subdirectories (class folders)
        subdirs = [d for d in data_dir.iterdir() if d.is_dir()]
        result['subdirectories'] = [d.name for d in subdirs]
        
        # Validate images
        image_extensions = {'.png', '.jpg', '.jpeg', '.dcm', '.dicom'}
        
        for file_path in data_dir.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                result['total_images'] += 1
                
                if DataValidator.validate_image_file(file_path):
                    result['valid_images'] += 1
                else:
                    result['invalid_images'].append(str(file_path))
        
        if result['total_images'] == 0:
            result['valid'] = False
            result['errors'].append("No image files found in dataset")
        
        if result['invalid_images']:
            result['errors'].append(f"Found {len(result['invalid_images'])} invalid images")
        
        return result


class DatasetSplitter:
    """Split datasets into train/validation/test sets."""
    
    @staticmethod
    def split_dataset(image_paths: List[Path], 
                     labels: List[int],
                     train_ratio: float = 0.7,
                     val_ratio: float = 0.2,
                     test_ratio: float = 0.1,
                     random_state: int = 42) -> Dict[str, Tuple[List[Path], List[int]]]:
        """
        Split dataset into train/validation/test sets.
        
        Args:
            image_paths: List of image file paths
            labels: List of corresponding labels
            train_ratio: Ratio for training set
            val_ratio: Ratio for validation set
            test_ratio: Ratio for test set
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary with train/val/test splits
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Ratios must sum to 1.0"
        
        # First split: separate test set
        if test_ratio > 0:
            X_temp, X_test, y_temp, y_test = train_test_split(
                image_paths, labels, 
                test_size=test_ratio, 
                random_state=random_state,
                stratify=labels
            )
        else:
            X_temp, y_temp = image_paths, labels
            X_test, y_test = [], []
        
        # Second split: separate train and validation
        if val_ratio > 0:
            val_size = val_ratio / (train_ratio + val_ratio)
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp,
                test_size=val_size,
                random_state=random_state,
                stratify=y_temp
            )
        else:
            X_train, y_train = X_temp, y_temp
            X_val, y_val = [], []
        
        return {
            'train': (X_train, y_train),
            'val': (X_val, y_val),
            'test': (X_test, y_test)
        }
    
    @staticmethod
    def save_split_info(splits: Dict[str, Tuple[List[Path], List[int]]], 
                       output_dir: Union[str, Path]):
        """
        Save dataset split information to CSV files.
        
        Args:
            splits: Dictionary with dataset splits
            output_dir: Directory to save split files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for split_name, (paths, labels) in splits.items():
            if paths:  # Only save non-empty splits
                df = pd.DataFrame({
                    'image_path': [str(p) for p in paths],
                    'label': labels
                })
                
                output_file = output_dir / f"{split_name}_split.csv"
                df.to_csv(output_file, index=False)
                logger.info(f"Saved {split_name} split with {len(df)} samples to {output_file}")


class DatasetLoader:
    """Load and manage datasets."""
    
    @staticmethod
    def load_from_directory(data_dir: Union[str, Path],
                          class_names: Optional[List[str]] = None) -> Tuple[List[Path], List[int], List[str]]:
        """
        Load dataset from directory structure (one subdirectory per class).
        
        Args:
            data_dir: Path to dataset directory
            class_names: Optional list of class names (for ordering)
            
        Returns:
            Tuple of (image_paths, labels, class_names)
        """
        data_dir = Path(data_dir)
        image_paths = []
        labels = []
        
        # Get class directories
        class_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
        
        if not class_dirs:
            # No subdirectories, assume all images are in root directory
            image_extensions = {'.png', '.jpg', '.jpeg', '.dcm', '.dicom'}
            for file_path in data_dir.iterdir():
                if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                    image_paths.append(file_path)
                    labels.append(0)  # Default class
            
            return image_paths, labels, ['unknown']
        
        # Sort class directories
        if class_names:
            # Use provided order
            class_dirs = [data_dir / name for name in class_names if (data_dir / name).exists()]
        else:
            class_dirs = sorted(class_dirs, key=lambda x: x.name)
            class_names = [d.name for d in class_dirs]
        
        # Load images from each class directory
        image_extensions = {'.png', '.jpg', '.jpeg', '.dcm', '.dicom'}
        
        for class_idx, class_dir in enumerate(class_dirs):
            for file_path in class_dir.iterdir():
                if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                    if DataValidator.validate_image_file(file_path):
                        image_paths.append(file_path)
                        labels.append(class_idx)
                    else:
                        logger.warning(f"Skipping invalid image: {file_path}")
        
        logger.info(f"Loaded {len(image_paths)} images from {len(class_names)} classes")
        return image_paths, labels, class_names
    
    @staticmethod
    def load_from_csv(csv_path: Union[str, Path],
                     data_dir: Union[str, Path],
                     image_column: str = 'image_path',
                     label_column: str = 'label') -> Tuple[List[Path], List[int]]:
        """
        Load dataset from CSV file.
        
        Args:
            csv_path: Path to CSV file
            data_dir: Base directory for image paths
            image_column: Name of column containing image paths
            label_column: Name of column containing labels
            
        Returns:
            Tuple of (image_paths, labels)
        """
        csv_path = Path(csv_path)
        data_dir = Path(data_dir)
        
        df = pd.read_csv(csv_path)
        
        image_paths = []
        labels = []
        
        for _, row in df.iterrows():
            image_path = data_dir / row[image_column]
            
            if DataValidator.validate_image_file(image_path):
                image_paths.append(image_path)
                labels.append(int(row[label_column]))
            else:
                logger.warning(f"Skipping invalid image: {image_path}")
        
        logger.info(f"Loaded {len(image_paths)} images from CSV")
        return image_paths, labels


class DatasetStatistics:
    """Calculate and display dataset statistics."""
    
    @staticmethod
    def calculate_statistics(labels: List[int], 
                           class_names: Optional[List[str]] = None) -> Dict[str, any]:
        """
        Calculate dataset statistics.
        
        Args:
            labels: List of labels
            class_names: Optional list of class names
            
        Returns:
            Dictionary with statistics
        """
        labels_array = np.array(labels)
        unique_labels, counts = np.unique(labels_array, return_counts=True)
        
        stats = {
            'total_samples': len(labels),
            'num_classes': len(unique_labels),
            'class_distribution': {},
            'class_balance': {}
        }
        
        for label, count in zip(unique_labels, counts):
            class_name = class_names[label] if class_names else f"class_{label}"
            stats['class_distribution'][class_name] = int(count)
            stats['class_balance'][class_name] = float(count / len(labels))
        
        return stats
    
    @staticmethod
    def print_statistics(stats: Dict[str, any]):
        """Print dataset statistics in a formatted way."""
        print(f"\nDataset Statistics:")
        print(f"Total samples: {stats['total_samples']}")
        print(f"Number of classes: {stats['num_classes']}")
        print(f"\nClass distribution:")
        
        for class_name, count in stats['class_distribution'].items():
            percentage = stats['class_balance'][class_name] * 100
            print(f"  {class_name}: {count} ({percentage:.1f}%)")


def create_data_loaders(dataset: Dataset,
                       batch_size: int = 32,
                       val_split: float = 0.2,
                       num_workers: int = 2,
                       pin_memory: bool = True) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation data loaders.
    
    Args:
        dataset: PyTorch dataset
        batch_size: Batch size for data loaders
        val_split: Fraction of data to use for validation
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    dataset_size = len(dataset)
    val_size = int(val_split * dataset_size)
    train_size = dataset_size - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader