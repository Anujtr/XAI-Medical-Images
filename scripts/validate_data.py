#!/usr/bin/env python3
"""
Validate dataset integrity and structure for XAI Medical Images.
"""

import os
import argparse
from pathlib import Path
import pandas as pd
import sys
import logging
from typing import Dict, List, Tuple, Optional

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_utils import DataValidator, DatasetStatistics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetValidator:
    """Comprehensive dataset validation."""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.validator = DataValidator()
        self.stats = DatasetStatistics()
        self.validation_results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'statistics': {}
        }
    
    def validate_directory_structure(self) -> bool:
        """Validate basic directory structure."""
        logger.info("Validating directory structure...")
        
        if not self.data_dir.exists():
            self.validation_results['errors'].append(
                f"Data directory does not exist: {self.data_dir}"
            )
            return False
        
        # Check for expected subdirectories or images
        subdirs = [d for d in self.data_dir.iterdir() if d.is_dir()]
        image_files = list(self.data_dir.glob('*.png')) + \
                     list(self.data_dir.glob('*.jpg')) + \
                     list(self.data_dir.glob('*.jpeg')) + \
                     list(self.data_dir.glob('*.dcm'))
        
        if not subdirs and not image_files:
            self.validation_results['errors'].append(
                "No subdirectories or image files found in data directory"
            )
            return False
        
        if subdirs:
            logger.info(f"Found {len(subdirs)} subdirectories: {[d.name for d in subdirs]}")
        if image_files:
            logger.info(f"Found {len(image_files)} image files in root directory")
        
        return True
    
    def validate_images(self) -> Tuple[int, int, List[str]]:
        """Validate all images in the dataset."""
        logger.info("Validating image files...")
        
        valid_count = 0
        total_count = 0
        invalid_files = []
        
        # Get all image files
        image_extensions = ['.png', '.jpg', '.jpeg', '.dcm', '.dicom']
        
        for ext in image_extensions:
            for image_file in self.data_dir.rglob(f'*{ext}'):
                total_count += 1
                
                if self.validator.validate_image_file(image_file):
                    valid_count += 1
                else:
                    invalid_files.append(str(image_file))
                    logger.warning(f"Invalid image: {image_file}")
        
        logger.info(f"Image validation: {valid_count}/{total_count} valid")
        
        if invalid_files:
            self.validation_results['warnings'].extend([
                f"Invalid image file: {f}" for f in invalid_files
            ])
        
        return valid_count, total_count, invalid_files
    
    def validate_csv_labels(self, csv_path: Optional[str] = None) -> bool:
        """Validate CSV label file if present."""
        if csv_path:
            csv_file = Path(csv_path)
        else:
            # Look for common CSV files
            csv_candidates = list(self.data_dir.glob('*.csv'))
            if not csv_candidates:
                logger.info("No CSV label file found")
                return True
            csv_file = csv_candidates[0]
        
        logger.info(f"Validating CSV file: {csv_file}")
        
        required_columns = ['image_path', 'label']  # Basic requirements
        is_valid, error_msg = self.validator.validate_csv_file(csv_file, required_columns)
        
        if not is_valid:
            self.validation_results['errors'].append(f"CSV validation failed: {error_msg}")
            return False
        
        # Additional CSV validation
        try:
            df = pd.read_csv(csv_file)
            
            # Check for missing values
            missing_values = df.isnull().sum()
            if missing_values.any():
                self.validation_results['warnings'].append(
                    f"CSV has missing values: {missing_values.to_dict()}"
                )
            
            # Check label distribution
            if 'label' in df.columns:
                label_counts = df['label'].value_counts()
                logger.info(f"Label distribution: {label_counts.to_dict()}")
                
                # Check for class imbalance
                min_count = label_counts.min()
                max_count = label_counts.max()
                if max_count / min_count > 10:  # More than 10:1 ratio
                    self.validation_results['warnings'].append(
                        f"Severe class imbalance detected: {label_counts.to_dict()}"
                    )
            
            # Validate image paths in CSV
            if 'image_path' in df.columns:
                missing_images = []
                for _, row in df.iterrows():
                    img_path = self.data_dir / row['image_path']
                    if not img_path.exists():
                        missing_images.append(str(img_path))
                
                if missing_images:
                    self.validation_results['errors'].extend([
                        f"Image referenced in CSV not found: {img}" for img in missing_images[:10]
                    ])
                    if len(missing_images) > 10:
                        self.validation_results['errors'].append(
                            f"... and {len(missing_images) - 10} more missing images"
                        )
        
        except Exception as e:
            self.validation_results['errors'].append(f"Error processing CSV: {e}")
            return False
        
        return True
    
    def calculate_dataset_statistics(self) -> Dict:
        """Calculate comprehensive dataset statistics."""
        logger.info("Calculating dataset statistics...")
        
        # Get all images and labels
        image_paths = []
        labels = []
        
        # Check if we have subdirectory structure (class folders)
        subdirs = [d for d in self.data_dir.iterdir() if d.is_dir()]
        
        if subdirs:
            # Subdirectory structure
            for class_idx, class_dir in enumerate(sorted(subdirs)):
                class_images = []
                for ext in ['.png', '.jpg', '.jpeg', '.dcm']:
                    class_images.extend(list(class_dir.glob(f'*{ext}')))
                
                for img_path in class_images:
                    if self.validator.validate_image_file(img_path):
                        image_paths.append(img_path)
                        labels.append(class_idx)
            
            class_names = [d.name for d in sorted(subdirs)]
        else:
            # Single directory - assume all images are class 0
            for ext in ['.png', '.jpg', '.jpeg', '.dcm']:
                for img_path in self.data_dir.glob(f'*{ext}'):
                    if self.validator.validate_image_file(img_path):
                        image_paths.append(img_path)
                        labels.append(0)
            
            class_names = ['unknown']
        
        if labels:
            stats = self.stats.calculate_statistics(labels, class_names)
            self.validation_results['statistics'] = stats
            return stats
        else:
            return {}
    
    def check_disk_space(self) -> None:
        """Check available disk space."""
        import shutil
        
        total, used, free = shutil.disk_usage(self.data_dir)
        
        # Convert to GB
        total_gb = total / (1024**3)
        used_gb = used / (1024**3)
        free_gb = free / (1024**3)
        
        logger.info(f"Disk space - Total: {total_gb:.1f}GB, Used: {used_gb:.1f}GB, Free: {free_gb:.1f}GB")
        
        # Warning if less than 5GB free
        if free_gb < 5:
            self.validation_results['warnings'].append(
                f"Low disk space: only {free_gb:.1f}GB remaining"
            )
    
    def check_permissions(self) -> None:
        """Check file permissions."""
        if not os.access(self.data_dir, os.R_OK):
            self.validation_results['errors'].append(
                f"No read permission for directory: {self.data_dir}"
            )
        
        # Check write permission for processing
        if not os.access(self.data_dir, os.W_OK):
            self.validation_results['warnings'].append(
                f"No write permission for directory: {self.data_dir}"
            )
    
    def run_full_validation(self, csv_path: Optional[str] = None) -> Dict:
        """Run complete dataset validation."""
        logger.info(f"Starting full validation of dataset: {self.data_dir}")
        
        # Basic structure validation
        if not self.validate_directory_structure():
            self.validation_results['valid'] = False
            return self.validation_results
        
        # Image validation
        valid_images, total_images, invalid_files = self.validate_images()
        
        if valid_images == 0:
            self.validation_results['errors'].append("No valid images found in dataset")
            self.validation_results['valid'] = False
        elif invalid_files:
            self.validation_results['valid'] = False
        
        # CSV validation
        if not self.validate_csv_labels(csv_path):
            self.validation_results['valid'] = False
        
        # Statistics
        stats = self.calculate_dataset_statistics()
        
        # System checks
        self.check_disk_space()
        self.check_permissions()
        
        # Final validation status
        if self.validation_results['errors']:
            self.validation_results['valid'] = False
        
        return self.validation_results
    
    def print_validation_report(self) -> None:
        """Print formatted validation report."""
        print("\n" + "="*60)
        print("DATASET VALIDATION REPORT")
        print("="*60)
        
        # Overall status
        status = "✅ VALID" if self.validation_results['valid'] else "❌ INVALID"
        print(f"Overall Status: {status}")
        print(f"Dataset Path: {self.data_dir}")
        
        # Statistics
        if self.validation_results['statistics']:
            print(f"\nDataset Statistics:")
            stats = self.validation_results['statistics']
            print(f"  Total samples: {stats['total_samples']}")
            print(f"  Number of classes: {stats['num_classes']}")
            
            if 'class_distribution' in stats:
                print(f"  Class distribution:")
                for class_name, count in stats['class_distribution'].items():
                    percentage = stats['class_balance'][class_name] * 100
                    print(f"    {class_name}: {count} ({percentage:.1f}%)")
        
        # Errors
        if self.validation_results['errors']:
            print(f"\n❌ Errors ({len(self.validation_results['errors'])}):")
            for error in self.validation_results['errors']:
                print(f"  • {error}")
        
        # Warnings
        if self.validation_results['warnings']:
            print(f"\n⚠️  Warnings ({len(self.validation_results['warnings'])}):")
            for warning in self.validation_results['warnings']:
                print(f"  • {warning}")
        
        if not self.validation_results['errors'] and not self.validation_results['warnings']:
            print(f"\n✅ No issues found!")
        
        print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Validate dataset for XAI Medical Images')
    parser.add_argument(
        '--data_dir',
        type=str,
        required=True,
        help='Path to dataset directory'
    )
    parser.add_argument(
        '--csv_file',
        type=str,
        help='Path to CSV label file (optional)'
    )
    parser.add_argument(
        '--output_report',
        type=str,
        help='Save validation report to file'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        validator = DatasetValidator(args.data_dir)
        results = validator.run_full_validation(args.csv_file)
        
        # Print report
        validator.print_validation_report()
        
        # Save report if requested
        if args.output_report:
            import json
            with open(args.output_report, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"Validation report saved to: {args.output_report}")
        
        # Exit with appropriate code
        sys.exit(0 if results['valid'] else 1)
        
    except KeyboardInterrupt:
        logger.info("Validation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()