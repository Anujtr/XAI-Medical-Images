#!/usr/bin/env python3
"""
Download sample chest X-ray data for testing and development.
"""

import os
import requests
import argparse
from pathlib import Path
import zipfile
import tempfile
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SampleDataDownloader:
    """Download and setup sample data for development."""
    
    SAMPLE_DATASETS = {
        'mini_chestxray': {
            'url': 'https://github.com/ieee8023/covid-chestxray-dataset/archive/master.zip',
            'description': 'COVID-19 chest X-ray dataset sample',
            'size': '~50MB'
        },
        'pediatric_pneumonia': {
            'url': 'https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia',
            'description': 'Pediatric pneumonia dataset (requires Kaggle API)',
            'size': '~1.2GB'
        }
    }
    
    def __init__(self, output_dir: str = 'data/sample'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def download_file(self, url: str, filename: str) -> Path:
        """Download a file from URL."""
        filepath = self.output_dir / filename
        
        if filepath.exists():
            logger.info(f"File already exists: {filepath}")
            return filepath
        
        logger.info(f"Downloading {filename} from {url}")
        
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\rProgress: {percent:.1f}%", end='', flush=True)
        
        print()  # New line after progress
        logger.info(f"Downloaded: {filepath}")
        return filepath
    
    def extract_zip(self, zip_path: Path, extract_to: Path) -> None:
        """Extract ZIP file."""
        logger.info(f"Extracting {zip_path} to {extract_to}")
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        
        logger.info(f"Extraction complete")
    
    def create_sample_images(self, num_samples: int = 20) -> None:
        """Create synthetic sample images for testing."""
        import numpy as np
        from PIL import Image
        
        logger.info(f"Creating {num_samples} synthetic sample images")
        
        # Create directories
        normal_dir = self.output_dir / 'normal'
        abnormal_dir = self.output_dir / 'abnormal'
        normal_dir.mkdir(exist_ok=True)
        abnormal_dir.mkdir(exist_ok=True)
        
        for i in range(num_samples // 2):
            # Create normal-looking image (more uniform)
            normal_image = np.random.randint(100, 200, (512, 512), dtype=np.uint8)
            # Add some structure
            normal_image[100:400, 100:400] += np.random.randint(-30, 30, (300, 300))
            normal_image = np.clip(normal_image, 0, 255)
            
            Image.fromarray(normal_image, mode='L').save(
                normal_dir / f'normal_{i:03d}.png'
            )
            
            # Create abnormal-looking image (more varied)
            abnormal_image = np.random.randint(50, 250, (512, 512), dtype=np.uint8)
            # Add bright spots (simulating pathology)
            spots = np.random.randint(0, 2, (512, 512)) * 100
            abnormal_image = np.clip(abnormal_image + spots, 0, 255)
            
            Image.fromarray(abnormal_image, mode='L').save(
                abnormal_dir / f'abnormal_{i:03d}.png'
            )
        
        logger.info(f"Created synthetic images in {self.output_dir}")
    
    def download_covid_dataset(self) -> None:
        """Download COVID-19 chest X-ray dataset sample."""
        dataset_info = self.SAMPLE_DATASETS['mini_chestxray']
        
        # Download the dataset
        zip_file = self.download_file(
            dataset_info['url'], 
            'covid-chestxray-dataset.zip'
        )
        
        # Extract
        extract_dir = self.output_dir / 'covid_dataset'
        self.extract_zip(zip_file, extract_dir)
        
        # Clean up zip file
        zip_file.unlink()
        
        logger.info(f"COVID-19 dataset ready in {extract_dir}")
    
    def setup_test_data(self) -> None:
        """Setup minimal test data structure."""
        logger.info("Setting up test data structure")
        
        # Create directory structure
        dirs = [
            'raw/normal',
            'raw/abnormal', 
            'processed/normal',
            'processed/abnormal',
            'sample/normal',
            'sample/abnormal'
        ]
        
        for dir_path in dirs:
            (self.output_dir.parent / dir_path).mkdir(parents=True, exist_ok=True)
        
        # Create sample CSV file
        csv_content = """image_path,label,finding
normal/normal_001.png,0,No Finding
normal/normal_002.png,0,No Finding
abnormal/abnormal_001.png,1,Pneumonia
abnormal/abnormal_002.png,1,Atelectasis
"""
        
        csv_path = self.output_dir / 'sample_labels.csv'
        with open(csv_path, 'w') as f:
            f.write(csv_content)
        
        logger.info(f"Created sample CSV: {csv_path}")
    
    def list_datasets(self) -> None:
        """List available datasets."""
        print("\nAvailable Sample Datasets:")
        print("-" * 50)
        
        for name, info in self.SAMPLE_DATASETS.items():
            print(f"Name: {name}")
            print(f"Description: {info['description']}")
            print(f"Size: {info['size']}")
            print(f"URL: {info['url']}")
            print("-" * 30)


def main():
    parser = argparse.ArgumentParser(description='Download sample data for XAI Medical Images')
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default='data/sample',
        help='Output directory for sample data'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['covid', 'synthetic', 'test_structure'],
        default='synthetic',
        help='Dataset to download'
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=20,
        help='Number of synthetic samples to create'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List available datasets'
    )
    
    args = parser.parse_args()
    
    downloader = SampleDataDownloader(args.output_dir)
    
    if args.list:
        downloader.list_datasets()
        return
    
    try:
        if args.dataset == 'covid':
            downloader.download_covid_dataset()
        elif args.dataset == 'synthetic':
            downloader.create_sample_images(args.num_samples)
        elif args.dataset == 'test_structure':
            downloader.setup_test_data()
            downloader.create_sample_images(args.num_samples)
        
        logger.info("Sample data setup complete!")
        
    except KeyboardInterrupt:
        logger.info("Download interrupted by user")
    except Exception as e:
        logger.error(f"Error downloading sample data: {e}")
        raise


if __name__ == '__main__':
    main()