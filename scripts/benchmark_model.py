#!/usr/bin/env python3
"""
Benchmark model performance and inference speed for XAI Medical Images.
"""

import os
import sys
import time
import argparse
import torch
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Tuple
import json
import psutil
import gc

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from routes import XRayAnalyzer
from preprocess import create_preprocessor
from gradcam import create_gradcam
import torchvision.models as models

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelBenchmark:
    """Benchmark model performance and resource usage."""
    
    def __init__(self, model_path: str = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        self.results = {
            'device': str(self.device),
            'model_path': model_path,
            'benchmarks': {}
        }
        
        # Initialize components
        self.analyzer = None
        self.model = None
        self.preprocessor = create_preprocessor()
        
        logger.info(f"Benchmarking on device: {self.device}")
    
    def setup_model(self) -> None:
        """Setup model for benchmarking."""
        logger.info("Setting up model...")
        
        # Load model
        self.model = models.resnet50(pretrained=True)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 2)
        
        if self.model_path and os.path.exists(self.model_path):
            checkpoint = torch.load(self.model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            logger.info(f"Loaded model from {self.model_path}")
        
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize analyzer
        self.analyzer = XRayAnalyzer(self.model_path)
    
    def get_system_info(self) -> Dict:
        """Get system information."""
        info = {
            'cpu_count': psutil.cpu_count(),
            'cpu_count_logical': psutil.cpu_count(logical=True),
            'memory_total_gb': psutil.virtual_memory().total / (1024**3),
            'memory_available_gb': psutil.virtual_memory().available / (1024**3),
            'python_version': sys.version,
            'pytorch_version': torch.__version__,
        }
        
        if torch.cuda.is_available():
            info.update({
                'cuda_available': True,
                'cuda_version': torch.version.cuda,
                'gpu_count': torch.cuda.device_count(),
                'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else None,
                'gpu_memory_gb': torch.cuda.get_device_properties(0).total_memory / (1024**3) if torch.cuda.device_count() > 0 else None
            })
        else:
            info['cuda_available'] = False
        
        return info
    
    def benchmark_model_loading(self) -> Dict:
        """Benchmark model loading time."""
        logger.info("Benchmarking model loading...")
        
        times = []
        memory_usage = []
        
        for i in range(5):  # Run 5 times
            # Clear GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            start_memory = psutil.Process().memory_info().rss / (1024**2)  # MB
            
            start_time = time.time()
            
            # Load model
            model = models.resnet50(pretrained=True)
            model.fc = torch.nn.Linear(model.fc.in_features, 2)
            
            if self.model_path and os.path.exists(self.model_path):
                checkpoint = torch.load(self.model_path, map_location=self.device)
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
            
            model.to(self.device)
            model.eval()
            
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / (1024**2)  # MB
            
            times.append(end_time - start_time)
            memory_usage.append(end_memory - start_memory)
            
            del model
        
        return {
            'avg_time_seconds': float(np.mean(times)),
            'std_time_seconds': float(np.std(times)),
            'min_time_seconds': float(np.min(times)),
            'max_time_seconds': float(np.max(times)),
            'avg_memory_mb': float(np.mean(memory_usage)),
            'std_memory_mb': float(np.std(memory_usage))
        }
    
    def benchmark_inference_speed(self, num_samples: int = 100) -> Dict:
        """Benchmark inference speed."""
        logger.info(f"Benchmarking inference speed with {num_samples} samples...")
        
        if not self.model:
            self.setup_model()
        
        # Create random test data
        test_data = torch.randn(1, 3, 224, 224).to(self.device)
        
        # Warmup
        logger.info("Warming up model...")
        for _ in range(10):
            with torch.no_grad():
                _ = self.model(test_data)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # Benchmark
        times = []
        
        for i in range(num_samples):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            start_time = time.time()
            
            with torch.no_grad():
                output = self.model(test_data)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            end_time = time.time()
            times.append(end_time - start_time)
            
            if (i + 1) % 20 == 0:
                logger.info(f"Completed {i + 1}/{num_samples} inference runs")
        
        return {
            'num_samples': num_samples,
            'avg_time_seconds': float(np.mean(times)),
            'std_time_seconds': float(np.std(times)),
            'min_time_seconds': float(np.min(times)),
            'max_time_seconds': float(np.max(times)),
            'throughput_samples_per_second': float(1.0 / np.mean(times)),
            'percentile_95_seconds': float(np.percentile(times, 95)),
            'percentile_99_seconds': float(np.percentile(times, 99))
        }
    
    def benchmark_gradcam_generation(self, num_samples: int = 50) -> Dict:
        """Benchmark Grad-CAM generation speed."""
        logger.info(f"Benchmarking Grad-CAM generation with {num_samples} samples...")
        
        if not self.model:
            self.setup_model()
        
        gradcam = create_gradcam(self.model)
        test_data = torch.randn(1, 3, 224, 224).to(self.device)
        test_image = np.random.randint(0, 255, (224, 224), dtype=np.uint8)
        
        # Warmup
        logger.info("Warming up Grad-CAM...")
        for _ in range(5):
            _ = gradcam.generate_cam(test_data)
        
        # Benchmark CAM generation
        cam_times = []
        overlay_times = []
        total_times = []
        
        for i in range(num_samples):
            # CAM generation
            start_time = time.time()
            heatmap = gradcam.generate_cam(test_data)
            cam_time = time.time() - start_time
            cam_times.append(cam_time)
            
            # Overlay generation
            start_time = time.time()
            _ = gradcam.overlay_heatmap(test_image, heatmap)
            overlay_time = time.time() - start_time
            overlay_times.append(overlay_time)
            
            total_times.append(cam_time + overlay_time)
            
            if (i + 1) % 10 == 0:
                logger.info(f"Completed {i + 1}/{num_samples} Grad-CAM runs")
        
        return {
            'num_samples': num_samples,
            'cam_generation': {
                'avg_time_seconds': float(np.mean(cam_times)),
                'std_time_seconds': float(np.std(cam_times)),
                'min_time_seconds': float(np.min(cam_times)),
                'max_time_seconds': float(np.max(cam_times))
            },
            'overlay_generation': {
                'avg_time_seconds': float(np.mean(overlay_times)),
                'std_time_seconds': float(np.std(overlay_times)),
                'min_time_seconds': float(np.min(overlay_times)),
                'max_time_seconds': float(np.max(overlay_times))
            },
            'total_gradcam': {
                'avg_time_seconds': float(np.mean(total_times)),
                'std_time_seconds': float(np.std(total_times)),
                'throughput_samples_per_second': float(1.0 / np.mean(total_times))
            }
        }
    
    def benchmark_preprocessing(self, num_samples: int = 100) -> Dict:
        """Benchmark preprocessing speed."""
        logger.info(f"Benchmarking preprocessing with {num_samples} samples...")
        
        # Create test images
        test_images = [
            np.random.randint(0, 255, (512, 512), dtype=np.uint8)
            for _ in range(num_samples)
        ]
        
        # Benchmark preprocessing
        times = []
        
        for i, test_image in enumerate(test_images):
            start_time = time.time()
            
            # Preprocess for model
            model_input = self.preprocessor.preprocess_for_model(test_image)
            display_image = self.preprocessor.preprocess_for_display(test_image)
            
            end_time = time.time()
            times.append(end_time - start_time)
            
            if (i + 1) % 20 == 0:
                logger.info(f"Completed {i + 1}/{num_samples} preprocessing runs")
        
        return {
            'num_samples': num_samples,
            'avg_time_seconds': float(np.mean(times)),
            'std_time_seconds': float(np.std(times)),
            'min_time_seconds': float(np.min(times)),
            'max_time_seconds': float(np.max(times)),
            'throughput_samples_per_second': float(1.0 / np.mean(times))
        }
    
    def benchmark_memory_usage(self) -> Dict:
        """Benchmark memory usage."""
        logger.info("Benchmarking memory usage...")
        
        if not self.model:
            self.setup_model()
        
        # Get baseline memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        baseline_memory = psutil.Process().memory_info().rss / (1024**2)  # MB
        baseline_gpu_memory = torch.cuda.memory_allocated() / (1024**2) if torch.cuda.is_available() else 0
        
        # Memory with model loaded
        model_memory = psutil.Process().memory_info().rss / (1024**2)  # MB
        model_gpu_memory = torch.cuda.memory_allocated() / (1024**2) if torch.cuda.is_available() else 0
        
        # Memory during inference
        test_data = torch.randn(1, 3, 224, 224).to(self.device)
        
        with torch.no_grad():
            _ = self.model(test_data)
        
        inference_memory = psutil.Process().memory_info().rss / (1024**2)  # MB
        inference_gpu_memory = torch.cuda.memory_allocated() / (1024**2) if torch.cuda.is_available() else 0
        
        # Peak memory during Grad-CAM
        gradcam = create_gradcam(self.model)
        _ = gradcam.generate_cam(test_data)
        
        gradcam_memory = psutil.Process().memory_info().rss / (1024**2)  # MB
        gradcam_gpu_memory = torch.cuda.memory_allocated() / (1024**2) if torch.cuda.is_available() else 0
        
        return {
            'baseline_memory_mb': float(baseline_memory),
            'model_memory_mb': float(model_memory),
            'inference_memory_mb': float(inference_memory),
            'gradcam_memory_mb': float(gradcam_memory),
            'model_overhead_mb': float(model_memory - baseline_memory),
            'inference_overhead_mb': float(inference_memory - model_memory),
            'gradcam_overhead_mb': float(gradcam_memory - inference_memory),
            'gpu': {
                'baseline_memory_mb': float(baseline_gpu_memory),
                'model_memory_mb': float(model_gpu_memory),
                'inference_memory_mb': float(inference_gpu_memory),
                'gradcam_memory_mb': float(gradcam_gpu_memory)
            } if torch.cuda.is_available() else None
        }
    
    def run_full_benchmark(self) -> Dict:
        """Run complete benchmark suite."""
        logger.info("Starting full benchmark suite...")
        
        # System info
        self.results['system_info'] = self.get_system_info()
        
        # Model loading benchmark
        self.results['benchmarks']['model_loading'] = self.benchmark_model_loading()
        
        # Setup model for other benchmarks
        self.setup_model()
        
        # Inference speed
        self.results['benchmarks']['inference_speed'] = self.benchmark_inference_speed()
        
        # Grad-CAM generation
        self.results['benchmarks']['gradcam_generation'] = self.benchmark_gradcam_generation()
        
        # Preprocessing
        self.results['benchmarks']['preprocessing'] = self.benchmark_preprocessing()
        
        # Memory usage
        self.results['benchmarks']['memory_usage'] = self.benchmark_memory_usage()
        
        logger.info("Benchmark suite completed!")
        return self.results
    
    def print_benchmark_report(self) -> None:
        """Print formatted benchmark report."""
        print("\n" + "="*70)
        print("MODEL BENCHMARK REPORT")
        print("="*70)
        
        # System info
        sys_info = self.results['system_info']
        print(f"Device: {self.results['device']}")
        print(f"CPU: {sys_info['cpu_count']} cores ({sys_info['cpu_count_logical']} logical)")
        print(f"Memory: {sys_info['memory_total_gb']:.1f}GB total, {sys_info['memory_available_gb']:.1f}GB available")
        
        if sys_info.get('cuda_available'):
            print(f"GPU: {sys_info.get('gpu_name', 'Unknown')} ({sys_info.get('gpu_memory_gb', 0):.1f}GB)")
        else:
            print("GPU: Not available")
        
        # Benchmarks
        benchmarks = self.results['benchmarks']
        
        # Model loading
        if 'model_loading' in benchmarks:
            loading = benchmarks['model_loading']
            print(f"\n📦 Model Loading:")
            print(f"  Average time: {loading['avg_time_seconds']:.3f}s ± {loading['std_time_seconds']:.3f}s")
            print(f"  Memory usage: {loading['avg_memory_mb']:.1f}MB ± {loading['std_memory_mb']:.1f}MB")
        
        # Inference speed
        if 'inference_speed' in benchmarks:
            inference = benchmarks['inference_speed']
            print(f"\n🚀 Inference Speed:")
            print(f"  Average time: {inference['avg_time_seconds']*1000:.1f}ms ± {inference['std_time_seconds']*1000:.1f}ms")
            print(f"  Throughput: {inference['throughput_samples_per_second']:.1f} samples/second")
            print(f"  95th percentile: {inference['percentile_95_seconds']*1000:.1f}ms")
        
        # Grad-CAM
        if 'gradcam_generation' in benchmarks:
            gradcam = benchmarks['gradcam_generation']
            print(f"\n🎯 Grad-CAM Generation:")
            print(f"  CAM generation: {gradcam['cam_generation']['avg_time_seconds']*1000:.1f}ms")
            print(f"  Overlay generation: {gradcam['overlay_generation']['avg_time_seconds']*1000:.1f}ms")
            print(f"  Total time: {gradcam['total_gradcam']['avg_time_seconds']*1000:.1f}ms")
            print(f"  Throughput: {gradcam['total_gradcam']['throughput_samples_per_second']:.1f} samples/second")
        
        # Preprocessing
        if 'preprocessing' in benchmarks:
            preproc = benchmarks['preprocessing']
            print(f"\n⚡ Preprocessing:")
            print(f"  Average time: {preproc['avg_time_seconds']*1000:.1f}ms ± {preproc['std_time_seconds']*1000:.1f}ms")
            print(f"  Throughput: {preproc['throughput_samples_per_second']:.1f} samples/second")
        
        # Memory usage
        if 'memory_usage' in benchmarks:
            memory = benchmarks['memory_usage']
            print(f"\n💾 Memory Usage:")
            print(f"  Model overhead: {memory['model_overhead_mb']:.1f}MB")
            print(f"  Inference overhead: {memory['inference_overhead_mb']:.1f}MB")
            print(f"  Grad-CAM overhead: {memory['gradcam_overhead_mb']:.1f}MB")
            
            if memory.get('gpu'):
                gpu_mem = memory['gpu']
                print(f"  GPU model memory: {gpu_mem['model_memory_mb']:.1f}MB")
                print(f"  GPU inference memory: {gpu_mem['inference_memory_mb']:.1f}MB")
        
        print("="*70)


def main():
    parser = argparse.ArgumentParser(description='Benchmark XAI Medical Images model')
    parser.add_argument(
        '--model_path',
        type=str,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        help='Save benchmark results to JSON file'
    )
    parser.add_argument(
        '--inference_samples',
        type=int,
        default=100,
        help='Number of samples for inference benchmark'
    )
    parser.add_argument(
        '--gradcam_samples',
        type=int,
        default=50,
        help='Number of samples for Grad-CAM benchmark'
    )
    parser.add_argument(
        '--preprocessing_samples',
        type=int,
        default=100,
        help='Number of samples for preprocessing benchmark'
    )
    
    args = parser.parse_args()
    
    try:
        benchmark = ModelBenchmark(args.model_path)
        
        # Run benchmarks
        results = benchmark.run_full_benchmark()
        
        # Print report
        benchmark.print_benchmark_report()
        
        # Save results
        if args.output_file:
            with open(args.output_file, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Benchmark results saved to: {args.output_file}")
        
    except KeyboardInterrupt:
        logger.info("Benchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        raise


if __name__ == '__main__':
    main()