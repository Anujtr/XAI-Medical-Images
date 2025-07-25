"""
Health check system for XAI Medical Images application.
"""

import time
import psutil
import torch
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
import requests
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health check status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"  
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    name: str
    status: HealthStatus
    message: str
    details: Optional[Dict] = None
    timestamp: Optional[str] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


class HealthChecker:
    """Comprehensive health checking system."""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.start_time = time.time()
        self.last_successful_inference = None
        self.consecutive_failures = 0
        self.max_consecutive_failures = 5
        
        logger.info("Health checker initialized")
    
    def check_system_resources(self) -> HealthCheckResult:
        """Check system resource availability."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1.0)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Disk space
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            
            details = {
                'cpu_percent': cpu_percent,
                'memory_percent': memory_percent,
                'memory_available_gb': memory.available / (1024**3),
                'disk_percent': disk_percent,
                'disk_free_gb': disk.free / (1024**3)
            }
            
            # Determine status based on thresholds
            if cpu_percent > 95 or memory_percent > 95 or disk_percent > 95:
                return HealthCheckResult(
                    name="system_resources",
                    status=HealthStatus.CRITICAL,
                    message="Critical resource usage detected",
                    details=details
                )
            elif cpu_percent > 80 or memory_percent > 85 or disk_percent > 85:
                return HealthCheckResult(
                    name="system_resources",
                    status=HealthStatus.DEGRADED,
                    message="High resource usage detected",
                    details=details
                )
            else:
                return HealthCheckResult(
                    name="system_resources",
                    status=HealthStatus.HEALTHY,
                    message="System resources are healthy",
                    details=details
                )
        
        except Exception as e:
            return HealthCheckResult(
                name="system_resources",
                status=HealthStatus.UNHEALTHY,
                message=f"Failed to check system resources: {str(e)}"
            )
    
    def check_model_availability(self) -> HealthCheckResult:
        """Check if the model is loaded and available."""
        try:
            # Check if model file exists (if specified)
            if self.model_path:
                model_file = Path(self.model_path)
                if not model_file.exists():
                    return HealthCheckResult(
                        name="model_availability",
                        status=HealthStatus.CRITICAL,
                        message=f"Model file not found: {self.model_path}"
                    )
                
                # Check file size (basic sanity check)
                file_size_mb = model_file.stat().st_size / (1024**2)
                if file_size_mb < 1:  # Model should be at least 1MB
                    return HealthCheckResult(
                        name="model_availability",
                        status=HealthStatus.UNHEALTHY,
                        message=f"Model file too small: {file_size_mb:.1f}MB"
                    )
            
            # Try to import and create model (basic test)
            try:
                import torchvision.models as models
                model = models.resnet50(pretrained=False)
                model.fc = torch.nn.Linear(model.fc.in_features, 2)
                del model  # Clean up
            except Exception as e:
                return HealthCheckResult(
                    name="model_availability",
                    status=HealthStatus.UNHEALTHY,
                    message=f"Failed to create model: {str(e)}"
                )
            
            details = {
                'model_path': self.model_path,
                'model_file_exists': bool(self.model_path and Path(self.model_path).exists()),
                'pytorch_available': True
            }
            
            return HealthCheckResult(
                name="model_availability",
                status=HealthStatus.HEALTHY,
                message="Model is available",
                details=details
            )
        
        except Exception as e:
            return HealthCheckResult(
                name="model_availability",
                status=HealthStatus.UNHEALTHY,
                message=f"Model availability check failed: {str(e)}"
            )
    
    def check_gpu_status(self) -> HealthCheckResult:
        """Check GPU availability and status."""
        try:
            if not torch.cuda.is_available():
                return HealthCheckResult(
                    name="gpu_status",
                    status=HealthStatus.HEALTHY,
                    message="GPU not available (using CPU)",
                    details={'cuda_available': False}
                )
            
            # GPU is available
            gpu_count = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(current_device)
            
            # Check GPU memory
            memory_allocated = torch.cuda.memory_allocated(current_device)
            memory_reserved = torch.cuda.memory_reserved(current_device)
            memory_total = torch.cuda.get_device_properties(current_device).total_memory
            
            memory_usage_percent = (memory_allocated / memory_total) * 100
            
            details = {
                'cuda_available': True,
                'gpu_count': gpu_count,
                'current_device': current_device,
                'device_name': device_name,
                'memory_allocated_mb': memory_allocated / (1024**2),
                'memory_reserved_mb': memory_reserved / (1024**2),
                'memory_total_mb': memory_total / (1024**2),
                'memory_usage_percent': memory_usage_percent
            }
            
            # Check GPU health
            if memory_usage_percent > 95:
                return HealthCheckResult(
                    name="gpu_status",
                    status=HealthStatus.CRITICAL,
                    message="GPU memory usage critical",
                    details=details
                )
            elif memory_usage_percent > 85:
                return HealthCheckResult(
                    name="gpu_status",
                    status=HealthStatus.DEGRADED,
                    message="GPU memory usage high",
                    details=details
                )
            else:
                return HealthCheckResult(
                    name="gpu_status",
                    status=HealthStatus.HEALTHY,
                    message="GPU is healthy",
                    details=details
                )
        
        except Exception as e:
            return HealthCheckResult(
                name="gpu_status",
                status=HealthStatus.UNHEALTHY,
                message=f"GPU status check failed: {str(e)}"
            )
    
    def check_dependencies(self) -> HealthCheckResult:
        """Check if all required dependencies are available."""
        try:
            required_modules = [
                'torch',
                'torchvision', 
                'opencv-python',
                'pydicom',
                'matplotlib',
                'numpy',
                'flask',
                'flask_cors',
                'PIL'
            ]
            
            missing_modules = []
            version_info = {}
            
            for module_name in required_modules:
                try:
                    if module_name == 'opencv-python':
                        import cv2
                        version_info['opencv'] = cv2.__version__
                    elif module_name == 'PIL':
                        import PIL
                        version_info['PIL'] = PIL.__version__
                    else:
                        module = __import__(module_name)
                        if hasattr(module, '__version__'):
                            version_info[module_name] = module.__version__
                except ImportError:
                    missing_modules.append(module_name)
            
            if missing_modules:
                return HealthCheckResult(
                    name="dependencies",
                    status=HealthStatus.CRITICAL,
                    message=f"Missing required modules: {', '.join(missing_modules)}",
                    details={'missing_modules': missing_modules, 'versions': version_info}
                )
            else:
                return HealthCheckResult(
                    name="dependencies",
                    status=HealthStatus.HEALTHY,
                    message="All dependencies available",
                    details={'versions': version_info}
                )
        
        except Exception as e:
            return HealthCheckResult(
                name="dependencies",
                status=HealthStatus.UNHEALTHY,
                message=f"Dependency check failed: {str(e)}"
            )
    
    def check_file_permissions(self) -> HealthCheckResult:
        """Check file system permissions for required directories."""
        try:
            required_dirs = [
                'static/uploads',
                'logs',
                'models/checkpoints'
            ]
            
            permission_issues = []
            
            for dir_path in required_dirs:
                path = Path(dir_path)
                
                # Check if directory exists
                if not path.exists():
                    try:
                        path.mkdir(parents=True, exist_ok=True)
                    except Exception as e:
                        permission_issues.append(f"Cannot create {dir_path}: {str(e)}")
                        continue
                
                # Check read permission
                if not os.access(path, os.R_OK):
                    permission_issues.append(f"No read permission for {dir_path}")
                
                # Check write permission
                if not os.access(path, os.W_OK):
                    permission_issues.append(f"No write permission for {dir_path}")
            
            if permission_issues:
                return HealthCheckResult(
                    name="file_permissions",
                    status=HealthStatus.DEGRADED,
                    message="File permission issues detected",
                    details={'issues': permission_issues}
                )
            else:
                return HealthCheckResult(
                    name="file_permissions",
                    status=HealthStatus.HEALTHY,
                    message="File permissions are correct"
                )
        
        except Exception as e:
            return HealthCheckResult(
                name="file_permissions",
                status=HealthStatus.UNHEALTHY,
                message=f"Permission check failed: {str(e)}"
            )
    
    def check_model_inference(self) -> HealthCheckResult:
        """Test model inference with dummy data."""
        try:
            import torchvision.models as models
            
            # Create model
            model = models.resnet50(pretrained=False)
            model.fc = torch.nn.Linear(model.fc.in_features, 2)
            
            # Load checkpoint if available
            if self.model_path and Path(self.model_path).exists():
                try:
                    checkpoint = torch.load(self.model_path, map_location='cpu')
                    if 'model_state_dict' in checkpoint:
                        model.load_state_dict(checkpoint['model_state_dict'])
                    else:
                        model.load_state_dict(checkpoint)
                except Exception as e:
                    logger.warning(f"Could not load model checkpoint: {e}")
            
            model.eval()
            
            # Test inference with dummy data
            start_time = time.time()
            with torch.no_grad():
                dummy_input = torch.randn(1, 3, 224, 224)
                output = model(dummy_input)
            inference_time = (time.time() - start_time) * 1000  # ms
            
            # Clean up
            del model, dummy_input, output
            
            # Update tracking
            self.last_successful_inference = datetime.now()
            self.consecutive_failures = 0
            
            return HealthCheckResult(
                name="model_inference",
                status=HealthStatus.HEALTHY,
                message=f"Model inference successful ({inference_time:.1f}ms)",
                details={
                    'inference_time_ms': inference_time,
                    'last_successful_inference': self.last_successful_inference.isoformat()
                }
            )
        
        except Exception as e:
            self.consecutive_failures += 1
            
            if self.consecutive_failures >= self.max_consecutive_failures:
                status = HealthStatus.CRITICAL
            else:
                status = HealthStatus.UNHEALTHY
            
            return HealthCheckResult(
                name="model_inference",
                status=status,
                message=f"Model inference failed: {str(e)}",
                details={
                    'consecutive_failures': self.consecutive_failures,
                    'max_consecutive_failures': self.max_consecutive_failures
                }
            )
    
    def check_external_services(self) -> HealthCheckResult:
        """Check connectivity to external services if any."""
        try:
            # For now, just check internet connectivity
            # In a real deployment, you might check database, cache, etc.
            
            try:
                response = requests.get('https://httpbin.org/status/200', timeout=5)
                internet_available = response.status_code == 200
            except:
                internet_available = False
            
            details = {
                'internet_connectivity': internet_available
            }
            
            if not internet_available:
                return HealthCheckResult(
                    name="external_services",
                    status=HealthStatus.DEGRADED,
                    message="Limited external connectivity",
                    details=details
                )
            else:
                return HealthCheckResult(
                    name="external_services",
                    status=HealthStatus.HEALTHY,
                    message="External services accessible",
                    details=details
                )
        
        except Exception as e:
            return HealthCheckResult(
                name="external_services",
                status=HealthStatus.UNHEALTHY,
                message=f"External service check failed: {str(e)}"
            )
    
    def run_all_checks(self) -> List[HealthCheckResult]:
        """Run all health checks and return results."""
        checks = [
            self.check_system_resources,
            self.check_model_availability,
            self.check_gpu_status,
            self.check_dependencies,
            self.check_file_permissions,
            self.check_model_inference,
            self.check_external_services
        ]
        
        results = []
        for check in checks:
            try:
                result = check()
                results.append(result)
            except Exception as e:
                # If a health check itself fails, record that
                results.append(HealthCheckResult(
                    name=check.__name__,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Health check failed: {str(e)}"
                ))
        
        return results
    
    def get_overall_status(self, results: List[HealthCheckResult]) -> Tuple[HealthStatus, str]:
        """Determine overall system health from individual check results."""
        status_counts = {status: 0 for status in HealthStatus}
        
        for result in results:
            status_counts[result.status] += 1
        
        # Determine overall status based on individual results
        if status_counts[HealthStatus.CRITICAL] > 0:
            return HealthStatus.CRITICAL, "System has critical issues"
        elif status_counts[HealthStatus.UNHEALTHY] > 0:
            return HealthStatus.UNHEALTHY, "System has unhealthy components"
        elif status_counts[HealthStatus.DEGRADED] > 0:
            return HealthStatus.DEGRADED, "System performance is degraded"
        else:
            return HealthStatus.HEALTHY, "All systems healthy"
    
    def get_health_summary(self) -> Dict:
        """Get a comprehensive health summary."""
        results = self.run_all_checks()
        overall_status, overall_message = self.get_overall_status(results)
        
        uptime_seconds = time.time() - self.start_time
        
        return {
            'timestamp': datetime.now().isoformat(),
            'overall_status': overall_status.value,
            'overall_message': overall_message,
            'uptime_seconds': uptime_seconds,
            'uptime_human': self._format_uptime(uptime_seconds),
            'checks': [
                {
                    'name': result.name,
                    'status': result.status.value,
                    'message': result.message,
                    'details': result.details,
                    'timestamp': result.timestamp
                }
                for result in results
            ],
            'summary': {
                'total_checks': len(results),
                'healthy': sum(1 for r in results if r.status == HealthStatus.HEALTHY),
                'degraded': sum(1 for r in results if r.status == HealthStatus.DEGRADED),
                'unhealthy': sum(1 for r in results if r.status == HealthStatus.UNHEALTHY),
                'critical': sum(1 for r in results if r.status == HealthStatus.CRITICAL)
            }
        }
    
    def _format_uptime(self, uptime_seconds: float) -> str:
        """Format uptime in human-readable format."""
        uptime_td = timedelta(seconds=int(uptime_seconds))
        days = uptime_td.days
        hours, remainder = divmod(uptime_td.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        if days > 0:
            return f"{days}d {hours}h {minutes}m"
        elif hours > 0:
            return f"{hours}h {minutes}m"
        else:
            return f"{minutes}m {seconds}s"


# Global health checker instance
_health_checker: Optional[HealthChecker] = None


def get_health_checker(model_path: Optional[str] = None) -> HealthChecker:
    """Get the global health checker instance."""
    global _health_checker
    if _health_checker is None:
        _health_checker = HealthChecker(model_path)
    return _health_checker


def initialize_health_checks(model_path: Optional[str] = None):
    """Initialize the health checking system."""
    global _health_checker
    _health_checker = HealthChecker(model_path)
    logger.info("Health checking system initialized")


def get_health_status() -> Dict:
    """Get current health status (convenience function)."""
    checker = get_health_checker()
    return checker.get_health_summary()