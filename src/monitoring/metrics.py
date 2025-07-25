"""
Application metrics collection and reporting for XAI Medical Images.
"""

import time
import psutil
import torch
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import threading
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class SystemMetrics:
    """System resource metrics."""
    timestamp: str
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    disk_usage_percent: float
    disk_free_gb: float
    gpu_memory_used_mb: Optional[float] = None
    gpu_memory_total_mb: Optional[float] = None
    gpu_utilization_percent: Optional[float] = None


@dataclass
class ApplicationMetrics:
    """Application-specific metrics."""
    timestamp: str
    total_requests: int
    successful_requests: int
    failed_requests: int
    avg_response_time_ms: float
    p95_response_time_ms: float
    p99_response_time_ms: float
    current_active_requests: int
    model_inference_count: int
    gradcam_generation_count: int
    upload_count: int
    error_count: int


@dataclass
class ModelMetrics:
    """Model performance metrics."""
    timestamp: str
    inference_time_ms: float
    preprocessing_time_ms: float
    gradcam_time_ms: float
    total_time_ms: float
    prediction: str
    confidence: float
    image_size_bytes: int
    model_version: str


class MetricsCollector:
    """Collect and manage application metrics."""
    
    def __init__(self):
        self.start_time = time.time()
        self.request_times = []
        self.model_metrics = []
        self.lock = threading.Lock()
        
        # Counters
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.model_inference_count = 0
        self.gradcam_generation_count = 0
        self.upload_count = 0
        self.error_count = 0
        self.current_active_requests = 0
        
        # Keep last 1000 response times for percentile calculation
        self.max_response_times = 1000
        
        logger.info("Metrics collector initialized")
    
    def collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics."""
        # CPU and memory
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # GPU metrics if available
        gpu_memory_used = None
        gpu_memory_total = None
        gpu_utilization = None
        
        if torch.cuda.is_available():
            try:
                gpu_memory_used = torch.cuda.memory_allocated() / (1024**2)  # MB
                gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**2)  # MB
                # Note: GPU utilization requires nvidia-ml-py for accurate readings
            except Exception as e:
                logger.warning(f"Could not collect GPU metrics: {e}")
        
        return SystemMetrics(
            timestamp=datetime.now().isoformat(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_used_mb=memory.used / (1024**2),
            memory_available_mb=memory.available / (1024**2),
            disk_usage_percent=disk.used / disk.total * 100,
            disk_free_gb=disk.free / (1024**3),
            gpu_memory_used_mb=gpu_memory_used,
            gpu_memory_total_mb=gpu_memory_total,
            gpu_utilization_percent=gpu_utilization
        )
    
    def collect_application_metrics(self) -> ApplicationMetrics:
        """Collect current application metrics."""
        with self.lock:
            # Calculate response time percentiles
            if self.request_times:
                avg_response_time = sum(self.request_times) / len(self.request_times)
                sorted_times = sorted(self.request_times)
                p95_index = int(len(sorted_times) * 0.95)
                p99_index = int(len(sorted_times) * 0.99)
                p95_response_time = sorted_times[p95_index] if p95_index < len(sorted_times) else sorted_times[-1]
                p99_response_time = sorted_times[p99_index] if p99_index < len(sorted_times) else sorted_times[-1]
            else:
                avg_response_time = 0.0
                p95_response_time = 0.0
                p99_response_time = 0.0
            
            return ApplicationMetrics(
                timestamp=datetime.now().isoformat(),
                total_requests=self.total_requests,
                successful_requests=self.successful_requests,
                failed_requests=self.failed_requests,
                avg_response_time_ms=avg_response_time,
                p95_response_time_ms=p95_response_time,
                p99_response_time_ms=p99_response_time,
                current_active_requests=self.current_active_requests,
                model_inference_count=self.model_inference_count,
                gradcam_generation_count=self.gradcam_generation_count,
                upload_count=self.upload_count,
                error_count=self.error_count
            )
    
    def record_request_start(self) -> int:
        """Record the start of a request. Returns request ID."""
        with self.lock:
            self.total_requests += 1
            self.current_active_requests += 1
            return self.total_requests
    
    def record_request_end(self, request_id: int, success: bool, response_time_ms: float):
        """Record the end of a request."""
        with self.lock:
            self.current_active_requests = max(0, self.current_active_requests - 1)
            
            if success:
                self.successful_requests += 1
            else:
                self.failed_requests += 1
            
            # Store response time (keep only last N for memory efficiency)
            self.request_times.append(response_time_ms)
            if len(self.request_times) > self.max_response_times:
                self.request_times.pop(0)
    
    def record_model_inference(self, metrics: ModelMetrics):
        """Record model inference metrics."""
        with self.lock:
            self.model_inference_count += 1
            self.model_metrics.append(metrics)
            
            # Keep only last 100 model metrics for memory efficiency
            if len(self.model_metrics) > 100:
                self.model_metrics.pop(0)
    
    def record_gradcam_generation(self):
        """Record Grad-CAM generation event."""
        with self.lock:
            self.gradcam_generation_count += 1
    
    def record_upload(self):
        """Record file upload event."""
        with self.lock:
            self.upload_count += 1
    
    def record_error(self):
        """Record error event."""
        with self.lock:
            self.error_count += 1
    
    def get_uptime_seconds(self) -> float:
        """Get application uptime in seconds."""
        return time.time() - self.start_time
    
    def reset_counters(self):
        """Reset all counters (useful for periodic reporting)."""
        with self.lock:
            self.total_requests = 0
            self.successful_requests = 0
            self.failed_requests = 0
            self.model_inference_count = 0
            self.gradcam_generation_count = 0
            self.upload_count = 0
            self.error_count = 0
            self.request_times.clear()
            self.model_metrics.clear()
        
        logger.info("Metrics counters reset")


class MetricsReporter:
    """Report metrics to various destinations."""
    
    def __init__(self, collector: MetricsCollector, log_file: Optional[str] = None):
        self.collector = collector
        self.log_file = log_file
        self.is_running = False
        self.report_thread = None
        self.report_interval = 60  # seconds
    
    def start_periodic_reporting(self, interval_seconds: int = 60):
        """Start periodic metrics reporting."""
        self.report_interval = interval_seconds
        self.is_running = True
        self.report_thread = threading.Thread(target=self._periodic_report_worker, daemon=True)
        self.report_thread.start()
        logger.info(f"Started periodic metrics reporting every {interval_seconds} seconds")
    
    def stop_periodic_reporting(self):
        """Stop periodic metrics reporting."""
        self.is_running = False
        if self.report_thread:
            self.report_thread.join(timeout=5)
        logger.info("Stopped periodic metrics reporting")
    
    def _periodic_report_worker(self):
        """Worker thread for periodic reporting."""
        while self.is_running:
            try:
                self.report_all_metrics()
                time.sleep(self.report_interval)
            except Exception as e:
                logger.error(f"Error in periodic metrics reporting: {e}")
                time.sleep(self.report_interval)
    
    def report_all_metrics(self):
        """Report all current metrics."""
        system_metrics = self.collector.collect_system_metrics()
        app_metrics = self.collector.collect_application_metrics()
        
        # Log to file if specified
        if self.log_file:
            self._log_metrics_to_file(system_metrics, app_metrics)
        
        # Log to application logger
        self._log_metrics_to_logger(system_metrics, app_metrics)
    
    def _log_metrics_to_file(self, system_metrics: SystemMetrics, app_metrics: ApplicationMetrics):
        """Log metrics to JSON file."""
        try:
            metrics_data = {
                'system': asdict(system_metrics),
                'application': asdict(app_metrics),
                'uptime_seconds': self.collector.get_uptime_seconds()
            }
            
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(metrics_data) + '\n')
        
        except Exception as e:
            logger.error(f"Error writing metrics to file {self.log_file}: {e}")
    
    def _log_metrics_to_logger(self, system_metrics: SystemMetrics, app_metrics: ApplicationMetrics):
        """Log metrics to application logger."""
        uptime_hours = self.collector.get_uptime_seconds() / 3600
        
        logger.info(
            f"METRICS - "
            f"Uptime: {uptime_hours:.1f}h, "
            f"Requests: {app_metrics.total_requests} "
            f"({app_metrics.successful_requests} success, {app_metrics.failed_requests} failed), "
            f"Active: {app_metrics.current_active_requests}, "
            f"Avg Response: {app_metrics.avg_response_time_ms:.1f}ms, "
            f"CPU: {system_metrics.cpu_percent:.1f}%, "
            f"Memory: {system_metrics.memory_percent:.1f}% "
            f"({system_metrics.memory_used_mb:.0f}MB), "
            f"Model Inferences: {app_metrics.model_inference_count}, "
            f"Uploads: {app_metrics.upload_count}"
        )
        
        # Log GPU metrics if available
        if system_metrics.gpu_memory_used_mb is not None:
            gpu_usage_percent = (system_metrics.gpu_memory_used_mb / system_metrics.gpu_memory_total_mb * 100 
                               if system_metrics.gpu_memory_total_mb > 0 else 0)
            logger.info(
                f"GPU METRICS - "
                f"Memory: {system_metrics.gpu_memory_used_mb:.0f}MB / "
                f"{system_metrics.gpu_memory_total_mb:.0f}MB ({gpu_usage_percent:.1f}%)"
            )
    
    def export_metrics_json(self) -> Dict[str, Any]:
        """Export all metrics as JSON-serializable dictionary."""
        system_metrics = self.collector.collect_system_metrics()
        app_metrics = self.collector.collect_application_metrics()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'uptime_seconds': self.collector.get_uptime_seconds(),
            'system': asdict(system_metrics),
            'application': asdict(app_metrics),
            'recent_model_metrics': [asdict(m) for m in self.collector.model_metrics[-10:]]  # Last 10
        }


# Global metrics collector instance
_metrics_collector: Optional[MetricsCollector] = None
_metrics_reporter: Optional[MetricsReporter] = None


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector instance."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector


def get_metrics_reporter() -> MetricsReporter:
    """Get the global metrics reporter instance."""
    global _metrics_reporter
    if _metrics_reporter is None:
        collector = get_metrics_collector()
        _metrics_reporter = MetricsReporter(collector, log_file='logs/metrics.jsonl')
    return _metrics_reporter


def initialize_metrics(log_file: Optional[str] = None, report_interval: int = 60):
    """Initialize metrics collection and reporting."""
    global _metrics_collector, _metrics_reporter
    
    _metrics_collector = MetricsCollector()
    _metrics_reporter = MetricsReporter(_metrics_collector, log_file)
    
    # Start periodic reporting
    _metrics_reporter.start_periodic_reporting(report_interval)
    
    logger.info("Metrics system initialized")


def shutdown_metrics():
    """Shutdown metrics collection and reporting."""
    global _metrics_reporter
    
    if _metrics_reporter:
        _metrics_reporter.stop_periodic_reporting()
    
    logger.info("Metrics system shutdown")


# Context manager for request timing
class RequestTimer:
    """Context manager for timing requests."""
    
    def __init__(self, collector: MetricsCollector):
        self.collector = collector
        self.request_id = None
        self.start_time = None
    
    def __enter__(self):
        self.request_id = self.collector.record_request_start()
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            response_time_ms = (time.time() - self.start_time) * 1000
            success = exc_type is None
            self.collector.record_request_end(self.request_id, success, response_time_ms)
            
            if not success:
                self.collector.record_error()