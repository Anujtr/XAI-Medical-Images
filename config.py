"""
Configuration management for XAI Medical Images application.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any
import yaml
import json


@dataclass
class ModelConfig:
    """Model configuration settings."""
    architecture: str = 'resnet50'
    num_classes: int = 2
    pretrained: bool = True
    checkpoint_path: Optional[str] = None
    device: str = 'auto'  # 'auto', 'cpu', 'cuda'
    
    
@dataclass
class TrainingConfig:
    """Training configuration settings."""
    batch_size: int = 32
    learning_rate: float = 1e-4
    num_epochs: int = 50
    weight_decay: float = 1e-4
    val_split: float = 0.2
    save_every: int = 10
    early_stopping_patience: int = 10
    use_tensorboard: bool = True
    

@dataclass
class DataConfig:
    """Data configuration settings."""
    data_dir: str = 'data'
    csv_file: Optional[str] = None
    image_size: tuple = (224, 224)
    binary_classification: bool = True
    num_workers: int = 2
    augmentation: bool = True
    

@dataclass
class GradCAMConfig:
    """Grad-CAM configuration settings."""
    target_layer: str = 'layer4'
    alpha: float = 0.4
    colormap: str = 'jet'
    use_relu: bool = True
    

@dataclass
class FlaskConfig:
    """Flask application configuration."""
    host: str = '0.0.0.0'
    port: int = 5000
    debug: bool = False
    max_content_length: int = 16 * 1024 * 1024  # 16MB
    upload_folder: str = 'static/uploads'
    allowed_extensions: List[str] = None
    cors_enabled: bool = True
    
    def __post_init__(self):
        if self.allowed_extensions is None:
            self.allowed_extensions = ['png', 'jpg', 'jpeg', 'dcm', 'dicom']


@dataclass
class LoggingConfig:
    """Logging configuration settings."""
    level: str = 'INFO'
    format: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    log_dir: str = 'logs'
    max_bytes: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    

@dataclass
class SecurityConfig:
    """Security configuration settings."""
    secret_key: Optional[str] = None
    enable_csrf: bool = True
    max_requests_per_minute: int = 60
    trusted_hosts: List[str] = None
    
    def __post_init__(self):
        if self.trusted_hosts is None:
            self.trusted_hosts = ['localhost', '127.0.0.1']


class Config:
    """Main configuration class."""
    
    def __init__(self, config_file: Optional[str] = None, environment: str = 'development'):
        self.environment = environment
        self.config_file = config_file
        
        # Initialize default configurations
        self.model = ModelConfig()
        self.training = TrainingConfig()
        self.data = DataConfig()
        self.gradcam = GradCAMConfig()
        self.flask = FlaskConfig()
        self.logging = LoggingConfig()
        self.security = SecurityConfig()
        
        # Load configuration from file if provided
        if config_file and os.path.exists(config_file):
            self.load_from_file(config_file)
        
        # Override with environment variables
        self.load_from_env()
        
        # Apply environment-specific settings
        self._apply_environment_settings()
    
    def load_from_file(self, config_file: str):
        """Load configuration from YAML or JSON file."""
        config_path = Path(config_file)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_file}")
        
        with open(config_path, 'r') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                config_data = yaml.safe_load(f)
            elif config_path.suffix.lower() == '.json':
                config_data = json.load(f)
            else:
                raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")
        
        # Update configurations with loaded data
        self._update_from_dict(config_data)
    
    def load_from_env(self):
        """Load configuration from environment variables."""
        env_mappings = {
            # Flask config
            'FLASK_HOST': ('flask', 'host'),
            'FLASK_PORT': ('flask', 'port', int),
            'FLASK_DEBUG': ('flask', 'debug', lambda x: x.lower() == 'true'),
            'MAX_CONTENT_LENGTH': ('flask', 'max_content_length', int),
            
            # Model config
            'MODEL_CHECKPOINT_PATH': ('model', 'checkpoint_path'),
            'MODEL_DEVICE': ('model', 'device'),
            'MODEL_NUM_CLASSES': ('model', 'num_classes', int),
            
            # Training config
            'BATCH_SIZE': ('training', 'batch_size', int),
            'LEARNING_RATE': ('training', 'learning_rate', float),
            'NUM_EPOCHS': ('training', 'num_epochs', int),
            
            # Data config
            'DATA_DIR': ('data', 'data_dir'),
            'CSV_FILE': ('data', 'csv_file'),
            'NUM_WORKERS': ('data', 'num_workers', int),
            
            # Logging config
            'LOG_LEVEL': ('logging', 'level'),
            'LOG_DIR': ('logging', 'log_dir'),
            
            # Security config
            'SECRET_KEY': ('security', 'secret_key'),
        }
        
        for env_var, config_path in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                section = config_path[0]
                key = config_path[1]
                converter = config_path[2] if len(config_path) > 2 else str
                
                try:
                    converted_value = converter(env_value)
                    setattr(getattr(self, section), key, converted_value)
                except (ValueError, TypeError) as e:
                    print(f"Warning: Could not convert environment variable {env_var}={env_value}: {e}")
    
    def _update_from_dict(self, config_data: Dict[str, Any]):
        """Update configuration from dictionary."""
        for section_name, section_data in config_data.items():
            if hasattr(self, section_name) and isinstance(section_data, dict):
                section_obj = getattr(self, section_name)
                for key, value in section_data.items():
                    if hasattr(section_obj, key):
                        setattr(section_obj, key, value)
    
    def _apply_environment_settings(self):
        """Apply environment-specific settings."""
        if self.environment == 'development':
            self.flask.debug = True
            self.logging.level = 'DEBUG'
            self.security.enable_csrf = False
        elif self.environment == 'production':
            self.flask.debug = False
            self.logging.level = 'INFO'
            self.security.enable_csrf = True
            # Generate secret key if not provided
            if not self.security.secret_key:
                self.security.secret_key = os.urandom(32).hex()
        elif self.environment == 'testing':
            self.flask.debug = False
            self.logging.level = 'WARNING'
            self.training.num_epochs = 2  # Faster testing
            self.training.batch_size = 4
    
    def save_to_file(self, config_file: str):
        """Save current configuration to file."""
        config_data = self.to_dict()
        config_path = Path(config_file)
        
        # Create directory if it doesn't exist
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                yaml.safe_dump(config_data, f, default_flow_style=False, indent=2)
            elif config_path.suffix.lower() == '.json':
                json.dump(config_data, f, indent=2, default=str)
            else:
                raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'model': self.model.__dict__,
            'training': self.training.__dict__,
            'data': self.data.__dict__,
            'gradcam': self.gradcam.__dict__,
            'flask': self.flask.__dict__,
            'logging': self.logging.__dict__,
            'security': {k: v for k, v in self.security.__dict__.items() if k != 'secret_key'}
        }
    
    def validate(self):
        """Validate configuration settings."""
        errors = []
        
        # Validate paths
        if self.model.checkpoint_path and not Path(self.model.checkpoint_path).exists():
            errors.append(f"Model checkpoint path does not exist: {self.model.checkpoint_path}")
        
        if not Path(self.data.data_dir).exists():
            errors.append(f"Data directory does not exist: {self.data.data_dir}")
        
        # Validate numeric ranges
        if self.training.learning_rate <= 0:
            errors.append("Learning rate must be positive")
        
        if self.training.batch_size <= 0:
            errors.append("Batch size must be positive")
        
        if not 0 < self.training.val_split < 1:
            errors.append("Validation split must be between 0 and 1")
        
        if not 0 <= self.gradcam.alpha <= 1:
            errors.append("Grad-CAM alpha must be between 0 and 1")
        
        # Validate Flask settings
        if not 1 <= self.flask.port <= 65535:
            errors.append("Flask port must be between 1 and 65535")
        
        if errors:
            raise ValueError(f"Configuration validation failed:\n" + "\n".join(f"- {error}" for error in errors))
    
    def get_model_save_path(self) -> str:
        """Get the path where model checkpoints should be saved."""
        return os.path.join('models', 'checkpoints', f'{self.model.architecture}_best.pth')
    
    def get_log_file_path(self) -> str:
        """Get the path for log files."""
        os.makedirs(self.logging.log_dir, exist_ok=True)
        return os.path.join(self.logging.log_dir, 'app.log')


def load_config(config_file: Optional[str] = None, environment: Optional[str] = None) -> Config:
    """
    Load configuration from file and environment.
    
    Args:
        config_file: Path to configuration file
        environment: Environment name (development, production, testing)
    
    Returns:
        Configured Config instance
    """
    if environment is None:
        environment = os.getenv('ENVIRONMENT', 'development')
    
    # Try to find config file if not specified
    if config_file is None:
        possible_files = [
            f'config/{environment}.yaml',
            f'config/{environment}.yml',
            f'config/{environment}.json',
            'config/default.yaml',
            'config/default.yml',
            'config/default.json'
        ]
        
        for file_path in possible_files:
            if os.path.exists(file_path):
                config_file = file_path
                break
    
    config = Config(config_file, environment)
    config.validate()
    
    return config


# Global configuration instance
_global_config: Optional[Config] = None


def get_config() -> Config:
    """Get the global configuration instance."""
    global _global_config
    if _global_config is None:
        _global_config = load_config()
    return _global_config


def set_config(config: Config):
    """Set the global configuration instance."""
    global _global_config
    _global_config = config