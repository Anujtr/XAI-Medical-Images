"""
Model registry and versioning system for XAI Medical Images.
"""

import os
import json
import hashlib
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import torch
import logging

logger = logging.getLogger(__name__)


@dataclass
class ModelMetadata:
    """Metadata for a registered model."""
    name: str
    version: str
    description: str
    created_at: str
    file_path: str
    file_size_bytes: int
    file_hash: str
    architecture: str
    num_classes: int
    training_config: Dict[str, Any]
    performance_metrics: Dict[str, float]
    tags: List[str]
    author: Optional[str] = None
    training_dataset: Optional[str] = None
    validation_accuracy: Optional[float] = None
    notes: Optional[str] = None


@dataclass 
class ModelVersion:
    """Information about a specific model version."""
    version: str
    created_at: str
    file_path: str
    metadata: ModelMetadata
    is_active: bool = False
    is_deprecated: bool = False


class ModelRegistry:
    """Registry for managing model versions and metadata."""
    
    def __init__(self, registry_dir: str = "models"):
        self.registry_dir = Path(registry_dir)
        self.checkpoints_dir = self.registry_dir / "checkpoints"
        self.metadata_file = self.registry_dir / "registry.json"
        
        # Create directories
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing registry
        self.models = self._load_registry()
        
        logger.info(f"Model registry initialized at {self.registry_dir}")
    
    def _load_registry(self) -> Dict[str, Dict[str, ModelVersion]]:
        """Load the model registry from disk."""
        if not self.metadata_file.exists():
            return {}
        
        try:
            with open(self.metadata_file, 'r') as f:
                registry_data = json.load(f)
            
            # Convert to ModelVersion objects
            models = {}
            for model_name, versions_data in registry_data.items():
                models[model_name] = {}
                for version, version_data in versions_data.items():
                    metadata = ModelMetadata(**version_data['metadata'])
                    models[model_name][version] = ModelVersion(
                        version=version_data['version'],
                        created_at=version_data['created_at'],
                        file_path=version_data['file_path'],
                        metadata=metadata,
                        is_active=version_data.get('is_active', False),
                        is_deprecated=version_data.get('is_deprecated', False)
                    )
            
            logger.info(f"Loaded {len(models)} models from registry")
            return models
        
        except Exception as e:
            logger.error(f"Error loading registry: {e}")
            return {}
    
    def _save_registry(self):
        """Save the model registry to disk."""
        try:
            # Convert to serializable format
            registry_data = {}
            for model_name, versions in self.models.items():
                registry_data[model_name] = {}
                for version, model_version in versions.items():
                    registry_data[model_name][version] = {
                        'version': model_version.version,
                        'created_at': model_version.created_at,
                        'file_path': model_version.file_path,
                        'metadata': asdict(model_version.metadata),
                        'is_active': model_version.is_active,
                        'is_deprecated': model_version.is_deprecated
                    }
            
            # Write to file
            with open(self.metadata_file, 'w') as f:
                json.dump(registry_data, f, indent=2)
            
            logger.debug("Registry saved to disk")
        
        except Exception as e:
            logger.error(f"Error saving registry: {e}")
            raise
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of a file."""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    def _generate_version(self, model_name: str) -> str:
        """Generate next version number for a model."""
        if model_name not in self.models:
            return "1.0.0"
        
        # Get highest version
        versions = list(self.models[model_name].keys())
        version_numbers = []
        
        for version in versions:
            try:
                parts = version.split('.')
                if len(parts) == 3:
                    major, minor, patch = map(int, parts)
                    version_numbers.append((major, minor, patch))
            except ValueError:
                continue
        
        if not version_numbers:
            return "1.0.0"
        
        # Increment patch version
        latest = max(version_numbers)
        return f"{latest[0]}.{latest[1]}.{latest[2] + 1}"
    
    def register_model(self,
                      model_path: str,
                      name: str,
                      description: str,
                      architecture: str = "resnet50",
                      num_classes: int = 2,
                      training_config: Optional[Dict] = None,
                      performance_metrics: Optional[Dict] = None,
                      tags: Optional[List[str]] = None,
                      version: Optional[str] = None,
                      author: Optional[str] = None,
                      training_dataset: Optional[str] = None,
                      notes: Optional[str] = None) -> str:
        """
        Register a new model version.
        
        Args:
            model_path: Path to the model file
            name: Model name
            description: Model description
            architecture: Model architecture name
            num_classes: Number of output classes
            training_config: Training configuration used
            performance_metrics: Model performance metrics
            tags: List of tags for categorization
            version: Specific version (auto-generated if None)
            author: Model author
            training_dataset: Dataset used for training
            notes: Additional notes
            
        Returns:
            Version string of the registered model
        """
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Generate version if not provided
        if version is None:
            version = self._generate_version(name)
        
        # Check if version already exists
        if name in self.models and version in self.models[name]:
            raise ValueError(f"Version {version} already exists for model {name}")
        
        # Copy model file to registry
        file_extension = model_path.suffix
        registry_filename = f"{name}_v{version}{file_extension}"
        registry_filepath = self.checkpoints_dir / registry_filename
        
        shutil.copy2(model_path, registry_filepath)
        
        # Calculate metadata
        file_size = registry_filepath.stat().st_size
        file_hash = self._calculate_file_hash(registry_filepath)
        created_at = datetime.now().isoformat()
        
        # Create metadata
        metadata = ModelMetadata(
            name=name,
            version=version,
            description=description,
            created_at=created_at,
            file_path=str(registry_filepath),
            file_size_bytes=file_size,
            file_hash=file_hash,
            architecture=architecture,
            num_classes=num_classes,
            training_config=training_config or {},
            performance_metrics=performance_metrics or {},
            tags=tags or [],
            author=author,
            training_dataset=training_dataset,
            notes=notes
        )
        
        # Add validation accuracy to metadata if provided in performance_metrics
        if performance_metrics and 'validation_accuracy' in performance_metrics:
            metadata.validation_accuracy = performance_metrics['validation_accuracy']
        
        # Create model version
        model_version = ModelVersion(
            version=version,
            created_at=created_at,
            file_path=str(registry_filepath),
            metadata=metadata,
            is_active=False
        )
        
        # Add to registry
        if name not in self.models:
            self.models[name] = {}
        
        self.models[name][version] = model_version
        
        # Save registry
        self._save_registry()
        
        logger.info(f"Registered model {name} version {version}")
        return version
    
    def get_model_versions(self, name: str) -> List[ModelVersion]:
        """Get all versions of a model."""
        if name not in self.models:
            return []
        
        return list(self.models[name].values())
    
    def get_model_version(self, name: str, version: str) -> Optional[ModelVersion]:
        """Get a specific model version."""
        if name not in self.models or version not in self.models[name]:
            return None
        
        return self.models[name][version]
    
    def get_latest_version(self, name: str) -> Optional[ModelVersion]:
        """Get the latest version of a model."""
        if name not in self.models:
            return None
        
        versions = self.models[name]
        if not versions:
            return None
        
        # Sort versions and return latest
        sorted_versions = sorted(versions.keys(), key=lambda v: self._version_key(v), reverse=True)
        return versions[sorted_versions[0]]
    
    def get_active_version(self, name: str) -> Optional[ModelVersion]:
        """Get the active version of a model."""
        if name not in self.models:
            return None
        
        for version in self.models[name].values():
            if version.is_active:
                return version
        
        return None
    
    def set_active_version(self, name: str, version: str):
        """Set a specific version as active."""
        if name not in self.models or version not in self.models[name]:
            raise ValueError(f"Model {name} version {version} not found")
        
        # Deactivate all versions
        for v in self.models[name].values():
            v.is_active = False
        
        # Activate specified version
        self.models[name][version].is_active = True
        
        # Save registry
        self._save_registry()
        
        logger.info(f"Set {name} version {version} as active")
    
    def deprecate_version(self, name: str, version: str):
        """Mark a model version as deprecated."""
        if name not in self.models or version not in self.models[name]:
            raise ValueError(f"Model {name} version {version} not found")
        
        self.models[name][version].is_deprecated = True
        
        # If this was the active version, deactivate it
        if self.models[name][version].is_active:
            self.models[name][version].is_active = False
        
        # Save registry
        self._save_registry()
        
        logger.info(f"Deprecated {name} version {version}")
    
    def delete_version(self, name: str, version: str, force: bool = False):
        """Delete a model version."""
        if name not in self.models or version not in self.models[name]:
            raise ValueError(f"Model {name} version {version} not found")
        
        model_version = self.models[name][version]
        
        # Prevent deletion of active version unless forced
        if model_version.is_active and not force:
            raise ValueError("Cannot delete active version. Use force=True or deactivate first.")
        
        # Delete file
        file_path = Path(model_version.file_path)
        if file_path.exists():
            file_path.unlink()
        
        # Remove from registry
        del self.models[name][version]
        
        # Remove model entirely if no versions left
        if not self.models[name]:
            del self.models[name]
        
        # Save registry
        self._save_registry()
        
        logger.info(f"Deleted {name} version {version}")
    
    def list_models(self) -> List[str]:
        """List all registered model names."""
        return list(self.models.keys())
    
    def search_models(self, query: str = None, tags: List[str] = None, 
                     architecture: str = None) -> List[ModelVersion]:
        """Search models by various criteria."""
        results = []
        
        for model_name, versions in self.models.items():
            for version in versions.values():
                # Skip deprecated versions unless specifically searching for them
                if version.is_deprecated and query != "deprecated":
                    continue
                
                # Text search in name and description
                if query:
                    if query.lower() not in model_name.lower() and \
                       query.lower() not in version.metadata.description.lower():
                        continue
                
                # Tag filter
                if tags:
                    if not any(tag in version.metadata.tags for tag in tags):
                        continue
                
                # Architecture filter
                if architecture:
                    if version.metadata.architecture != architecture:
                        continue
                
                results.append(version)
        
        return results
    
    def get_model_info(self, name: str) -> Dict[str, Any]:
        """Get comprehensive information about a model."""
        if name not in self.models:
            raise ValueError(f"Model {name} not found")
        
        versions = self.models[name]
        active_version = self.get_active_version(name)
        latest_version = self.get_latest_version(name)
        
        return {
            'name': name,
            'total_versions': len(versions),
            'active_version': active_version.version if active_version else None,
            'latest_version': latest_version.version if latest_version else None,
            'versions': [
                {
                    'version': v.version,
                    'created_at': v.created_at,
                    'is_active': v.is_active,
                    'is_deprecated': v.is_deprecated,
                    'description': v.metadata.description,
                    'validation_accuracy': v.metadata.validation_accuracy,
                    'file_size_mb': v.metadata.file_size_bytes / (1024**2),
                    'tags': v.metadata.tags
                }
                for v in versions.values()
            ]
        }
    
    def load_model(self, name: str, version: str = None) -> torch.nn.Module:
        """Load a model from the registry."""
        if version is None:
            # Load active version, or latest if no active version
            model_version = self.get_active_version(name)
            if model_version is None:
                model_version = self.get_latest_version(name)
        else:
            model_version = self.get_model_version(name, version)
        
        if model_version is None:
            raise ValueError(f"Model {name} version {version} not found")
        
        # Load the model
        try:
            import torchvision.models as models
            
            # Create model architecture
            if model_version.metadata.architecture == "resnet50":
                model = models.resnet50(pretrained=False)
                model.fc = torch.nn.Linear(model.fc.in_features, model_version.metadata.num_classes)
            else:
                raise ValueError(f"Unsupported architecture: {model_version.metadata.architecture}")
            
            # Load weights
            checkpoint = torch.load(model_version.file_path, map_location='cpu')
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            model.eval()
            
            logger.info(f"Loaded model {name} version {model_version.version}")
            return model
        
        except Exception as e:
            logger.error(f"Error loading model {name} version {model_version.version}: {e}")
            raise
    
    def compare_versions(self, name: str, version1: str, version2: str) -> Dict[str, Any]:
        """Compare two versions of a model."""
        v1 = self.get_model_version(name, version1)
        v2 = self.get_model_version(name, version2)
        
        if v1 is None or v2 is None:
            raise ValueError("One or both versions not found")
        
        return {
            'model_name': name,
            'version1': {
                'version': v1.version,
                'created_at': v1.created_at,
                'description': v1.metadata.description,
                'validation_accuracy': v1.metadata.validation_accuracy,
                'file_size_mb': v1.metadata.file_size_bytes / (1024**2),
                'performance_metrics': v1.metadata.performance_metrics
            },
            'version2': {
                'version': v2.version,
                'created_at': v2.created_at,
                'description': v2.metadata.description,
                'validation_accuracy': v2.metadata.validation_accuracy,
                'file_size_mb': v2.metadata.file_size_bytes / (1024**2),
                'performance_metrics': v2.metadata.performance_metrics
            },
            'comparison': {
                'accuracy_diff': (v2.metadata.validation_accuracy or 0) - (v1.metadata.validation_accuracy or 0),
                'size_diff_mb': (v2.metadata.file_size_bytes - v1.metadata.file_size_bytes) / (1024**2),
                'newer_version': v2.version if v2.created_at > v1.created_at else v1.version
            }
        }
    
    def _version_key(self, version: str) -> Tuple[int, int, int]:
        """Convert version string to tuple for sorting."""
        try:
            parts = version.split('.')
            if len(parts) >= 3:
                return (int(parts[0]), int(parts[1]), int(parts[2]))
            else:
                return (0, 0, 0)
        except ValueError:
            return (0, 0, 0)
    
    def export_registry(self, output_file: str):
        """Export the entire registry to a JSON file."""
        try:
            registry_data = {}
            for model_name, versions in self.models.items():
                registry_data[model_name] = {}
                for version, model_version in versions.items():
                    registry_data[model_name][version] = {
                        'version': model_version.version,
                        'created_at': model_version.created_at,
                        'file_path': model_version.file_path,
                        'metadata': asdict(model_version.metadata),
                        'is_active': model_version.is_active,
                        'is_deprecated': model_version.is_deprecated
                    }
            
            with open(output_file, 'w') as f:
                json.dump(registry_data, f, indent=2)
            
            logger.info(f"Registry exported to {output_file}")
        
        except Exception as e:
            logger.error(f"Error exporting registry: {e}")
            raise


# Global model registry instance
_model_registry: Optional[ModelRegistry] = None


def get_model_registry(registry_dir: str = "models") -> ModelRegistry:
    """Get the global model registry instance."""
    global _model_registry
    if _model_registry is None:
        _model_registry = ModelRegistry(registry_dir)
    return _model_registry