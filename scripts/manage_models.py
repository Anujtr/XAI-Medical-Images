#!/usr/bin/env python3
"""
Model management script for XAI Medical Images.
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, Any

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from model_registry import get_model_registry, ModelRegistry


def list_models(registry: ModelRegistry):
    """List all registered models."""
    models = registry.list_models()
    
    if not models:
        print("No models registered.")
        return
    
    print(f"\nRegistered Models ({len(models)}):")
    print("-" * 50)
    
    for model_name in models:
        info = registry.get_model_info(model_name)
        active_version = info['active_version']
        latest_version = info['latest_version']
        total_versions = info['total_versions']
        
        print(f"📦 {model_name}")
        print(f"   Versions: {total_versions}")
        print(f"   Latest: {latest_version}")
        print(f"   Active: {active_version or 'None'}")
        print()


def show_model_info(registry: ModelRegistry, model_name: str):
    """Show detailed information about a model."""
    try:
        info = registry.get_model_info(model_name)
        
        print(f"\n📦 Model: {info['name']}")
        print(f"Total Versions: {info['total_versions']}")
        print(f"Active Version: {info['active_version'] or 'None'}")
        print(f"Latest Version: {info['latest_version']}")
        
        print(f"\nVersions:")
        print("-" * 80)
        
        for version_info in sorted(info['versions'], key=lambda x: x['created_at'], reverse=True):
            status = []
            if version_info['is_active']:
                status.append('ACTIVE')
            if version_info['is_deprecated']:
                status.append('DEPRECATED')
            
            status_str = f" [{', '.join(status)}]" if status else ""
            
            print(f"🔸 Version {version_info['version']}{status_str}")
            print(f"  Created: {version_info['created_at']}")
            print(f"  Description: {version_info['description']}")
            if version_info['validation_accuracy']:
                print(f"  Validation Accuracy: {version_info['validation_accuracy']:.3f}")
            print(f"  File Size: {version_info['file_size_mb']:.1f} MB")
            if version_info['tags']:
                print(f"  Tags: {', '.join(version_info['tags'])}")
            print()
    
    except ValueError as e:
        print(f"Error: {e}")


def register_model(registry: ModelRegistry, args):
    """Register a new model."""
    try:
        # Parse training config if provided
        training_config = {}
        if args.training_config:
            if os.path.exists(args.training_config):
                with open(args.training_config, 'r') as f:
                    training_config = json.load(f)
            else:
                # Try to parse as JSON string
                training_config = json.loads(args.training_config)
        
        # Parse performance metrics if provided
        performance_metrics = {}
        if args.performance_metrics:
            if os.path.exists(args.performance_metrics):
                with open(args.performance_metrics, 'r') as f:
                    performance_metrics = json.load(f)
            else:
                # Try to parse as JSON string
                performance_metrics = json.loads(args.performance_metrics)
        
        # Parse tags
        tags = args.tags.split(',') if args.tags else []
        
        version = registry.register_model(
            model_path=args.model_path,
            name=args.name,
            description=args.description,
            architecture=args.architecture,
            num_classes=args.num_classes,
            training_config=training_config,
            performance_metrics=performance_metrics,
            tags=tags,
            version=args.version,
            author=args.author,
            training_dataset=args.training_dataset,
            notes=args.notes
        )
        
        print(f"✅ Successfully registered {args.name} version {version}")
        
        if args.set_active:
            registry.set_active_version(args.name, version)
            print(f"✅ Set version {version} as active")
    
    except Exception as e:
        print(f"❌ Error registering model: {e}")
        sys.exit(1)


def set_active_version(registry: ModelRegistry, model_name: str, version: str):
    """Set a model version as active."""
    try:
        registry.set_active_version(model_name, version)
        print(f"✅ Set {model_name} version {version} as active")
    except Exception as e:
        print(f"❌ Error: {e}")


def deprecate_version(registry: ModelRegistry, model_name: str, version: str):
    """Deprecate a model version."""
    try:
        registry.deprecate_version(model_name, version)
        print(f"✅ Deprecated {model_name} version {version}")
    except Exception as e:
        print(f"❌ Error: {e}")


def delete_version(registry: ModelRegistry, model_name: str, version: str, force: bool = False):
    """Delete a model version."""
    try:
        registry.delete_version(model_name, version, force=force)
        print(f"✅ Deleted {model_name} version {version}")
    except Exception as e:
        print(f"❌ Error: {e}")


def compare_versions(registry: ModelRegistry, model_name: str, version1: str, version2: str):
    """Compare two model versions."""
    try:
        comparison = registry.compare_versions(model_name, version1, version2)
        
        print(f"\n🔍 Comparing {model_name} versions {version1} vs {version2}")
        print("-" * 60)
        
        print(f"\n📊 Version {version1}:")
        v1 = comparison['version1']
        print(f"  Created: {v1['created_at']}")
        print(f"  Description: {v1['description']}")
        print(f"  Validation Accuracy: {v1['validation_accuracy']}")
        print(f"  File Size: {v1['file_size_mb']:.1f} MB")
        
        print(f"\n📊 Version {version2}:")
        v2 = comparison['version2']
        print(f"  Created: {v2['created_at']}")
        print(f"  Description: {v2['description']}")
        print(f"  Validation Accuracy: {v2['validation_accuracy']}")
        print(f"  File Size: {v2['file_size_mb']:.1f} MB")
        
        print(f"\n📈 Comparison:")
        comp = comparison['comparison']
        if comp['accuracy_diff'] > 0:
            print(f"  Accuracy: +{comp['accuracy_diff']:.3f} (better)")
        elif comp['accuracy_diff'] < 0:
            print(f"  Accuracy: {comp['accuracy_diff']:.3f} (worse)")
        else:
            print(f"  Accuracy: No change")
        
        if comp['size_diff_mb'] > 0:
            print(f"  Size: +{comp['size_diff_mb']:.1f} MB (larger)")
        elif comp['size_diff_mb'] < 0:
            print(f"  Size: {comp['size_diff_mb']:.1f} MB (smaller)")
        else:
            print(f"  Size: No change")
        
        print(f"  Newer version: {comp['newer_version']}")
    
    except Exception as e:
        print(f"❌ Error: {e}")


def search_models(registry: ModelRegistry, query: str = None, tags: str = None, architecture: str = None):
    """Search models by criteria."""
    tag_list = tags.split(',') if tags else None
    
    results = registry.search_models(query=query, tags=tag_list, architecture=architecture)
    
    if not results:
        print("No models found matching criteria.")
        return
    
    print(f"\n🔍 Search Results ({len(results)} found):")
    print("-" * 60)
    
    for model_version in results:
        metadata = model_version.metadata
        status = []
        if model_version.is_active:
            status.append('ACTIVE')
        if model_version.is_deprecated:
            status.append('DEPRECATED')
        
        status_str = f" [{', '.join(status)}]" if status else ""
        
        print(f"📦 {metadata.name} v{metadata.version}{status_str}")
        print(f"  Description: {metadata.description}")
        print(f"  Architecture: {metadata.architecture}")
        if metadata.validation_accuracy:
            print(f"  Validation Accuracy: {metadata.validation_accuracy:.3f}")
        if metadata.tags:
            print(f"  Tags: {', '.join(metadata.tags)}")
        print()


def export_registry(registry: ModelRegistry, output_file: str):
    """Export registry to file."""
    try:
        registry.export_registry(output_file)
        print(f"✅ Registry exported to {output_file}")
    except Exception as e:
        print(f"❌ Error: {e}")


def main():
    parser = argparse.ArgumentParser(description='Manage XAI Medical Images models')
    parser.add_argument('--registry_dir', type=str, default='models', 
                       help='Model registry directory')
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # List models
    subparsers.add_parser('list', help='List all registered models')
    
    # Show model info
    info_parser = subparsers.add_parser('info', help='Show detailed model information')
    info_parser.add_argument('model_name', help='Model name')
    
    # Register model
    register_parser = subparsers.add_parser('register', help='Register a new model')
    register_parser.add_argument('model_path', help='Path to model file')
    register_parser.add_argument('name', help='Model name')
    register_parser.add_argument('description', help='Model description')
    register_parser.add_argument('--architecture', default='resnet50', help='Model architecture')
    register_parser.add_argument('--num_classes', type=int, default=2, help='Number of classes')
    register_parser.add_argument('--training_config', help='Training config JSON file or string')
    register_parser.add_argument('--performance_metrics', help='Performance metrics JSON file or string')
    register_parser.add_argument('--tags', help='Comma-separated tags')
    register_parser.add_argument('--version', help='Specific version (auto-generated if not provided)')
    register_parser.add_argument('--author', help='Model author')
    register_parser.add_argument('--training_dataset', help='Training dataset name/description')
    register_parser.add_argument('--notes', help='Additional notes')
    register_parser.add_argument('--set_active', action='store_true', help='Set as active version')
    
    # Set active version
    active_parser = subparsers.add_parser('activate', help='Set model version as active')
    active_parser.add_argument('model_name', help='Model name')
    active_parser.add_argument('version', help='Version to activate')
    
    # Deprecate version
    deprecate_parser = subparsers.add_parser('deprecate', help='Deprecate model version')
    deprecate_parser.add_argument('model_name', help='Model name')
    deprecate_parser.add_argument('version', help='Version to deprecate')
    
    # Delete version
    delete_parser = subparsers.add_parser('delete', help='Delete model version')
    delete_parser.add_argument('model_name', help='Model name')
    delete_parser.add_argument('version', help='Version to delete')
    delete_parser.add_argument('--force', action='store_true', help='Force delete even if active')
    
    # Compare versions
    compare_parser = subparsers.add_parser('compare', help='Compare two model versions')
    compare_parser.add_argument('model_name', help='Model name')
    compare_parser.add_argument('version1', help='First version')
    compare_parser.add_argument('version2', help='Second version')
    
    # Search models
    search_parser = subparsers.add_parser('search', help='Search models')
    search_parser.add_argument('--query', help='Text query (searches name and description)')
    search_parser.add_argument('--tags', help='Comma-separated tags to filter by')
    search_parser.add_argument('--architecture', help='Architecture to filter by')
    
    # Export registry
    export_parser = subparsers.add_parser('export', help='Export registry to file')
    export_parser.add_argument('output_file', help='Output file path')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Initialize registry
    registry = get_model_registry(args.registry_dir)
    
    # Execute command
    try:
        if args.command == 'list':
            list_models(registry)
        
        elif args.command == 'info':
            show_model_info(registry, args.model_name)
        
        elif args.command == 'register':
            register_model(registry, args)
        
        elif args.command == 'activate':
            set_active_version(registry, args.model_name, args.version)
        
        elif args.command == 'deprecate':
            deprecate_version(registry, args.model_name, args.version)
        
        elif args.command == 'delete':
            delete_version(registry, args.model_name, args.version, args.force)
        
        elif args.command == 'compare':
            compare_versions(registry, args.model_name, args.version1, args.version2)
        
        elif args.command == 'search':
            search_models(registry, args.query, args.tags, args.architecture)
        
        elif args.command == 'export':
            export_registry(registry, args.output_file)
    
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()