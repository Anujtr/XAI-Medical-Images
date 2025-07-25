#!/usr/bin/env python3
"""
Flask application entry point for XAI Medical Images.
"""

import os
import sys
from flask_cors import CORS

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.routes import create_app


def main():
    """Main application entry point."""
    # Look for model checkpoint
    model_path = os.path.join(os.path.dirname(__file__), 'models', 'checkpoints', 'best_model.pth')
    if not os.path.exists(model_path):
        model_path = None
        print("No model checkpoint found. Using pretrained ResNet-50.")
    
    # Create Flask app
    app = create_app(model_path)
    
    # Enable CORS for all routes
    CORS(app)
    
    # Configuration
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    host = os.environ.get('FLASK_HOST', '0.0.0.0')
    
    print(f"Starting XAI Medical Images server on {host}:{port}")
    print(f"Debug mode: {debug}")
    
    # Run the app
    app.run(
        host=host,
        port=port,
        debug=debug,
        threaded=True
    )


if __name__ == '__main__':
    main()