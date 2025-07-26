"""
Medical BERT Classifier - Simplified Last-Layer Training
A streamlined approach to medical text classification using specialized BERT models
"""

__version__ = "1.0.0"
__author__ = "Medical NLP Team"
__description__ = "Efficient medical text classification with last-layer BERT fine-tuning"

# Package metadata
__all__ = [
    "models",
    "training", 
    "data",
    "__version__",
    "__author__",
    "__description__"
]

# Import main modules for easy access
try:
    from . import models
    from . import training
    from . import data
except ImportError:
    # Handle cases where dependencies aren't installed
    pass

def get_version():
    """Get package version"""
    return __version__

def get_info():
    """Get package information"""
    return {
        "name": "medical-bert-classifier-simplified",
        "version": __version__,
        "description": __description__,
        "author": __author__
    }
