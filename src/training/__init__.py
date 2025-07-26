"""
Training Module
Last-layer training components for medical BERT classifiers
"""

from .last_layer_trainer import (
    LastLayerTrainer,
    MedicalTextDataset,
    create_datasets_from_csv
)

__all__ = [
    "LastLayerTrainer",
    "MedicalTextDataset", 
    "create_datasets_from_csv"
]

def get_trainer_info():
    """Get information about the training approach"""
    return {
        "approach": "last_layer_only",
        "description": "Freeze BERT backbone and train only the classification layer",
        "advantages": [
            "Fast training",
            "Low memory usage", 
            "Reduced overfitting risk",
            "Quick experimentation"
        ],
        "suitable_for": [
            "Small datasets",
            "CPU training",
            "Rapid prototyping",
            "Model comparison"
        ]
    }
