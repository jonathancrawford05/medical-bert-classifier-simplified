"""
Medical BERT Models Module
Specialized BERT implementations for medical text classification
"""

from .base_classifier import BaseMedicalBERTClassifier, MedicalBERTForSequenceClassification, LastLayerTrainer
from .bio_bert import BioBERTClassifier, BioBERTConfig, create_biobert_model
from .clinical_bert import ClinicalBERTClassifier, ClinicalBERTConfig, create_clinical_bert_model

__all__ = [
    # Base classes
    "BaseMedicalBERTClassifier",
    "MedicalBERTForSequenceClassification", 
    "LastLayerTrainer",
    
    # Bio-BERT
    "BioBERTClassifier",
    "BioBERTConfig", 
    "create_biobert_model",
    
    # Clinical-Bio-BERT
    "ClinicalBERTClassifier",
    "ClinicalBERTConfig",
    "create_clinical_bert_model"
]

# Model registry for easy access
MODEL_REGISTRY = {
    'bio_bert': {
        'class': BioBERTClassifier,
        'create_func': create_biobert_model,
        'config': BioBERTConfig,
        'model_name': 'dmis-lab/biobert-base-cased-v1.1'
    },
    'clinical_bert': {
        'class': ClinicalBERTClassifier,
        'create_func': create_clinical_bert_model,
        'config': ClinicalBERTConfig,
        'model_name': 'emilyalsentzer/Bio_ClinicalBERT'
    }
}

def get_available_models():
    """Get list of available medical BERT models"""
    return list(MODEL_REGISTRY.keys())

def create_model(model_name: str, num_classes: int = 10):
    """Factory function to create any available model"""
    if model_name not in MODEL_REGISTRY:
        available = ', '.join(get_available_models())
        raise ValueError(f"Unknown model '{model_name}'. Available models: {available}")
    
    return MODEL_REGISTRY[model_name]['create_func'](num_classes=num_classes)
