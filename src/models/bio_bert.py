"""
Bio-BERT Implementation for Medical Text Classification
Specialized for biomedical literature and terminology
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional
import yaml
from .base_classifier import BaseMedicalBERTClassifier


class BioBERTClassifier(BaseMedicalBERTClassifier):
    """
    Bio-BERT implementation optimized for biomedical text classification
    
    Bio-BERT is pre-trained on PubMed abstracts and PMC full-text articles,
    making it particularly effective for biomedical terminology and concepts.
    """
    
    def __init__(
        self, 
        model_name: str = "dmis-lab/biobert-base-cased-v1.1",
        num_classes: int = 10,
        max_length: int = 512,
        dropout_rate: float = 0.3,
        freeze_backbone: bool = True
    ):
        super().__init__(
            model_name=model_name,
            num_classes=num_classes,
            max_length=max_length,
            dropout_rate=dropout_rate,
            freeze_backbone=freeze_backbone
        )
        
        # Bio-BERT specific configuration
        self.biomedical_domains = {
            'medications': ['drug', 'medication', 'treatment', 'therapy', 'prescription'],
            'procedures': ['surgery', 'procedure', 'intervention', 'operation'],
            'diagnoses': ['disease', 'condition', 'disorder', 'syndrome'],
            'anatomy': ['organ', 'tissue', 'cell', 'system'],
            'symptoms': ['symptom', 'sign', 'complaint', 'presentation']
        }
        
    def preprocess_text(self, text: str) -> str:
        """
        Bio-BERT specific text preprocessing
        
        Bio-BERT is case-sensitive and trained on formal biomedical text,
        so we preserve case and handle biomedical terminology carefully.
        """
        # Preserve original case for biomedical entities
        text = text.strip()
        
        # Handle common biomedical abbreviations
        biomedical_abbrevs = {
            'mg': 'milligrams',
            'ml': 'milliliters', 
            'kg': 'kilograms',
            'bp': 'blood pressure',
            'hr': 'heart rate',
            'temp': 'temperature'
        }
        
        # Expand critical abbreviations for better understanding
        for abbrev, expansion in biomedical_abbrevs.items():
            text = text.replace(f' {abbrev} ', f' {expansion} ')
            text = text.replace(f' {abbrev.upper()} ', f' {expansion} ')
        
        return text
    
    def get_biomedical_confidence(self, text: str) -> float:
        """
        Calculate confidence score for biomedical content
        
        Returns higher confidence for text containing biomedical terminology
        """
        text_lower = text.lower()
        biomedical_terms = 0
        total_terms = 0
        
        for domain, terms in self.biomedical_domains.items():
            for term in terms:
                if term in text_lower:
                    biomedical_terms += 1
                total_terms += 1
        
        return biomedical_terms / total_terms if total_terms > 0 else 0.0
    
    def predict_with_biomedical_context(
        self, 
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        text: str = None
    ) -> Dict:
        """
        Enhanced prediction with biomedical context awareness
        """
        with torch.no_grad():
            logits = self.forward(input_ids, attention_mask)
            probabilities = torch.softmax(logits, dim=-1)
            predicted_class = torch.argmax(logits, dim=-1)
            
            result = {
                'predicted_class': predicted_class.item(),
                'probabilities': probabilities.squeeze().tolist(),
                'confidence': torch.max(probabilities).item()
            }
            
            # Add biomedical confidence if text provided
            if text:
                result['biomedical_confidence'] = self.get_biomedical_confidence(text)
                
            return result


class BioBERTConfig:
    """Configuration class for Bio-BERT specific settings"""
    
    def __init__(self):
        self.model_name = "dmis-lab/biobert-base-cased-v1.1"
        self.case_sensitive = True
        self.domain = "biomedical_literature"
        self.optimal_batch_sizes = {
            'cpu': 8,
            'gpu': 16
        }
        self.recommended_learning_rates = {
            'last_layer_only': 2e-4,
            'full_fine_tuning': 2e-5
        }
        
        # Bio-BERT performs best on these medical text types
        self.optimal_text_types = [
            'research_abstracts',
            'clinical_literature',
            'drug_descriptions',
            'biomedical_procedures',
            'disease_descriptions'
        ]
        
        # Special tokens and handling
        self.special_tokens = {
            'cls_token': '[CLS]',
            'sep_token': '[SEP]',
            'mask_token': '[MASK]',
            'unk_token': '[UNK]',
            'pad_token': '[PAD]'
        }
        
    def get_optimization_config(self) -> Dict:
        """Get optimized configuration for Bio-BERT training"""
        return {
            'learning_rate': self.recommended_learning_rates['last_layer_only'],
            'warmup_ratio': 0.1,
            'weight_decay': 0.01,
            'max_grad_norm': 1.0,
            'scheduler': 'linear_warmup'
        }
        
    def get_preprocessing_config(self) -> Dict:
        """Get text preprocessing configuration"""
        return {
            'preserve_case': True,
            'expand_abbreviations': True,
            'handle_biomedical_entities': True,
            'max_length': 512,
            'padding': 'max_length',
            'truncation': True
        }


def create_biobert_model(num_classes: int = 10, config_path: str = None) -> BioBERTClassifier:
    """
    Factory function to create Bio-BERT classifier with optimal settings
    
    Args:
        num_classes: Number of classification classes
        config_path: Path to configuration file (optional)
        
    Returns:
        BioBERTClassifier instance
    """
    if config_path:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        num_classes = config.get('num_classes', num_classes)
    
    bio_config = BioBERTConfig()
    
    return BioBERTClassifier(
        model_name=bio_config.model_name,
        num_classes=num_classes,
        max_length=512,
        dropout_rate=0.3,
        freeze_backbone=True
    )


def test_biobert_model():
    """Test function for Bio-BERT model"""
    print("Testing Bio-BERT Model...")
    
    # Create model
    model = create_biobert_model(num_classes=10)
    model.print_parameter_summary()
    
    # Test text samples
    test_samples = [
        "Patient prescribed metformin 500mg twice daily for type 2 diabetes",
        "Hemoglobin A1c levels measured at 7.2% indicating good glycemic control",
        "Myocardial infarction diagnosed via elevated cardiac enzymes and ECG changes",
        "Coronary angioplasty performed with stent placement in LAD vessel"
    ]
    
    for text in test_samples:
        processed = model.preprocess_text(text)
        bio_confidence = model.get_biomedical_confidence(text)
        print(f"Original: {text}")
        print(f"Processed: {processed}")
        print(f"Biomedical confidence: {bio_confidence:.3f}")
        print("-" * 50)


if __name__ == "__main__":
    test_biobert_model()
