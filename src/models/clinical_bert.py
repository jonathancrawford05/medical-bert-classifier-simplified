"""
Clinical-Bio-BERT Implementation for Medical Text Classification
Specialized for clinical notes and electronic health records
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import yaml
import re
from .base_classifier import BaseMedicalBERTClassifier


class ClinicalBERTClassifier(BaseMedicalBERTClassifier):
    """
    Clinical-Bio-BERT implementation optimized for clinical text classification
    
    Clinical-Bio-BERT is Bio-BERT further pre-trained on clinical notes from MIMIC-III,
    making it particularly effective for clinical documentation and EHR text.
    """
    
    def __init__(
        self, 
        model_name: str = "emilyalsentzer/Bio_ClinicalBERT",
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
        
        # Clinical-specific configuration
        self.clinical_domains = {
            'medications': ['prescribed', 'medication', 'drug', 'dosage', 'therapy', 'treatment'],
            'laboratory': ['lab', 'test', 'result', 'level', 'value', 'analysis'],
            'diagnoses': ['diagnosed', 'condition', 'disease', 'disorder', 'syndrome'],
            'procedures': ['procedure', 'surgery', 'intervention', 'performed', 'operation'],
            'symptoms': ['complaint', 'symptom', 'pain', 'reports', 'experiences'],
            'vitals': ['blood pressure', 'heart rate', 'temperature', 'weight', 'height'],
            'family_history': ['family history', 'mother', 'father', 'sibling', 'genetic'],
            'social': ['social', 'lifestyle', 'smoking', 'alcohol', 'occupation'],
            'tobacco': ['smoking', 'tobacco', 'cigarettes', 'quit', 'pack per day']
        }
        
        # Clinical abbreviations commonly found in EHRs
        self.clinical_abbreviations = {
            # Vital signs and measurements
            'bp': 'blood pressure',
            'hr': 'heart rate', 
            'rr': 'respiratory rate',
            'temp': 'temperature',
            'wt': 'weight',
            'ht': 'height',
            'bmi': 'body mass index',
            
            # Laboratory
            'cbc': 'complete blood count',
            'bmp': 'basic metabolic panel',
            'hgb': 'hemoglobin',
            'hct': 'hematocrit',
            'wbc': 'white blood cell',
            'rbc': 'red blood cell',
            'plt': 'platelet',
            'glucose': 'blood glucose',
            'a1c': 'hemoglobin a1c',
            
            # Medical history
            'hx': 'history',
            'fhx': 'family history',
            'pmh': 'past medical history',
            'psh': 'past surgical history',
            'soc hx': 'social history',
            
            # Clinical terms
            'dx': 'diagnosis',
            'rx': 'prescription',
            'tx': 'treatment',
            'sx': 'symptoms',
            'px': 'procedure',
            
            # Time references
            'y.o.': 'year old',
            'y/o': 'year old',
            'yo': 'year old',
            
            # Common medical terms
            'pt': 'patient',
            'pts': 'patients',
            'c/o': 'complains of',
            'r/o': 'rule out',
            's/p': 'status post',
            'w/': 'with',
            'w/o': 'without'
        }
        
    def preprocess_text(self, text: str) -> str:
        """
        Clinical-BERT specific text preprocessing
        
        Handles clinical abbreviations and standardizes clinical terminology
        """
        text = text.strip()
        
        # Expand clinical abbreviations for better understanding
        text_lower = text.lower()
        for abbrev, expansion in self.clinical_abbreviations.items():
            # Use word boundaries to avoid partial matches
            pattern = r'\b' + re.escape(abbrev) + r'\b'
            text_lower = re.sub(pattern, expansion, text_lower)
        
        # Handle common clinical patterns
        text_lower = self._handle_clinical_patterns(text_lower)
        
        return text_lower
    
    def _handle_clinical_patterns(self, text: str) -> str:
        """Handle common clinical text patterns"""
        
        # Standardize dosage patterns
        text = re.sub(r'(\d+)\s*mg\b', r'\1 milligrams', text)
        text = re.sub(r'(\d+)\s*ml\b', r'\1 milliliters', text)
        text = re.sub(r'(\d+)\s*kg\b', r'\1 kilograms', text)
        
        # Standardize time patterns
        text = re.sub(r'(\d+)\s*x\s*daily', r'\1 times daily', text)
        text = re.sub(r'bid\b', 'twice daily', text)
        text = re.sub(r'tid\b', 'three times daily', text)
        text = re.sub(r'qid\b', 'four times daily', text)
        
        # Standardize vital signs patterns
        text = re.sub(r'(\d+)/(\d+)\s*mmhg', r'blood pressure \1 over \2', text)
        text = re.sub(r'(\d+)\s*bpm', r'\1 beats per minute', text)
        
        return text
    
    def get_clinical_confidence(self, text: str) -> float:
        """
        Calculate confidence score for clinical content
        
        Returns higher confidence for text containing clinical terminology
        """
        text_lower = text.lower()
        clinical_terms = 0
        total_terms = 0
        
        for domain, terms in self.clinical_domains.items():
            for term in terms:
                if term in text_lower:
                    clinical_terms += 1
                total_terms += 1
        
        # Check for clinical abbreviations
        for abbrev in self.clinical_abbreviations:
            if abbrev in text_lower:
                clinical_terms += 1
            total_terms += 1
        
        return clinical_terms / total_terms if total_terms > 0 else 0.0
    
    def identify_clinical_domain(self, text: str) -> Tuple[str, float]:
        """
        Identify the most likely clinical domain for the text
        
        Returns:
            Tuple of (domain_name, confidence_score)
        """
        text_lower = text.lower()
        domain_scores = {}
        
        for domain, terms in self.clinical_domains.items():
            score = 0
            for term in terms:
                if term in text_lower:
                    score += 1
            domain_scores[domain] = score / len(terms)
        
        if domain_scores:
            best_domain = max(domain_scores, key=domain_scores.get)
            return best_domain, domain_scores[best_domain]
        
        return 'unknown', 0.0
    
    def predict_with_clinical_context(
        self, 
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        text: str = None
    ) -> Dict:
        """
        Enhanced prediction with clinical context awareness
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
            
            # Add clinical context if text provided
            if text:
                result['clinical_confidence'] = self.get_clinical_confidence(text)
                domain, domain_confidence = self.identify_clinical_domain(text)
                result['clinical_domain'] = domain
                result['domain_confidence'] = domain_confidence
                
            return result
    
    def extract_clinical_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract clinical entities from text using pattern matching
        
        This is a simplified approach - in production you might use 
        specialized NER models for clinical text
        """
        text_lower = text.lower()
        entities = {
            'medications': [],
            'dosages': [],
            'vital_signs': [],
            'laboratory_values': [],
            'procedures': []
        }
        
        # Extract medications (simple pattern)
        med_patterns = [
            r'\b(metformin|lisinopril|atorvastatin|levothyroxine|amlodipine)\b',
            r'\b(\w+)\s+\d+\s*mg\b'
        ]
        
        for pattern in med_patterns:
            matches = re.findall(pattern, text_lower)
            entities['medications'].extend(matches)
        
        # Extract dosages
        dosage_patterns = [r'\b\d+\s*mg\b', r'\b\d+\s*ml\b', r'\btimes daily\b']
        for pattern in dosage_patterns:
            matches = re.findall(pattern, text_lower)
            entities['dosages'].extend(matches)
        
        # Extract vital signs
        vital_patterns = [
            r'\bbp?\s*\d+/\d+\b',
            r'\bheart rate\s*\d+\b',
            r'\btemperature\s*\d+\.?\d*\b'
        ]
        for pattern in vital_patterns:
            matches = re.findall(pattern, text_lower)
            entities['vital_signs'].extend(matches)
        
        return entities


class ClinicalBERTConfig:
    """Configuration class for Clinical-BERT specific settings"""
    
    def __init__(self):
        self.model_name = "emilyalsentzer/Bio_ClinicalBERT"
        self.case_sensitive = False
        self.domain = "clinical_notes"
        self.optimal_batch_sizes = {
            'cpu': 8,
            'gpu': 16
        }
        self.recommended_learning_rates = {
            'last_layer_only': 2e-4,
            'full_fine_tuning': 2e-5
        }
        
        # Clinical-BERT performs best on these text types
        self.optimal_text_types = [
            'clinical_notes',
            'discharge_summaries',
            'progress_notes',
            'consultation_reports',
            'nursing_notes',
            'medication_lists',
            'laboratory_reports'
        ]
        
    def get_optimization_config(self) -> Dict:
        """Get optimized configuration for Clinical-BERT training"""
        return {
            'learning_rate': self.recommended_learning_rates['last_layer_only'],
            'warmup_ratio': 0.1,
            'weight_decay': 0.01,
            'max_grad_norm': 1.0,
            'scheduler': 'linear_warmup',
            'label_smoothing': 0.1  # Helps with clinical text ambiguity
        }
        
    def get_preprocessing_config(self) -> Dict:
        """Get text preprocessing configuration"""
        return {
            'preserve_case': False,
            'expand_abbreviations': True,
            'handle_clinical_patterns': True,
            'standardize_dosages': True,
            'max_length': 512,
            'padding': 'max_length',
            'truncation': True
        }


def create_clinical_bert_model(num_classes: int = 10, config_path: str = None) -> ClinicalBERTClassifier:
    """
    Factory function to create Clinical-BERT classifier with optimal settings
    
    Args:
        num_classes: Number of classification classes
        config_path: Path to configuration file (optional)
        
    Returns:
        ClinicalBERTClassifier instance
    """
    if config_path:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        num_classes = config.get('num_classes', num_classes)
    
    clinical_config = ClinicalBERTConfig()
    
    return ClinicalBERTClassifier(
        model_name=clinical_config.model_name,
        num_classes=num_classes,
        max_length=512,
        dropout_rate=0.3,
        freeze_backbone=True
    )


def test_clinical_bert_model():
    """Test function for Clinical-BERT model"""
    print("Testing Clinical-BERT Model...")
    
    # Create model
    model = create_clinical_bert_model(num_classes=10)
    model.print_parameter_summary()
    
    # Test text samples
    test_samples = [
        "Pt c/o chest pain, BP 140/90, HR 85 bpm, prescribed metformin 500mg bid",
        "Labs: CBC w/ diff, BMP, HgbA1c 7.2%, LDL 140 mg/dL",
        "PMH significant for DM type 2, HTN, hyperlipidemia s/p MI 2019",
        "FHx positive for CAD in father, DM in mother, no known genetic conditions",
        "Social hx: former smoker, quit 5 years ago, 20 pack-year history"
    ]
    
    for text in test_samples:
        processed = model.preprocess_text(text)
        clinical_confidence = model.get_clinical_confidence(text)
        domain, domain_conf = model.identify_clinical_domain(text)
        entities = model.extract_clinical_entities(text)
        
        print(f"Original: {text}")
        print(f"Processed: {processed}")
        print(f"Clinical confidence: {clinical_confidence:.3f}")
        print(f"Clinical domain: {domain} (confidence: {domain_conf:.3f})")
        print(f"Entities: {entities}")
        print("-" * 80)


if __name__ == "__main__":
    test_clinical_bert_model()
