"""
Base Medical BERT Classifier with Last-Layer Training
Implements simplified fine-tuning approach focusing on classifier layer only
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, List
import yaml
import os
from abc import ABC, abstractmethod
from transformers import (
    AutoModel, 
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)


class BaseMedicalBERTClassifier(nn.Module, ABC):
    """
    Abstract base class for medical BERT classifiers with last-layer training
    """
    
    def __init__(
        self, 
        model_name: str,
        num_classes: int,
        max_length: int = 512,
        dropout_rate: float = 0.3,
        freeze_backbone: bool = True
    ):
        super().__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        self.max_length = max_length
        self.dropout_rate = dropout_rate
        self.freeze_backbone = freeze_backbone
        
        # Load pre-trained BERT model
        self.bert = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Freeze BERT parameters for last-layer training
        if freeze_backbone:
            self._freeze_bert_layers()
            
        # Classification head (only trainable component)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.bert.config.hidden_size, num_classes)
        )
        
    def _freeze_bert_layers(self):
        """Freeze all BERT parameters except classifier"""
        for param in self.bert.parameters():
            param.requires_grad = False
            
    def forward(
        self, 
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through the model
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            token_type_ids: Token type IDs [batch_size, seq_len] (optional)
            
        Returns:
            logits: Classification logits [batch_size, num_classes]
        """
        # Get BERT outputs
        bert_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # Use [CLS] token representation
        pooled_output = bert_outputs.last_hidden_state[:, 0]  # [CLS] token
        
        # Apply classifier
        logits = self.classifier(pooled_output)
        
        return logits
    
    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """Get only trainable parameters (classifier layer)"""
        return [p for p in self.parameters() if p.requires_grad]
    
    def get_num_trainable_parameters(self) -> int:
        """Count trainable parameters"""
        return sum(p.numel() for p in self.get_trainable_parameters())
    
    def print_parameter_summary(self):
        """Print summary of model parameters"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = self.get_num_trainable_parameters()
        frozen_params = total_params - trainable_params
        
        print(f"\n{'='*50}")
        print(f"Model: {self.model_name}")
        print(f"{'='*50}")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
        print(f"Frozen parameters: {frozen_params:,} ({frozen_params/total_params*100:.2f}%)")
        print(f"{'='*50}")


class MedicalBERTForSequenceClassification(nn.Module):
    """
    Simplified wrapper for Hugging Face transformers compatibility
    """
    
    def __init__(self, config_path: str = "config/classes.yaml"):
        super().__init__()
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.num_classes = self.config['num_classes']
        self.class_labels = self.config['class_labels']
        
    def create_model(self, model_name: str) -> BaseMedicalBERTClassifier:
        """Factory method to create specific model instances"""
        
        if "biobert" in model_name.lower():
            return BioBERTClassifier(
                model_name=model_name,
                num_classes=self.num_classes
            )
        elif "clinical" in model_name.lower():
            return ClinicalBERTClassifier(
                model_name=model_name,
                num_classes=self.num_classes
            )
        elif "blue" in model_name.lower():
            return BlueBERTClassifier(
                model_name=model_name,
                num_classes=self.num_classes
            )
        else:
            # Default to base implementation
            return BaseMedicalBERTClassifier(
                model_name=model_name,
                num_classes=self.num_classes
            )


class BioBERTClassifier(BaseMedicalBERTClassifier):
    """BioBERT implementation for medical text classification"""
    
    def __init__(self, model_name: str, num_classes: int, **kwargs):
        super().__init__(model_name, num_classes, **kwargs)
        
    def preprocess_text(self, text: str) -> str:
        """BioBERT-specific text preprocessing"""
        # BioBERT handles biomedical terminology well
        # Keep original case for better biomedical entity recognition
        return text.strip()


class ClinicalBERTClassifier(BaseMedicalBERTClassifier):
    """Clinical-BioBERT implementation for clinical text classification"""
    
    def __init__(self, model_name: str, num_classes: int, **kwargs):
        super().__init__(model_name, num_classes, **kwargs)
        
    def preprocess_text(self, text: str) -> str:
        """Clinical-BERT-specific text preprocessing"""
        # Clinical-BERT is trained on clinical notes
        # Normalize common clinical abbreviations
        text = text.strip()
        # Add any clinical-specific preprocessing here
        return text


class BlueBERTClassifier(BaseMedicalBERTClassifier):
    """BlueBERT implementation for mixed biomedical/clinical text"""
    
    def __init__(self, model_name: str, num_classes: int, **kwargs):
        super().__init__(model_name, num_classes, **kwargs)
        
    def preprocess_text(self, text: str) -> str:
        """BlueBERT-specific text preprocessing"""
        # BlueBERT handles both biomedical and clinical text
        text = text.strip().lower()  # BlueBERT is uncased
        return text


class LastLayerTrainer:
    """
    Simplified trainer focusing on last-layer training only
    Based on approaches from the attached notebooks
    """
    
    def __init__(
        self, 
        model: BaseMedicalBERTClassifier,
        train_dataloader: torch.utils.data.DataLoader,
        val_dataloader: torch.utils.data.DataLoader,
        config_path: str = "config/training.yaml"
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        
        # Load training configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # Setup device
        self.device = self._setup_device()
        self.model.to(self.device)
        
        # Setup optimizer (only for trainable parameters)
        self.optimizer = torch.optim.AdamW(
            self.model.get_trainable_parameters(),
            lr=self._get_learning_rate(),
            weight_decay=self.config['optimizer'].get('weight_decay', 0.01)
        )
        
        # Setup loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Setup metrics tracking
        self.train_losses = []
        self.val_accuracies = []
        self.best_val_accuracy = 0.0
        
    def _setup_device(self) -> torch.device:
        """Setup training device (CPU/GPU)"""
        if self.config['device']['auto_detect'] and torch.cuda.is_available():
            if self.config['device']['prefer_gpu']:
                return torch.device(f"cuda:{self.config['device']['cuda_device_id']}")
        return torch.device("cpu")
    
    def _get_learning_rate(self) -> float:
        """Get learning rate based on device"""
        if self.device.type == "cuda":
            return self.config['gpu_config']['learning_rate']
        else:
            return self.config['cpu_config']['learning_rate']
            
    def _get_batch_config(self) -> Dict:
        """Get batch configuration based on device"""
        if self.device.type == "cuda":
            return self.config['gpu_config']
        else:
            return self.config['cpu_config']
    
    def train_epoch(self) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in self.train_dataloader:
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
            loss = self.criterion(logits, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if self.config['advanced'].get('gradient_clipping'):
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config['advanced']['gradient_clipping']
                )
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
        return total_loss / num_batches
    
    def evaluate(self) -> Tuple[float, float]:
        """Evaluate model on validation set"""
        self.model.eval()
        total_correct = 0
        total_samples = 0
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
                loss = self.criterion(logits, labels)
                
                predictions = torch.argmax(logits, dim=1)
                total_correct += (predictions == labels).sum().item()
                total_samples += labels.size(0)
                total_loss += loss.item()
                num_batches += 1
        
        accuracy = total_correct / total_samples
        avg_loss = total_loss / num_batches
        
        return accuracy, avg_loss
    
    def train(self, num_epochs: Optional[int] = None) -> Dict:
        """Main training loop"""
        if num_epochs is None:
            num_epochs = self._get_batch_config()['num_epochs']
            
        print(f"Training on device: {self.device}")
        self.model.print_parameter_summary()
        
        for epoch in range(num_epochs):
            # Train epoch
            train_loss = self.train_epoch()
            
            # Evaluate
            val_accuracy, val_loss = self.evaluate()
            
            # Track metrics
            self.train_losses.append(train_loss)
            self.val_accuracies.append(val_accuracy)
            
            # Save best model
            if val_accuracy > self.best_val_accuracy:
                self.best_val_accuracy = val_accuracy
                self._save_model()
            
            print(f"Epoch {epoch+1}/{num_epochs} - "
                  f"Train Loss: {train_loss:.4f}, "
                  f"Val Accuracy: {val_accuracy:.4f}, "
                  f"Val Loss: {val_loss:.4f}")
        
        return {
            'train_losses': self.train_losses,
            'val_accuracies': self.val_accuracies,
            'best_val_accuracy': self.best_val_accuracy
        }
    
    def _save_model(self):
        """Save the best model"""
        output_dir = self.config['output']['output_dir']
        os.makedirs(output_dir, exist_ok=True)
        
        model_path = os.path.join(output_dir, f"best_{self.model.model_name.replace('/', '_')}.pt")
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_accuracy': self.best_val_accuracy,
            'config': self.config
        }, model_path)
        
        print(f"Model saved to {model_path}")
