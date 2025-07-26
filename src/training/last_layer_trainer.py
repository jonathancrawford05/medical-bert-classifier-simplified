"""
Last-Layer Trainer for Medical BERT Classifiers
Implements simplified fine-tuning approach based on attached notebook patterns
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
import yaml
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import os
from tqdm.auto import tqdm
import time
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


class MedicalTextDataset(Dataset):
    """
    Custom dataset for medical text classification
    Based on patterns from the attached notebooks
    """
    
    def __init__(
        self, 
        texts: List[str], 
        labels: List[int], 
        tokenizer, 
        max_length: int = 512
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenize text following notebook patterns
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class LastLayerTrainer:
    """
    Simplified trainer focusing on last-layer training only
    Implements approaches from the Fine-tuning Transformers notebooks
    """
    
    def __init__(
        self, 
        model,
        train_dataset: Dataset,
        val_dataset: Dataset,
        config_path: str = "config/training.yaml"
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        
        # Load training configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # Setup device (CPU/GPU)
        self.device = self._setup_device()
        self.model.to(self.device)
        
        # Create data loaders
        self.train_dataloader = self._create_dataloader(train_dataset, is_train=True)
        self.val_dataloader = self._create_dataloader(val_dataset, is_train=False)
        
        # Setup optimizer and scheduler (following notebook patterns)
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        
        # Setup loss function and metrics
        self.criterion = nn.CrossEntropyLoss()
        
        # Training tracking
        self.train_losses = []
        self.val_accuracies = []
        self.val_losses = []
        self.best_val_accuracy = 0.0
        self.best_model_state = None
        
        print(f"Training setup complete. Device: {self.device}")
        self.model.print_parameter_summary()
    
    def _setup_device(self) -> torch.device:
        """Setup training device based on configuration"""
        device_config = self.config['device']
        
        if device_config['auto_detect'] and torch.cuda.is_available():
            if device_config['prefer_gpu']:
                device = torch.device(f"cuda:{device_config['cuda_device_id']}")
                print(f"Using GPU: {torch.cuda.get_device_name()}")
            else:
                device = torch.device("cpu")
                print("Using CPU (GPU available but CPU preferred)")
        else:
            device = torch.device("cpu")
            print("Using CPU")
            
        return device
    
    def _get_device_config(self) -> Dict:
        """Get device-specific configuration"""
        if self.device.type == "cuda":
            return self.config['gpu_config']
        else:
            return self.config['cpu_config']
    
    def _create_dataloader(self, dataset: Dataset, is_train: bool) -> DataLoader:
        """Create data loader with device-appropriate settings"""
        device_config = self._get_device_config()
        
        batch_size = device_config['batch_size']['train' if is_train else 'eval']
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=is_train,
            num_workers=device_config['dataloader_num_workers'],
            pin_memory=device_config.get('pin_memory', False)
        )
    
    def _setup_optimizer(self):
        """Setup optimizer following notebook patterns"""
        device_config = self._get_device_config()
        optimizer_config = self.config['optimizer']
        
        # Only optimize trainable parameters (classifier layer)
        trainable_params = self.model.get_trainable_parameters()
        
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=device_config['learning_rate'],
            betas=optimizer_config['betas'],
            eps=optimizer_config['eps'],
            weight_decay=device_config['weight_decay']
        )
        
        print(f"Optimizer setup: AdamW with LR={device_config['learning_rate']}")
        print(f"Trainable parameters: {self.model.get_num_trainable_parameters():,}")
        
        return optimizer
    
    def _setup_scheduler(self):
        """Setup learning rate scheduler"""
        device_config = self._get_device_config()
        total_steps = len(self.train_dataloader) * device_config['num_epochs']
        warmup_steps = device_config['warmup_steps']
        
        scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        print(f"Scheduler setup: Linear warmup with {warmup_steps} warmup steps")
        return scheduler
    
    def train_epoch(self) -> float:
        """
        Train for one epoch
        Following patterns from the Fine-tuning notebook
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        # Progress bar for training
        pbar = tqdm(self.train_dataloader, desc="Training")
        
        for batch in pbar:
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
            loss = self.criterion(logits, labels)
            
            # Backward pass (following notebook pattern)
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping (from notebook best practices)
            if self.config['advanced'].get('gradient_clipping'):
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config['advanced']['gradient_clipping']
                )
            
            self.optimizer.step()
            self.scheduler.step()
            
            # Track loss
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        return total_loss / num_batches
    
    def evaluate(self) -> Tuple[float, float, Dict]:
        """
        Evaluate model on validation set
        Following evaluation patterns from notebooks
        """
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            pbar = tqdm(self.val_dataloader, desc="Evaluating")
            
            for batch in pbar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
                loss = self.criterion(logits, labels)
                
                # Get predictions
                predictions = torch.argmax(logits, dim=1)
                
                # Collect for metrics
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                total_loss += loss.item()
        
        # Calculate metrics (following notebook patterns)
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='weighted'
        )
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'predictions': all_predictions,
            'labels': all_labels
        }
        
        avg_loss = total_loss / len(self.val_dataloader)
        
        return accuracy, avg_loss, metrics
    
    def train(self, num_epochs: Optional[int] = None) -> Dict:
        """
        Main training loop following notebook patterns
        """
        if num_epochs is None:
            num_epochs = self._get_device_config()['num_epochs']
        
        print(f"\nStarting training for {num_epochs} epochs...")
        print(f"Training samples: {len(self.train_dataset)}")
        print(f"Validation samples: {len(self.val_dataset)}")
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 50)
            
            # Train epoch
            train_loss = self.train_epoch()
            
            # Evaluate
            val_accuracy, val_loss, metrics = self.evaluate()
            
            # Track metrics
            self.train_losses.append(train_loss)
            self.val_accuracies.append(val_accuracy)
            self.val_losses.append(val_loss)
            
            # Save best model (following notebook pattern)
            if val_accuracy > self.best_val_accuracy:
                self.best_val_accuracy = val_accuracy
                self.best_model_state = self.model.state_dict().copy()\n                print(f"New best model! Accuracy: {val_accuracy:.4f}")
            
            # Print epoch summary
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Val Accuracy: {val_accuracy:.4f}")
            print(f"Val F1: {metrics['f1']:.4f}")
            
            # Early stopping check
            if self._should_early_stop():
                print("Early stopping triggered!")
                break
        
        training_time = time.time() - start_time
        print(f"\nTraining completed in {training_time:.2f} seconds")
        print(f"Best validation accuracy: {self.best_val_accuracy:.4f}")
        
        # Load best model
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)
        
        return {
            'train_losses': self.train_losses,
            'val_accuracies': self.val_accuracies,
            'val_losses': self.val_losses,
            'best_val_accuracy': self.best_val_accuracy,
            'training_time': training_time,
            'final_metrics': metrics
        }
    
    def _should_early_stop(self) -> bool:
        """Check if early stopping should be triggered"""
        early_stop_config = self.config['early_stopping']
        
        if not early_stop_config['enabled']:
            return False
        
        patience = early_stop_config['patience']
        min_delta = early_stop_config['min_delta']
        
        if len(self.val_accuracies) < patience:
            return False
        
        # Check if no improvement in last 'patience' epochs
        recent_accuracies = self.val_accuracies[-patience:]
        max_recent = max(recent_accuracies)
        
        return (self.best_val_accuracy - max_recent) > min_delta
    
    def plot_training_progress(self, save_path: str = None):
        """
        Plot training progress (following notebook visualization patterns)
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot losses
        epochs = range(1, len(self.train_losses) + 1)
        ax1.plot(epochs, self.train_losses, 'b-', label='Training Loss')
        ax1.plot(epochs, self.val_losses, 'r-', label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot accuracy
        ax2.plot(epochs, self.val_accuracies, 'g-', label='Validation Accuracy')
        ax2.axhline(y=self.best_val_accuracy, color='r', linestyle='--', 
                   label=f'Best Accuracy: {self.best_val_accuracy:.4f}')
        ax2.set_title('Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training plot saved to: {save_path}")
        
        plt.show()
    
    def plot_confusion_matrix(self, class_names: List[str], save_path: str = None):
        """Plot confusion matrix from last evaluation"""
        if not hasattr(self, 'val_accuracies') or not self.val_accuracies:
            print("No evaluation results available for confusion matrix")
            return
        
        # Get predictions from last evaluation
        _, _, metrics = self.evaluate()
        
        cm = confusion_matrix(metrics['labels'], metrics['predictions'])
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to: {save_path}")
        
        plt.show()
    
    def save_model(self, save_path: str):
        """Save the trained model"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_accuracy': self.best_val_accuracy,
            'training_config': self.config,
            'train_losses': self.train_losses,
            'val_accuracies': self.val_accuracies,
            'val_losses': self.val_losses
        }
        
        torch.save(checkpoint, save_path)
        print(f"Model saved to: {save_path}")
    
    def load_model(self, load_path: str):
        """Load a trained model"""
        checkpoint = torch.load(load_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_accuracy = checkpoint['best_val_accuracy']
        self.train_losses = checkpoint['train_losses']
        self.val_accuracies = checkpoint['val_accuracies']
        self.val_losses = checkpoint['val_losses']
        
        print(f"Model loaded from: {load_path}")
        print(f"Best validation accuracy: {self.best_val_accuracy:.4f}")


def create_datasets_from_csv(
    data_path: str,
    tokenizer,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[MedicalTextDataset, MedicalTextDataset]:
    """
    Create train/validation datasets from CSV file
    Following data loading patterns from notebooks
    """
    # Load data
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} samples from {data_path}")
    
    # Split data
    from sklearn.model_selection import train_test_split
    
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['text'].tolist(),
        df['label_id'].tolist(),
        test_size=test_size,
        random_state=random_state,
        stratify=df['label_id'].tolist()
    )
    
    # Create datasets
    train_dataset = MedicalTextDataset(train_texts, train_labels, tokenizer)
    val_dataset = MedicalTextDataset(val_texts, val_labels, tokenizer)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    return train_dataset, val_dataset


def main():
    """Example usage of the last-layer trainer"""
    from src.models.clinical_bert import create_clinical_bert_model
    
    # Create model
    model = create_clinical_bert_model(num_classes=10)
    
    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    
    # Create datasets (assuming data exists)
    data_path = "data/synthetic_training_data_10class.csv"
    if os.path.exists(data_path):
        train_dataset, val_dataset = create_datasets_from_csv(data_path, tokenizer)
        
        # Create trainer
        trainer = LastLayerTrainer(model, train_dataset, val_dataset)
        
        # Train model
        results = trainer.train()
        
        # Plot results
        trainer.plot_training_progress("results/training_progress.png")
        
        # Save model
        trainer.save_model("models/clinical_bert_10class.pt")
        
        print("Training completed successfully!")
    else:
        print(f"Data file not found: {data_path}")
        print("Please run data generation first.")


if __name__ == "__main__":
    main()
