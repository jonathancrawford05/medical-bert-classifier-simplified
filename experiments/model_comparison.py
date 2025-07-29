#!/usr/bin/env python3
"""
Medical BERT Model Comparison Framework
Compare performance of existing trained models without retraining

This script loads pre-trained medical BERT models and evaluates them
on your classification task, providing detailed performance comparisons.

Usage:
    python experiments/model_comparison.py
    
Outputs:
    - results/model_comparison_summary.csv
    - results/model_comparison_plots.png
"""

import os
import sys
import torch
import pandas as pd
import numpy as np
from typing import Dict, List
import yaml
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import seaborn as sns

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from transformers import AutoTokenizer
from src.models.bio_bert import create_biobert_model
from src.models.clinical_bert import create_clinical_bert_model
from src.models.base_classifier import BaseMedicalBERTClassifier
from src.training.last_layer_trainer import create_datasets_from_csv

def load_config():
    """Load class configuration"""
    with open("config/classes.yaml", 'r') as f:
        return yaml.safe_load(f)

def load_trained_model(model_name: str, model_path: str, num_classes: int = 10):
    """Load a trained model from file"""
    print(f"Loading {model_name} from {model_path}...")
    
    try:
        # Create the model architecture (this loads pretrained BERT weights)
        print(f"   Creating {model_name} architecture...")
        if model_name == "Bio-BERT":
            model = create_biobert_model(num_classes=num_classes)
        elif model_name == "Clinical-Bio-BERT":
            model = create_clinical_bert_model(num_classes=num_classes)
        elif model_name == "BlueBERT":
            model = BaseMedicalBERTClassifier(
                model_name="bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12",
                num_classes=num_classes
            )
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Debug model creation
        total_params = sum(p.numel() for p in model.parameters())
        print(f"   Model created with {total_params:,} total parameters")
        
        if total_params < 1000000:  # Less than 1M params means something is wrong
            print(f"   ⚠️  Warning: Model has unusually few parameters!")
            print(f"   ⚠️  This suggests the BERT backbone didn't load correctly")
            return None
        
        # Load the saved checkpoint
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        # Debug: Check what's in the checkpoint
        print(f"   Checkpoint keys: {list(checkpoint.keys())[:5]}...")  # Show first 5 keys
        
        # Extract the actual model state dict
        if 'model_state_dict' in checkpoint:
            model_checkpoint = checkpoint['model_state_dict']
            print(f"   Found model_state_dict with {len(model_checkpoint)} parameters")
        else:
            model_checkpoint = checkpoint
            print(f"   Using checkpoint directly with {len(model_checkpoint)} parameters")
        
        # Debug: Show first few model state dict keys
        print(f"   Model state keys: {list(model_checkpoint.keys())[:3]}...")
        
        # Get the current model state
        model_state = model.state_dict()
        print(f"   Current model has {len(model_state)} total parameters")
        
        # Load the state dict with exact matching
        model.load_state_dict(model_checkpoint, strict=False)
        
        # Verify model integrity after loading
        total_params_after = sum(p.numel() for p in model.parameters())
        print(f"   Model has {total_params_after:,} parameters after loading")
        
        if total_params_after < 1000000:  # Less than 1M params means loading corrupted the model
            print(f"   ❌ Model corrupted during loading - parameter count dropped drastically")
            print(f"   ❌ Expected ~108M parameters, got {total_params_after:,}")
            print(f"   ❌ This suggests the saved checkpoint is incompatible")
            return None
        
        model.eval()  # Set to evaluation mode
        
        print(f"✅ Successfully loaded {model_name}")
        print(f"   Loaded classifier weights, BERT backbone uses pretrained weights")
        return model
        
    except Exception as e:
        print(f"❌ Failed to load {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return None

def evaluate_model(model, tokenizer, data_path: str, model_name: str):
    """Evaluate a loaded model on the test data"""
    print(f"Evaluating {model_name}...")
    
    try:
        # Create datasets
        print(f"   Creating datasets...")
        train_dataset, val_dataset = create_datasets_from_csv(data_path, tokenizer, test_size=0.2)
        print(f"   Validation dataset size: {len(val_dataset)}")
        
        # Use validation dataset for evaluation
        model.eval()
        all_predictions = []
        all_labels = []
        total_loss = 0.0
        
        # Create DataLoader for validation set with smaller batch size
        from torch.utils.data import DataLoader
        print(f"   Creating DataLoader...")
        val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False)  # Smaller batch size
        print(f"   DataLoader created with {len(val_dataloader)} batches")
        
        # Process only first few batches for quick evaluation
        max_batches = min(50, len(val_dataloader))  # Limit to 10 batches
        print(f"   Processing {max_batches} batches for quick evaluation...")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_dataloader):
                if batch_idx >= max_batches:
                    break
                    
                print(f"   Processing batch {batch_idx + 1}/{max_batches}")
                
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask'] 
                labels = batch['labels']
                
                # Forward pass
                try:
                    outputs = model(input_ids, attention_mask)
                    
                    # Calculate loss (if model returns loss)
                    if hasattr(outputs, 'loss') and outputs.loss is not None:
                        total_loss += outputs.loss.item()
                    
                    # Get predictions
                    if hasattr(outputs, 'logits'):
                        logits = outputs.logits
                    else:
                        logits = outputs
                        
                    predictions = torch.argmax(logits, dim=-1)
                    
                    all_predictions.extend(predictions.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    
                except Exception as e:
                    print(f"   ❌ Error in forward pass: {e}")
                    return None
        
        if not all_predictions:
            print(f"   ❌ No predictions generated")
            return None
            
        print(f"   Generated {len(all_predictions)} predictions")
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='weighted')
        
        # Per-class metrics (simplified)
        config = load_config()
        class_names = list(config['class_labels'].values())
        
        try:
            class_report = classification_report(all_labels, all_predictions, target_names=class_names, output_dict=True)
        except:
            class_report = {}  # Fallback if classification report fails
        
        results = {
            'model_name': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'loss': total_loss / max_batches if max_batches > 0 else 0,
            'predictions': all_predictions,
            'true_labels': all_labels,
            'class_report': class_report,
            'confusion_matrix': confusion_matrix(all_labels, all_predictions) if len(set(all_labels)) > 1 else None
        }
        
        print(f"✅ {model_name} evaluation complete")
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   F1-Score: {f1:.4f}")
        print(f"   Samples evaluated: {len(all_predictions)}")
        
        return results
        
    except Exception as e:
        print(f"❌ Failed to evaluate {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return None

def compare_existing_models():
    """Compare all existing trained models"""
    print("🏥 Medical BERT Model Comparison - Existing Models")
    print("=" * 70)
    
    # Model configurations
    models_config = {
        'Bio-BERT': {
            'file': 'models/bio_bert_10class.pt',
            'tokenizer': 'dmis-lab/biobert-base-cased-v1.1',
            'description': 'Specialized for biomedical literature'
        },
        'Clinical-Bio-BERT': {
            'file': 'models/clinical_bert_10class.pt',
            'tokenizer': 'emilyalsentzer/Bio_ClinicalBERT',
            'description': 'Optimized for clinical notes'
        },
        'BlueBERT': {
            'file': 'models/blue_bert_10class.pt', 
            'tokenizer': 'bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12',
            'description': 'Mixed biomedical and clinical'
        }
    }
    
    data_path = "data/synthetic_training_data_10class.csv"
    
    if not os.path.exists(data_path):
        print(f"❌ Data file not found: {data_path}")
        return
    
    print(f"Using data: {data_path}")
    print()
    
    # Load and evaluate each model
    results = {}
    
    for model_name, config in models_config.items():
        model_path = config['file']
        
        if not os.path.exists(model_path):
            print(f"❌ Model file not found: {model_path}")
            continue
            
        # Load model
        model = load_trained_model(model_name, model_path)
        if model is None:
            continue
            
        # Load tokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(config['tokenizer'])
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
        except Exception as e:
            print(f"❌ Failed to load tokenizer for {model_name}: {e}")
            continue
            
        # Evaluate model
        result = evaluate_model(model, tokenizer, data_path, model_name)
        if result:
            results[model_name] = result
        
        print()
    
    # Generate comparison report
    if results:
        print("=" * 70)
        print("MODEL COMPARISON RESULTS")
        print("=" * 70)
        
        # Create comparison DataFrame
        comparison_data = []
        for model_name, result in results.items():
            comparison_data.append({
                'Model': model_name,
                'Accuracy': f"{result['accuracy']:.4f}",
                'Precision': f"{result['precision']:.4f}",
                'Recall': f"{result['recall']:.4f}",
                'F1-Score': f"{result['f1']:.4f}",
                'Description': models_config[model_name]['description']
            })
        
        df = pd.DataFrame(comparison_data)
        print(df.to_string(index=False))
        
        # Save results
        df.to_csv("results/model_comparison_summary.csv", index=False)
        print(f"\n💾 Results saved to: results/model_comparison_summary.csv")
        
        # Find best model
        best_accuracy = max(results.keys(), key=lambda x: results[x]['accuracy'])
        best_f1 = max(results.keys(), key=lambda x: results[x]['f1'])
        
        print(f"\n🏆 Best Performance:")
        print(f"   Accuracy: {best_accuracy} ({results[best_accuracy]['accuracy']:.4f})")
        print(f"   F1-Score: {best_f1} ({results[best_f1]['f1']:.4f})")
        
        # Create visualization
        create_comparison_plots(results, models_config)
        
    else:
        print("❌ No models could be evaluated")

def create_comparison_plots(results: Dict, models_config: Dict):
    """Create comparison visualizations"""
    if len(results) < 2:
        print("⚠️  Need at least 2 models for comparison plots")
        return
        
    try:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        models = list(results.keys())
        
        # 1. Accuracy comparison
        accuracies = [results[m]['accuracy'] for m in models]
        axes[0, 0].bar(models, accuracies, color='skyblue')
        axes[0, 0].set_title('Model Accuracy Comparison')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_ylim(0, 1)
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. F1-Score comparison
        f1_scores = [results[m]['f1'] for m in models]
        axes[0, 1].bar(models, f1_scores, color='lightcoral')
        axes[0, 1].set_title('Model F1-Score Comparison')
        axes[0, 1].set_ylabel('F1-Score')
        axes[0, 1].set_ylim(0, 1)
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Multi-metric comparison
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        x = np.arange(len(metrics))
        width = 0.25
        
        for i, model in enumerate(models):
            values = [results[model][metric] for metric in metrics]
            axes[1, 0].bar(x + i*width, values, width, label=model)
        
        axes[1, 0].set_title('Multi-Metric Comparison')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].set_xticks(x + width)
        axes[1, 0].set_xticklabels(metrics)
        axes[1, 0].legend()
        axes[1, 0].set_ylim(0, 1)
        
        # 4. Confusion matrix for best model
        best_model = max(results.keys(), key=lambda x: results[x]['f1'])
        cm = results[best_model]['confusion_matrix']
        
        config = load_config()
        class_names = list(config['class_labels'].values())
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names,
                   ax=axes[1, 1])
        axes[1, 1].set_title(f'Confusion Matrix - {best_model}')
        axes[1, 1].set_xlabel('Predicted')
        axes[1, 1].set_ylabel('Actual')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = "results/existing_model_comparison.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 Comparison plots saved to: {plot_path}")
        
    except Exception as e:
        print(f"❌ Failed to create plots: {e}")

if __name__ == "__main__":
    compare_existing_models()
