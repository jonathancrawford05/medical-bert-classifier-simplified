"""
Medical BERT Model Comparison Framework
Comprehensive evaluation and comparison of medical BERT models for text classification
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import time
import os
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    confusion_matrix, classification_report
)
from transformers import AutoTokenizer

# Import our model implementations
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.bio_bert import create_biobert_model, BioBERTConfig
from src.models.clinical_bert import create_clinical_bert_model, ClinicalBERTConfig
from src.models.base_classifier import BaseMedicalBERTClassifier
from src.training.last_layer_trainer import LastLayerTrainer, create_datasets_from_csv


class MedicalBERTComparison:
    """
    Framework for comparing multiple medical BERT models
    """
    
    def __init__(self, config_path: str = "config/classes.yaml"):
        """Initialize comparison framework"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.num_classes = self.config['num_classes']
        self.class_labels = self.config['class_labels']
        self.class_names = list(self.class_labels.values())
        
        # Model configurations
        self.models_config = {
            'Bio-BERT': {
                'model_name': 'dmis-lab/biobert-base-cased-v1.1',
                'create_func': create_biobert_model,
                'config_class': BioBERTConfig,
                'description': 'BioBERT pre-trained on PubMed abstracts and PMC full-text articles',
                'domain': 'biomedical_literature',
                'strengths': ['Biomedical terminology', 'Research literature', 'Drug names'],
                'use_cases': ['Literature mining', 'Drug discovery', 'Biomedical NER']
            },
            'Clinical-Bio-BERT': {
                'model_name': 'emilyalsentzer/Bio_ClinicalBERT',
                'create_func': create_clinical_bert_model,
                'config_class': ClinicalBERTConfig,
                'description': 'BioBERT further pre-trained on clinical notes from MIMIC-III',
                'domain': 'clinical_notes',
                'strengths': ['Clinical terminology', 'EHR text', 'Medical abbreviations'],
                'use_cases': ['Clinical note analysis', 'EHR processing', 'Clinical decision support']
            },
            'BlueBERT': {
                'model_name': 'bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12',
                'create_func': self._create_bluebert_model,
                'config_class': None,
                'description': 'BERT pre-trained on both PubMed abstracts and MIMIC-III clinical notes',
                'domain': 'biomedical_clinical_mixed',
                'strengths': ['Mixed domain', 'General medical text', 'Versatile'],
                'use_cases': ['General medical NLP', 'Cross-domain tasks', 'Medical chatbots']
            }
        }
        
        self.results = {}
        self.comparison_metrics = {}
        
    def _create_bluebert_model(self, num_classes: int = 10) -> BaseMedicalBERTClassifier:
        """Create BlueBERT model using base implementation"""
        return BaseMedicalBERTClassifier(
            model_name="bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12",
            num_classes=num_classes,
            max_length=512,
            dropout_rate=0.3,
            freeze_backbone=True
        )
    
    def train_single_model(
        self, 
        model_name: str,
        data_path: str,
        num_epochs: int = 3,
        test_size: float = 0.2
    ) -> Dict:
        """
        Train a single medical BERT model and collect results
        """
        print(f"\n{'='*60}")
        print(f"Training {model_name}")
        print(f"{'='*60}")
        
        config = self.models_config[model_name]
        
        try:
            # Create model
            print("Creating model...")
            model = config['create_func'](num_classes=self.num_classes)
            
            # Create tokenizer
            tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Prepare data
            print("Preparing datasets...")
            train_dataset, val_dataset = create_datasets_from_csv(
                data_path, tokenizer, test_size=test_size
            )
            
            # Create trainer
            print("Setting up trainer...")
            trainer = LastLayerTrainer(model, train_dataset, val_dataset)
            
            # Train model
            print(f"Training for {num_epochs} epochs...")
            start_time = time.time()
            training_results = trainer.train(num_epochs=num_epochs)
            training_time = time.time() - start_time
            
            # Collect final evaluation metrics
            accuracy, loss, metrics = trainer.evaluate()
            
            # Calculate detailed metrics
            precision, recall, f1, _ = precision_recall_fscore_support(
                metrics['labels'], metrics['predictions'], average='weighted'
            )
            
            # Per-class metrics
            class_report = classification_report(
                metrics['labels'], 
                metrics['predictions'],
                target_names=self.class_names,
                output_dict=True
            )
            
            # Compile results
            result = {
                'model_name': model_name,
                'model_config': config,
                'trainer': trainer,
                'training_results': training_results,
                'training_time': training_time,
                'final_accuracy': accuracy,
                'final_loss': loss,
                'final_precision': precision,
                'final_recall': recall,
                'final_f1': f1,
                'class_report': class_report,
                'confusion_matrix': confusion_matrix(metrics['labels'], metrics['predictions']),
                'trainable_params': model.get_num_trainable_parameters(),
                'total_params': sum(p.numel() for p in model.parameters()),
                'predictions': metrics['predictions'],
                'true_labels': metrics['labels']
            }
            
            # Success summary
            print(f"✅ {model_name} training completed!")
            print(f"   Training time: {training_time:.2f} seconds")
            print(f"   Best accuracy: {training_results['best_val_accuracy']:.4f}")
            print(f"   Final F1: {f1:.4f}")
            print(f"   Trainable parameters: {result['trainable_params']:,}")
            
            return result
            
        except Exception as e:
            print(f"❌ {model_name} training failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def compare_all_models(
        self, 
        data_path: str,
        models_to_compare: Optional[List[str]] = None,
        num_epochs: int = 3
    ) -> Dict:
        """
        Train and compare all specified medical BERT models
        """
        if models_to_compare is None:
            models_to_compare = list(self.models_config.keys())
        
        print("🏥 Medical BERT Model Comparison Framework")
        print("=" * 70)
        print(f"Comparing models: {', '.join(models_to_compare)}")
        print(f"Training epochs: {num_epochs}")
        print(f"Data source: {data_path}")
        
        # Train each model
        for model_name in models_to_compare:
            if model_name in self.models_config:
                result = self.train_single_model(model_name, data_path, num_epochs)
                if result:
                    self.results[model_name] = result
            else:
                print(f"⚠️  Unknown model: {model_name}")
        
        # Generate comparison metrics
        if self.results:
            self._generate_comparison_metrics()
            self._print_comparison_summary()
        else:
            print("❌ No models were successfully trained for comparison")
        
        return self.results
    
    def _generate_comparison_metrics(self):
        """Generate comparison metrics from all trained models"""
        self.comparison_metrics = {
            'accuracy': {},
            'f1_score': {},
            'training_time': {},
            'trainable_params': {},
            'model_size_ratio': {},
            'per_class_f1': {}
        }
        
        for model_name, result in self.results.items():
            self.comparison_metrics['accuracy'][model_name] = result['final_accuracy']
            self.comparison_metrics['f1_score'][model_name] = result['final_f1']
            self.comparison_metrics['training_time'][model_name] = result['training_time']
            self.comparison_metrics['trainable_params'][model_name] = result['trainable_params']
            
            # Calculate trainable parameter ratio
            ratio = result['trainable_params'] / result['total_params']
            self.comparison_metrics['model_size_ratio'][model_name] = ratio
            
            # Per-class F1 scores
            class_f1 = {}
            for class_name in self.class_names:
                class_f1[class_name] = result['class_report'][class_name]['f1-score']
            self.comparison_metrics['per_class_f1'][model_name] = class_f1
    
    def _print_comparison_summary(self):
        """Print a formatted comparison summary"""
        print(f"\n{'='*70}")
        print("MODEL COMPARISON SUMMARY")
        print(f"{'='*70}")
        
        # Create comparison DataFrame
        comparison_data = []
        for model_name, result in self.results.items():
            comparison_data.append({
                'Model': model_name,
                'Accuracy': f"{result['final_accuracy']:.4f}",
                'F1-Score': f"{result['final_f1']:.4f}",
                'Training Time (s)': f"{result['training_time']:.2f}",
                'Trainable Params': f"{result['trainable_params']:,}",
                'Param Ratio': f"{self.comparison_metrics['model_size_ratio'][model_name]:.3%}",
                'Domain': result['model_config']['domain']
            })
        
        df = pd.DataFrame(comparison_data)
        print(df.to_string(index=False))
        
        # Best model analysis
        best_accuracy_model = max(self.results.keys(), 
                                 key=lambda x: self.results[x]['final_accuracy'])
        best_f1_model = max(self.results.keys(), 
                           key=lambda x: self.results[x]['final_f1'])
        fastest_model = min(self.results.keys(), 
                           key=lambda x: self.results[x]['training_time'])
        
        print(f"\n🏆 Best Performance:")
        print(f"   Accuracy: {best_accuracy_model} ({self.results[best_accuracy_model]['final_accuracy']:.4f})")
        print(f"   F1-Score: {best_f1_model} ({self.results[best_f1_model]['final_f1']:.4f})")
        print(f"   Speed: {fastest_model} ({self.results[fastest_model]['training_time']:.2f}s)")
    
    def plot_comparison_results(self, save_path: str = None):
        """Create comprehensive comparison visualizations"""
        if not self.results:
            print("No results available for plotting")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        models = list(self.results.keys())
        
        # 1. Accuracy comparison
        accuracies = [self.results[m]['final_accuracy'] for m in models]
        axes[0, 0].bar(models, accuracies, color='skyblue')
        axes[0, 0].set_title('Model Accuracy Comparison')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_ylim(0, 1)
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. F1-Score comparison
        f1_scores = [self.results[m]['final_f1'] for m in models]
        axes[0, 1].bar(models, f1_scores, color='lightcoral')
        axes[0, 1].set_title('Model F1-Score Comparison')
        axes[0, 1].set_ylabel('F1-Score')
        axes[0, 1].set_ylim(0, 1)
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Training time comparison
        times = [self.results[m]['training_time'] for m in models]
        axes[0, 2].bar(models, times, color='lightgreen')
        axes[0, 2].set_title('Training Time Comparison')
        axes[0, 2].set_ylabel('Training Time (seconds)')
        axes[0, 2].tick_params(axis='x', rotation=45)
        
        # 4. Per-class F1 heatmap
        class_f1_data = []
        for model in models:
            class_f1_data.append([
                self.comparison_metrics['per_class_f1'][model][class_name] 
                for class_name in self.class_names
            ])
        
        im = axes[1, 0].imshow(class_f1_data, cmap='Blues', aspect='auto')
        axes[1, 0].set_title('Per-Class F1-Score Heatmap')
        axes[1, 0].set_yticks(range(len(models)))
        axes[1, 0].set_yticklabels(models)
        axes[1, 0].set_xticks(range(len(self.class_names)))
        axes[1, 0].set_xticklabels(self.class_names, rotation=45)
        plt.colorbar(im, ax=axes[1, 0])
        
        # 5. Trainable parameters comparison
        params = [self.results[m]['trainable_params'] for m in models]
        axes[1, 1].bar(models, params, color='orange')
        axes[1, 1].set_title('Trainable Parameters')
        axes[1, 1].set_ylabel('Number of Parameters')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        # 6. Accuracy vs Training Time scatter
        axes[1, 2].scatter(times, accuracies, s=100, alpha=0.7)
        for i, model in enumerate(models):
            axes[1, 2].annotate(model, (times[i], accuracies[i]), 
                               xytext=(5, 5), textcoords='offset points')
        axes[1, 2].set_xlabel('Training Time (seconds)')
        axes[1, 2].set_ylabel('Accuracy')
        axes[1, 2].set_title('Accuracy vs Training Time')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Comparison plots saved to: {save_path}")
        
        plt.show()
    
    def analyze_class_performance(self, class_name: str):
        """Analyze performance on a specific class across all models"""
        if not self.results:
            print("No results available for analysis")
            return
        
        print(f"\n📊 Class Performance Analysis: {class_name}")
        print("=" * 50)
        
        class_metrics = []
        for model_name, result in self.results.items():
            if class_name in result['class_report']:
                metrics = result['class_report'][class_name]
                class_metrics.append({
                    'Model': model_name,
                    'Precision': f"{metrics['precision']:.4f}",
                    'Recall': f"{metrics['recall']:.4f}",
                    'F1-Score': f"{metrics['f1-score']:.4f}",
                    'Support': metrics['support']
                })
        
        if class_metrics:
            df = pd.DataFrame(class_metrics)
            print(df.to_string(index=False))
            
            # Best performing model for this class
            best_model = max(class_metrics, key=lambda x: float(x['F1-Score']))
            print(f"\n🏆 Best model for {class_name}: {best_model['Model']}")
            print(f"   F1-Score: {best_model['F1-Score']}")
        else:
            print(f"No results found for class: {class_name}")
    
    def save_results(self, output_dir: str = "results"):
        """Save comparison results to files"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save summary CSV
        summary_data = []
        for model_name, result in self.results.items():
            summary_data.append({
                'Model': model_name,
                'Domain': result['model_config']['domain'],
                'Description': result['model_config']['description'],
                'Accuracy': result['final_accuracy'],
                'F1_Score': result['final_f1'],
                'Precision': result['final_precision'],
                'Recall': result['final_recall'],
                'Training_Time_Seconds': result['training_time'],
                'Trainable_Parameters': result['trainable_params'],
                'Total_Parameters': result['total_params'],
                'Parameter_Ratio': result['trainable_params'] / result['total_params']
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_path = os.path.join(output_dir, "model_comparison_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        
        # Save detailed per-class results
        for model_name, result in self.results.items():
            class_df = pd.DataFrame(result['class_report']).transpose()
            class_path = os.path.join(output_dir, f"{model_name}_class_report.csv")
            class_df.to_csv(class_path)
        
        print(f"✅ Results saved to: {output_dir}")


def main():
    """Example usage of the comparison framework"""
    # Initialize comparison framework
    comparator = MedicalBERTComparison()
    
    # Compare all models
    data_path = "../data/synthetic_training_data_10class.csv"
    if os.path.exists(data_path):
        results = comparator.compare_all_models(
            data_path=data_path,
            models_to_compare=['Bio-BERT', 'Clinical-Bio-BERT'],  # Start with these two
            num_epochs=2  # Quick comparison
        )
        
        if results:
            # Generate visualizations
            comparator.plot_comparison_results("results/model_comparison.png")
            
            # Analyze specific classes
            comparator.analyze_class_performance('Sx')  # New symptoms class
            comparator.analyze_class_performance('Dx')  # Diagnoses
            
            # Save results
            comparator.save_results()
        
    else:
        print(f"Data file not found: {data_path}")
        print("Please run data generation first.")


if __name__ == "__main__":
    main()
