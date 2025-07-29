#!/usr/bin/env python3
"""
Resume Training from Existing Checkpoints
Continue training medical BERT models from saved checkpoints
"""

import os
import sys
import torch
import argparse
from pathlib import Path

# Add src to path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.models.bio_bert import create_biobert_model
from src.models.clinical_bert import create_clinical_bert_model
from src.models.base_classifier import BaseMedicalBERTClassifier
from src.training.last_layer_trainer import LastLayerTrainer, create_datasets_from_csv
from transformers import AutoTokenizer
import yaml

def load_config(config_name: str = "classes") -> dict:
    """Load configuration from YAML file"""
    config_path = f"config/{config_name}.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def resume_training(model_name: str, target_epochs: int = 10, data_path: str = None):
    """Resume training from existing checkpoint"""
    
    model_path = f"models/{model_name}_10class.pt"
    
    if not os.path.exists(model_path):
        print(f"❌ Checkpoint not found: {model_path}")
        return None
    
    print(f"🔄 Resuming {model_name} training to {target_epochs} total epochs")
    print("=" * 60)
    
    try:
        # Load existing checkpoint
        print("Loading existing checkpoint...")
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        # Extract training history
        current_epoch = len(checkpoint.get('train_losses', []))
        best_val_accuracy = checkpoint.get('best_val_accuracy', 0.0)
        train_losses = checkpoint.get('train_losses', [])
        val_accuracies = checkpoint.get('val_accuracies', [])
        val_losses = checkpoint.get('val_losses', [])
        
        print(f"✅ Checkpoint loaded:")
        print(f"   Current epochs completed: {current_epoch}")
        print(f"   Best validation accuracy: {best_val_accuracy:.4f}")
        
        if current_epoch >= target_epochs:
            print(f"⚠️  Model already trained for {current_epoch} epochs (target: {target_epochs})")
            print("No additional training needed!")
            return checkpoint
        
        remaining_epochs = target_epochs - current_epoch
        print(f"🎯 Will train for {remaining_epochs} more epochs")
        
        # Create model architecture
        print("Creating model architecture...")
        if model_name == "bio_bert":
            model = create_biobert_model(num_classes=10)
            tokenizer_name = "dmis-lab/biobert-base-cased-v1.1"
        elif model_name == "clinical_bert":
            model = create_clinical_bert_model(num_classes=10)
            tokenizer_name = "emilyalsentzer/Bio_ClinicalBERT"
        elif model_name == "blue_bert":
            model = BaseMedicalBERTClassifier(
                model_name="bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12",
                num_classes=10
            )
            tokenizer_name = "bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12"
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Load model weights
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Model weights restored")
        
        # Create tokenizer and datasets
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        if data_path is None:
            data_path = "data/synthetic_training_data_10class.csv"
        
        train_dataset, val_dataset = create_datasets_from_csv(data_path, tokenizer)
        
        # Create trainer
        trainer = LastLayerTrainer(model, train_dataset, val_dataset)
        
        # Restore optimizer and scheduler states
        if 'optimizer_state_dict' in checkpoint:
            print("✅ Restoring optimizer state...")
            trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint:
            print("✅ Restoring scheduler state...")
            trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Restore trainer state
        trainer.best_val_accuracy = best_val_accuracy
        trainer.train_losses = train_losses.copy()
        trainer.val_accuracies = val_accuracies.copy() 
        trainer.val_losses = val_losses.copy()
        
        print(f"\\n🚀 Training {remaining_epochs} additional epochs...")
        print("=" * 60)
        
        # Train for remaining epochs only
        results = trainer.train(num_epochs=remaining_epochs)
        
        # Combine original and new training history
        final_results = {
            'train_losses': trainer.train_losses,
            'val_accuracies': trainer.val_accuracies,
            'val_losses': trainer.val_losses,
            'best_val_accuracy': trainer.best_val_accuracy,
            'total_epochs': len(trainer.train_losses),
            'training_time': results.get('training_time', 0)
        }
        
        # Save extended model
        extended_model_path = f"models/{model_name}_10class.pt"
        trainer.save_model(extended_model_path)
        
        # Save plots
        plot_path = f"results/{model_name}_extended_training_progress.png"
        trainer.plot_training_progress(plot_path)
        
        print(f"\\n✅ Extended training completed!")
        print(f"   Total epochs: {target_epochs}")
        print(f"   Best validation accuracy: {trainer.best_val_accuracy:.4f}")
        print(f"   Model saved: {extended_model_path}")
        
        return final_results
        
    except Exception as e:
        print(f"❌ Failed to resume training: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main function for resuming training"""
    parser = argparse.ArgumentParser(
        description="Resume Medical BERT training from existing checkpoints",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python resume_training_fixed.py --model bio_bert --epochs 10
  python resume_training_fixed.py --model clinical_bert --epochs 15
  python resume_training_fixed.py --all --epochs 10
        """
    )
    
    parser.add_argument(
        '--model',
        choices=['bio_bert', 'clinical_bert', 'blue_bert'],
        help='Which model to continue training'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=10,
        help='Total target epochs (default: 10)'
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='Resume training for all three models'
    )
    
    args = parser.parse_args()
    
    if not args.all and not args.model:
        print("❌ Please specify either --model or --all")
        return
    
    # Setup environment
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    
    print("🏥 Medical BERT Resume Training (FIXED)")
    print("=" * 70)
    
    if args.all:
        # Resume training for all models
        models = ['bio_bert', 'clinical_bert', 'blue_bert']
        results = {}
        
        for model_name in models:
            print(f"\\n{'='*70}")
            result = resume_training(model_name, args.epochs)
            if result:
                results[model_name] = result
        
        # Summary
        if results:
            print(f"\\n🎉 Resume Training Summary")
            print("=" * 70)
            for model_name, result in results.items():
                print(f"{model_name:15}: {result['best_val_accuracy']:.4f} accuracy ({result['total_epochs']} epochs)")
    else:
        # Resume training for single model
        result = resume_training(args.model, args.epochs)
        
        if result:
            print(f"\\n🎉 {args.model} resume training completed successfully!")
        else:
            print(f"\\n❌ {args.model} resume training failed")

if __name__ == "__main__":
    main()
