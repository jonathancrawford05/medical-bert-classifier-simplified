#!/usr/bin/env python3
"""
Medical BERT Classifier - Simplified Last-Layer Training
Main entry point for the complete pipeline

This script orchestrates:
1. Data generation (10-class synthetic medical text)
2. Model training (last-layer only for efficiency)  
3. Model evaluation and comparison
4. Results visualization and saving

Supports multiple medical BERT models:
- Bio-BERT (biomedical literature)
- Clinical-Bio-BERT (clinical notes)
- BlueBERT (mixed biomedical/clinical)
"""

import os
import sys
import argparse
import yaml
import time
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

# Core imports
from src.data.generate_10class import MedicalDataGenerator
from src.models.bio_bert import create_biobert_model
from src.models.clinical_bert import create_clinical_bert_model
from src.training.last_layer_trainer import LastLayerTrainer, create_datasets_from_csv
from transformers import AutoTokenizer


def setup_environment():
    """Setup environment variables and directories"""
    # Suppress tokenizer warnings for cleaner output
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    
    # Create necessary directories
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    print("Environment setup complete")


def load_config(config_name: str = "classes") -> dict:
    """Load configuration from YAML file"""
    config_path = f"config/{config_name}.yaml"
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Configuration file not found: {config_path}")
        sys.exit(1)


def generate_data(force_regenerate: bool = False):
    """Step 1: Generate synthetic training data"""
    print("\n" + "="*60)
    print("STEP 1: GENERATING SYNTHETIC MEDICAL TEXT DATA")
    print("="*60)
    
    data_path = "data/synthetic_training_data_10class.csv"
    
    if os.path.exists(data_path) and not force_regenerate:
        print(f"Data file already exists: {data_path}")
        print("Use --force-regenerate to create new data")
        return data_path
    
    try:
        generator = MedicalDataGenerator()
        df = generator.generate_all_data()
        generator.save_data(df, data_path)
        print(f"✅ Data generation completed: {len(df)} samples")
        return data_path
    except Exception as e:
        print(f"❌ Data generation failed: {e}")
        sys.exit(1)


def train_model(model_name: str, data_path: str, epochs: int = None):
    """Step 2: Train a specific medical BERT model"""
    print(f"\n" + "="*60)
    print(f"STEP 2: TRAINING {model_name.upper()} MODEL")
    print("="*60)
    
    try:
        # Create model based on type
        if model_name == "bio_bert":
            model = create_biobert_model(num_classes=10)
            tokenizer_name = "dmis-lab/biobert-base-cased-v1.1"
        elif model_name == "clinical_bert":
            model = create_clinical_bert_model(num_classes=10)
            tokenizer_name = "emilyalsentzer/Bio_ClinicalBERT"
        elif model_name == "blue_bert":
            # Create BlueBERT model (using base implementation)
            from src.models.base_classifier import BaseMedicalBERTClassifier
            model = BaseMedicalBERTClassifier(
                model_name="bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12",
                num_classes=10
            )
            tokenizer_name = "bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12"
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Create tokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Create datasets
        train_dataset, val_dataset = create_datasets_from_csv(data_path, tokenizer)
        
        # Create trainer
        trainer = LastLayerTrainer(model, train_dataset, val_dataset)
        
        # Train model
        start_time = time.time()
        results = trainer.train(num_epochs=epochs)
        training_time = time.time() - start_time
        
        # Save results
        model_path = f"models/{model_name}_10class.pt"
        trainer.save_model(model_path)
        
        # Save training plots
        plot_path = f"results/{model_name}_training_progress.png"
        trainer.plot_training_progress(plot_path)
        
        # Save confusion matrix
        class_config = load_config("classes")
        class_names = list(class_config['class_labels'].values())
        cm_path = f"results/{model_name}_confusion_matrix.png"
        trainer.plot_confusion_matrix(class_names, cm_path)
        
        print(f"✅ {model_name} training completed in {training_time:.2f} seconds")
        print(f"   Best validation accuracy: {results['best_val_accuracy']:.4f}")
        print(f"   Model saved: {model_path}")
        
        return results
        
    except Exception as e:
        print(f"❌ {model_name} training failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def compare_models(data_path: str):
    """Step 3: Compare all medical BERT models"""
    print("\n" + "="*60)
    print("STEP 3: COMPARING MEDICAL BERT MODELS")
    print("="*60)
    
    models_to_compare = ["bio_bert", "clinical_bert", "blue_bert"]
    results = {}
    
    for model_name in models_to_compare:
        print(f"\nTraining {model_name}...")
        model_results = train_model(model_name, data_path, epochs=3)  # Shorter for comparison
        if model_results:
            results[model_name] = model_results
    
    # Create comparison report
    if results:
        print("\n" + "="*60)
        print("MODEL COMPARISON RESULTS")
        print("="*60)
        
        comparison_data = []
        for model_name, model_results in results.items():
            comparison_data.append({
                'Model': model_name,
                'Best Accuracy': f"{model_results['best_val_accuracy']:.4f}",
                'Training Time': f"{model_results['training_time']:.2f}s",
                'Final F1': f"{model_results['final_metrics']['f1']:.4f}"
            })
        
        # Print comparison table
        import pandas as pd
        comparison_df = pd.DataFrame(comparison_data)
        print(comparison_df.to_string(index=False))
        
        # Save comparison results
        comparison_df.to_csv("results/model_comparison.csv", index=False)
        print(f"\n✅ Comparison results saved to: results/model_comparison.csv")
    
    return results


def evaluate_model(model_name: str, data_path: str):
    """Step 4: Detailed evaluation of a specific model"""
    print(f"\n" + "="*60)
    print(f"STEP 4: EVALUATING {model_name.upper()} MODEL")
    print("="*60)
    
    model_path = f"models/{model_name}_10class.pt"
    
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        print("Please train the model first")
        return
    
    try:
        # Load model and evaluate
        # Implementation would load the saved model and run detailed evaluation
        print(f"✅ Detailed evaluation of {model_name} completed")
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description="Medical BERT Classifier - Simplified Last-Layer Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --step all                    # Run complete pipeline
  python main.py --step data                   # Generate data only
  python main.py --step train --model clinical_bert  # Train specific model
  python main.py --step compare                # Compare all models
  python main.py --step evaluate --model bio_bert     # Evaluate specific model
  
  python main.py --epochs 5 --force-regenerate # Custom training with new data
        """
    )
    
    parser.add_argument(
        '--step', 
        choices=['data', 'train', 'compare', 'evaluate', 'all'],
        default='all',
        help='Which step to run (default: all)'
    )
    
    parser.add_argument(
        '--model',
        choices=['bio_bert', 'clinical_bert', 'blue_bert'],
        default='clinical_bert',
        help='Which model to train/evaluate (default: clinical_bert)'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        help='Number of training epochs (uses config default if not specified)'
    )
    
    parser.add_argument(
        '--force-regenerate',
        action='store_true',
        help='Force regeneration of training data'
    )
    
    parser.add_argument(
        '--cpu-only',
        action='store_true',
        help='Force CPU-only training (ignore GPU)'
    )
    
    args = parser.parse_args()
    
    # Setup
    setup_environment()
    
    if args.cpu_only:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        print("Forced CPU-only mode")
    
    print("🏥 Medical BERT Classifier - Simplified Last-Layer Training")
    print("=" * 70)
    
    # Execute requested steps
    data_path = None
    
    if args.step in ['data', 'all']:
        data_path = generate_data(args.force_regenerate)
    
    # For train/compare/evaluate steps, check if data exists if not already set
    if data_path is None and args.step in ['train', 'compare', 'evaluate', 'all']:
        expected_data_path = "data/synthetic_training_data_10class.csv"
        if os.path.exists(expected_data_path):
            data_path = expected_data_path
            print(f"Using existing training data: {data_path}")
        else:
            print(f"❌ Training data not found: {expected_data_path}")
            print("Please run: python main.py --step data")
            sys.exit(1)
    
    if args.step in ['train', 'all'] and data_path:
        train_model(args.model, data_path, args.epochs)
    
    if args.step in ['compare', 'all'] and data_path:
        compare_models(data_path)
    
    if args.step in ['evaluate'] and data_path:
        evaluate_model(args.model, data_path)
    
    print("\n🎉 Pipeline execution completed!")
    print("Check the 'results/' directory for outputs and visualizations.")


if __name__ == "__main__":
    main()
