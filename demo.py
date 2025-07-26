#!/usr/bin/env python3
"""
Medical BERT Classifier - Quick Demo
Demonstrates the complete pipeline with minimal setup
"""

import os
import sys
import time
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Suppress warnings for cleaner demo output
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import warnings
warnings.filterwarnings('ignore')

def run_demo():
    """Run a quick demonstration of the medical BERT classifier"""
    
    print("🏥 Medical BERT Classifier - Quick Demo")
    print("=" * 60)
    print("This demo shows the complete pipeline in action:")
    print("1. Generate synthetic medical text data")
    print("2. Train Clinical-Bio-BERT with last-layer fine-tuning")
    print("3. Evaluate performance and show results")
    print()
    
    # Step 1: Data Generation
    print("📊 Step 1: Generating Synthetic Medical Data")
    print("-" * 40)
    
    try:
        from src.data.generate_10class import MedicalDataGenerator
        
        # Create smaller dataset for demo
        generator = MedicalDataGenerator()
        
        # Override config for demo (smaller dataset)
        generator.data_config['samples_per_class'] = 100  # Small for demo
        
        df = generator.generate_all_data()
        
        print(f"✅ Generated {len(df)} samples across 10 classes")
        print(f"Classes: {', '.join(df['label'].unique())}")
        
        # Show sample from new Sx class
        sx_sample = df[df['label'] == 'Sx'].iloc[0]
        print(f"\nNew Sx (Symptoms) class example:")
        print(f"'{sx_sample['text']}'")
        
        # Save demo data
        demo_data_path = "data/demo_data.csv"
        os.makedirs("data", exist_ok=True)
        df.to_csv(demo_data_path, index=False)
        
    except Exception as e:
        print(f"❌ Data generation failed: {e}")
        return False
    
    print()
    
    # Step 2: Model Training
    print("🧠 Step 2: Training Clinical-Bio-BERT")
    print("-" * 40)
    
    try:
        from src.models.clinical_bert import create_clinical_bert_model
        from src.training.last_layer_trainer import LastLayerTrainer, create_datasets_from_csv
        from transformers import AutoTokenizer
        
        # Create model
        print("Creating Clinical-Bio-BERT model...")
        model = create_clinical_bert_model(num_classes=10)
        
        # Create tokenizer
        tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Prepare datasets
        print("Preparing datasets...")
        train_dataset, val_dataset = create_datasets_from_csv(demo_data_path, tokenizer, test_size=0.3)
        
        # Create trainer
        print("Setting up trainer...")
        trainer = LastLayerTrainer(model, train_dataset, val_dataset)
        
        # Quick training (2 epochs for demo)
        print("Training model (2 epochs for demo)...")
        start_time = time.time()
        results = trainer.train(num_epochs=2)
        training_time = time.time() - start_time
        
        print(f"✅ Training completed in {training_time:.2f} seconds")
        print(f"Best validation accuracy: {results['best_val_accuracy']:.4f}")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print()
    
    # Step 3: Evaluation and Testing
    print("📈 Step 3: Evaluation and Testing")
    print("-" * 40)
    
    try:
        # Final evaluation
        accuracy, loss, metrics = trainer.evaluate()
        print(f"Final accuracy: {accuracy:.4f}")
        print(f"Final F1 score: {metrics['f1']:.4f}")
        
        # Test individual predictions
        test_samples = {
            'Rx': "Patient prescribed metformin 500mg twice daily for diabetes",
            'Sx': "Patient reports severe chest pain radiating to left arm",  # NEW!
            'Dx': "Diagnosed with type 2 diabetes mellitus",
            'Lx': "Hemoglobin A1c measured at 7.2%",
            'Vitals': "Blood pressure 140/90 mmHg, heart rate 85 bpm"
        }
        
        print(f"\nTesting individual predictions:")
        correct_predictions = 0
        
        for true_class, text in test_samples.items():
            # Tokenize
            encoding = tokenizer(text, truncation=True, padding='max_length', 
                               max_length=512, return_tensors='pt')
            
            # Predict
            model.eval()
            import torch
            with torch.no_grad():
                logits = model(input_ids=encoding['input_ids'], 
                              attention_mask=encoding['attention_mask'])
                predicted_class_id = torch.argmax(logits, dim=-1).item()
                confidence = torch.softmax(logits, dim=-1).max().item()
            
            # Get class names
            from src.data import get_class_names
            class_names = get_class_names()
            predicted_class = class_names[predicted_class_id]
            
            is_correct = predicted_class == true_class
            if is_correct:
                correct_predictions += 1
            
            status = "✅" if is_correct else "❌"
            print(f"{status} True: {true_class} | Predicted: {predicted_class} | Confidence: {confidence:.3f}")
        
        print(f"\nIndividual test accuracy: {correct_predictions}/{len(test_samples)} ({correct_predictions/len(test_samples)*100:.1f}%)")
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        return False
    
    # Summary
    print()
    print("🎉 Demo Summary")
    print("=" * 60)
    print(f"✅ Successfully demonstrated medical BERT classifier!")
    print(f"✅ Trained on {len(df)} synthetic medical text samples")
    print(f"✅ Achieved {accuracy:.4f} accuracy with last-layer training")
    print(f"✅ Training time: {training_time:.2f} seconds")
    print(f"✅ Model uses only {trainer.model.get_num_trainable_parameters():,} trainable parameters")
    
    total_params = sum(p.numel() for p in trainer.model.parameters())
    trainable_ratio = trainer.model.get_num_trainable_parameters() / total_params
    print(f"✅ Efficiency: Only {trainable_ratio:.2%} of parameters are trainable")
    
    print(f"\nKey Features Demonstrated:")
    print(f"- 🏥 Medical BERT specialization (Clinical-Bio-BERT)")
    print(f"- 🚀 Efficient last-layer training approach")
    print(f"- 📊 10-class medical taxonomy including new Sx class")
    print(f"- 💻 CPU-optimized training (works without GPU)")
    print(f"- ⚡ Fast training and inference")
    
    print(f"\nNext Steps:")
    print(f"- Run full pipeline: python main.py --step all")
    print(f"- Compare models: python main.py --step compare")
    print(f"- Try notebook: jupyter lab notebooks/medical_bert_training.ipynb")
    
    return True

def main():
    """Main demo function"""
    try:
        success = run_demo()
        if success:
            print(f"\n🌟 Demo completed successfully!")
        else:
            print(f"\n⚠️  Demo encountered issues. Please check the errors above.")
        return success
    except KeyboardInterrupt:
        print(f"\n⏹️  Demo interrupted by user")
        return False
    except Exception as e:
        print(f"\n💥 Demo crashed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
