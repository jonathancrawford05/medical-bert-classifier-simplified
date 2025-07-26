#!/usr/bin/env python3
"""
Quick Setup Test for Medical BERT Classifier
Tests basic functionality and dependencies
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all main modules can be imported"""
    print("Testing imports...")
    
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
    except ImportError as e:
        print(f"❌ PyTorch import failed: {e}")
        return False
    
    try:
        import transformers
        print(f"✅ Transformers {transformers.__version__}")
    except ImportError as e:
        print(f"❌ Transformers import failed: {e}")
        return False
    
    try:
        import pandas as pd
        print(f"✅ Pandas {pd.__version__}")
    except ImportError as e:
        print(f"❌ Pandas import failed: {e}")
        return False
    
    try:
        import yaml
        print("✅ PyYAML")
    except ImportError as e:
        print(f"❌ PyYAML import failed: {e}")
        return False
        
    try:
        import sklearn
        print(f"✅ Scikit-learn {sklearn.__version__}")
    except ImportError as e:
        print(f"❌ Scikit-learn import failed: {e}")
        return False
    
    return True

def test_config_files():
    """Test that configuration files exist and are readable"""
    print("\nTesting configuration files...")
    
    config_files = [
        "config/classes.yaml",
        "config/models.yaml", 
        "config/training.yaml"
    ]
    
    for config_file in config_files:
        if os.path.exists(config_file):
            try:
                import yaml
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
                print(f"✅ {config_file}")
            except Exception as e:
                print(f"❌ {config_file} - Error reading: {e}")
                return False
        else:
            print(f"❌ {config_file} - File not found")
            return False
    
    return True

def test_module_imports():
    """Test that our custom modules can be imported"""
    print("\nTesting custom module imports...")
    
    try:
        from src.models.bio_bert import create_biobert_model
        print("✅ Bio-BERT module")
    except ImportError as e:
        print(f"❌ Bio-BERT import failed: {e}")
        return False
    
    try:
        from src.models.clinical_bert import create_clinical_bert_model
        print("✅ Clinical-BERT module")
    except ImportError as e:
        print(f"❌ Clinical-BERT import failed: {e}")
        return False
    
    try:
        from src.data.generate_10class import MedicalDataGenerator
        print("✅ Data generation module")
    except ImportError as e:
        print(f"❌ Data generation import failed: {e}")
        return False
    
    try:
        from src.training.last_layer_trainer import LastLayerTrainer
        print("✅ Training module")
    except ImportError as e:
        print(f"❌ Training import failed: {e}")
        return False
    
    return True

def test_model_creation():
    """Test that models can be created"""
    print("\nTesting model creation...")
    
    try:
        from src.models.clinical_bert import create_clinical_bert_model
        model = create_clinical_bert_model(num_classes=10)
        trainable_params = model.get_num_trainable_parameters()
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✅ Clinical-BERT model created")
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
        return True
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return False

def test_data_generation():
    """Test synthetic data generation"""
    print("\nTesting data generation...")
    
    try:
        from src.data.generate_10class import MedicalDataGenerator
        generator = MedicalDataGenerator()
        
        # Test small sample generation
        rx_data = generator.generate_rx_data(5)
        sx_data = generator.generate_sx_data(5)  # Test new Sx class
        
        print(f"✅ Data generation working")
        print(f"   Sample Rx: {rx_data[0][0][:50]}...")
        print(f"   Sample Sx: {sx_data[0][0][:50]}...")
        return True
    except Exception as e:
        print(f"❌ Data generation failed: {e}")
        return False

def test_device_setup():
    """Test device configuration"""
    print("\nTesting device setup...")
    
    try:
        import torch
        
        # Test CPU
        cpu_device = torch.device("cpu")
        print(f"✅ CPU device available: {cpu_device}")
        
        # Test GPU (if available)
        if torch.cuda.is_available():
            gpu_device = torch.device("cuda")
            gpu_name = torch.cuda.get_device_name()
            print(f"✅ GPU device available: {gpu_device} ({gpu_name})")
        else:
            print("ℹ️  GPU not available (CPU-only mode)")
        
        return True
    except Exception as e:
        print(f"❌ Device setup failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🏥 Medical BERT Classifier - Setup Test")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_config_files,
        test_module_imports,
        test_model_creation,
        test_data_generation,
        test_device_setup
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test_func.__name__} crashed: {e}")
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed! Setup is complete.")
        print("\nNext steps:")
        print("1. Run: python main.py --step data")
        print("2. Run: python main.py --step train --model clinical_bert")
        print("3. Or run: make demo")
        return True
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        print("\nTroubleshooting:")
        print("1. Install dependencies: pip install -r requirements/cpu.txt")
        print("2. Check Python version (3.8+ required)")
        print("3. Verify all files are in the correct locations")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
