# Quick Setup Guide

🏥 **Medical BERT Classifier - Simplified Last-Layer Training**

This guide will get you up and running in under 5 minutes!

## 🚀 Quick Start (3 Steps)

### Step 1: Install Dependencies
```bash
# CPU-only setup (recommended for most users)
pip install -r requirements/cpu.txt

# OR for GPU acceleration (if you have CUDA)
pip install -r requirements/gpu.txt
```

### Step 2: Test Your Setup
```bash
# Run the setup test to verify everything works
python test_setup.py
```

### Step 3: Run the Demo
```bash
# Quick demonstration of the complete pipeline
python demo.py

# OR run the full pipeline
python main.py --step all
```

That's it! 🎉

---

## 🛠️ Detailed Setup Options

### Option A: Complete Pipeline
```bash
# Generate data, train model, and evaluate
python main.py --step all
```

### Option B: Step-by-Step
```bash
# 1. Generate synthetic medical data
python main.py --step data

# 2. Train Clinical-Bio-BERT model
python main.py --step train --model clinical_bert

# 3. Compare all medical BERT models
python main.py --step compare
```

### Option C: Interactive Notebook
```bash
# Launch Jupyter Lab with training notebook
jupyter lab notebooks/medical_bert_training.ipynb
```

### Option D: Using Make Commands
```bash
# Show all available commands
make help

# Quick demo
make demo

# Train specific models
make clinical    # Clinical-Bio-BERT
make bio         # Bio-BERT
make compare     # All models
```

---

## 💻 System Requirements

### Minimum Requirements
- **Python**: 3.8+
- **RAM**: 8GB (16GB recommended)
- **Storage**: 2GB free space
- **OS**: Windows, macOS, or Linux

### Optional (for GPU acceleration)
- **CUDA**: 11.8+ or 12.1+
- **GPU RAM**: 6GB+ VRAM
- **GPU**: NVIDIA GPU with CUDA support

---

## 🏥 What You'll Get

### 10-Class Medical Text Classification
- **Rx**: Medications & Prescriptions
- **Lx**: Laboratory Tests & Results  
- **Dx**: Medical Diagnoses
- **Px**: Medical Procedures
- **Sx**: Symptoms & Complaints ⭐ *NEW!*
- **Other**: Administrative Content
- **Vitals**: Vital Signs & Measurements
- **FHx**: Family Medical History
- **SDOH**: Social Determinants of Health
- **Tobacco**: Smoking & Tobacco Use

### Medical BERT Models
- **Bio-BERT**: For biomedical literature
- **Clinical-Bio-BERT**: For clinical notes
- **BlueBERT**: For mixed medical text

### Key Features
- ⚡ **Fast training**: Last-layer only (2-5 minutes)
- 💻 **CPU optimized**: Works without GPU
- 🔧 **Configurable**: Easy YAML configuration
- 📊 **Comprehensive**: Built-in evaluation and comparison

---

## 🔧 Troubleshooting

### Common Issues

**Problem**: `ImportError: No module named 'torch'`
```bash
# Solution: Install PyTorch
pip install torch>=2.0.0
```

**Problem**: `CUDA out of memory`
```bash
# Solution: Use CPU mode
python main.py --step train --cpu-only
```

**Problem**: `Tokenizer warnings`
```bash
# Solution: Already handled (warnings suppressed)
# No action needed
```

**Problem**: Slow training on CPU
```bash
# Solution: Use smaller dataset for testing
python main.py --step all --epochs 2
```

### Getting Help

1. **Run setup test**: `python test_setup.py`
2. **Check requirements**: `make check-deps`
3. **View project info**: `make info`
4. **Clean and restart**: `make clean-all && make setup`

---

## 📖 Next Steps

### After Setup
1. **Explore the notebook**: `jupyter lab notebooks/medical_bert_training.ipynb`
2. **Compare models**: `python main.py --step compare`
3. **Customize classes**: Edit `config/classes.yaml`
4. **Adjust training**: Edit `config/training.yaml`

### For Real Data
1. Prepare your CSV with columns: `text`, `label`, `label_id`
2. Update data path in configurations
3. Retrain models with your data

### For Production
1. Save trained models: `trainer.save_model("my_model.pt")`
2. Load for inference: `trainer.load_model("my_model.pt")`
3. Deploy with REST API (future feature)

---

## 🎯 Example Commands

```bash
# Quick test with Bio-BERT
python main.py --step train --model bio_bert --epochs 2

# Compare models with custom epochs
python main.py --step compare --epochs 5

# Generate new data and retrain
python main.py --step data --force-regenerate
python main.py --step train --model clinical_bert

# CPU-only training
python main.py --step all --cpu-only

# Clean everything and start fresh
make clean-all && make setup && make demo
```

---

## 💡 Tips for Best Results

1. **Start with the demo**: `python demo.py`
2. **Use Clinical-Bio-BERT**: Best for medical text
3. **Try GPU if available**: Faster training
4. **Experiment with epochs**: 3-10 usually sufficient
5. **Check confusion matrix**: Identify problem classes
6. **Use real data**: Replace synthetic with actual clinical notes

---

## 🆘 Still Having Issues?

1. **Check Python version**: `python --version` (3.8+ required)
2. **Verify setup**: `python test_setup.py`
3. **Clean install**: Remove and reinstall dependencies
4. **Check memory**: Ensure 8GB+ RAM available
5. **Update packages**: `pip install --upgrade -r requirements.txt`

**Remember**: This is designed to work out-of-the-box with minimal setup! 🚀
