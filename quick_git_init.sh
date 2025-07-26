#!/bin/bash
# Quick Git Setup - Minimal version of init_git.sh

# Initialize git and commit all files with a simple message
git init
git add .
git commit -m "🏥 Initial commit: Medical BERT Classifier - Simplified Last-Layer Training

Complete project setup with:
- Last-layer training for medical BERT models  
- 10-class taxonomy (including new Sx class)
- Bio-BERT, Clinical-Bio-BERT, BlueBERT support
- CPU-optimized training with GPU acceleration
- YAML configuration system
- Comprehensive documentation"

echo "✅ Git repository initialized and initial commit created!"
echo "🚀 Next: python test_setup.py"
