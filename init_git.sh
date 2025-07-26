#!/bin/bash

# Medical BERT Classifier - Git Repository Initialization Script
# This script initializes a git repository and commits all project files

set -e  # Exit on any error

echo "🏥 Medical BERT Classifier - Git Repository Setup"
echo "=================================================="

# Check if we're already in a git repository
if [ -d ".git" ]; then
    echo "⚠️  Git repository already exists in this directory."
    echo "Do you want to reinitialize? This will preserve existing history. (y/N)"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        echo "❌ Aborted. Existing git repository preserved."
        exit 1
    fi
fi

# Initialize git repository
echo "📦 Initializing git repository..."
git init

# Set up gitignore first (best practice)
echo "🙈 Adding .gitignore..."
git add .gitignore

# Check git configuration
echo "🔧 Checking git configuration..."
if ! git config user.name > /dev/null 2>&1; then
    echo "⚠️  Git user.name not set. Please configure:"
    echo "   git config --global user.name \"Your Name\""
    echo "   git config --global user.email \"your.email@example.com\""
    echo ""
    echo "Continue anyway? (y/N)"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        echo "❌ Please configure git first, then run this script again."
        exit 1
    fi
fi

# Show current directory structure
echo ""
echo "📁 Current project structure:"
echo "=============================="
find . -type f -not -path "./.git/*" | head -20
echo ""

# Show what files will be committed
echo "📋 Files to be committed:"
echo "========================="
echo ""

# Add all project files
git add .

# Show status
echo "Git status after staging:"
git status --short

echo ""
echo "📊 Summary of staged files:"
git diff --cached --stat

echo ""
echo "🔍 Review staged files (y/N)?"
read -r review
if [[ "$review" =~ ^[Yy]$ ]]; then
    echo ""
    echo "Staged files:"
    git diff --cached --name-only | while read file; do
        echo "  ✅ $file"
    done
fi

echo ""
echo "💬 Commit message options:"
echo "1. Use default comprehensive message (recommended)"
echo "2. Enter custom commit message"
echo "3. Cancel and exit"
echo ""
echo "Choose option (1-3): "
read -r option

case $option in
    1)
        # Default comprehensive commit message
        commit_message="🏥 Initial commit: Medical BERT Classifier - Simplified Last-Layer Training

✨ Features:
- Last-layer training approach for efficient medical text classification
- 10-class medical taxonomy including new Sx (Symptoms) class
- Multiple medical BERT models: Bio-BERT, Clinical-Bio-BERT, BlueBERT
- CPU-optimized training with GPU acceleration support
- YAML-based configuration system
- Comprehensive documentation and setup guides

📊 Classification Classes:
- Rx: Medications & Prescriptions
- Lx: Laboratory Tests & Results  
- Dx: Medical Diagnoses
- Px: Medical Procedures
- Sx: Symptoms & Complaints (NEW!)
- Other: Administrative Content
- Vitals: Vital Signs & Measurements
- FHx: Family Medical History
- SDOH: Social Determinants of Health
- Tobacco: Smoking & Tobacco Use

🚀 Quick Start:
- python test_setup.py (verify setup)
- python demo.py (quick demonstration)
- python main.py --step all (full pipeline)
- jupyter lab notebooks/medical_bert_training.ipynb (interactive)

💻 Technical Stack:
- PyTorch + Transformers
- Specialized medical BERT models
- Last-layer fine-tuning approach
- CPU-first design with GPU support
- Synthetic medical data generation
- Model comparison framework

📁 Project Structure:
- src/: Source code with models, training, and data modules
- config/: YAML configuration files
- notebooks/: Interactive Jupyter notebook
- experiments/: Model comparison framework
- requirements/: CPU and GPU dependency specifications

🎯 Benefits:
- Fast training (2-5 minutes vs 10-30 minutes)
- Low memory usage (8GB vs 16GB+ RAM)
- Only ~0.1% of parameters trainable
- Medical domain specialization
- Production-ready architecture"
        ;;
    2)
        echo "Enter your commit message:"
        read -r commit_message
        if [ -z "$commit_message" ]; then
            echo "❌ Empty commit message. Aborting."
            exit 1
        fi
        ;;
    3)
        echo "❌ Cancelled by user."
        exit 1
        ;;
    *)
        echo "❌ Invalid option. Aborting."
        exit 1
        ;;
esac

echo ""
echo "📝 Commit message preview:"
echo "=========================="
echo "$commit_message"
echo ""

echo "Proceed with commit? (y/N)"
read -r confirm
if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
    echo "❌ Commit cancelled."
    echo "Files remain staged. You can commit later with: git commit"
    exit 1
fi

# Create the initial commit
echo "💾 Creating initial commit..."
git commit -m "$commit_message"

echo ""
echo "✅ Initial commit created successfully!"

# Show commit info
echo ""
echo "📋 Commit details:"
echo "=================="
git log --oneline -1
echo ""
git show --stat HEAD

echo ""
echo "🌿 Branch setup options:"
echo "1. Stay on main branch"
echo "2. Create and switch to develop branch"
echo "3. Create feature branch for development"
echo ""
echo "Choose option (1-3): "
read -r branch_option

case $branch_option in
    1)
        echo "✅ Staying on main branch"
        ;;
    2)
        echo "🌿 Creating develop branch..."
        git checkout -b develop
        echo "✅ Switched to develop branch"
        ;;
    3)
        echo "Enter feature branch name (e.g., feature/enhancements):"
        read -r branch_name
        if [ -n "$branch_name" ]; then
            git checkout -b "$branch_name"
            echo "✅ Created and switched to branch: $branch_name"
        else
            echo "⚠️  Empty branch name. Staying on main."
        fi
        ;;
    *)
        echo "⚠️  Invalid option. Staying on main branch."
        ;;
esac

echo ""
echo "🎉 Git repository setup complete!"
echo ""
echo "📋 Repository status:"
echo "==================="
echo "Branch: $(git branch --show-current)"
echo "Commits: $(git rev-list --count HEAD)"
echo "Files tracked: $(git ls-files | wc -l)"
echo ""

echo "🚀 Next steps:"
echo "============="
echo "1. Test the setup: python test_setup.py"
echo "2. Run demo: python demo.py"
echo "3. Add remote repository (if needed):"
echo "   git remote add origin <repository-url>"
echo "   git push -u origin main"
echo ""

echo "📚 Useful git commands:"
echo "====================="
echo "git status          # Check repository status"
echo "git log --oneline   # View commit history"
echo "git branch -a       # List all branches"
echo "git diff            # See unstaged changes"
echo ""

echo "✨ Repository ready for development!"
