.PHONY: help install setup clean data train compare evaluate notebook test format lint all

# Default target
help: ## Show this help message
	@echo "Medical BERT Classifier - Simplified Last-Layer Training"
	@echo "Available commands:"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# Environment setup
setup: ## Complete project setup
	@echo "Setting up Medical BERT Classifier..."
	pip install -r requirements/cpu.txt
	@echo "✅ Setup complete!"

setup-gpu: ## Setup with GPU support
	@echo "Setting up Medical BERT Classifier with GPU support..."
	pip install -r requirements/gpu.txt
	@echo "✅ GPU setup complete!"

install: setup ## Alias for setup

# Data operations
data: ## Generate synthetic training data
	@echo "Generating synthetic medical text data..."
	python main.py --step data
	@echo "✅ Data generation complete!"

data-force: ## Force regenerate training data
	@echo "Force regenerating synthetic medical text data..."
	python main.py --step data --force-regenerate
	@echo "✅ Data regeneration complete!"

# Training operations
train: ## Train default model (Clinical-Bio-BERT)
	@echo "Training Clinical-Bio-BERT model..."
	python main.py --step train --model clinical_bert
	@echo "✅ Training complete!"

train-bio: ## Train Bio-BERT model
	@echo "Training Bio-BERT model..."
	python main.py --step train --model bio_bert
	@echo "✅ Bio-BERT training complete!"

train-clinical: ## Train Clinical-Bio-BERT model
	@echo "Training Clinical-Bio-BERT model..."
	python main.py --step train --model clinical_bert
	@echo "✅ Clinical-Bio-BERT training complete!"

train-blue: ## Train BlueBERT model
	@echo "Training BlueBERT model..."
	python main.py --step train --model blue_bert
	@echo "✅ BlueBERT training complete!"

train-all: ## Train all medical BERT models
	@echo "Training all medical BERT models..."
	python main.py --step compare
	@echo "✅ All models training complete!"

train-fast: ## Quick training (2 epochs) for testing
	@echo "Quick training for testing..."
	python main.py --step train --model clinical_bert --epochs 2
	@echo "✅ Quick training complete!"

train-cpu: ## Force CPU-only training
	@echo "Training with CPU only..."
	python main.py --step train --model clinical_bert --cpu-only
	@echo "✅ CPU training complete!"

# Evaluation and comparison
compare: ## Compare all medical BERT models
	@echo "Comparing medical BERT models..."
	python main.py --step compare
	@echo "✅ Model comparison complete!"

evaluate: ## Evaluate default model
	@echo "Evaluating Clinical-Bio-BERT model..."
	python main.py --step evaluate --model clinical_bert
	@echo "✅ Evaluation complete!"

# Complete pipeline
all: ## Run complete pipeline (data + train + compare)
	@echo "Running complete pipeline..."
	python main.py --step all
	@echo "✅ Complete pipeline finished!"

all-fast: ## Run complete pipeline with quick training
	@echo "Running complete pipeline (fast)..."
	python main.py --step all --epochs 2
	@echo "✅ Fast pipeline finished!"

# Interactive tools
notebook: ## Start Jupyter Lab with training notebook
	@echo "Starting Jupyter Lab..."
	jupyter lab notebooks/medical_bert_training.ipynb

jupyter: notebook ## Alias for notebook

# Testing and quality
test: ## Run tests (placeholder for future tests)
	@echo "Running tests..."
	@echo "⚠️  Test suite not yet implemented"

format: ## Format code with black (if available)
	@echo "Formatting code..."
	@if command -v black >/dev/null 2>&1; then \
		black src/ main.py experiments/; \
		echo "✅ Code formatted!"; \
	else \
		echo "⚠️  Black not installed. Run: pip install black"; \
	fi

lint: ## Lint code with flake8 (if available)
	@echo "Linting code..."
	@if command -v flake8 >/dev/null 2>&1; then \
		flake8 src/ main.py experiments/ --max-line-length=88; \
		echo "✅ Code linted!"; \
	else \
		echo "⚠️  Flake8 not installed. Run: pip install flake8"; \
	fi

# Cleanup operations
clean: ## Clean generated files and cache
	@echo "Cleaning up..."
	rm -rf __pycache__ src/__pycache__ experiments/__pycache__
	rm -rf .pytest_cache
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete
	find . -name "*~" -delete
	@echo "✅ Cleanup complete!"

clean-data: ## Remove generated data files
	@echo "Removing generated data..."
	rm -f data/synthetic_training_data_10class.csv
	rm -f data/synthetic_training_data_10class_summary.txt
	@echo "✅ Data cleanup complete!"

clean-models: ## Remove trained models
	@echo "Removing trained models..."
	rm -f models/*.pt
	rm -f models/*.pth
	@echo "✅ Models cleanup complete!"

clean-results: ## Remove result files
	@echo "Removing results..."
	rm -f results/*.png
	rm -f results/*.csv
	rm -f results/*.txt
	@echo "✅ Results cleanup complete!"

clean-all: clean clean-data clean-models clean-results ## Remove all generated files
	@echo "✅ Complete cleanup finished!"

# Information and diagnostics
info: ## Show project information
	@echo "Medical BERT Classifier - Project Information"
	@echo "============================================="
	@echo "Python version: $$(python --version 2>&1)"
	@echo "PyTorch version: $$(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'Not installed')"
	@echo "CUDA available: $$(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'PyTorch not available')"
	@echo "Project structure:"
	@find . -name "*.py" -not -path "./__pycache__/*" | head -10
	@echo "Configuration files:"
	@ls -la config/

check-deps: ## Check if dependencies are installed
	@echo "Checking dependencies..."
	@python -c "import torch; print('✅ PyTorch installed')" || echo "❌ PyTorch missing"
	@python -c "import transformers; print('✅ Transformers installed')" || echo "❌ Transformers missing"
	@python -c "import pandas; print('✅ Pandas installed')" || echo "❌ Pandas missing"
	@python -c "import sklearn; print('✅ Scikit-learn installed')" || echo "❌ Scikit-learn missing"
	@python -c "import yaml; print('✅ PyYAML installed')" || echo "❌ PyYAML missing"

# Quick shortcuts for common tasks
quick-test: data-force train-fast ## Quick test: generate data and fast train
	@echo "✅ Quick test complete!"

demo: ## Run a quick demo of the system
	@echo "Running Medical BERT Classifier demo..."
	python main.py --step all --epochs 2
	@echo "✅ Demo complete! Check results/ directory for outputs."

# Documentation
docs: ## Generate documentation (placeholder)
	@echo "📚 Documentation generation not yet implemented"
	@echo "For now, see README.md and docstrings in the code"

# Development helpers
dev-setup: ## Setup development environment
	@echo "Setting up development environment..."
	pip install -r requirements/cpu.txt
	@if command -v pip >/dev/null 2>&1; then \
		pip install black flake8 pytest jupyter; \
		echo "✅ Development tools installed!"; \
	fi

# Model-specific shortcuts
bio: train-bio ## Shortcut for Bio-BERT training
clinical: train-clinical ## Shortcut for Clinical-Bio-BERT training
blue: train-blue ## Shortcut for BlueBERT training

# Utility targets
size: ## Show project size
	@echo "Project size:"
	@du -sh .
	@echo "Code lines:"
	@find . -name "*.py" -not -path "./__pycache__/*" | xargs wc -l | tail -1

tree: ## Show project tree structure
	@echo "Project structure:"
	@tree -I '__pycache__|*.pyc|*.pyo' . || ls -la

# Error handling
check-data: ## Check if training data exists
	@if [ -f "data/synthetic_training_data_10class.csv" ]; then \
		echo "✅ Training data found"; \
		wc -l data/synthetic_training_data_10class.csv; \
	else \
		echo "❌ Training data not found. Run: make data"; \
	fi

check-models: ## Check for trained models
	@echo "Checking for trained models..."
	@if [ -d "models" ] && [ "$$(ls -A models/)" ]; then \
		echo "✅ Trained models found:"; \
		ls -la models/; \
	else \
		echo "❌ No trained models found. Run: make train"; \
	fi

# Performance monitoring
profile: ## Run training with performance profiling (if available)
	@echo "⚠️  Performance profiling not yet implemented"
	@echo "For basic timing, use: time make train"

# Advanced features (future)
deploy: ## Deploy model (placeholder for future)
	@echo "🚀 Model deployment not yet implemented"

serve: ## Start model serving (placeholder for future)
	@echo "🌐 Model serving not yet implemented"
