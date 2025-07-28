"""
Industry-standard configuration management using Pydantic
This is how most production ML systems handle configs
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Literal, Union
from pathlib import Path
import yaml
import json


class DeviceConfig(BaseModel):
    """Device configuration schema"""
    auto_detect: bool = True
    prefer_gpu: bool = True
    fallback_to_cpu: bool = True
    cuda_device_id: int = 0


class BatchSizeConfig(BaseModel):
    """Batch size configuration"""
    train: int = Field(gt=0, description="Training batch size")
    eval: int = Field(gt=0, description="Evaluation batch size") 
    predict: int = Field(gt=0, description="Prediction batch size")


class ComputeConfig(BaseModel):
    """Compute-specific configuration"""
    batch_size: BatchSizeConfig
    learning_rate: float = Field(gt=0, le=1, description="Learning rate")
    num_epochs: int = Field(gt=0, description="Number of training epochs")
    warmup_steps: int = Field(ge=0, description="Warmup steps")
    weight_decay: float = Field(ge=0, le=1, description="Weight decay")
    gradient_accumulation_steps: int = Field(gt=0, description="Gradient accumulation")
    dataloader_num_workers: int = Field(ge=0, description="DataLoader workers")
    pin_memory: bool = False
    mixed_precision: Optional[bool] = None  # Only for GPU config


class OptimizerConfig(BaseModel):
    """Optimizer configuration"""
    type: Literal["AdamW", "Adam", "SGD"] = "AdamW"
    betas: List[float] = Field(default=[0.9, 0.999], min_items=2, max_items=2)
    eps: float = Field(default=1e-8, gt=0)
    amsgrad: bool = False
    
    @validator('betas')
    def validate_betas(cls, v):
        if not all(0 <= beta <= 1 for beta in v):
            raise ValueError('Beta values must be between 0 and 1')
        return v


class SchedulerConfig(BaseModel):
    """Learning rate scheduler configuration"""
    type: Literal["linear_warmup", "cosine", "step"] = "linear_warmup"
    warmup_ratio: float = Field(default=0.1, ge=0, le=1)


class EarlyStoppingConfig(BaseModel):
    """Early stopping configuration"""
    enabled: bool = True
    patience: int = Field(gt=0, description="Early stopping patience")
    monitor: str = "val_accuracy"
    mode: Literal["min", "max"] = "max"
    min_delta: float = Field(ge=0, description="Minimum change threshold")
    restore_best_weights: bool = True


class ClassifierConfig(BaseModel):
    """Classifier layer configuration"""
    dropout: float = Field(ge=0, le=1, description="Dropout rate")
    hidden_size: int = Field(gt=0, description="Hidden layer size")
    activation: Literal["gelu", "relu", "tanh"] = "gelu"
    bias: bool = True
    weight_initialization: str = "xavier_uniform"


class DataConfig(BaseModel):
    """Data configuration"""
    train_split: float = Field(gt=0, le=1, description="Training split ratio")
    val_split: float = Field(gt=0, le=1, description="Validation split ratio") 
    test_split: float = Field(ge=0, le=1, description="Test split ratio")
    shuffle_train: bool = True
    shuffle_eval: bool = False
    random_seed: int = 42
    
    @validator('val_split')
    def validate_splits(cls, v, values):
        train_split = values.get('train_split', 0)
        if train_split + v > 1:
            raise ValueError('train_split + val_split cannot exceed 1.0')
        return v


class TrainingStrategyConfig(BaseModel):
    """Training strategy configuration"""
    approach: str = "last_layer_only"
    freeze_backbone: bool = True
    trainable_components: List[str] = ["classifier_layer"]
    description: str = "Freeze BERT encoder and train only the final classification layer"


class RegularizationConfig(BaseModel):
    """Regularization configuration"""
    label_smoothing: float = Field(ge=0, le=1, default=0.1)
    dropout_rate: float = Field(ge=0, le=1, default=0.3)
    layer_norm_eps: float = Field(gt=0, default=1e-12)


class EvaluationConfig(BaseModel):
    """Evaluation configuration"""
    metrics: List[str] = ["accuracy", "precision", "recall", "f1", "confusion_matrix"]
    average: str = "weighted"
    class_report: bool = True
    save_predictions: bool = True


class CheckpointingConfig(BaseModel):
    """Checkpointing configuration"""
    enabled: bool = True
    save_best_only: bool = True
    save_frequency: str = "epoch"
    max_checkpoints: int = Field(gt=0, default=3)


class LoggingConfig(BaseModel):
    """Logging configuration"""
    level: str = "INFO"
    log_frequency: int = Field(gt=0, default=50)
    tensorboard: bool = False
    wandb: bool = False
    save_logs: bool = True


class ComparisonConfig(BaseModel):
    """Model comparison configuration"""
    enabled: bool = True
    models_to_compare: List[str] = ["bio_bert", "clinical_bio_bert", "blue_bert"]
    cross_validation: bool = True
    cv_folds: int = Field(gt=0, default=3)
    save_comparison_results: bool = True


class PredictionConfig(BaseModel):
    """Prediction configuration"""
    confidence_threshold: float = Field(ge=0, le=1, default=0.5)
    return_probabilities: bool = True
    batch_prediction: bool = True


class MonitoringConfig(BaseModel):
    """Performance monitoring configuration"""
    track_memory: bool = True
    track_training_time: bool = True
    track_inference_time: bool = True
    cpu_utilization: bool = True
    gpu_utilization: bool = False


class ReproducibilityConfig(BaseModel):
    """Reproducibility configuration"""
    deterministic: bool = True
    benchmark: bool = False
    set_seed_everywhere: bool = True


class AdvancedConfig(BaseModel):
    """Advanced settings configuration"""
    gradient_clipping: float = Field(gt=0, default=1.0)
    accumulate_grad_batches: int = Field(gt=0, default=4)
    precision: int = Field(default=32)
    auto_lr_find: bool = False
    auto_scale_batch_size: bool = False


class ExperimentalConfig(BaseModel):
    """Experimental features configuration"""
    knowledge_distillation: bool = False
    ensemble_methods: bool = False
    active_learning: bool = False
    few_shot_learning: bool = False


class ResourceLimitsConfig(BaseModel):
    """Resource limits configuration"""
    max_memory_gb: int = Field(gt=0, default=16)
    max_training_hours: float = Field(gt=0, default=2)
    checkpoint_memory_limit_gb: int = Field(gt=0, default=4)


class ErrorHandlingConfig(BaseModel):
    """Error handling configuration"""
    continue_on_error: bool = False
    retry_failed_batches: int = Field(ge=0, default=3)
    fallback_batch_size: int = Field(gt=0, default=4)


class OutputConfig(BaseModel):
    """Output configuration"""
    save_model: bool = True
    save_tokenizer: bool = True
    save_config: bool = True
    save_training_args: bool = True
    output_dir: str = "models"
    run_name: Optional[str] = None


class TrainingConfig(BaseModel):
    """Complete training configuration with validation"""
    
    # Core configuration sections
    training_strategy: TrainingStrategyConfig
    device: DeviceConfig
    cpu_config: ComputeConfig
    gpu_config: ComputeConfig
    data: DataConfig
    classifier: ClassifierConfig
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    early_stopping: EarlyStoppingConfig
    
    # Additional configuration sections
    regularization: RegularizationConfig
    evaluation: EvaluationConfig
    checkpointing: CheckpointingConfig
    logging: LoggingConfig
    comparison: ComparisonConfig
    prediction: PredictionConfig
    monitoring: MonitoringConfig
    reproducibility: ReproducibilityConfig
    advanced: AdvancedConfig
    experimental: ExperimentalConfig
    resource_limits: ResourceLimitsConfig
    error_handling: ErrorHandlingConfig
    output: OutputConfig
    
    class Config:
        """Pydantic configuration"""
        validate_assignment = True  # Validate on assignment too
        extra = "forbid"  # Forbid extra fields not in schema
        

class ConfigManager:
    """
    Industry-standard configuration manager
    Used by companies like Uber, Netflix, etc.
    """
    
    @staticmethod
    def load_from_yaml(config_path: Union[str, Path]) -> TrainingConfig:
        """Load and validate configuration from YAML"""
        try:
            with open(config_path, 'r') as f:
                raw_config = yaml.safe_load(f)
            
            # Pydantic automatically handles type conversion and validation
            config = TrainingConfig(**raw_config)
            
            print(f"✅ Configuration loaded and validated from {config_path}")
            return config
            
        except Exception as e:
            print(f"❌ Configuration error in {config_path}: {e}")
            raise
    
    @staticmethod  
    def load_from_dict(config_dict: dict) -> TrainingConfig:
        """Load configuration from dictionary"""
        return TrainingConfig(**config_dict)
    
    @staticmethod
    def save_to_yaml(config: TrainingConfig, output_path: Union[str, Path]):
        """Save configuration to YAML"""
        with open(output_path, 'w') as f:
            yaml.dump(config.dict(), f, default_flow_style=False)
    
    @staticmethod
    def save_to_json(config: TrainingConfig, output_path: Union[str, Path]):
        """Save configuration to JSON"""
        with open(output_path, 'w') as f:
            json.dump(config.dict(), f, indent=2)


def create_default_config() -> TrainingConfig:
    """Create a default configuration with proper types"""
    return TrainingConfig(
        training_strategy=TrainingStrategyConfig(),
        device=DeviceConfig(),
        cpu_config=ComputeConfig(
            batch_size=BatchSizeConfig(train=8, eval=16, predict=32),
            learning_rate=2e-4,
            num_epochs=10,
            warmup_steps=50,
            weight_decay=0.01,
            gradient_accumulation_steps=4,
            dataloader_num_workers=2
        ),
        gpu_config=ComputeConfig(
            batch_size=BatchSizeConfig(train=16, eval=32, predict=64),
            learning_rate=2e-4,
            num_epochs=5,
            warmup_steps=100,
            weight_decay=0.01,
            gradient_accumulation_steps=1,
            dataloader_num_workers=4,
            pin_memory=True,
            mixed_precision=True
        ),
        data=DataConfig(train_split=0.8, val_split=0.2, test_split=0.0),
        classifier=ClassifierConfig(dropout=0.3, hidden_size=768),
        optimizer=OptimizerConfig(),
        scheduler=SchedulerConfig(),
        early_stopping=EarlyStoppingConfig(patience=3, min_delta=0.001),
        regularization=RegularizationConfig(),
        evaluation=EvaluationConfig(),
        checkpointing=CheckpointingConfig(),
        logging=LoggingConfig(),
        comparison=ComparisonConfig(),
        prediction=PredictionConfig(),
        monitoring=MonitoringConfig(),
        reproducibility=ReproducibilityConfig(),
        advanced=AdvancedConfig(),
        experimental=ExperimentalConfig(),
        resource_limits=ResourceLimitsConfig(),
        error_handling=ErrorHandlingConfig(),
        output=OutputConfig()
    )


# Usage example:
if __name__ == "__main__":
    # This is how you'd use it in production
    
    # Method 1: Load from YAML (with automatic validation)
    try:
        config = ConfigManager.load_from_yaml("config/training.yaml")
        print("Config loaded successfully!")
        print(f"Learning rate: {config.cpu_config.learning_rate}")  # Always a float!
        print(f"Batch size: {config.cpu_config.batch_size.train}")  # Always an int!
        
    except Exception as e:
        print(f"Config error: {e}")
        
    # Method 2: Create from code with validation
    config = create_default_config()
    
    # Method 3: Access with type safety
    lr = config.cpu_config.learning_rate  # IDE knows this is float
    epochs = config.cpu_config.num_epochs  # IDE knows this is int
