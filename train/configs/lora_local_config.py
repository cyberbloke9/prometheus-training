# Optimized LoRA Training Configuration for Local Setup
# This configuration is designed for consumer-grade hardware (RTX 3090/4090, etc.)

from dataclasses import dataclass
from typing import ClassVar, List

@dataclass
class lora_local_config:
    """
    Optimized LoRA configuration for local training on single GPU
    Memory requirement: ~16-24GB VRAM for 13B model with these settings
    """
    # LoRA hyperparameters
    r: int = 32  # LoRA rank (higher = more capacity, more memory)
    lora_alpha: int = 64  # Scaling factor (typically 2x rank)

    # Target modules - expanding beyond default for better performance
    target_modules: ClassVar[List[str]] = [
        "q_proj",  # Query projection
        "k_proj",  # Key projection
        "v_proj",  # Value projection
        "o_proj",  # Output projection
        "gate_proj",  # For LLaMA-2 FFN
        "up_proj",   # For LLaMA-2 FFN
        "down_proj"  # For LLaMA-2 FFN
    ]

    bias: str = "none"  # Don't train bias terms
    task_type: str = "CAUSAL_LM"
    lora_dropout: float = 0.05  # Regularization
    inference_mode: bool = False

@dataclass
class lora_low_memory_config:
    """
    Ultra low memory LoRA config for 8-16GB VRAM GPUs
    Works with 13B models using 8-bit quantization
    """
    r: int = 16  # Lower rank for less memory
    lora_alpha: int = 32

    target_modules: ClassVar[List[str]] = [
        "q_proj",
        "v_proj"  # Only Q and V for minimal setup
    ]

    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    lora_dropout: float = 0.1  # Higher dropout for regularization
    inference_mode: bool = False

@dataclass
class lora_high_performance_config:
    """
    High performance LoRA config for multi-GPU or high VRAM setups
    Requires 40GB+ VRAM or multiple GPUs
    """
    r: int = 64  # High rank for maximum expressiveness
    lora_alpha: int = 128

    target_modules: ClassVar[List[str]] = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ]

    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    lora_dropout: float = 0.05
    inference_mode: bool = False

@dataclass
class train_config_lora_local:
    """
    Training configuration optimized for local LoRA training
    """
    # Model settings
    model_name: str = "meta-llama/Llama-2-13b-chat-hf"
    hf_cache_dir: str = "C:/Users/Prithvi Putta/hf_cache"

    # Dataset settings
    dataset: str = "feedback_collection_dataset"
    data_path: str = "./sample_train_data.json"

    # Training hyperparameters
    batch_size_training: int = 1  # Small batch for memory efficiency
    gradient_accumulation_steps: int = 32  # Effective batch size = 32
    num_epochs: int = 3
    lr: float = 2e-4  # Higher LR works well with LoRA
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03  # Warmup for first 3% of training

    # Optimization settings
    use_peft: bool = True
    peft_method: str = "lora"
    quantization: bool = True  # Enable 8-bit for memory savings
    use_fp16: bool = False
    mixed_precision: bool = True  # Use BF16 if available

    # Memory optimization
    use_fast_kernels: bool = True  # Flash Attention
    gradient_checkpointing: bool = True  # Save memory at cost of speed

    # Distributed training (set to False for single GPU)
    enable_fsdp: bool = False
    low_cpu_fsdp: bool = False

    # Saving and logging
    save_model: bool = True
    output_dir: str = "C:/Users/Prithvi Putta/prometheus/lora_models"
    save_steps: int = 100  # Save checkpoint every 100 steps
    logging_steps: int = 10
    eval_steps: int = 50

    # Scheduler
    scheduler: str = "cosine"

    # Other settings
    seed: int = 42
    num_workers_dataloader: int = 2
    run_validation: bool = False
    experiment_name: str = "prometheus-lora-local"

    # Weights & Biases (optional)
    use_wandb: bool = False  # Set to True if you want tracking
    wandb_entity: str = "YOUR_WANDB_ENTITY"  # Change this
    wandb_project: str = "prometheus-lora"
