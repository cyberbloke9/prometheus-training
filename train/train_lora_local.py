#!/usr/bin/env python3
"""
Prometheus LoRA Training Script - Optimized for Local Hardware
This script provides an easy-to-use interface for training Prometheus with LoRA
on consumer-grade GPUs.

Usage:
    # Default configuration (13B model, ~16-24GB VRAM)
    python train_lora_local.py

    # Low memory mode (8-16GB VRAM)
    python train_lora_local.py --memory_mode low

    # High performance mode (40GB+ VRAM)
    python train_lora_local.py --memory_mode high

    # Custom settings
    python train_lora_local.py --batch_size 2 --lora_rank 64 --learning_rate 3e-4
"""

import os
import sys
import argparse
import torch
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from llama_finetuning import main as train_main


def check_gpu():
    """Check GPU availability and memory"""
    if not torch.cuda.is_available():
        print(" ERROR: No CUDA-capable GPU detected!")
        print("This script requires a NVIDIA GPU with CUDA support.")
        sys.exit(1)

    device_count = torch.cuda.device_count()
    print(f" Found {device_count} GPU(s)")

    for i in range(device_count):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
        print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")

    return device_count


def get_memory_preset(memory_mode):
    """Get training configuration based on available memory"""
    presets = {
        "low": {
            "batch_size": 1,
            "gradient_accumulation_steps": 64,
            "lora_rank": 16,
            "lora_alpha": 32,
            "target_modules": "q_proj,v_proj",
            "quantization": True,
            "max_length": 2048,
            "description": "Low Memory (8-16GB VRAM)"
        },
        "medium": {
            "batch_size": 1,
            "gradient_accumulation_steps": 32,
            "lora_rank": 32,
            "lora_alpha": 64,
            "target_modules": "q_proj,k_proj,v_proj,o_proj",
            "quantization": True,
            "max_length": 2048,
            "description": "Medium Memory (16-24GB VRAM)"
        },
        "high": {
            "batch_size": 2,
            "gradient_accumulation_steps": 16,
            "lora_rank": 64,
            "lora_alpha": 128,
            "target_modules": "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
            "quantization": False,
            "max_length": 4096,
            "description": "High Memory (40GB+ VRAM)"
        }
    }
    return presets.get(memory_mode, presets["medium"])


def main():
    parser = argparse.ArgumentParser(
        description="Train Prometheus with LoRA on local hardware",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with default settings (medium memory)
  python train_lora_local.py

  # Train with low memory settings
  python train_lora_local.py --memory_mode low

  # Train with custom data
  python train_lora_local.py --data_path ./my_training_data.json

  # Train 7B model instead of 13B
  python train_lora_local.py --model_name meta-llama/Llama-2-7b-chat-hf --memory_mode low
        """
    )

    # Model and data arguments
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-2-13b-chat-hf",
        help="Base model to fine-tune (default: Llama-2-13b-chat-hf)"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="./sample_train_data.json",
        help="Path to training data JSON file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="C:/Users/Prithvi Putta/prometheus/lora_models",
        help="Directory to save trained model"
    )

    # Memory and performance arguments
    parser.add_argument(
        "--memory_mode",
        type=str,
        choices=["low", "medium", "high"],
        default="medium",
        help="Preset memory configuration (low: 8-16GB, medium: 16-24GB, high: 40GB+)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Training batch size (overrides memory_mode preset)"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=None,
        help="Gradient accumulation steps (effective_batch = batch_size * this)"
    )

    # LoRA specific arguments
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=None,
        help="LoRA rank (r parameter, higher = more capacity)"
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=None,
        help="LoRA alpha (scaling factor)"
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.05,
        help="LoRA dropout rate"
    )

    # Training hyperparameters
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=2048,
        help="Maximum sequence length"
    )

    # Optimization flags
    parser.add_argument(
        "--no_quantization",
        action="store_true",
        help="Disable 8-bit quantization (requires more memory)"
    )
    parser.add_argument(
        "--use_flash_attention",
        action="store_true",
        default=True,
        help="Use Flash Attention for memory efficiency"
    )

    # Logging and checkpointing
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="prometheus-lora",
        help="Experiment name for logging"
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Enable Weights & Biases logging"
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default="",
        help="W&B entity name"
    )

    # Advanced options
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Resume training from checkpoint path"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    # Check GPU availability
    print("\n" + "="*60)
    print(" Prometheus LoRA Training - Local Setup")
    print("="*60 + "\n")

    gpu_count = check_gpu()

    # Get memory preset and merge with user arguments
    preset = get_memory_preset(args.memory_mode)
    print(f"\n Memory Mode: {preset['description']}")
    print(f"   Batch Size: {args.batch_size or preset['batch_size']}")
    print(f"   Gradient Accumulation: {args.gradient_accumulation_steps or preset['gradient_accumulation_steps']}")
    print(f"   Effective Batch Size: {(args.batch_size or preset['batch_size']) * (args.gradient_accumulation_steps or preset['gradient_accumulation_steps'])}")
    print(f"   LoRA Rank: {args.lora_rank or preset['lora_rank']}")
    print(f"   Quantization: {preset['quantization'] and not args.no_quantization}\n")

    # Prepare training arguments
    training_args = {
        # Model and data
        "model_name": args.model_name,
        "data_path": args.data_path,
        "dataset": "feedback_collection_dataset",
        "hf_cache_dir": "C:/Users/Prithvi Putta/hf_cache",

        # Output
        "output_dir": args.output_dir,
        "experiment_name": args.experiment_name,

        # Batch size and accumulation
        "batch_size_training": args.batch_size or preset["batch_size"],
        "gradient_accumulation_steps": args.gradient_accumulation_steps or preset["gradient_accumulation_steps"],

        # LoRA configuration
        "use_peft": True,
        "peft_method": "lora",
        "lora_r": args.lora_rank or preset["lora_rank"],
        "lora_alpha": args.lora_alpha or preset["lora_alpha"],
        "lora_dropout": args.lora_dropout,

        # Training hyperparameters
        "lr": args.learning_rate,
        "num_epochs": args.num_epochs,
        "seed": args.seed,

        # Optimization
        "quantization": preset["quantization"] and not args.no_quantization,
        "use_fast_kernels": args.use_flash_attention,
        "mixed_precision": True,
        "use_fp16": False,

        # Distributed training
        "enable_fsdp": False,
        "one_gpu": True,

        # Checkpointing
        "save_model": True,
        "save_optimizer": False,

        # Scheduler
        "scheduler": "cosine",

        # Validation
        "run_validation": False,
        "num_workers_dataloader": 2,
    }

    # Add W&B configuration if enabled
    if args.use_wandb:
        if not args.wandb_entity:
            print("  WARNING: --use_wandb enabled but --wandb_entity not set!")
            print("   W&B logging may fail. Please update llama_finetuning.py lines 81 and 94")
            print("   to use your W&B entity, or disable with WANDB_MODE=disabled\n")

    # Print training configuration
    print("\n Training Configuration:")
    print(f"   Model: {args.model_name}")
    print(f"   Data: {args.data_path}")
    print(f"   Output: {args.output_dir}")
    print(f"   Learning Rate: {args.learning_rate}")
    print(f"   Epochs: {args.num_epochs}")
    print(f"   Max Length: {preset['max_length']}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Disable W&B if not explicitly enabled
    if not args.use_wandb:
        os.environ["WANDB_MODE"] = "disabled"

    print("\n" + "="*60)
    print(" Starting Training...")
    print("="*60 + "\n")

    # Launch training
    try:
        train_main(**training_args)
        print("\n" + "="*60)
        print(" Training completed successfully!")
        print(f" Model saved to: {args.output_dir}")
        print("="*60 + "\n")
    except KeyboardInterrupt:
        print("\n  Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
