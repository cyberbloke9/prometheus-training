#!/usr/bin/env python3
"""
Unified Prometheus Training Interface
Train Prometheus with LoRA using either local hardware or Tinker API

This script provides a single interface that automatically chooses the best
training method based on available resources and user preferences.

Usage:
    # Auto-detect: Use Tinker if API key available, otherwise local
    python prometheus_train.py --data_path ./data.json

    # Force local training
    python prometheus_train.py --data_path ./data.json --mode local

    # Force Tinker training
    python prometheus_train.py --data_path ./data.json --mode tinker

    # Interactive mode (asks for preferences)
    python prometheus_train.py --interactive
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from typing import Optional, Dict

# Fix Windows console encoding
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


class PrometheusTrainingManager:
    """Manage Prometheus training across local and Tinker backends"""

    def __init__(self):
        self.script_dir = Path(__file__).parent
        self.has_tinker = self._check_tinker_available()
        self.has_gpu = self._check_gpu_available()

    def _check_tinker_available(self) -> bool:
        """Check if Tinker API is available"""
        # Check for API key
        if not os.environ.get("TINKER_API_KEY"):
            return False

        # Check for SDK
        try:
            import tinker
            return True
        except ImportError:
            return False

    def _check_gpu_available(self) -> bool:
        """Check if CUDA GPU is available"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def get_gpu_info(self) -> Optional[Dict]:
        """Get GPU information"""
        if not self.has_gpu:
            return None

        try:
            import torch
            gpu_count = torch.cuda.device_count()
            gpus = []
            for i in range(gpu_count):
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
                gpus.append({
                    "id": i,
                    "name": torch.cuda.get_device_name(i),
                    "memory_gb": gpu_memory
                })
            return {
                "count": gpu_count,
                "gpus": gpus
            }
        except:
            return None

    def recommend_mode(self) -> str:
        """Recommend training mode based on available resources"""
        if self.has_tinker:
            return "tinker"
        elif self.has_gpu:
            return "local"
        else:
            return "none"

    def print_status(self):
        """Print system status and capabilities"""
        print("\n" + "="*70)
        print("🔍 Prometheus Training System Status")
        print("="*70)

        # Tinker availability
        print(f"\n{'✓' if self.has_tinker else '✗'} Tinker API: ", end="")
        if self.has_tinker:
            print("Available")
        else:
            if not os.environ.get("TINKER_API_KEY"):
                print("Not available (No API key set)")
                print("  → Set TINKER_API_KEY environment variable to enable")
            else:
                print("Not available (SDK not installed)")
                print("  → Install with: pip install tinker-sdk")

        # GPU availability
        print(f"\n{'✓' if self.has_gpu else '✗'} Local GPU: ", end="")
        if self.has_gpu:
            gpu_info = self.get_gpu_info()
            print(f"Available ({gpu_info['count']} GPU(s))")
            for gpu in gpu_info['gpus']:
                print(f"  → GPU {gpu['id']}: {gpu['name']} ({gpu['memory_gb']:.1f} GB)")
        else:
            print("Not available")
            print("  → CUDA-capable GPU required for local training")

        # Recommendation
        recommended = self.recommend_mode()
        print(f"\n💡 Recommended Mode: ", end="")
        if recommended == "tinker":
            print("Tinker API (Best for most use cases)")
            print("  → No GPU required, automatic scaling, pay-per-use")
        elif recommended == "local":
            print("Local Training")
            print("  → GPU available, full control, no API costs")
        else:
            print("None (Insufficient resources)")
            print("  → Either set up Tinker API or install CUDA GPU")

        print("="*70 + "\n")

    def train_local(self, args: argparse.Namespace):
        """Execute local training"""
        print("\n🚀 Starting LOCAL training...")

        cmd = [
            sys.executable,
            str(self.script_dir / "train_lora_local.py"),
            "--data_path", args.data_path,
            "--model_name", args.model,
            "--output_dir", args.output_dir,
            "--memory_mode", args.memory_mode,
            "--learning_rate", str(args.learning_rate),
            "--num_epochs", str(args.num_epochs),
            "--lora_rank", str(args.lora_rank),
            "--experiment_name", args.experiment_name,
        ]

        if args.use_wandb:
            cmd.append("--use_wandb")
            if args.wandb_entity:
                cmd.extend(["--wandb_entity", args.wandb_entity])

        if args.no_quantization:
            cmd.append("--no_quantization")

        print(f"Command: {' '.join(cmd)}\n")
        result = subprocess.run(cmd)
        return result.returncode

    def train_tinker(self, args: argparse.Namespace):
        """Execute Tinker training"""
        print("\n🚀 Starting TINKER training...")

        cmd = [
            sys.executable,
            str(self.script_dir / "train_tinker.py"),
            "--data_path", args.data_path,
            "--model", args.model,
            "--output_dir", args.output_dir,
            "--learning_rate", str(args.learning_rate),
            "--num_epochs", str(args.num_epochs),
            "--lora_rank", str(args.lora_rank),
            "--experiment_name", args.experiment_name,
            "--batch_size", str(args.batch_size),
        ]

        if args.download_model:
            cmd.append("--download")

        if args.test_inference:
            cmd.append("--test_inference")

        print(f"Command: {' '.join(cmd)}\n")
        result = subprocess.run(cmd)
        return result.returncode

    def interactive_mode(self):
        """Interactive configuration wizard"""
        print("\n" + "="*70)
        print("🧙 Prometheus Training Wizard")
        print("="*70 + "\n")

        self.print_status()

        # Get training mode
        recommended = self.recommend_mode()
        if recommended == "none":
            print("❌ Cannot proceed: No training backend available")
            sys.exit(1)

        print(f"Recommended mode: {recommended}")
        mode = input(f"Choose training mode (local/tinker) [{recommended}]: ").strip().lower()
        if not mode:
            mode = recommended

        if mode not in ["local", "tinker"]:
            print(f"❌ Invalid mode: {mode}")
            sys.exit(1)

        # Validate mode
        if mode == "tinker" and not self.has_tinker:
            print("❌ Tinker API not available")
            sys.exit(1)
        if mode == "local" and not self.has_gpu:
            print("❌ No GPU available for local training")
            sys.exit(1)

        # Get configuration
        print("\n📋 Training Configuration:")

        data_path = input("Training data path [./sample_train_data.json]: ").strip()
        if not data_path:
            data_path = "./sample_train_data.json"

        model = input("Base model [meta-llama/Llama-2-13b-chat-hf]: ").strip()
        if not model:
            model = "meta-llama/Llama-2-13b-chat-hf"

        # Create args object
        class Args:
            pass

        args = Args()
        args.mode = mode
        args.data_path = data_path
        args.model = model
        args.output_dir = "C:/Users/Prithvi Putta/prometheus/trained_models"
        args.memory_mode = "medium"
        args.learning_rate = 2e-4
        args.num_epochs = 3
        args.lora_rank = 32
        args.batch_size = 4
        args.experiment_name = "prometheus-interactive"
        args.use_wandb = False
        args.wandb_entity = ""
        args.no_quantization = False
        args.download_model = True
        args.test_inference = True

        print("\n" + "="*70)
        print("Starting training with configuration:")
        print(f"  Mode: {mode}")
        print(f"  Data: {data_path}")
        print(f"  Model: {model}")
        print("="*70 + "\n")

        if mode == "local":
            return self.train_local(args)
        else:
            return self.train_tinker(args)


def main():
    parser = argparse.ArgumentParser(
        description="Unified Prometheus Training Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-detect best training method
  python prometheus_train.py --data_path ./data.json

  # Force local training
  python prometheus_train.py --data_path ./data.json --mode local

  # Force Tinker training
  python prometheus_train.py --data_path ./data.json --mode tinker

  # Interactive wizard
  python prometheus_train.py --interactive

  # Custom configuration
  python prometheus_train.py --data_path ./data.json --mode local \\
      --model meta-llama/Llama-2-7b-chat-hf --lora_rank 64 --num_epochs 5
        """
    )

    # Mode selection
    parser.add_argument(
        "--mode",
        type=str,
        choices=["auto", "local", "tinker"],
        default="auto",
        help="Training mode: auto (detect best), local (GPU), or tinker (API)"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Interactive configuration wizard"
    )

    # Data and model
    parser.add_argument(
        "--data_path",
        type=str,
        default="./sample_train_data.json",
        help="Path to training data JSON"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-2-13b-chat-hf",
        help="Base model to fine-tune"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="C:/Users/Prithvi Putta/prometheus/trained_models",
        help="Output directory for trained models"
    )

    # Training hyperparameters
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=32,
        help="LoRA rank parameter"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size (Tinker only)"
    )

    # Local-specific options
    parser.add_argument(
        "--memory_mode",
        type=str,
        choices=["low", "medium", "high"],
        default="medium",
        help="Memory preset for local training"
    )
    parser.add_argument(
        "--no_quantization",
        action="store_true",
        help="Disable quantization (local only)"
    )

    # Tinker-specific options
    parser.add_argument(
        "--download_model",
        action="store_true",
        default=True,
        help="Download model after Tinker training"
    )
    parser.add_argument(
        "--test_inference",
        action="store_true",
        help="Test inference after training"
    )

    # Logging
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="prometheus-train",
        help="Experiment name for logging"
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Enable Weights & Biases logging (local only)"
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default="",
        help="W&B entity name"
    )

    args = parser.parse_args()

    # Initialize manager
    manager = PrometheusTrainingManager()

    # Interactive mode
    if args.interactive:
        return manager.interactive_mode()

    # Print status
    manager.print_status()

    # Determine training mode
    if args.mode == "auto":
        args.mode = manager.recommend_mode()
        if args.mode == "none":
            print("❌ No training backend available")
            print("\nPlease either:")
            print("  1. Set up Tinker API (recommended)")
            print("     → Visit: https://thinkingmachines.ai/tinker/")
            print("  2. Install CUDA GPU for local training")
            sys.exit(1)
        print(f"🎯 Auto-selected mode: {args.mode.upper()}\n")

    # Validate mode
    if args.mode == "tinker" and not manager.has_tinker:
        print("❌ Tinker API not available")
        print("Please set TINKER_API_KEY or install tinker-sdk")
        sys.exit(1)

    if args.mode == "local" and not manager.has_gpu:
        print("❌ No GPU available for local training")
        sys.exit(1)

    # Execute training
    if args.mode == "local":
        return manager.train_local(args)
    elif args.mode == "tinker":
        return manager.train_tinker(args)


if __name__ == "__main__":
    sys.exit(main())
