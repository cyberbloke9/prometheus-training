#!/usr/bin/env python3
"""
Prometheus Training with Tinker API
This script integrates Prometheus training with Thinking Machines Lab's Tinker API
for distributed LoRA fine-tuning without managing infrastructure.

Prerequisites:
    1. Join Tinker waitlist: https://thinkingmachines.ai/tinker/
    2. Get API key from Tinker console
    3. Set environment variable: export TINKER_API_KEY="your_api_key"
    4. Install Tinker SDK: pip install tinker-sdk

Usage:
    # Basic training
    python train_tinker.py --data_path ./sample_train_data.json

    # With custom model and LoRA config
    python train_tinker.py --model meta-llama/Llama-2-13b-chat-hf --lora_rank 32

    # Monitor training
    python train_tinker.py --data_path ./data.json --monitor
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import time

try:
    import tinker
    TINKER_AVAILABLE = True
except ImportError:
    TINKER_AVAILABLE = False
    print("⚠️  Warning: tinker-sdk not installed. Install with: pip install tinker-sdk")


class PrometheusDatasetConverter:
    """Convert Prometheus training data format to Tinker-compatible format"""

    @staticmethod
    def load_prometheus_data(data_path: str) -> List[Dict]:
        """Load Prometheus training data from JSON file"""
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data

    @staticmethod
    def convert_to_tinker_format(prometheus_data: List[Dict]) -> List[Dict]:
        """
        Convert Prometheus format to standard instruction-response format
        Tinker expects: {"instruction": str, "output": str}
        """
        converted = []
        for item in prometheus_data:
            converted.append({
                "instruction": item.get("instruction", ""),
                "input": item.get("input", ""),
                "output": item.get("output", "")
            })
        return converted

    @staticmethod
    def save_tinker_format(data: List[Dict], output_path: str):
        """Save converted data in Tinker-compatible format"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"✓ Converted data saved to: {output_path}")


class TinkerPrometheusTrainer:
    """Prometheus training interface using Tinker API"""

    def __init__(self, api_key: Optional[str] = None):
        if not TINKER_AVAILABLE:
            raise ImportError(
                "Tinker SDK not installed. Install with: pip install tinker-sdk\n"
                "Note: Tinker is currently in private beta. "
                "Request access at https://thinkingmachines.ai/tinker/"
            )

        # Get API key from environment or parameter
        self.api_key = api_key or os.environ.get("TINKER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Tinker API key not found. Set TINKER_API_KEY environment variable "
                "or pass api_key parameter."
            )

        # Initialize Tinker service client
        self.service_client = tinker.ServiceClient(api_key=self.api_key)
        self.training_client = None
        self.sampling_client = None

    def create_training_client(
        self,
        base_model: str,
        lora_rank: int = 32,
        lora_alpha: Optional[int] = None,
        target_modules: Optional[List[str]] = None
    ):
        """
        Create LoRA training client

        Args:
            base_model: HuggingFace model ID (e.g., "meta-llama/Llama-2-13b-chat-hf")
            lora_rank: LoRA rank parameter
            lora_alpha: LoRA alpha (defaults to 2 * rank)
            target_modules: List of modules to apply LoRA (defaults to Tinker's selection)
        """
        if lora_alpha is None:
            lora_alpha = 2 * lora_rank

        print(f"\n🔧 Creating Tinker training client...")
        print(f"   Model: {base_model}")
        print(f"   LoRA Rank: {lora_rank}")
        print(f"   LoRA Alpha: {lora_alpha}")

        config = {
            "base_model": base_model,
            "rank": lora_rank,
        }

        # Add optional parameters if provided
        if lora_alpha:
            config["alpha"] = lora_alpha
        if target_modules:
            config["target_modules"] = target_modules

        self.training_client = self.service_client.create_lora_training_client(**config)
        print("✓ Training client created successfully")
        return self.training_client

    def train(
        self,
        data_path: str,
        num_epochs: int = 3,
        learning_rate: float = 2e-4,
        batch_size: int = 4,
        max_length: int = 2048,
        warmup_steps: int = 100,
        save_steps: int = 500,
        experiment_name: str = "prometheus-tinker"
    ):
        """
        Execute training loop using Tinker API

        Args:
            data_path: Path to training data (Prometheus format JSON)
            num_epochs: Number of training epochs
            learning_rate: Learning rate
            batch_size: Batch size per device
            max_length: Maximum sequence length
            warmup_steps: Number of warmup steps
            save_steps: Save checkpoint every N steps
            experiment_name: Name for this training run
        """
        if not self.training_client:
            raise ValueError("Training client not initialized. Call create_training_client() first.")

        # Convert and load data
        print("\n📊 Loading and converting training data...")
        converter = PrometheusDatasetConverter()
        prometheus_data = converter.load_prometheus_data(data_path)
        print(f"✓ Loaded {len(prometheus_data)} training examples")

        # Convert to Tinker format
        tinker_data_path = data_path.replace(".json", "_tinker_format.json")
        tinker_data = converter.convert_to_tinker_format(prometheus_data)
        converter.save_tinker_format(tinker_data, tinker_data_path)

        print("\n" + "="*60)
        print("🚀 Starting Tinker Training")
        print("="*60)
        print(f"Experiment: {experiment_name}")
        print(f"Epochs: {num_epochs}")
        print(f"Learning Rate: {learning_rate}")
        print(f"Batch Size: {batch_size}")
        print(f"Max Length: {max_length}")
        print("="*60 + "\n")

        # Training loop using Tinker primitives
        # Note: This is a conceptual implementation based on Tinker documentation
        # Actual implementation may vary based on final Tinker API

        try:
            step = 0
            total_steps = len(prometheus_data) * num_epochs // batch_size

            for epoch in range(num_epochs):
                print(f"\n📈 Epoch {epoch + 1}/{num_epochs}")

                for batch_idx in range(0, len(prometheus_data), batch_size):
                    batch = prometheus_data[batch_idx:batch_idx + batch_size]

                    # Forward and backward pass
                    # This uses Tinker's forward_backward primitive
                    loss = self.training_client.forward_backward(
                        batch=batch,
                        max_length=max_length
                    )

                    # Optimization step
                    if (step + 1) % 1 == 0:  # Update every step (adjust for gradient accumulation)
                        self.training_client.optim_step(
                            learning_rate=learning_rate,
                            warmup_steps=warmup_steps,
                            current_step=step
                        )

                    # Logging
                    if step % 10 == 0:
                        print(f"  Step {step}/{total_steps} | Loss: {loss:.4f}")

                    # Checkpointing
                    if (step + 1) % save_steps == 0:
                        print(f"\n💾 Saving checkpoint at step {step}...")
                        self.training_client.save_state(
                            checkpoint_name=f"{experiment_name}_step_{step}"
                        )

                    step += 1

            print("\n✅ Training completed successfully!")

        except Exception as e:
            print(f"\n❌ Training failed: {e}")
            raise

    def save_and_create_sampling_client(self, model_name: str):
        """Save trained weights and create sampling client for inference"""
        print("\n💾 Saving final model weights...")
        self.sampling_client = self.training_client.save_weights_and_get_sampling_client(
            name=model_name
        )
        print(f"✓ Model saved as: {model_name}")
        return self.sampling_client

    def download_model(self, output_path: str):
        """Download trained model weights"""
        if not self.sampling_client:
            raise ValueError("No sampling client available. Train and save model first.")

        print(f"\n⬇️  Downloading model to: {output_path}")
        rest_client = self.service_client.create_rest_client()

        # Download checkpoint archive
        future = rest_client.download_checkpoint_archive_from_tinker_path(
            self.sampling_client.model_path
        )

        # Wait for download to complete
        archive_path = future.result()
        print(f"✓ Model downloaded to: {archive_path}")

        # Extract if needed
        if archive_path.endswith('.tar.gz'):
            import tarfile
            with tarfile.open(archive_path, 'r:gz') as tar:
                tar.extractall(output_path)
            print(f"✓ Model extracted to: {output_path}")

        return output_path

    def test_inference(self, prompt: str, max_tokens: int = 256):
        """Test model with a sample prompt"""
        if not self.sampling_client:
            raise ValueError("No sampling client available. Train and save model first.")

        print(f"\n🧪 Testing inference...")
        print(f"Prompt: {prompt[:100]}...")

        result = self.sampling_client.sample(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=1.0,
            top_p=0.9
        )

        print(f"\nGenerated: {result}")
        return result


def main():
    parser = argparse.ArgumentParser(
        description="Train Prometheus using Tinker API",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Required arguments
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to Prometheus training data (JSON format)"
    )

    # Model and LoRA configuration
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-2-13b-chat-hf",
        help="Base model to fine-tune"
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=32,
        help="LoRA rank parameter"
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=None,
        help="LoRA alpha (defaults to 2 * rank)"
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
        "--batch_size",
        type=int,
        default=4,
        help="Batch size per device"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=2048,
        help="Maximum sequence length"
    )

    # Experiment settings
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="prometheus-tinker",
        help="Experiment name"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="C:/Users/Prithvi Putta/prometheus/tinker_models",
        help="Directory to download trained model"
    )

    # API settings
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="Tinker API key (or set TINKER_API_KEY env var)"
    )

    # Actions
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download trained model after training"
    )
    parser.add_argument(
        "--test_inference",
        action="store_true",
        help="Test model with sample inference after training"
    )

    args = parser.parse_args()

    # Check API key
    api_key = args.api_key or os.environ.get("TINKER_API_KEY")
    if not api_key:
        print("❌ ERROR: Tinker API key not found!")
        print("\nPlease either:")
        print("  1. Set environment variable: export TINKER_API_KEY='your_key'")
        print("  2. Pass --api_key argument")
        print("\nTo get an API key:")
        print("  1. Join waitlist: https://thinkingmachines.ai/tinker/")
        print("  2. Get API key from Tinker console")
        sys.exit(1)

    print("\n" + "="*60)
    print("🚀 Prometheus Training with Tinker API")
    print("="*60)

    try:
        # Initialize trainer
        trainer = TinkerPrometheusTrainer(api_key=api_key)

        # Create training client
        trainer.create_training_client(
            base_model=args.model,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha
        )

        # Train model
        trainer.train(
            data_path=args.data_path,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            max_length=args.max_length,
            experiment_name=args.experiment_name
        )

        # Save model
        model_name = f"{args.experiment_name}_{int(time.time())}"
        trainer.save_and_create_sampling_client(model_name)

        # Test inference if requested
        if args.test_inference:
            test_prompt = """###Task Description:
An instruction (might include an Input inside it), a response to evaluate, a reference answer that gets a score of 5, and a score rubric representing a evaluation criteria are given.
1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score that is an integer between 1 and 5.

###The instruction to evaluate:
Explain what machine learning is in simple terms.

###Response to evaluate:
Machine learning is when computers learn from data.

###Reference Answer (Score 5):
Machine learning is a branch of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed. It uses algorithms to analyze data, identify patterns, and make decisions or predictions.

###Score Rubrics:
[Clarity and Completeness]
Score 1: Incomplete or unclear explanation
Score 2: Basic explanation with some clarity issues
Score 3: Clear explanation covering main concepts
Score 4: Comprehensive and clear explanation
Score 5: Exceptionally clear and thorough explanation

###Feedback: """
            trainer.test_inference(test_prompt)

        # Download model if requested
        if args.download:
            os.makedirs(args.output_dir, exist_ok=True)
            trainer.download_model(args.output_dir)

        print("\n" + "="*60)
        print("✅ All tasks completed successfully!")
        print("="*60 + "\n")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
