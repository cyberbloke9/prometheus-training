#!/usr/bin/env python3
"""
Download and prepare publicly available datasets for Prometheus training
"""

import os
import json
from datasets import load_dataset
from pathlib import Path

print("="*70)
print("Downloading Publicly Available Evaluation Datasets")
print("="*70)

# Create data directory
data_dir = Path("C:/Users/Prithvi Putta/prometheus/training_data")
data_dir.mkdir(exist_ok=True)

datasets_info = []

# 1. Prometheus Feedback Collection (Original dataset from paper)
print("\n[1/4] Downloading Prometheus Feedback Collection...")
try:
    dataset = load_dataset("kaist-ai/Feedback-Collection", split="train")
    print(f"  -> Loaded {len(dataset)} examples")

    # Save to JSON
    output_path = data_dir / "feedback_collection.json"
    dataset.to_json(output_path)
    print(f"  -> Saved to: {output_path}")

    datasets_info.append({
        "name": "Feedback Collection",
        "source": "kaist-ai/Feedback-Collection",
        "samples": len(dataset),
        "path": str(output_path),
        "description": "Original Prometheus training dataset with evaluation tasks"
    })
except Exception as e:
    print(f"  -> Error: {e}")

# 2. UltraFeedback (Instruction-response pairs with quality ratings)
print("\n[2/4] Downloading UltraFeedback...")
try:
    dataset = load_dataset("openbmb/UltraFeedback", split="train")
    print(f"  -> Loaded {len(dataset)} examples")

    # Take a subset (it's very large)
    dataset = dataset.select(range(min(10000, len(dataset))))
    print(f"  -> Using subset of {len(dataset)} examples")

    output_path = data_dir / "ultrafeedback.json"
    dataset.to_json(output_path)
    print(f"  -> Saved to: {output_path}")

    datasets_info.append({
        "name": "UltraFeedback",
        "source": "openbmb/UltraFeedback",
        "samples": len(dataset),
        "path": str(output_path),
        "description": "Instruction-response pairs with detailed quality ratings"
    })
except Exception as e:
    print(f"  -> Error: {e}")

# 3. HH-RLHF (Anthropic's Human Preference Dataset)
print("\n[3/4] Downloading HH-RLHF (Helpful & Harmless)...")
try:
    dataset = load_dataset("Anthropic/hh-rlhf", split="train")
    print(f"  -> Loaded {len(dataset)} examples")

    # Take a subset
    dataset = dataset.select(range(min(5000, len(dataset))))
    print(f"  -> Using subset of {len(dataset)} examples")

    output_path = data_dir / "hh_rlhf.json"
    dataset.to_json(output_path)
    print(f"  -> Saved to: {output_path}")

    datasets_info.append({
        "name": "HH-RLHF",
        "source": "Anthropic/hh-rlhf",
        "samples": len(dataset),
        "path": str(output_path),
        "description": "Human preference data for helpful and harmless responses"
    })
except Exception as e:
    print(f"  -> Error: {e}")

# 4. AlpacaEval (Instruction following evaluation)
print("\n[4/4] Downloading AlpacaEval...")
try:
    dataset = load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval", split="eval")
    print(f"  -> Loaded {len(dataset)} examples")

    output_path = data_dir / "alpaca_eval.json"
    dataset.to_json(output_path)
    print(f"  -> Saved to: {output_path}")

    datasets_info.append({
        "name": "AlpacaEval",
        "source": "tatsu-lab/alpaca_eval",
        "samples": len(dataset),
        "path": str(output_path),
        "description": "Instruction following evaluation dataset"
    })
except Exception as e:
    print(f"  -> Error: {e}")

# Save dataset info
print("\n" + "="*70)
print("Download Summary")
print("="*70)

summary_path = data_dir / "datasets_info.json"
with open(summary_path, 'w') as f:
    json.dump(datasets_info, f, indent=2)

print(f"\nDownloaded {len(datasets_info)} datasets:")
for info in datasets_info:
    print(f"\n  {info['name']}:")
    print(f"    Source: {info['source']}")
    print(f"    Samples: {info['samples']}")
    print(f"    Path: {info['path']}")
    print(f"    Description: {info['description']}")

print(f"\nDataset info saved to: {summary_path}")
print("\nNext step: Convert these datasets to Prometheus format")
print("="*70)
