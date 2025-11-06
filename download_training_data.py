#!/usr/bin/env python3
"""
Download and prepare Prometheus training data from HuggingFace
Run this script on RunPod before starting training
"""

import os
import json
from datasets import load_dataset
from pathlib import Path

def download_and_prepare_data():
    """Download Feedback Collection dataset and format for Prometheus training"""

    print("=" * 60)
    print("Downloading Prometheus Training Data")
    print("=" * 60)

    # Create directories
    data_dir = Path("/workspace/prometheus/prometheus_formatted_data")
    data_dir.mkdir(parents=True, exist_ok=True)

    output_file = data_dir / "prometheus_train.json"

    # Check if already downloaded
    if output_file.exists():
        print(f"\n✓ Data already exists at {output_file}")
        with open(output_file, 'r') as f:
            data = json.load(f)
        print(f"✓ Found {len(data)} training examples")
        return

    print("\n📥 Downloading Feedback Collection dataset from HuggingFace...")
    print("   This will download ~21,000 examples (~770MB)")

    # Download dataset
    dataset = load_dataset("kaist-ai/Feedback-Collection", split="train[:21000]")

    print(f"✓ Downloaded {len(dataset)} examples")

    # Convert to Prometheus format
    print("\n🔄 Converting to Prometheus training format...")
    formatted_data = []

    for item in dataset:
        formatted_item = {
            "instruction": item["instruction"],
            "output": item["output"]
        }
        formatted_data.append(formatted_item)

    # Save to JSON
    print(f"\n💾 Saving to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(formatted_data, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Training data ready!")
    print(f"   Location: {output_file}")
    print(f"   Examples: {len(formatted_data)}")
    print(f"   Size: {output_file.stat().st_size / (1024*1024):.1f} MB")
    print("\n" + "=" * 60)

if __name__ == "__main__":
    download_and_prepare_data()
