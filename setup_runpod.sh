#!/bin/bash
# Setup script for RunPod H100 training
# Run this once after cloning the repo

set -e  # Exit on error

echo "=========================================="
echo " Prometheus Training Setup for RunPod"
echo "=========================================="
echo ""

# Step 1: Install dependencies
echo "📦 Step 1: Installing Python dependencies..."
pip install -r requirements.txt

# Step 2: Download training data
echo ""
echo "📥 Step 2: Downloading training data..."
python download_training_data.py

# Step 3: Set environment variables
echo ""
echo "⚙️  Step 3: Setting environment variables..."
export WANDB_MODE=disabled
echo "   WANDB_MODE=disabled"

echo ""
echo "✅ Setup complete!"
echo ""
echo "To start training, run:"
echo "   cd train"
echo "   python llama_finetuning.py"
echo ""
echo "=========================================="
