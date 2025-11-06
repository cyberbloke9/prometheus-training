# Prometheus LoRA Training Guide

Complete guide for training Prometheus with LoRA using local hardware or Tinker API.

## 📋 Table of Contents

- [Quick Start](#quick-start)
- [Setup Instructions](#setup-instructions)
- [Training Methods](#training-methods)
- [Configuration Options](#configuration-options)
- [Troubleshooting](#troubleshooting)
- [Cost Comparison](#cost-comparison)

## 🚀 Quick Start

### Option 1: Unified Interface (Recommended)

```bash
cd C:\Users\Prithvi Putta\prometheus\train

# Auto-detect best training method
python prometheus_train.py --data_path ../sample_train_data.json

# Or use interactive wizard
python prometheus_train.py --interactive
```

### Option 2: Direct Training

**Local Training:**
```bash
python train_lora_local.py --data_path ../sample_train_data.json
```

**Tinker Training:**
```bash
export TINKER_API_KEY="your_api_key_here"
python train_tinker.py --data_path ../sample_train_data.json
```

## 🛠️ Setup Instructions

### 1. Install Dependencies

```bash
cd C:\Users\Prithvi Putta\prometheus

# Install base requirements
pip install -r requirements.txt

# For Flash Attention (optional but recommended)
pip install flash-attn --no-build-isolation

# For Tinker API (when you get access)
pip install tinker-sdk
```

### 2. Set up HuggingFace Authentication

```bash
# Install HuggingFace CLI
pip install huggingface_hub

# Login to access LLaMA models
huggingface-cli login
```

Enter your HuggingFace token from: https://huggingface.co/settings/tokens

### 3. Configure Environment

**For Local Training:**
```bash
# Windows CMD
set HF_HOME=C:\Users\Prithvi Putta\hf_cache

# Windows PowerShell
$env:HF_HOME="C:\Users\Prithvi Putta\hf_cache"
```

**For Tinker Training:**
```bash
# Set your Tinker API key
set TINKER_API_KEY=your_api_key_here
```

## 🎯 Training Methods

### Method 1: Local LoRA Training

**Recommended for:** Users with GPU hardware (RTX 3090, 4090, A100, etc.)

**Memory Requirements:**

| Memory Mode | VRAM Required | Model Size | LoRA Rank |
|-------------|---------------|------------|-----------|
| Low         | 8-16 GB       | 13B        | 16        |
| Medium      | 16-24 GB      | 13B        | 32        |
| High        | 40+ GB        | 13B        | 64        |

**Basic Usage:**

```bash
# Default (medium memory)
python train_lora_local.py \
    --data_path ../sample_train_data.json \
    --model_name meta-llama/Llama-2-13b-chat-hf

# Low memory mode
python train_lora_local.py \
    --memory_mode low \
    --data_path ../sample_train_data.json

# High performance mode
python train_lora_local.py \
    --memory_mode high \
    --data_path ../sample_train_data.json
```

**Custom Configuration:**

```bash
python train_lora_local.py \
    --data_path ../sample_train_data.json \
    --model_name meta-llama/Llama-2-13b-chat-hf \
    --lora_rank 32 \
    --lora_alpha 64 \
    --learning_rate 2e-4 \
    --num_epochs 3 \
    --batch_size 1 \
    --gradient_accumulation_steps 32 \
    --output_dir C:\Users\Prithvi Putta\prometheus\my_model
```

### Method 2: Tinker API Training

**Recommended for:** Everyone (especially without GPU or with limited compute)

**Advantages:**
- No GPU required
- No electricity costs
- Automatic distributed training
- Pay only for usage
- Access to large GPUs (A100, H100)

**Setup:**

1. Join Tinker waitlist: https://thinkingmachines.ai/tinker/
2. Get API key from Tinker console
3. Set environment variable:
   ```bash
   export TINKER_API_KEY="your_api_key"
   ```

**Basic Usage:**

```bash
python train_tinker.py \
    --data_path ../sample_train_data.json \
    --model meta-llama/Llama-2-13b-chat-hf
```

**With Model Download:**

```bash
python train_tinker.py \
    --data_path ../sample_train_data.json \
    --model meta-llama/Llama-2-13b-chat-hf \
    --download \
    --test_inference \
    --output_dir C:\Users\Prithvi Putta\prometheus\tinker_models
```

### Method 3: Unified Interface

**Recommended for:** Flexible workflow, switching between local and Tinker

**Auto-detect:**

```bash
# Automatically choose best method
python prometheus_train.py --data_path ../sample_train_data.json
```

**Force specific mode:**

```bash
# Force local training
python prometheus_train.py --data_path ../sample_train_data.json --mode local

# Force Tinker training
python prometheus_train.py --data_path ../sample_train_data.json --mode tinker
```

**Interactive wizard:**

```bash
python prometheus_train.py --interactive
```

## ⚙️ Configuration Options

### LoRA Hyperparameters

| Parameter | Default | Description | Impact |
|-----------|---------|-------------|--------|
| `lora_rank` | 32 | LoRA rank (r) | Higher = more capacity, more memory |
| `lora_alpha` | 64 | Scaling factor | Usually 2x rank |
| `lora_dropout` | 0.05 | Dropout rate | Regularization |
| `target_modules` | q,k,v,o | Which layers to train | More = better quality, more memory |

**Presets:**

```python
# Low memory (8-16GB)
lora_rank=16, target_modules=["q_proj", "v_proj"]

# Medium (16-24GB) - Recommended
lora_rank=32, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]

# High performance (40GB+)
lora_rank=64, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
```

### Training Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `learning_rate` | 2e-4 | Learning rate for LoRA |
| `num_epochs` | 3 | Number of training epochs |
| `batch_size` | 1 | Samples per GPU |
| `gradient_accumulation_steps` | 32 | Accumulate gradients (effective batch size = batch_size × this) |
| `max_length` | 2048 | Maximum sequence length |

### Memory Optimization Flags

```bash
# Enable all optimizations
python train_lora_local.py \
    --memory_mode low \
    --use_flash_attention \
    --quantization \
    --gradient_checkpointing
```

| Flag | Memory Savings | Speed Impact |
|------|----------------|--------------|
| `quantization` | 50% | ~10% slower |
| `use_flash_attention` | 30% | 15% faster |
| `gradient_checkpointing` | 40% | 20% slower |

## 🔧 Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory

**Error:** `RuntimeError: CUDA out of memory`

**Solutions:**
```bash
# Reduce batch size
--batch_size 1

# Increase gradient accumulation
--gradient_accumulation_steps 64

# Use lower memory mode
--memory_mode low

# Enable quantization
--quantization

# Use smaller model
--model_name meta-llama/Llama-2-7b-chat-hf
```

#### 2. Tinker API Key Not Found

**Error:** `Tinker API key not found`

**Solution:**
```bash
# Windows CMD
set TINKER_API_KEY=your_key_here

# Windows PowerShell
$env:TINKER_API_KEY="your_key_here"

# Or pass directly
python train_tinker.py --api_key your_key_here
```

#### 3. HuggingFace Authentication Required

**Error:** `Access denied for model`

**Solution:**
```bash
huggingface-cli login
# Enter token from https://huggingface.co/settings/tokens
```

#### 4. Flash Attention Not Available

**Error:** `flash_attn module not found`

**Solution:**
```bash
# Install Flash Attention
pip install flash-attn --no-build-isolation

# Or disable in training
--use_fast_kernels=False
```

#### 5. Wandb Entity Hardcoded

**Error:** `wandb authentication error for jshin49`

**Solution:**
```bash
# Disable W&B
set WANDB_MODE=disabled

# Or edit llama_finetuning.py lines 81 and 94
# Change entity="jshin49" to your W&B entity
```

## 💰 Cost Comparison

### Training 13B Model for 3 Epochs (10K steps)

| Method | Hardware Cost | Electricity | Time | Total Cost | Notes |
|--------|---------------|-------------|------|------------|-------|
| **Local Full Fine-Tuning** | $40,000 (4x A100) | $27/3 days | 12 hrs | $40,027 | One-time hardware |
| **Local LoRA (RTX 4090)** | $1,600 | $1.50/2 days | 48 hrs | $1,601.50 | Reusable hardware |
| **Local LoRA + Quantization** | $1,200 (RTX 3090) | $1.00/2 days | 60 hrs | $1,201 | Lower-end GPU |
| **Tinker API** | $0 | $0 | 12-24 hrs | ~$1,900 | Pay-per-use |
| **Google Colab Pro+** | $50/month | $0 | 24 hrs | $50/month | Temporary access |

**Recommendations:**

- **One-time training:** Use Tinker API ($1,900)
- **Frequent training:** Local LoRA with RTX 4090 ($1,600 + $1.50/training)
- **Experimentation:** Colab Pro+ ($50/month unlimited)
- **Production:** Local setup for <$2,000 with reusable hardware

## 📊 Expected Results

### Training Metrics

**Typical LoRA Training (3 epochs):**

- Initial Loss: ~2.0-2.5
- Final Loss: ~0.5-1.0
- Training Time: 12-48 hours (depending on hardware)
- Model Size: ~200-500 MB (LoRA adapters only)

### Quality Comparison

| Method | Quality | Training Time | Cost | Memory |
|--------|---------|---------------|------|--------|
| Full Fine-Tuning | 100% | 12 hrs | $27 | 80GB |
| LoRA (r=64) | 95-98% | 18 hrs | $1.50 | 24GB |
| LoRA (r=32) | 90-95% | 24 hrs | $1.20 | 18GB |
| LoRA (r=16) | 85-92% | 36 hrs | $1.00 | 14GB |

**Recommendation:** LoRA r=32 offers best quality/cost tradeoff

## 🎓 Best Practices

### Data Preparation

1. **Format your data** in Prometheus format:
```json
[
  {
    "instruction": "###Task Description:\\n...\\n###Feedback: ",
    "input": "",
    "output": "Feedback text [RESULT] 4"
  }
]
```

2. **Data Quality > Quantity:**
   - 1,000 high-quality examples > 10,000 noisy examples
   - Ensure consistent scoring rubrics
   - Validate reference answers are truly score 5

3. **Data Split:**
   - Training: 80-90%
   - Validation: 10-20%
   - Keep validation data representative

### Training Strategy

1. **Start Small:**
   - Begin with 7B model to test pipeline
   - Use low memory mode for initial experiments
   - Validate on small subset first

2. **Hyperparameter Tuning:**
   - Learning rate: Try [1e-4, 2e-4, 3e-4]
   - LoRA rank: Try [16, 32, 64]
   - Batch size: Maximize within memory limits

3. **Monitoring:**
   - Watch training loss (should decrease)
   - Check for overfitting (validation loss increasing)
   - Sample generations periodically

### Deployment

1. **Merge LoRA weights** (optional):
```python
from peft import PeftModel
from transformers import LlamaForCausalLM

base_model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-13b-chat-hf")
model = PeftModel.from_pretrained(base_model, "path/to/lora")
merged_model = model.merge_and_unload()
merged_model.save_pretrained("path/to/merged_model")
```

2. **Test thoroughly:**
   - Run on validation set
   - Compare with original Prometheus
   - Check score distributions

3. **Optimize inference:**
   - Use TGI (Text Generation Inference) server
   - Enable Flash Attention
   - Consider quantization (int8, int4)

## 📚 Additional Resources

- **Prometheus Paper:** https://arxiv.org/abs/2310.08491
- **LoRA Paper:** https://arxiv.org/abs/2106.09685
- **Tinker Docs:** https://tinker-docs.thinkingmachines.ai/
- **HuggingFace Docs:** https://huggingface.co/docs/peft/

## 🤝 Support

- **Issues:** File at GitHub repository
- **Questions:** Use discussion board
- **Tinker Support:** https://thinkingmachines.ai/tinker/

---

**Last Updated:** 2025-11-03
