# Prometheus LoRA Training - READY TO START!

## ✅ Setup Complete

All systems are ready for Prometheus LoRA training!

### System Status

**Hardware:**
- GPU: NVIDIA GeForce RTX 3060
- VRAM: 12.9 GB
- Driver: 560.94
- CUDA: 12.6

**Software:**
- Python: 3.12.6
- PyTorch: 2.5.1+cu121 (CUDA-enabled)
- CUDA Available: ✅ Yes
- PEFT: 0.17.1
- Transformers: 4.53.0
- Datasets: 4.3.0

### Training Data Ready

**Combined Dataset:**
- Training: 18,900 examples
- Validation: 2,100 examples
- Total: 21,000 examples

**Sources:**
1. Feedback Collection (10,000) - Official Prometheus dataset
2. UltraFeedback (5,000) - Quality-rated instructions
3. HH-RLHF (6,000) - Human preference data

**Location:**
- Train: `C:\Users\Prithvi Putta\prometheus\prometheus_formatted_data\prometheus_train.json`
- Val: `C:\Users\Prithvi Putta\prometheus\prometheus_formatted_data\prometheus_val.json`

---

## 🚀 How to Start Training

### Option 1: Interactive Mode (Recommended)

```bash
cd C:\Users\Prithvi Putta\prometheus\train
python prometheus_train.py --interactive
```

The wizard will guide you through configuration.

### Option 2: Direct Training with Defaults

```bash
cd C:\Users\Prithvi Putta\prometheus\train
python train_lora_local.py \
    --data_path ../prometheus_formatted_data/prometheus_train.json \
    --memory_mode medium \
    --num_epochs 3 \
    --learning_rate 2e-4
```

### Option 3: Quick Launch (Windows)

Double-click: `train_local.bat`

---

## 📊 Recommended Settings for RTX 3060 (12GB)

### Medium Memory Mode (Recommended)
- Batch Size: 1
- Gradient Accumulation: 32
- LoRA Rank: 32
- Effective Batch: 32
- Estimated Time: 24-36 hours
- Memory Usage: ~10-11GB

### Conservative Mode (Safer)
- Batch Size: 1
- Gradient Accumulation: 64
- LoRA Rank: 16
- Effective Batch: 64
- Estimated Time: 36-48 hours
- Memory Usage: ~8-9GB

### Configuration
```bash
python train_lora_local.py \
    --data_path ../prometheus_formatted_data/prometheus_train.json \
    --memory_mode medium \
    --model_name meta-llama/Llama-2-13b-chat-hf \
    --num_epochs 3 \
    --learning_rate 2e-4 \
    --lora_rank 32 \
    --output_dir ../trained_models/prometheus_lora \
    --experiment_name prometheus-rtx3060
```

---

## 📋 Training Checklist

Before starting:

- [x] GPU detected and working
- [x] CUDA-enabled PyTorch installed
- [x] Training data prepared (18,900 examples)
- [x] PEFT (LoRA) library installed
- [x] Output directories created
- [ ] HuggingFace authentication (optional but recommended)
- [ ] Enough disk space (~30GB for model + checkpoints)

### Optional: HuggingFace Login

```bash
pip install huggingface_hub
huggingface-cli login
```

This is needed to download LLaMA models from HuggingFace.
Get token from: https://huggingface.co/settings/tokens

---

## 🎯 Expected Training Timeline

**For 3 Epochs on RTX 3060:**

| Phase | Duration | What Happens |
|-------|----------|--------------|
| Setup | 5-10 min | Model download (26GB), initialization |
| Epoch 1 | 8-12 hrs | Initial training, loss drops rapidly |
| Epoch 2 | 8-12 hrs | Refinement, loss continues decreasing |
| Epoch 3 | 8-12 hrs | Final tuning, convergence |
| **Total** | **24-36 hrs** | Complete training |

**Progress Indicators:**
- Initial Loss: ~2.0-2.5
- Target Loss: ~0.5-1.0
- Checkpoints saved every 100 steps

---

## 💡 Training Tips

### 1. Monitor GPU Usage

Open another terminal:
```bash
nvidia-smi -l 1
```

This updates GPU stats every second.

### 2. Expected Memory Usage

- Model Loading: ~4-5GB
- Training: ~9-11GB peak
- If OOM Error: Reduce to `--memory_mode low`

### 3. Checkpoint Management

Models saved to: `C:\Users\Prithvi Putta\prometheus\lora_models`

Each checkpoint ~200-500MB (LoRA adapters only)

### 4. Interrupt and Resume

- Safe to Ctrl+C to stop
- Can resume from checkpoints (not auto-enabled)
- Consider adding `--resume_from_checkpoint` if needed

---

## 🔍 Monitoring Training

### Check Training Progress

Training outputs:
- Step number
- Loss value (should decrease)
- Learning rate
- GPU memory usage

Example output:
```
Step 100/1772 | Loss: 1.234 | LR: 0.0002 | GPU: 10.2GB
Step 200/1772 | Loss: 1.156 | LR: 0.0002 | GPU: 10.2GB
```

### Loss Expectations

- Epoch 1: 2.0 → 1.2
- Epoch 2: 1.2 → 0.8
- Epoch 3: 0.8 → 0.6

---

## 📦 After Training

### 1. Test Your Model

```python
from peft import PeftModel
from transformers import LlamaForCausalLM, LlamaTokenizer

# Load base model
base_model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-13b-chat-hf")

# Load LoRA adapters
model = PeftModel.from_pretrained(
    base_model,
    "C:/Users/Prithvi Putta/prometheus/lora_models"
)

# Test evaluation
tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-13b-chat-hf")
# ... run inference
```

### 2. Merge Weights (Optional)

```python
merged_model = model.merge_and_unload()
merged_model.save_pretrained("C:/Users/Prithvi Putta/prometheus/merged_model")
```

### 3. Deploy for Inference

Use Text Generation Inference (TGI) or similar server for production.

---

## 🚨 Troubleshooting

### CUDA Out of Memory

**Solution:**
```bash
# Reduce memory usage
python train_lora_local.py --memory_mode low
```

### Model Download Fails

**Solution:**
```bash
# Login to HuggingFace first
huggingface-cli login
```

### Training Too Slow

**Current Speed:** ~40-50 steps/hour expected
**To Speed Up:**
- Reduce gradient accumulation
- Use smaller model (Llama-2-7b)
- Reduce max_length to 1024

### W&B Error

**Solution:**
```bash
# Disable Weights & Biases
set WANDB_MODE=disabled
```

---

## 📞 Support

If you encounter issues:

1. Check `setup_verification.py` output
2. Review `LORA_TRAINING_GUIDE.md`
3. Check GPU with `nvidia-smi`
4. Verify data format with sample

---

## ✨ You're Ready!

Everything is set up. To start training:

```bash
cd C:\Users\Prithvi Putta\prometheus\train
python prometheus_train.py --interactive
```

**Good luck with your training!** 🚀

---

**Setup completed:** 2025-11-03
**System:** RTX 3060 12GB
**Training data:** 18,900 examples ready
**Estimated time:** 24-36 hours for 3 epochs
