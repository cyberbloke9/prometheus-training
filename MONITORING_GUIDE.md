# Prometheus Training - Live Monitoring Guide

## ✅ Training Started Successfully!

Your Prometheus LoRA training is now running in the background. The LLaMA-2-13B model is being downloaded (~26GB).

---

## 📊 4 Ways to Monitor Training

### **Method 1: Live Monitor Dashboard (Recommended)**

Open a **new terminal** and run:

```bash
cd C:\Users\Prithvi Putta\prometheus
python monitor_training.py
```

**Features:**
- Real-time GPU usage and memory
- Current training step and loss
- Temperature monitoring
- Updates every 5 seconds
- Press Ctrl+C to exit (training continues)

---

### **Method 2: GPU Monitoring with nvidia-smi**

Open a **new terminal** and run:

```bash
# Update every 1 second
nvidia-smi -l 1
```

**Shows:**
- GPU utilization %
- Memory usage
- Temperature
- Running processes

---

### **Method 3: Check Training Log File**

```bash
# View latest training output
Get-Content C:\Users\Prithvi Putta\prometheus\train\training_log.txt -Tail 50 -Wait
```

Or open in Notepad and refresh:
```
C:\Users\Prithvi Putta\prometheus\train\training_log.txt
```

---

### **Method 4: PowerShell Real-time Tail**

```powershell
# Windows PowerShell
Get-Content C:\Users\Prithvi Putta\prometheus\train\training_log.txt -Tail 50 -Wait
```

---

## 📈 What to Expect

### **Phase 1: Model Download (10-30 minutes)**
```
Fetching 3 files: 0%|          | 0/3
Downloading model files...
```
- LLaMA-2-13B model: ~26GB
- Downloads to: C:\Users\Prithvi Putta\.cache\huggingface

### **Phase 2: Model Loading & Initialization (5-10 minutes)**
```
Loading checkpoint shards...
Applying LoRA adapters...
Preparing 8-bit quantization...
```
- Model loads into GPU memory (~10-11GB)
- LoRA adapters initialized

### **Phase 3: Training Loop (24-36 hours)**
```
Epoch 1/3 | Step 100/1772 | Loss: 1.234 | LR: 0.0002
Epoch 1/3 | Step 200/1772 | Loss: 1.156 | LR: 0.0002
```
- 18,900 training examples
- 591 steps per epoch with batch size 32
- 1,772 total steps for 3 epochs

---

## 🔍 Training Progress Indicators

### **Loss Values:**
- **Initial (Step 0-100)**: 2.0 - 2.5 (expected)
- **Early (Step 100-500)**: 1.5 - 2.0 (decreasing)
- **Mid (Step 500-1000)**: 1.0 - 1.5 (steady decrease)
- **Late (Step 1000+)**: 0.6 - 1.0 (converging)

### **GPU Usage:**
- **Utilization**: Should be 90-100% during training
- **Memory**: ~10-11GB / 12.9GB
- **Temperature**: 60-80°C normal, <85°C safe

### **Training Speed:**
- **Steps per hour**: ~40-60 steps/hour expected
- **Time per epoch**: 8-12 hours
- **Total time**: 24-36 hours for 3 epochs

---

## 🎯 Key Checkpoints

Training automatically saves checkpoints every 100 steps to:
```
C:\Users\Prithvi Putta\prometheus\lora_models\
```

**Checkpoint structure:**
```
lora_models/
├── adapter_config.json
├── adapter_model.bin  (~200-500MB)
└── training_log.txt
```

---

## ⚠️ What to Watch For

### **Good Signs:**
- ✅ GPU utilization 90-100%
- ✅ Loss decreasing over time
- ✅ Memory usage stable around 10-11GB
- ✅ Temperature under 85°C
- ✅ Regular checkpoint saves

### **Warning Signs:**
- ⚠️ Loss increasing or not decreasing
- ⚠️ GPU utilization < 50% (stuck?)
- ⚠️ Temperature > 85°C (cooling issue)
- ⚠️ Memory usage = 12.9GB (might OOM)

### **Error Conditions:**
- ❌ "CUDA out of memory" → Reduce batch size or use low memory mode
- ❌ Process stopped → Check training_log.txt for errors
- ❌ Loss = NaN → Learning rate too high

---

## 🛠️ Training Control Commands

### **Check if Training is Running:**
```bash
tasklist | findstr python
```

### **Monitor GPU:**
```bash
nvidia-smi
```

### **View Recent Log:**
```bash
Get-Content C:\Users\Prithvi Putta\prometheus\train\training_log.txt -Tail 20
```

### **Stop Training (if needed):**
```bash
# Find process ID
tasklist | findstr python

# Kill specific process
taskkill /PID <process_id> /F
```

---

## 📱 Quick Status Check Script

Create `check_training.bat`:

```batch
@echo off
echo ========================================
echo Prometheus Training Status
echo ========================================
echo.

echo GPU Status:
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader

echo.
echo Latest Training Output:
powershell -Command "Get-Content C:\Users\Prithvi Putta\prometheus\train\training_log.txt -Tail 10"

echo.
echo ========================================
pause
```

---

## 📊 Expected Timeline

| Time | Progress | What's Happening |
|------|----------|------------------|
| 0-30 min | Setup | Model download (26GB) |
| 30-45 min | Init | Model loading & initialization |
| 1-12 hrs | Epoch 1 | First training pass, loss drops |
| 12-24 hrs | Epoch 2 | Refinement, continued learning |
| 24-36 hrs | Epoch 3 | Final tuning, convergence |
| **36 hrs** | **Done** | **Training complete!** |

---

## 🎉 When Training Completes

You'll see:
```
============================================================
 Training completed successfully!
 Model saved to: C:/Users/Prithvi Putta/prometheus/lora_models
============================================================
```

**Next steps:**
1. Find your model at: `C:\Users\Prithvi Putta\prometheus\lora_models`
2. Test inference with your trained model
3. Evaluate on validation set
4. Deploy for production use

---

## 💡 Pro Tips

1. **Let it run overnight**: Training takes 24-36 hours
2. **Don't close the terminal**: Training runs in background
3. **Monitor periodically**: Check every few hours
4. **Keep GPU cool**: Ensure good ventilation
5. **Save checkpoints**: Training saves every 100 steps
6. **Check logs regularly**: Watch for errors or issues

---

## 🔗 Useful Links

- Training logs: `C:\Users\Prithvi Putta\prometheus\train\training_log.txt`
- Model output: `C:\Users\Prithvi Putta\prometheus\lora_models\`
- Monitoring script: `C:\Users\Prithvi Putta\prometheus\monitor_training.py`

---

**Training started:** November 4, 2025
**Estimated completion:** November 5-6, 2025
**Total training examples:** 18,900
**Model:** LLaMA-2-13B with LoRA
