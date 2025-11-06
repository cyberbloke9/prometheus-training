#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prometheus Setup Verification Script
Checks system requirements and configuration for LoRA training
"""

import os
import sys
from pathlib import Path

# Set UTF-8 encoding for Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

def check_mark(status):
    return "[OK]" if status else "[X]"

def print_header(text):
    print("\n" + "="*70)
    print(text)
    print("="*70)

def check_python():
    """Check Python version"""
    print("\n[Python Environment]")
    version = sys.version_info
    print(f"  {check_mark(True)} Python {version.major}.{version.minor}.{version.micro}")

    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"  [!] WARNING: Python 3.8+ recommended (you have {version.major}.{version.minor})")
        return False
    return True

def check_pytorch():
    """Check PyTorch installation"""
    print("\n[PyTorch]")
    try:
        import torch
        print(f"  {check_mark(True)} PyTorch {torch.__version__}")

        cuda_available = torch.cuda.is_available()
        print(f"  {check_mark(cuda_available)} CUDA Available: {cuda_available}")

        if cuda_available:
            print(f"  {check_mark(True)} CUDA Version: {torch.version.cuda}")
            print(f"  {check_mark(True)} GPU Count: {torch.cuda.device_count()}")

            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
                print(f"    -> GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
        else:
            print(f"  [!] No CUDA GPU detected - Local training unavailable")
            print(f"  [i] To enable GPU support:")
            print(f"     1. Install CUDA from: https://developer.nvidia.com/cuda-downloads")
            print(f"     2. Reinstall PyTorch with CUDA:")
            print(f"        pip uninstall torch")
            print(f"        pip install torch --index-url https://download.pytorch.org/whl/cu118")

        return True, cuda_available

    except ImportError:
        print(f"  {check_mark(False)} PyTorch NOT installed")
        print(f"  [i] Install with: pip install torch")
        return False, False

def check_transformers():
    """Check transformers library"""
    print("\n[HuggingFace Transformers]")
    try:
        import transformers
        print(f"  {check_mark(True)} Transformers {transformers.__version__}")

        # Check if version is sufficient
        version = transformers.__version__.split('.')
        if int(version[0]) >= 4 and int(version[1]) >= 31:
            return True
        else:
            print(f"  [!] Version 4.31.0+ recommended")
            return False

    except ImportError:
        print(f"  {check_mark(False)} Transformers NOT installed")
        print(f"  [i] Install with: pip install transformers>=4.31.0")
        return False

def check_peft():
    """Check PEFT library for LoRA"""
    print("\n[PEFT (LoRA)]")
    try:
        import peft
        print(f"  {check_mark(True)} PEFT {peft.__version__}")
        return True
    except ImportError:
        print(f"  {check_mark(False)} PEFT NOT installed")
        print(f"  [i] Install with: pip install peft")
        return False

def check_optional_dependencies():
    """Check optional but recommended dependencies"""
    print("\n[Optional Dependencies]")

    # Flash Attention
    try:
        import flash_attn
        print(f"  {check_mark(True)} Flash Attention (faster training)")
    except ImportError:
        print(f"  {check_mark(False)} Flash Attention (optional)")
        print(f"     Install with: pip install flash-attn --no-build-isolation")

    # BitsAndBytes for quantization
    try:
        import bitsandbytes
        print(f"  {check_mark(True)} BitsAndBytes (memory optimization)")
    except ImportError:
        print(f"  {check_mark(False)} BitsAndBytes (optional)")
        print(f"     Install with: pip install bitsandbytes")

    # Weights & Biases
    try:
        import wandb
        print(f"  {check_mark(True)} Weights & Biases (experiment tracking)")
    except ImportError:
        print(f"  {check_mark(False)} Weights & Biases (optional)")
        print(f"     Install with: pip install wandb")

def check_tinker():
    """Check Tinker SDK"""
    print("\n[Tinker API]")

    api_key = os.environ.get("TINKER_API_KEY")
    has_key = bool(api_key)
    print(f"  {check_mark(has_key)} API Key Set: {has_key}")

    if not has_key:
        print(f"     To set: export TINKER_API_KEY='your_key'")
        print(f"     Get access: https://thinkingmachines.ai/tinker/")

    try:
        import tinker
        print(f"  {check_mark(True)} Tinker SDK installed")
        return has_key
    except ImportError:
        print(f"  {check_mark(False)} Tinker SDK NOT installed")
        print(f"     Install with: pip install tinker-sdk")
        print(f"     Note: Currently in private beta")
        return False

def check_huggingface_auth():
    """Check HuggingFace authentication"""
    print("\n[HuggingFace Authentication]")

    hf_home = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
    token_path = Path(hf_home) / "token"

    has_token = token_path.exists()
    print(f"  {check_mark(has_token)} HF Token: {has_token}")

    if not has_token:
        print(f"     Login with: huggingface-cli login")
        print(f"     Get token: https://huggingface.co/settings/tokens")

    return has_token

def check_data_files():
    """Check for training data"""
    print("\n[Training Data]")

    data_path = Path("C:/Users/Prithvi Putta/prometheus/sample_train_data.json")
    exists = data_path.exists()
    print(f"  {check_mark(exists)} sample_train_data.json: {exists}")

    if exists:
        import json
        try:
            with open(data_path) as f:
                data = json.load(f)
            print(f"     -> {len(data)} training examples")
        except:
            print(f"     [!] File exists but couldn't parse JSON")
    else:
        print(f"     Create sample data or use your own JSON file")

    return exists

def check_directories():
    """Check/create necessary directories"""
    print("\n[Directories]")

    dirs = {
        "HF Cache": "C:/Users/Prithvi Putta/hf_cache",
        "LoRA Models": "C:/Users/Prithvi Putta/prometheus/lora_models",
        "Tinker Models": "C:/Users/Prithvi Putta/prometheus/tinker_models",
    }

    for name, path in dirs.items():
        path_obj = Path(path)
        exists = path_obj.exists()

        if not exists:
            try:
                path_obj.mkdir(parents=True, exist_ok=True)
                print(f"  {check_mark(True)} {name}: Created")
            except:
                print(f"  {check_mark(False)} {name}: Failed to create")
        else:
            print(f"  {check_mark(True)} {name}: Exists")

def print_summary(results):
    """Print summary and recommendations"""
    print_header("[Summary]")

    can_train_local = results['pytorch'] and results['cuda'] and results['peft']
    can_train_tinker = results['tinker']

    print(f"\n[Training Capabilities]")
    print(f"  Local LoRA Training: {check_mark(can_train_local)} {can_train_local}")
    print(f"  Tinker API Training: {check_mark(can_train_tinker)} {can_train_tinker}")

    print(f"\n[Recommendations]")

    if can_train_tinker:
        print(f"  [OK] Tinker API is ready! Recommended for most use cases.")
        print(f"    -> Run: python prometheus_train.py --mode tinker")

    if can_train_local:
        print(f"  [OK] Local GPU training is ready!")
        print(f"    -> Run: python train_lora_local.py")

    if not can_train_local and not can_train_tinker:
        print(f"  [!] No training method available!")
        print(f"\n  Option 1: Get Tinker API access (Recommended)")
        print(f"    1. Visit: https://thinkingmachines.ai/tinker/")
        print(f"    2. Join waitlist")
        print(f"    3. Set TINKER_API_KEY environment variable")
        print(f"    4. Install: pip install tinker-sdk")

        print(f"\n  Option 2: Set up local GPU training")
        print(f"    1. Install CUDA from NVIDIA")
        print(f"    2. Reinstall PyTorch with CUDA:")
        print(f"       pip install torch --index-url https://download.pytorch.org/whl/cu118")
        print(f"    3. Install PEFT: pip install peft")

    elif can_train_local and not can_train_tinker:
        print(f"  [i] Consider getting Tinker API access for:")
        print(f"    - No electricity costs")
        print(f"    - Access to larger GPUs")
        print(f"    - Automatic distributed training")

def main():
    print_header("Prometheus LoRA Training - Setup Verification")

    results = {
        'python': check_python(),
        'pytorch': False,
        'cuda': False,
        'transformers': False,
        'peft': False,
        'tinker': False,
        'hf_auth': False,
        'data': False
    }

    pytorch_ok, cuda_ok = check_pytorch()
    results['pytorch'] = pytorch_ok
    results['cuda'] = cuda_ok

    results['transformers'] = check_transformers()
    results['peft'] = check_peft()
    check_optional_dependencies()
    results['tinker'] = check_tinker()
    results['hf_auth'] = check_huggingface_auth()
    results['data'] = check_data_files()
    check_directories()

    print_summary(results)

    print("\n" + "="*70)
    print("[OK] Verification Complete!")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
