#!/usr/bin/env python3
"""
Simple script to login to HuggingFace
Usage: python hf_login_simple.py YOUR_TOKEN_HERE
"""

import sys
from huggingface_hub import login

if len(sys.argv) < 2:
    print("="*60)
    print("HuggingFace Login")
    print("="*60)
    print("\nUsage:")
    print("  python hf_login_simple.py YOUR_TOKEN_HERE")
    print("\nGet your token from: https://huggingface.co/settings/tokens")
    print()
    sys.exit(1)

token = sys.argv[1].strip()

print("\n" + "="*60)
print("Authenticating with HuggingFace...")
print("="*60)

try:
    login(token=token, add_to_git_credential=True)
    print("\nSUCCESS! You are now authenticated!")
    print("="*60)
    print("\nYour token has been saved. You can now:")
    print("  1. Access LLaMA-2 models")
    print("  2. Start training")
    print("\nNext step:")
    print("  cd C:\\Users\\Prithvi Putta\\prometheus\\train")
    print("  set WANDB_MODE=disabled")
    print("  python train_lora_local.py --data_path ../prometheus_formatted_data/prometheus_train.json")
    print()
except Exception as e:
    print(f"\nError: {e}")
    sys.exit(1)
