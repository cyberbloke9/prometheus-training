#!/usr/bin/env python3
"""
Simple script to login to HuggingFace
"""

from huggingface_hub import login

print("="*60)
print("HuggingFace Login")
print("="*60)
print("\nPlease paste your HuggingFace token below.")
print("(Get it from: https://huggingface.co/settings/tokens)")
print()

token = input("Enter your HuggingFace token: ").strip()

if not token:
    print("\nError: No token provided!")
    exit(1)

print("\nAuthenticating with HuggingFace...")

try:
    login(token=token, add_to_git_credential=True)
    print("\n" + "="*60)
    print("SUCCESS! You are now authenticated with HuggingFace!")
    print("="*60)
    print("\nYou can now start training:")
    print("  cd C:\\Users\\Prithvi Putta\\prometheus\\train")
    print("  python train_lora_local.py --data_path ../prometheus_formatted_data/prometheus_train.json")
    print()
except Exception as e:
    print(f"\nError during authentication: {e}")
    exit(1)
