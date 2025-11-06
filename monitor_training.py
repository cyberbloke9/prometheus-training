#!/usr/bin/env python3
"""
Live Training Monitor for Prometheus
Displays real-time training progress, GPU usage, and loss curves
"""

import os
import time
import subprocess
from pathlib import Path

def get_gpu_stats():
    """Get GPU utilization and memory usage"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=2
        )
        if result.returncode == 0:
            gpu_util, mem_used, mem_total, temp = result.stdout.strip().split(', ')
            return {
                'gpu_util': int(gpu_util),
                'mem_used': int(mem_used),
                'mem_total': int(mem_total),
                'temp': int(temp)
            }
    except:
        pass
    return None

def read_training_log():
    """Read the latest training progress from log file"""
    log_path = Path("C:/Users/Prithvi Putta/prometheus/train/training_log.txt")
    if not log_path.exists():
        return None

    try:
        with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()

        # Look for training progress lines
        latest_info = {
            'epoch': None,
            'step': None,
            'loss': None,
            'lr': None
        }

        for line in reversed(lines[-50:]):  # Check last 50 lines
            if 'Epoch' in line and 'Step' in line:
                # Parse training progress
                if 'Loss' in line:
                    parts = line.split('|')
                    for part in parts:
                        if 'Epoch' in part:
                            latest_info['epoch'] = part.split(':')[1].strip() if ':' in part else None
                        elif 'Step' in part:
                            latest_info['step'] = part.split(':')[1].strip() if ':' in part else None
                        elif 'Loss' in part:
                            latest_info['loss'] = part.split(':')[1].strip() if ':' in part else None
                        elif 'LR' in part:
                            latest_info['lr'] = part.split(':')[1].strip() if ':' in part else None
                if latest_info['loss']:
                    break

        return latest_info
    except:
        return None

def clear_screen():
    """Clear terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

def draw_progress_bar(progress, width=40):
    """Draw a progress bar"""
    filled = int(width * progress)
    bar = '█' * filled + '░' * (width - filled)
    return f"[{bar}] {progress*100:.1f}%"

def main():
    print("Starting Prometheus Training Monitor...")
    print("Press Ctrl+C to exit\n")
    time.sleep(2)

    while True:
        try:
            clear_screen()

            # Header
            print("="*70)
            print(" "*20 + "PROMETHEUS TRAINING MONITOR")
            print("="*70)
            print()

            # GPU Stats
            gpu_stats = get_gpu_stats()
            if gpu_stats:
                print("GPU STATUS:")
                print(f"  Utilization: {gpu_stats['gpu_util']}%")
                print(f"  Memory: {gpu_stats['mem_used']} MB / {gpu_stats['mem_total']} MB " +
                      f"({gpu_stats['mem_used']/gpu_stats['mem_total']*100:.1f}%)")
                print(f"  Temperature: {gpu_stats['temp']}°C")

                # GPU progress bar
                gpu_progress = gpu_stats['gpu_util'] / 100
                print(f"  {draw_progress_bar(gpu_progress)}")

                # Memory progress bar
                mem_progress = gpu_stats['mem_used'] / gpu_stats['mem_total']
                print(f"  {draw_progress_bar(mem_progress)}")
            else:
                print("GPU STATUS: Unable to fetch")

            print()
            print("-"*70)
            print()

            # Training Progress
            train_info = read_training_log()
            if train_info and train_info['loss']:
                print("TRAINING PROGRESS:")
                if train_info['epoch']:
                    print(f"  Epoch: {train_info['epoch']}")
                if train_info['step']:
                    print(f"  Step: {train_info['step']}")
                if train_info['loss']:
                    print(f"  Loss: {train_info['loss']}")
                if train_info['lr']:
                    print(f"  Learning Rate: {train_info['lr']}")
            else:
                print("TRAINING PROGRESS:")
                print("  Waiting for training to start...")
                print("  Model is being downloaded or initializing...")

            print()
            print("-"*70)
            print()

            # Instructions
            print("MONITORING:")
            print("  This screen updates every 5 seconds")
            print("  Training log: C:/Users/Prithvi Putta/prometheus/train/training_log.txt")
            print("  Press Ctrl+C to exit monitor (training continues)")

            print()
            print("="*70)

            # Update every 5 seconds
            time.sleep(5)

        except KeyboardInterrupt:
            print("\n\nMonitor stopped. Training continues in background.")
            break
        except Exception as e:
            print(f"\nError: {e}")
            time.sleep(5)

if __name__ == "__main__":
    main()
