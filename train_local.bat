@echo off
REM Quick Launch Script for Local LoRA Training
REM Windows Batch File

echo ========================================
echo Prometheus Local LoRA Training
echo ========================================
echo.

cd /d "%~dp0train"

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found!
    echo Please install Python 3.8+ and add to PATH
    pause
    exit /b 1
)

REM Check for CUDA
python -c "import torch; assert torch.cuda.is_available()" >nul 2>&1
if errorlevel 1 (
    echo WARNING: CUDA not detected!
    echo Local training requires NVIDIA GPU with CUDA
    echo.
    choice /C YN /M "Continue anyway? "
    if errorlevel 2 exit /b 1
)

REM Disable W&B by default
set WANDB_MODE=disabled

REM Set HuggingFace cache directory
set HF_HOME=C:\Users\Prithvi Putta\hf_cache

echo.
echo Configuration:
echo   Mode: Local LoRA Training
echo   Memory: Medium (16-24GB VRAM)
echo   Data: sample_train_data.json
echo.
echo Starting training...
echo.

python train_lora_local.py ^
    --data_path ../sample_train_data.json ^
    --memory_mode medium ^
    --num_epochs 3 ^
    --learning_rate 2e-4

if errorlevel 1 (
    echo.
    echo ========================================
    echo Training FAILED!
    echo ========================================
    pause
    exit /b 1
)

echo.
echo ========================================
echo Training COMPLETED Successfully!
echo ========================================
echo.
echo Model saved to: C:\Users\Prithvi Putta\prometheus\lora_models
echo.
pause
