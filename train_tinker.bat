@echo off
REM Quick Launch Script for Tinker API Training
REM Windows Batch File

echo ========================================
echo Prometheus Tinker API Training
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

REM Check for Tinker API key
if "%TINKER_API_KEY%"=="" (
    echo ERROR: TINKER_API_KEY not set!
    echo.
    echo Please set your Tinker API key:
    echo   set TINKER_API_KEY=your_api_key_here
    echo.
    echo To get an API key:
    echo   1. Visit: https://thinkingmachines.ai/tinker/
    echo   2. Join waitlist and get access
    echo   3. Get API key from Tinker console
    echo.
    pause
    exit /b 1
)

REM Check if Tinker SDK is installed
python -c "import tinker" >nul 2>&1
if errorlevel 1 (
    echo ERROR: Tinker SDK not installed!
    echo.
    echo Install with: pip install tinker-sdk
    echo.
    pause
    exit /b 1
)

echo.
echo Configuration:
echo   Mode: Tinker API (Distributed Training)
echo   Data: sample_train_data.json
echo   Model: Llama-2-13b-chat-hf
echo.
echo Starting training...
echo.

python train_tinker.py ^
    --data_path ../sample_train_data.json ^
    --model meta-llama/Llama-2-13b-chat-hf ^
    --num_epochs 3 ^
    --learning_rate 2e-4 ^
    --download ^
    --test_inference

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
echo Model downloaded to: C:\Users\Prithvi Putta\prometheus\tinker_models
echo.
pause
