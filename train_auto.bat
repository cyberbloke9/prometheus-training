@echo off
REM Unified Training Script - Auto-detects best method
REM Windows Batch File

echo ========================================
echo Prometheus Unified Training Interface
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

echo Detecting available training methods...
echo.

python prometheus_train.py ^
    --data_path ../sample_train_data.json ^
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
pause
