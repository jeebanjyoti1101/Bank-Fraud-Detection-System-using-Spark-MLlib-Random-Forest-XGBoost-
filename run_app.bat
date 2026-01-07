@echo off
REM Bank Fraud Detection System - Windows Launch Script

echo ========================================
echo Bank Fraud Detection System
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.7 or higher
    pause
    exit /b 1
)

REM Check if in correct directory
if not exist "app\app.py" (
    echo ERROR: Please run this script from the fraud-detection directory
    echo Current directory should contain app\app.py
    pause
    exit /b 1
)

REM Check if virtual environment exists
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment
        pause
        exit /b 1
    )
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install requirements
echo Installing/updating requirements...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install requirements
    pause
    exit /b 1
)

REM Check if models exist
if not exist "models\rf_model.pkl" (
    echo.
    echo WARNING: Trained models not found!
    echo Please train models first:
    echo Run: python scripts\train.py
    echo.
    echo Press any key to continue anyway...
    pause >nul
)

REM Set environment variables
set FLASK_ENV=development
set FLASK_DEBUG=True

REM Start the Flask application
echo.
echo Starting Flask application...
echo Open your browser to: http://localhost:5000
echo Press Ctrl+C to stop the server
echo.

cd app
python app.py
