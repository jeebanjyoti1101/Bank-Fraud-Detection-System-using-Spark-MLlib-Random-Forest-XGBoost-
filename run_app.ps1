# Bank Fraud Detection System - PowerShell Launch Script

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Bank Fraud Detection System" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if Python is installed
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✅ Python detected: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ ERROR: Python is not installed or not in PATH" -ForegroundColor Red
    Write-Host "Please install Python 3.7 or higher" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# Check if in correct directory
if (-not (Test-Path "app\app.py")) {
    Write-Host "❌ ERROR: Please run this script from the fraud-detection directory" -ForegroundColor Red
    Write-Host "Current directory should contain app\app.py" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# Check if virtual environment exists
if (-not (Test-Path "venv")) {
    Write-Host "📦 Creating virtual environment..." -ForegroundColor Yellow
    python -m venv venv
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ ERROR: Failed to create virtual environment" -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }
    Write-Host "✅ Virtual environment created" -ForegroundColor Green
}

# Activate virtual environment
Write-Host "🔄 Activating virtual environment..." -ForegroundColor Yellow
& "venv\Scripts\Activate.ps1"

# Install requirements
Write-Host "📦 Installing/updating requirements..." -ForegroundColor Yellow
pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ ERROR: Failed to install requirements" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}
Write-Host "✅ Requirements installed" -ForegroundColor Green

# Check if models exist
if (-not (Test-Path "models\feature_columns.json")) {
    Write-Host ""
    Write-Host "⚠️  WARNING: Trained models not found!" -ForegroundColor Yellow
    Write-Host "Please train models first:" -ForegroundColor Yellow
    Write-Host "1. Use Google Colab notebook: notebooks\train_on_colab.ipynb" -ForegroundColor Cyan
    Write-Host "2. Or train locally: python scripts\train_spark.py" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Press any key to continue anyway..." -ForegroundColor Yellow
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
}

# Set environment variables
$env:FLASK_ENV = "development"
$env:FLASK_DEBUG = "True"

# Start the Flask application
Write-Host ""
Write-Host "🚀 Starting Flask application..." -ForegroundColor Green
Write-Host "🌐 Open your browser to: http://localhost:5000" -ForegroundColor Cyan
Write-Host "⏹️  Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host ""

Set-Location app
python app.py
