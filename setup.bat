@echo off
REM Setup script for House Price Prediction Project (Windows)

echo 🏠 House Price Prediction - Setup Script
echo ========================================

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python is not installed. Please install Python 3.8+ first.
    pause
    exit /b 1
)

echo ✅ Python found
python --version

REM Create virtual environment
echo 📦 Creating virtual environment...
python -m venv venv

REM Activate virtual environment
echo 🔧 Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo ⬆️ Upgrading pip...
python -m pip install --upgrade pip

REM Install dependencies
echo 📚 Installing dependencies...
pip install -r requirements.txt

echo.
echo ✅ Setup completed successfully!
echo.
echo 🚀 To get started:
echo 1. Activate the virtual environment: venv\Scripts\activate.bat
echo 2. Run the training script: python train.py
echo 3. Launch the web app: streamlit run app.py
echo 4. Open Jupyter notebook: jupyter notebook notebooks\EDA.ipynb
echo.
echo Happy coding! 🎉
pause
