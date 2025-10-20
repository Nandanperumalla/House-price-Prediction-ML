#!/bin/bash

# Setup script for House Price Prediction Project

echo "🏠 House Price Prediction - Setup Script"
echo "========================================"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

echo "✅ Python3 found: $(python3 --version)"

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📚 Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "✅ Setup completed successfully!"
echo ""
echo "🚀 To get started:"
echo "1. Activate the virtual environment: source venv/bin/activate"
echo "2. Run the training script: python train.py"
echo "3. Launch the web app: streamlit run app.py"
echo "4. Open Jupyter notebook: jupyter notebook notebooks/EDA.ipynb"
echo ""
echo "Happy coding! 🎉"
