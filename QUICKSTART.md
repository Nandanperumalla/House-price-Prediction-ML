# 🏠 House Price Prediction Project - Quick Start Guide

## 🚀 Quick Start

### Option 1: Automated Setup (Recommended)

**For macOS/Linux:**
```bash
chmod +x setup.sh
./setup.sh
```

**For Windows:**
```cmd
setup.bat
```

### Option 2: Manual Setup

1. **Create virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the training pipeline:**
   ```bash
   python train.py
   ```

4. **Launch the web application:**
   ```bash
   streamlit run app.py
   ```

## 📁 Project Files Overview

| File | Description |
|------|-------------|
| `app.py` | 🌐 Streamlit web application |
| `train.py` | 🤖 Complete training pipeline |
| `predict.py` | 🎯 Prediction script |
| `requirements.txt` | 📦 Python dependencies |
| `README.md` | 📚 Comprehensive documentation |
| `setup.sh` / `setup.bat` | 🔧 Automated setup scripts |

### Source Code (`src/`)

| File | Description |
|------|-------------|
| `data_preprocessing.py` | 🧹 Data cleaning and feature engineering |
| `model_training.py` | 🤖 Model training and comparison |
| `evaluation.py` | 📊 Model evaluation and visualization |

### Notebooks (`notebooks/`)

| File | Description |
|------|-------------|
| `EDA.ipynb` | 📊 Exploratory Data Analysis |

## 🎯 Key Features

- ✅ **Complete ML Pipeline**: From data preprocessing to model deployment
- ✅ **Multiple Algorithms**: Linear Regression, Random Forest, XGBoost, and more
- ✅ **Interactive Web App**: Streamlit-based interface for predictions
- ✅ **Comprehensive EDA**: Jupyter notebook with detailed analysis
- ✅ **Model Evaluation**: Detailed metrics and visualizations
- ✅ **Feature Engineering**: Advanced feature creation and selection
- ✅ **Production Ready**: Modular code structure and documentation

## 🔧 Usage Examples

### 1. Train Models
```python
from src.model_training import ModelTrainer
from src.data_preprocessing import DataPreprocessor

# Load and preprocess data
preprocessor = DataPreprocessor()
X, y = preprocessor.preprocess_pipeline(df)

# Train models
trainer = ModelTrainer()
trainer.initialize_models()
trainer.train_models(X_train, y_train, X_test, y_test)
```

### 2. Make Predictions
```python
import joblib

# Load trained model
model = joblib.load('models/best_model.pkl')

# Make prediction
property_data = {
    'bedroomcnt': 3,
    'bathroomcnt': 2.5,
    'calculatedfinishedsquarefeet': 2000,
    'taxvaluedollarcnt': 500000,
    'yearbuilt': 2010,
    'regionidzip': 12345
}

prediction = model.predict(pd.DataFrame([property_data]))
```

### 3. Web Application
```bash
streamlit run app.py
```
Then navigate to `http://localhost:8501`

## 📊 Expected Results

- **RMSE**: ~0.11 (Root Mean Square Error)
- **MAE**: ~0.09 (Mean Absolute Error)
- **R² Score**: ~0.72 (Coefficient of Determination)
- **MAPE**: ~8.5% (Mean Absolute Percentage Error)

## 🛠️ Customization

### Adding New Features
1. Modify `data_preprocessing.py` to add new feature engineering functions
2. Update the preprocessing pipeline in `preprocess_pipeline()` method
3. Retrain models with new features

### Adding New Models
1. Add new model to `initialize_models()` in `model_training.py`
2. Include hyperparameter tuning if needed
3. Update model comparison logic

### Customizing Web App
1. Modify `app.py` to add new pages or features
2. Update the UI components and styling
3. Add new visualizations using Plotly

## 🚨 Troubleshooting

### Common Issues

1. **ModuleNotFoundError**: Make sure virtual environment is activated
2. **FileNotFoundError**: Ensure all required files are in correct locations
3. **Memory Issues**: Reduce dataset size or use data sampling
4. **Slow Training**: Use fewer estimators or enable parallel processing

### Getting Help

- Check the comprehensive `README.md` for detailed documentation
- Review the Jupyter notebook `EDA.ipynb` for data analysis examples
- Examine the source code in `src/` for implementation details

## 🎉 Next Steps

1. **Add Real Data**: Replace sample data with actual Zillow dataset
2. **Improve Models**: Experiment with different algorithms and hyperparameters
3. **Enhance Features**: Add more property and location-based features
4. **Deploy**: Deploy the web application to cloud platforms
5. **Monitor**: Add model monitoring and retraining capabilities

---

**Happy Predicting! 🏠📈**
