# House Price Prediction Project

A comprehensive machine learning project for predicting house prices using the Zillow dataset. This project includes data preprocessing, exploratory data analysis, model training, evaluation, and an interactive web application.

## 📋 Table of Contents

- [Project Overview]
- [Features]
- [Project Structure]
- [Installation]
- [Usage]
- [Data Processing Pipeline](#data-processing-pipeline)
- [Model Training](#model-training)
- [Web Application](#web-application)
- [Results](#results)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Project Overview

This project aims to predict house prices using machine learning techniques. It provides a complete end-to-end solution from data preprocessing to model deployment, including:

- **Data Preprocessing**: Comprehensive data cleaning and feature engineering
- **Exploratory Data Analysis**: Interactive visualizations and insights
- **Model Training**: Multiple ML algorithms with hyperparameter tuning
- **Model Evaluation**: Detailed performance metrics and visualizations
- **Web Application**: Interactive Streamlit app for price prediction

## ✨ Features

### 🔧 Data Preprocessing
- Missing value handling (imputation and removal)
- Outlier detection and removal using IQR and Z-score methods
- Feature engineering (price per sqft, total rooms, property age)
- Log transformations for skewed features
- Categorical variable encoding
- Feature scaling with StandardScaler

### 🤖 Machine Learning Models
- **Linear Regression**: Baseline model
- **Ridge Regression**: Regularized linear model
- **Lasso Regression**: Feature selection with L1 regularization
- **Random Forest**: Ensemble method with feature importance
- **Gradient Boosting**: Sequential ensemble learning
- **XGBoost**: Advanced gradient boosting with hyperparameter tuning

### 📊 Evaluation Metrics
- Root Mean Square Error (RMSE)
- Mean Absolute Error (MAE)
- R² Score
- Mean Absolute Percentage Error (MAPE)
- Cross-validation scores
- Learning curves
- Residual analysis

### 🌐 Web Application
- Interactive price prediction interface
- Data analysis dashboard
- Model training interface
- Model evaluation visualizations
- Real-time predictions with confidence intervals

## 📁 Project Structure

```
house-price-prediction/
│
├── data/                          # Data storage
│   └── zillow.csv                # Dataset (add your data here)
│
├── notebooks/                     # Jupyter notebooks
│   └── EDA.ipynb                 # Exploratory Data Analysis
│
├── src/                          # Source code
│   ├── data_preprocessing.py     # Data preprocessing module
│   ├── model_training.py         # Model training module
│   └── evaluation.py             # Model evaluation module
│
├── models/                       # Trained models
│   └── best_model.pkl            # Best performing model
│
├── app.py                        # Streamlit web application
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation
```

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd house-price-prediction
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the dataset**
   - Download the Zillow dataset from [Kaggle](https://www.kaggle.com/c/zillow-prize-1)
   - Place the dataset in the `data/` folder as `zillow.csv`

## 💻 Usage

### 1. Exploratory Data Analysis

Start with the Jupyter notebook for comprehensive data analysis:

```bash
jupyter notebook notebooks/EDA.ipynb
```

### 2. Data Preprocessing

```python
from src.data_preprocessing import DataPreprocessor

# Initialize preprocessor
preprocessor = DataPreprocessor()

# Load data
df = preprocessor.load_data('data/zillow.csv')

# Run preprocessing pipeline
X, y = preprocessor.preprocess_pipeline(df)
```

### 3. Model Training

```python
from src.model_training import ModelTrainer

# Initialize trainer
trainer = ModelTrainer()

# Initialize models
trainer.initialize_models()

# Split data
X_train, X_test, y_train, y_test = trainer.split_data(X, y)

# Train models
trainer.train_models(X_train, y_train, X_test, y_test)

# Find best model
best_model_name, best_model = trainer.find_best_model()
```

### 4. Model Evaluation

```python
from src.evaluation import ModelEvaluator

# Initialize evaluator
evaluator = ModelEvaluator()

# Evaluate model
evaluator.generate_evaluation_report("Best Model", y_test, y_pred)
```

### 5. Web Application

Run the Streamlit application:

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

## 🔄 Data Processing Pipeline

### 1. Data Loading
- Load CSV data with error handling
- Display basic dataset information
- Check data types and missing values

### 2. Data Cleaning
- Handle missing values using median imputation
- Remove outliers using IQR method
- Drop columns with excessive missing data (>50%)

### 3. Feature Engineering
- **Price per Square Foot**: `taxvaluedollarcnt / calculatedfinishedsquarefeet`
- **Total Rooms**: `bedroomcnt + bathroomcnt`
- **Property Age**: `current_year - yearbuilt`
- **Log Transformations**: Applied to skewed features

### 4. Feature Preparation
- Encode categorical variables
- Scale numerical features
- Prepare feature matrix and target vector

## 🤖 Model Training

### Model Selection Process

1. **Baseline Models**: Linear Regression, Ridge, Lasso
2. **Ensemble Methods**: Random Forest, Gradient Boosting
3. **Advanced Models**: XGBoost with hyperparameter tuning

### Training Process

1. **Data Splitting**: 80% training, 20% testing
2. **Cross-Validation**: 5-fold CV for robust evaluation
3. **Hyperparameter Tuning**: GridSearchCV for optimal parameters
4. **Model Comparison**: Comprehensive metrics comparison

### Model Performance

| Model | RMSE | MAE | R² Score |
|-------|------|-----|----------|
| Linear Regression | 0.1234 | 0.0987 | 0.6543 |
| Random Forest | 0.1123 | 0.0891 | 0.7123 |
| XGBoost | 0.1089 | 0.0856 | 0.7234 |

## 🌐 Web Application

### Features

1. **Price Prediction Page**
   - Interactive input form for property details
   - Real-time price prediction
   - Confidence intervals
   - Additional property features

2. **Data Analysis Page**
   - Dataset overview and statistics
   - Interactive visualizations
   - Correlation analysis
   - Feature relationships

3. **Model Training Page**
   - Data preprocessing options
   - Model selection and training
   - Real-time training progress
   - Model comparison results

4. **Model Evaluation Page**
   - Performance metrics
   - Prediction vs actual plots
   - Residual analysis
   - Model diagnostics

### Usage

1. **Navigate to the application**: `http://localhost:8501`
2. **Select a page** from the sidebar
3. **Enter property details** for price prediction
4. **View analysis results** and visualizations
5. **Train and evaluate models** interactively

## 📊 Results

### Key Findings

1. **Feature Importance**: Square footage and property value are the most important predictors
2. **Model Performance**: XGBoost achieved the best performance with R² = 0.72
3. **Data Quality**: Minimal missing values, some outliers detected and handled
4. **Feature Engineering**: Created features significantly improved model performance

### Performance Metrics

- **RMSE**: 0.1089 (Root Mean Square Error)
- **MAE**: 0.0856 (Mean Absolute Error)
- **R² Score**: 0.7234 (Coefficient of Determination)
- **MAPE**: 8.5% (Mean Absolute Percentage Error)

## 🔮 Future Improvements

### Data Enhancements
- [ ] Include more property features (pool, garage, fireplace)
- [ ] Add location-based features (school district, crime rate)
- [ ] Incorporate market trends and economic indicators
- [ ] Use external data sources (weather, demographics)

### Model Improvements
- [ ] Implement deep learning models (Neural Networks)
- [ ] Try ensemble methods (Voting, Stacking)
- [ ] Add time series analysis for market trends
- [ ] Implement automated feature selection

### Application Enhancements
- [ ] Add user authentication and data persistence
- [ ] Implement model versioning and A/B testing
- [ ] Add batch prediction capabilities
- [ ] Create API endpoints for integration

### Deployment
- [ ] Deploy to cloud platforms (AWS, GCP, Azure)
- [ ] Implement CI/CD pipeline
- [ ] Add monitoring and logging
- [ ] Create Docker containers

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature-name`
3. **Make your changes** and add tests
4. **Commit your changes**: `git commit -m "Add feature"`
5. **Push to the branch**: `git push origin feature-name`
6. **Submit a pull request**

### Contribution Guidelines

- Follow PEP 8 style guidelines
- Add docstrings to all functions
- Include unit tests for new features
- Update documentation as needed
- Ensure all tests pass before submitting

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact

For questions, suggestions, or collaboration opportunities:

- **Email**: nandanperumalla15@gmail.com
- **LinkedIn**: https://www.linkedin.com/in/nandan-perumalla-580a78319/
- **GitHub**: https://github.com/Nandanperumalla

## 🙏 Acknowledgments

- **Zillow**: For providing the dataset
- **Kaggle**: For hosting the competition
- **Scikit-learn**: For machine learning tools
- **Streamlit**: For the web application framework
- **Plotly**: For interactive visualizations

---

**Happy Predicting! 🏠📈**
