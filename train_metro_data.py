"""
Training Pipeline for Metropolitan Area House Price Prediction

This script works with metropolitan area-level Zillow data instead of individual property data.
"""

import pandas as pd
import numpy as np
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_preprocessing import DataPreprocessor
from model_training import ModelTrainer
from evaluation import ModelEvaluator
import joblib

def load_metro_data():
    """
    Load and process metropolitan area Zillow data.
    
    Returns:
        pd.DataFrame: Processed metro area data
    """
    print("📊 Loading Metropolitan Area Zillow Data...")
    
    # Check for available data files
    data_files = {
        'zhvi': 'data/Metro_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv',
        'zori': 'data/Metro_zori_uc_sfrcondomfr_sm_month.csv',
        'zhvf_growth': 'data/Metro_zhvf_growth_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv'
    }
    
    available_files = {}
    for name, path in data_files.items():
        if os.path.exists(path):
            available_files[name] = path
            print(f"✅ Found {name}: {path}")
        else:
            print(f"❌ Missing {name}: {path}")
    
    if not available_files:
        print("❌ No metro data files found!")
        return None
    
    try:
        # Load the main ZHVI (Zillow Home Value Index) data
        if 'zhvi' in available_files:
            print("Loading ZHVI data...")
            zhvi_df = pd.read_csv(available_files['zhvi'])
            print(f"ZHVI data shape: {zhvi_df.shape}")
            
            # Process the time series data
            df = process_metro_time_series(zhvi_df, 'zhvi')
            
            # Add other datasets if available
            if 'zori' in available_files:
                print("Adding ZORI (rental) data...")
                zori_df = pd.read_csv(available_files['zori'])
                zori_processed = process_metro_time_series(zori_df, 'zori')
                df = df.merge(zori_processed, on=['RegionID', 'RegionName'], how='left', suffixes=('', '_zori'))
            
            if 'zhvf_growth' in available_files:
                print("Adding growth data...")
                growth_df = pd.read_csv(available_files['zhvf_growth'])
                growth_processed = process_metro_time_series(growth_df, 'growth')
                df = df.merge(growth_processed, on=['RegionID', 'RegionName'], how='left', suffixes=('', '_growth'))
            
            print(f"Final processed data shape: {df.shape}")
            return df
            
    except Exception as e:
        print(f"❌ Error loading metro data: {e}")
        return None

def process_metro_time_series(df, data_type):
    """
    Process metropolitan area time series data into features.
    
    Args:
        df: Raw metro data DataFrame
        data_type: Type of data ('zhvi', 'zori', 'growth')
        
    Returns:
        pd.DataFrame: Processed data with features
    """
    print(f"Processing {data_type} time series data...")
    
    # Get date columns (all columns that look like dates)
    date_columns = [col for col in df.columns if col.startswith('20')]
    
    if not date_columns:
        print(f"❌ No date columns found in {data_type} data")
        return df[['RegionID', 'RegionName']].copy()
    
    # Convert date columns to numeric
    numeric_data = df[date_columns].apply(pd.to_numeric, errors='coerce')
    
    # Create features from time series
    features = pd.DataFrame()
    features['RegionID'] = df['RegionID']
    features['RegionName'] = df['RegionName']
    
    # Basic statistics
    features[f'{data_type}_mean'] = numeric_data.mean(axis=1)
    features[f'{data_type}_std'] = numeric_data.std(axis=1)
    features[f'{data_type}_min'] = numeric_data.min(axis=1)
    features[f'{data_type}_max'] = numeric_data.max(axis=1)
    features[f'{data_type}_latest'] = numeric_data.iloc[:, -1]  # Most recent value
    
    # Growth features (if we have enough data)
    if len(date_columns) > 12:  # At least 1 year of data
        # 1-year growth
        features[f'{data_type}_growth_1y'] = (
            (numeric_data.iloc[:, -1] - numeric_data.iloc[:, -13]) / 
            numeric_data.iloc[:, -13] * 100
        )
        
        # 5-year growth (if available)
        if len(date_columns) > 60:
            features[f'{data_type}_growth_5y'] = (
                (numeric_data.iloc[:, -1] - numeric_data.iloc[:, -61]) / 
                numeric_data.iloc[:, -61] * 100
            )
    
    # Volatility (standard deviation of monthly changes)
    if len(date_columns) > 1:
        monthly_changes = numeric_data.diff(axis=1).iloc[:, 1:]  # Skip first NaN
        features[f'{data_type}_volatility'] = monthly_changes.std(axis=1)
    
    # Trend (slope of linear regression)
    if len(date_columns) > 6:
        slopes = []
        for idx in range(len(numeric_data)):
            y = numeric_data.iloc[idx].dropna()
            if len(y) > 6:
                x = np.arange(len(y))
                slope = np.polyfit(x, y, 1)[0]
                slopes.append(slope)
            else:
                slopes.append(np.nan)
        features[f'{data_type}_trend'] = slopes
    
    return features

def create_target_variable(df):
    """
    Create a target variable for prediction.
    Since we don't have individual property prices, we'll predict future metro area price changes.
    
    Args:
        df: Processed metro data
        
    Returns:
        pd.DataFrame: Data with target variable
    """
    print("Creating target variable...")
    
    # Use ZHVI growth as target (predicting future price growth)
    if 'zhvi_growth_1y' in df.columns:
        df['target'] = df['zhvi_growth_1y']
        print("Using 1-year ZHVI growth as target variable")
    elif 'zhvi_latest' in df.columns:
        # Create a synthetic target based on price levels
        df['target'] = df['zhvi_latest'] / 1000  # Scale down for easier prediction
        print("Using scaled ZHVI latest value as target variable")
    else:
        print("❌ Cannot create target variable")
        return None
    
    # Remove rows with missing targets
    df = df.dropna(subset=['target'])
    print(f"Data shape after removing missing targets: {df.shape}")
    
    return df

def main():
    """
    Main training pipeline for metro area data.
    """
    print("🏠 Metropolitan Area House Price Prediction")
    print("=" * 60)
    
    # Step 1: Data Loading and Preprocessing
    print("\n📊 Step 1: Data Loading and Preprocessing")
    print("-" * 40)
    
    # Load metro data
    df = load_metro_data()
    
    if df is None:
        print("❌ Failed to load metro data. Using sample data instead...")
        from data_preprocessing import create_sample_data
        df = create_sample_data()
    
    # Create target variable
    df = create_target_variable(df)
    
    if df is None:
        print("❌ Failed to create target variable")
        return
    
    print(f"Final dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Prepare features and target
    # Remove non-feature columns
    non_feature_cols = ['RegionID', 'RegionName', 'target']
    feature_cols = [col for col in df.columns if col not in non_feature_cols]
    
    X = df[feature_cols]
    y = df['target']
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    # Handle missing values
    X = X.fillna(X.mean())
    
    # Step 2: Model Training
    print("\n🤖 Step 2: Model Training")
    print("-" * 40)
    
    trainer = ModelTrainer()
    trainer.initialize_models()
    
    # Split data
    X_train, X_test, y_train, y_test = trainer.split_data(X, y)
    
    # Train models
    trainer.train_models(X_train, y_train, X_test, y_test)
    
    # Find best model
    best_model_name, best_model = trainer.find_best_model()
    
    # Step 3: Model Evaluation
    print("\n📈 Step 3: Model Evaluation")
    print("-" * 40)
    
    evaluator = ModelEvaluator()
    
    # Make predictions with best model
    y_pred = best_model.predict(X_test)
    
    # Generate comprehensive evaluation report
    evaluator.generate_evaluation_report(best_model_name, y_test, y_pred)
    
    # Step 4: Model Persistence
    print("\n💾 Step 4: Model Persistence")
    print("-" * 40)
    
    # Save best model
    model_path = "models/metro_best_model.pkl"
    os.makedirs("models", exist_ok=True)
    
    try:
        joblib.dump(best_model, model_path)
        print(f"✅ Best model saved: {model_path}")
        
    except Exception as e:
        print(f"❌ Error saving model: {e}")
    
    # Step 5: Summary
    print("\n📋 Training Summary")
    print("-" * 40)
    
    print(f"✅ Dataset processed: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"✅ Models trained: {len(trainer.models)}")
    print(f"✅ Best model: {best_model_name}")
    print(f"✅ Model saved: {model_path}")
    
    # Display final metrics
    if best_model_name in trainer.model_scores:
        scores = trainer.model_scores[best_model_name]
        print(f"\n🎯 Final Performance Metrics:")
        print(f"   RMSE: {scores['test_rmse']:.4f}")
        print(f"   MAE: {scores['test_mae']:.4f}")
        print(f"   R²: {scores['test_r2']:.4f}")
    
    print("\n🚀 Metro area training pipeline completed successfully!")
    print("\n📝 Note: This model predicts metro area price trends, not individual property prices.")
    print("For individual property prediction, you need property-level data.")

if __name__ == "__main__":
    main()
