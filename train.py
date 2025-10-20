"""
Complete Training Pipeline for House Price Prediction

This script demonstrates the complete machine learning pipeline from data loading
to model training and evaluation.
"""

import pandas as pd
import numpy as np
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_preprocessing import DataPreprocessor, create_sample_data
from model_training import ModelTrainer
from evaluation import ModelEvaluator
import joblib

def main():
    """
    Main training pipeline.
    """
    print("🏠 House Price Prediction - Complete Training Pipeline")
    print("=" * 60)
    
    # Step 1: Data Loading and Preprocessing
    print("\n📊 Step 1: Data Loading and Preprocessing")
    print("-" * 40)
    
    preprocessor = DataPreprocessor()
    
    # For demonstration, we'll use sample data
    # In production, load actual data: df = preprocessor.load_data('data/zillow.csv')
    df = create_sample_data()
    
    print(f"Original dataset shape: {df.shape}")
    
    # Run preprocessing pipeline
    X, y = preprocessor.preprocess_pipeline(df)
    
    print(f"Processed features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
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
    model_path = "models/best_model.pkl"
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
    
    print("\n🚀 Training pipeline completed successfully!")
    print("Ready to use the model for predictions!")

if __name__ == "__main__":
    main()
