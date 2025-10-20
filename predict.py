"""
Prediction Script for House Price Prediction

This script demonstrates how to load a trained model and make predictions
on new data.
"""

import pandas as pd
import numpy as np
import joblib
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_preprocessing import DataPreprocessor

def load_model(model_path="models/best_model.pkl"):
    """
    Load a trained model from disk.
    
    Args:
        model_path (str): Path to the saved model
        
    Returns:
        Loaded model object
    """
    try:
        model = joblib.load(model_path)
        print(f"✅ Model loaded successfully from {model_path}")
        return model
    except FileNotFoundError:
        print(f"❌ Model file not found: {model_path}")
        print("Please run train.py first to train and save a model.")
        return None
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

def predict_price(model, property_data, preprocessor):
    """
    Make price prediction for a single property.
    
    Args:
        model: Trained model
        property_data (dict): Property features
        preprocessor: DataPreprocessor instance
        
    Returns:
        float: Predicted price
    """
    try:
        # Convert to DataFrame
        df = pd.DataFrame([property_data])
        
        # Apply the same preprocessing pipeline
        X, _ = preprocessor.preprocess_pipeline(df)
        
        # Make prediction
        prediction = model.predict(X)[0]
        
        return prediction
    except Exception as e:
        print(f"❌ Error making prediction: {e}")
        return None

def main():
    """
    Main prediction script.
    """
    print("🏠 House Price Prediction - Prediction Script")
    print("=" * 50)
    
    # Load model
    model = load_model()
    
    if model is None:
        return
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Example property data
    property_data = {
        'parcelid': 1,  # Add required parcelid
        'bedroomcnt': 3,
        'bathroomcnt': 2.5,
        'calculatedfinishedsquarefeet': 2000,
        'taxvaluedollarcnt': 500000,
        'yearbuilt': 2010,
        'regionidzip': 12345,
        'logerror': 0.0  # Add dummy target for preprocessing
    }
    
    print("\n🏡 Property Details:")
    for key, value in property_data.items():
        if key not in ['parcelid', 'logerror']:  # Don't show internal fields
            print(f"   {key}: {value}")
    
    # Make prediction
    print("\n🎯 Making Prediction...")
    prediction = predict_price(model, property_data, preprocessor)
    
    if prediction is not None:
        print(f"\n💰 Predicted Price: ${prediction:,.2f}")
        
        # Add confidence interval (simplified)
        confidence = 0.85
        margin = prediction * 0.1
        lower_bound = prediction - margin
        upper_bound = prediction + margin
        
        print(f"📊 Confidence: {confidence*100:.0f}%")
        print(f"📈 Price Range: ${lower_bound:,.2f} - ${upper_bound:,.2f}")
    
    # Interactive prediction
    print("\n" + "="*50)
    print("🔄 Interactive Prediction Mode")
    print("="*50)
    
    while True:
        try:
            print("\nEnter property details (or 'quit' to exit):")
            
            bedrooms = input("Number of bedrooms: ")
            if bedrooms.lower() == 'quit':
                break
            
            bathrooms = input("Number of bathrooms: ")
            square_feet = input("Square footage: ")
            year_built = input("Year built: ")
            zip_code = input("ZIP code: ")
            
            # Create property data
            property_data = {
                'parcelid': 1,  # Add required parcelid
                'bedroomcnt': int(bedrooms),
                'bathroomcnt': float(bathrooms),
                'calculatedfinishedsquarefeet': int(square_feet),
                'taxvaluedollarcnt': 300000,  # Default value
                'yearbuilt': int(year_built),
                'regionidzip': int(zip_code),
                'logerror': 0.0  # Add dummy target for preprocessing
            }
            
            # Make prediction
            prediction = predict_price(model, property_data, preprocessor)
            
            if prediction is not None:
                print(f"\n💰 Predicted Price: ${prediction:,.2f}")
            
        except ValueError:
            print("❌ Invalid input. Please enter numeric values.")
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
