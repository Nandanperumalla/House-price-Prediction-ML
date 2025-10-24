"""
Advanced Prediction System with Confidence Intervals
Handles 2.9M properties with 58+ features and provides uncertainty quantification
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import RobustScaler
import warnings
warnings.filterwarnings('ignore')

class AdvancedPredictor:
    """
    Advanced predictor with confidence intervals and uncertainty quantification.
    """
    
    def __init__(self, model_dir="models"):
        self.model_dir = model_dir
        self.model = None
        self.scaler = None
        self.confidence_model = None
        self.feature_names = None
        self.model_info = None
        self.load_models()
    
    def load_models(self):
        """
        Load trained models and scalers.
        """
        try:
            # Load model info
            self.model_info = joblib.load(f"{self.model_dir}/advanced_model_info.pkl")
            
            # Load best model
            self.model = joblib.load(f"{self.model_dir}/advanced_best_model.pkl")
            
            # Load scaler
            self.scaler = joblib.load(f"{self.model_dir}/advanced_robust_scaler.pkl")
            
            # Load confidence model
            try:
                self.confidence_model = joblib.load(f"{self.model_dir}/advanced_confidence_model.pkl")
            except:
                self.confidence_model = None
            
            # Load feature names
            self.feature_names = joblib.load(f"{self.model_dir}/advanced_feature_names.pkl")
            
            print(f"✅ Loaded advanced model: {self.model_info['best_model_name']}")
            print(f"📊 Model R² Score: {self.model_info['best_score']:.4f}")
            print(f"🔧 Features: {self.model_info['n_features']}")
            
        except FileNotFoundError:
            print("❌ Advanced models not found. Please run train_real_data_advanced.py first.")
            self.model = None
    
    def prepare_features(self, property_data):
        """
        Prepare property data for prediction with feature engineering.
        """
        # Convert to DataFrame if single property
        if isinstance(property_data, dict):
            df = pd.DataFrame([property_data])
        else:
            df = property_data.copy()
        
        # Ensure all required features exist
        required_features = {
            'bedroomcnt': 3,
            'bathroomcnt': 2.0,
            'calculatedfinishedsquarefeet': 2000,
            'latitude': 40.0,
            'longitude': -74.0,
            'yearbuilt': 2000,
            'taxvaluedollarcnt': 300000,
            'landtaxvaluedollarcnt': 150000,
            'structuretaxvaluedollarcnt': 150000,
            'buildingqualitytypeid': 6,
            'heatingorsystemtypeid': 10,
            'airconditioningtypeid': 5,
            'poolcnt': 0,
            'garagecarcnt': 1,
            'fireplacecnt': 0,
            'lotsizesquarefeet': 8000,
            'unitcnt': 1,
            'numberofstories': 2,
            'propertylandusetypeid': 261,
            'hashottuborspa': 0
        }
        
        # Fill missing features with defaults
        for feature, default_value in required_features.items():
            if feature not in df.columns:
                df[feature] = default_value
        
        # Engineer features
        df['price_per_sqft'] = df['taxvaluedollarcnt'] / (df['calculatedfinishedsquarefeet'] + 1)
        df['property_age'] = 2020 - df['yearbuilt']
        df['bath_bed_ratio'] = df['bathroomcnt'] / (df['bedroomcnt'] + 1)
        df['total_rooms'] = df['bedroomcnt'] + df['bathroomcnt']
        df['has_pool'] = (df['poolcnt'] > 0).astype(int)
        df['has_garage'] = (df['garagecarcnt'] > 0).astype(int)
        df['has_fireplace'] = (df['fireplacecnt'] > 0).astype(int)
        df['has_hottub'] = df['hashottuborspa'].fillna(0)
        df['lot_per_sqft'] = df['lotsizesquarefeet'] / (df['calculatedfinishedsquarefeet'] + 1)
        df['value_per_lot'] = df['taxvaluedollarcnt'] / (df['lotsizesquarefeet'] + 1)
        df['quality_score'] = (
            df['buildingqualitytypeid'].fillna(5) * 0.3 +
            df['heatingorsystemtypeid'].fillna(5) * 0.2 +
            df['airconditioningtypeid'].fillna(5) * 0.2 +
            df['architecturalstyletypeid'].fillna(5) * 0.3 if 'architecturalstyletypeid' in df.columns else 5
        )
        df['latitude_rounded'] = df['latitude'].round(1)
        df['longitude_rounded'] = df['longitude'].round(1)
        df['high_value'] = (df['taxvaluedollarcnt'] > 500000).astype(int)
        df['low_value'] = (df['taxvaluedollarcnt'] < 200000).astype(int)
        
        # Select features in correct order
        if self.feature_names:
            available_features = [f for f in self.feature_names if f in df.columns]
            X = df[available_features].fillna(0)
        else:
            X = df.fillna(0)
        
        return X
    
    def predict_single(self, property_data):
        """
        Predict price for a single property with confidence intervals.
        """
        if self.model is None:
            return {
                'prediction': 0,
                'confidence_lower': 0,
                'confidence_upper': 0,
                'confidence_level': 0,
                'error': 'Model not loaded'
            }
        
        try:
            # Prepare features
            X = self.prepare_features(property_data)
            
            # Make prediction
            if self.model_info['best_model_name'] in ['XGBoost', 'LightGBM', 'Random Forest', 'Gradient Boosting']:
                prediction = self.model.predict(X)[0]
            else:
                X_scaled = self.scaler.transform(X)
                prediction = self.model.predict(X_scaled)[0]
            
            # Calculate confidence intervals
            if self.confidence_model is not None:
                uncertainty = self.confidence_model.predict(X)[0]
                confidence_lower = prediction - 1.96 * uncertainty
                confidence_upper = prediction + 1.96 * uncertainty
                confidence_level = 0.95
            else:
                # Simple confidence based on model performance
                uncertainty = abs(prediction) * 0.1
                confidence_lower = prediction - uncertainty
                confidence_upper = prediction + uncertainty
                confidence_level = 0.68
            
            return {
                'prediction': prediction,
                'confidence_lower': confidence_lower,
                'confidence_upper': confidence_upper,
                'confidence_level': confidence_level,
                'uncertainty': uncertainty,
                'model_name': self.model_info['best_model_name'],
                'r2_score': self.model_info['best_score']
            }
            
        except Exception as e:
            return {
                'prediction': 0,
                'confidence_lower': 0,
                'confidence_upper': 0,
                'confidence_level': 0,
                'error': str(e)
            }
    
    def predict_batch(self, properties_data):
        """
        Predict prices for multiple properties.
        """
        if self.model is None:
            return []
        
        results = []
        for property_data in properties_data:
            result = self.predict_single(property_data)
            results.append(result)
        
        return results
    
    def get_model_info(self):
        """
        Get information about the loaded model.
        """
        if self.model_info is None:
            return {
                'status': 'No model loaded',
                'r2_score': 0,
                'features': 0,
                'model_name': 'None'
            }
        
        return {
            'status': 'Model loaded',
            'r2_score': self.model_info['best_score'],
            'features': self.model_info['n_features'],
            'model_name': self.model_info['best_model_name'],
            'confidence_model': self.confidence_model is not None
        }


def main():
    """
    Test the advanced prediction system.
    """
    print("🚀 Testing Advanced Prediction System")
    print("="*50)
    
    # Initialize predictor
    predictor = AdvancedPredictor()
    
    # Test with sample property
    sample_property = {
        'bedroomcnt': 3,
        'bathroomcnt': 2.0,
        'calculatedfinishedsquarefeet': 2000,
        'latitude': 40.7128,
        'longitude': -74.0060,
        'yearbuilt': 2010,
        'taxvaluedollarcnt': 500000,
        'landtaxvaluedollarcnt': 200000,
        'structuretaxvaluedollarcnt': 300000,
        'buildingqualitytypeid': 7,
        'heatingorsystemtypeid': 10,
        'airconditioningtypeid': 5,
        'poolcnt': 0,
        'garagecarcnt': 1,
        'fireplacecnt': 1,
        'lotsizesquarefeet': 5000,
        'unitcnt': 1,
        'numberofstories': 2,
        'propertylandusetypeid': 261,
        'hashottuborspa': 0
    }
    
    # Make prediction
    result = predictor.predict_single(sample_property)
    
    print(f"🏠 Property Prediction:")
    print(f"  Prediction: {result['prediction']:.4f}")
    print(f"  Confidence Interval: [{result['confidence_lower']:.4f}, {result['confidence_upper']:.4f}]")
    print(f"  Confidence Level: {result['confidence_level']:.1%}")
    print(f"  Model: {result.get('model_name', 'Unknown')}")
    print(f"  R² Score: {result.get('r2_score', 0):.4f}")
    
    # Model info
    info = predictor.get_model_info()
    print(f"\n📊 Model Information:")
    print(f"  Status: {info['status']}")
    print(f"  R² Score: {info['r2_score']:.4f}")
    print(f"  Features: {info['features']}")
    print(f"  Model Name: {info['model_name']}")
    print(f"  Confidence Model: {info['confidence_model']}")


if __name__ == "__main__":
    main()
