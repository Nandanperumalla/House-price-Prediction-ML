"""
Model Training Module for House Price Prediction

This module handles training multiple machine learning models
and comparing their performance for house price prediction.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class ModelTrainer:
    """
    A comprehensive model training class for house price prediction.
    """
    
    def __init__(self):
        self.models = {}
        self.model_scores = {}
        self.best_model = None
        self.best_model_name = None
        self.feature_importance = {}
        
    def initialize_models(self):
        """
        Initialize various regression models for comparison.
        """
        self.models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=0.1),
            'Random Forest': RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=100,
                random_state=42
            ),
            'XGBoost': XGBRegressor(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            )
        }
        
        print("Models initialized successfully!")
        print(f"Available models: {list(self.models.keys())}")
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """
        Split data into training and testing sets.
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target variable
            test_size (float): Proportion of data for testing
            random_state (int): Random seed for reproducibility
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        print(f"Data split completed:")
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        return X_train, X_test, y_train, y_test
    
    def train_models(self, X_train, y_train, X_test, y_test):
        """
        Train all models and evaluate their performance.
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test target
        """
        print("=" * 60)
        print("TRAINING MODELS")
        print("=" * 60)
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            
            try:
                # Train the model
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred_train = model.predict(X_train)
                y_pred_test = model.predict(X_test)
                
                # Calculate metrics
                train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
                test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
                train_mae = mean_absolute_error(y_train, y_pred_train)
                test_mae = mean_absolute_error(y_test, y_pred_test)
                train_r2 = r2_score(y_train, y_pred_train)
                test_r2 = r2_score(y_test, y_pred_test)
                
                # Store scores
                self.model_scores[name] = {
                    'train_rmse': train_rmse,
                    'test_rmse': test_rmse,
                    'train_mae': train_mae,
                    'test_mae': test_mae,
                    'train_r2': train_r2,
                    'test_r2': test_r2
                }
                
                # Store feature importance for tree-based models
                if hasattr(model, 'feature_importances_'):
                    self.feature_importance[name] = model.feature_importances_
                
                print(f"✓ {name} trained successfully")
                print(f"  Test RMSE: {test_rmse:.4f}")
                print(f"  Test MAE: {test_mae:.4f}")
                print(f"  Test R²: {test_r2:.4f}")
                
            except Exception as e:
                print(f"✗ Error training {name}: {e}")
    
    def cross_validate_models(self, X, y, cv=5):
        """
        Perform cross-validation for all models.
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target variable
            cv (int): Number of cross-validation folds
        """
        print("\n" + "=" * 60)
        print("CROSS-VALIDATION RESULTS")
        print("=" * 60)
        
        cv_results = {}
        
        for name, model in self.models.items():
            try:
                # Perform cross-validation
                cv_scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error')
                cv_rmse = np.sqrt(-cv_scores)
                
                cv_results[name] = {
                    'mean_rmse': cv_rmse.mean(),
                    'std_rmse': cv_rmse.std(),
                    'cv_scores': cv_rmse
                }
                
                print(f"{name}:")
                print(f"  Mean RMSE: {cv_rmse.mean():.4f} (±{cv_rmse.std():.4f})")
                
            except Exception as e:
                print(f"✗ Error in cross-validation for {name}: {e}")
        
        return cv_results
    
    def find_best_model(self, metric='test_rmse'):
        """
        Find the best performing model based on specified metric.
        
        Args:
            metric (str): Metric to use for comparison
            
        Returns:
            tuple: (best_model_name, best_model)
        """
        if not self.model_scores:
            print("No models have been trained yet!")
            return None, None
        
        # Find model with best (lowest) metric
        best_score = float('inf')
        best_name = None
        
        for name, scores in self.model_scores.items():
            if scores[metric] < best_score:
                best_score = scores[metric]
                best_name = name
        
        self.best_model_name = best_name
        self.best_model = self.models[best_name]
        
        print(f"\nBest model: {best_name}")
        print(f"Best {metric}: {best_score:.4f}")
        
        return best_name, self.best_model
    
    def hyperparameter_tuning(self, X_train, y_train, model_name='XGBoost'):
        """
        Perform hyperparameter tuning for the specified model.
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            model_name (str): Name of the model to tune
        """
        print(f"\nPerforming hyperparameter tuning for {model_name}...")
        
        if model_name == 'XGBoost':
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0]
            }
            model = XGBRegressor(random_state=42, n_jobs=-1)
            
        elif model_name == 'Random Forest':
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            model = RandomForestRegressor(random_state=42, n_jobs=-1)
            
        else:
            print(f"Hyperparameter tuning not implemented for {model_name}")
            return
        
        # Perform grid search
        grid_search = GridSearchCV(
            model, param_grid, cv=3, scoring='neg_mean_squared_error',
            n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        # Update the model with best parameters
        self.models[model_name] = grid_search.best_estimator_
        
        print(f"Best parameters for {model_name}:")
        for param, value in grid_search.best_params_.items():
            print(f"  {param}: {value}")
        
        print(f"Best cross-validation RMSE: {np.sqrt(-grid_search.best_score_):.4f}")
    
    def plot_model_comparison(self):
        """
        Create visualizations comparing model performance.
        """
        if not self.model_scores:
            print("No model scores available for plotting!")
            return
        
        # Prepare data for plotting
        model_names = list(self.model_scores.keys())
        test_rmse = [scores['test_rmse'] for scores in self.model_scores.values()]
        test_r2 = [scores['test_r2'] for scores in self.model_scores.values()]
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # RMSE comparison
        bars1 = ax1.bar(model_names, test_rmse, color='skyblue', alpha=0.7)
        ax1.set_title('Model Comparison - RMSE', fontsize=14, fontweight='bold')
        ax1.set_ylabel('RMSE', fontsize=12)
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars1, test_rmse):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{value:.4f}', ha='center', va='bottom')
        
        # R² comparison
        bars2 = ax2.bar(model_names, test_r2, color='lightcoral', alpha=0.7)
        ax2.set_title('Model Comparison - R² Score', fontsize=14, fontweight='bold')
        ax2.set_ylabel('R² Score', fontsize=12)
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars2, test_r2):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
    
    def plot_feature_importance(self, feature_names, top_n=10):
        """
        Plot feature importance for tree-based models.
        
        Args:
            feature_names (list): Names of features
            top_n (int): Number of top features to display
        """
        if not self.feature_importance:
            print("No feature importance data available!")
            return
        
        # Create subplots for each model with feature importance
        n_models = len(self.feature_importance)
        fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 6))
        
        if n_models == 1:
            axes = [axes]
        
        for i, (model_name, importance) in enumerate(self.feature_importance.items()):
            # Get top N features
            top_indices = np.argsort(importance)[-top_n:][::-1]
            top_features = [feature_names[idx] for idx in top_indices]
            top_importance = importance[top_indices]
            
            # Create horizontal bar plot
            axes[i].barh(range(len(top_features)), top_importance, color='lightgreen', alpha=0.7)
            axes[i].set_yticks(range(len(top_features)))
            axes[i].set_yticklabels(top_features)
            axes[i].set_xlabel('Feature Importance')
            axes[i].set_title(f'{model_name} - Top {top_n} Features')
            axes[i].invert_yaxis()
        
        plt.tight_layout()
        plt.show()
    
    def save_model(self, model_name=None, filepath=None):
        """
        Save the best model or specified model to disk.
        
        Args:
            model_name (str): Name of the model to save
            filepath (str): Path to save the model
        """
        if model_name is None:
            model_name = self.best_model_name
            model = self.best_model
        else:
            model = self.models.get(model_name)
        
        if model is None:
            print(f"Model {model_name} not found!")
            return
        
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"models/{model_name.lower().replace(' ', '_')}_{timestamp}.pkl"
        
        try:
            joblib.dump(model, filepath)
            print(f"Model saved successfully: {filepath}")
        except Exception as e:
            print(f"Error saving model: {e}")
    
    def load_model(self, filepath):
        """
        Load a saved model from disk.
        
        Args:
            filepath (str): Path to the saved model
            
        Returns:
            Loaded model object
        """
        try:
            model = joblib.load(filepath)
            print(f"Model loaded successfully: {filepath}")
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    
    def generate_report(self):
        """
        Generate a comprehensive model performance report.
        """
        if not self.model_scores:
            print("No model scores available for report!")
            return
        
        print("\n" + "=" * 80)
        print("COMPREHENSIVE MODEL PERFORMANCE REPORT")
        print("=" * 80)
        
        # Create a DataFrame for better visualization
        report_data = []
        for name, scores in self.model_scores.items():
            report_data.append({
                'Model': name,
                'Train RMSE': scores['train_rmse'],
                'Test RMSE': scores['test_rmse'],
                'Train MAE': scores['train_mae'],
                'Test MAE': scores['test_mae'],
                'Train R²': scores['train_r2'],
                'Test R²': scores['test_r2']
            })
        
        report_df = pd.DataFrame(report_data)
        report_df = report_df.sort_values('Test RMSE')
        
        print(report_df.to_string(index=False, float_format='%.4f'))
        
        # Highlight the best model
        best_model = report_df.iloc[0]
        print(f"\n🏆 BEST PERFORMING MODEL: {best_model['Model']}")
        print(f"   Test RMSE: {best_model['Test RMSE']:.4f}")
        print(f"   Test R²: {best_model['Test R²']:.4f}")


def create_sample_data():
    """
    Create sample data for testing the model training pipeline.
    """
    np.random.seed(42)
    n_samples = 1000
    
    # Create features
    X = pd.DataFrame({
        'bedroomcnt': np.random.randint(1, 6, n_samples),
        'bathroomcnt': np.random.uniform(1, 4, n_samples),
        'calculatedfinishedsquarefeet': np.random.randint(800, 4000, n_samples),
        'taxvaluedollarcnt': np.random.randint(100000, 1000000, n_samples),
        'yearbuilt': np.random.randint(1950, 2020, n_samples),
        'regionidzip': np.random.randint(10000, 99999, n_samples)
    })
    
    # Create target with some relationship to features
    y = (X['taxvaluedollarcnt'] / 100000 + 
         X['calculatedfinishedsquarefeet'] / 1000 + 
         X['bedroomcnt'] * 0.1 + 
         np.random.normal(0, 0.1, n_samples))
    
    return X, y


if __name__ == "__main__":
    # Test the model training pipeline
    trainer = ModelTrainer()
    
    # Create sample data
    X, y = create_sample_data()
    
    # Initialize models
    trainer.initialize_models()
    
    # Split data
    X_train, X_test, y_train, y_test = trainer.split_data(X, y)
    
    # Train models
    trainer.train_models(X_train, y_train, X_test, y_test)
    
    # Find best model
    trainer.find_best_model()
    
    # Generate report
    trainer.generate_report()
    
    # Save best model
    trainer.save_model()
