"""
Model Evaluation Module for House Price Prediction

This module provides comprehensive evaluation metrics and visualizations
for house price prediction models.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error
)
from sklearn.model_selection import learning_curve
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')


class ModelEvaluator:
    """
    A comprehensive model evaluation class for house price prediction.
    """
    
    def __init__(self):
        self.evaluation_results = {}
        self.residuals = {}
        
    def calculate_metrics(self, y_true, y_pred, model_name):
        """
        Calculate comprehensive evaluation metrics.
        
        Args:
            y_true (array-like): True values
            y_pred (array-like): Predicted values
            model_name (str): Name of the model
            
        Returns:
            dict: Dictionary containing all metrics
        """
        metrics = {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'mape': mean_absolute_percentage_error(y_true, y_pred) * 100,
            'adjusted_r2': 1 - (1 - r2_score(y_true, y_pred)) * (len(y_true) - 1) / (len(y_true) - len(y_true) - 1)
        }
        
        # Store metrics
        self.evaluation_results[model_name] = metrics
        
        return metrics
    
    def print_metrics(self, model_name):
        """
        Print evaluation metrics for a specific model.
        
        Args:
            model_name (str): Name of the model
        """
        if model_name not in self.evaluation_results:
            print(f"No evaluation results found for {model_name}")
            return
        
        metrics = self.evaluation_results[model_name]
        
        print(f"\n{'='*50}")
        print(f"EVALUATION METRICS - {model_name.upper()}")
        print(f"{'='*50}")
        print(f"Mean Squared Error (MSE):     {metrics['mse']:.4f}")
        print(f"Root Mean Squared Error (RMSE): {metrics['rmse']:.4f}")
        print(f"Mean Absolute Error (MAE):    {metrics['mae']:.4f}")
        print(f"R² Score:                     {metrics['r2']:.4f}")
        print(f"Adjusted R² Score:            {metrics['adjusted_r2']:.4f}")
        print(f"Mean Absolute Percentage Error: {metrics['mape']:.2f}%")
    
    def plot_predictions_vs_actual(self, y_true, y_pred, model_name, figsize=(10, 8)):
        """
        Create scatter plot of predicted vs actual values.
        
        Args:
            y_true (array-like): True values
            y_pred (array-like): Predicted values
            model_name (str): Name of the model
            figsize (tuple): Figure size
        """
        plt.figure(figsize=figsize)
        
        # Create scatter plot
        plt.scatter(y_true, y_pred, alpha=0.6, color='blue')
        
        # Add perfect prediction line
        min_val = min(min(y_true), min(y_pred))
        max_val = max(max(y_true), max(y_pred))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        
        # Calculate and display R²
        r2 = r2_score(y_true, y_pred)
        plt.text(0.05, 0.95, f'R² = {r2:.4f}', transform=plt.gca().transAxes, 
                fontsize=12, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.xlabel('Actual Values', fontsize=12)
        plt.ylabel('Predicted Values', fontsize=12)
        plt.title(f'Predicted vs Actual Values - {model_name}', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_residuals(self, y_true, y_pred, model_name, figsize=(15, 5)):
        """
        Create residual plots for model evaluation.
        
        Args:
            y_true (array-like): True values
            y_pred (array-like): Predicted values
            model_name (str): Name of the model
            figsize (tuple): Figure size
        """
        residuals = y_true - y_pred
        self.residuals[model_name] = residuals
        
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Residuals vs Predicted
        axes[0].scatter(y_pred, residuals, alpha=0.6, color='blue')
        axes[0].axhline(y=0, color='red', linestyle='--')
        axes[0].set_xlabel('Predicted Values')
        axes[0].set_ylabel('Residuals')
        axes[0].set_title('Residuals vs Predicted')
        axes[0].grid(True, alpha=0.3)
        
        # Histogram of residuals
        axes[1].hist(residuals, bins=30, alpha=0.7, color='green', edgecolor='black')
        axes[1].set_xlabel('Residuals')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Distribution of Residuals')
        axes[1].grid(True, alpha=0.3)
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[2])
        axes[2].set_title('Q-Q Plot of Residuals')
        axes[2].grid(True, alpha=0.3)
        
        plt.suptitle(f'Residual Analysis - {model_name}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def plot_learning_curve(self, model, X, y, model_name, cv=5, figsize=(10, 6)):
        """
        Plot learning curve to assess model performance with different training sizes.
        
        Args:
            model: Trained model
            X (array-like): Features
            y (array-like): Target variable
            model_name (str): Name of the model
            cv (int): Number of cross-validation folds
            figsize (tuple): Figure size
        """
        train_sizes, train_scores, val_scores = learning_curve(
            model, X, y, cv=cv, n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 10),
            scoring='neg_mean_squared_error'
        )
        
        # Calculate mean and std
        train_mean = -np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = -np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        plt.figure(figsize=figsize)
        
        # Plot learning curves
        plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Score')
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
        
        plt.plot(train_sizes, val_mean, 'o-', color='red', label='Validation Score')
        plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')
        
        plt.xlabel('Training Set Size')
        plt.ylabel('Mean Squared Error')
        plt.title(f'Learning Curve - {model_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_feature_importance_comparison(self, feature_importance_dict, feature_names, top_n=10):
        """
        Compare feature importance across different models.
        
        Args:
            feature_importance_dict (dict): Dictionary with model names as keys and importance arrays as values
            feature_names (list): Names of features
            top_n (int): Number of top features to display
        """
        n_models = len(feature_importance_dict)
        fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 6))
        
        if n_models == 1:
            axes = [axes]
        
        for i, (model_name, importance) in enumerate(feature_importance_dict.items()):
            # Get top N features
            top_indices = np.argsort(importance)[-top_n:][::-1]
            top_features = [feature_names[idx] for idx in top_indices]
            top_importance = importance[top_indices]
            
            # Create horizontal bar plot
            bars = axes[i].barh(range(len(top_features)), top_importance, 
                               color=plt.cm.viridis(np.linspace(0, 1, len(top_features))))
            axes[i].set_yticks(range(len(top_features)))
            axes[i].set_yticklabels(top_features)
            axes[i].set_xlabel('Feature Importance')
            axes[i].set_title(f'{model_name} - Top {top_n} Features')
            axes[i].invert_yaxis()
            
            # Add value labels
            for j, (bar, value) in enumerate(zip(bars, top_importance)):
                axes[i].text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                           f'{value:.3f}', ha='left', va='center')
        
        plt.tight_layout()
        plt.show()
    
    def create_interactive_plots(self, y_true, y_pred, model_name):
        """
        Create interactive plots using Plotly.
        
        Args:
            y_true (array-like): True values
            y_pred (array-like): Predicted values
            model_name (str): Name of the model
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Predicted vs Actual', 'Residuals vs Predicted', 
                          'Residuals Distribution', 'Error Distribution'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        residuals = y_true - y_pred
        
        # Predicted vs Actual
        fig.add_trace(
            go.Scatter(x=y_true, y=y_pred, mode='markers', name='Predictions',
                      marker=dict(color='blue', opacity=0.6)),
            row=1, col=1
        )
        
        # Add perfect prediction line
        min_val = min(min(y_true), min(y_pred))
        max_val = max(max(y_true), max(y_pred))
        fig.add_trace(
            go.Scatter(x=[min_val, max_val], y=[min_val, max_val], 
                      mode='lines', name='Perfect Prediction',
                      line=dict(color='red', dash='dash')),
            row=1, col=1
        )
        
        # Residuals vs Predicted
        fig.add_trace(
            go.Scatter(x=y_pred, y=residuals, mode='markers', name='Residuals',
                      marker=dict(color='green', opacity=0.6)),
            row=1, col=2
        )
        
        # Add zero line
        fig.add_trace(
            go.Scatter(x=[min(y_pred), max(y_pred)], y=[0, 0], 
                      mode='lines', name='Zero Line',
                      line=dict(color='red', dash='dash')),
            row=1, col=2
        )
        
        # Residuals Distribution
        fig.add_trace(
            go.Histogram(x=residuals, name='Residuals Distribution',
                        marker=dict(color='orange', opacity=0.7)),
            row=2, col=1
        )
        
        # Error Distribution
        errors = np.abs(residuals)
        fig.add_trace(
            go.Histogram(x=errors, name='Absolute Errors',
                        marker=dict(color='purple', opacity=0.7)),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title=f'Interactive Model Evaluation - {model_name}',
            height=800,
            showlegend=False
        )
        
        fig.show()
    
    def compare_models(self, model_results):
        """
        Compare multiple models using various metrics.
        
        Args:
            model_results (dict): Dictionary with model names as keys and (y_true, y_pred) tuples as values
        """
        comparison_data = []
        
        for model_name, (y_true, y_pred) in model_results.items():
            metrics = self.calculate_metrics(y_true, y_pred, model_name)
            comparison_data.append({
                'Model': model_name,
                'RMSE': metrics['rmse'],
                'MAE': metrics['mae'],
                'R²': metrics['r2'],
                'MAPE': metrics['mape']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('RMSE')
        
        print("\n" + "="*80)
        print("MODEL COMPARISON RESULTS")
        print("="*80)
        print(comparison_df.to_string(index=False, float_format='%.4f'))
        
        # Create comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # RMSE comparison
        axes[0, 0].bar(comparison_df['Model'], comparison_df['RMSE'], color='skyblue', alpha=0.7)
        axes[0, 0].set_title('RMSE Comparison')
        axes[0, 0].set_ylabel('RMSE')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # MAE comparison
        axes[0, 1].bar(comparison_df['Model'], comparison_df['MAE'], color='lightcoral', alpha=0.7)
        axes[0, 1].set_title('MAE Comparison')
        axes[0, 1].set_ylabel('MAE')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # R² comparison
        axes[1, 0].bar(comparison_df['Model'], comparison_df['R²'], color='lightgreen', alpha=0.7)
        axes[1, 0].set_title('R² Score Comparison')
        axes[1, 0].set_ylabel('R² Score')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # MAPE comparison
        axes[1, 1].bar(comparison_df['Model'], comparison_df['MAPE'], color='gold', alpha=0.7)
        axes[1, 1].set_title('MAPE Comparison')
        axes[1, 1].set_ylabel('MAPE (%)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        return comparison_df
    
    def generate_evaluation_report(self, model_name, y_true, y_pred):
        """
        Generate a comprehensive evaluation report for a model.
        
        Args:
            model_name (str): Name of the model
            y_true (array-like): True values
            y_pred (array-like): Predicted values
        """
        print(f"\n{'='*60}")
        print(f"COMPREHENSIVE EVALUATION REPORT - {model_name.upper()}")
        print(f"{'='*60}")
        
        # Calculate metrics
        metrics = self.calculate_metrics(y_true, y_pred, model_name)
        
        # Print metrics
        self.print_metrics(model_name)
        
        # Create visualizations
        self.plot_predictions_vs_actual(y_true, y_pred, model_name)
        self.plot_residuals(y_true, y_pred, model_name)
        
        # Model interpretation
        print(f"\n{'='*50}")
        print("MODEL INTERPRETATION")
        print(f"{'='*50}")
        
        if metrics['r2'] > 0.8:
            print("✅ Excellent model performance (R² > 0.8)")
        elif metrics['r2'] > 0.6:
            print("✅ Good model performance (R² > 0.6)")
        elif metrics['r2'] > 0.4:
            print("⚠️  Moderate model performance (R² > 0.4)")
        else:
            print("❌ Poor model performance (R² < 0.4)")
        
        if metrics['mape'] < 10:
            print("✅ Low prediction error (MAPE < 10%)")
        elif metrics['mape'] < 20:
            print("⚠️  Moderate prediction error (MAPE < 20%)")
        else:
            print("❌ High prediction error (MAPE > 20%)")


def create_sample_data():
    """
    Create sample data for testing the evaluation pipeline.
    """
    np.random.seed(42)
    n_samples = 500
    
    # Create true values
    y_true = np.random.normal(500000, 150000, n_samples)
    
    # Create predictions with some relationship to true values
    y_pred = y_true + np.random.normal(0, 50000, n_samples)
    
    return y_true, y_pred


if __name__ == "__main__":
    # Test the evaluation pipeline
    evaluator = ModelEvaluator()
    
    # Create sample data
    y_true, y_pred = create_sample_data()
    
    # Generate comprehensive evaluation report
    evaluator.generate_evaluation_report("Sample Model", y_true, y_pred)
    
    # Create interactive plots
    evaluator.create_interactive_plots(y_true, y_pred, "Sample Model")
