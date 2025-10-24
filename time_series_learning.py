"""
Time Series Learning System for Market Changes
Learns from historical data to adapt to market trends over time
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class TimeSeriesLearner:
    """
    Time series learning system that adapts to market changes over time.
    """
    
    def __init__(self):
        self.market_models = {}
        self.trend_models = {}
        self.seasonal_models = {}
        self.last_update = None
        self.market_indicators = {}
        
    def load_historical_data(self, data_path):
        """
        Load historical market data for time series learning.
        """
        try:
            # Load your metro data (time series)
            df = pd.read_csv(data_path)
            print(f"✅ Loaded historical data: {len(df):,} records")
            return df
        except FileNotFoundError:
            print("❌ Historical data not found. Creating sample data...")
            return self.create_sample_time_series_data()
    
    def create_sample_time_series_data(self):
        """
        Create sample time series data for demonstration.
        """
        print("🔄 Creating sample time series data...")
        
        # Create date range
        dates = pd.date_range(start='2000-01-01', end='2024-12-01', freq='MS')
        
        # Create sample data for multiple metro areas
        metro_areas = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix']
        
        data = []
        for metro in metro_areas:
            for date in dates:
                # Simulate market trends
                year = date.year
                month = date.month
                
                # Base price with trends
                base_price = 300000 + (year - 2000) * 10000  # General upward trend
                
                # Seasonal effects
                seasonal_factor = 1 + 0.1 * np.sin(2 * np.pi * month / 12)
                
                # Market cycles (boom/bust)
                cycle_factor = 1 + 0.2 * np.sin(2 * np.pi * (year - 2000) / 7)
                
                # Random noise
                noise = np.random.normal(0, 0.05)
                
                # Metro-specific adjustments
                metro_multiplier = {
                    'New York': 1.5,
                    'Los Angeles': 1.3,
                    'Chicago': 0.9,
                    'Houston': 0.8,
                    'Phoenix': 0.7
                }
                
                price = base_price * seasonal_factor * cycle_factor * metro_multiplier[metro] * (1 + noise)
                
                data.append({
                    'date': date,
                    'metro_area': metro,
                    'zhvi': price,
                    'year': year,
                    'month': month,
                    'quarter': (month - 1) // 3 + 1
                })
        
        df = pd.DataFrame(data)
        print(f"✅ Created sample time series data: {len(df):,} records")
        return df
    
    def engineer_time_features(self, df):
        """
        Engineer time-based features for learning.
        """
        print("🔧 Engineering time-based features...")
        
        # Convert date to datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Time features
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        df['day_of_year'] = df['date'].dt.dayofyear
        
        # Cyclical features
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['quarter_sin'] = np.sin(2 * np.pi * df['quarter'] / 4)
        df['quarter_cos'] = np.cos(2 * np.pi * df['quarter'] / 4)
        
        # Market phase features
        df['years_since_2000'] = df['year'] - 2000
        df['is_recession'] = ((df['year'] >= 2008) & (df['year'] <= 2009)).astype(int)
        df['is_boom'] = ((df['year'] >= 2003) & (df['year'] <= 2007)).astype(int)
        
        # Price change features
        df = df.sort_values(['metro_area', 'date'])
        df['price_change'] = df.groupby('metro_area')['zhvi'].pct_change()
        df['price_change_3m'] = df.groupby('metro_area')['zhvi'].pct_change(3)
        df['price_change_12m'] = df.groupby('metro_area')['zhvi'].pct_change(12)
        
        # Moving averages
        df['ma_3m'] = df.groupby('metro_area')['zhvi'].rolling(3).mean().reset_index(0, drop=True)
        df['ma_12m'] = df.groupby('metro_area')['zhvi'].rolling(12).mean().reset_index(0, drop=True)
        
        # Volatility
        df['volatility_3m'] = df.groupby('metro_area')['price_change'].rolling(3).std().reset_index(0, drop=True)
        df['volatility_12m'] = df.groupby('metro_area')['price_change'].rolling(12).std().reset_index(0, drop=True)
        
        print(f"✅ Engineered time features. Total features: {len(df.columns)}")
        return df
    
    def train_market_models(self, df):
        """
        Train models for different market aspects.
        """
        print("🤖 Training market learning models...")
        
        # Prepare features
        feature_columns = [
            'year', 'month', 'quarter', 'day_of_year',
            'month_sin', 'month_cos', 'quarter_sin', 'quarter_cos',
            'years_since_2000', 'is_recession', 'is_boom',
            'price_change', 'price_change_3m', 'price_change_12m',
            'ma_3m', 'ma_12m', 'volatility_3m', 'volatility_12m'
        ]
        
        # Train models for each metro area
        for metro in df['metro_area'].unique():
            metro_data = df[df['metro_area'] == metro].copy()
            metro_data = metro_data.dropna()
            
            if len(metro_data) < 50:  # Need enough data
                continue
            
            X = metro_data[feature_columns].fillna(0)
            y = metro_data['zhvi']
            
            # Train market trend model
            trend_model = LinearRegression()
            trend_model.fit(X, y)
            self.trend_models[metro] = trend_model
            
            # Train seasonal model
            seasonal_features = ['month_sin', 'month_cos', 'quarter_sin', 'quarter_cos']
            X_seasonal = metro_data[seasonal_features].fillna(0)
            seasonal_model = LinearRegression()
            seasonal_model.fit(X_seasonal, y)
            self.seasonal_models[metro] = seasonal_model
            
            # Train comprehensive market model
            market_model = RandomForestRegressor(n_estimators=100, random_state=42)
            market_model.fit(X, y)
            self.market_models[metro] = market_model
            
            # Calculate market indicators
            recent_data = metro_data.tail(12)  # Last year
            if len(recent_data) > 0:
                self.market_indicators[metro] = {
                    'current_trend': recent_data['price_change'].mean(),
                    'volatility': recent_data['volatility_3m'].mean(),
                    'seasonal_strength': recent_data['price_change'].std(),
                    'market_phase': 'bull' if recent_data['price_change'].mean() > 0 else 'bear'
                }
            
            print(f"✅ Trained models for {metro}")
        
        print(f"✅ Trained models for {len(self.market_models)} metro areas")
    
    def predict_market_trends(self, metro_area, future_months=12):
        """
        Predict future market trends for a metro area.
        """
        if metro_area not in self.market_models:
            return None
        
        # Create future dates
        last_date = pd.Timestamp.now().replace(day=1)
        future_dates = pd.date_range(start=last_date, periods=future_months + 1, freq='MS')[1:]
        
        predictions = []
        
        for date in future_dates:
            # Create features for future date
            features = {
                'year': date.year,
                'month': date.month,
                'quarter': date.quarter,
                'day_of_year': date.dayofyear,
                'month_sin': np.sin(2 * np.pi * date.month / 12),
                'month_cos': np.cos(2 * np.pi * date.month / 12),
                'quarter_sin': np.sin(2 * np.pi * date.quarter / 4),
                'quarter_cos': np.cos(2 * np.pi * date.quarter / 4),
                'years_since_2000': date.year - 2000,
                'is_recession': 0,  # Assume no recession
                'is_boom': 0,  # Assume no boom
                'price_change': 0.01,  # Assume 1% monthly growth
                'price_change_3m': 0.03,  # Assume 3% quarterly growth
                'price_change_12m': 0.12,  # Assume 12% annual growth
                'ma_3m': 0,  # Will be calculated
                'ma_12m': 0,  # Will be calculated
                'volatility_3m': 0.02,  # Assume 2% volatility
                'volatility_12m': 0.05   # Assume 5% annual volatility
            }
            
            # Make prediction
            X = pd.DataFrame([features])
            prediction = self.market_models[metro_area].predict(X)[0]
            
            predictions.append({
                'date': date,
                'predicted_price': prediction,
                'metro_area': metro_area
            })
        
        return predictions
    
    def get_market_insights(self, metro_area):
        """
        Get current market insights for a metro area.
        """
        if metro_area not in self.market_indicators:
            return None
        
        indicators = self.market_indicators[metro_area]
        
        insights = {
            'metro_area': metro_area,
            'current_trend': indicators['current_trend'],
            'volatility': indicators['volatility'],
            'market_phase': indicators['market_phase'],
            'trend_direction': 'Upward' if indicators['current_trend'] > 0 else 'Downward',
            'volatility_level': 'High' if indicators['volatility'] > 0.05 else 'Low',
            'recommendation': self._get_market_recommendation(indicators)
        }
        
        return insights
    
    def _get_market_recommendation(self, indicators):
        """
        Get market recommendation based on indicators.
        """
        trend = indicators['current_trend']
        volatility = indicators['volatility']
        
        if trend > 0.02 and volatility < 0.03:
            return "Strong buy - upward trend with low volatility"
        elif trend > 0.01 and volatility < 0.05:
            return "Buy - positive trend with moderate volatility"
        elif trend > -0.01 and volatility < 0.05:
            return "Hold - stable market"
        elif trend < -0.01 and volatility < 0.05:
            return "Sell - downward trend"
        else:
            return "Wait - high volatility market"
    
    def update_models(self, new_data):
        """
        Update models with new data (incremental learning).
        """
        print("🔄 Updating models with new data...")
        
        # This would implement incremental learning
        # For now, we'll retrain with all data
        df = self.load_historical_data("data/metro_data.csv")
        df = self.engineer_time_features(df)
        self.train_market_models(df)
        
        self.last_update = pd.Timestamp.now()
        print("✅ Models updated successfully")
    
    def save_models(self, save_dir="models"):
        """
        Save time series models.
        """
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        # Save market models
        joblib.dump(self.market_models, f"{save_dir}/time_series_market_models.pkl")
        joblib.dump(self.trend_models, f"{save_dir}/time_series_trend_models.pkl")
        joblib.dump(self.seasonal_models, f"{save_dir}/time_series_seasonal_models.pkl")
        joblib.dump(self.market_indicators, f"{save_dir}/market_indicators.pkl")
        
        print(f"✅ Time series models saved to {save_dir}/")


def main():
    """
    Test the time series learning system.
    """
    print("🚀 Testing Time Series Learning System")
    print("="*50)
    
    # Initialize learner
    learner = TimeSeriesLearner()
    
    # Load and prepare data
    df = learner.load_historical_data("data/metro_data.csv")
    df = learner.engineer_time_features(df)
    
    # Train models
    learner.train_market_models(df)
    
    # Test predictions
    metro_areas = ['New York', 'Los Angeles', 'Chicago']
    
    for metro in metro_areas:
        print(f"\n🏙️ {metro} Market Analysis:")
        
        # Get market insights
        insights = learner.get_market_insights(metro)
        if insights:
            print(f"  Trend: {insights['trend_direction']} ({insights['current_trend']:.2%})")
            print(f"  Volatility: {insights['volatility_level']} ({insights['volatility']:.2%})")
            print(f"  Phase: {insights['market_phase']}")
            print(f"  Recommendation: {insights['recommendation']}")
        
        # Predict future trends
        predictions = learner.predict_market_trends(metro, 6)
        if predictions:
            print(f"  Next 6 months predictions:")
            for pred in predictions[:3]:  # Show first 3
                print(f"    {pred['date'].strftime('%Y-%m')}: ${pred['predicted_price']:,.0f}")
    
    # Save models
    learner.save_models()
    
    print("\n✅ Time series learning system ready!")


if __name__ == "__main__":
    main()
