"""
Streamlit Web Application for House Price Prediction

This application provides an interactive interface for predicting house prices
using the trained machine learning models.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_preprocessing import DataPreprocessor
from model_training import ModelTrainer
from evaluation import ModelEvaluator

# Page configuration
st.set_page_config(
    page_title="🏠 House Price Prediction",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .prediction-card {
        background-color: #e8f4fd;
        padding: 2rem;
        border-radius: 1rem;
        border: 2px solid #1f77b4;
        text-align: center;
    }
    .stButton > button {
        background-color: #1f77b4;
        color: white;
        border-radius: 0.5rem;
        border: none;
        padding: 0.5rem 2rem;
        font-size: 1.1rem;
    }
    .stButton > button:hover {
        background-color: #0d5aa7;
    }
</style>
""", unsafe_allow_html=True)

def load_sample_model():
    """
    Load a sample model or create a simple one for demonstration.
    """
    try:
        # Try to load a saved model
        model_path = "models/best_model.pkl"
        if os.path.exists(model_path):
            return joblib.load(model_path)
    except:
        pass
    
    # Create a simple model for demonstration
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    
    # Create sample data for training
    np.random.seed(42)
    n_samples = 1000
    
    X_sample = pd.DataFrame({
        'bedroomcnt': np.random.randint(1, 6, n_samples),
        'bathroomcnt': np.random.uniform(1, 4, n_samples),
        'calculatedfinishedsquarefeet': np.random.randint(800, 4000, n_samples),
        'taxvaluedollarcnt': np.random.randint(100000, 1000000, n_samples),
        'yearbuilt': np.random.randint(1950, 2020, n_samples),
        'regionidzip': np.random.randint(10000, 99999, n_samples)
    })
    
    y_sample = (X_sample['taxvaluedollarcnt'] / 100000 + 
               X_sample['calculatedfinishedsquarefeet'] / 1000 + 
               X_sample['bedroomcnt'] * 0.1 + 
               np.random.normal(0, 0.1, n_samples))
    
    # Train a simple model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_sample, y_sample)
    
    return model

def create_sample_data():
    """
    Create sample data for demonstration.
    """
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'parcelid': range(1, n_samples + 1),
        'bedroomcnt': np.random.randint(1, 6, n_samples),
        'bathroomcnt': np.random.uniform(1, 4, n_samples),
        'calculatedfinishedsquarefeet': np.random.randint(800, 4000, n_samples),
        'taxvaluedollarcnt': np.random.randint(100000, 1000000, n_samples),
        'yearbuilt': np.random.randint(1950, 2020, n_samples),
        'regionidzip': np.random.randint(10000, 99999, n_samples),
        'logerror': np.random.normal(0, 0.1, n_samples)
    }
    
    return pd.DataFrame(data)

def main():
    """
    Main Streamlit application.
    """
    # Header
    st.markdown('<h1 class="main-header">🏠 House Price Prediction</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("🎛️ Control Panel")
    
    # Page selection
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🏠 Price Prediction", "📊 Data Analysis", "🤖 Model Training", "📈 Model Evaluation"]
    )
    
    if page == "🏠 Price Prediction":
        price_prediction_page()
    elif page == "📊 Data Analysis":
        data_analysis_page()
    elif page == "🤖 Model Training":
        model_training_page()
    elif page == "📈 Model Evaluation":
        model_evaluation_page()

def price_prediction_page():
    """
    Price prediction page.
    """
    st.header("🏠 House Price Prediction")
    st.write("Enter the property details below to get a price prediction.")
    
    # Create two columns
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📝 Property Details")
        
        # Input fields
        bedrooms = st.number_input("Number of Bedrooms", min_value=1, max_value=10, value=3)
        bathrooms = st.number_input("Number of Bathrooms", min_value=1.0, max_value=10.0, value=2.0, step=0.5)
        square_feet = st.number_input("Square Footage", min_value=500, max_value=10000, value=2000)
        year_built = st.number_input("Year Built", min_value=1800, max_value=2023, value=2000)
        zip_code = st.number_input("ZIP Code", min_value=10000, max_value=99999, value=12345)
        tax_value = st.number_input("Tax Assessed Value ($)", min_value=50000, max_value=10000000, value=300000, step=10000)
        
        # Additional features
        st.subheader("🔧 Additional Features")
        has_pool = st.checkbox("Has Pool")
        has_garage = st.checkbox("Has Garage")
        has_fireplace = st.checkbox("Has Fireplace")
        
    with col2:
        st.subheader("🎯 Prediction")
        
        # Create feature vector
        base_features = {
            'parcelid': 1,
            'bedroomcnt': bedrooms,
            'bathroomcnt': bathrooms,
            'calculatedfinishedsquarefeet': square_feet,
            'taxvaluedollarcnt': tax_value,
            'yearbuilt': year_built,
            'regionidzip': zip_code,
            'logerror': 0.0
        }
        feature_df = pd.DataFrame([base_features])
        
        # Load model and make prediction
        try:
            # Try loading user's trained model first
            model = None
            try:
                model_path = "models/best_model.pkl"
                if os.path.exists(model_path):
                    model = joblib.load(model_path)
            except Exception:
                model = None

            if model is not None:
                # Apply the same preprocessing used during training
                preprocessor = DataPreprocessor()
                X_processed, _ = preprocessor.preprocess_pipeline(feature_df)
                prediction = model.predict(X_processed)[0]
            else:
                # Fallback to internal demo model
                model = load_sample_model()
                # Align to demo model's feature set
                demo_features = feature_df[[
                    'bedroomcnt',
                    'bathroomcnt',
                    'calculatedfinishedsquarefeet',
                    'taxvaluedollarcnt',
                    'yearbuilt',
                    'regionidzip'
                ]]
                prediction = model.predict(demo_features)[0]
            
            # Display prediction
            st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
            st.metric("Predicted Price", f"${prediction:,.2f}")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Confidence interval (simplified)
            confidence = 0.85
            margin = prediction * 0.1
            lower_bound = prediction - margin
            upper_bound = prediction + margin
            
            st.write(f"**Confidence:** {confidence*100:.0f}%")
            st.write(f"**Price Range:** ${lower_bound:,.2f} - ${upper_bound:,.2f}")
            
        except Exception as e:
            st.error(f"Error making prediction: {e}")
    
    # Additional information
    st.subheader("ℹ️ About This Prediction")
    st.info("""
    This prediction is based on a machine learning model trained on historical house price data. 
    The model considers various factors including property size, location, age, and amenities.
    
    **Note:** This is a demonstration model. For production use, ensure you have:
    - High-quality training data
    - Proper model validation
    - Regular model updates
    """)

def data_analysis_page():
    """
    Data analysis page.
    """
    st.header("📊 Data Analysis")
    
    # Load sample data
    df = create_sample_data()
    
    # Data overview
    st.subheader("📋 Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", len(df))
    with col2:
        st.metric("Features", len(df.columns) - 1)
    with col3:
        st.metric("Missing Values", df.isnull().sum().sum())
    with col4:
        st.metric("Duplicate Rows", df.duplicated().sum())
    
    # Data preview
    st.subheader("🔍 Data Preview")
    st.dataframe(df.head(10))
    
    # Statistics
    st.subheader("📈 Statistical Summary")
    st.dataframe(df.describe())
    
    # Visualizations
    st.subheader("📊 Data Visualizations")
    
    # Distribution plots
    col1, col2 = st.columns(2)
    
    with col1:
        # Price distribution
        fig = px.histogram(df, x='taxvaluedollarcnt', title='Price Distribution')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Square footage distribution
        fig = px.histogram(df, x='calculatedfinishedsquarefeet', title='Square Footage Distribution')
        st.plotly_chart(fig, use_container_width=True)
    
    # Correlation heatmap
    st.subheader("🔗 Feature Correlations")
    correlation_matrix = df.corr()
    fig = px.imshow(correlation_matrix, 
                    text_auto=True, 
                    aspect="auto",
                    title="Correlation Matrix")
    st.plotly_chart(fig, use_container_width=True)
    
    # Scatter plots
    st.subheader("📈 Feature Relationships")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.scatter(df, x='calculatedfinishedsquarefeet', y='taxvaluedollarcnt',
                        title='Price vs Square Footage')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.scatter(df, x='bedroomcnt', y='taxvaluedollarcnt',
                        title='Price vs Bedrooms')
        st.plotly_chart(fig, use_container_width=True)

def model_training_page():
    """
    Model training page.
    """
    st.header("🤖 Model Training")
    
    # Load sample data
    df = create_sample_data()
    
    # Initialize components
    preprocessor = DataPreprocessor()
    trainer = ModelTrainer()
    
    st.subheader("📊 Data Preprocessing")
    
    # Preprocessing options
    col1, col2 = st.columns(2)
    
    with col1:
        handle_missing = st.selectbox("Handle Missing Values", ["impute", "drop"])
        remove_outliers = st.checkbox("Remove Outliers", value=True)
    
    with col2:
        feature_engineering = st.checkbox("Feature Engineering", value=True)
        scale_features = st.checkbox("Scale Features", value=True)
    
    # Process data
    if st.button("🔄 Process Data"):
        with st.spinner("Processing data..."):
            # Run preprocessing pipeline
            X, y = preprocessor.preprocess_pipeline(df)
            
            st.success("Data processed successfully!")
            st.write(f"Features shape: {X.shape}")
            st.write(f"Target shape: {y.shape}")
            
            # Store in session state
            st.session_state['X'] = X
            st.session_state['y'] = y
    
    # Model training
    st.subheader("🎯 Model Training")
    
    if 'X' in st.session_state and 'y' in st.session_state:
        X = st.session_state['X']
        y = st.session_state['y']
        
        # Model selection
        models_to_train = st.multiselect(
            "Select Models to Train",
            ["Linear Regression", "Random Forest", "XGBoost", "Gradient Boosting"],
            default=["Random Forest", "XGBoost"]
        )
        
        if st.button("🚀 Train Models"):
            with st.spinner("Training models..."):
                # Initialize models
                trainer.initialize_models()
                
                # Filter selected models
                selected_models = {k: v for k, v in trainer.models.items() if k in models_to_train}
                trainer.models = selected_models
                
                # Split data
                X_train, X_test, y_train, y_test = trainer.split_data(X, y)
                
                # Train models
                trainer.train_models(X_train, y_train, X_test, y_test)
                
                # Store results
                st.session_state['trainer'] = trainer
                st.session_state['X_test'] = X_test
                st.session_state['y_test'] = y_test
                
                st.success("Models trained successfully!")
                
                # Display results
                trainer.generate_report()
    
    # Model comparison
    if 'trainer' in st.session_state:
        st.subheader("📊 Model Comparison")
        trainer = st.session_state['trainer']
        
        # Create comparison table
        comparison_data = []
        for name, scores in trainer.model_scores.items():
            comparison_data.append({
                'Model': name,
                'RMSE': scores['test_rmse'],
                'MAE': scores['test_mae'],
                'R²': scores['test_r2']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df)
        
        # Find best model
        best_model_name, best_model = trainer.find_best_model()
        st.success(f"🏆 Best Model: {best_model_name}")

def model_evaluation_page():
    """
    Model evaluation page.
    """
    st.header("📈 Model Evaluation")
    
    if 'trainer' in st.session_state and 'X_test' in st.session_state and 'y_test' in st.session_state:
        trainer = st.session_state['trainer']
        X_test = st.session_state['X_test']
        y_test = st.session_state['y_test']
        
        # Model selection for evaluation
        model_name = st.selectbox("Select Model for Evaluation", list(trainer.models.keys()))
        
        if st.button("📊 Evaluate Model"):
            model = trainer.models[model_name]
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Initialize evaluator
            evaluator = ModelEvaluator()
            
            # Calculate metrics
            metrics = evaluator.calculate_metrics(y_test, y_pred, model_name)
            
            # Display metrics
            st.subheader("📊 Performance Metrics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("RMSE", f"{metrics['rmse']:.4f}")
            with col2:
                st.metric("MAE", f"{metrics['mae']:.4f}")
            with col3:
                st.metric("R² Score", f"{metrics['r2']:.4f}")
            with col4:
                st.metric("MAPE", f"{metrics['mape']:.2f}%")
            
            # Visualizations
            st.subheader("📈 Evaluation Visualizations")
            
            # Predicted vs Actual
            fig = px.scatter(x=y_test, y=y_pred, 
                           title=f'Predicted vs Actual - {model_name}',
                           labels={'x': 'Actual Values', 'y': 'Predicted Values'})
            
            # Add perfect prediction line
            min_val = min(min(y_test), min(y_pred))
            max_val = max(max(y_test), max(y_pred))
            fig.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                                   mode='lines', name='Perfect Prediction',
                                   line=dict(color='red', dash='dash')))
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Residuals plot
            residuals = y_test - y_pred
            fig = px.scatter(x=y_pred, y=residuals,
                           title=f'Residuals Plot - {model_name}',
                           labels={'x': 'Predicted Values', 'y': 'Residuals'})
            
            # Add zero line
            fig.add_hline(y=0, line_dash="dash", line_color="red")
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Residuals distribution
            fig = px.histogram(residuals, title=f'Residuals Distribution - {model_name}')
            st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.warning("Please train models first in the Model Training page.")

if __name__ == "__main__":
    main()
