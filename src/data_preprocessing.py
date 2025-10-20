"""
Data Preprocessing Module for House Price Prediction

This module handles data cleaning, feature engineering, and preprocessing
for the Zillow house price prediction dataset.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')


class DataPreprocessor:
    """
    A comprehensive data preprocessing class for house price prediction.
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.imputer = SimpleImputer(strategy='median')
        self.feature_columns = None
        self.is_fitted = False
        
    def load_data(self, file_path):
        """
        Load the dataset from CSV file.
        
        Args:
            file_path (str): Path to the CSV file
            
        Returns:
            pd.DataFrame: Loaded dataset
        """
        try:
            df = pd.read_csv(file_path)
            print(f"Dataset loaded successfully. Shape: {df.shape}")
            return df
        except FileNotFoundError:
            print(f"Error: File {file_path} not found.")
            return None
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def basic_info(self, df):
        """
        Display basic information about the dataset.
        
        Args:
            df (pd.DataFrame): Input dataset
        """
        print("=" * 50)
        print("DATASET INFORMATION")
        print("=" * 50)
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print("\nData Types:")
        print(df.dtypes)
        print("\nMissing Values:")
        missing_data = df.isnull().sum()
        missing_percent = (missing_data / len(df)) * 100
        missing_df = pd.DataFrame({
            'Missing Count': missing_data,
            'Missing Percentage': missing_percent
        })
        print(missing_df[missing_df['Missing Count'] > 0])
        
    def handle_missing_values(self, df, strategy='drop'):
        """
        Handle missing values in the dataset.
        
        Args:
            df (pd.DataFrame): Input dataset
            strategy (str): Strategy to handle missing values ('drop', 'impute', 'median')
            
        Returns:
            pd.DataFrame: Dataset with handled missing values
        """
        df_clean = df.copy()
        
        if strategy == 'drop':
            # Drop rows with missing values in target variable
            if 'logerror' in df_clean.columns:
                df_clean = df_clean.dropna(subset=['logerror'])
            
            # Drop columns with more than 50% missing values
            threshold = 0.5
            cols_to_drop = df_clean.columns[df_clean.isnull().mean() > threshold]
            if len(cols_to_drop) > 0:
                print(f"Dropping columns with >{threshold*100}% missing values: {list(cols_to_drop)}")
                df_clean = df_clean.drop(columns=cols_to_drop)
                
        elif strategy == 'impute':
            # Impute numerical columns with median
            numerical_cols = df_clean.select_dtypes(include=[np.number]).columns
            df_clean[numerical_cols] = self.imputer.fit_transform(df_clean[numerical_cols])
            
        print(f"Missing values handled. New shape: {df_clean.shape}")
        return df_clean
    
    def remove_outliers(self, df, columns=None, method='iqr'):
        """
        Remove outliers from the dataset.
        
        Args:
            df (pd.DataFrame): Input dataset
            columns (list): Columns to check for outliers
            method (str): Method to detect outliers ('iqr', 'zscore')
            
        Returns:
            pd.DataFrame: Dataset with outliers removed
        """
        df_clean = df.copy()
        
        if columns is None:
            # Select numerical columns
            columns = df_clean.select_dtypes(include=[np.number]).columns.tolist()
            # Remove target variable if present
            if 'logerror' in columns:
                columns.remove('logerror')
        
        initial_shape = df_clean.shape
        
        for col in columns:
            if col in df_clean.columns:
                if method == 'iqr':
                    Q1 = df_clean[col].quantile(0.25)
                    Q3 = df_clean[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    df_clean = df_clean[(df_clean[col] >= lower_bound) & 
                                      (df_clean[col] <= upper_bound)]
                elif method == 'zscore':
                    z_scores = np.abs((df_clean[col] - df_clean[col].mean()) / df_clean[col].std())
                    df_clean = df_clean[z_scores < 3]
        
        print(f"Outliers removed. Shape changed from {initial_shape} to {df_clean.shape}")
        return df_clean
    
    def feature_engineering(self, df):
        """
        Create new features from existing ones.
        
        Args:
            df (pd.DataFrame): Input dataset
            
        Returns:
            pd.DataFrame: Dataset with new features
        """
        df_fe = df.copy()
        
        # Create price per square foot if both price and square footage exist
        if 'taxvaluedollarcnt' in df_fe.columns and 'calculatedfinishedsquarefeet' in df_fe.columns:
            df_fe['price_per_sqft'] = df_fe['taxvaluedollarcnt'] / df_fe['calculatedfinishedsquarefeet']
            df_fe['price_per_sqft'] = df_fe['price_per_sqft'].replace([np.inf, -np.inf], np.nan)
        
        # Create total rooms feature
        if 'bedroomcnt' in df_fe.columns and 'bathroomcnt' in df_fe.columns:
            df_fe['total_rooms'] = df_fe['bedroomcnt'] + df_fe['bathroomcnt']
        
        # Create age of property if year built exists
        if 'yearbuilt' in df_fe.columns:
            current_year = 2023  # Assuming current year
            df_fe['property_age'] = current_year - df_fe['yearbuilt']
            df_fe['property_age'] = df_fe['property_age'].clip(lower=0)  # Remove negative ages
        
        # Create log transformations for skewed features
        skewed_features = ['taxvaluedollarcnt', 'calculatedfinishedsquarefeet']
        for feature in skewed_features:
            if feature in df_fe.columns:
                df_fe[f'log_{feature}'] = np.log1p(df_fe[feature])
        
        print("Feature engineering completed.")
        return df_fe
    
    def encode_categorical_features(self, df):
        """
        Encode categorical features.
        
        Args:
            df (pd.DataFrame): Input dataset
            
        Returns:
            pd.DataFrame: Dataset with encoded categorical features
        """
        df_encoded = df.copy()
        
        # Identify categorical columns
        categorical_cols = df_encoded.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            if col in df_encoded.columns:
                # Use label encoding for categorical variables
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                self.label_encoders[col] = le
        
        print(f"Categorical features encoded: {list(categorical_cols)}")
        return df_encoded
    
    def prepare_features(self, df, target_column='logerror'):
        """
        Prepare features for model training.
        
        Args:
            df (pd.DataFrame): Input dataset
            target_column (str): Name of the target column
            
        Returns:
            tuple: (X, y) where X is features and y is target
        """
        df_prep = df.copy()
        
        # Remove non-feature columns
        non_feature_cols = ['parcelid', target_column]
        feature_cols = [col for col in df_prep.columns if col not in non_feature_cols]
        
        # Select only numerical features
        numerical_features = df_prep[feature_cols].select_dtypes(include=[np.number]).columns
        X = df_prep[numerical_features]
        y = df_prep[target_column] if target_column in df_prep.columns else None
        
        # Store feature columns for later use
        self.feature_columns = numerical_features.tolist()
        
        print(f"Features prepared. Shape: {X.shape}")
        return X, y
    
    def scale_features(self, X_train, X_test=None):
        """
        Scale features using StandardScaler.
        
        Args:
            X_train (pd.DataFrame): Training features
            X_test (pd.DataFrame): Test features (optional)
            
        Returns:
            tuple: Scaled training and test features
        """
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        if X_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
            return X_train_scaled, X_test_scaled
        
        self.is_fitted = True
        return X_train_scaled
    
    def preprocess_pipeline(self, df, target_column='logerror'):
        """
        Complete preprocessing pipeline.
        
        Args:
            df (pd.DataFrame): Input dataset
            target_column (str): Name of the target column
            
        Returns:
            tuple: (X, y) processed features and target
        """
        print("Starting preprocessing pipeline...")
        
        # Step 1: Handle missing values
        df_clean = self.handle_missing_values(df, strategy='impute')
        
        # Step 2: Remove outliers
        df_no_outliers = self.remove_outliers(df_clean)
        
        # Step 3: Feature engineering
        df_featured = self.feature_engineering(df_no_outliers)
        
        # Step 4: Encode categorical features
        df_encoded = self.encode_categorical_features(df_featured)
        
        # Step 5: Prepare features
        X, y = self.prepare_features(df_encoded, target_column)
        
        print("Preprocessing pipeline completed successfully!")
        return X, y


def create_sample_data():
    """
    Create sample data for testing the preprocessing pipeline.
    
    Returns:
        pd.DataFrame: Sample dataset
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
    
    # Add some missing values
    missing_indices = np.random.choice(n_samples, size=50, replace=False)
    # Convert to float first to allow NaN values
    data['calculatedfinishedsquarefeet'] = data['calculatedfinishedsquarefeet'].astype(float)
    data['calculatedfinishedsquarefeet'][missing_indices] = np.nan
    
    return pd.DataFrame(data)


if __name__ == "__main__":
    # Test the preprocessing pipeline with sample data
    preprocessor = DataPreprocessor()
    
    # Create sample data
    sample_df = create_sample_data()
    
    # Display basic info
    preprocessor.basic_info(sample_df)
    
    # Run preprocessing pipeline
    X, y = preprocessor.preprocess_pipeline(sample_df)
    
    print(f"\nFinal processed data shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Feature columns: {preprocessor.feature_columns}")
