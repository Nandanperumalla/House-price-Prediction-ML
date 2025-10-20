"""
Zillow Dataset Preparation and Analysis Script

This script helps you understand the Zillow dataset structure and prepare it for training.
"""

import pandas as pd
import numpy as np
import os

def analyze_zillow_dataset():
    """
    Analyze the Zillow dataset structure and provide insights.
    """
    print("🏠 Zillow Dataset Analysis")
    print("=" * 50)
    
    # Check for data files
    data_files = {
        'train': 'data/train_2016.csv',
        'properties': 'data/properties_2016.csv',
        'sample': 'data/sample_submission.csv'
    }
    
    print("📁 Checking for data files:")
    for name, path in data_files.items():
        exists = os.path.exists(path)
        status = "✅ Found" if exists else "❌ Missing"
        print(f"   {name}: {path} - {status}")
    
    # Load and analyze training data
    if os.path.exists(data_files['train']):
        print(f"\n📊 Training Data Analysis:")
        train_df = pd.read_csv(data_files['train'])
        print(f"   Shape: {train_df.shape}")
        print(f"   Columns: {list(train_df.columns)}")
        print(f"   Missing values: {train_df.isnull().sum().sum()}")
        print(f"   Target variable (logerror) stats:")
        print(f"     Mean: {train_df['logerror'].mean():.4f}")
        print(f"     Std: {train_df['logerror'].std():.4f}")
        print(f"     Min: {train_df['logerror'].min():.4f}")
        print(f"     Max: {train_df['logerror'].max():.4f}")
    
    # Load and analyze properties data
    if os.path.exists(data_files['properties']):
        print(f"\n🏡 Properties Data Analysis:")
        properties_df = pd.read_csv(data_files['properties'])
        print(f"   Shape: {properties_df.shape}")
        print(f"   Columns: {len(properties_df.columns)}")
        print(f"   Missing values: {properties_df.isnull().sum().sum()}")
        
        # Show column categories
        numerical_cols = properties_df.select_dtypes(include=[np.number]).columns
        categorical_cols = properties_df.select_dtypes(include=['object']).columns
        
        print(f"   Numerical columns: {len(numerical_cols)}")
        print(f"   Categorical columns: {len(categorical_cols)}")
        
        # Show some key columns
        key_columns = ['bedroomcnt', 'bathroomcnt', 'calculatedfinishedsquarefeet', 
                      'taxvaluedollarcnt', 'yearbuilt', 'regionidzip']
        available_key_cols = [col for col in key_columns if col in properties_df.columns]
        
        if available_key_cols:
            print(f"   Key columns available: {available_key_cols}")
    
    # Show how to merge datasets
    if os.path.exists(data_files['train']) and os.path.exists(data_files['properties']):
        print(f"\n🔗 Dataset Merging:")
        train_df = pd.read_csv(data_files['train'])
        properties_df = pd.read_csv(data_files['properties'])
        
        merged_df = train_df.merge(properties_df, on='parcelid', how='left')
        print(f"   Merged dataset shape: {merged_df.shape}")
        print(f"   Missing values after merge: {merged_df.isnull().sum().sum()}")

def create_sample_zillow_data():
    """
    Create a sample dataset that mimics the Zillow structure for testing.
    """
    print("\n🔄 Creating Sample Zillow Dataset...")
    
    np.random.seed(42)
    n_samples = 1000
    
    # Create training data (similar to train_2016.csv)
    train_data = {
        'parcelid': range(1, n_samples + 1),
        'logerror': np.random.normal(0, 0.1, n_samples),
        'transactiondate': pd.date_range('2016-01-01', periods=n_samples, freq='D')[:n_samples]
    }
    
    # Create properties data (similar to properties_2016.csv)
    properties_data = {
        'parcelid': range(1, n_samples + 1),
        'airconditioningtypeid': np.random.choice([1, 2, 3, 4, 5], n_samples),
        'bathroomcnt': np.random.uniform(1, 4, n_samples),
        'bedroomcnt': np.random.randint(1, 6, n_samples),
        'buildingqualitytypeid': np.random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], n_samples),
        'calculatedfinishedsquarefeet': np.random.randint(800, 4000, n_samples),
        'fips': np.random.choice([6037, 6059, 6111], n_samples),
        'fireplacecnt': np.random.randint(0, 3, n_samples),
        'garagecarcnt': np.random.randint(0, 3, n_samples),
        'garagetotalsqft': np.random.randint(0, 1000, n_samples),
        'hashottuborspa': np.random.choice([0, 1], n_samples),
        'heatingorsystemtypeid': np.random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], n_samples),
        'latitude': np.random.uniform(33.5, 34.5, n_samples),
        'longitude': np.random.uniform(-118.5, -117.5, n_samples),
        'lotsizesquarefeet': np.random.randint(2000, 20000, n_samples),
        'poolcnt': np.random.choice([0, 1], n_samples),
        'poolsizesum': np.random.randint(0, 1000, n_samples),
        'propertycountylandusecode': np.random.choice(['0100', '0101', '0102', '0103'], n_samples),
        'propertylandusetypeid': np.random.choice([31, 46, 47, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 273, 274, 275, 276, 279], n_samples),
        'propertyzoningdesc': np.random.choice(['LAR1', 'LAR2', 'LAR3', 'LAR4', 'LAR5'], n_samples),
        'rawcensustractandblock': np.random.randint(100000000000, 999999999999, n_samples),
        'regionidcity': np.random.randint(1000, 9999, n_samples),
        'regionidcounty': np.random.choice([3101, 3102, 3103], n_samples),
        'regionidneighborhood': np.random.randint(10000, 99999, n_samples),
        'regionidzip': np.random.randint(10000, 99999, n_samples),
        'roomcnt': np.random.randint(3, 10, n_samples),
        'storytypeid': np.random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], n_samples),
        'typeconstructiontypeid': np.random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], n_samples),
        'unitcnt': np.random.randint(1, 5, n_samples),
        'yardbuildingsqft17': np.random.randint(0, 1000, n_samples),
        'yardbuildingsqft26': np.random.randint(0, 1000, n_samples),
        'yearbuilt': np.random.randint(1900, 2020, n_samples),
        'numberofstories': np.random.randint(1, 3, n_samples),
        'structuretaxvaluedollarcnt': np.random.randint(100000, 1000000, n_samples),
        'taxvaluedollarcnt': np.random.randint(100000, 1000000, n_samples),
        'assessmentyear': np.random.choice([2015, 2016], n_samples),
        'landtaxvaluedollarcnt': np.random.randint(50000, 500000, n_samples),
        'taxamount': np.random.randint(1000, 20000, n_samples),
        'taxdelinquencyflag': np.random.choice(['Y', 'N'], n_samples),
        'taxdelinquencyyear': np.random.choice([2015, 2016, np.nan], n_samples),
        'censustractandblock': np.random.randint(100000000000, 999999999999, n_samples)
    }
    
    # Add some missing values to make it realistic
    missing_indices = np.random.choice(n_samples, size=100, replace=False)
    properties_data['calculatedfinishedsquarefeet'] = properties_data['calculatedfinishedsquarefeet'].astype(float)
    properties_data['calculatedfinishedsquarefeet'][missing_indices] = np.nan
    
    # Create DataFrames
    train_df = pd.DataFrame(train_data)
    properties_df = pd.DataFrame(properties_data)
    
    # Save sample datasets
    os.makedirs('data', exist_ok=True)
    train_df.to_csv('data/sample_train_2016.csv', index=False)
    properties_df.to_csv('data/sample_properties_2016.csv', index=False)
    
    print("✅ Sample datasets created:")
    print("   data/sample_train_2016.csv")
    print("   data/sample_properties_2016.csv")
    
    return train_df, properties_df

def main():
    """
    Main function to analyze and prepare Zillow dataset.
    """
    # Analyze existing dataset
    analyze_zillow_dataset()
    
    # Create sample data if real data is not available
    if not os.path.exists('data/train_2016.csv') or not os.path.exists('data/properties_2016.csv'):
        print("\n" + "="*50)
        print("📥 DOWNLOAD INSTRUCTIONS")
        print("="*50)
        print("To get the real Zillow dataset:")
        print("1. Go to: https://www.kaggle.com/c/zillow-prize-1")
        print("2. Download these files:")
        print("   - train_2016.csv")
        print("   - properties_2016.csv")
        print("   - sample_submission.csv")
        print("3. Place them in the 'data/' folder")
        print("4. Run: python train_with_real_data.py")
        
        # Create sample data for testing
        create_sample_zillow_data()
    
    print("\n🚀 Ready to train models!")
    print("Run: python train_with_real_data.py")

if __name__ == "__main__":
    main()
