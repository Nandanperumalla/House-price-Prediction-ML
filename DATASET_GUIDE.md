# 📊 How to Add the Zillow Dataset

## 🎯 **Quick Start**

### **Step 1: Download the Dataset**

1. **Go to Kaggle**: Visit [Zillow Prize: Zillow's Home Value Prediction](https://www.kaggle.com/c/zillow-prize-1)

2. **Create Account**: Sign up for a free Kaggle account if you don't have one

3. **Download Files**: Download these essential files:
   - `train_2016.csv` - Training data with logerror (target variable)
   - `properties_2016.csv` - Property features
   - `sample_submission.csv` - Submission format (optional)

### **Step 2: Place Files in Project**

```
house-price-prediction/
├── data/
│   ├── train_2016.csv          # ← Place downloaded file here
│   ├── properties_2016.csv     # ← Place downloaded file here
│   └── sample_submission.csv   # ← Place downloaded file here (optional)
```

### **Step 3: Run Data Preparation**

```bash
cd "/Users/nandanp/Documents/ml/Ml project/house-price-prediction"
source venv/bin/activate
python prepare_data.py
```

### **Step 4: Train with Real Data**

```bash
python train_with_real_data.py
```

---

## 📋 **Detailed Instructions**

### **Option 1: Manual Download (Recommended)**

1. **Visit Kaggle Competition Page**:
   - URL: https://www.kaggle.com/c/zillow-prize-1
   - Click "Download All" or download individual files

2. **Required Files**:
   - **train_2016.csv**: Contains `parcelid`, `logerror`, `transactiondate`
   - **properties_2016.csv**: Contains property features like bedrooms, bathrooms, square footage, etc.

3. **File Structure**:
   ```
   data/
   ├── train_2016.csv          # ~2.9M rows, 3 columns
   ├── properties_2016.csv     # ~2.9M rows, 58 columns
   └── sample_submission.csv  # Submission format
   ```

### **Option 2: Using Kaggle API**

1. **Install Kaggle API**:
   ```bash
   pip install kaggle
   ```

2. **Setup API Credentials**:
   - Go to Kaggle Account Settings
   - Create API token
   - Place `kaggle.json` in `~/.kaggle/`

3. **Download Dataset**:
   ```bash
   kaggle competitions download -c zillow-prize-1 -p data/
   ```

### **Option 3: Sample Data (For Testing)**

If you want to test the pipeline without downloading the full dataset:

```bash
python prepare_data.py
```

This will create sample datasets that mimic the Zillow structure.

---

## 🔍 **Dataset Overview**

### **Training Data (train_2016.csv)**
- **parcelid**: Unique property identifier
- **logerror**: Target variable (log of error between actual and predicted price)
- **transactiondate**: Date of transaction

### **Properties Data (properties_2016.csv)**
Key features include:
- **bedroomcnt**: Number of bedrooms
- **bathroomcnt**: Number of bathrooms
- **calculatedfinishedsquarefeet**: Square footage
- **taxvaluedollarcnt**: Tax assessed value
- **yearbuilt**: Year property was built
- **regionidzip**: ZIP code
- **latitude/longitude**: Geographic coordinates
- **poolcnt**: Number of pools
- **fireplacecnt**: Number of fireplaces
- And many more...

---

## 🚀 **Usage Examples**

### **1. Analyze Dataset Structure**
```bash
python prepare_data.py
```

### **2. Train Models with Real Data**
```bash
python train_with_real_data.py
```

### **3. Train Models with Sample Data**
```bash
python train.py
```

### **4. Launch Web Application**
```bash
streamlit run app.py
```

---

## ⚠️ **Important Notes**

1. **File Size**: The full dataset is quite large (~1GB+)
2. **Memory Requirements**: Training may require 8GB+ RAM
3. **Processing Time**: Full dataset training can take 30+ minutes
4. **Missing Values**: The dataset has many missing values that need handling

---

## 🔧 **Troubleshooting**

### **Common Issues**:

1. **"File not found" error**:
   - Ensure files are in the correct `data/` directory
   - Check file names match exactly

2. **Memory errors**:
   - Use sample data for testing
   - Reduce dataset size for initial testing

3. **Missing columns**:
   - The dataset structure may vary
   - Check column names in your downloaded files

### **Data Validation**:
```python
# Check if files exist
import os
print("train_2016.csv exists:", os.path.exists("data/train_2016.csv"))
print("properties_2016.csv exists:", os.path.exists("data/properties_2016.csv"))

# Check file sizes
if os.path.exists("data/train_2016.csv"):
    size = os.path.getsize("data/train_2016.csv") / (1024*1024)  # MB
    print(f"train_2016.csv size: {size:.1f} MB")
```

---

## 📈 **Expected Results**

With the real Zillow dataset, you should see:
- **Better model performance** (R² > 0.1)
- **More realistic predictions**
- **Better feature importance insights**
- **More comprehensive EDA results**

---

## 🎉 **Next Steps**

Once you have the dataset:
1. Run the data preparation script
2. Train models with real data
3. Compare performance with sample data
4. Experiment with feature engineering
5. Deploy the web application

**Happy Modeling! 🏠📈**
