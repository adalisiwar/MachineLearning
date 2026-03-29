"""
Utility functions for ML Retail project.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy import stats
import os

def load_data(raw_path='data/raw/retail_customers_COMPLETE_CATEGORICAL.csv'):
    """Load raw CSV data."""
    df = pd.read_csv(raw_path)
    print(f"Loaded data: {df.shape}")
    return df

def parse_registration_date(df):
    """Parse inconsistent RegistrationDate formats (defensive)."""
    if 'RegistrationDate' not in df.columns:
        return df
    
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=UserWarning)
        df['RegistrationDate'] = pd.to_datetime(df['RegistrationDate'], dayfirst=True, errors='coerce', format='mixed')
    df['RegYear'] = df['RegistrationDate'].dt.year
    df['RegMonth'] = df['RegistrationDate'].dt.month
    df['RegDay'] = df['RegistrationDate'].dt.day
    df['RegWeekday'] = df['RegistrationDate'].dt.weekday
    return df.drop('RegistrationDate', axis=1)

def plot_correlation(df, num_features=20):
    """Plot correlation heatmap for top features."""
    # Use only numeric columns
    num_df = df.select_dtypes(include=[np.number])
    if num_df.shape[1] > 1:
        corr_matrix = num_df.corr()
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0, square=True)
        plt.title('Feature Correlation Heatmap')
        os.makedirs('reports', exist_ok=True)
        plt.savefig('reports/correlation_heatmap.png')
        plt.tight_layout()
        plt.show()
    else:
        print("Skipping correlation plot: insufficient numeric features.")

def detect_outliers(df, columns, method='iqr', threshold=1.5):
    """Detect outliers in numerical columns."""
    outliers = {}
    for col in columns:
        if method == 'iqr':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - threshold * IQR
            upper = Q3 + threshold * IQR
            outliers[col] = df[(df[col] < lower) | (df[col] > upper)].index.tolist()
    return outliers

def feature_engineering(df):
    """Create new features per PDF (defensive)."""
    if 'MonetaryTotal' in df.columns and 'Recency' in df.columns:
        df['MonetaryPerDay'] = df['MonetaryTotal'] / (df['Recency'] + 1)
    if 'MonetaryTotal' in df.columns and 'Frequency' in df.columns:
        df['AvgBasketValue'] = df['MonetaryTotal'] / df['Frequency']
    if 'Recency' in df.columns and 'CustomerTenureDays' in df.columns:
        df['TenureRatio'] = df['Recency'] / (df['CustomerTenureDays'] + 1)
    return df

if __name__ == "__main__":
    df = load_data()
    print(df.head())
