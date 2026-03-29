"""
Main preprocessing pipeline for Retail ML project.
"""
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
import joblib
from utils import load_data, parse_registration_date, feature_engineering, detect_outliers, plot_correlation

# Config
RAW_PATH = 'data/raw/retail_customers_COMPLETE_CATEGORICAL.csv'
PROCESSED_PATH = 'data/processed/processed_data.csv'
TRAIN_TEST_PATHS = {
    'X_train': 'data/train_test/X_train.csv',
    'X_test': 'data/train_test/X_test.csv',
    'y_train': 'data/train_test/y_train.csv',
    'y_test': 'data/train_test/y_test.csv'
}
PCA_VAR = 0.95  # Retain 95% variance
TEST_SIZE = 0.2
RANDOM_STATE = 42

def main():
    # 1. Load and initial exploration
    df = load_data(RAW_PATH)
    print("Initial shape:", df.shape)
    print(df['Churn'].value_counts())  # Target

    # 2. Parse dates and feature engineering
    df = parse_registration_date(df)
    df = feature_engineering(df)

    # 3. Identify numerical/categorical columns (adjust based on PDF features)
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.drop('Churn', errors='ignore').tolist()
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    print("Num cols:", len(num_cols), "Cat cols:", len(cat_cols))

    # Detect outliers (cap them) - use available numerical cols
    outlier_cols = ['Recency', 'Frequency', 'MonetaryTotal']
    available_outliers = [col for col in outlier_cols if col in df.columns]
    if available_outliers:
        outliers = detect_outliers(df, available_outliers, threshold=1.5)
        for col, idxs in outliers.items():
            if idxs:
                df.loc[idxs, col] = df[col].median()  # Cap to median
        print(f"Capped outliers in: {available_outliers}")

    # 4. Preprocessing pipeline
    # Impute num: KNN, cat: constant 'missing'
    num_transformer = Pipeline([
        ('imputer', KNNImputer(n_neighbors=5)),
        ('scaler', StandardScaler())
    ])
    cat_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))  # Ordinal for simplicity
    ])
    preprocessor = ColumnTransformer([
        ('num', num_transformer, num_cols),
        ('cat', cat_transformer, cat_cols)
    ])

    # Separate X, y (target: Churn)
    X = df.drop('Churn', axis=1)
    y = df['Churn']

    # Fit and transform
    X_processed = preprocessor.fit_transform(X)
    joblib.dump(preprocessor, 'models/preprocessor.pkl')

    # 5. PCA
    pca = PCA(n_components=PCA_VAR)
    X_pca = pca.fit_transform(X_processed)
    joblib.dump(pca, 'models/pca.pkl')
    print(f"PCA: {X_pca.shape[1]} components, explained variance: {pca.explained_variance_ratio_.sum():.3f}")

    # 6. Train/test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X_pca, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # 7. Save
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('data/train_test', exist_ok=True)
    pd.DataFrame(X_processed).to_csv(PROCESSED_PATH, index=False)
    pd.DataFrame(X_train).to_csv(TRAIN_TEST_PATHS['X_train'], index=False)
    pd.DataFrame(X_test).to_csv(TRAIN_TEST_PATHS['X_test'], index=False)
    y_train.to_csv(TRAIN_TEST_PATHS['y_train'], index=False)
    y_test.to_csv(TRAIN_TEST_PATHS['y_test'], index=False)

    # 8. Reports
    plot_correlation(pd.DataFrame(X_processed))
    print("Preprocessing complete! Check data/processed/ and data/train_test/")

if __name__ == "__main__":
    main()
