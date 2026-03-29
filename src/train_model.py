"""
Churn classification modeling for Retail ML project.
Load preprocessed PCA data, train/tune models, evaluate, save best.
"""
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, classification_report
import xgboost as xgb
import joblib
from utils import plot_correlation  # Optional viz

# Config
TRAIN_TEST_PATHS = {
    'X_train': 'data/train_test/X_train.csv',
    'X_test': 'data/train_test/X_test.csv',
    'y_train': 'data/train_test/y_train.csv',
    'y_test': 'data/train_test/y_test.csv'
}
MODELS_DIR = 'models'

# Models & params (per PDF: hyperparam tuning)
MODELS = {
    'LogisticRegression': {
        'model': LogisticRegression(random_state=42, max_iter=1000),
        'params': {
            'C': [0.1, 1, 10],
            'penalty': ['l1', 'l2']
        }
    },
    'RandomForest': {
        'model': RandomForestClassifier(random_state=42),
        'params': {
            'n_estimators': [100, 200],
            'max_depth': [5, 10, None]
        }
    },
    'XGBoost': {
        'model': xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
        'params': {
            'n_estimators': [100, 200],
            'max_depth': [3, 6],
            'learning_rate': [0.01, 0.1]
        }
    }
}

def main():
    # 1. Load data
    X_train = pd.read_csv(TRAIN_TEST_PATHS['X_train'])
    X_test = pd.read_csv(TRAIN_TEST_PATHS['X_test'])
    y_train = pd.read_csv(TRAIN_TEST_PATHS['y_train']).squeeze()
    y_test = pd.read_csv(TRAIN_TEST_PATHS['y_test']).squeeze()

    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    print(f"Churn train: {y_train.value_counts().to_dict()}")

    # 2. Model training & tuning
    best_score = 0
    best_model = None
    results = {}

    for name, config in MODELS.items():
        print(f"\nTuning {name}...")
        gs = GridSearchCV(
            config['model'], config['params'],
            cv=5, scoring='roc_auc', n_jobs=-1, verbose=1
        )
        gs.fit(X_train, y_train)
        results[name] = gs.best_score_

        # Test eval
