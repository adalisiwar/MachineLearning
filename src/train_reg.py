import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns

def train_regression_on_raw_data():
    print(" Entraînement du modèle de régression sur données brutes...")

    # 1. Chargement des données spécifiques créées par le préprocesseur
    try:
        X_train = pd.read_csv('data/regression_specific/X_train_reg.csv')
        X_test = pd.read_csv('data/regression_specific/X_test_reg.csv')
        y_train = pd.read_csv('data/regression_specific/y_train_reg.csv').values.ravel()
        y_test = pd.read_csv('data/regression_specific/y_test_reg.csv').values.ravel()
    except Exception as e:
        print(f" Erreur : Fichiers spécifiques introuvables. Lancez d'abord le preprocessing. ({e})")
        return

    # 2. Transformation Logarithmique
    # Indispensable pour les montants financiers (réduit l'asymétrie)
    y_train_log = np.log1p(np.maximum(y_train, 0))
    y_test_log = np.log1p(np.maximum(y_test, 0))

    # 3. Modèle RandomForest optimisé
    print(f" Apprentissage sur {X_train.shape[1]} variables...")
    model = RandomForestRegressor(
        n_estimators=200, 
        max_depth=15, 
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    
    # Validation croisée pour vérifier la robustesse (5-fold)
    print(" Calcul de la stabilité via Cross-Validation...")
    cv_scores = cross_val_score(model, X_train, y_train_log, cv=5, scoring='r2')
    
    model.fit(X_train, y_train_log)

    # 4. Évaluation sur le jeu de test
    y_pred_log = model.predict(X_test)
    y_pred_real = np.expm1(y_pred_log)

    # Calcul des métriques
    r2_log = r2_score(y_test_log, y_pred_log)
    r2_real = r2_score(y_test, y_pred_real)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_real))
    mae = mean_absolute_error(y_test, y_pred_real)

    print("\n RÉSULTATS DE L'ENTRAÎNEMENT :")
    print(f"------------------------------------")
    print(f"R² Score (Échelle Log)    : {r2_log:.4f}")
    print(f"R² Score (Échelle Réelle) : {r2_real:.4f}")
    print(f"Stabilité CV (Moyenne)    : {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
    print(f"RMSE (Erreur moyenne)     : {rmse:.2f} DT")
    print(f"MAE (Erreur absolue)      : {mae:.2f} DT")
    print(f"------------------------------------")

    # 5. Sauvegarde du modèle
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/regression_model_raw.pkl')
    print(" Modèle sauvegardé : 'models/regression_model_raw.pkl'")

    # 6. Simulation de test rapide (Inference Test)
    print("\n Test de prédiction sur les 3 premiers clients du test set :")
    for i in range(3):
        sample = X_test.iloc[[i]]
        real_val = y_test[i]
        pred_log = model.predict(sample)
        pred_real = np.expm1(pred_log)[0]
        diff = abs(real_val - pred_real)
        print(f"Client {i+1} -> Réel: {real_val:.2f} DT | Prédit: {pred_real:.2f} DT (Écart: {diff:.2f} DT)")

    # 7. Visualisations
    plt.figure(figsize=(15, 6))

    # Graphique 1 : Réel vs Prédit
    plt.subplot(1, 2, 1)
    plt.scatter(y_test, y_pred_real, alpha=0.4, color='teal')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.title(f'Performance : Réel vs Prédit (R²={r2_real:.3f})')
    plt.xlabel('Valeurs Réelles (DT)')
    plt.ylabel('Prédictions (DT)')

    # Graphique 2 : Importance des variables
    plt.subplot(1, 2, 2)
    importances = model.feature_importances_
    feat_importances = pd.Series(importances, index=X_train.columns)
    feat_importances.nlargest(10).plot(kind='barh', color='skyblue')
    plt.title('Top 10 des variables prédictives')
    plt.xlabel('Score d\'importance')

    plt.tight_layout()
    os.makedirs('reports', exist_ok=True)
    plt.savefig('reports/regression_raw_final_report.png')
    print("\n Graphique de rapport sauvegardé : 'reports/regression_raw_final_report.png'")

if __name__ == "__main__":
    train_regression_on_raw_data()