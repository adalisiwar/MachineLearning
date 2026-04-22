import pandas as pd
import numpy as np
import joblib
import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# Configuration des chemins vers les modèles sauvegardés
# Note : 'classifier' pointe maintenant vers le nom généré dans le Canvas
MODEL_PATHS = {
    'classifier': 'models/best_model_churn.pkl',
    'regressor': 'models/regression_model.pkl',
    'kmeans': 'models/kmeans_model.pkl'
}

# Chemins des données de test
DATA_TEST_PATH = 'data/train_test/X_test.csv'
TARGET_TEST_PATH = 'data/train_test/y_test.csv'

def run_comprehensive_predictions():
    print(" Lancement des tests de prédiction (Classification, Régression, Clustering)...")

    # 1. Chargement des modèles
    models = {}
    try:
        for key, path in MODEL_PATHS.items():
            if os.path.exists(path):
                models[key] = joblib.load(path)
                print(f" Modèle '{key}' chargé avec succès.")
            else:
                print(f" Attention : Le modèle '{key}' à l'emplacement {path} est introuvable.")
    except Exception as e:
        print(f" Erreur lors du chargement des modèles : {e}")
        return

    # 2. Chargement des données de test
    if not os.path.exists(DATA_TEST_PATH):
        print(f" Erreur : {DATA_TEST_PATH} introuvable. Lancez le preprocessing d'abord.")
        return
        
    X_test = pd.read_csv(DATA_TEST_PATH)
    
    # Chargement de la cible réelle pour comparaison (si disponible)
    y_true = None
    if os.path.exists(TARGET_TEST_PATH):
        y_true = pd.read_csv(TARGET_TEST_PATH).values.ravel()

    print(f" Analyse de {len(X_test)} clients sur {X_test.shape[1]} composantes PCA.")

    # 3. Calcul des prédictions
    results = pd.DataFrame(index=range(len(X_test)))

    # --- Classification (Churn) ---
    if 'classifier' in models:
        results['Predicted_Churn'] = models['classifier'].predict(X_test)
        results['Churn_Probability_%'] = (models['classifier'].predict_proba(X_test)[:, 1] * 100).round(2)
    
    # --- Régression (Dépenses) ---
    if 'regressor' in models:
        # Note : Si votre régisseur a été entraîné sur le log (retrain_regression.py), 
        # il faut appliquer np.expm1()
        raw_preds = models['regressor'].predict(X_test)
        # On tente de détecter si c'est une échelle log ou réelle (basé sur la moyenne)
        if raw_preds.mean() < 20: # Probablement une échelle log
            results['Predicted_Spending_DT'] = np.expm1(raw_preds).round(2)
        else:
            results['Predicted_Spending_DT'] = raw_preds.round(2)

    # --- Clustering (Segments) ---
    if 'kmeans' in models:
        # Utilisation de .values pour éviter les warnings de noms de colonnes
        results['Customer_Segment'] = models['kmeans'].predict(X_test.values)
        # Mappage des noms de segments (à adapter selon vos conclusions K-Means)
        segment_map = {0: "Stable", 1: "VIP", 2: "Occasionnel", 3: "À Risque"}
        results['Segment_Name'] = results['Customer_Segment'].map(segment_map)

    # Ajout des valeurs réelles si présentes
    if y_true is not None:
        results['Real_Status'] = y_true

    # 4. Sauvegarde des résultats
    os.makedirs('data/results', exist_ok=True)
    output_path = 'data/results/test_predictions_complet.csv'
    results.to_csv(output_path, index=False)

    # 5. Affichage du résumé statistique
    print("\n" + "="*60)
    print("  RÉSUMÉ DES PRÉDICTIONS")
    print("="*60)
    print(f"Nombre total de clients analysés  : {len(results)}")
    
    if 'Predicted_Churn' in results:
        churn_count = results['Predicted_Churn'].sum()
        churn_rate = (churn_count / len(results)) * 100
        print(f"Clients détectés en Churn        : {churn_count} ({churn_rate:.2f}%)")
    
    if 'Predicted_Spending_DT' in results:
        print(f"Estimation dépense moyenne        : {results['Predicted_Spending_DT'].mean():.2f} DT")
        print(f"Potentiel financier total estimé : {results['Predicted_Spending_DT'].sum():.2f} DT")
    
    if 'Segment_Name' in results:
        print("\nRépartition des segments :")
        print(results['Segment_Name'].value_counts())
    
    print("="*60)
    print(f"\n Résultats détaillés exportés : {output_path}")

if __name__ == "__main__":
    run_comprehensive_predictions()