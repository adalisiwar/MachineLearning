import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_for_regression():
    print(" Création d'un dataset spécifique pour la régression (Raw Features)...")

    # 1. Chargement des données brutes
    # On privilégie le fichier complet mentionné dans l'erreur
    possible_files = [
        'data/raw/retail_customers_COMPLETE_CATEGORICAL.csv', 
        'data/cleaned_data.csv', 
        'data/customer_data.csv'
    ]
    df = None
    
    for file in possible_files:
        if os.path.exists(file):
            df = pd.read_csv(file)
            print(f" Fichier trouvé et chargé : {file}")
            break
            
    if df is None:
        print(" Erreur : Aucun fichier de données trouvé.")
        return

    # 2. Sélection des variables explicatives
    # On identifie dynamiquement la colonne cible
    possible_targets = ['MonetaryTotal', 'Monetary', 'TotalSpending']
    target_col = next((c for c in possible_targets if c in df.columns), None)
    
    if not target_col:
        print(f" Erreur : Colonne cible non trouvée. Colonnes : {df.columns.tolist()}")
        return
    
    # NETTOYAGE CRUCIAL : On ne garde que les colonnes numériques
    # L'erreur venait des colonnes comme 'InvoiceDate' ou des identifiants clients
    numeric_df = df.select_dtypes(include=[np.number])
    
    # On exclut la cible et les colonnes non-prédictives identifiées
    exclude = [target_col, 'Churn', 'CustomerID', 'customer_id', 'cluster', 'Cluster', 'Unnamed: 0']
    features = numeric_df.drop(columns=[c for c in exclude if c in numeric_df.columns], errors='ignore')
    target = df[target_col]

    # Suppression des lignes avec des valeurs manquantes (NaN) s'il y en a
    # Le StandardScaler ne supporte pas les NaN
    full_data = pd.concat([features, target], axis=1).dropna()
    features = full_data.drop(columns=[target_col])
    target = full_data[target_col]

    print(f" Variables sélectionnées : {features.columns.tolist()}")

    # 3. Traitement des Outliers
    upper_limit = target.quantile(0.99)
    target = np.clip(target, 0, upper_limit)

    # 4. Split Train/Test
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )

    # 5. Scaling (Standardisation)
    # Maintenant que X_train ne contient que des flottants/entiers, cela fonctionnera
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 6. Sauvegarde
    os.makedirs('data/regression_specific', exist_ok=True)
    
    pd.DataFrame(X_train_scaled, columns=features.columns).to_csv('data/regression_specific/X_train_reg.csv', index=False)
    pd.DataFrame(X_test_scaled, columns=features.columns).to_csv('data/regression_specific/X_test_reg.csv', index=False)
    pd.Series(y_train).to_csv('data/regression_specific/y_train_reg.csv', index=False)
    pd.Series(y_test).to_csv('data/regression_specific/y_test_reg.csv', index=False)

    print(f" Dataset de régression créé avec {features.shape[1]} variables numériques.")
    print(" Fichiers sauvegardés dans 'data/regression_specific/'")

if __name__ == "__main__":
    preprocess_for_regression()