"""
utils.py - Fonctions utilitaires partagées
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import os

def save_model(model, filepath='models/model.pkl'):
    """Sauvegarde un modèle entraîné"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)
    print(f" Modèle sauvegardé: {filepath}")

def save_metrics(metrics, filepath='reports/metrics.txt'):
    """Sauvegarde les métriques"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value:.4f}\n")
    print(f" Métriques sauvegardées: {filepath}")

def plot_confusion_matrix(y_true, y_pred, save_path='reports/confusion_matrix.png'):
    """Affiche et sauvegarde la matrice de confusion"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Fidèle (0)', 'Churn (1)'],
                yticklabels=['Fidèle (0)', 'Churn (1)'])
    plt.title('Matrice de Confusion')
    plt.ylabel('Vrai label')
    plt.xlabel('Prédiction')
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show() # Assure l'affichage à l'écran
    plt.close()
    return cm



def plot_roc_curve(y_true, y_pred_proba, save_path='reports/roc_curve.png'):
    """Affiche et sauvegarde la courbe ROC"""
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('Taux de Faux Positifs')
    plt.ylabel('Taux de Vrais Positifs')
    plt.title('Courbe ROC')
    plt.legend(loc="lower right")
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show() # Assure l'affichage à l'écran
    plt.close()
    return roc_auc



def plot_feature_importance(model, feature_names, top_n=10,
                            save_path='reports/feature_importance.png'):
    """Affiche l'importance des features (Adapté PCA)"""
    if not hasattr(model, 'feature_importances_'):
        return None
    
    importances = model.feature_importances_
    
    # Sécurité : Si top_n est plus grand que le nombre de colonnes réelles
    actual_top_n = min(len(importances), top_n)
    indices = np.argsort(importances)[::-1][:actual_top_n]
    
    plt.figure(figsize=(10, 8))
    plt.title(f'Top {actual_top_n} Features (Composantes PCA)')
    plt.barh(range(actual_top_n), importances[indices], align='center')
    plt.yticks(range(actual_top_n), [feature_names[i] for i in indices])
    plt.xlabel('Importance')
    plt.gca().invert_yaxis()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    plt.close()

def print_classification_report(y_true, y_pred):
    """Affiche le rapport de classification"""
    print("\n" + "="*60)
    print("RAPPORT DE CLASSIFICATION")
    print("="*60)
    print(classification_report(y_true, y_pred,
                                target_names=['Fidèle', 'Churn']))

def calculate_metrics(y_true, y_pred, y_pred_proba):
    """Calcule toutes les métriques"""
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1-Score': f1_score(y_true, y_pred),
        'ROC-AUC': roc_auc_score(y_true, y_pred_proba)
    }
    
    print("\n" + "="*60)
    print("MÉTRIQUES")
    print("="*60)
    for metric, value in metrics.items():
        print(f"{metric:12}: {value:.4f}")
    
    return metrics

def get_resampling_strategy(y_train):
    """Analyse le déséquilibre"""
    from collections import Counter
    counter = Counter(y_train)
    print(f"\n Distribution des classes:")
    print(f"   Fidèle (0): {counter[0]} ({counter[0]/len(y_train)*100:.1f}%)")
    print(f"   Churn (1):  {counter[1]} ({counter[1]/len(y_train)*100:.1f}%)")
    return counter

def compare_models(results_dict):
    """Compare plusieurs modèles"""
    df_results = pd.DataFrame(results_dict).T
    df_results = df_results.reset_index().rename(columns={'index': 'Model'})
    print("\nCOMPARAISON FINALE DES MODÈLES:")
    print(df_results.round(4))
    os.makedirs('../output', exist_ok=True)
    df_results.to_csv('../output/model_comparison.csv', index=False)
    return df_results