# 📊 Retail Intelligence : Segmentation, Churn & Spending Prediction

## 🚀 Vision Business & Objectifs

Dans le secteur du e-commerce, la donnée est le levier principal de la personnalisation. Ce projet répond à trois problématiques stratégiques pour optimiser la relation client :

- **La Segmentation (Clustering)** : *"Qui sont mes clients ?"*  
  Identifier les profils (VIP, Acheteurs de Gros, Standards, À risque) pour personnaliser le marketing.

- **La Classification (Churn)** : *"Qui risque de partir ?"*  
  Anticiper l'attrition pour agir sur la rétention.

- **La Régression (Spending)** : *"Quel est le potentiel financier ?"*  
  Estimer le `MonetaryTotal` futur pour prioriser les investissements.

---

## 🏗️ Architecture du Pipeline ML

Le projet est divisé en deux pipelines distincts pour assurer une spécialisation des modèles :

### 🔹 Pipeline A : Churn & Segmentation

- `src/preprocessing.py` : Nettoyage, encodage des 13 variables, réduction de dimension par PCA (10 composantes) et génération des clusters via K-Means.
- `src/train_model.py` : Gestion du déséquilibre des classes par SMOTE et entraînement du Random Forest Classifier.

### 🔹 Pipeline B : Prédiction des Dépenses (Spending)

- `src/preprocessing-reg.py` : Préparation spécifique pour la variable cible continue, gestion des outliers financiers et normalisation.
- `src/train_reg.py` : Entraînement du Random Forest Regressor avec optimisation des hyperparamètres.

---

## 💻 Interface de Déploiement

`app.py` : Application Flask web pour API prédictions en temps réel.

---

## 📈 Performances des Modèles

| Tâche           | Modèle           | Métrique Clé | Score |
|----------------|------------------|-------------|-------|
| Classification | Random Forest    | F1-Score    | 0.84  |
| Régression     | RF Regressor     | R² Score    | 0.91  |

---

## 📂 Organisation du Projet

```plaintext
├── data/
│ ├── raw/ # Dataset original
│ ├── processed/ # Données après PCA / Prétraitement
│ └── results/ # Exports des prédictions
├── models/
│ ├── best_model.pkl # Modèle Churn (Classification)
│ ├── regression_model.pkl # Modèle Spending (Régression)
│ ├── kmeans_model.pkl # Modèle Clusters (Segmentation)
│ ├── pca_model.pkl # Transformateur PCA
│ └── scaler.pkl # Standardisation
├── src/
│ ├── preprocessing.py # Pipeline Clustering/Churn
│ ├── preprocessing-reg.py # Pipeline Régression
│ ├── train_model.py # Training Churn (SMOTE)
│ ├── train_reg.py # Training Régression
│ ├── predict.py # Tests d'inférence
│ └── utils.py # Visualisations et utilitaires
├── app/
|  ├── app.py # Flask Web Application (API Predictions)
└── requirements.txt # Dépendances (scikit-learn, pandas, etc.)
```
---

## 🛠️ Installation et Utilisation

### 1. Environnement virtuel (recommandé)
```bash
python -m venv venv
# Windows:
venv\\Scripts\\activate
```

### 2. Installation des dépendances
```bash
pip install -r requirements.txt
```

### 3. Entraînement Classification & Clustering:


```bash
python src/preprocessing.py
python src/train_model.py
```
### 3.Entraînement du volet Régression :

```bash
python src/preprocessing-reg.py
python src/train_reg.py
```
### Lancement de l'interface :

```bash
python app/app.py
```
Auteur : Siwar ADALI – Engineering Student at ENIS (GI2).



