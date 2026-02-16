# Analyse Comportementale Clientèle Retail - Phase 1 : Préprocessing des Données

## Description du Projet
Ce projet vise à analyser le comportement des clients d'une entreprise e-commerce de cadeaux pour personnaliser les stratégies marketing, réduire le churn et optimiser les ventes. La première phase se concentre sur l'exploration et la préparation des données : nettoyage, imputation des valeurs manquantes, encodage des features catégorielles, normalisation, et réduction de dimension via ACP. Le dataset comprend 52 features issues de transactions réelles, avec des problèmes de qualité à résoudre (valeurs manquantes, aberrantes, formats inconsistants).

## Instructions d'Installation
- **Environnement virtuel** : Créez et activez un environnement virtuel avec Python.
  - `python -m venv venv`
  - Activation : `venv\Scripts\activate` (Windows) ou `source venv/bin/activate` (Linux/Mac).
- **Dépendances** : Installez les packages requis.
  - `pip install -r requirements.txt`
  - Générer requirements.txt après installation : `pip freeze > requirements.txt`.
- **Outils requis** : Python 3.x, VS Code, Jupyter Notebook, GitHub pour dépôt.

## Structure du Projet
Voici l'arborescence principale du projet (focus sur la phase 1) :
- `data/` : Dossier pour les données.
  - `raw/` : Données brutes originales (ex. : dataset initial avec 52 features).
  - `processed/` : Données nettoyées après preprocessing.
  - `train_test/` : Données splitées (train/test) pour les phases suivantes.
- `notebooks/` : Notebooks Jupyter pour prototypage (ex. : exploration et preprocessing).
- `src/` : Scripts Python pour production.
  - `preprocessing.py` : Script principal pour nettoyage, imputation, encodage, normalisation, ACP.
  - `utils.py` : Fonctions utilitaires (ex. : visualisation, parsing).
- `requirements.txt` : Liste des dépendances.
- `README.md` : Ce fichier de documentation.

## Guide d'Utilisation - Phase 1 : Préprocessing
Suivez ces étapes pour exécuter le preprocessing dans VS Code ou via scripts :

- **Étape 1 : Exploration des données** :
  - Chargez le dataset brut depuis `data/raw/`.
  - Utilisez `notebooks/exploration.ipynb` pour analyser la qualité (valeurs manquantes, aberrantes, corrélations).
  - Exemples : Heatmap de corrélation, détection de multicolinéarité (VIF > 10).

- **Étape 2 : Nettoyage et préparation** :
  - Exécutez `src/preprocessing.py` pour :
    - Imputer les valeurs manquantes (ex. : Age avec KNN Imputer, Satisfaction avec médiane).
    - Corriger les valeurs aberrantes (ex. : SupportTickets, Satisfaction).
    - Parser les dates (ex. : RegistrationDate avec `pd.to_datetime`).
    - Supprimer les features inutiles (ex. : NewsletterSubscribed si constante).
  - Encodage : Appliquez One-Hot ou Ordinal Encoding aux catégorielles (ex. : Region, Gender).
  - Normalisation : Utilisez `StandardScaler` sur les numériques (évitez data leakage en appliquant après split).

