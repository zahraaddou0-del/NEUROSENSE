# -*- coding: utf-8 -*-
"""
model_training.py
Entraînement local du modèle - À exécuter sur votre ordinateur
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import warnings
warnings.filterwarnings('ignore')

print("🚀 Démarrage de l'entraînement du modèle NeuroSense...")

# 1. Chargement des données
print("📂 Chargement des données...")
try:
    # Essayer de charger depuis une URL
    df = pd.read_csv("https://raw.githubusercontent.com/nagatejakachapuram/Autism-Prediction-System-ML/main/train.csv")
    print(f"✅ Données chargées : {len(df)} échantillons")
except:
    # Si échec, créer des données synthétiques
    print("⚠️ Création de données synthétiques...")
    np.random.seed(42)
    n = 1000
    df = pd.DataFrame({
        'A1_Score': np.random.binomial(1, 0.4, n), 'A2_Score': np.random.binomial(1, 0.45, n),
        'A3_Score': np.random.binomial(1, 0.5, n), 'A4_Score': np.random.binomial(1, 0.4, n),
        'A5_Score': np.random.binomial(1, 0.55, n), 'A6_Score': np.random.binomial(1, 0.45, n),
        'A7_Score': np.random.binomial(1, 0.5, n), 'A8_Score': np.random.binomial(1, 0.4, n),
        'A9_Score': np.random.binomial(1, 0.5, n), 'A10_Score': np.random.binomial(1, 0.6, n),
        'age': np.random.normal(8, 5, n).clip(1, 50),
        'gender': np.random.choice(['m', 'f'], n, p=[0.6, 0.4]),
        'ethnicity': np.random.choice(['White-European', 'Asian', 'Latino', 'Middle Eastern', 'Others'], n),
        'jaundice': np.random.binomial(1, 0.1, n),
        'austim': np.random.binomial(1, 0.15, n),
        'Class/ASD': np.random.binomial(1, 0.3, n)
    })

# 2. Prétraitement
print("🛠️ Prétraitement des données...")

# Création d'une nouvelle caractéristique
df['total_score'] = df[['A1_Score', 'A2_Score', 'A3_Score', 'A4_Score', 'A5_Score',
                         'A6_Score', 'A7_Score', 'A8_Score', 'A9_Score', 'A10_Score']].sum(axis=1)

# Sélection des caractéristiques
feature_cols = ['A1_Score', 'A2_Score', 'A3_Score', 'A4_Score', 'A5_Score', 
                'A6_Score', 'A7_Score', 'A8_Score', 'A9_Score', 'A10_Score',
                'age', 'gender', 'ethnicity', 'jaundice', 'austim', 'total_score']

X = df[feature_cols].copy()
y = df['Class/ASD']

# Encodage des variables catégorielles
label_encoders = {}
for col in ['gender', 'ethnicity']:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

# Normalisation
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Division des données
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 4. Entraînement du modèle XGBoost (meilleure performance)
print("🧠 Entraînement du modèle XGBoost...")
model = XGBClassifier(n_estimators=150, max_depth=6, learning_rate=0.1, 
                      random_state=42, use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# 5. Évaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Précision du modèle : {accuracy:.2%}")

# 6. Sauvegarde
print("💾 Sauvegarde des fichiers...")
joblib.dump(model, 'best_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(label_encoders, 'label_encoders.pkl')
joblib.dump(feature_cols, 'features.pkl')

print("🎉 Entraînement terminé ! Fichiers sauvegardés :")
print("   - best_model.pkl")
print("   - scaler.pkl")
print("   - label_encoders.pkl")
print("   - features.pkl")
