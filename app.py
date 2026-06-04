# -*- coding: utf-8 -*-
"""
NeuroSense — Détection Précoce de l'Autisme
Version corrigée (remplacement d'imblearn + gestion des types)
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings, time
warnings.filterwarnings("ignore")
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_curve, auc,
                             roc_auc_score)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# ── Remplacement de imblearn ──────────────────────────────────────
def random_oversample(X, y, random_state=42):
    """Sur-échantillonnage aléatoire avec conversion explicite des types."""
    np.random.seed(random_state)
    # Convertir en tableaux numpy purs
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    
    unique, counts = np.unique(y, return_counts=True)
    if len(unique) != 2:
        return X, y
    maj_class = unique[np.argmax(counts)]
    min_class = unique[np.argmin(counts)]
    X_min = X[y == min_class]
    y_min = y[y == min_class]
    X_maj = X[y == maj_class]
    y_maj = y[y == maj_class]
    
    n_min = len(X_min)
    n_maj = len(X_maj)
    if n_min >= n_maj:
        return X, y
    n_to_add = n_maj - n_min
    indices = np.random.choice(n_min, n_to_add, replace=True)
    X_min_extra = X_min[indices]
    y_min_extra = y_min[indices]
    
    X_res = np.vstack([X_maj, X_min, X_min_extra])
    y_res = np.hstack([y_maj, y_min, y_min_extra])
    
    # Mélanger
    shuffle = np.random.permutation(len(X_res))
    X_res = X_res[shuffle].astype(np.float64)
    y_res = y_res[shuffle].astype(np.int64)
    return X_res, y_res

# ── TensorFlow / Keras (optionnel) ────────────────────────────────
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import (Dense, Dropout, Conv1D,
                                         MaxPooling1D, Flatten,
                                         GlobalAveragePooling1D)
    from tensorflow.keras.callbacks import EarlyStopping
    TF_OK = True
except ImportError:
    TF_OK = False

# ... (le reste du code jusqu'à la fonction pipeline_complet est inchangé, 
#      mais on modifie l'ordre : oversampling APRES le scaler ? Non, on garde avant split)

def pipeline_complet(df_raw: pd.DataFrame):
    target = "Class_ASD"
    df = ingenierie_features(df_raw)
    df, le_dict, scaler = pretraiter(df, is_train=True)  # scaler appliqué ici

    X = df.drop(columns=[target]).values.astype(np.float64)
    y = df[target].values.astype(np.int64)

    # Oversampling
    X, y = random_oversample(X, y, random_state=42)

    # Split
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Vérification des NaN/inf
    if np.any(np.isnan(X_tr)) or np.any(np.isinf(X_tr)):
        st.error("Les données contiennent des NaN ou des inf après oversampling.")
        st.stop()
    if np.any(np.isnan(y_tr)) or np.any(np.isinf(y_tr)):
        st.error("Les cibles contiennent des NaN ou des inf.")
        st.stop()

    resultats = {}
    progress = st.progress(0, text="🤖 Entraînement des modèles…")
    n_models = len(modeles_classiques()) + (2 if TF_OK else 0)
    step = 0

    for nom, clf in modeles_classiques().items():
        try:
            clf.fit(X_tr, y_tr)
        except Exception as e:
            st.error(f"Erreur avec {nom} : {str(e)}")
            raise
        yp = clf.predict(X_te)
        yprob = clf.predict_proba(X_te)[:, 1] if hasattr(clf, "predict_proba") else None
        acc = accuracy_score(y_te, yp)
        roc = roc_auc_score(y_te, yprob) if yprob is not None else acc
        resultats[nom] = dict(modele=clf, accuracy=acc, auc=roc,
                              y_pred=yp, y_proba=yprob, history=None)
        step += 1
        progress.progress(step / n_models, text=f"✅ {nom}")

    if TF_OK:
        # ANN et CNN (inchangé)
        ann, yp, yprob, acc, roc, hist = entrainer_deep(X_tr, X_te, y_tr, y_te, "ANN")
        resultats["🧠 ANN"] = dict(modele=ann, accuracy=acc, auc=roc,
                                   y_pred=yp, y_proba=yprob, history=hist)
        step += 1
        progress.progress(step / n_models, text="✅ ANN")

        cnn, yp, yprob, acc, roc, hist = entrainer_deep(X_tr, X_te, y_tr, y_te, "CNN")
        resultats["📡 CNN"] = dict(modele=cnn, accuracy=acc, auc=roc,
                                   y_pred=yp, y_proba=yprob, history=hist)
        step += 1
        progress.progress(1.0, text="✅ CNN")

    progress.empty()

    best_name = max(resultats, key=lambda k: resultats[k]["auc"])
    best = resultats[best_name]
    col_names = list(df.drop(columns=[target]).columns)
    return (best["modele"], best_name, resultats, le_dict, scaler,
            X_tr, X_te, y_tr, y_te, best["y_pred"], best["accuracy"], col_names)

# Le reste du code (pages, CSS, etc.) est identique à la version précédente,
# sauf la suppression de l'import imblearn.
# Assurez-vous que toutes les fonctions (ingenierie_features, pretraiter,
# modeles_classiques, build_ann, build_cnn, entrainer_deep, plot_learning_curve)
# sont bien présentes telles qu'elles étaient.
