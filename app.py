
# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════════════════════╗
║                         🧠 NeuroSense — Détection Précoce de l'Autisme               ║
║                                                                                      ║
║  ⚡ 10 MODÈLES (8 classiques + ANN + CNN)                                            ║
║  📈 TOUTES LES COURBES : ROC, Learning Curves, Historique Deep Learning             ║
║  📊 TOUTES LES MÉTRIQUES : Accuracy, Precision, Recall, F1, AUC, Matrice confusion  ║
║  🗳️ SYSTÈME DE VOTE : Décision collective de tous les modèles                       ║
║                                                                                      ║
║  Projets fusionnés :                                                                ║
║  • claredang          → ANN, CNN, Learning Curves, 3 datasets                       ║
║  • nagatejakachapuram → Feature Engineering, ROC-AUC, Oversampling, SVC, XGBoost    ║
║  • yashmahes          → Random Forest, Gradient Boosting, Tkinter                   ║
║  • Shehab-Hegab       → Logistic Regression, KNN                                    ║
║  • prasanna24062004   → Random Forest, SVC, Streamlit                               ║
║  • MASANAMUTHU22      → XGBoost                                                     ║
║  • Ankita-M-24        → Decision Tree                                               ║
║  • Anvesh-3           → KNN                                                         ║
╚══════════════════════════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import time
import os
from datetime import datetime

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ──────────────────────────────────────────────────────────────────────────────────────
# IMPORTS SCIKIT-LEARN
# ──────────────────────────────────────────────────────────────────────────────────────
from sklearn.model_selection import train_test_split, learning_curve, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_curve, roc_auc_score, precision_score, recall_score, f1_score,
    ConfusionMatrixDisplay, precision_recall_curve, average_precision_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import RandomOverSampler, SMOTE

# ──────────────────────────────────────────────────────────────────────────────────────
# IMPORTS TENSORFLOW / KERAS (ANN + CNN)
# ──────────────────────────────────────────────────────────────────────────────────────
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import (
        Dense, Dropout, Conv1D, MaxPooling1D, Flatten,
        GlobalAveragePooling1D, BatchNormalization, Input
    )
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.optimizers import Adam
    TF_OK = True
except ImportError:
    TF_OK = False
    st.warning("⚠️ TensorFlow non installé. ANN et CNN seront désactivés.")

# ═══════════════════════════════════════════════════════════════════════════════════════
# CONFIGURATION STREAMLIT
# ═══════════════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="NeuroSense — Détection Autisme",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ═══════════════════════════════════════════════════════════════════════════════════════
# CSS PERSONNALISÉ
# ═══════════════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
* { font-family: 'Inter', sans-serif; }

.stApp { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }

.main-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 2rem;
    border-radius: 20px;
    text-align: center;
    margin-bottom: 2rem;
    box-shadow: 0 10px 40px rgba(0,0,0,0.2);
}
.main-header h1 { color: white; font-size: 2.8rem; margin-bottom: 0.5rem; }
.main-header p { color: rgba(255,255,255,0.85); font-size: 1.1rem; }

.card {
    background: white;
    border-radius: 20px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 4px 20px rgba(0,0,0,0.1);
}

.stButton > button {
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white;
    border: none;
    border-radius: 50px;
    padding: 0.75rem 2rem;
    font-size: 1rem;
    font-weight: 600;
    transition: all 0.3s;
}
.stButton > button:hover {
    transform: scale(1.05);
    box-shadow: 0 5px 20px rgba(102,126,234,0.4);
}

.result-card {
    border-radius: 20px;
    padding: 2rem;
    text-align: center;
    margin-top: 1.5rem;
}

.badge {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 50px;
    font-size: 0.7rem;
    font-weight: 600;
    margin-right: 0.5rem;
}
.badge-purple { background: linear-gradient(135deg, #667eea, #764ba2); color: white; }
.badge-green { background: #4CAF50; color: white; }
.badge-orange { background: #ff9800; color: white; }
.badge-pink { background: #e91e63; color: white; }

.winner-box {
    background: linear-gradient(135deg, #f6d365, #fda085);
    border-radius: 15px;
    padding: 1rem;
    text-align: center;
    font-weight: bold;
    font-size: 1.1rem;
    margin: 1rem 0;
}

.fade-in {
    animation: fadeIn 0.6s ease-out;
}
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}

.metric-card {
    background: linear-gradient(135deg, #667eea15, #764ba215);
    border-radius: 15px;
    padding: 1rem;
    text-align: center;
    border: 1px solid rgba(255,255,255,0.2);
}

.plot-container {
    background: white;
    border-radius: 15px;
    padding: 1rem;
    margin: 1rem 0;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════════════════
_DEFAULTS = {
    "page": 1,
    "role": None,
    "infos_enfant": {},
    "reponses": {},
    "model_entraine": False,
    "best_model": None,
    "best_name": "",
    "all_results": {},
    "best_accuracy": 0,
    "X_train": None,
    "X_test": None,
    "y_train": None,
    "y_test": None,
    "y_pred": None,
    "scaler": None,
    "df_train": None,
    "col_names": None,
    "le_dict": None,
    "model_performances": {},
    "training_history": {}
}

for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ═══════════════════════════════════════════════════════════════════════════════════════
# FONCTIONS DE CHARGEMENT ET PRÉTRAITEMENT
# ═══════════════════════════════════════════════════════════════════════════════════════

@st.cache_data
def charger_donnees():
    """Charge ou génère des données réalistes."""
    try:
        df = pd.read_csv("autism_screening.csv")
    except FileNotFoundError:
        try:
            df = pd.read_csv("train.csv")
        except FileNotFoundError:
            # Génération de données synthétiques avancées
            np.random.seed(42)
            n = 2000
            
            labels = np.random.choice([0, 1], n, p=[0.68, 0.32])
            scores = np.zeros((n, 10), dtype=int)
            
            for i in range(n):
                if labels[i] == 1:
                    p = [0.15, 0.85]
                else:
                    p = [0.85, 0.15]
                scores[i] = np.random.choice([0, 1], 10, p=p)
            
            data = {f"A{i+1}_Score": scores[:, i] for i in range(10)}
            data.update({
                "age": np.random.randint(2, 60, n),
                "gender": np.random.choice(["m", "f"], n, p=[0.48, 0.52]),
                "ethnicity": np.random.choice(
                    ["White-European", "Asian", "Black", "Middle-Eastern", "Hispanic", "Others"], n
                ),
                "jaundice": np.random.choice([0, 1], n, p=[0.85, 0.15]),
                "family_member_with_ASD": np.random.choice([0, 1], n, p=[0.80, 0.20]),
                "country_of_res": np.random.choice(
                    ["France", "USA", "UK", "Canada", "India", "Australia", "Others"], n
                ),
                "Class_ASD": labels,
            })
            df = pd.DataFrame(data)
    
    return df


def ingenierie_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature Engineering avancé
    Sources: nagatejakachapuram + claredang
    """
    df = df.copy()
    
    # Nettoyage
    df.replace({"?": "Others", "": "Others", "unknown": "Others"}, inplace=True)
    
    for c in df.select_dtypes("object").columns:
        df[c].fillna("Others", inplace=True)
    for c in df.select_dtypes("number").columns:
        df[c].fillna(df[c].median(), inplace=True)
    
    # Suppression colonnes inutiles
    drop_cols = ["ID", "id", "age_desc", "used_app_before", "relation"]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True, errors="ignore")
    
    # Questions scores
    q_cols = [f"A{i}_Score" for i in range(1, 11) if f"A{i}_Score" in df.columns]
    for c in q_cols:
        if df[c].dtype == object:
            df[c] = df[c].map({"Yes": 1, "yes": 1, "1": 1, "No": 0, "no": 0, "0": 0}).fillna(0).astype(int)
    
    # sum_score
    if q_cols:
        df["sum_score"] = df[q_cols].sum(axis=1)
        df["avg_score"] = df[q_cols].mean(axis=1)
        df["high_risk_questions"] = (df[q_cols].sum(axis=1) >= 7).astype(int)
    
    # age_group
    if "age" in df.columns:
        df["age"] = pd.to_numeric(df["age"], errors="coerce").fillna(df["age"].median())
        df["age_group"] = pd.cut(
            df["age"],
            bins=[0, 4, 12, 18, 35, 60, 200],
            labels=[0, 1, 2, 3, 4, 5]
        ).astype(int)
        
        # Interaction features
        if q_cols:
            df["age_score_interaction"] = df["age"] * df["avg_score"]
    
    return df


def pretraiter(df: pd.DataFrame, is_train=True, le_dict=None, scaler=None):
    """Prétraitement avec encodage et normalisation."""
    df = df.copy()
    target = "Class_ASD"
    
    cat_cols = [c for c in df.select_dtypes("object").columns if c != target]
    
    if is_train:
        le_dict = {}
        for c in cat_cols:
            le = LabelEncoder()
            df[c] = le.fit_transform(df[c].astype(str))
            le_dict[c] = le
        
        if target in df.columns and df[target].dtype == object:
            le_t = LabelEncoder()
            df[target] = le_t.fit_transform(df[target].astype(str))
            le_dict["__target__"] = le_t
    else:
        for c in cat_cols:
            if c in le_dict:
                le = le_dict[c]
                df[c] = df[c].astype(str).apply(
                    lambda x: x if x in le.classes_ else le.classes_[0]
                )
                df[c] = le.transform(df[c])
    
    num_cols = [c for c in df.select_dtypes("number").columns if c != target]
    
    if is_train:
        scaler = StandardScaler()
        if num_cols:
            df[num_cols] = scaler.fit_transform(df[num_cols])
    else:
        if scaler and num_cols:
            df[num_cols] = scaler.transform(df[num_cols])
    
    return df, le_dict, scaler


# ═══════════════════════════════════════════════════════════════════════════════════════
# MODÈLES CLASSIQUES (8 modèles)
# ═══════════════════════════════════════════════════════════════════════════════════════

def modeles_classiques():
    """Dictionnaire des 8 modèles classiques."""
    return {
        "🌲 Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=15, min_samples_split=5,
            min_samples_leaf=2, random_state=42, n_jobs=-1
        ),
        "⚡ XGBoost": XGBClassifier(
            n_estimators=150, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            use_label_encoder=False, eval_metric="logloss", random_state=42
        ),
        "🚀 Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150, learning_rate=0.1, max_depth=5,
            subsample=0.8, random_state=42
        ),
        "📐 Logistic Regression": LogisticRegression(
            C=1.0, max_iter=1000, solver="lbfgs", random_state=42
        ),
        "🔷 SVC": SVC(
            C=1.0, kernel="rbf", gamma="scale", probability=True, random_state=42
        ),
        "🌿 Decision Tree": DecisionTreeClassifier(
            max_depth=10, min_samples_split=5, min_samples_leaf=2, random_state=42
        ),
        "👥 KNN": KNeighborsClassifier(
            n_neighbors=7, weights="distance", metric="minkowski", p=2
        ),
        "🔔 Naive Bayes": GaussianNB()
    }


# ═══════════════════════════════════════════════════════════════════════════════════════
# MODÈLES DEEP LEARNING (ANN + CNN)
# ═══════════════════════════════════════════════════════════════════════════════════════

def build_ann(input_dim: int) -> Sequential:
    """Construction du réseau de neurones artificiels (ANN)."""
    model = Sequential([
        Dense(256, activation="relu", input_shape=(input_dim,)),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(128, activation="relu"),
        BatchNormalization(),
        Dropout(0.2),
        
        Dense(64, activation="relu"),
        BatchNormalization(),
        Dropout(0.2),
        
        Dense(32, activation="relu"),
        Dropout(0.1),
        
        Dense(1, activation="sigmoid")
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
    )
    return model


def build_cnn(input_dim: int) -> Sequential:
    """Construction du réseau de neurones convolutif (CNN 1D)."""
    model = Sequential([
        Input(shape=(input_dim, 1)),
        
        Conv1D(64, kernel_size=3, activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        
        Conv1D(128, kernel_size=3, activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        
        Conv1D(64, kernel_size=3, activation="relu", padding="same"),
        GlobalAveragePooling1D(),
        
        Dense(64, activation="relu"),
        Dropout(0.3),
        
        Dense(32, activation="relu"),
        Dropout(0.2),
        
        Dense(1, activation="sigmoid")
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
    )
    return model


def entrainer_deep(X_tr, X_te, y_tr, y_te, kind="ANN"):
    """Entraîne un modèle Deep Learning et retourne les résultats."""
    es = EarlyStopping(
        monitor="val_loss", patience=15, restore_best_weights=True, verbose=0
    )
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=7, min_lr=0.00001, verbose=0
    )
    
    if kind == "ANN":
        model = build_ann(X_tr.shape[1])
        Xtr, Xte = X_tr, X_te
    else:
        model = build_cnn(X_tr.shape[1])
        Xtr = X_tr.reshape(-1, X_tr.shape[1], 1)
        Xte = X_te.reshape(-1, X_te.shape[1], 1)
    
    history = model.fit(
        Xtr, y_tr,
        epochs=100,
        batch_size=32,
        validation_split=0.15,
        callbacks=[es, reduce_lr],
        verbose=0
    )
    
    y_proba = model.predict(Xte, verbose=0).ravel()
    y_pred = (y_proba > 0.5).astype(int)
    acc = accuracy_score(y_te, y_pred)
    roc = roc_auc_score(y_te, y_proba)
    
    return model, y_pred, y_proba, acc, roc, history


# ═══════════════════════════════════════════════════════════════════════════════════════
# FONCTIONS D'ÉVALUATION ET VISUALISATION
# ═══════════════════════════════════════════════════════════════════════════════════════

def plot_roc_curves(results, y_test, best_name):
    """Dessine toutes les courbes ROC."""
    fig, ax = plt.subplots(figsize=(10, 7))
    
    colors = plt.cm.tab20(np.linspace(0, 1, len(results)))
    
    for (nom, res), col in zip(sorted(results.items(), key=lambda x: x[1]["auc"], reverse=True), colors):
        if res["y_proba"] is not None:
            fpr, tpr, _ = roc_curve(y_test, res["y_proba"])
            lw = 3 if nom == best_name else 1.5
            ls = "-" if nom == best_name else "--"
            ax.plot(fpr, tpr, lw=lw, ls=ls, color=col,
                    label=f"{nom} (AUC = {res['auc']:.3f})")
    
    ax.plot([0, 1], [0, 1], "k:", lw=2, label="Modèle aléatoire (AUC = 0.5)")
    ax.set_xlabel("Taux de faux positifs (1 - Spécificité)", fontsize=12)
    ax.set_ylabel("Taux de vrais positifs (Sensibilité)", fontsize=12)
    ax.set_title("📈 Courbes ROC de tous les modèles", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    
    return fig


def plot_pr_curves(results, y_test, best_name):
    """Dessine toutes les courbes Precision-Recall."""
    fig, ax = plt.subplots(figsize=(10, 7))
    
    colors = plt.cm.tab20(np.linspace(0, 1, len(results)))
    
    for (nom, res), col in zip(sorted(results.items(), key=lambda x: x[1]["auc"], reverse=True), colors):
        if res["y_proba"] is not None:
            precision, recall, _ = precision_recall_curve(y_test, res["y_proba"])
            ap = average_precision_score(y_test, res["y_proba"])
            lw = 3 if nom == best_name else 1.5
            ls = "-" if nom == best_name else "--"
            ax.plot(recall, precision, lw=lw, ls=ls, color=col,
                    label=f"{nom} (AP = {ap:.3f})")
    
    ax.set_xlabel("Rappel (Recall)", fontsize=12)
    ax.set_ylabel("Précision (Precision)", fontsize=12)
    ax.set_title("📊 Courbes Précision-Rappel de tous les modèles", fontsize=14, fontweight="bold")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    
    return fig


def plot_learning_curve(estimator, X, y, model_name):
    """Dessine la courbe d'apprentissage."""
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        train_sizes=np.linspace(0.1, 1.0, 10),
        n_jobs=-1,
        scoring="accuracy"
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    fig, ax = plt.subplots(figsize=(9, 6))
    
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                    alpha=0.2, color="#667eea")
    ax.fill_between(train_sizes, test_mean - test_std, test_mean + test_std,
                    alpha=0.2, color="#f5576c")
    
    ax.plot(train_sizes, train_mean, "o-", color="#667eea", lw=2, label="Score entraînement")
    ax.plot(train_sizes, test_mean, "s-", color="#f5576c", lw=2, label="Score validation")
    
    ax.set_xlabel("Taille de l'échantillon d'entraînement", fontsize=12)
    ax.set_ylabel("Score de précision", fontsize=12)
    ax.set_title(f"📉 Courbe d'apprentissage — {model_name}", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    
    return fig


def plot_confusion_matrix(y_test, y_pred, model_name):
    """Dessine la matrice de confusion."""
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Non-ASD", "ASD"],
                yticklabels=["Non-ASD", "ASD"],
                annot_kws={"size": 14})
    
    ax.set_xlabel("Prédiction", fontsize=12)
    ax.set_ylabel("Vérité terrain", fontsize=12)
    ax.set_title(f"📊 Matrice de confusion — {model_name}", fontsize=14, fontweight="bold")
    
    return fig


def plot_feature_importance(model, feature_names, model_name, top_n=15):
    """Dessine l'importance des caractéristiques."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        indices = np.argsort(importances)[-top_n:]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(range(len(indices)), importances[indices], color="#667eea")
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels([feature_names[i] for i in indices])
        ax.set_xlabel("Importance", fontsize=12)
        ax.set_title(f"🎯 Importance des caractéristiques — {model_name}", fontsize=14, fontweight="bold")
        ax.invert_yaxis()
        return fig
    elif hasattr(model, "coef_"):
        coef = np.abs(model.coef_[0])
        indices = np.argsort(coef)[-top_n:]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(range(len(indices)), coef[indices], color="#764ba2")
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels([feature_names[i] for i in indices])
        ax.set_xlabel("|Coefficient|", fontsize=12)
        ax.set_title(f"🎯 Importance des coefficients — {model_name}", fontsize=14, fontweight="bold")
        ax.invert_yaxis()
        return fig
    return None


def plot_model_comparison(results):
    """Dessine un graphique comparatif de tous les modèles."""
    names = list(results.keys())
    aucs = [results[n]["auc"] for n in names]
    accs = [results[n]["accuracy"] for n in names]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(names))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, accs, width, label="Accuracy", color="#667eea", alpha=0.8)
    bars2 = ax.bar(x + width/2, aucs, width, label="ROC-AUC", color="#f5576c", alpha=0.8)
    
    ax.set_xlabel("Modèles", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("🏆 Comparaison des performances de tous les modèles", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([n.split(" ")[0] for n in names], rotation=45, ha="right")
    ax.legend(loc="lower right", fontsize=11)
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3, axis="y")
    
    # Ajouter les valeurs sur les barres
    for bar, val in zip(bars1, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    
    for bar, val in zip(bars2, aucs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    
    fig.tight_layout()
    return fig


def plot_training_history(history, model_name):
    """Dessine l'historique d'entraînement d'un modèle Deep Learning."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Accuracy
    axes[0].plot(history.history["accuracy"], "o-", color="#667eea", label="Train", lw=2)
    axes[0].plot(history.history["val_accuracy"], "s-", color="#f5576c", label="Validation", lw=2)
    axes[0].set_xlabel("Époque", fontsize=11)
    axes[0].set_ylabel("Accuracy", fontsize=11)
    axes[0].set_title(f"{model_name} — Accuracy", fontsize=12, fontweight="bold")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Loss
    axes[1].plot(history.history["loss"], "o-", color="#667eea", label="Train", lw=2)
    axes[1].plot(history.history["val_loss"], "s-", color="#f5576c", label="Validation", lw=2)
    axes[1].set_xlabel("Époque", fontsize=11)
    axes[1].set_ylabel("Loss", fontsize=11)
    axes[1].set_title(f"{model_name} — Loss", fontsize=12, fontweight="bold")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════════════
# PIPELINE D'ENTRAÎNEMENT COMPLET
# ═══════════════════════════════════════════════════════════════════════════════════════

def pipeline_complet(df_raw: pd.DataFrame):
    """Entraîne TOUS les modèles et retourne les résultats."""
    target = "Class_ASD"
    
    with st.spinner("🔧 Ingénierie des caractéristiques..."):
        df = ingenierie_features(df_raw)
    
    with st.spinner("📊 Prétraitement des données..."):
        df, le_dict, scaler = pretraiter(df, is_train=True)
    
    X = df.drop(columns=[target]).values
    y = df[target].values
    
    with st.spinner("⚖️ Équilibrage des classes (SMOTE)..."):
        smote = SMOTE(random_state=42)
        X, y = smote.fit_resample(X, y)
    
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    results = {}
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    all_models = modeles_classiques()
    n_models = len(all_models) + (2 if TF_OK else 0)
    step = 0
    
    # Modèles classiques
    for nom, clf in all_models.items():
        status_text.text(f"🔄 Entraînement: {nom}")
        clf.fit(X_tr, y_tr)
        
        y_pred = clf.predict(X_te)
        y_proba = clf.predict_proba(X_te)[:, 1] if hasattr(clf, "predict_proba") else None
        
        acc = accuracy_score(y_te, y_pred)
        prec = precision_score(y_te, y_pred, zero_division=0)
        rec = recall_score(y_te, y_pred, zero_division=0)
        f1 = f1_score(y_te, y_pred, zero_division=0)
        roc = roc_auc_score(y_te, y_proba) if y_proba is not None else acc
        
        results[nom] = {
            "modele": clf,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "auc": roc,
            "y_pred": y_pred,
            "y_proba": y_proba,
            "history": None
        }
        
        step += 1
        progress_bar.progress(step / n_models)
    
    # Deep Learning
    if TF_OK:
        status_text.text("🔄 Entraînement: 🧠 ANN")
        ann, yp, yprob, acc, roc, hist = entrainer_deep(X_tr, X_te, y_tr, y_te, "ANN")
        results["🧠 ANN"] = {
            "modele": ann, "accuracy": acc, "auc": roc,
            "y_pred": yp, "y_proba": yprob, "history": hist,
            "precision": precision_score(y_te, yp, zero_division=0),
            "recall": recall_score(y_te, yp, zero_division=0),
            "f1": f1_score(y_te, yp, zero_division=0)
        }
        step += 1
        progress_bar.progress(step / n_models)
        
        status_text.text("🔄 Entraînement: 📡 CNN")
        cnn, yp, yprob, acc, roc, hist = entrainer_deep(X_tr, X_te, y_tr, y_te, "CNN")
        results["📡 CNN"] = {
            "modele": cnn, "accuracy": acc, "auc": roc,
            "y_pred": yp, "y_proba": yprob, "history": hist,
            "precision": precision_score(y_te, yp, zero_division=0),
            "recall": recall_score(y_te, yp, zero_division=0),
            "f1": f1_score(y_te, yp, zero_division=0)
        }
        step += 1
        progress_bar.progress(1.0)
    
    status_text.text("✅ Entraînement terminé !")
    time.sleep(1)
    status_text.empty()
    progress_bar.empty()
    
    # Sélection du meilleur modèle (basé sur AUC)
    best_name = max(results, key=lambda k: results[k]["auc"])
    best = results[best_name]
    
    col_names = list(df.drop(columns=[target]).columns)
    
    return (
        best["modele"], best_name, results, le_dict, scaler,
        X_tr, X_te, y_tr, y_te, best["y_pred"], best["accuracy"], col_names
    )


# ═══════════════════════════════════════════════════════════════════════════════════════
# FONCTION DE PRÉDICTION POUR UN PATIENT
# ═══════════════════════════════════════════════════════════════════════════════════════

def preparer_patient(reponses, infos, scaler, n_features):
    """Prépare les données d'un patient pour la prédiction."""
    # Questions A1-A10
    q_scores = [reponses.get(f"A{i}", 0) for i in range(1, 11)]
    
    # Informations démographiques
    ethnie_map = {
        "Blanc": 0, "Asiatique": 1, "Noir": 2,
        "Arabe": 3, "Autre": 4, "White-European": 0,
        "Asian": 1, "Black": 2, "Middle-Eastern": 3,
        "Hispanic": 4, "Others": 5
    }
    
    age = infos.get("age", 5)
    gender = 0 if infos.get("genre") in ["garcon", "m"] else 1
    ethnicity = ethnie_map.get(infos.get("ethnie"), 5)
    jaundice = infos.get("jaundice", 0)
    family_asd = infos.get("family_asd", 0)
    
    # Features calculées
    sum_score = sum(q_scores)
    avg_score = sum_score / 10
    high_risk = 1 if sum_score >= 7 else 0
    
    # Age group
    if age <= 4:
        age_group = 0
    elif age <= 12:
        age_group = 1
    elif age <= 18:
        age_group = 2
    elif age <= 35:
        age_group = 3
    elif age <= 60:
        age_group = 4
    else:
        age_group = 5
    
    age_interaction = age * avg_score
    
    input_vector = q_scores + [
        age, gender, ethnicity, jaundice, family_asd,
        sum_score, avg_score, high_risk, age_group, age_interaction
    ]
    
    # Aligner sur le nombre de features attendues
    vec = np.zeros(n_features)
    for i, v in enumerate(input_vector[:n_features]):
        vec[i] = v
    
    return scaler.transform(vec.reshape(1, -1))


# ═══════════════════════════════════════════════════════════════════════════════════════
# CHARGEMENT ET ENTRAÎNEMENT INITIAL
# ═══════════════════════════════════════════════════════════════════════════════════════

if st.session_state.df_train is None:
    with st.spinner("📂 Chargement des données..."):
        st.session_state.df_train = charger_donnees()
    st.rerun()

if not st.session_state.model_entraine and st.session_state.df_train is not None:
    (model, best_name, results, le_dict, scaler,
     X_tr, X_te, y_tr, y_te, y_pred, accuracy, col_names) = \
        pipeline_complet(st.session_state.df_train)
    
    st.session_state.update({
        "best_model": model,
        "best_name": best_name,
        "all_results": results,
        "le_dict": le_dict,
        "scaler": scaler,
        "X_train": X_tr,
        "X_test": X_te,
        "y_train": y_tr,
        "y_test": y_te,
        "y_pred": y_pred,
        "best_accuracy": accuracy,
        "col_names": col_names,
        "model_entraine": True,
    })
    st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding:1rem;">
        <h2 style="color:white;">🧠 NeuroSense</h2>
        <p style="color:rgba(255,255,255,0.8);">IA pour la détection précoce</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    
    if st.session_state.model_entraine:
        results = st.session_state.all_results
        best = st.session_state.best_name
        
        st.metric("🏆 Meilleur modèle", best.split(" ")[-1])
        st.metric("🎯 Accuracy", f"{st.session_state.best_accuracy:.1%}")
        st.metric("📊 AUC", f"{results[best]['auc']:.3f}")
        
        st.markdown("---")
        st.markdown("### 🥇 Top 5 modèles")
        sorted_models = sorted(results.items(), key=lambda x: x[1]["auc"], reverse=True)[:5]
        
        for i, (name, metrics) in enumerate(sorted_models, 1):
            medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
            st.markdown(f"{medal} **{name.split(' ')[-1]}**")
            st.caption(f"AUC: {metrics['auc']:.3f} | Acc: {metrics['accuracy']:.1%}")
    
    st.markdown("---")
    st.markdown("### 📌 Progression")
    pages = ["🎭 Rôle", "👶 Infos", "📋 Questionnaire", "📊 Résultat"]
    for i, p in enumerate(pages, 1):
        if i < st.session_state.page:
            st.markdown(f"✅ {i}. {p}")
        elif i == st.session_state.page:
            st.markdown(f"🔵 **{i}. {p}**")
        else:
            st.markdown(f"⚪ {i}. {p}")
    
    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.75rem; text-align:center; color:rgba(255,255,255,0.6);">
        <p>📚 Sources : claredang • nagatejakachapuram<br>
        yashmahes • Shehab-Hegab • prasanna24062004<br>
        MASANAMUTHU22 • Ankita-M-24 • Anvesh-3</p>
        <p>© 2024 — Outil d'aide à la décision</p>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════════════
# EN-TÊTE
# ═══════════════════════════════════════════════════════════════════════════════════════

n_models = len(st.session_state.all_results) if st.session_state.model_entraine else "..."

st.markdown(f"""
<div class="main-header fade-in">
    <h1>🧠 NeuroSense</h1>
    <p>Détection précoce des Troubles du Spectre Autistique par Intelligence Artificielle</p>
    <div style="margin-top: 1rem;">
        <span class="badge badge-purple">🤖 {n_models} modèles</span>
        <span class="badge badge-green">🧠 ANN + CNN</span>
        <span class="badge badge-orange">🏆 Sélection automatique</span>
        <span class="badge badge-pink">📈 Learning Curves</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════════════
# PAGE 1 — CHOIX DU RÔLE
# ═══════════════════════════════════════════════════════════════════════════════════════

if st.session_state.page == 1:
    st.markdown('<div class="card fade-in">', unsafe_allow_html=True)
    st.subheader("👋 Qui êtes-vous ?")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style="text-align:center; background:linear-gradient(135deg,#667eea20,#764ba220);
                    border-radius:20px; padding:2rem;">
            <span style="font-size:4rem;">👨‍👩‍👧</span>
            <h3>Parent</h3>
            <p>Complétez le questionnaire pour votre enfant</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("📝 Je suis un parent", key="btn_parent", use_container_width=True):
            st.session_state.role = "parent"
            st.session_state.page = 2
            st.rerun()
    
    with col2:
        st.markdown("""
        <div style="text-align:center; background:linear-gradient(135deg,#667eea20,#764ba220);
                    border-radius:20px; padding:2rem;">
            <span style="font-size:4rem;">👨‍⚕️</span>
            <h3>Médecin</h3>
            <p>Évaluez votre patient avec notre outil</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("🩺 Je suis médecin", key="btn_medecin", use_container_width=True):
            st.session_state.role = "medecin"
            st.session_state.page = 2
            st.rerun()
    
    st.markdown("</div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════════════
# PAGE 2 — INFORMATIONS DE L'ENFANT
# ═══════════════════════════════════════════════════════════════════════════════════════

elif st.session_state.page == 2:
    st.markdown('<div class="card fade-in">', unsafe_allow_html=True)
    st.subheader("👶 Informations de l'enfant")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        nom = st.text_input("📝 Prénom de l'enfant", placeholder="ex: Adam, Sara, Lucas...")
        age = st.number_input("🎂 Âge (années)", min_value=2, max_value=60, value=5)
    
    with col2:
        genre = st.radio("⚥ Genre", ["m", "f"],
                        format_func=lambda x: "👦 Garçon" if x == "m" else "👧 Fille",
                        horizontal=True)
        ethnie = st.selectbox("🌍 Origine ethnique",
                             ["White-European", "Asian", "Black", "Middle-Eastern", "Hispanic", "Others"])
    
    col3, col4 = st.columns(2)
    
    with col3:
        jaundice = st.radio("🟡 Ictère à la naissance ?", [0, 1],
                           format_func=lambda x: "❌ Non" if x == 0 else "✅ Oui",
                           horizontal=True)
    
    with col4:
        family_asd = st.radio("👨‍👩‍👧 Antécédents familiaux d'autisme ?", [0, 1],
                             format_func=lambda x: "❌ Non" if x == 0 else "✅ Oui",
                             horizontal=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    col_back, col_next, _ = st.columns([1, 2, 1])
    
    with col_back:
        if st.button("⬅️ Retour", use_container_width=True):
            st.session_state.page = 1
            st.rerun()
    
    with col_next:
        if st.button("📝 Commencer le questionnaire", type="primary", use_container_width=True):
            if nom and nom.strip():
                st.session_state.infos_enfant = {
                    "nom": nom, "age": age, "genre": genre,
                    "ethnie": ethnie, "jaundice": jaundice,
                    "family_asd": family_asd
                }
                st.session_state.page = 3
                st.rerun()
            else:
                st.error("⚠️ Veuillez entrer le prénom de l'enfant")


# ═══════════════════════════════════════════════════════════════════════════════════════
# PAGE 3 — QUESTIONNAIRE
# ═══════════════════════════════════════════════════════════════════════════════════════

elif st.session_state.page == 3:
    infos = st.session_state.infos_enfant
    
    st.markdown(f"""
    <div class="card fade-in">
        <h3>📋 Questionnaire d'évaluation</h3>
        <p><strong>{infos.get('nom', '')}</strong> • Âge : {infos.get('age', '')} ans</p>
        <p><small>Basé sur les 10 questions standard du screening ASD (AQ-10)</small></p>
    </div>
    """, unsafe_allow_html=True)
    
    QUESTIONS = [
        ("A1", "😊", "Difficultés à comprendre les expressions faciales ?",
         "Ne comprend pas quand quelqu'un est triste, content ou fâché"),
        ("A2", "💬", "Difficultés à maintenir une conversation ?",
         "Ne sait pas quand parler, quand s'arrêter, change de sujet brusquement"),
        ("A3", "🔄", "Comportements répétitifs ?",
         "Se balance, tourne, tape des mains, répète les mêmes mots"),
        ("A4", "🎯", "Intérêts très spécifiques et intenses ?",
         "Toujours le même sujet, collectionne des objets inhabituels"),
        ("A5", "😐", "Semble distant ou sans émotion ?",
         "Ne réagit pas quand on l'appelle, semble dans sa bulle"),
        ("A6", "🔊", "Sensibilité aux bruits ou textures ?",
         "N'aime pas l'aspirateur, les étiquettes, certaines lumières"),
        ("A7", "🎮", "Préfère jouer seul ?",
         "Ne cherche pas à faire des amis, joue en solitaire"),
        ("A8", "📖", "Langage très littéral ?",
         "Ne comprend pas les blagues, l'ironie ou les métaphores"),
        ("A9", "👀", "Évite le contact visuel ?",
         "Ne regarde pas dans les yeux, détourne le regard"),
        ("A10", "📅", "Très attaché à ses routines ?",
         "Se fâche quand on change ses habitudes ou son environnement"),
    ]
    
    for idx, (qid, icon, question, detail) in enumerate(QUESTIONS, 1):
        col_icon, col_q = st.columns([1, 5])
        
        with col_icon:
            st.markdown(f"""
            <div style="background:linear-gradient(135deg,#667eea,#764ba2);
                        width:45px;height:45px;border-radius:22.5px;
                        display:flex;align-items:center;justify-content:center;">
                <span style="font-size:1.5rem;">{icon}</span>
            </div>
            """, unsafe_allow_html=True)
        
        with col_q:
            st.markdown(f"**Question {idx}/10** — {question}")
            st.caption(f"💡 {detail}")
            rep = st.radio("", [0, 1],
                          format_func=lambda x: "❌ Non" if x == 0 else "✅ Oui",
                          key=f"q_{qid}",
                          horizontal=True,
                          label_visibility="collapsed")
            if rep is not None:
                st.session_state.reponses[qid] = rep
        
        st.markdown("---")
    
    total_reponses = len(st.session_state.reponses)
    
    if total_reponses > 0:
        score = sum(st.session_state.reponses.values())
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#667eea,#764ba2);
                    border-radius:20px; padding:1rem; text-align:center; margin:1rem 0;">
            <span style="color:white; font-size:1.2rem;">
                📊 Progression : {total_reponses}/10
            </span>
            <br>
            <span style="color:white; font-size:2rem; font-weight:bold;">
                Score : {score}/{total_reponses}
            </span>
        </div>
        """, unsafe_allow_html=True)
    
    col_back, col_next, _ = st.columns([1, 2, 1])
    
    with col_back:
        if st.button("⬅️ Retour", use_container_width=True):
            st.session_state.page = 2
            st.rerun()
    
    with col_next:
        if total_reponses == 10:
            if st.button("🔮 Voir le résultat", type="primary", use_container_width=True):
                st.session_state.page = 4
                st.rerun()
        else:
            st.warning(f"⚠️ {10 - total_reponses} question(s) restante(s)")


# ═══════════════════════════════════════════════════════════════════════════════════════
# PAGE 4 — RÉSULTATS COMPLETS AVEC TOUTES LES COURBES
# ═══════════════════════════════════════════════════════════════════════════════════════

elif st.session_state.page == 4:
    if not st.session_state.model_entraine:
        st.error("❌ Modèles non chargés. Veuillez rafraîchir la page.")
        st.stop()
    
    infos = st.session_state.infos_enfant
    reponses = st.session_state.reponses
    results = st.session_state.all_results
    best_name = st.session_state.best_name
    best_model = st.session_state.best_model
    
    total_score = sum(reponses.values())
    
    # Préparer les données du patient
    n_features = st.session_state.scaler.n_features_in_
    patient_scaled = preparer_patient(reponses, infos, st.session_state.scaler, n_features)
    
    # Prédictions de tous les modèles
    all_predictions = {}
    all_probas = {}
    
    with st.spinner("🧠 Analyse par tous les modèles..."):
        for name, res in results.items():
            clf = res["modele"]
            try:
                if name in ["🧠 ANN", "📡 CNN"]:
                    if name == "📡 CNN":
                        X_in = patient_scaled.reshape(-1, n_features, 1)
                    else:
                        X_in = patient_scaled
                    proba = float(clf.predict(X_in, verbose=0).ravel()[0])
                else:
                    proba = float(clf.predict_proba(patient_scaled)[0][1])
                
                pred = int(proba > 0.5)
            except Exception:
                proba = 0.5
                pred = 0
            
            all_predictions[name] = pred
            all_probas[name] = proba
    
    # Résultat du meilleur modèle
    best_pred = all_predictions[best_name]
    best_proba = all_probas[best_name]
    
    # Vote majoritaire
    votes_positive = sum(all_predictions.values())
    consensus = votes_positive / len(all_predictions)
    consensus_pred = int(consensus > 0.5)
    
    # ──────────────────────────────────────────────────────────────────────────────────
    # AFFICHAGE DU RÉSULTAT PRINCIPAL
    # ──────────────────────────────────────────────────────────────────────────────────
    
    st.markdown(f'<div class="winner-box fade-in">🏆 Décision finale — {best_name}</div>', 
                unsafe_allow_html=True)
    
    if best_pred == 1:
        st.markdown("""
        <div class="result-card" style="background:linear-gradient(135deg,#f093fb,#f5576c);">
            <span style="font-size:4rem;">🚨</span>
            <h1 style="color:white;">Risque élevé détecté</h1>
            <p style="color:white; font-size:1.2rem;">
                Une évaluation clinique approfondie est recommandée
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="result-card" style="background:linear-gradient(135deg,#84fab0,#8fd3f4);">
            <span style="font-size:4rem;">✅</span>
            <h1 style="color:#2c3e50;">Risque faible</h1>
            <p style="color:#2c3e50; font-size:1.2rem;">
                Le développement semble dans la norme
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Métriques principales
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📊 Score questionnaire", f"{total_score}/10")
    col2.metric("🤖 Probabilité (meilleur)", f"{best_proba:.1%}")
    col3.metric("🎯 Consensus des modèles", f"{consensus:.0%}")
    col4.metric("📈 AUC meilleur modèle", f"{results[best_name]['auc']:.3f}")
    
    st.progress(best_proba)
    st.markdown("---")
    
    # ──────────────────────────────────────────────────────────────────────────────────
    # VOTE DE TOUS LES MODÈLES
    # ──────────────────────────────────────────────────────────────────────────────────
    
    st.subheader("🗳️ Vote des 10 modèles")
    
    cols = st.columns(4)
    for idx, (name, proba) in enumerate(sorted(all_probas.items(), key=lambda x: x[1], reverse=True)):
        pred = all_predictions[name]
        color = "#ffcccc" if pred else "#ccffcc"
        medal = "🏆 " if name == best_name else ""
        
        with cols[idx % 4]:
            st.markdown(f"""
            <div style="background:{color}; border-radius:12px; padding:0.8rem; 
                        margin:0.4rem; text-align:center;">
                <strong>{medal}{name}</strong><br>
                {'⚠️ Risque' if pred else '✅ Normal'}<br>
                <small>Prob: {proba:.1%}</small>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#667eea,#764ba2);
                border-radius:20px; padding:1.2rem; text-align:center; margin:1rem 0;">
        <h3 style="color:white;">🗳️ {votes_positive}/{len(all_predictions)} modèles détectent un risque</h3>
        <p style="color:rgba(255,255,255,0.9);">
            {'⚠️ Une consultation avec un spécialiste est recommandée' if consensus_pred else '✅ Pas de signes d\'alerte majeurs'}
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ──────────────────────────────────────────────────────────────────────────────────
    # SECTION 1: COMPARAISON DE TOUS LES MODÈLES
    # ──────────────────────────────────────────────────────────────────────────────────
    
    st.subheader("📊 Comparaison des performances de tous les modèles")
    
    # Graphique de comparaison
    with st.container():
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        fig_comp = plot_model_comparison(results)
        st.pyplot(fig_comp)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Tableau des métriques
    with st.expander("📋 Voir le tableau détaillé des métriques"):
        metrics_data = []
        for name, res in results.items():
            metrics_data.append({
                "Modèle": name,
                "Accuracy": f"{res['accuracy']:.3f}",
                "Précision": f"{res['precision']:.3f}",
                "Rappel": f"{res['recall']:.3f}",
                "F1-Score": f"{res['f1']:.3f}",
                "AUC": f"{res['auc']:.3f}"
            })
        
        df_metrics = pd.DataFrame(metrics_data)
        st.dataframe(df_metrics, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # ──────────────────────────────────────────────────────────────────────────────────
    # SECTION 2: COURBES ROC ET PR DE TOUS LES MODÈLES
    # ──────────────────────────────────────────────────────────────────────────────────
    
    st.subheader("📈 Courbes d'évaluation de tous les modèles")
    
    col_roc, col_pr = st.columns(2)
    
    with col_roc:
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        st.markdown("#### Courbes ROC")
        fig_roc = plot_roc_curves(results, st.session_state.y_test, best_name)
        st.pyplot(fig_roc)
        plt.close()
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_pr:
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        st.markdown("#### Courbes Précision-Rappel")
        fig_pr = plot_pr_curves(results, st.session_state.y_test, best_name)
        st.pyplot(fig_pr)
        plt.close()
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ──────────────────────────────────────────────────────────────────────────────────
    # SECTION 3: COURBE D'APPRENTISSAGE DU MEILLEUR MODÈLE
    # ──────────────────────────────────────────────────────────────────────────────────
    
    st.subheader("📉 Courbe d'apprentissage du meilleur modèle")
    
    with st.container():
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        
        if best_name not in ["🧠 ANN", "📡 CNN"]:
            X_combined = np.vstack([st.session_state.X_train, st.session_state.X_test])
            y_combined = np.concatenate([st.session_state.y_train, st.session_state.y_test])
            
            fig_lc = plot_learning_curve(best_model, X_combined, y_combined, best_name)
            st.pyplot(fig_lc)
            plt.close()
        else:
            st.info("ℹ️ Les courbes d'apprentissage pour les modèles Deep Learning sont disponibles dans la section suivante.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ──────────────────────────────────────────────────────────────────────────────────
    # SECTION 4: MATRICE DE CONFUSION ET RAPPORT
    # ──────────────────────────────────────────────────────────────────────────────────
    
    st.subheader("📊 Évaluation détaillée du meilleur modèle")
    
    col_cm, col_cr = st.columns(2)
    
    with col_cm:
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        st.markdown("#### Matrice de confusion")
        fig_cm = plot_confusion_matrix(
            st.session_state.y_test, 
            st.session_state.y_pred,
            best_name
        )
        st.pyplot(fig_cm)
        plt.close()
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_cr:
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        st.markdown("#### Rapport de classification")
        
        report = classification_report(
            st.session_state.y_test,
            st.session_state.y_pred,
            target_names=["Non-ASD", "ASD"],
            output_dict=True
        )
        
        df_report = pd.DataFrame(report).transpose().round(3)
        st.dataframe(df_report, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ──────────────────────────────────────────────────────────────────────────────────
    # SECTION 5: HISTORIQUE DEEP LEARNING (si disponible)
    # ──────────────────────────────────────────────────────────────────────────────────
    
    if TF_OK:
        dl_models = [(name, res) for name, res in results.items() 
                     if name in ["🧠 ANN", "📡 CNN"] and res["history"] is not None]
        
        if dl_models:
            st.subheader("🧠 Historique d'entraînement - Deep Learning")
            
            dl_cols = st.columns(len(dl_models))
            for idx, (name, res) in enumerate(dl_models):
                with dl_cols[idx]:
                    st.markdown(f'<div class="plot-container">', unsafe_allow_html=True)
                    st.markdown(f"#### {name}")
                    fig_hist = plot_training_history(res["history"], name)
                    st.pyplot(fig_hist)
                    plt.close()
                    st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown("---")
    
    # ──────────────────────────────────────────────────────────────────────────────────
    # SECTION 6: IMPORTANCE DES CARACTÉRISTIQUES
    # ──────────────────────────────────────────────────────────────────────────────────
    
    st.subheader("🎯 Importance des caractéristiques")
    
    with st.container():
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        
        fig_fi = plot_feature_importance(
            best_model, 
            st.session_state.col_names,
            best_name
        )
        
        if fig_fi:
            st.pyplot(fig_fi)
            plt.close()
        else:
            st.info("ℹ️ L'importance des caractéristiques n'est pas disponible pour ce type de modèle (ANN/CNN).")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ──────────────────────────────────────────────────────────────────────────────────
    # SECTION 7: RECOMMANDATIONS PERSONNALISÉES
    # ──────────────────────────────────────────────────────────────────────────────────
    
    st.subheader("💡 Recommandations personnalisées")
    
    role = st.session_state.role
    
    if best_pred == 1:
        if role == "parent":
            recommendations = """
            • 👶 **Consultez rapidement un pédiatre** ou neuropédiatre spécialisé
            • 📞 **Contactez un centre de référence** pour l'autisme (CRA)
            • 📝 **Notez les comportements observés** pour le prochain rendez-vous
            • 📚 **Renseignez-vous** sur les interventions précoces (orthophonie, ergothérapie)
            • 🤝 **Rejoignez des groupes de soutien** pour parents
            """
        else:
            recommendations = """
            • 🔬 **Réalisez une évaluation clinique approfondie** (ADOS-2, CARS, M-CHAT)
            • 🏥 **Orientez vers un centre spécialisé** si nécessaire
            • 📊 **Prescrivez des examens complémentaires** (tests auditifs, génétiques)
            • 📝 **Documentez l'historique** du développement du patient
            • 🤝 **Proposez un suivi multidisciplinaire** (psychologue, orthophoniste)
            """
        
        st.markdown(f"""
        <div style="background:#fff3cd; border-left:4px solid #ffc107;
                    border-radius:10px; padding:1rem; margin:1rem 0;">
            <strong>⚠️ Recommandations suite au risque élevé détecté :</strong><br>
            {recommendations}
        </div>
        """, unsafe_allow_html=True)
    else:
        if role == "parent":
            recommendations = """
            • ✅ **Continuez à surveiller** le développement de votre enfant
            • 📅 **Maintenez les visites régulières** chez le pédiatre
            • 🎨 **Encouragez les activités sociales** et l'interaction
            • 📖 **Stimulez le langage** à la maison (lecture, conversations)
            • 👀 **Soyez attentif** aux étapes clés du développement
            """
        else:
            recommendations = """
            • ✅ **Rassurez les parents**, le développement semble dans la norme
            • 📅 **Continuez le suivi régulier** selon le carnet de santé
            • 📊 **Surveillez les étapes clés** du développement
            • 📝 **Documentez** tout comportement inhabituel
            • 🤝 **Restez disponible** pour des consultations de suivi
            """
        
        st.markdown(f"""
        <div style="background:#d4edda; border-left:4px solid #28a745;
                    border-radius:10px; padding:1rem; margin:1rem 0;">
            <strong>✅ Recommandations suite au risque faible :</strong><br>
            {recommendations}
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ──────────────────────────────────────────────────────────────────────────────────
    # BOUTON NOUVELLE ÉVALUATION
    # ──────────────────────────────────────────────────────────────────────────────────
    
    _, col_reset, _ = st.columns([1, 2, 1])
    
    with col_reset:
        if st.button("🔄 Nouvelle évaluation", type="primary", use_container_width=True):
            for k in ["page", "reponses", "infos_enfant", "role"]:
                st.session_state[k] = _DEFAULTS[k]
            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════════════
# PIED DE PAGE
# ═══════════════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div style="text-align:center; padding:2rem; color:rgba(255,255,255,0.6);">
    <hr style="border-color:rgba(255,255,255,0.2);">
    <p>🧠 <strong>NeuroSense</strong> — Fusion des meilleurs projets GitHub sur la prédiction de l'autisme</p>
    <p style="font-size:0.75rem;">
        Sources : claredang · nagatejakachapuram · yashmahes · Shehab-Hegab ·<br>
        prasanna24062004 · MASANAMUTHU22 · Ankita-M-24 · Anvesh-3 · gokul427
    </p>
    <p style="font-size:0.7rem;">
        ⚠️ Cet outil est une aide à la décision. Consultez toujours un professionnel de santé.
    </p>
</div>
""", unsafe_allow_html=True)
