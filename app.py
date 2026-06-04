# -*- coding: utf-8 -*-
"""
NeuroSense — Détection Précoce de l'Autisme
Version: Entraînement d'abord → Interface ensuite
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import time
import os

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ── Scikit-learn ──────────────────────────────────────────────────
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_curve, roc_auc_score)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample
from xgboost import XGBClassifier

# ── TensorFlow ────────────────────────────────────────────────────
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, GlobalAveragePooling1D
    from tensorflow.keras.callbacks import EarlyStopping
    TF_OK = True
except ImportError:
    TF_OK = False

# ══════════════════════════════════════════════════════════════════
# Configuration Streamlit
# ══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="NeuroSense — Détection Autisme",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════════
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
.main-header h1 { color: white; font-size: 2.5rem; margin-bottom: 0.5rem; }
.main-header p { color: rgba(255,255,255,0.85); font-size: 1rem; }

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
    font-size: 0.75rem;
    font-weight: 600;
    margin-right: 0.5rem;
}
.badge-purple { background: linear-gradient(135deg, #667eea, #764ba2); color: white; }

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

.loading-container {
    text-align: center;
    padding: 50px;
    color: white;
}
.loading-spinner {
    border: 4px solid rgba(255,255,255,0.3);
    border-radius: 50%;
    border-top: 4px solid white;
    width: 50px;
    height: 50px;
    animation: spin 1s linear infinite;
    margin: 20px auto;
}
@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# Session State
# ══════════════════════════════════════════════════════════════════
_DEFAULTS = {
    "page": 1,
    "role": None,
    "infos_enfant": {},
    "reponses": {},
    "modeles_prets": False,
    "model": None,
    "best_name": "",
    "all_results": {},
    "accuracy": 0,
    "X_train": None,
    "X_test": None,
    "y_train": None,
    "y_test": None,
    "y_pred": None,
    "scaler": None,
    "df_train": None,
    "col_names": None,
    "le_dict": None,
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ══════════════════════════════════════════════════════════════════
# Fonctions
# ══════════════════════════════════════════════════════════════════
@st.cache_data
def charger_donnees():
    """Charge ou génère des données."""
    try:
        df = pd.read_csv("autism_screening.csv")
    except FileNotFoundError:
        np.random.seed(42)
        n = 1200
        labels = np.random.choice([0, 1], n, p=[0.68, 0.32])
        scores = np.zeros((n, 10), dtype=int)
        for i in range(n):
            p = [0.20, 0.80] if labels[i] == 1 else [0.82, 0.18]
            scores[i] = np.random.choice([0, 1], 10, p=p)
        data = {f"A{j+1}_Score": scores[:, j] for j in range(10)}
        data.update({
            "age": np.random.randint(2, 13, n),
            "gender": np.random.choice(["garcon", "fille"], n),
            "ethnicity": np.random.choice(["Blanc", "Asiatique", "Noir", "Arabe", "Autre"], n),
            "jaundice": np.random.choice([0, 1], n, p=[0.85, 0.15]),
            "family_member_with_ASD": np.random.choice([0, 1], n, p=[0.80, 0.20]),
            "contry_of_res": np.random.choice(["France", "USA", "UK", "Autre"], n),
            "Class_ASD": labels,
        })
        df = pd.DataFrame(data)
    return df

def ingenierie_features(df):
    df = df.copy()
    df.replace({"?": "Others", "": "Others"}, inplace=True)
    for c in df.select_dtypes("object").columns:
        df[c].fillna("Others", inplace=True)
    for c in df.select_dtypes("number").columns:
        df[c].fillna(df[c].median(), inplace=True)
    
    drop_cols = ["ID", "id", "age_desc", "used_app_before", "relation"]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True, errors="ignore")
    
    q_cols = [f"A{i}_Score" for i in range(1, 11) if f"A{i}_Score" in df.columns]
    for c in q_cols:
        if df[c].dtype == object:
            df[c] = df[c].map({"Yes": 1, "yes": 1, "1": 1, "No": 0, "no": 0, "0": 0}).fillna(0).astype(int)
    df["sum_score"] = df[q_cols].sum(axis=1)
    
    if "age" in df.columns:
        df["age"] = pd.to_numeric(df["age"], errors="coerce").fillna(5)
        df["age_group"] = pd.cut(df["age"], bins=[0, 4, 12, 18, 40, 200], labels=[0, 1, 2, 3, 4]).astype(int)
    return df

def pretraiter(df, is_train=True, le_dict=None, scaler=None):
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
                df[c] = df[c].astype(str).apply(lambda x: x if x in le.classes_ else le.classes_[0])
                df[c] = le.transform(df[c])
    
    num_cols = [c for c in df.select_dtypes("number").columns if c != target]
    if is_train:
        scaler = StandardScaler()
        df[num_cols] = scaler.fit_transform(df[num_cols])
    else:
        if scaler and num_cols:
            df[num_cols] = scaler.transform(df[num_cols])
    
    return df, le_dict, scaler

def modeles_classiques():
    return {
        "🌲 Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "⚡ XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42),
        "🚀 Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
        "📐 Logistic Regression": LogisticRegression(max_iter=500, random_state=42),
        "🔷 SVM": SVC(probability=True, kernel="rbf", random_state=42),
        "🌿 Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=42),
        "👥 KNN": KNeighborsClassifier(n_neighbors=5),
        "🔔 Naive Bayes": GaussianNB(),
    }

def build_ann(input_dim):
    model = Sequential([
        Dense(128, activation="relu", input_shape=(input_dim,)),
        Dropout(0.3),
        Dense(64, activation="relu"),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

def build_cnn(input_dim):
    model = Sequential([
        Conv1D(64, kernel_size=3, activation="relu", input_shape=(input_dim, 1), padding="same"),
        MaxPooling1D(pool_size=2),
        Conv1D(32, kernel_size=3, activation="relu", padding="same"),
        GlobalAveragePooling1D(),
        Dense(64, activation="relu"),
        Dropout(0.3),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

def entrainer_deep(X_tr, X_te, y_tr, y_te, kind="ANN"):
    es = EarlyStopping(patience=10, restore_best_weights=True)
    if kind == "ANN":
        model = build_ann(X_tr.shape[1])
        Xtr, Xte = X_tr, X_te
    else:
        model = build_cnn(X_tr.shape[1])
        Xtr = X_tr.reshape(-1, X_tr.shape[1], 1)
        Xte = X_te.reshape(-1, X_te.shape[1], 1)
    
    hist = model.fit(Xtr, y_tr, epochs=50, batch_size=32, validation_split=0.15, callbacks=[es], verbose=0)
    y_proba = model.predict(Xte, verbose=0).ravel()
    y_pred = (y_proba > 0.5).astype(int)
    acc = accuracy_score(y_te, y_pred)
    roc = roc_auc_score(y_te, y_proba)
    return model, y_pred, y_proba, acc, roc, hist

def safe_oversample(X, y, random_state=42):
    np.random.seed(random_state)
    classes = np.unique(y)
    max_count = max(np.bincount(y.astype(int)))
    X_list, y_list = [], []
    for cls in classes:
        X_cls = X[y == cls]
        y_cls = y[y == cls]
        if len(X_cls) < max_count:
            X_cls, y_cls = resample(X_cls, y_cls, replace=True, n_samples=max_count, random_state=random_state)
        X_list.append(X_cls)
        y_list.append(y_cls)
    return np.vstack(X_list), np.hstack(y_list)

# ══════════════════════════════════════════════════════════════════
# FONCTION PRINCIPALE D'ENTRAÎNEMENT
# ══════════════════════════════════════════════════════════════════
def entrainer_tous_les_modeles():
    """Entraîne tous les modèles et stocke les résultats dans session_state."""
    
    st.markdown("""
    <div class="loading-container">
        <h2>🧠 NeuroSense</h2>
        <p>Entraînement des modèles d'intelligence artificielle en cours...</p>
        <div class="loading-spinner"></div>
        <p style="font-size:0.9rem; margin-top:20px;">Cela peut prendre 30 à 60 secondes</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Chargement des données
    progress_text = st.empty()
    progress_text.markdown("📂 Chargement des données...")
    
    df_raw = charger_donnees()
    df = ingenierie_features(df_raw)
    
    progress_text.markdown("🔧 Prétraitement des données...")
    df, le_dict, scaler = pretraiter(df, is_train=True)
    
    X = df.drop(columns=["Class_ASD"]).values
    y = df["Class_ASD"].values
    
    progress_text.markdown("⚖️ Équilibrage des données (oversampling)...")
    X, y = safe_oversample(X, y)
    
    progress_text.markdown("✂️ Division entraînement/test...")
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    resultats = {}
    progress_bar = st.progress(0)
    
    # Modèles classiques
    modeles = modeles_classiques()
    n_modeles = len(modeles) + (2 if TF_OK else 0)
    idx = 0
    
    for nom, clf in modeles.items():
        progress_text.markdown(f"🤖 Entraînement de {nom}...")
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)
        y_proba = clf.predict_proba(X_te)[:, 1] if hasattr(clf, "predict_proba") else None
        acc = accuracy_score(y_te, y_pred)
        roc = roc_auc_score(y_te, y_proba) if y_proba is not None else acc
        resultats[nom] = {"modele": clf, "accuracy": acc, "auc": roc, "y_pred": y_pred, "y_proba": y_proba, "history": None}
        idx += 1
        progress_bar.progress(idx / n_modeles)
    
    # Deep Learning
    if TF_OK:
        progress_text.markdown("🧠 Entraînement du réseau de neurones ANN...")
        ann, yp, yprob, acc, roc, hist = entrainer_deep(X_tr, X_te, y_tr, y_te, "ANN")
        resultats["🧠 ANN"] = {"modele": ann, "accuracy": acc, "auc": roc, "y_pred": yp, "y_proba": yprob, "history": hist}
        idx += 1
        progress_bar.progress(idx / n_modeles)
        
        progress_text.markdown("📡 Entraînement du réseau CNN...")
        cnn, yp, yprob, acc, roc, hist = entrainer_deep(X_tr, X_te, y_tr, y_te, "CNN")
        resultats["📡 CNN"] = {"modele": cnn, "accuracy": acc, "auc": roc, "y_pred": yp, "y_proba": yprob, "history": hist}
        idx += 1
        progress_bar.progress(idx / n_modeles)
    
    # Meilleur modèle
    best_name = max(resultats, key=lambda k: resultats[k]["auc"])
    best = resultats[best_name]
    col_names = list(df.drop(columns=["Class_ASD"]).columns)
    
    st.session_state.update({
        "modeles_prets": True,
        "model": best["modele"],
        "best_name": best_name,
        "all_results": resultats,
        "accuracy": best["accuracy"],
        "X_train": X_tr,
        "X_test": X_te,
        "y_train": y_tr,
        "y_test": y_te,
        "y_pred": best["y_pred"],
        "scaler": scaler,
        "col_names": col_names,
        "le_dict": le_dict,
    })
    
    progress_text.empty()
    progress_bar.empty()
    st.rerun()

# ══════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 🧠 NeuroSense")
    st.markdown("*IA pour la détection précoce*")
    st.markdown("---")
    
    if st.session_state.modeles_prets:
        st.metric("🏆 Meilleur modèle", st.session_state.best_name.split(" ", 1)[-1] if " " in st.session_state.best_name else st.session_state.best_name)
        st.metric("🎯 Accuracy", f"{st.session_state.accuracy:.1%}")
    else:
        st.info("⏳ Entraînement en cours...")
    
    st.markdown("---")
    st.markdown("### 📌 Progression")
    pages = ["Choix du rôle", "Infos enfant", "Questionnaire", "Résultat"]
    for i, p in enumerate(pages, 1):
        if i < st.session_state.page:
            st.markdown(f"✅ {i}. {p}")
        elif i == st.session_state.page:
            st.markdown(f"🔵 **{i}. {p}**")
        else:
            st.markdown(f"⚪ {i}. {p}")

# ══════════════════════════════════════════════════════════════════
# AFFICHAGE PRINCIPAL
# ══════════════════════════════════════════════════════════════════

# Si les modèles ne sont pas encore prêts, on les entraîne
if not st.session_state.modeles_prets:
    entrainer_tous_les_modeles()

# PAGE 1 - CHOIX DU RÔLE
elif st.session_state.page == 1:
    st.markdown("""
    <div class="main-header">
        <h1>🧠 NeuroSense</h1>
        <p>Détection précoce des Troubles du Spectre Autistique</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("👋 Qui êtes-vous ?")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style="text-align:center; padding:2rem;">
            <span style="font-size:4rem;">👨‍👩‍👧</span>
            <h3>Parent</h3>
            <p>Complétez le questionnaire pour votre enfant</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("📝 Je suis un parent", use_container_width=True, key="btn_parent"):
            st.session_state.role = "parent"
            st.session_state.page = 2
            st.rerun()
    
    with col2:
        st.markdown("""
        <div style="text-align:center; padding:2rem;">
            <span style="font-size:4rem;">👨‍⚕️</span>
            <h3>Médecin</h3>
            <p>Évaluez votre patient avec notre outil</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("🩺 Je suis médecin", use_container_width=True, key="btn_medecin"):
            st.session_state.role = "medecin"
            st.session_state.page = 2
            st.rerun()
    
    st.markdown("</div>", unsafe_allow_html=True)

# PAGE 2 - INFOS ENFANT
elif st.session_state.page == 2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("👶 Informations de l'enfant")
    
    col1, col2 = st.columns(2)
    with col1:
        nom = st.text_input("📝 Prénom de l'enfant", placeholder="ex: Adam, Sara...")
        age = st.number_input("🎂 Âge (années)", min_value=2, max_value=12, value=5)
    with col2:
        genre = st.selectbox("⚥ Genre", ["garcon", "fille"], format_func=lambda x: "👦 Garçon" if x == "garcon" else "👧 Fille")
        ethnie = st.selectbox("🌍 Origine", ["Blanc", "Asiatique", "Noir", "Arabe", "Autre"])
    
    col3, col4 = st.columns(2)
    with col3:
        jaundice = st.radio("🟡 Ictère à la naissance ?", [0, 1], format_func=lambda x: "❌ Non" if x == 0 else "✅ Oui", horizontal=True)
    with col4:
        family_asd = st.radio("👨‍👩‍👧 Antécédents familiaux d'autisme ?", [0, 1], format_func=lambda x: "❌ Non" if x == 0 else "✅ Oui", horizontal=True)
    
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
                    "nom": nom, "age": age, "genre": genre, "ethnie": ethnie,
                    "jaundice": jaundice, "family_asd": family_asd
                }
                st.session_state.page = 3
                st.rerun()
            else:
                st.error("⚠️ Veuillez entrer le prénom de l'enfant")

# PAGE 3 - QUESTIONNAIRE
elif st.session_state.page == 3:
    infos = st.session_state.infos_enfant
    st.markdown(f"""
    <div class="card">
        <h3>📋 Questionnaire d'évaluation</h3>
        <p>Enfant : <strong>{infos.get('nom', '')}</strong> | Âge : {infos.get('age', '')} ans</p>
    </div>
    """, unsafe_allow_html=True)
    
    QUESTIONS = [
        ("A1", "😊", "Difficultés à comprendre les expressions faciales ?"),
        ("A2", "💬", "Difficultés à maintenir une conversation ?"),
        ("A3", "🔄", "Comportements répétitifs ?"),
        ("A4", "🎯", "Intérêts très spécifiques et intenses ?"),
        ("A5", "😐", "Semble distant ou sans émotion ?"),
        ("A6", "🔊", "Sensibilité aux bruits ou textures ?"),
        ("A7", "🎮", "Préfère jouer seul ?"),
        ("A8", "📖", "Langage très littéral ?"),
        ("A9", "👀", "Évite le contact visuel ?"),
        ("A10", "📅", "Très attaché à ses routines ?"),
    ]
    
    for qid, icon, question in QUESTIONS:
        col_icon, col_q = st.columns([1, 5])
        with col_icon:
            st.markdown(f"<div style='font-size:2rem;'>{icon}</div>", unsafe_allow_html=True)
        with col_q:
            st.markdown(f"**{question}**")
            rep = st.radio("", [0, 1], format_func=lambda x: "❌ Non" if x == 0 else "✅ Oui",
                          key=f"q_{qid}", horizontal=True, label_visibility="collapsed")
            if rep is not None:
                st.session_state.reponses[qid] = rep
        st.markdown("---")
    
    total = len(st.session_state.reponses)
    if total > 0:
        score = sum(st.session_state.reponses.values())
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#667eea,#764ba2); border-radius:20px; padding:1rem; text-align:center;">
            <span style="color:white;">📊 Progression : {total}/10 | Score : {score}/{total}</span>
        </div>
        """, unsafe_allow_html=True)
    
    col_back, col_next, _ = st.columns([1, 2, 1])
    with col_back:
        if st.button("⬅️ Retour", use_container_width=True):
            st.session_state.page = 2
            st.rerun()
    with col_next:
        if total == 10:
            if st.button("🔮 Voir le résultat", type="primary", use_container_width=True):
                st.session_state.page = 4
                st.rerun()
        else:
            st.warning(f"⚠️ {10 - total} question(s) restante(s)")

# PAGE 4 - RÉSULTAT
elif st.session_state.page == 4:
    infos = st.session_state.infos_enfant
    total_score = sum(st.session_state.reponses.values())
    ar = st.session_state.all_results
    best_name = st.session_state.best_name
    
    # Préparation des données patient
    ethnie_map = {"Blanc": 0, "Asiatique": 1, "Noir": 2, "Arabe": 3, "Autre": 4}
    input_data = (
        [st.session_state.reponses.get(f"A{i}", 0) for i in range(1, 11)] +
        [infos.get("age", 5),
         0 if infos.get("genre") == "garcon" else 1,
         ethnie_map.get(infos.get("ethnie"), 0),
         infos.get("jaundice", 0),
         infos.get("family_asd", 0),
         sum([st.session_state.reponses.get(f"A{i}", 0) for i in range(1, 11)]),
         0 if infos.get("age", 5) <= 4 else 1 if infos.get("age", 5) <= 12 else 2]
    )
    
    n_feat = st.session_state.scaler.n_features_in_
    vec = np.zeros(n_feat)
    for i, v in enumerate(input_data[:n_feat]):
        vec[i] = v
    vec_scaled = st.session_state.scaler.transform(vec.reshape(1, -1))
    
    # Prédiction du meilleur modèle
    best_clf = st.session_state.model
    try:
        if "ANN" in best_name or "CNN" in best_name:
            prob = float(best_clf.predict(vec_scaled, verbose=0).ravel()[0])
        else:
            prob = float(best_clf.predict_proba(vec_scaled)[0][1])
    except:
        prob = 0.5
    
    prediction = int(prob > 0.5)
    
    # Affichage résultat
    st.markdown(f'<div class="winner-box">🏆 Résultat — {best_name}</div>', unsafe_allow_html=True)
    
    if prediction == 1:
        st.markdown("""
        <div class="result-card" style="background:linear-gradient(135deg,#f093fb,#f5576c);">
            <span style="font-size:4rem;">🚨</span>
            <h1 style="color:white;">Risque élevé détecté</h1>
            <p style="color:white;">Une évaluation clinique approfondie est recommandée</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="result-card" style="background:linear-gradient(135deg,#84fab0,#8fd3f4);">
            <span style="font-size:4rem;">✅</span>
            <h1 style="color:#2c3e50;">Risque faible</h1>
            <p style="color:#2c3e50;">Le développement semble dans la norme</p>
        </div>
        """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("📊 Score total", f"{total_score}/10")
    col2.metric("🤖 Probabilité", f"{prob:.1%}")
    col3.metric("🎯 Accuracy modèle", f"{st.session_state.accuracy:.1%}")
    st.progress(prob)
    
    # Vote de tous les modèles
    st.markdown("---")
    st.subheader("🗳️ Vote de tous les modèles")
    cols = st.columns(3)
    votes = 0
    for idx, (nom, res) in enumerate(sorted(ar.items(), key=lambda x: x[1]["auc"], reverse=True)):
        clf = res["modele"]
        try:
            if "ANN" in nom or "CNN" in nom:
                p = float(clf.predict(vec_scaled, verbose=0).ravel()[0])
            else:
                p = float(clf.predict_proba(vec_scaled)[0][1])
        except:
            p = 0.5
        pred_oui = int(p > 0.5)
        if pred_oui:
            votes += 1
        color = "#ffcccc" if pred_oui else "#ccffcc"
        with cols[idx % 3]:
            st.markdown(f"""
            <div style="background:{color}; border-radius:10px; padding:0.5rem; margin:0.2rem; text-align:center;">
                <small><strong>{nom}</strong></small><br>
                <small>{'⚠️ Risque' if pred_oui else '✅ Normal'} ({p:.0%})</small>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#667eea,#764ba2); border-radius:20px; padding:1rem; text-align:center; margin:1rem 0;">
        <h3 style="color:white;">🗳️ {votes}/{len(ar)} modèles détectent un risque</h3>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    _, col_reset, _ = st.columns([1, 2, 1])
    with col_reset:
        if st.button("🔄 Nouvelle évaluation", type="primary", use_container_width=True):
            for k in ["page", "reponses", "infos_enfant", "role"]:
                st.session_state[k] = _DEFAULTS[k]
            st.rerun()
