# -*- coding: utf-8 -*-
"""
NeuroSense — Détection Précoce de l'Autisme
Version avec 7 modèles (tous les modèles des 9 projets GitHub)
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import time
import os
import joblib

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ── Scikit-learn ──────────────────────────────────────────────────
from sklearn.model_selection import train_test_split, learning_curve, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_curve, roc_auc_score,
                             precision_score, recall_score, f1_score)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample
from xgboost import XGBClassifier

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

.model-comparison {
    background: white;
    border-radius: 15px;
    padding: 1rem;
    margin-top: 1rem;
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
    "best_model": None,
    "best_name": "",
    "all_models": {},
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
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ══════════════════════════════════════════════════════════════════
# Fonctions
# ══════════════════════════════════════════════════════════════════
@st.cache_data
def charger_donnees():
    """Charge ou génère des données réalistes."""
    try:
        # Essayer de charger le dataset original
        df = pd.read_csv("autism_screening.csv")
    except FileNotFoundError:
        # Générer des données synthétiques réalistes
        np.random.seed(42)
        n = 1500
        
        # Générer les scores pour les 10 questions (A1 à A10)
        # Les personnes avec ASD ont tendance à avoir des scores plus élevés
        labels = np.random.choice([0, 1], n, p=[0.68, 0.32])
        
        scores = np.zeros((n, 10), dtype=int)
        for i in range(n):
            if labels[i] == 1:  # ASD
                # Probabilité plus élevée de répondre Oui (1)
                p = [0.20, 0.80]  # [Non, Oui]
            else:  # Non-ASD
                p = [0.82, 0.18]  # [Non, Oui]
            scores[i] = np.random.choice([0, 1], 10, p=p)
        
        data = {f"A{j+1}_Score": scores[:, j] for j in range(10)}
        data.update({
            "age": np.random.randint(2, 60, n),
            "gender": np.random.choice(["m", "f"], n, p=[0.48, 0.52]),
            "ethnicity": np.random.choice(["White-European", "Asian", "Black", "Middle-Eastern", "Hispanic", "Others"], n),
            "jaundice": np.random.choice([0, 1], n, p=[0.85, 0.15]),
            "family_member_with_ASD": np.random.choice([0, 1], n, p=[0.80, 0.20]),
            "country_of_res": np.random.choice(["France", "USA", "UK", "Canada", "India", "Australia", "Others"], n),
            "Class_ASD": labels,
        })
        df = pd.DataFrame(data)
    
    return df

def ingenierie_features(df):
    """Prétraitement avancé des features."""
    df = df.copy()
    
    # Remplacer les valeurs manquantes
    df.replace({"?": "Others", "": "Others", "unknown": "Others"}, inplace=True)
    
    for c in df.select_dtypes("object").columns:
        df[c].fillna("Others", inplace=True)
    
    for c in df.select_dtypes("number").columns:
        df[c].fillna(df[c].median(), inplace=True)
    
    # Supprimer les colonnes inutiles
    drop_cols = ["ID", "id", "age_desc", "used_app_before", "relation"]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True, errors="ignore")
    
    # S'assurer que les scores des questions sont binaires
    q_cols = [f"A{i}_Score" for i in range(1, 11) if f"A{i}_Score" in df.columns]
    for c in q_cols:
        if df[c].dtype == object:
            df[c] = df[c].map({"Yes": 1, "yes": 1, "1": 1, "No": 0, "no": 0, "0": 0}).fillna(0).astype(int)
    
    # Créer un score total
    if q_cols:
        df["sum_score"] = df[q_cols].sum(axis=1)
    
    # Catégoriser l'âge
    if "age" in df.columns:
        df["age"] = pd.to_numeric(df["age"], errors="coerce").fillna(df["age"].median())
        df["age_group"] = pd.cut(df["age"], bins=[0, 4, 12, 18, 35, 60, 200], 
                                 labels=[0, 1, 2, 3, 4, 5]).astype(int)
    
    return df

def pretraiter(df, is_train=True, le_dict=None, scaler=None):
    """Prétraite les données avec encodage et normalisation."""
    df = df.copy()
    target = "Class_ASD"
    
    # Colonnes catégorielles
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
    
    # Normalisation
    num_cols = [c for c in df.select_dtypes("number").columns if c != target]
    
    if is_train:
        scaler = StandardScaler()
        if num_cols:
            df[num_cols] = scaler.fit_transform(df[num_cols])
    else:
        if scaler and num_cols:
            df[num_cols] = scaler.transform(df[num_cols])
    
    return df, le_dict, scaler

def modeles_classiques():
    """Dictionnaire de tous les modèles utilisés dans les 9 projets."""
    return {
        "🌲 Random Forest (yashmahes, prasanna)": RandomForestClassifier(
            n_estimators=100, 
            max_depth=15,
            min_samples_split=5,
            random_state=42
        ),
        "⚡ XGBoost (nagateja, MASANAMUTHU)": XGBClassifier(
            use_label_encoder=False, 
            eval_metric="logloss",
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        ),
        "📐 Logistic Regression (nagateja, Shehab)": LogisticRegression(
            max_iter=1000, 
            C=1.0,
            random_state=42
        ),
        "🔷 SVC (nagateja, prasanna)": SVC(
            probability=True, 
            kernel="rbf",
            C=1.0,
            gamma="scale",
            random_state=42
        ),
        "🌿 Decision Tree (claredang, Ankita)": DecisionTreeClassifier(
            max_depth=10,
            min_samples_split=5,
            random_state=42
        ),
        "👥 KNN (Shehab, Anvesh)": KNeighborsClassifier(
            n_neighbors=7,
            weights="distance",
            metric="minkowski"
        ),
        "🎯 Gradient Boosting (yashmahes)": GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
    }

def safe_oversample(X, y, random_state=42):
    """Oversampling équilibré des classes."""
    np.random.seed(random_state)
    classes = np.unique(y)
    max_count = max(np.bincount(y.astype(int)))
    
    X_list, y_list = [], []
    
    for cls in classes:
        X_cls = X[y == cls]
        y_cls = y[y == cls]
        
        if len(X_cls) < max_count:
            X_cls, y_cls = resample(
                X_cls, y_cls, 
                replace=True, 
                n_samples=max_count, 
                random_state=random_state
            )
        
        X_list.append(X_cls)
        y_list.append(y_cls)
    
    return np.vstack(X_list), np.hstack(y_list)

def evaluer_modele(clf, X_test, y_test, model_name):
    """Évalue un modèle et retourne les métriques."""
    y_pred = clf.predict(X_test)
    
    if hasattr(clf, "predict_proba"):
        y_proba = clf.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
    else:
        y_proba = None
        auc = accuracy_score(y_test, y_pred)
    
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "auc": auc,
        "y_pred": y_pred,
        "y_proba": y_proba
    }

def entrainer_tous_les_modeles():
    """Entraîne tous les modèles et sélectionne le meilleur."""
    with st.spinner("🧠 Chargement et entraînement des 7 modèles d'IA..."):
        # Charger les données
        df_raw = charger_donnees()
        df = ingenierie_features(df_raw)
        df, le_dict, scaler = pretraiter(df, is_train=True)
        
        # Préparer les features
        X = df.drop(columns=["Class_ASD"]).values
        y = df["Class_ASD"].values
        
        # Équilibrer les classes
        X, y = safe_oversample(X, y)
        
        # Split train/test
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Entraîner chaque modèle
        all_models = modeles_classiques()
        results = {}
        performances = {}
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, (name, clf) in enumerate(all_models.items()):
            status_text.text(f"🔄 Entraînement: {name}")
            
            # Entraîner
            clf.fit(X_tr, y_tr)
            
            # Évaluer
            metrics = evaluer_modele(clf, X_te, y_te, name)
            results[name] = {"model": clf, **metrics}
            
            # Validation croisée
            cv_scores = cross_val_score(clf, X_tr, y_tr, cv=5, scoring="accuracy")
            performances[name] = {
                "cv_mean": cv_scores.mean(),
                "cv_std": cv_scores.std(),
                "test_acc": metrics["accuracy"]
            }
            
            progress_bar.progress((i + 1) / len(all_models))
        
        # Trouver le meilleur modèle (basé sur l'AUC)
        best_name = max(results, key=lambda k: results[k]["auc"])
        best_result = results[best_name]
        
        status_text.text(f"✅ Meilleur modèle: {best_name} (AUC: {best_result['auc']:.3f})")
        time.sleep(1)
        status_text.empty()
        progress_bar.empty()
        
        # Sauvegarder dans session state
        st.session_state.update({
            "modeles_prets": True,
            "best_model": best_result["model"],
            "best_name": best_name,
            "all_models": all_models,
            "all_results": results,
            "best_accuracy": best_result["accuracy"],
            "X_train": X_tr,
            "X_test": X_te,
            "y_train": y_tr,
            "y_test": y_te,
            "y_pred": best_result["y_pred"],
            "scaler": scaler,
            "col_names": list(df.drop(columns=["Class_ASD"]).columns),
            "le_dict": le_dict,
            "model_performances": performances,
        })
        
        return results

def afficher_comparaison_modeles():
    """Affiche un tableau comparatif des performances des modèles."""
    if not st.session_state.all_results:
        return
    
    st.markdown("### 📊 Comparaison des 7 modèles")
    
    # Créer un DataFrame pour l'affichage
    data = []
    for name, metrics in st.session_state.all_results.items():
        data.append({
            "Modèle": name,
            "Accuracy": f"{metrics['accuracy']:.2%}",
            "Précision": f"{metrics['precision']:.2%}",
            "Rappel": f"{metrics['recall']:.2%}",
            "F1-Score": f"{metrics['f1']:.2%}",
            "AUC": f"{metrics['auc']:.3f}"
        })
    
    df_results = pd.DataFrame(data)
    
    # Afficher le tableau
    st.dataframe(
        df_results,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Modèle": st.column_config.TextColumn("Modèle", width="medium"),
            "Accuracy": st.column_config.TextColumn("Accuracy", width="small"),
            "Précision": st.column_config.TextColumn("Précision", width="small"),
            "Rappel": st.column_config.TextColumn("Rappel", width="small"),
            "F1-Score": st.column_config.TextColumn("F1-Score", width="small"),
            "AUC": st.column_config.TextColumn("AUC", width="small"),
        }
    )
    
    # Graphique des performances
    fig, ax = plt.subplots(figsize=(10, 5))
    model_names = [n.split(" (")[0] for n in df_results["Modèle"]]
    accuracies = [float(s.strip("%"))/100 for s in df_results["Accuracy"]]
    
    colors = ['#667eea', '#764ba2', '#f6d365', '#fda085', '#84fab0', '#8fd3f4', '#f093fb']
    bars = ax.barh(model_names, accuracies, color=colors[:len(model_names)])
    ax.set_xlabel("Accuracy")
    ax.set_title("Comparaison des performances des modèles")
    ax.set_xlim(0, 1)
    
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                f"{acc:.1%}", va='center')
    
    st.pyplot(fig)
    plt.close()

# ══════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 🧠 NeuroSense")
    st.markdown("*IA pour la détection précoce*")
    st.markdown("---")
    
    if st.session_state.modeles_prets:
        st.metric("🏆 Meilleur modèle", st.session_state.best_name.split(" (")[0])
        st.metric("🎯 Accuracy", f"{st.session_state.best_accuracy:.1%}")
        
        # Afficher les 3 meilleurs modèles
        st.markdown("---")
        st.markdown("### 🥇 Top 3 modèles")
        sorted_models = sorted(
            st.session_state.all_results.items(),
            key=lambda x: x[1]["auc"],
            reverse=True
        )[:3]
        
        for i, (name, metrics) in enumerate(sorted_models, 1):
            medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉"
            st.markdown(f"{medal} **{name.split(' (')[0]}**")
            st.caption(f"AUC: {metrics['auc']:.3f}")
    
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
# PAGE 1 - CHOIX DU RÔLE
# ══════════════════════════════════════════════════════════════════
if st.session_state.page == 1:
    st.markdown("""
    <div class="main-header">
        <h1>🧠 NeuroSense</h1>
        <p>Détection précoce des Troubles du Spectre Autistique</p>
        <p><small>Basé sur 7 modèles de Machine Learning issus de 9 projets de recherche</small></p>
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
    
    # Démarrer l'entraînement en arrière-plan
    if not st.session_state.modeles_prets:
        entrainer_tous_les_modeles()

# ══════════════════════════════════════════════════════════════════
# PAGE 2 - INFOS ENFANT
# ══════════════════════════════════════════════════════════════════
elif st.session_state.page == 2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("👶 Informations de l'enfant")
    
    col1, col2 = st.columns(2)
    with col1:
        nom = st.text_input("📝 Prénom de l'enfant", placeholder="ex: Adam, Sara...")
        age = st.number_input("🎂 Âge (années)", min_value=2, max_value=60, value=5)
    with col2:
        genre = st.selectbox("⚥ Genre", ["m", "f"], format_func=lambda x: "👦 Garçon" if x == "m" else "👧 Fille")
        ethnie = st.selectbox("🌍 Origine", ["White-European", "Asian", "Black", "Middle-Eastern", "Hispanic", "Others"])
    
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

# ══════════════════════════════════════════════════════════════════
# PAGE 3 - QUESTIONNAIRE
# ══════════════════════════════════════════════════════════════════
elif st.session_state.page == 3:
    infos = st.session_state.infos_enfant
    st.markdown(f"""
    <div class="card">
        <h3>📋 Questionnaire d'évaluation</h3>
        <p>Enfant : <strong>{infos.get('nom', '')}</strong> | Âge : {infos.get('age', '')} ans</p>
        <p><small>Basé sur les 10 questions standard du screening ASD (AQ-10)</small></p>
    </div>
    """, unsafe_allow_html=True)
    
    QUESTIONS = [
        ("A1", "😊", "Difficultés à comprendre les expressions faciales des autres ?"),
        ("A2", "💬", "Difficultés à maintenir une conversation normale ?"),
        ("A3", "🔄", "Présence de comportements répétitifs (battement des mains, balancement) ?"),
        ("A4", "🎯", "Intérêts très spécifiques et intenses qui dominent son attention ?"),
        ("A5", "😐", "Semble distant, indifférent ou sans émotion dans certaines situations ?"),
        ("A6", "🔊", "Sensibilité inhabituelle aux bruits, textures ou lumières ?"),
        ("A7", "🎮", "Préfère jouer seul plutôt qu'avec d'autres enfants ?"),
        ("A8", "📖", "Comprend le langage de façon très littérale (sans saisir l'ironie) ?"),
        ("A9", "👀", "Évite le contact visuel direct ?"),
        ("A10", "📅", "Très attaché à ses routines et résiste aux changements ?"),
    ]
    
    for qid, icon, question in QUESTIONS:
        col_icon, col_q = st.columns([1, 5])
        with col_icon:
            st.markdown(f"<div style='font-size:2rem;'>{icon}</div>", unsafe_allow_html=True)
        with col_q:
            st.markdown(f"**{question}**")
            rep = st.radio(
                "", 
                [0, 1], 
                format_func=lambda x: "❌ Non" if x == 0 else "✅ Oui",
                key=f"q_{qid}", 
                horizontal=True, 
                label_visibility="collapsed"
            )
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

# ══════════════════════════════════════════════════════════════════
# PAGE 4 - RÉSULTAT
# ══════════════════════════════════════════════════════════════════
elif st.session_state.page == 4:
    if not st.session_state.modeles_prets:
        st.warning("⏳ Chargement des modèles en cours... Veuillez patienter.")
        entrainer_tous_les_modeles()
        time.sleep(1)
        st.rerun()
    
    infos = st.session_state.infos_enfant
    total_score = sum(st.session_state.reponses.values())
    results = st.session_state.all_results
    best_name = st.session_state.best_name
    
    # Mapping des catégories
    ethnie_map = {
        "White-European": 0, "Asian": 1, "Black": 2, 
        "Middle-Eastern": 3, "Hispanic": 4, "Others": 5
    }
    
    # Construction du vecteur d'entrée
    input_data = (
        [st.session_state.reponses.get(f"A{i}", 0) for i in range(1, 11)] +
        [
            infos.get("age", 5),
            0 if infos.get("genre") == "m" else 1,
            ethnie_map.get(infos.get("ethnie"), 5),
            infos.get("jaundice", 0),
            infos.get("family_asd", 0),
            sum([st.session_state.reponses.get(f"A{i}", 0) for i in range(1, 11)]),
        ]
    )
    
    # Ajouter l'age_group si nécessaire
    age = infos.get("age", 5)
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
    
    input_data.append(age_group)
    
    # Alignement avec le scaler
    n_feat = st.session_state.scaler.n_features_in_
    vec = np.zeros(n_feat)
    for i, v in enumerate(input_data[:n_feat]):
        vec[i] = v
    vec_scaled = st.session_state.scaler.transform(vec.reshape(1, -1))
    
    # Prédictions de tous les modèles
    all_predictions = {}
    all_probas = {}
    
    for name, res in results.items():
        clf = res["model"]
        try:
            proba = float(clf.predict_proba(vec_scaled)[0][1])
            pred = int(proba > 0.5)
        except Exception as e:
            proba = 0.5
            pred = 0
        all_predictions[name] = pred
        all_probas[name] = proba
    
    # Prédiction du meilleur modèle
    best_pred = all_predictions[best_name]
    best_proba = all_probas[best_name]
    
    # Vote majoritaire
    votes_positive = sum(all_predictions.values())
    consensus = votes_positive / len(all_predictions)
    consensus_pred = int(consensus > 0.5)
    
    # Affichage résultat principal
    st.markdown(f'<div class="winner-box">🏆 Résultat final — {best_name.split(" (")[0]} (meilleur modèle)</div>', unsafe_allow_html=True)
    
    if best_pred == 1:
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
    
    # Métriques principales
    col1, col2, col3 = st.columns(3)
    col1.metric("📊 Score questionnaire", f"{total_score}/10")
    col2.metric("🤖 Probabilité (meilleur modèle)", f"{best_proba:.1%}")
    col3.metric("🎯 Consensus des modèles", f"{consensus:.0%}")
    
    # Barre de probabilité
    st.progress(best_proba)
    
    # Vote de tous les modèles
    st.markdown("---")
    st.subheader("🗳️ Vote des 7 modèles")
    
    cols = st.columns(3)
    for idx, (name, proba) in enumerate(sorted(all_probas.items(), key=lambda x: x[1], reverse=True)):
        pred_oui = all_predictions[name]
        color = "#ffcccc" if pred_oui else "#ccffcc"
        
        with cols[idx % 3]:
            st.markdown(f"""
            <div style="background:{color}; border-radius:10px; padding:0.75rem; margin:0.3rem; text-align:center;">
                <small><strong>{name}</strong></small><br>
                <small>{'⚠️ Risque' if pred_oui else '✅ Normal'} ({proba:.0%})</small>
            </div>
            """, unsafe_allow_html=True)
    
    # Résumé du consensus
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#667eea,#764ba2); border-radius:20px; padding:1.5rem; text-align:center; margin:1rem 0;">
        <h3 style="color:white;">🗳️ Résultat du consensus</h3>
        <p style="color:white; font-size:1.2rem;">
            {votes_positive}/{len(all_predictions)} modèles détectent un risque
        </p>
        <p style="color:white;">
            {'⚠️ Une consultation avec un spécialiste est recommandée' if consensus_pred == 1 else '✅ Pas de signes d\'alerte majeurs'}
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Afficher la comparaison détaillée
    with st.expander("📈 Voir la comparaison détaillée des 7 modèles"):
        afficher_comparaison_modeles()
    
    # Bouton nouvelle évaluation
    st.markdown("---")
    _, col_reset, _ = st.columns([1, 2, 1])
    with col_reset:
        if st.button("🔄 Nouvelle évaluation", type="primary", use_container_width=True):
            for k in ["page", "reponses", "infos_enfant", "role"]:
                st.session_state[k] = _DEFAULTS[k]
            st.rerun()
