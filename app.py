# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════╗
║          NeuroSense — Détection Précoce de l'Autisme             ║
║  Fusion des meilleurs repos :                                    ║
║  • claredang    → ANN, CNN, 6 modèles, 3 datasets, Learning Curves║
║  • nagatejakachapuram → Feature Engineering, ROC-AUC, Oversampling║
║  • NeuroSense   → Interface moderne 4 pages                      ║
╚══════════════════════════════════════════════════════════════════╝
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

# ── Scikit-learn ──────────────────────────────────────────────────
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing   import LabelEncoder, StandardScaler
from sklearn.metrics         import (accuracy_score, classification_report,
                                     confusion_matrix, roc_curve, auc,
                                     roc_auc_score)
from sklearn.linear_model    import LogisticRegression
from sklearn.svm             import SVC
from sklearn.neighbors       import KNeighborsClassifier
from sklearn.naive_bayes     import GaussianNB
from sklearn.ensemble        import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree            import DecisionTreeClassifier
from xgboost                 import XGBClassifier

# ── Fonction d'oversampling manuel CORRIGÉE ────────────────────────
def RandomOverSampler_fit_resample(X, y, random_state=42):
    """
    Oversampling manuel totalement corrigé.
    Remplace imblearn.RandomOverSampler.
    """
    np.random.seed(random_state)
    classes = np.unique(y)
    max_count = 0
    for cls in classes:
        count = np.sum(y == cls)
        if count > max_count:
            max_count = count
    
    X_res = []
    y_res = []
    
    for cls in classes:
        # Récupérer les indices de la classe actuelle
        indices = np.where(y == cls)[0]
        X_class = X[indices]
        y_class = y[indices]
        
        # Ajouter toutes les instances originales
        X_res.append(X_class)
        y_res.append(y_class)
        
        # Si c'est une classe minoritaire, ajouter des doublons
        if len(X_class) < max_count:
            n_need = max_count - len(X_class)
            # Choisir aléatoirement des indices à dupliquer
            extra_indices = np.random.choice(indices, n_need, replace=True)
            X_res.append(X[extra_indices])
            y_res.append(y[extra_indices])
    
    # Concaténer tous les résultats
    X_final = np.vstack(X_res)
    y_final = np.concatenate(y_res)
    
    return X_final, y_final

# ── TensorFlow / Keras (ANN + CNN) ────────────────────────────────
try:
    import tensorflow as tf
    from tensorflow.keras.models     import Sequential
    from tensorflow.keras.layers     import (Dense, Dropout, Conv1D,
                                             MaxPooling1D, Flatten,
                                             GlobalAveragePooling1D)
    from tensorflow.keras.callbacks  import EarlyStopping
    from tensorflow.keras.utils      import to_categorical
    TF_OK = True
except ImportError:
    TF_OK = False

# ══════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title = "NeuroSense — Détection Autisme",
    page_icon  = "🧠",
    layout     = "wide",
    initial_sidebar_state = "expanded"
)

# ══════════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
* { font-family:'Inter',sans-serif; }

.stApp { background:linear-gradient(135deg,#667eea 0%,#764ba2 100%); }

.main-header {
    background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);
    padding:2rem; border-radius:20px; text-align:center;
    margin-bottom:2rem; box-shadow:0 10px 40px rgba(0,0,0,.2);
}
.main-header h1 { color:white; font-size:3rem; margin-bottom:.5rem; }
.main-header p  { color:rgba(255,255,255,.85); font-size:1.1rem; }

.card {
    background:white; border-radius:20px; padding:1.5rem;
    margin-bottom:1.5rem; box-shadow:0 4px 20px rgba(0,0,0,.1);
}

.stButton>button {
    background:linear-gradient(135deg,#667eea,#764ba2);
    color:white; border:none; border-radius:50px;
    padding:.75rem 2rem; font-size:1rem; font-weight:600;
    transition:all .3s;
}
.stButton>button:hover { transform:scale(1.05); box-shadow:0 5px 20px rgba(102,126,234,.4); }

.result-card { border-radius:20px; padding:2rem; text-align:center; margin-top:1.5rem; }

.badge {
    display:inline-block; padding:.25rem .75rem;
    border-radius:50px; font-size:.75rem; font-weight:600; margin-right:.5rem;
}
.badge-purple { background:linear-gradient(135deg,#667eea,#764ba2); color:white; }

.winner-box {
    background:linear-gradient(135deg,#f6d365,#fda085);
    border-radius:15px; padding:1rem; text-align:center;
    font-weight:bold; font-size:1.1rem; margin:1rem 0;
}

@keyframes fadeIn {
    from{opacity:0;transform:translateY(20px);}
    to  {opacity:1;transform:translateY(0);}
}
.fade-in { animation:fadeIn .6s ease-out; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════
_DEFAULTS = dict(
    page=1, role=None, infos_enfant={}, reponses={},
    model_entraine=False, model=None, best_name="",
    all_results={}, accuracy=0,
    X_train=None, X_test=None,
    y_train=None, y_test=None,
    y_pred=None,  scaler=None,
    df_train=None,
)
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ══════════════════════════════════════════════════════════════════
# DONNÉES  (nagatejakachapuram : feature engineering complet)
# ══════════════════════════════════════════════════════════════════
@st.cache_data
def charger_donnees():
    """Charge ou génère des données réalistes avec corrélations."""
    try:
        df = pd.read_csv("autism_screening.csv")
        found = True
    except FileNotFoundError:
        np.random.seed(42)
        n = 1200
        labels = np.random.choice([0, 1], n, p=[0.68, 0.32])
        scores = np.zeros((n, 10), dtype=int)
        for i in range(n):
            p = [0.20, 0.80] if labels[i] == 1 else [0.82, 0.18]
            scores[i] = np.random.choice([0, 1], 10, p=p)
        data = {f"A{j+1}_Score": scores[:, j] for j in range(10)}
        data.update(dict(
            age           = np.random.randint(2, 13, n),
            gender        = np.random.choice(["garcon", "fille"], n),
            ethnicity     = np.random.choice(["Blanc","Asiatique","Noir","Arabe","Autre"], n),
            jaundice      = np.random.choice([0, 1], n, p=[0.85, 0.15]),
            family_member_with_ASD = np.random.choice([0, 1], n, p=[0.80, 0.20]),
            contry_of_res = np.random.choice(["France","USA","UK","Autre"], n),
            Class_ASD     = labels,
        ))
        df    = pd.DataFrame(data)
        found = False
    return df, found


def ingenierie_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature Engineering — nagatejakachapuram style :
    • sum_score  : total des 10 questions
    • age_group  : Toddler / Kid / Teenager / Young / Senior
    • Suppression colonnes inutiles
    """
    df = df.copy()
    # Nettoyage
    df.replace({"?": "Others", "": "Others"}, inplace=True)
    for c in df.select_dtypes("object").columns:
        df[c].fillna("Others", inplace=True)
    for c in df.select_dtypes("number").columns:
        df[c].fillna(df[c].median(), inplace=True)

    # Supprimer colonnes inutiles
    drop_cols = ["ID", "id", "age_desc", "used_app_before", "relation"]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    # sum_score
    q_cols = [f"A{i}_Score" for i in range(1, 11) if f"A{i}_Score" in df.columns]
    for c in q_cols:
        if df[c].dtype == object:
            df[c] = df[c].map({"Yes":1,"yes":1,"1":1,"No":0,"no":0,"0":0}).fillna(0).astype(int)
    df["sum_score"] = df[q_cols].sum(axis=1)

    # age_group — converti en int directement pour éviter Categorical dtype
    if "age" in df.columns:
        df["age"] = pd.to_numeric(df["age"], errors="coerce").fillna(5)
        def age_to_group(a):
            if a <= 4:   return 0  # Toddler
            elif a <= 12: return 1  # Kid
            elif a <= 18: return 2  # Teenager
            elif a <= 40: return 3  # Young
            else:         return 4  # Senior
        df["age_group"] = df["age"].apply(age_to_group).astype(int)
    return df


def pretraiter(df: pd.DataFrame, is_train=True, le_dict=None, scaler=None):
    """Encodage + normalisation."""
    df = df.copy()
    target = "Class_ASD"

    # Convertir toute colonne Categorical résiduelle en string
    for c in df.columns:
        if hasattr(df[c], "cat"):
            df[c] = df[c].astype(str)

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
                    lambda x: x if x in le.classes_ else le.classes_[0])
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


# ══════════════════════════════════════════════════════════════════
# MODÈLES CLASSIQUES  (nagatejakachapuram + claredang)
# ══════════════════════════════════════════════════════════════════
def modeles_classiques():
    return {
        "🌲 Random Forest":       RandomForestClassifier(n_estimators=200, random_state=42),
        "⚡ XGBoost":             XGBClassifier(use_label_encoder=False,
                                                eval_metric="logloss", random_state=42),
        "🚀 Gradient Boosting":   GradientBoostingClassifier(n_estimators=150, random_state=42),
        "📐 Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "🔷 SVM":                 SVC(probability=True, kernel="rbf", random_state=42),
        "🌿 Decision Tree":       DecisionTreeClassifier(max_depth=10, random_state=42),
        "👥 KNN":                 KNeighborsClassifier(n_neighbors=7),
        "🔔 Naive Bayes":         GaussianNB(),
    }


# ══════════════════════════════════════════════════════════════════
# ANN + CNN  (claredang — meilleur repo)
# ══════════════════════════════════════════════════════════════════
def build_ann(input_dim: int):
    if not TF_OK:
        return None
    model = Sequential([
        Dense(128, activation="relu", input_shape=(input_dim,)),
        Dropout(0.3),
        Dense(64,  activation="relu"),
        Dropout(0.2),
        Dense(32,  activation="relu"),
        Dense(1,   activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def build_cnn(input_dim: int):
    if not TF_OK:
        return None
    model = Sequential([
        Conv1D(64, kernel_size=3, activation="relu",
               input_shape=(input_dim, 1), padding="same"),
        MaxPooling1D(pool_size=2),
        Conv1D(32, kernel_size=3, activation="relu", padding="same"),
        GlobalAveragePooling1D(),
        Dense(64, activation="relu"),
        Dropout(0.3),
        Dense(1,   activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def entrainer_deep(X_tr, X_te, y_tr, y_te, kind="ANN"):
    """Entraîne ANN ou CNN et retourne résultats + historique."""
    if not TF_OK:
        return None, None, None, 0, 0, None
    
    es = EarlyStopping(patience=10, restore_best_weights=True)
    if kind == "ANN":
        model = build_ann(X_tr.shape[1])
        Xtr, Xte = X_tr, X_te
    else:
        model = build_cnn(X_tr.shape[1])
        Xtr = X_tr.reshape(-1, X_tr.shape[1], 1)
        Xte = X_te.reshape(-1, X_te.shape[1], 1)

    hist = model.fit(Xtr, y_tr, epochs=80, batch_size=32,
                     validation_split=0.15, callbacks=[es], verbose=0)

    y_proba = model.predict(Xte, verbose=0).ravel()
    y_pred  = (y_proba > 0.5).astype(int)
    acc     = accuracy_score(y_te, y_pred)
    roc     = roc_auc_score(y_te, y_proba)
    return model, y_pred, y_proba, acc, roc, hist


# ══════════════════════════════════════════════════════════════════
# PIPELINE D'ENTRAÎNEMENT COMPLET
# ══════════════════════════════════════════════════════════════════
def pipeline_complet(df_raw: pd.DataFrame):
    """
    1. Feature Engineering
    2. Prétraitement
    3. Oversampling
    4. Entraînement de TOUS les modèles (classiques + ANN + CNN)
    5. Sélection automatique du meilleur (ROC-AUC)
    """
    target = "Class_ASD"

    df = ingenierie_features(df_raw)
    df, le_dict, scaler = pretraiter(df, is_train=True)

    X = df.drop(columns=[target]).values
    y = df[target].values

    # Oversampling manuel CORRIGÉ
    try:
        X, y = RandomOverSampler_fit_resample(X, y, random_state=42)
    except Exception as e:
        st.warning(f"Oversampling ignoré: {e}")
        # Continuer avec les données originales

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    resultats = {}
    progress  = st.progress(0, text="🤖 Entraînement des modèles…")
    n_models  = len(modeles_classiques()) + (2 if TF_OK else 0)
    step = 0

    # ── Modèles classiques ────────────────────────────────────────
    for nom, clf in modeles_classiques().items():
        try:
            clf.fit(X_tr, y_tr)
            yp    = clf.predict(X_te)
            yprob = clf.predict_proba(X_te)[:, 1] if hasattr(clf, "predict_proba") else None
            acc   = accuracy_score(y_te, yp)
            roc   = roc_auc_score(y_te, yprob) if yprob is not None else acc
            resultats[nom] = dict(modele=clf, accuracy=acc, auc=roc,
                                   y_pred=yp, y_proba=yprob, history=None)
        except Exception as e:
            resultats[nom] = dict(modele=clf, accuracy=0, auc=0,
                                   y_pred=None, y_proba=None, history=None)
        step += 1
        progress.progress(step / n_models, text=f"✅ {nom}")

    # ── ANN ───────────────────────────────────────────────────────
    if TF_OK:
        try:
            ann, yp, yprob, acc, roc, hist = entrainer_deep(X_tr, X_te, y_tr, y_te, "ANN")
            if ann is not None:
                resultats["🧠 ANN"] = dict(modele=ann, accuracy=acc, auc=roc,
                                            y_pred=yp, y_proba=yprob, history=hist)
        except Exception as e:
            resultats["🧠 ANN"] = dict(modele=None, accuracy=0, auc=0,
                                        y_pred=None, y_proba=None, history=None)
        step += 1
        progress.progress(step / n_models, text="✅ ANN")

        try:
            cnn, yp, yprob, acc, roc, hist = entrainer_deep(X_tr, X_te, y_tr, y_te, "CNN")
            if cnn is not None:
                resultats["📡 CNN"] = dict(modele=cnn, accuracy=acc, auc=roc,
                                            y_pred=yp, y_proba=yprob, history=hist)
        except Exception as e:
            resultats["📡 CNN"] = dict(modele=None, accuracy=0, auc=0,
                                        y_pred=None, y_proba=None, history=None)
        step += 1
        progress.progress(1.0, text="✅ CNN")

    progress.empty()

    # ── Meilleur modèle (ROC-AUC) ─────────────────────────────────
    # Filtrer les modèles valides
    valid_results = {k: v for k, v in resultats.items() if v["auc"] > 0}
    if valid_results:
        best_name = max(valid_results, key=lambda k: valid_results[k]["auc"])
        best = valid_results[best_name]
    else:
        best_name = list(resultats.keys())[0]
        best = resultats[best_name]

    col_names = list(df.drop(columns=[target]).columns)

    return (best["modele"], best_name, resultats, le_dict, scaler,
            X_tr, X_te, y_tr, y_te, best.get("y_pred", np.zeros(len(y_te))), 
            best.get("accuracy", 0), col_names)


# ══════════════════════════════════════════════════════════════════
# LEARNING CURVE  (claredang — signature)
# ══════════════════════════════════════════════════════════════════
def plot_learning_curve(estimator, X, y, nom):
    """Reproduit les learning curves du repo claredang."""
    try:
        sizes, tr_scores, val_scores = learning_curve(
            estimator, X, y,
            cv=5, scoring="accuracy",
            train_sizes=np.linspace(0.1, 1.0, 8),
            n_jobs=-1
        )
        tr_mean  = np.mean(tr_scores,  axis=1)
        val_mean = np.mean(val_scores, axis=1)

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(sizes, tr_mean,  "o-", color="#667eea", label="Entraînement")
        ax.plot(sizes, val_mean, "s-", color="#f5576c", label="Validation")
        ax.fill_between(sizes,
                        tr_mean  - np.std(tr_scores,  axis=1),
                        tr_mean  + np.std(tr_scores,  axis=1),
                        alpha=0.15, color="#667eea")
        ax.fill_between(sizes,
                        val_mean - np.std(val_scores, axis=1),
                        val_mean + np.std(val_scores, axis=1),
                        alpha=0.15, color="#f5576c")
        ax.set_xlabel("Taille de l'échantillon d'entraînement")
        ax.set_ylabel("Précision")
        ax.set_title(f"Courbe d'apprentissage — {nom}")
        ax.legend(); ax.grid(True, alpha=0.3)
        return fig
    except Exception:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.text(0.5, 0.5, "Learning curve non disponible", ha="center", va="center")
        return fig


# ══════════════════════════════════════════════════════════════════
# CHARGEMENT + ENTRAÎNEMENT (une seule fois)
# ══════════════════════════════════════════════════════════════════
if st.session_state.df_train is None:
    with st.spinner("📂 Chargement des données…"):
        df_raw, _ = charger_donnees()
        st.session_state.df_train = df_raw
    st.rerun()

if not st.session_state.model_entraine and st.session_state.df_train is not None:
    with st.spinner("🧠 Entraînement des modèles en cours... (patientez)"):
        (model, best_name, all_results, le_dict, scaler,
         X_tr, X_te, y_tr, y_te, y_pred, accuracy, col_names) = \
            pipeline_complet(st.session_state.df_train)

        st.session_state.update(dict(
            model          = model,
            best_name      = best_name,
            all_results    = all_results,
            le_dict        = le_dict,
            scaler         = scaler,
            X_train        = X_tr,
            X_test         = X_te,
            y_train        = y_tr,
            y_test         = y_te,
            y_pred         = y_pred,
            accuracy       = accuracy,
            col_names      = col_names,
            model_entraine = True,
        ))
    st.rerun()

# ══════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:1rem;">
        <h2 style="color:white;">🧠 NeuroSense</h2>
        <p style="color:rgba(255,255,255,.7);">IA pour la détection précoce</p>
    </div>""", unsafe_allow_html=True)
    st.markdown("---")

    if st.session_state.model_entraine:
        ar = st.session_state.all_results
        bn = st.session_state.best_name
        st.metric("🏆 Meilleur modèle", bn.split(" ", 1)[-1] if " " in bn else bn)
        st.metric("🎯 Accuracy",        f"{st.session_state.accuracy:.1%}")
        if bn in ar:
            st.metric("📊 ROC-AUC",         f"{ar[bn]['auc']:.3f}")
        st.metric("🧠 Deep Learning",   "✅ ANN + CNN" if TF_OK else "❌ TF non installé")

        st.markdown("---")
        st.subheader("📊 Classement")
        valid_items = [(nom, res) for nom, res in ar.items() if res["auc"] > 0]
        for i, (nom, res) in enumerate(
            sorted(valid_items, key=lambda x: x[1]["auc"], reverse=True), 1
        ):
            medal = "🥇" if i == 1 else ("🥈" if i == 2 else ("🥉" if i == 3 else "  "))
            short_name = nom.split(" ", 1)[-1] if " " in nom else nom
            st.caption(f"{medal} {short_name}: **{res['auc']:.3f}**")

    st.markdown("---")
    st.subheader("📌 Progression")
    for i, e in enumerate(["Choix du rôle","Infos enfant","Questionnaire","Résultat"], 1):
        if   i < st.session_state.page: st.markdown(f"✅ {i}. {e}")
        elif i == st.session_state.page: st.markdown(f"🔵 **{i}. {e}**")
        else: st.markdown(f"⚪ {i}. {e}")

# ══════════════════════════════════════════════════════════════════
# EN-TÊTE
# ══════════════════════════════════════════════════════════════════
n_mod = len(st.session_state.all_results) if st.session_state.model_entraine else "..."
st.markdown(f"""
<div class="main-header fade-in">
    <h1>🧠 NeuroSense</h1>
    <p>Détection précoce des Troubles du Spectre Autistique par Intelligence Artificielle</p>
    <div style="margin-top:1rem;">
        <span class="badge badge-purple">🤖 {n_mod} modèles comparés</span>
        <span class="badge" style="background:#4CAF50;color:white;">🧠 ANN + CNN</span>
        <span class="badge" style="background:#ff9800;color:white;">🏆 Meilleur auto-sélectionné</span>
        <span class="badge" style="background:#e91e63;color:white;">📈 Learning Curves</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# PAGE 1 — RÔLE
# ══════════════════════════════════════════════════════════════════
if st.session_state.page == 1:
    st.markdown('<div class="card fade-in">', unsafe_allow_html=True)
    st.subheader("👋 Qui êtes-vous ?")
    st.markdown("---")
    c1, c2 = st.columns(2)
    for col, role, icon, label, desc, btn_label in [
        (c1, "parent",  "👨‍👩‍👧", "Parent",  "Complétez le questionnaire pour votre enfant",  "📝 Je suis un parent"),
        (c2, "medecin", "👨‍⚕️", "Médecin", "Évaluez votre patient avec notre outil d'aide", "🩺 Je suis médecin"),
    ]:
        with col:
            st.markdown(f"""
            <div style="text-align:center;background:linear-gradient(135deg,#667eea20,#764ba220);
                        border-radius:20px;padding:2rem;">
                <span style="font-size:4rem;">{icon}</span>
                <h3>{label}</h3><p>{desc}</p>
            </div>""", unsafe_allow_html=True)
            if st.button(btn_label, key=f"btn_{role}", use_container_width=True):
                st.session_state.role = role
                st.session_state.page = 2
                st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# PAGE 2 — INFOS ENFANT
# ══════════════════════════════════════════════════════════════════
elif st.session_state.page == 2:
    st.markdown('<div class="card fade-in">', unsafe_allow_html=True)
    st.subheader("👶 Informations de l'enfant")
    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        nom = st.text_input("📝 Nom de l'enfant", placeholder="ex: Adam, Sara, Lucas…")
        age = st.number_input("🎂 Âge (années)", min_value=2, max_value=12, value=5)
    with c2:
        genre  = st.radio("⚥ Genre", ["garcon","fille"],
                           format_func=lambda x:"👦 Garçon" if x=="garcon" else "👧 Fille",
                           horizontal=True)
        ethnie = st.selectbox("🌍 Origine ethnique",
                               ["Blanc","Asiatique","Noir","Arabe","Autre"])
    st.markdown("---")
    c3, c4 = st.columns(2)
    with c3:
        jaundice   = st.radio("🟡 Ictère à la naissance ?", [0,1],
                               format_func=lambda x:"❌ Non" if x==0 else "✅ Oui",
                               horizontal=True)
    with c4:
        family_asd = st.radio("👨‍👩‍👧 Antécédents familiaux d'autisme ?", [0,1],
                               format_func=lambda x:"❌ Non" if x==0 else "✅ Oui",
                               horizontal=True)
    st.markdown("</div>", unsafe_allow_html=True)
    bc1, bc2, _ = st.columns([1,2,1])
    with bc1:
        if st.button("⬅️ Retour", use_container_width=True):
            st.session_state.page = 1; st.rerun()
    with bc2:
        if st.button("📝 Commencer le questionnaire", type="primary", use_container_width=True):
            if nom and nom.strip():
                st.session_state.infos_enfant = dict(
                    nom=nom, age=age, genre=genre,
                    ethnie=ethnie, jaundice=jaundice, family_asd=family_asd
                )
                st.session_state.page = 3; st.rerun()
            else:
                st.error("⚠️ Veuillez entrer le nom de l'enfant")

# ══════════════════════════════════════════════════════════════════
# PAGE 3 — QUESTIONNAIRE
# ══════════════════════════════════════════════════════════════════
elif st.session_state.page == 3:
    inf = st.session_state.infos_enfant
    st.markdown(f"""
    <div class="card fade-in">
        <h3>📋 Questionnaire d'évaluation</h3>
        <p>Enfant : <strong>{inf.get('nom','')}</strong> | Âge : {inf.get('age','')} ans</p>
    </div>""", unsafe_allow_html=True)

    QUESTIONS = [
        ("A1",  "😊", "Difficultés à comprendre les expressions faciales ?",
                       "Ne comprend pas quand quelqu'un est triste, content ou fâché"),
        ("A2",  "💬", "Difficultés à maintenir une conversation ?",
                       "Ne sait pas quand parler, quand s'arrêter, change de sujet brusquement"),
        ("A3",  "🔄", "Comportements répétitifs ?",
                       "Se balance, tourne, tape des mains, répète les mêmes mots"),
        ("A4",  "🎯", "Intérêts très spécifiques et intenses ?",
                       "Toujours le même sujet, collectionne des objets inhabituels"),
        ("A5",  "😐", "Semble distant ou sans émotion ?",
                       "Ne réagit pas quand on l'appelle, semble dans sa bulle"),
        ("A6",  "🔊", "Sensibilité aux bruits ou textures ?",
                       "N'aime pas l'aspirateur, les étiquettes, certaines lumières"),
        ("A7",  "🎮", "Préfère jouer seul ?",
                       "Ne cherche pas à faire des amis, joue en solitaire"),
        ("A8",  "📖", "Langage très littéral ?",
                       "Ne comprend pas les blagues, l'ironie ou les métaphores"),
        ("A9",  "👀", "Évite le contact visuel ?",
                       "Ne regarde pas dans les yeux, détourne le regard"),
        ("A10", "📅", "Très attaché à ses routines ?",
                       "Se fâche quand on change ses habitudes ou son environnement"),
    ]

    for idx, (qid, icon, question, detail) in enumerate(QUESTIONS, 1):
        qc1, qc2 = st.columns([1, 5])
        with qc1:
            st.markdown(f"""
            <div style="background:linear-gradient(135deg,#667eea,#764ba2);
                        width:50px;height:50px;border-radius:25px;
                        display:flex;align-items:center;justify-content:center
