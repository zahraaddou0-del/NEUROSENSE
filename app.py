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

# ── Imblearn (avec fallback si erreur) ────────────────────────────
try:
    from imblearn.over_sampling  import RandomOverSampler
    IMBLEARN_OK = True
except Exception as e:
    IMBLEARN_OK = False
    print(f"⚠️ imbalanced-learn non disponible: {e}")
    # Alternative manuelle simple pour l'oversampling
    class SimpleRandomOverSampler:
        def __init__(self, random_state=42):
            self.random_state = random_state
        def fit_resample(self, X, y):
            np.random.seed(self.random_state)
            # Trouver la classe minoritaire
            unique, counts = np.unique(y, return_counts=True)
            maj_class = unique[np.argmax(counts)]
            min_class = unique[np.argmin(counts)]
            
            X_maj = X[y == maj_class]
            X_min = X[y == min_class]
            y_maj = y[y == maj_class]
            y_min = y[y == min_class]
            
            # Oversampler la classe minoritaire
            n_to_sample = len(X_maj) - len(X_min)
            if n_to_sample > 0:
                indices = np.random.choice(len(X_min), n_to_sample, replace=True)
                X_min_oversampled = X_min[indices]
                y_min_oversampled = y_min[indices]
                
                X_resampled = np.vstack([X_maj, X_min, X_min_oversampled])
                y_resampled = np.hstack([y_maj, y_min, y_min_oversampled])
            else:
                X_resampled = X
                y_resampled = y
            
            return X_resampled, y_resampled

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
def build_ann(input_dim: int) -> "Sequential":
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


def build_cnn(input_dim: int) -> "Sequential":
    model = Sequential([
        Conv1D(64, kernel_size=3, activation="relu",
               input_shape=(input_dim, 1), padding="same"),
        MaxPooling1D(pool_size=2),
        Conv1D(32, kernel_size=3, activation="relu", padding="same"),
        GlobalAveragePooling1D(),
        Dense(64, activation="relu"),
        Dropout(0.3),
        Dense(1,  activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def entrainer_deep(X_tr, X_te, y_tr, y_te, kind="ANN"):
    """Entraîne ANN ou CNN et retourne résultats + historique."""
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

    # Oversampling (nagatejakachapuram) - with fallback
    if IMBLEARN_OK:
        ros = RandomOverSampler(random_state=42)
        X, y = ros.fit_resample(X, y)
    else:
        ros = SimpleRandomOverSampler(random_state=42)
        X, y = ros.fit_resample(X, y)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    resultats = {}
    progress  = st.progress(0, text="🤖 Entraînement des modèles…")
    n_models  = len(modeles_classiques()) + (2 if TF_OK else 0)
    step = 0

    # ── Modèles classiques ────────────────────────────────────────
    for nom, clf in modeles_classiques().items():
        clf.fit(X_tr, y_tr)
        yp    = clf.predict(X_te)
        yprob = clf.predict_proba(X_te)[:, 1] if hasattr(clf, "predict_proba") else None
        acc   = accuracy_score(y_te, yp)
        roc   = roc_auc_score(y_te, yprob) if yprob is not None else acc
        resultats[nom] = dict(modele=clf, accuracy=acc, auc=roc,
                               y_pred=yp, y_proba=yprob, history=None)
        step += 1
        progress.progress(step / n_models, text=f"✅ {nom}")

    # ── ANN ───────────────────────────────────────────────────────
    if TF_OK:
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

    # ── Meilleur modèle (ROC-AUC) ─────────────────────────────────
    best_name = max(resultats, key=lambda k: resultats[k]["auc"])
    best      = resultats[best_name]

    col_names = list(df.drop(columns=[target]).columns)

    return (best["modele"], best_name, resultats, le_dict, scaler,
            X_tr, X_te, y_tr, y_te, best["y_pred"], best["accuracy"], col_names)


# ══════════════════════════════════════════════════════════════════
# LEARNING CURVE  (claredang — signature)
# ══════════════════════════════════════════════════════════════════
def plot_learning_curve(estimator, X, y, nom):
    """Reproduit les learning curves du repo claredang."""
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


# ══════════════════════════════════════════════════════════════════
# CHARGEMENT + ENTRAÎNEMENT (une seule fois)
# ══════════════════════════════════════════════════════════════════
if st.session_state.df_train is None:
    with st.spinner("📂 Chargement des données…"):
        df_raw, _ = charger_donnees()
        st.session_state.df_train = df_raw
    st.rerun()

if not st.session_state.model_entraine and st.session_state.df_train is not None:
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

# ... (le reste du code reste identique - la partie interface utilisateur ne change pas)
