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
from sklearn.utils           import resample   # pour oversampling sans imblearn
from xgboost                 import XGBClassifier

# ── Fonction d'oversampling manuel (remplace imblearn) ────────────
def manual_oversample(X, y, random_state=42):
    """Équilibrage des classes par sur-échantillonnage (sans imblearn)."""
    np.random.seed(random_state)
    classes = np.unique(y)
    max_count = max(np.bincount(y.astype(int)))
    X_res, y_res = [], []
    for cls in classes:
        X_cls = X[y == cls]
        y_cls = y[y == cls]
        if len(X_cls) < max_count:
            X_cls, y_cls = resample(X_cls, y_cls,
                                    replace=True,
                                    n_samples=max_count,
                                    random_state=random_state)
        X_res.append(X_cls)
        y_res.append(y_cls)
    return np.vstack(X_res), np.hstack(y_res)

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
# CSS (identique à l'original)
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

    # age_group
    if "age" in df.columns:
        df["age"] = pd.to_numeric(df["age"], errors="coerce").fillna(5)
        df["age_group"] = pd.cut(df["age"],
                                  bins  =[0, 4, 12, 18, 40, 200],
                                  labels=["Toddler","Kid","Teenager","Young","Senior"])
    return df


def pretraiter(df: pd.DataFrame, is_train=True, le_dict=None, scaler=None):
    """Encodage + normalisation."""
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
# PIPELINE D'ENTRAÎNEMENT COMPLET (corrigé sans imblearn)
# ══════════════════════════════════════════════════════════════════
def pipeline_complet(df_raw: pd.DataFrame):
    """
    1. Feature Engineering
    2. Prétraitement
    3. Oversampling (manuel)
    4. Entraînement de TOUS les modèles (classiques + ANN + CNN)
    5. Sélection automatique du meilleur (ROC-AUC)
    """
    target = "Class_ASD"

    df = ingenierie_features(df_raw)
    df, le_dict, scaler = pretraiter(df, is_train=True)

    X = df.drop(columns=[target]).values
    y = df[target].values

    # Oversampling manuel (remplace imblearn)
    X, y = manual_oversample(X, y, random_state=42)

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

# ══════════════════════════════════════════════════════════════════
# SIDEBAR (inchangé)
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
        st.metric("🏆 Meilleur modèle", bn.split(" ", 1)[-1])
        st.metric("🎯 Accuracy",        f"{st.session_state.accuracy:.1%}")
        st.metric("📊 ROC-AUC",         f"{ar[bn]['auc']:.3f}")
        st.metric("🧠 Deep Learning",   "✅ ANN + CNN" if TF_OK else "❌ TF non installé")

        st.markdown("---")
        st.subheader("📊 Classement")
        for i, (nom, res) in enumerate(
            sorted(ar.items(), key=lambda x: x[1]["auc"], reverse=True), 1
        ):
            medal = "🥇" if i == 1 else ("🥈" if i == 2 else ("🥉" if i == 3 else "  "))
            st.caption(f"{medal} {nom.split(' ',1)[-1]}: **{res['auc']:.3f}**")

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
# PAGE 1 — RÔLE (identique à l'original)
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
# PAGE 2 — INFOS ENFANT (identique)
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
# PAGE 3 — QUESTIONNAIRE (identique, avec les 10 questions)
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
                        display:flex;align-items:center;justify-content:center;">
                <span style="font-size:1.8rem;">{icon}</span>
            </div>""", unsafe_allow_html=True)
        with qc2:
            st.markdown(f"**Question {idx}/10** — {question}")
            st.caption(f"💡 {detail}")
            rep = st.radio("", [0,1],
                           format_func=lambda x:"❌ Non" if x==0 else "✅ Oui",
                           key=f"q_{qid}",
                           index=st.session_state.reponses.get(qid, None),
                           horizontal=True, label_visibility="collapsed")
            if rep is not None:
                st.session_state.reponses[qid] = rep
        st.markdown("---")

    total_rep = len(st.session_state.reponses)
    if total_rep > 0:
        score_tmp = sum(st.session_state.reponses.values())
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#667eea,#764ba2);
                    border-radius:20px;padding:1rem;text-align:center;margin:1rem 0;">
            <span style="color:white;font-size:1.2rem;">
                📊 Progression : {total_rep}/10</span><br>
            <span style="color:white;font-size:2rem;font-weight:bold;">
                Score actuel : {score_tmp}/{total_rep}</span>
        </div>""", unsafe_allow_html=True)

    bc1, _, bc3 = st.columns([1,2,1])
    with bc1:
        if st.button("⬅️ Retour", use_container_width=True):
            st.session_state.page = 2; st.rerun()
    with bc3:
        if total_rep == 10:
            if st.button("🔮 Voir le résultat", type="primary", use_container_width=True):
                st.session_state.page = 4; st.rerun()
        else:
            st.warning(f"⚠️ {10-total_rep} question(s) restante(s)")

# ══════════════════════════════════════════════════════════════════
# PAGE 4 — RÉSULTATS + GRAPHIQUES COMPLETS (identique)
# ══════════════════════════════════════════════════════════════════
elif st.session_state.page == 4:
    inf         = st.session_state.infos_enfant
    total_score = sum(st.session_state.reponses.values())
    ar          = st.session_state.all_results
    bn          = st.session_state.best_name

    if not st.session_state.model_entraine:
        st.error("❌ Modèle non disponible. Rafraîchissez la page.")
        st.stop()

    # ── Préparer entrée patient ────────────────────────────────────
    ethnie_map = {"Blanc":0,"Asiatique":1,"Noir":2,"Arabe":3,"Autre":4}
    input_raw  = (
        [st.session_state.reponses.get(f"A{i}", 0) for i in range(1, 11)] +
        [inf.get("age", 5),
         0 if inf.get("genre")=="garcon" else 1,
         ethnie_map.get(inf.get("ethnie"), 0),
         inf.get("jaundice", 0),
         inf.get("family_asd", 0)]
    )
    # sum_score + age_group (encodé 0-4) manquants → on les ajoute
    sum_sc    = int(sum(input_raw[:10]))
    age_val   = inf.get("age", 5)
    age_grp   = (0 if age_val<=4 else 1 if age_val<=12 else
                 2 if age_val<=18 else 3 if age_val<=40 else 4)
    contry    = 0  # par défaut
    input_ext = input_raw + [sum_sc, age_grp]
    # aligner sur n_features du scaler
    n_feat    = st.session_state.scaler.n_features_in_
    vec       = np.zeros(n_feat)
    for i, v in enumerate(input_ext[:n_feat]):
        vec[i] = v
    vec_scaled = st.session_state.scaler.transform(vec.reshape(1, -1))

    with st.spinner("🧠 Analyse par tous les modèles…"):
        time.sleep(0.4)

    # ── Vote de tous les modèles ───────────────────────────────────
    st.subheader("🗳️ Vote de tous les modèles")
    votes_asd, model_preds = 0, {}
    cols3 = st.columns(3)
    for idx, (nom, res) in enumerate(
        sorted(ar.items(), key=lambda x: x[1]["auc"], reverse=True)
    ):
        clf = res["modele"]
        try:
            if nom in ("🧠 ANN", "📡 CNN"):
                x_in = vec_scaled.reshape(-1, n_feat, 1) if nom=="📡 CNN" else vec_scaled
                prob = float(clf.predict(x_in, verbose=0).ravel()[0])
                pred = int(prob > 0.5)
            else:
                pred = int(clf.predict(vec_scaled)[0])
                prob = float(clf.predict_proba(vec_scaled)[0][1]) \
                       if hasattr(clf,"predict_proba") else float(pred)
        except Exception:
            pred, prob = 0, 0.0
        if pred == 1: votes_asd += 1
        model_preds[nom] = (pred, prob)

        color  = "#ffb3b3" if pred==1 else "#b3f0c8"
        result = "⚠️ Risque" if pred==1 else "✅ Normal"
        crown  = "🏆 " if nom==bn else ""
        with cols3[idx % 3]:
            st.markdown(f"""
            <div style="background:{color};border-radius:12px;
                        padding:.8rem;margin-bottom:.8rem;text-align:center;">
                <strong>{crown}{nom}</strong><br>
                <span style="font-size:1.1rem;">{result}</span><br>
                Prob: {prob:.1%} | AUC: {res['auc']:.3f}
            </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#667eea,#764ba2);
                border-radius:20px;padding:1.2rem;text-align:center;margin:1rem 0;">
        <h3 style="color:white;">🗳️ {votes_asd}/{len(ar)} modèles détectent un risque</h3>
    </div>""", unsafe_allow_html=True)
    st.markdown("---")

    # ── Résultat du MEILLEUR modèle ────────────────────────────────
    best_pred, best_prob = model_preds.get(bn, (0, 0.0))
    st.markdown(f'<div class="winner-box">🏆 Décision finale — {bn}</div>',
                unsafe_allow_html=True)

    if best_pred == 1:
        st.markdown("""
        <div class="result-card fade-in"
             style="background:linear-gradient(135deg,#f093fb,#f5576c);">
            <span style="font-size:4rem;">🚨</span>
            <h1 style="color:white;">Risque élevé détecté</h1>
            <p style="color:white;font-size:1.2rem;">
                Une évaluation clinique approfondie est recommandée</p>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="result-card fade-in"
             style="background:linear-gradient(135deg,#84fab0,#8fd3f4);">
            <span style="font-size:4rem;">✅</span>
            <h1 style="color:#2c3e50;">Risque faible</h1>
            <p style="color:#2c3e50;font-size:1.2rem;">
                Le développement semble dans la norme</p>
        </div>""", unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("📊 Score total",    f"{total_score}/10")
    c2.metric("🤖 Probabilité",    f"{best_prob:.1%}")
    c3.metric("🎯 Accuracy",       f"{st.session_state.accuracy:.1%}")
    c4.metric("📊 ROC-AUC",        f"{ar[bn]['auc']:.3f}")
    st.progress(best_prob)
    st.markdown("---")

    # ═══════════════════════════════════════════════════════════════
    # GRAPHIQUES COMPLETS (inchangés)
    # ═══════════════════════════════════════════════════════════════
    st.subheader("📊 Analyse complète — Tous les modèles")

    # 1. Comparaison AUC
    st.markdown("#### 🏆 Classement ROC-AUC de tous les modèles")
    noms_sorted = sorted(ar.keys(), key=lambda k: ar[k]["auc"])
    aucs_s      = [ar[n]["auc"] for n in noms_sorted]
    colors_s    = ["gold" if n==bn else "#667eea" for n in noms_sorted]

    fig_cmp, ax_cmp = plt.subplots(figsize=(10, 5))
    bars = ax_cmp.barh(noms_sorted, aucs_s, color=colors_s)
    for bar, val in zip(bars, aucs_s):
        ax_cmp.text(val+.003, bar.get_y()+bar.get_height()/2,
                    f"{val:.3f}", va="center", fontsize=9)
    ax_cmp.axvline(.8, color="red", ls="--", alpha=.5, label="Seuil 0.80")
    ax_cmp.set_xlim(0, 1.08)
    ax_cmp.set_xlabel("ROC-AUC")
    ax_cmp.set_title("Comparaison de tous les modèles (🥇 = sélectionné)")
    ax_cmp.legend(); ax_cmp.grid(True, alpha=.3)
    st.pyplot(fig_cmp)
    st.markdown("---")

    # 2. Matrice confusion + rapport
    cg1, cg2 = st.columns(2)
    with cg1:
        st.markdown(f"#### 📊 Matrice de confusion — {bn}")
        cm = confusion_matrix(st.session_state.y_test, st.session_state.y_pred)
        fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm,
                    xticklabels=["Non-ASD","ASD"],
                    yticklabels=["Non-ASD","ASD"])
        ax_cm.set_xlabel("Prédiction"); ax_cm.set_ylabel("Réalité")
        st.pyplot(fig_cm)
    with cg2:
        st.markdown("#### 📈 Rapport de classification")
        rdf = pd.DataFrame(
            classification_report(st.session_state.y_test,
                                   st.session_state.y_pred,
                                   output_dict=True)
        ).transpose().round(3)
        st.dataframe(rdf, use_container_width=True)
    st.markdown("---")

    # 3. Courbes ROC de tous les modèles
    st.markdown("#### 📉 Courbes ROC — Tous les modèles")
    fig_roc, ax_roc = plt.subplots(figsize=(9, 6))
    palette = plt.cm.tab10(np.linspace(0, 1, len(ar)))
    for (nom, res), col in zip(
        sorted(ar.items(), key=lambda x: x[1]["auc"], reverse=True), palette
    ):
        if res["y_proba"] is not None:
            fpr_r, tpr_r, _ = roc_curve(st.session_state.y_test, res["y_proba"])
            lw = 3 if nom==bn else 1.5
            ls = "-" if nom==bn else "--"
            ax_roc.plot(fpr_r, tpr_r, lw=lw, ls=ls, color=col,
                        label=f"{nom} ({res['auc']:.3f})")
    ax_roc.plot([0,1],[0,1],"k:",lw=1,label="Aléatoire (0.500)")
    ax_roc.set_xlabel("Taux faux positifs"); ax_roc.set_ylabel("Taux vrais positifs")
    ax_roc.set_title("Courbes ROC — Comparaison complète")
    ax_roc.legend(fontsize=8, loc="lower right"); ax_roc.grid(True, alpha=.3)
    st.pyplot(fig_roc)
    st.markdown("---")

    # 4. Courbe d'apprentissage
    st.markdown(f"#### 📈 Courbe d'apprentissage — {bn}")
    best_clf = st.session_state.model
    if not hasattr(best_clf, "predict_proba") or bn in ("🧠 ANN","📡 CNN"):
        st.info("ℹ️ Learning Curve disponible uniquement pour les modèles scikit-learn.")
    else:
        try:
            X_lc = np.vstack([st.session_state.X_train, st.session_state.X_test])
            y_lc = np.concatenate([st.session_state.y_train, st.session_state.y_test])
            fig_lc = plot_learning_curve(best_clf, X_lc, y_lc, bn)
            st.pyplot(fig_lc)
        except Exception as e:
            st.warning(f"Learning curve non disponible : {e}")
    st.markdown("---")

    # 5. Deep Learning — historique
    if TF_OK:
        dl_names = [k for k in ("🧠 ANN","📡 CNN") if k in ar and ar[k]["history"]]
        if dl_names:
            st.markdown("#### 🧠 Historique d'entraînement — Deep Learning")
            dcols = st.columns(len(dl_names))
            for dcol, dl_name in zip(dcols, dl_names):
                hist = ar[dl_name]["history"].history
                fig_h, ax_h = plt.subplots(figsize=(5, 4))
                ax_h.plot(hist["accuracy"],     label="Train",      color="#667eea")
                ax_h.plot(hist["val_accuracy"], label="Validation", color="#f5576c")
                ax_h.set_title(f"{dl_name} — Accuracy")
                ax_h.set_xlabel("Époque"); ax_h.set_ylabel("Accuracy")
                ax_h.legend(); ax_h.grid(True, alpha=.3)
                dcol.pyplot(fig_h)
            st.markdown("---")

    # 6. Importance des caractéristiques
    st.markdown(f"#### 🎯 Importance des caractéristiques — {bn}")
    feat_names = getattr(st.session_state, "col_names",
                         [f"f{i}" for i in range(st.session_state.X_train.shape[1])])
    if hasattr(best_clf, "feature_importances_"):
        imp = best_clf.feature_importances_
        idx = np.argsort(imp)[::-1][:12]
        fig_fi, ax_fi = plt.subplots(figsize=(9, 5))
        ax_fi.barh([feat_names[i] for i in idx[::-1]],
                   imp[idx[::-1]], color="#667eea")
        ax_fi.set_xlabel("Importance")
        ax_fi.set_title("Top caractéristiques (tree-based)")
        st.pyplot(fig_fi)
    elif hasattr(best_clf, "coef_"):
        coef = np.abs(best_clf.coef_[0])
        idx  = np.argsort(coef)[::-1][:12]
        fig_fi, ax_fi = plt.subplots(figsize=(9, 5))
        ax_fi.barh([feat_names[i] for i in idx[::-1]],
                   coef[idx[::-1]], color="#764ba2")
        ax_fi.set_xlabel("|Coefficient|")
        ax_fi.set_title("Importance des coefficients (Logistic Regression)")
        st.pyplot(fig_fi)
    else:
        st.info("ℹ️ Importance non disponible pour ce modèle (ANN/CNN).")
    st.markdown("---")

    # ── Recommandations ────────────────────────────────────────────
    st.subheader("💡 Recommandations personnalisées")
    role = st.session_state.role
    if best_pred == 1:
        msg = (
            "• Consultez rapidement un pédiatre ou neuropédiatre<br>"
            "• Contactez un centre de référence pour l'autisme<br>"
            "• Notez les comportements observés pour le prochain rendez-vous"
        ) if role == "parent" else (
            "• Réalisez une évaluation clinique approfondie (ADOS, CARS, M-CHAT)<br>"
            "• Orientez vers un centre spécialisé si nécessaire<br>"
            "• Prescrivez des examens complémentaires si indiqués"
        )
        st.markdown(f"""
        <div style="background:#fff3cd;border-left:4px solid #ffc107;
                    padding:1rem;border-radius:10px;">
            <strong>⚠️ Recommandations :</strong><br>{msg}</div>
        """, unsafe_allow_html=True)
    else:
        msg = (
            "• Continuez à surveiller le développement de votre enfant<br>"
            "• Consultez régulièrement votre pédiatre"
        ) if role == "parent" else (
            "• Rassurez les parents, le développement semble dans la norme<br>"
            "• Continuez le suivi régulier"
        )
        st.markdown(f"""
        <div style="background:#d4edda;border-left:4px solid #28a745;
                    padding:1rem;border-radius:10px;">
            <strong>✅ Recommandations :</strong><br>{msg}</div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    _, bc2, _ = st.columns([1,2,1])
    with bc2:
        if st.button("🔄 Nouvelle évaluation", type="primary", use_container_width=True):
            for k in ["page","reponses","infos_enfant","role"]:
                st.session_state[k] = _DEFAULTS[k]
            st.rerun()

# ══════════════════════════════════════════════════════════════════
# PIED DE PAGE (identique)
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<div style="text-align:center;padding:2rem;color:rgba(255,255,255,.7);">
    <hr style="border-color:rgba(255,255,255,.2);">
    <p>🧠 NeuroSense — Fusion des meilleurs projets GitHub sur la prédiction de l'autisme</p>
    <p style="font-size:.8rem;">
        Sources : claredang · nagatejakachapuram · yashmahes · Shehab-Hegab et al.<br>
        © 2024 — Outil d'aide à la décision — Consultez toujours un professionnel de santé
    </p>
</div>
""", unsafe_allow_html=True)
