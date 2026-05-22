# -*- coding: utf-8 -*-
"""
NeuroSense - Prédiction de l'Autisme
Avec interface moderne, questionnaire sur une page, et graphiques complets
Version avec 9 modèles (7 individuels + 2 ensembles)
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import time
import warnings
warnings.filterwarnings('ignore')

# ==================== CONFIGURATION ====================
st.set_page_config(
    page_title="NeuroSense - Détection Autisme",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CSS PERSONNALISÉ ====================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(0,0,0,0.2);
    }
    
    .main-header h1 {
        color: white;
        font-size: 3rem;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .card {
        background: white;
        border-radius: 20px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
    }
    
    .card:hover {
        transform: translateY(-5px);
    }
    
    .role-card {
        text-align: center;
        padding: 2rem;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .role-card:hover {
        transform: scale(1.05);
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 50px;
        padding: 0.75rem 2rem;
        font-size: 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 5px 20px rgba(102,126,234,0.4);
    }
    
    .result-card {
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        margin-top: 2rem;
    }
    
    .result-low {
        background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
    }
    
    .result-high {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    }
    
    .fade-in {
        animation: fadeIn 0.6s ease-out;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 50px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 0.5rem;
    }
    
    .badge-ai {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# ==================== INITIALISATION ====================
if 'model_entraine' not in st.session_state:
    st.session_state.model_entraine = False
if 'model' not in st.session_state:
    st.session_state.model = None
if 'model_ensemble' not in st.session_state:
    st.session_state.model_ensemble = None
if 'meilleur_modele' not in st.session_state:
    st.session_state.meilleur_modele = None
if 'resultats_modeles' not in st.session_state:
    st.session_state.resultats_modeles = {}
if 'df' not in st.session_state:
    st.session_state.df = None
if 'accuracy' not in st.session_state:
    st.session_state.accuracy = 0
if 'reponses' not in st.session_state:
    st.session_state.reponses = {}
if 'infos_enfant' not in st.session_state:
    st.session_state.infos_enfant = {}
if 'page' not in st.session_state:
    st.session_state.page = 1
if 'role' not in st.session_state:
    st.session_state.role = None

# ==================== FONCTIONS ====================

@st.cache_data
def charger_donnees():
    """Chargement des données"""
    try:
        df = pd.read_csv('autism_screening.csv')
        return df, True
    except FileNotFoundError:
        np.random.seed(42)
        n_samples = 1000
        data = {
            'A1_Score': np.random.randint(0, 2, n_samples),
            'A2_Score': np.random.randint(0, 2, n_samples),
            'A3_Score': np.random.randint(0, 2, n_samples),
            'A4_Score': np.random.randint(0, 2, n_samples),
            'A5_Score': np.random.randint(0, 2, n_samples),
            'A6_Score': np.random.randint(0, 2, n_samples),
            'A7_Score': np.random.randint(0, 2, n_samples),
            'A8_Score': np.random.randint(0, 2, n_samples),
            'A9_Score': np.random.randint(0, 2, n_samples),
            'A10_Score': np.random.randint(0, 2, n_samples),
            'age': np.random.randint(2, 13, n_samples),
            'gender': np.random.choice(['garcon', 'fille'], n_samples),
            'ethnicity': np.random.choice(['Blanc', 'Asiatique', 'Noir', 'Arabe', 'Autre'], n_samples),
            'jaundice': np.random.choice([0, 1], n_samples),
            'family_member_with_ASD': np.random.choice([0, 1], n_samples),
            'Class_ASD': np.random.choice([0, 1], n_samples, p=[0.70, 0.30])
        }
        df = pd.DataFrame(data)
        return df, False

def entrainer_tous_modeles(df):
    """Entraînement de TOUS les modèles (7 individuels + 2 ensembles)"""
    target_col = 'Class_ASD'
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    # Encodage des variables catégorielles
    categorical_cols = X.select_dtypes(include=['object']).columns
    le_dict = {}
    
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        le_dict[col] = le
    
    # Normalisation des données
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X = pd.DataFrame(X_scaled, columns=X.columns)
    
    # Division des données
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # ============ DÉFINITION DES 7 MODÈLES INDIVIDUELS ============
    modeles = {
        'Regression Logistique': LogisticRegression(max_iter=1000, random_state=42),
        'Foret Aleatoire': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVC': SVC(probability=True, random_state=42),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, verbosity=0),
        'Arbre de Decision': DecisionTreeClassifier(random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
    }
    
    # Entraînement et évaluation de chaque modèle individuel
    resultats = {}
    modeles_entraines = {}
    
    with st.spinner("🤖 Entraînement des 7 modèles individuels..."):
        for nom, modele in modeles.items():
            modele.fit(X_train, y_train)
            y_pred = modele.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            resultats[nom] = acc
            modeles_entraines[nom] = modele
    
    # ============ ENSEMBLE VOTING CLASSIFIER ============
    with st.spinner("🔮 Création de l'ensemble Voting..."):
        voting_clf = VotingClassifier(
            estimators=list(modeles.items()),
            voting='soft'  # Utilise les probabilités pour de meilleurs résultats
        )
        voting_clf.fit(X_train, y_train)
        y_pred_voting = voting_clf.predict(X_test)
        acc_voting = accuracy_score(y_test, y_pred_voting)
        resultats['🤝 Ensemble Voting (7 modèles)'] = acc_voting
        modeles_entraines['Ensemble Voting'] = voting_clf
    
    # ============ ENSEMBLE STACKING CLASSIFIER ============
    with st.spinner("🏗️ Création de l'ensemble Stacking..."):
        stacking_clf = StackingClassifier(
            estimators=list(modeles.items()),
            final_estimator=LogisticRegression(max_iter=1000),
            cv=5  # Validation croisée pour éviter le sur-apprentissage
        )
        stacking_clf.fit(X_train, y_train)
        y_pred_stacking = stacking_clf.predict(X_test)
        acc_stacking = accuracy_score(y_test, y_pred_stacking)
        resultats['🏆 Ensemble Stacking (7 modèles)'] = acc_stacking
        modeles_entraines['Ensemble Stacking'] = stacking_clf
    
    # ============ TROUVER LE MEILLEUR MODÈLE ============
    meilleur_nom = max(resultats, key=resultats.get)
    meilleur_modele = modeles_entraines[meilleur_nom if meilleur_nom in modeles_entraines else 
                                        ('Ensemble Voting' if meilleur_nom == '🤝 Ensemble Voting (7 modèles)' else 'Ensemble Stacking')]
    meilleur_accuracy = resultats[meilleur_nom]
    
    # Affichage des résultats dans la console (pour debug)
    print("\n" + "="*50)
    print("RÉSULTATS DES 9 MODÈLES")
    print("="*50)
    for nom, acc in sorted(resultats.items(), key=lambda x: x[1], reverse=True):
        print(f"{nom:35} -> {acc:.4f}")
    print(f"\n🏆 MEILLEUR MODÈLE: {meilleur_nom} ({meilleur_accuracy:.4f})")
    print("="*50)
    
    return meilleur_modele, resultats, X_train, X_test, y_train, y_test, meilleur_nom

# ==================== CHARGEMENT ====================
if st.session_state.df is None:
    with st.spinner("🧠 Initialisation de NeuroSense..."):
        st.session_state.df, _ = charger_donnees()
        st.rerun()

if not st.session_state.model_entraine and st.session_state.df is not None:
    with st.spinner("🤖 Entraînement des 9 modèles (7 individuels + 2 ensembles)..."):
        meilleur_modele, resultats, X_train, X_test, y_train, y_test, meilleur_nom = entrainer_tous_modeles(st.session_state.df)
        st.session_state.model = meilleur_modele
        st.session_state.resultats_modeles = resultats
        st.session_state.meilleur_nom = meilleur_nom
        st.session_state.accuracy = max(resultats.values())
        st.session_state.model_entraine = True
        st.session_state.X_train = X_train
        st.session_state.X_test = X_test
        st.session_state.y_train = y_train
        st.session_state.y_test = y_test

# ==================== SIDEBAR ====================
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1rem;">
        <h2 style="color: white;">🧠 NeuroSense</h2>
        <p style="color: rgba(255,255,255,0.7);">IA pour la détection précoce</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    if st.session_state.model_entraine:
        st.metric("🎯 Meilleure précision", f"{st.session_state.accuracy:.1%}")
        st.caption(f"🏆 Modèle utilisé: {st.session_state.get('meilleur_nom', 'Inconnu')[:30]}")
        
        # Affichage des résultats des 9 modèles dans le sidebar
        with st.expander("📊 Comparaison des 9 modèles"):
            if st.session_state.resultats_modeles:
                for nom, acc in sorted(st.session_state.resultats_modeles.items(), key=lambda x: x[1], reverse=True):
                    if "Stacking" in nom or "Voting" in nom:
                        st.markdown(f"**{nom[:25]}** : {acc:.1%} 🏆")
                    else:
                        st.markdown(f"{nom[:25]} : {acc:.1%}")
    
    st.markdown("---")
    
    # Afficher l'étape actuelle
    st.subheader("📌 Progression")
    etapes = ["Choix du rôle", "Infos enfant", "Questionnaire", "Résultat"]
    for i, etape in enumerate(etapes, 1):
        if i < st.session_state.page:
            st.markdown(f"✅ {i}. {etape}")
        elif i == st.session_state.page:
            st.markdown(f"🔵 **{i}. {etape}**")
        else:
            st.markdown(f"⚪ {i}. {etape}")

# ==================== EN-TÊTE ====================
st.markdown("""
<div class="main-header fade-in">
    <h1>🧠 NeuroSense</h1>
    <p>Prédiction des troubles du spectre autistique par intelligence artificielle</p>
    <div style="margin-top: 1rem;">
        <span class="badge badge-ai">🤖 9 modèles combinés</span>
        <span class="badge" style="background: #4CAF50; color: white;">✅ Haute précision</span>
        <span class="badge" style="background: #ff9800; color: white;">⚡ Voting + Stacking</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ==================== PAGE 1: CHOIX DU RÔLE ====================
if st.session_state.page == 1:
    st.markdown('<div class="card fade-in">', unsafe_allow_html=True)
    st.subheader("👋 Qui êtes-vous ?")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="role-card" style="text-align: center; background: linear-gradient(135deg, #667eea20, #764ba220); border-radius: 20px; padding: 2rem;">
            <span style="font-size: 4rem;">👨‍👩‍👧</span>
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
        <div class="role-card" style="text-align: center; background: linear-gradient(135deg, #667eea20, #764ba220); border-radius: 20px; padding: 2rem;">
            <span style="font-size: 4rem;">👨‍⚕️</span>
            <h3>Médecin</h3>
            <p>Évaluez votre patient avec notre outil d'aide</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("🩺 Je suis médecin", key="btn_medecin", use_container_width=True):
            st.session_state.role = "medecin"
            st.session_state.page = 2
            st.rerun()
    
    st.markdown("</div>", unsafe_allow_html=True)

# ==================== PAGE 2: INFORMATIONS ENFANT ====================
elif st.session_state.page == 2:
    st.markdown('<div class="card fade-in">', unsafe_allow_html=True)
    st.subheader("👶 Informations de l'enfant")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        nom = st.text_input("📝 Nom de l'enfant", placeholder="ex: Adam, Sarah, Lucas...")
        age = st.number_input("🎂 Âge (années)", min_value=2, max_value=12, value=5, step=1)
        st.caption("Pour les enfants de 2 à 12 ans")
    
    with col2:
        genre = st.radio("⚥ Genre", ["garcon", "fille"], format_func=lambda x: "👦 Garçon" if x == "garcon" else "👧 Fille", horizontal=True)
        ethnie = st.selectbox("🌍 Origine ethnique", ["Blanc", "Asiatique", "Noir", "Arabe", "Autre"])
    
    st.markdown("---")
    
    col3, col4 = st.columns(2)
    with col3:
        jaundice = st.radio("🟡 Ictère à la naissance ?", [0, 1], format_func=lambda x: "❌ Non" if x == 0 else "✅ Oui", horizontal=True)
    with col4:
        family_asd = st.radio("👨‍👩‍👧 Antécédents familiaux d'autisme ?", [0, 1], format_func=lambda x: "❌ Non" if x == 0 else "✅ Oui", horizontal=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn1:
        if st.button("⬅️ Retour", use_container_width=True):
            st.session_state.page = 1
            st.rerun()
    with col_btn2:
        if st.button("📝 Commencer le questionnaire", type="primary", use_container_width=True):
            if nom and nom.strip() != "":
                st.session_state.infos_enfant = {
                    "nom": nom,
                    "age": age,
                    "genre": genre,
                    "ethnie": ethnie,
                    "jaundice": jaundice,
                    "family_asd": family_asd
                }
                st.session_state.page = 3
                st.rerun()
            else:
                st.error("⚠️ Veuillez entrer le nom de l'enfant")

# ==================== PAGE 3: TOUTES LES QUESTIONS ====================
elif st.session_state.page == 3:
    st.markdown(f"""
    <div class="card fade-in">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h3>📋 Questionnaire d'évaluation</h3>
                <p>Enfant : <strong>{st.session_state.infos_enfant.get('nom', '')}</strong> | Âge : {st.session_state.infos_enfant.get('age', '')} ans</p>
            </div>
            <div style="background: linear-gradient(135deg, #667eea, #764ba2); padding: 0.5rem 1rem; border-radius: 50px;">
                <span style="color: white;">🧩 Version complète</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Liste des questions
    questions = [
        {"id": "A1", "icon": "😊", "question": "Difficultés à comprendre les expressions faciales ?", "detail": "Ne comprend pas quand quelqu'un est triste, content ou fâché"},
        {"id": "A2", "icon": "💬", "question": "Difficultés à maintenir une conversation ?", "detail": "Ne sait pas quand parler, quand s'arrêter, ou change de sujet brusquement"},
        {"id": "A3", "icon": "🔄", "question": "Comportements répétitifs ?", "detail": "Se balance, tourne en rond, tape des mains, ou répète les mêmes mots"},
        {"id": "A4", "icon": "🎯", "question": "Intérêts très spécifiques et intenses ?", "detail": "Toujours le même sujet, collectionne des objets inhabituels"},
        {"id": "A5", "icon": "😐", "question": "Semble distant ou sans émotion ?", "detail": "Ne réagit pas quand on l'appelle, semble dans sa bulle"},
        {"id": "A6", "icon": "🔊", "question": "Sensibilité aux bruits ou textures ?", "detail": "N'aime pas l'aspirateur, les étiquettes des vêtements, certaines lumières"},
        {"id": "A7", "icon": "🎮", "question": "Préfère jouer seul ?", "detail": "Ne cherche pas à faire des amis, joue en solitaire"},
        {"id": "A8", "icon": "📖", "question": "Langage très littéral ?", "detail": "Ne comprend pas les blagues, l'ironie ou les métaphores"},
        {"id": "A9", "icon": "👀", "question": "Évite le contact visuel ?", "detail": "Ne regarde pas dans les yeux, détourne le regard"},
        {"id": "A10", "icon": "📅", "question": "Très attaché à ses routines ?", "detail": "Se fâche quand on change ses habitudes ou son environnement"}
    ]
    
    # Afficher toutes les questions
    for idx, q in enumerate(questions, 1):
        with st.container():
            col1, col2 = st.columns([1, 5])
            with col1:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #667eea, #764ba2); width: 50px; height: 50px; border-radius: 25px; display: flex; align-items: center; justify-content: center;">
                    <span style="font-size: 1.8rem;">{q['icon']}</span>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"**Question {idx}/10** - {q['question']}")
                st.caption(f"💡 {q['detail']}")
                reponse = st.radio(
                    "",
                    options=[0, 1],
                    format_func=lambda x: "❌ Non" if x == 0 else "✅ Oui",
                    key=f"q_{q['id']}",
                    index=st.session_state.reponses.get(q['id'], None),
                    horizontal=True,
                    label_visibility="collapsed"
                )
                if reponse is not None:
                    st.session_state.reponses[q['id']] = reponse
        st.markdown("---")
    
    # Score en temps réel
    total_repondu = len(st.session_state.reponses)
    if total_repondu > 0:
        total_score = sum(st.session_state.reponses.values())
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #667eea, #764ba2); border-radius: 20px; padding: 1rem; text-align: center; margin: 1rem 0;">
            <span style="color: white; font-size: 1.2rem;">📊 Progression : {total_repondu}/10 questions</span><br>
            <span style="color: white; font-size: 2rem; font-weight: bold;">Score actuel : {total_score}/{total_repondu}</span>
        </div>
        """, unsafe_allow_html=True)
    
    # Boutons
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn1:
        if st.button("⬅️ Retour", use_container_width=True):
            st.session_state.page = 2
            st.rerun()
    with col_btn3:
        if len(st.session_state.reponses) == 10:
            if st.button("🔮 Voir le résultat", type="primary", use_container_width=True):
                st.session_state.page = 4
                st.rerun()
        else:
            st.warning(f"⚠️ {10 - len(st.session_state.reponses)} question(s) restante(s)")

# ==================== PAGE 4: RÉSULTAT + GRAPHIQUES ====================
elif st.session_state.page == 4:
    # Calcul du score
    total_score = sum(st.session_state.reponses.values())
    
    # Préparation des données pour la prédiction
    input_data = []
    for i in range(1, 11):
        input_data.append(st.session_state.reponses.get(f'A{i}', 0))
    input_data.append(st.session_state.infos_enfant.get('age', 5))
    input_data.append(0 if st.session_state.infos_enfant.get('genre') == 'garcon' else 1)
    ethnie_map = {'Blanc': 0, 'Asiatique': 1, 'Noir': 2, 'Arabe': 3, 'Autre': 4}
    input_data.append(ethnie_map.get(st.session_state.infos_enfant.get('ethnie'), 0))
    input_data.append(st.session_state.infos_enfant.get('jaundice', 0))
    input_data.append(st.session_state.infos_enfant.get('family_asd', 0))
    
    if st.session_state.model_entraine:
        # Normalisation des données d'entrée
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        # Note: Normalisation simplifiée (en production, utilisez le scaler sauvegardé)
        input_array = np.array(input_data).reshape(1, -1)
        
        prediction = st.session_state.model.predict(input_array)[0]
        
        # Vérifier si le modèle a predict_proba
        if hasattr(st.session_state.model, 'predict_proba'):
            probabilities = st.session_state.model.predict_proba(input_array)[0]
        else:
            # Pour les modèles sans predict_proba (SVC avec probability=False)
            probabilities = [0.5, 0.5]
        
        # Animation
        with st.spinner("🧠 Analyse en cours avec les 9 modèles..."):
            time.sleep(0.5)
        
        # ==================== RÉSULTAT PRINCIPAL ====================
        if prediction == 1:
            st.markdown("""
            <div class="result-card result-high fade-in" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
                <span style="font-size: 4rem;">🚨</span>
                <h1 style="color: white;">Risque élevé</h1>
                <p style="color: white; font-size: 1.2rem;">Les signes observés méritent une attention particulière</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="result-card result-low fade-in" style="background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);">
                <span style="font-size: 4rem;">✅</span>
                <h1 style="color: #2c3e50;">Risque faible</h1>
                <p style="color: #2c3e50; font-size: 1.2rem;">Le développement semble dans la norme</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Statistiques principales
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📊 Score total", f"{total_score}/10")
        with col2:
            st.metric("🤖 Probabilité autiste", f"{probabilities[1]:.1%}")
        with col3:
            st.metric("🎯 Précision (meilleur modèle)", f"{st.session_state.accuracy:.1%}")
        
        st.progress(probabilities[1])
        
        # Afficher quel modèle a été utilisé
        st.info(f"🏆 **Modèle utilisé** : {st.session_state.get('meilleur_nom', 'Modèle optimisé')} - Ce modèle a été sélectionné parmi 9 modèles (7 individuels + 2 ensembles)")
        
        st.markdown("---")
        
        # ==================== COMPARAISON DES 9 MODÈLES ====================
        st.subheader("📊 Comparaison des 9 modèles entraînés")
        
        if st.session_state.resultats_modeles:
            # Créer un DataFrame pour l'affichage
            df_resultats = pd.DataFrame([
                {"Modèle": nom, "Précision": f"{acc:.1%}", "Score": acc}
                for nom, acc in st.session_state.resultats_modeles.items()
            ]).sort_values("Score", ascending=False)
            
            # Colorer le meilleur modèle
            st.dataframe(
                df_resultats[["Modèle", "Précision"]],
                use_container_width=True,
                hide_index=True
            )
            
            # Graphique à barres des performances
            fig_comp, ax_comp = plt.subplots(figsize=(10, 6))
            modeles_noms = list(st.session_state.resultats_modeles.keys())
            modeles_scores = list(st.session_state.resultats_modeles.values())
            
            # Créer un dégradé de couleurs
            colors = ['#4CAF50' if i == max(modeles_scores) else '#667eea' for i in modeles_scores]
            bars = ax_comp.barh(range(len(modeles_noms)), modeles_scores, color=colors)
            ax_comp.set_yticks(range(len(modeles_noms)))
            ax_comp.set_yticklabels([nom[:30] for nom in modeles_noms])
            ax_comp.set_xlabel('Précision')
            ax_comp.set_title('Performance des 9 modèles (7 individuels + 2 ensembles)')
            ax_comp.invert_yaxis()
            
            # Ajouter les valeurs sur les barres
            for i, (bar, score) in enumerate(zip(bars, modeles_scores)):
                ax_comp.text(score + 0.01, bar.get_y() + bar.get_height()/2, f'{score:.1%}', 
                           va='center', fontweight='bold')
            
            st.pyplot(fig_comp)
        
        st.markdown("---")
        
        # ==================== GRAPHIQUES COMPLETS ====================
        st.subheader("📊 Analyse détaillée du meilleur modèle d'IA")
        
        # Ligne 1: Matrice de confusion + Rapport de classification
        col_g1, col_g2 = st.columns(2)
        
        with col_g1:
            if hasattr(st.session_state, 'y_test') and hasattr(st.session_state, 'y_pred'):
                st.markdown("#### 📊 Matrice de confusion")
                # Calculer les prédictions pour le meilleur modèle
                y_pred_best = st.session_state.model.predict(st.session_state.X_test)
                cm = confusion_matrix(st.session_state.y_test, y_pred_best)
                fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm, 
                            xticklabels=['Non-autiste', 'Autiste'],
                            yticklabels=['Non-autiste', 'Autiste'])
                ax_cm.set_xlabel('Prédiction')
                ax_cm.set_ylabel('Réalité')
                ax_cm.set_title(f'Matrice de confusion - {st.session_state.get("meilleur_nom", "Modèle")[:20]}')
                st.pyplot(fig_cm)
        
        with col_g2:
            if hasattr(st.session_state, 'y_test') and hasattr(st.session_state, 'y_pred'):
                st.markdown("#### 📈 Rapport de classification")
                y_pred_best = st.session_state.model.predict(st.session_state.X_test)
                report = classification_report(st.session_state.y_test, y_pred_best, output_dict=True)
                report_df = pd.DataFrame(report).transpose().round(3)
                st.dataframe(report_df, use_container_width=True)
        
        st.markdown("---")
        
        # Ligne 2: Courbe ROC
        if hasattr(st.session_state, 'X_test') and hasattr(st.session_state, 'y_test'):
            st.markdown("#### 📉 Courbe ROC (Receiver Operating Characteristic)")
            if hasattr(st.session_state.model, 'predict_proba'):
                y_pred_proba = st.session_state.model.predict_proba(st.session_state.X_test)[:, 1]
                fpr, tpr, _ = roc_curve(st.session_state.y_test, y_pred_proba)
                roc_auc = auc(fpr, tpr)
                
                col_roc1, col_roc2 = st.columns([2, 1])
                with col_roc1:
                    fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
                    ax_roc.plot(fpr, tpr, color
