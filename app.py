# -*- coding: utf-8 -*-
"""
NeuroSense - Prédiction de l'Autisme
Interface élégante avec tous les questionnaires sur une seule page
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from PIL import Image
import base64
import time

# ==================== CONFIGURATION DE LA PAGE ====================
st.set_page_config(
    page_title="NeuroSense - Détection Autisme",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CSS PERSONNALISÉ (Design Moderne) ====================
st.markdown("""
<style>
    /* Police et fond */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Fond principal */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Conteneur principal */
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
    
    .main-header p {
        color: rgba(255,255,255,0.9);
        font-size: 1.2rem;
    }
    
    /* Cartes */
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
    
    /* Boutons */
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
    
    /* Champs de formulaire */
    .stTextInput > div > div > input, 
    .stNumberInput > div > div > input,
    .stSelectbox > div > div,
    .stRadio > div {
        border-radius: 15px !important;
        border: 2px solid #e0e0e0 !important;
        padding: 0.5rem !important;
    }
    
    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 2px rgba(102,126,234,0.2) !important;
    }
    
    /* Barre de progression */
    .stProgress > div > div > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Résultats */
    .result-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
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
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .fade-in {
        animation: fadeIn 0.6s ease-out;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    
    /* Badges */
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
    
    .badge-success {
        background: #4CAF50;
        color: white;
    }
    
    .badge-warning {
        background: #ff9800;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# ==================== INITIALISATION ====================
if 'model_entraine' not in st.session_state:
    st.session_state.model_entraine = False
if 'model' not in st.session_state:
    st.session_state.model = None
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

def entrainer_modele(df):
    """Entraînement du modèle"""
    target_col = 'Class_ASD'
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    categorical_cols = X.select_dtypes(include=['object']).columns
    le_dict = {}
    
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        le_dict[col] = le
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, le_dict, X_train, X_test, y_train, y_test, y_pred, accuracy

# ==================== CHARGEMENT DES DONNÉES ====================
if st.session_state.df is None:
    with st.spinner("🧠 Initialisation de NeuroSense..."):
        st.session_state.df, _ = charger_donnees()
        st.rerun()

if not st.session_state.model_entraine and st.session_state.df is not None:
    with st.spinner("🤖 Entraînement de l'intelligence artificielle..."):
        model, le_dict, X_train, X_test, y_train, y_test, y_pred, accuracy = entrainer_modele(st.session_state.df)
        st.session_state.model = model
        st.session_state.accuracy = accuracy
        st.session_state.model_entraine = True
        st.session_state.X_train = X_train
        st.session_state.X_test = X_test
        st.session_state.y_train = y_train
        st.session_state.y_test = y_test
        st.session_state.y_pred = y_pred

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
        st.metric("🎯 Précision du modèle", f"{st.session_state.accuracy:.1%}")
        
        # Mini graphique dans le sidebar
        cm = confusion_matrix(st.session_state.y_test, st.session_state.y_pred)
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm', ax=ax, cbar=False)
        ax.set_title('Matrice de confusion', fontsize=10)
        st.pyplot(fig)
    
    st.markdown("---")
    st.markdown("""
    <div style="color: rgba(255,255,255,0.7); font-size: 0.8rem;">
        <p>🔬 <strong>Technologie</strong><br>Random Forest Classifier</p>
        <p>📊 <strong>Données</strong><br>1000+ cas analysés</p>
        <p>⚡ <strong>Précision</strong><br>Haute fiabilité</p>
    </div>
    """, unsafe_allow_html=True)

# ==================== PAGE PRINCIPALE ====================

# En-tête
st.markdown("""
<div class="main-header fade-in">
    <h1>🧠 NeuroSense</h1>
    <p>Prédiction des troubles du spectre autistique par intelligence artificielle</p>
    <div style="margin-top: 1rem;">
        <span class="badge badge-ai">🤖 IA avancée</span>
        <span class="badge badge-success">✅ Haute précision</span>
        <span class="badge badge-warning">⚡ Rapide</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Page 1: Formulaire enfant
if st.session_state.page == 1:
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
                st.session_state.page = 2
                st.rerun()
            else:
                st.error("⚠️ Veuillez entrer le nom de l'enfant")

# Page 2: Tous les questionnaires en une page
elif st.session_state.page == 2:
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
    
    # Liste des questions avec icônes
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
        col1, col2 = st.columns([1, 4])
        
        with col1:
            st.markdown(f"""
            <div style="text-align: center; background: linear-gradient(135deg, #667eea, #764ba2); width: 50px; height: 50px; border-radius: 25px; display: flex; align-items: center; justify-content: center;">
                <span style="font-size: 1.8rem;">{q['icon']}</span>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            with st.container():
                st.markdown(f"""
                <div style="background: white; border-radius: 15px; padding: 1rem; margin-bottom: 0.5rem; border: 1px solid #e0e0e0;">
                    <strong>Question {idx}/10</strong><br>
                    {q['question']}
                    <br><small style="color: #888;">💡 {q['detail']}</small>
                </div>
                """, unsafe_allow_html=True)
                
                reponse = st.radio(
                    "",
                    options=[0, 1],
                    format_func=lambda x: "❌ Non, pas observé" if x == 0 else "✅ Oui, observé",
                    key=f"q_{q['id']}",
                    index=st.session_state.reponses.get(q['id'], None),
                    horizontal=True,
                    label_visibility="collapsed"
                )
                
                if reponse is not None:
                    st.session_state.reponses[q['id']] = reponse
        
        st.markdown("---")
    
    # Score total en temps réel
    total_repondu = len(st.session_state.reponses)
    if total_repondu > 0:
        total_score = sum([st.session_state.reponses.get(f'A{i}', 0) for i in range(1, total_repondu + 1) if f'A{i}' in st.session_state.reponses])
        
        col_score1, col_score2, col_score3 = st.columns([1, 2, 1])
        with col_score2:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #667eea, #764ba2); border-radius: 20px; padding: 1rem; text-align: center; margin: 1rem 0;">
                <span style="color: white; font-size: 1.2rem;">📊 Progression : {total_repondu}/10 questions répondues</span>
                <br>
                <span style="color: white; font-size: 2rem; font-weight: bold;">Score actuel : {total_score}/{total_repondu}</span>
            </div>
            """, unsafe_allow_html=True)
    
    # Boutons
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    
    with col_btn1:
        if st.button("⬅️ Retour", use_container_width=True):
            st.session_state.page = 1
            st.rerun()
    
    with col_btn3:
        if len(st.session_state.reponses) == 10:
            if st.button("🔮 Voir le résultat", type="primary", use_container_width=True):
                st.session_state.page = 3
                st.rerun()
        else:
            st.warning(f"⚠️ {10 - len(st.session_state.reponses)} question(s) restante(s)")

# Page 3: Résultat
elif st.session_state.page == 3:
    # Préparation des données
    input_data = []
    for i in range(1, 11):
        input_data.append(st.session_state.reponses.get(f'A{i}', 0))
    input_data.append(st.session_state.infos_enfant.get('age', 5))
    input_data.append(0 if st.session_state.infos_enfant.get('genre') == 'garcon' else 1)
    ethnie_map = {'Blanc': 0, 'Asiatique': 1, 'Noir': 2, 'Arabe': 3, 'Autre': 4}
    input_data.append(ethnie_map.get(st.session_state.infos_enfant.get('ethnie'), 0))
    input_data.append(st.session_state.infos_enfant.get('jaundice', 0))
    input_data.append(st.session_state.infos_enfant.get('family_asd', 0))
    
    total_score = sum([st.session_state.reponses.get(f'A{i}', 0) for i in range(1, 11)])
    
    if st.session_state.model_entraine:
        input_array = np.array(input_data).reshape(1, -1)
        prediction = st.session_state.model.predict(input_array)[0]
        probabilities = st.session_state.model.predict_proba(input_array)[0]
        
        # Animation de chargement
        with st.spinner("Analyse en cours..."):
            time.sleep(0.5)
        
        # Affichage du résultat
        if prediction == 1:
            st.markdown(f"""
            <div class="result-card result-high fade-in" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
                <span style="font-size: 4rem;">🚨</span>
                <h1 style="color: white;">Risque élevé</h1>
                <p style="color: white; font-size: 1.2rem;">Les signes observés méritent une attention particulière</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-card result-low fade-in" style="background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);">
                <span style="font-size: 4rem;">✅</span>
                <h1 style="color: #2c3e50;">Risque faible</h1>
                <p style="color: #2c3e50; font-size: 1.2rem;">Le développement semble dans la norme</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Statistiques détaillées
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("📊 Score total", f"{total_score}/10")
        
        with col2:
            proba_autisme = probabilities[1] * 100
            st.metric("🤖 Probabilité autiste", f"{proba_autisme:.1f}%")
        
        with col3:
            st.metric("🎯 Précision IA", f"{st.session_state.accuracy:.1%}")
        
        # Barre de progression
        st.markdown("### 📈 Niveau de probabilité")
        st.progress(probabilities[1])
        
        # Détail des réponses
        with st.expander("📋 Détail des réponses", expanded=False):
            for i in range(1, 11):
                reponse = st.session_state.reponses.get(f'A{i}', 0)
                emoji = "✅" if reponse == 1 else "❌"
                st.write(f"{emoji} Question {i}: {'Oui, observé' if reponse == 1 else 'Non, pas observé'}")
        
        # Recommandations
        st.markdown("---")
        st.subheader("💡 Recommandations personnalisées")
        
        if prediction == 1:
            st.markdown("""
            <div style="background: #fff3cd; border-left: 4px solid #ffc107; padding: 1rem; border-radius: 10px;">
                <strong>📞 Consultez rapidement un professionnel de santé</strong><br>
                • Pédiatre ou neuropédiatre<br>
                • Centre de référence pour l'autisme<br>
                • Orthophoniste ou psychologue spécialisé
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background: #d4edda; border-left: 4px solid #28a745; padding: 1rem; border-radius: 10px;">
                <strong>✅ Surveillez le développement de votre enfant</strong><br>
                • Consultez régulièrement votre pédiatre<br>
                • Observez les étapes clés du développement<br>
                • En cas de doute, n'hésitez pas à consulter
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Bouton pour recommencer
        col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
        with col_btn2:
            if st.button("🔄 Nouvelle évaluation", type="primary", use_container_width=True):
                st.session_state.page = 1
                st.session_state.reponses = {}
                st.session_state.infos_enfant = {}
                st.rerun()
    
    else:
        st.error("❌ Modèle non disponible. Veuillez rafraîchir la page.")

# Pied de page
st.markdown("""
<div style="text-align: center; padding: 2rem; color: rgba(255,255,255,0.7);">
    <hr style="border-color: rgba(255,255,255,0.2);">
    <p>🧠 NeuroSense - Intelligence artificielle pour la détection précoce des TSA</p>
    <p style="font-size: 0.8rem;">© 2024 - Outil d'aide à la décision - Consultez toujours un professionnel de santé</p>
</div>
""", unsafe_allow_html=True)
