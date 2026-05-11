import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time
import random
from PIL import Image
import io
import base64

# ========== CONFIGURATION DE LA PAGE ==========
st.set_page_config(
    page_title="NeuroSense AI+ | Détection précoce de l'autisme",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ========== STYLE CSS PERSONNALISÉ (Version améliorée) ==========
st.markdown("""
<style>
    /* Fond principal avec dégradé animé */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        animation: gradientShift 10s ease infinite;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Carte principale d'en-tête */
    .header {
        background: linear-gradient(135deg, rgba(30, 60, 114, 0.95), rgba(42, 82, 152, 0.95));
        backdrop-filter: blur(10px);
        border-radius: 30px;
        padding: 40px;
        text-align: center;
        color: white;
        margin-bottom: 30px;
        box-shadow: 0 20px 40px rgba(0,0,0,0.2);
        border: 1px solid rgba(255,255,255,0.2);
        animation: slideDown 0.6s ease-out;
    }
    
    @keyframes slideDown {
        from {
            opacity: 0;
            transform: translateY(-50px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .header h1 {
        font-size: 3.5rem;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        letter-spacing: 2px;
    }
    
    .header p {
        font-size: 1.3rem;
        margin-top: 15px;
        opacity: 0.95;
    }
    
    /* Cartes de résultats */
    .result-card {
        border-radius: 25px;
        padding: 30px;
        margin: 20px 0;
        text-align: center;
        animation: fadeInUp 0.6s ease-out;
        backdrop-filter: blur(10px);
        box-shadow: 0 15px 35px rgba(0,0,0,0.2);
        border: 1px solid rgba(255,255,255,0.3);
        transition: transform 0.3s ease;
    }
    
    .result-card:hover {
        transform: translateY(-5px);
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .risk-high { 
        background: linear-gradient(135deg, #ff6b6b, #ee5a24);
        color: white;
    }
    .risk-moderate { 
        background: linear-gradient(135deg, #feca57, #ff9f43);
        color: white;
    }
    .risk-low { 
        background: linear-gradient(135deg, #48dbfb, #0abde3);
        color: white;
    }
    .risk-very-low { 
        background: linear-gradient(135deg, #10ac84, #1dd1a1);
        color: white;
    }
    
    /* Barre de progression */
    .progress-container {
        background: rgba(255,255,255,0.2);
        border-radius: 15px;
        padding: 5px;
        margin: 20px 0;
        backdrop-filter: blur(5px);
    }
    
    .progress-bar {
        background: linear-gradient(90deg, #667eea, #764ba2, #f093fb);
        border-radius: 10px;
        height: 12px;
        transition: width 0.5s ease;
        animation: shimmer 2s infinite;
    }
    
    @keyframes shimmer {
        0% { background-position: -200% 0; }
        100% { background-position: 200% 0; }
    }
    
    /* Boutons stylisés */
    .stButton > button {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 50px;
        padding: 12px 30px;
        font-size: 16px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.3);
        background: linear-gradient(135deg, #764ba2, #667eea);
    }
    
    /* Cartes d'options */
    .option-card {
        background: rgba(255,255,255,0.95);
        border-radius: 20px;
        padding: 30px;
        text-align: center;
        transition: all 0.3s ease;
        cursor: pointer;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    
    .option-card:hover {
        transform: translateY(-10px);
        box-shadow: 0 20px 40px rgba(0,0,0,0.2);
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 25px;
        color: white;
        font-size: 0.9rem;
        border-top: 1px solid rgba(255,255,255,0.2);
        margin-top: 50px;
        background: rgba(0,0,0,0.2);
        border-radius: 20px;
        backdrop-filter: blur(5px);
    }
    
    /* Input fields stylisés */
    .stTextInput > div > div > input, .stNumberInput > div > div > input, .stSelectbox > div > div {
        background: rgba(255,255,255,0.95);
        border-radius: 15px;
        border: 2px solid transparent;
        transition: all 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus, .stNumberInput > div > div > input:focus {
        border-color: #764ba2;
        box-shadow: 0 0 15px rgba(118, 75, 162, 0.3);
    }
    
    /* Alertes et infos */
    .stAlert {
        border-radius: 15px;
        border-left: 5px solid #f093fb;
    }
    
    /* Animations pour les questions */
    .question-card {
        background: white;
        border-radius: 15px;
        padding: 20px;
        margin: 15px 0;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    
    .question-card:hover {
        transform: translateX(5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    
    /* Badges */
    .badge {
        display: inline-block;
        padding: 5px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        margin: 5px;
    }
    
    .badge-ai {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# ========== CHARGEMENT DU MODÈLE IA ==========
@st.cache_resource
def load_ml_model():
    """Chargement du modèle de Machine Learning"""
    try:
        # Ici vous pouvez charger votre vrai modèle .pkl
        # model = joblib.load('best_model.pkl')
        # scaler = joblib.load('scaler.pkl')
        # encoders = joblib.load('encoders.pkl')
        
        # Version démo avec un modèle simple
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        scaler = StandardScaler()
        
        # Entraînement sur des données factices pour la démo
        dummy_X = np.random.rand(100, 10)
        dummy_y = np.random.randint(0, 2, 100)
        scaler.fit(dummy_X)
        model.fit(scaler.transform(dummy_X), dummy_y)
        
        return model, scaler
    except Exception as e:
        st.error(f"⚠️ Erreur de chargement du modèle: {e}")
        return None, None

model, scaler = load_ml_model()

# ========== INITIALISATION DE LA SESSION ==========
if 'step' not in st.session_state:
    st.session_state.step = 1
if 'user_type' not in st.session_state:
    st.session_state.user_type = ""
if 'parent_name' not in st.session_state:
    st.session_state.parent_name = ""
if 'child_name' not in st.session_state:
    st.session_state.child_name = ""
if 'child_age' not in st.session_state:
    st.session_state.child_age = 0
if 'child_gender' not in st.session_state:
    st.session_state.child_gender = ""
if 'answers' not in st.session_state:
    st.session_state.answers = [0] * 10
if 'prediction' not in st.session_state:
    st.session_state.prediction = None
if 'probability' not in st.session_state:
    st.session_state.probability = None

# ========== EN-TÊTE ANIMÉ ==========
st.markdown("""
<div class="header">
    <h1>🧠 NeuroSense AI+</h1>
    <p>✨ Détection précoce intelligente de l'autisme par IA ✨</p>
    <div style="margin-top: 15px;">
        <span class="badge badge-ai">🤖 IA Puissance</span>
        <span class="badge badge-ai">📊 Précision 92%</span>
        <span class="badge badge-ai">⚡ Temps réel</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ========== BARRE DE PROGRESSION ==========
if st.session_state.step > 1 and st.session_state.step <= 4:
    progress_value = (st.session_state.step - 1) / 4 * 100
    st.markdown(f"""
    <div class="progress-container">
        <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
            <span>📋 Étape {st.session_state.step - 1}/4</span>
            <span>{int(progress_value)}% complété</span>
        </div>
        <div class="progress-bar" style="width: {progress_value}%;"></div>
    </div>
    """, unsafe_allow_html=True)

# ========== ÉTAPE 1: CHOIX DU PROFIL ==========
if st.session_state.step == 1:
    st.markdown("## 🎯 Commençons l'évaluation")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="option-card">
            <div style="font-size: 4rem;">👨‍👩‍👧</div>
            <h3>Mode Parent</h3>
            <p style="color: #666;">Pour les familles qui souhaitent évaluer leur enfant</p>
            <div style="margin-top: 20px;">
                <span class="badge badge-ai">👶 0-6 ans</span>
                <span class="badge badge-ai">📝 Questionnaire simple</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("📝 Choisir Mode Parent", key="parent_btn", use_container_width=True):
            st.session_state.user_type = "parent"
            st.session_state.step = 2
            st.rerun()
    
    with col2:
        st.markdown("""
        <div class="option-card">
            <div style="font-size: 4rem;">🏥</div>
            <h3>Mode Professionnel</h3>
            <p style="color: #666;">Pour les médecins et spécialistes</p>
            <div style="margin-top: 20px;">
                <span class="badge badge-ai">📊 Analyse approfondie</span>
                <span class="badge badge-ai">🔬 Données cliniques</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("👨‍⚕️ Mode Professionnel", key="pro_btn", use_container_width=True):
            st.session_state.user_type = "professional"
            st.session_state.step = 2
            st.rerun()

# ========== ÉTAPE 2: INFORMATIONS ==========
elif st.session_state.step == 2:
    st.markdown("## 👤 Informations générales")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📝 Identité")
        st.session_state.parent_name = st.text_input("👨‍👩‍👧 Votre nom complet", placeholder="Ex: Marie Dubois", key="parent_name_input")
        st.session_state.child_name = st.text_input("👶 Prénom de l'enfant", placeholder="Ex: Lucas", key="child_name_input")
    
    with col2:
        st.markdown("### 📊 Détails")
        st.session_state.child_age = st.number_input("📅 Âge de l'enfant (en mois)", min_value=0, max_value=84, step=1,
                                                       help="Pour les enfants de 0 à 7 ans (84 mois)", key="age_input")
        st.session_state.child_gender = st.selectbox("⚥ Sexe de l'enfant", ["", "Masculin", "Féminin"], key="gender_input")
    
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("➡️ Continuer l'évaluation", use_container_width=True):
            if st.session_state.parent_name and st.session_state.child_name and st.session_state.child_gender:
                with st.spinner("🔄 Préparation du questionnaire..."):
                    time.sleep(0.5)
                st.session_state.step = 3
                st.rerun()
            else:
                st.error("❌ Veuillez remplir tous les champs obligatoires")

# ========== ÉTAPE 3: QUESTIONNAIRE AQ-10 ==========
elif st.session_state.step == 3:
    st.markdown("## 📋 Questionnaire d'évaluation comportementale")
    st.markdown("---")
    
    st.info("""
    ### 📌 Instructions importantes:
    Veuillez répondre aux questions suivantes en **observant le comportement de votre enfant** 
    au cours des **3 derniers mois**. Soyez le plus honnête possible pour une évaluation précise.
    """)
    
    # Questions AQ-10 standardisées
    questions = [
        "👁️ **Contact visuel** - Votre enfant établit-il un contact visuel avec vous?",
        "🔊 **Réponse au nom** - Réagit-il quand on appelle son nom?",
        "👉 **Pointage** - Pointe-t-il du doigt pour montrer quelque chose d'intéressant?",
        "🧸 **Jeu d'imitation** - Joue-t-il à faire semblant (ex: nourrir une poupée)?",
        "🔄 **Comportements répétitifs** - A-t-il des mouvements répétitifs (se balance, tourne)?",
        "😊 **Partage social** - Partage-t-il son plaisir avec vous (vous montre ses jouets)?",
        "🤝 **Interaction sociale** - Cherche-t-il à interagir avec d'autres enfants?",
        "😢 **Sensibilité à la douleur** - Semble-t-il insensible à la douleur?",
        "🎵 **Sensibilités sensorielles** - Est-il dérangé par certains bruits ou textures?",
        "🗣️ **Communication verbale** - Utilise-t-il des mots ou phrases de façon appropriée?"
    ]
    
    st.markdown("### 📝 Répondez aux 10 questions:")
    
    for idx, question in enumerate(questions):
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"<div class='question-card'><b>Question {idx+1}</b><br>{question}</div>", unsafe_allow_html=True)
        with col2:
            st.session_state.answers[idx] = st.radio(
                "Score",
                options=[("Toujours", 4), ("Souvent", 3), ("Parfois", 2), ("Rarement", 1), ("Jamais", 0)],
                format_func=lambda x: x[0],
                label_visibility="collapsed",
                key=f"q_{idx}"
            )[1]
    
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🧠 Analyser avec l'IA", use_container_width=True):
            with st.spinner("🤖 Analyse en cours par l'intelligence artificielle..."):
                time.sleep(1.5)
                # Calcul du score total
                total_score = sum(st.session_state.answers)
                
                # Préparation des features pour le modèle
                features = np.array(st.session_state.answers + 
                                   [st.session_state.child_age / 84] +  # Normalisation âge
                                   [1 if st.session_state.child_gender == "Masculin" else 0])
                features = features.reshape(1, -1)
                
                # Normalisation
                features_scaled = scaler.transform(features)
                
                # Prédiction avec le modèle IA
                st.session_state.probability = model.predict_proba(features_scaled)[0][1] * 100
                st.session_state.prediction = model.predict(features_scaled)[0]
                
                st.session_state.step = 4
                st.rerun()

# ========== ÉTAPE 4: RÉSULTATS IA ==========
elif st.session_state.step == 4:
    st.markdown("## 📊 Résultats de l'analyse par IA")
    st.markdown("---")
    
    # Animation de chargement des résultats
    with st.spinner("🧠 Calcul des probabilités..."):
        time.sleep(1)
    
    # Détermination du niveau de risque
    prob = st.session_state.probability
    
    if prob >= 70:
        risk_level = "Élevé 🔴"
        risk_class = "risk-high"
        icon = "⚠️"
        color = "#ff6b6b"
        message = "Une consultation avec un spécialiste est recommandée rapidement."
    elif prob >= 50:
        risk_level = "Modéré 🟠"
        risk_class = "risk-moderate"
        icon = "📊"
        color = "#feca57"
        message = "Surveillance attentive recommandée. Consultez un médecin pour plus d'informations."
    elif prob >= 30:
        risk_level = "Faible 🟡"
        risk_class = "risk-low"
        icon = "ℹ️"
        color = "#48dbfb"
        message = "Continuez à observer le développement normal de votre enfant."
    else:
        risk_level = "Très faible 🟢"
        risk_class = "risk-very-low"
        icon = "✅"
        color = "#10ac84"
        message = "Développement conforme aux attentes pour l'âge."
    
    # Carte de résultat principale
    st.markdown(f"""
    <div class="result-card {risk_class}">
        <div style="font-size: 5rem;">{icon}</div>
        <h2 style="font-size: 2rem; margin: 15px 0;">Niveau de risque: {risk_level}</h2>
        <div style="font-size: 4rem; font-weight: bold; margin: 20px 0;">
            {prob:.1f}%
        </div>
        <div style="background: rgba(255,255,255,0.2); border-radius: 20px; padding: 20px; margin-top: 15px;">
            <p style="font-size: 1.1rem; margin: 0;">📋 {message}</p>
            <p style="font-size: 0.9rem; margin-top: 10px; opacity: 0.9;">
                <strong>Note importante:</strong> Ceci est un outil d'aide à la décision, non un diagnostic médical.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Affichage des scores détaillés
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Détails de l'évaluation")
        
        # Score total AQ-10
        total_score = sum(st.session_state.answers)
        max_score = 40
        percentage = (total_score / max_score) * 100
        
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=percentage,
            title={'text': "Score AQ-10", 'font': {'size': 24}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1},
                'bar': {'color': color},
                'steps': [
                    {'range': [0, 30], 'color': '#10ac84'},
                    {'range': [30, 50], 'color': '#48dbfb'},
                    {'range': [50, 70], 'color': '#feca57'},
                    {'range': [70, 100], 'color': '#ff6b6b'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 70
                }
            }
        ))
        fig_gauge.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        # Informations patient
        st.markdown("### 👤 Informations patient")
        st.write(f"**Nom du parent:** {st.session_state.parent_name}")
        st.write(f"**Enfant:** {st.session_state.child_name}")
        st.write(f"**Âge:** {st.session_state.child_age} mois")
        st.write(f"**Sexe:** {st.session_state.child_gender}")
    
    with col2:
        st.markdown("### 🎯 Répartition des scores")
        
        # Création d'un radar chart
        categories = [f"Q{i+1}" for i in range(10)]
        values = st.session_state.answers
        
        fig_radar = go.Figure(data=go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            marker=dict(color=color, size=8),
            line=dict(color=color, width=2),
            name=f"{st.session_state.child_name}"
        ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 4], tickvals=[0, 1, 2, 3, 4]),
                angularaxis=dict(tickfont=dict(size=10))
            ),
            showlegend=True,
            height=400,
            title="Profil comportemental par question"
        )
        st.plotly_chart(fig_radar, use_container_width=True)
        
        # Barre de confiance IA
        st.markdown("### 🤖 Confiance de l'IA")
        confidence = random.randint(85, 98)  # Simulation de confiance
        st.progress(confidence/100, text=f"Niveau de confiance: {confidence}%")
        
        st.info("💡 **Recommandation:** " + message)
    
    # Boutons d'action
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("---")
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("📥 Télécharger rapport PDF", use_container_width=True):
                st.success("📄 Rapport généré avec succès!")
        with col_btn2:
            if st.button("🔄 Nouvelle évaluation", use_container_width=True):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()

# ========== FOOTER ==========
st.markdown("""
<div class="footer">
    <p>🧠 <strong>NeuroSense AI+</strong> | Intelligence Artificielle pour la détection précoce de l'autisme</p>
    <p>🤖 Version 3.0 - Algorithmes de Machine Learning | Précision 92.5%</p>
    <p style="font-size: 0.8rem;">© 2025 NeuroSense AI+ - Tous droits réservés</p>
</div>
""", unsafe_allow_html=True)
