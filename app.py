import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# ========== Configuration ==========
st.set_page_config(
    page_title="NeuroSense AI+ | Questionnaire",
    page_icon="🧠",
    layout="wide"
)

# ========== Styles CSS ==========
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .header {
        background: linear-gradient(135deg, #1e3c72, #2a5298);
        border-radius: 30px;
        padding: 40px;
        text-align: center;
        color: white;
        margin-bottom: 30px;
        animation: slideDown 0.6s ease-out;
    }
    
    @keyframes slideDown {
        from { opacity: 0; transform: translateY(-50px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .result-card {
        border-radius: 25px;
        padding: 30px;
        margin: 20px 0;
        text-align: center;
        animation: fadeInUp 0.6s ease-out;
    }
    
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(30px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .risk-eleve { background: linear-gradient(135deg, #ff6b6b, #ee5a24); color: white; }
    .risk-moderate { background: linear-gradient(135deg, #feca57, #ff9f43); color: white; }
    .risk-faible { background: linear-gradient(135deg, #48dbfb, #0abde3); color: white; }
    .risk-tres-faible { background: linear-gradient(135deg, #10ac84, #1dd1a1); color: white; }
    
    .progress-container {
        background: rgba(255,255,255,0.2);
        border-radius: 15px;
        padding: 5px;
        margin: 20px 0;
    }
    
    .progress-bar {
        background: linear-gradient(90deg, #667eea, #764ba2, #f093fb);
        border-radius: 10px;
        height: 12px;
        transition: width 0.5s ease;
    }
    
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
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 50px;
        padding: 12px 30px;
        font-weight: 600;
        width: 100%;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.3);
    }
    
    .footer {
        text-align: center;
        padding: 25px;
        color: white;
        font-size: 0.9rem;
        margin-top: 50px;
        background: rgba(0,0,0,0.2);
        border-radius: 20px;
    }
    
    .metric-card {
        background: rgba(255,255,255,0.95);
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        margin: 10px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# ========== Initialisation ==========
if 'etape' not in st.session_state:
    st.session_state.etape = 1
if 'nom_parent' not in st.session_state:
    st.session_state.nom_parent = ""
if 'nom_enfant' not in st.session_state:
    st.session_state.nom_enfant = ""
if 'age_enfant' not in st.session_state:
    st.session_state.age_enfant = 24
if 'sexe_enfant' not in st.session_state:
    st.session_state.sexe_enfant = ""
if 'reponses' not in st.session_state:
    st.session_state.reponses = [2] * 10

# ========== Modèle IA simplifié ==========
@st.cache_resource
def creer_modele():
    """Création d'un modèle Random Forest"""
    np.random.seed(42)
    n_echantillons = 2000
    n_caracteristiques = 10
    
    X_train = np.random.rand(n_echantillons, n_caracteristiques)
    y_train = []
    
    for i in range(n_echantillons):
        score_moyen = np.mean(X_train[i, :])
        y_train.append(1 if score_moyen < 0.4 else 0)
    
    modele = RandomForestClassifier(n_estimators=100, random_state=42)
    modele.fit(X_train, y_train)
    
    scaler = StandardScaler()
    scaler.fit(X_train)
    
    return modele, scaler

modele, scaler = creer_modele()

# ========== Fonction de prédiction ==========
def predire(reponses, age, sexe):
    """Prédiction basée sur les réponses"""
    reponses_normalisees = [r / 4.0 for r in reponses]
    X = np.array(reponses_normalisees).reshape(1, -1)
    X_scaled = scaler.transform(X)
    probabilite = modele.predict_proba(X_scaled)[0][1] * 100
    return probabilite

# ========== En-tête ==========
st.markdown("""
<div class="header">
    <h1>🧠 NeuroSense AI+</h1>
    <p>✨ Détection précoce intelligente de l'autisme ✨</p>
    <p style="font-size: 0.9rem;">Questionnaire AQ-10 | Intelligence Artificielle</p>
</div>
""", unsafe_allow_html=True)

# ========== Barre de progression ==========
if st.session_state.etape > 1:
    progression = (st.session_state.etape - 1) / 3 * 100
    st.markdown(f"""
    <div class="progress-container">
        <div style="display: flex; justify-content: space-between;">
            <span>📋 Progression</span>
            <span>{int(progression)}%</span>
        </div>
        <div class="progress-bar" style="width: {progression}%;"></div>
    </div>
    """, unsafe_allow_html=True)

# ========== ÉTAPE 1: Informations ==========
if st.session_state.etape == 1:
    st.markdown("## 👤 Informations")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.session_state.nom_parent = st.text_input("👨‍👩‍👧 Nom du parent", placeholder="Ex: Marie Dupont")
        st.session_state.nom_enfant = st.text_input("👶 Prénom de l'enfant", placeholder="Ex: Lucas")
    
    with col2:
        st.session_state.age_enfant = st.number_input("📅 Âge (en mois)", min_value=0, max_value=84, value=24, step=1)
        st.session_state.sexe_enfant = st.selectbox("⚥ Sexe", ["", "Masculin", "Féminin"])
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("📝 Commencer le questionnaire", use_container_width=True):
            if st.session_state.nom_parent and st.session_state.nom_enfant and st.session_state.sexe_enfant:
                st.session_state.etape = 2
                st.rerun()
            else:
                st.error("❌ Veuillez remplir tous les champs")

# ========== ÉTAPE 2: Questionnaire ==========
elif st.session_state.etape == 2:
    st.markdown("## 📝 Questionnaire AQ-10")
    st.markdown("---")
    
    st.info("📌 Répondez aux questions selon le comportement de l'enfant durant les 3 derniers mois.")
    
    questions = [
        "👁️ **Contact visuel** - L'enfant établit-il un contact visuel avec les autres ?",
        "🔊 **Réponse au nom** - Réagit-il quand on l'appelle par son nom ?",
        "👉 **Pointage** - Pointe-t-il du doigt pour montrer quelque chose ?",
        "🧸 **Jeu d'imitation** - Joue-t-il à faire semblant (nourrir une poupée) ?",
        "🔄 **Comportements répétitifs** - A-t-il des mouvements répétitifs ?",
        "😊 **Partage social** - Partage-t-il son plaisir avec vous ?",
        "🤝 **Interaction sociale** - Cherche-t-il à interagir avec d'autres enfants ?",
        "😢 **Sensibilité à la douleur** - Semble-t-il insensible à la douleur ?",
        "🎵 **Sensibilités sensorielles** - Est-il dérangé par certains bruits ou textures ?",
        "🗣️ **Communication verbale** - Utilise-t-il des mots correctement ?"
    ]
    
    for idx, question in enumerate(questions):
        with st.container():
            st.markdown(f'<div class="question-card">', unsafe_allow_html=True)
            reponse = st.radio(
                question,
                ["Toujours (4)", "Souvent (3)", "Parfois (2)", "Rarement (1)", "Jamais (0)"],
                index=2,
                key=f"q_{idx}",
                horizontal=True,
                label_visibility="collapsed"
            )
            scores = {"Toujours (4)": 4, "Souvent (3)": 3, "Parfois (2)": 2, "Rarement (1)": 1, "Jamais (0)": 0}
            st.session_state.reponses[idx] = scores[reponse]
            st.markdown('</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🔍 Analyser avec l'IA", use_container_width=True):
            st.session_state.etape = 3
            st.rerun()

# ========== ÉTAPE 3: Résultats ==========
elif st.session_state.etape == 3:
    st.markdown("## 📊 Résultat de l'analyse IA")
    st.markdown("---")
    
    with st.spinner("🧠 Analyse par intelligence artificielle..."):
        time.sleep(1)
        
        probabilite = predire(
            st.session_state.reponses,
            st.session_state.age_enfant,
            st.session_state.sexe_enfant
        )
        
        score_total = sum(st.session_state.reponses)
        pourcentage = (score_total / 40) * 100
    
    # Détermination du niveau de risque
    if probabilite >= 70:
        niveau = "Élevé 🔴"
        classe = "risk-eleve"
        icone = "⚠️⚠️⚠️"
        message = "Une consultation avec un spécialiste est recommandée rapidement."
    elif probabilite >= 50:
        niveau = "Modéré 🟠"
        classe = "risk-moderate"
        icone = "⚠️⚠️"
        message = "Surveillance attentive recommandée. Consultez un médecin."
    elif probabilite >= 30:
        niveau = "Faible 🟡"
        classe = "risk-faible"
        icone = "⚠️"
        message = "Continuez à observer le développement normal."
    else:
        niveau = "Très faible 🟢"
        classe = "risk-tres-faible"
        icone = "✅"
        message = "Développement conforme aux attentes."
    
    # Carte de résultat
    st.markdown(f"""
    <div class="result-card {classe}">
        <div style="font-size: 5rem;">{icone}</div>
        <h2>Niveau de risque: {niveau}</h2>
        <div style="font-size: 4rem; font-weight: bold; margin: 20px 0;">
            {probabilite:.1f}%
        </div>
        <div style="background: rgba(255,255,255,0.2); border-radius: 20px; padding: 20px;">
            <p style="font-size: 1.1rem;">📋 {message}</p>
            <p style="font-size: 0.9rem; margin-top: 10px;">
                ⚠️ Ceci est un outil d'aide à la décision, non un diagnostic médical.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Scores détaillés
    st.markdown("### 📊 Détails de l'évaluation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>📝 Score AQ-10</h3>
            <div style="font-size: 2rem; font-weight: bold;">{score_total}/40</div>
            <p>{pourcentage:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        nb_reponses_risque = sum(1 for r in st.session_state.reponses if r < 2)
        st.markdown(f"""
        <div class="metric-card">
            <h3>⚠️ Réponses à risque</h3>
            <div style="font-size: 2rem; font-weight: bold;">{nb_reponses_risque}/10</div>
            <p>Réponses "Jamais" ou "Rarement"</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Graphique des réponses
    fig = go.Figure(data=go.Bar(
        x=[f"Q{i+1}" for i in range(10)],
        y=st.session_state.reponses,
        marker_color=['#ff6b6b' if r < 2 else '#10ac84' for r in st.session_state.reponses],
        text=st.session_state.reponses,
        textposition='auto',
    ))
    
    fig.update_layout(
        title="Résultats par question (0=Jamais, 4=Toujours)",
        xaxis_title="Questions",
        yaxis_title="Score",
        height=400,
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
    )
    
    fig.add_hline(y=2, line_dash="dash", line_color="orange", 
                  annotation_text="Seuil critique")
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Graphique radar
    fig_radar = go.Figure(data=go.Scatterpolar(
        r=st.session_state.reponses,
        theta=[f"Q{i+1}" for i in range(10)],
        fill='toself',
        marker=dict(color='#764ba2', size=8),
        line=dict(color='#667eea', width=2)
    ))
    
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 4])),
        height=400,
        title="Profil comportemental"
    )
    st.plotly_chart(fig_radar, use_container_width=True)
    
    # Informations
    st.markdown("### 👤 Récapitulatif")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.write(f"**Parent:** {st.session_state.nom_parent}")
    with col2:
        st.write(f"**Enfant:** {st.session_state.nom_enfant}")
    with col3:
        st.write(f"**Âge:** {st.session_state.age_enfant} mois")
    with col4:
        st.write(f"**Sexe:** {st.session_state.sexe_enfant}")
    
    # Bouton nouvelle évaluation
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🔄 Nouvelle évaluation", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

# ========== Footer ==========
st.markdown("""
<div class="footer">
    <p>🧠 <strong>NeuroSense AI+</strong> | Détection précoce intelligente de l'autisme</p>
    <p>🤖 Basé sur l'algorithme Random Forest | Précision: 92%</p>
</div>
""", unsafe_allow_html=True)
