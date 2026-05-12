import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# ========== Configuration ==========
st.set_page_config(
    page_title="NeuroSense AI+",
    page_icon="🧠",
    layout="wide"
)

# ========== Styles CSS (comme avant) ==========
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    * { font-family: 'Poppins', sans-serif; }
    .header {
        background: linear-gradient(135deg, #1e3c72, #2a5298);
        border-radius: 30px;
        padding: 40px;
        text-align: center;
        color: white;
        margin-bottom: 30px;
    }
    .result-card {
        border-radius: 25px;
        padding: 30px;
        margin: 20px 0;
        text-align: center;
    }
    .risk-eleve { background: linear-gradient(135deg, #ff6b6b, #ee5a24); color: white; }
    .risk-moderate { background: linear-gradient(135deg, #feca57, #ff9f43); color: white; }
    .risk-faible { background: linear-gradient(135deg, #48dbfb, #0abde3); color: white; }
    .risk-tres-faible { background: linear-gradient(135deg, #10ac84, #1dd1a1); color: white; }
    .question-card {
        background: white;
        border-radius: 15px;
        padding: 20px;
        margin: 15px 0;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border-radius: 50px;
        padding: 12px 30px;
        width: 100%;
    }
    .footer {
        text-align: center;
        padding: 25px;
        color: #666;
        margin-top: 50px;
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
    np.random.seed(42)
    X_train = np.random.rand(1000, 12)
    y_train = [1 if np.mean(X_train[i, :10]) < 0.5 else 0 for i in range(1000)]
    modele = RandomForestClassifier(n_estimators=100, random_state=42)
    modele.fit(X_train, y_train)
    scaler = StandardScaler()
    scaler.fit(X_train)
    return modele, scaler

modele, scaler = creer_modele()

# ========== En-tête ==========
st.markdown("""
<div class="header">
    <h1>🧠 NeuroSense AI+</h1>
    <p>Détection précoce intelligente de l'autisme</p>
</div>
""", unsafe_allow_html=True)

# ========== Étape 1: Informations ==========
if st.session_state.etape == 1:
    st.markdown("## 👤 Informations")
    
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.nom_parent = st.text_input("Nom du parent")
        st.session_state.nom_enfant = st.text_input("Prénom de l'enfant")
    with col2:
        st.session_state.age_enfant = st.number_input("Âge (mois)", 0, 84, 24)
        st.session_state.sexe_enfant = st.selectbox("Sexe", ["", "Masculin", "Féminin"])
    
    if st.button("Commencer le questionnaire"):
        if st.session_state.nom_parent and st.session_state.nom_enfant and st.session_state.sexe_enfant:
            st.session_state.etape = 2
            st.rerun()
        else:
            st.error("Veuillez remplir tous les champs")

# ========== Étape 2: Questionnaire ==========
elif st.session_state.etape == 2:
    st.markdown("## 📝 Questionnaire AQ-10")
    
    questions = [
        "Contact visuel", "Réponse au nom", "Pointage", "Jeu d'imitation",
        "Comportements répétitifs", "Partage social", "Interaction sociale",
        "Sensibilité à la douleur", "Sensibilités sensorielles", "Communication verbale"
    ]
    
    for i, q in enumerate(questions):
        with st.container():
            st.markdown(f'<div class="question-card">{q}</div>', unsafe_allow_html=True)
            st.session_state.reponses[i] = st.radio(
                "Score",
                ["Toujours(4)", "Souvent(3)", "Parfois(2)", "Rarement(1)", "Jamais(0)"],
                index=2,
                key=f"q{i}",
                horizontal=True,
                label_visibility="collapsed"
            )
            # Convertir le texte en nombre
            scores = {"Toujours(4)":4, "Souvent(3)":3, "Parfois(2)":2, "Rarement(1)":1, "Jamais(0)":0}
            st.session_state.reponses[i] = scores[st.session_state.reponses[i]]
    
    if st.button("Voir les résultats"):
        st.session_state.etape = 3
        st.rerun()

# ========== Étape 3: Résultats ==========
elif st.session_state.etape == 3:
    st.markdown("## 📊 Résultat")
    
    with st.spinner("Analyse par IA..."):
        time.sleep(1)
        
        # Préparation des données
        reponses_binaires = [1 if r >= 3 else 0 for r in st.session_state.reponses]
        age_norm = st.session_state.age_enfant / 84
        sexe_val = 1 if st.session_state.sexe_enfant == "Masculin" else 0
        
        features = reponses_binaires + [age_norm, sexe_val]
        while len(features) < 12:
            features.append(0)
        
        X = np.array(features).reshape(1, -1)
        X_scaled = scaler.transform(X)
        
        probabilite = modele.predict_proba(X_scaled)[0][1] * 100
        
        score_total = sum(st.session_state.reponses)
        pourcentage_q = (score_total / 40) * 100
    
    # Niveau de risque
    if probabilite >= 70:
        niveau = "Élevé 🔴"
        classe = "risk-eleve"
        message = "Consultation avec un spécialiste recommandée"
    elif probabilite >= 50:
        niveau = "Modéré 🟠"
        classe = "risk-moderate"
        message = "Surveillance attentive recommandée"
    elif probabilite >= 30:
        niveau = "Faible 🟡"
        classe = "risk-faible"
        message = "Continuer l'observation normale"
    else:
        niveau = "Très faible 🟢"
        classe = "risk-tres-faible"
        message = "Développement typique"
    
    st.markdown(f"""
    <div class="result-card {classe}">
        <h2>Niveau: {niveau}</h2>
        <div style="font-size: 4rem;">{probabilite:.1f}%</div>
        <p>{message}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Détails
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Score questionnaire", f"{score_total}/40")
        st.metric("Pourcentage", f"{pourcentage_q:.1f}%")
    with col2:
        st.write(f"**Enfant:** {st.session_state.nom_enfant}")
        st.write(f"**Âge:** {st.session_state.age_enfant} mois")
        st.write(f"**Sexe:** {st.session_state.sexe_enfant}")
    
    # Graphique
    fig = go.Figure(data=go.Bar(
        x=[f"Q{i+1}" for i in range(10)],
        y=st.session_state.reponses,
        marker_color=['#ff6b6b' if r < 2 else '#10ac84' for r in st.session_state.reponses]
    ))
    fig.update_layout(title="Résultats par question", height=350)
    st.plotly_chart(fig, use_container_width=True)
    
    if st.button("🔄 Nouvelle évaluation"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# ========== Footer ==========
st.markdown("""
<div class="footer">
    <p>NeuroSense AI+ | Outil d'aide à la décision</p>
</div>
""", unsafe_allow_html=True)
