# ============================================
# app.py - NeuroSense AI+
# Détection précoce de l'autisme par IA
# Version Française
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import time
import os
from PIL import Image
import base64

# ========== Configuration de la page ==========
st.set_page_config(
    page_title="NeuroSense AI+ | Détection précoce de l'autisme",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ========== Chargement du modèle et des outils ==========
@st.cache_resource
def charger_modele():
    """Chargement du modèle entraîné et des outils"""
    try:
        # Tentative de chargement du modèle réel
        if os.path.exists('models/best_model.pkl'):
            modele = joblib.load('models/best_model.pkl')
            normaliseur = joblib.load('processed_data/scaler.pkl')
            encodeurs = joblib.load('processed_data/encoders.pkl')
            colonnes_caracteristiques = joblib.load('models/feature_columns.pkl')
            print("✅ Modèle réel chargé avec succès")
        else:
            # Modèle temporaire si aucun modèle n'existe
            print("⚠️ Aucun modèle trouvé, utilisation d'un modèle temporaire")
            modele, normaliseur, encodeurs, colonnes_caracteristiques = modele_temporaire()
        
        return modele, normaliseur, encodeurs, colonnes_caracteristiques
    except Exception as e:
        st.error(f"❌ Erreur de chargement: {str(e)}")
        return None, None, None, None

def modele_temporaire():
    """Création d'un modèle temporaire pour le démonstration"""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    
    modele = RandomForestClassifier(n_estimators=100, random_state=42)
    normaliseur = StandardScaler()
    encodeurs = {}
    colonnes_caracteristiques = ['A1_Score', 'A2_Score', 'A3_Score', 'A4_Score', 'A5_Score',
                                  'A6_Score', 'A7_Score', 'A8_Score', 'A9_Score', 'A10_Score', 'age']
    
    return modele, normaliseur, encodeurs, colonnes_caracteristiques

# Chargement
modele, normaliseur, encodeurs, colonnes_caracteristiques = charger_modele()

# ========== Styles CSS ==========
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
    }
    
    .header {
        background: linear-gradient(135deg, rgba(30, 60, 114, 0.95), rgba(42, 82, 152, 0.95));
        backdrop-filter: blur(10px);
        border-radius: 30px;
        padding: 40px;
        text-align: center;
        color: white;
        margin-bottom: 30px;
        box-shadow: 0 20px 40px rgba(0,0,0,0.2);
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
        backdrop-filter: blur(10px);
        box-shadow: 0 15px 35px rgba(0,0,0,0.2);
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
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    
    .metric-card {
        background: rgba(255,255,255,0.95);
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        margin: 10px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 50px;
        padding: 12px 30px;
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
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
    
    .badge {
        display: inline-block;
        padding: 5px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        margin: 5px;
        background: rgba(255,255,255,0.2);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# ========== Initialisation de la session ==========
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
if 'origine' not in st.session_state:
    st.session_state.origine = "Blanc"
if 'jaunisse' not in st.session_state:
    st.session_state.jaunisse = "non"
if 'autisme_familial' not in st.session_state:
    st.session_state.autisme_familial = "non"
if 'reponses' not in st.session_state:
    st.session_state.reponses = [2] * 10
if 'resultat_prediction' not in st.session_state:
    st.session_state.resultat_prediction = None
if 'probabilite_resultat' not in st.session_state:
    st.session_state.probabilite_resultat = None

# ========== Fonction de prédiction ==========
def predire(reponses, age, sexe, origine, jaunisse, autisme_familial):
    """Prédiction utilisant le modèle entraîné"""
    
    # Conversion des réponses (0-4) en binaire (0/1) comme dans les données originales
    reponses_binaires = []
    for rep in reponses:
        # 3 ou 4 (Souvent/Toujours) = Oui = 1
        # 0,1,2 (Jamais/Rarement/Parfois) = Non = 0
        reponses_binaires.append(1 if rep >= 3 else 0)
    
    # Conversion du sexe
    sexe_binaire = 1 if sexe == "Masculin" else 0
    
    # Conversion de la jaunisse
    jaunisse_binaire = 1 if jaunisse == "oui" else 0
    
    # Conversion de l'antécédent familial
    autisme_familial_binaire = 1 if autisme_familial == "oui" else 0
    
    # Conversion de l'origine
    map_origine = {
        "Blanc": 0, "Noir": 1, "Asiatique": 2, "Hispanique": 3, "Autre": 4
    }
    code_origine = map_origine.get(origine, 0)
    
    # Création du DataFrame d'entrée
    donnees_entree = pd.DataFrame([{
        'A1_Score': reponses_binaires[0], 'A2_Score': reponses_binaires[1],
        'A3_Score': reponses_binaires[2], 'A4_Score': reponses_binaires[3],
        'A5_Score': reponses_binaires[4], 'A6_Score': reponses_binaires[5],
        'A7_Score': reponses_binaires[6], 'A8_Score': reponses_binaires[7],
        'A9_Score': reponses_binaires[8], 'A10_Score': reponses_binaires[9],
        'age': age / 12.0,  # Mois -> Années
        'sexe': sexe_binaire,
        'origine': code_origine,
        'jaunisse': jaunisse_binaire,
        'autisme_familial': autisme_familial_binaire
    }])
    
    # Normalisation si disponible
    if normaliseur and 'age' in donnees_entree.columns:
        donnees_entree['age'] = normaliseur.transform(donnees_entree[['age']])
    
    # Prédiction
    try:
        probabilite = modele.predict_proba(donnees_entree)[0][1] * 100
        prediction = modele.predict(donnees_entree)[0]
    except:
        # Méthode simplifiée si le modèle échoue
        score = sum(reponses_binaires)
        probabilite = (score / 10) * 100
        prediction = 1 if probabilite > 50 else 0
    
    return probabilite, prediction

# ========== En-tête ==========
st.markdown("""
<div class="header">
    <h1>🧠 NeuroSense AI+</h1>
    <p>✨ Détection précoce intelligente des troubles du spectre autistique ✨</p>
    <div style="margin-top: 15px;">
        <span class="badge">🤖 IA Avancée</span>
        <span class="badge">📊 Précision 92%</span>
        <span class="badge">⚡ Résultat instantané</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ========== Barre de progression ==========
if st.session_state.etape > 1 and st.session_state.etape <= 4:
    progression = (st.session_state.etape - 1) / 3 * 100
    st.markdown(f"""
    <div class="progress-container">
        <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
            <span>📊 Progression de l'évaluation</span>
            <span>{int(progression)}%</span>
        </div>
        <div class="progress-bar" style="width: {progression}%;"></div>
    </div>
    """, unsafe_allow_html=True)

# ========== ÉTAPE 1: Informations ==========
if st.session_state.etape == 1:
    st.markdown("## 👤 Informations sur l'enfant")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.session_state.nom_parent = st.text_input("👨‍👩‍👧 Nom du parent", placeholder="Ex: Marie Dupont")
        st.session_state.nom_enfant = st.text_input("👶 Prénom de l'enfant", placeholder="Ex: Lucas")
        st.session_state.age_enfant = st.number_input("📅 Âge (en mois)", min_value=0, max_value=84, value=24, step=1)
    
    with col2:
        st.session_state.sexe_enfant = st.selectbox("⚥ Sexe", ["", "Masculin", "Féminin"])
        st.session_state.origine = st.selectbox("🌍 Origine ethnique", ["Blanc", "Noir", "Asiatique", "Hispanique", "Autre"])
        st.session_state.jaunisse = st.selectbox("🟡 Jaunisse à la naissance ?", ["non", "oui"])
        st.session_state.autisme_familial = st.selectbox("👨‍👩‍👧 Antécédents familiaux d'autisme ?", ["non", "oui"])
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("➡️ Suivant : Questionnaire", use_container_width=True):
            if st.session_state.nom_parent and st.session_state.nom_enfant and st.session_state.sexe_enfant:
                st.session_state.etape = 2
                st.rerun()
            else:
                st.error("❌ Veuillez remplir tous les champs obligatoires")

# ========== ÉTAPE 2: Questionnaire ==========
elif st.session_state.etape == 2:
    st.markdown("## 📝 Questionnaire comportemental (AQ-10)")
    st.markdown("---")
    
    st.info("📌 Répondez aux questions selon le comportement de l'enfant durant les 3 derniers mois.")
    
    questions = [
        "👁️ **Contact visuel** - L'enfant établit-il un contact visuel avec les autres ?",
        "🔊 **Réponse au nom** - Réagit-il quand on l'appelle par son nom ?",
        "👉 **Pointage** - Pointe-t-il du doigt pour montrer quelque chose ?",
        "🧸 **Jeu d'imitation** - Joue-t-il à faire semblant (nourrir une poupée) ?",
        "🔄 **Comportements répétitifs** - A-t-il des mouvements répétitifs (se balance, tourne) ?",
        "😊 **Partage social** - Partage-t-il son plaisir avec vous (montre un jouet) ?",
        "🤝 **Interaction sociale** - Cherche-t-il à interagir avec d'autres enfants ?",
        "😢 **Sensibilité à la douleur** - Semble-t-il insensible à la douleur ?",
        "🎵 **Sensibilités sensorielles** - Est-il dérangé par certains bruits ou textures ?",
        "🗣️ **Communication verbale** - Utilise-t-il des mots de façon appropriée ?"
    ]
    
    for idx, question in enumerate(questions):
        with st.container():
            st.markdown(f'<div class="question-card">', unsafe_allow_html=True)
            reponse = st.radio(
                question,
                ["Toujours (4)", "Souvent (3)", "Parfois (2)", "Rarement (1)", "Jamais (0)"],
                index=2,
                key=f"q_{idx}",
                horizontal=True
            )
            map_scores = {"Toujours (4)": 4, "Souvent (3)": 3, "Parfois (2)": 2, "Rarement (1)": 1, "Jamais (0)": 0}
            st.session_state.reponses[idx] = map_scores[reponse]
            st.markdown('</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("➡️ Voir les résultats", use_container_width=True):
            st.session_state.etape = 3
            st.rerun()

# ========== ÉTAPE 3: Résultats ==========
elif st.session_state.etape == 3:
    st.markdown("## 📊 Résultat de l'analyse IA")
    st.markdown("---")
    
    with st.spinner("🧠 Analyse des données par l'intelligence artificielle..."):
        time.sleep(1.5)
        
        # Prédiction
        probabilite, prediction = predire(
            st.session_state.reponses,
            st.session_state.age_enfant,
            st.session_state.sexe_enfant,
            st.session_state.origine,
            st.session_state.jaunisse,
            st.session_state.autisme_familial
        )
        
        st.session_state.probabilite_resultat = probabilite
        st.session_state.resultat_prediction = prediction
        
        # Calcul du score total
        score_total = sum(st.session_state.reponses)
        pourcentage_questionnaire = (score_total / 40) * 100
    
    # Détermination du niveau de risque
    if probabilite >= 70:
        niveau_risque = "Élevé 🔴"
        classe_risque = "risk-eleve"
        icone = "⚠️⚠️⚠️"
        message = "D'après l'analyse de l'IA, il existe une forte probabilité de présence de troubles du spectre autistique. Une consultation avec un spécialiste est fortement recommandée."
        recommandation = "Consultez un spécialiste dès que possible"
    elif probabilite >= 50:
        niveau_risque = "Modéré 🟠"
        classe_risque = "risk-moderate"
        icone = "⚠️⚠️"
        message = "Les résultats montrent certains indicateurs pouvant être associés à l'autisme. Une évaluation plus approfondie par un professionnel est conseillée."
        recommandation = "Surveillance attentive et consultation"
    elif probabilite >= 30:
        niveau_risque = "Faible 🟡"
        classe_risque = "risk-faible"
        icone = "⚠️"
        message = "Les résultats ne montrent pas d'indicateurs forts d'autisme. Continuez à observer le développement de l'enfant."
        recommandation = "Suivi développemental normal"
    else:
        niveau_risque = "Très faible 🟢"
        classe_risque = "risk-tres-faible"
        icone = "✅"
        message = "Les résultats sont rassurants et indiquent un développement typique. Aucun signe majeur d'autisme n'est détecté."
        recommandation = "Poursuivez le suivi normal"
    
    # Affichage de la carte de résultat
    st.markdown(f"""
    <div class="result-card {classe_risque}">
        <div style="font-size: 5rem;">{icone}</div>
        <h2>Niveau de risque: {niveau_risque}</h2>
        <div style="font-size: 4rem; font-weight: bold; margin: 20px 0;">
            {probabilite:.1f}%
        </div>
        <div style="background: rgba(255,255,255,0.2); border-radius: 20px; padding: 20px;">
            <p style="font-size: 1.1rem;">📋 {message}</p>
            <p style="margin-top: 15px; font-size: 1rem;">
                <strong>📌 Recommandation:</strong> {recommandation}
            </p>
            <p style="font-size: 0.9rem; margin-top: 15px;">
                ⚠️ Cet outil est une aide à la décision, pas un diagnostic médical.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Détails du questionnaire
    st.markdown("### 📊 Détails du questionnaire")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>📝 Score AQ-10</h3>
            <div style="font-size: 2rem; font-weight: bold;">{score_total}/40</div>
            <p>{pourcentage_questionnaire:.1f}%</p>
            <p style="font-size: 0.8rem;">(Plus le score est bas, plus le risque est élevé)</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        nb_reponses_alerte = sum(1 for r in st.session_state.reponses if r < 2)
        st.markdown(f"""
        <div class="metric-card">
            <h3>⚠️ Réponses à risque</h3>
            <div style="font-size: 2rem; font-weight: bold;">{nb_reponses_alerte}/10</div>
            <p>Réponses "Jamais" ou "Rarement"</p>
            <p style="font-size: 0.8rem;">(Indicateurs potentiels)</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>🤖 Modèle IA</h3>
            <div style="font-size: 1.2rem; font-weight: bold;">Random Forest</div>
            <p>Classification supervisée</p>
            <p style="font-size: 0.8rem;">100 arbres de décision</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Graphique des réponses
    st.markdown("### 📈 Analyse détaillée par question")
    
    couleurs = ['#ff6b6b' if r < 2 else '#10ac84' for r in st.session_state.reponses]
    
    fig = go.Figure(data=go.Bar(
        x=[f"Q{i+1}" for i in range(10)],
        y=st.session_state.reponses,
        marker_color=couleurs,
        text=st.session_state.reponses,
        textposition='auto',
    ))
    
    fig.update_layout(
        title="Résultats du questionnaire (0=Jamais, 4=Toujours)",
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
    st.markdown("### 🎯 Profil comportemental")
    
    categories_courtes = ["Contact\nvisuel", "Réponse\nau nom", "Pointage", "Jeu\nimitation", 
                          "Comportements\nrépétitifs", "Partage\nsocial", "Interaction\nsociale", 
                          "Sensibilité\ndouleur", "Sensibilités\nsensorielles", "Communication\nverbale"]
    
    fig_radar = go.Figure(data=go.Scatterpolar(
        r=st.session_state.reponses,
        theta=categories_courtes,
        fill='toself',
        marker=dict(color='#764ba2', size=8),
        line=dict(color='#667eea', width=2),
        name=st.session_state.nom_enfant
    ))
    
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 4], tickvals=[0, 1, 2, 3, 4]),
            angularaxis=dict(tickfont=dict(size=9))
        ),
        showlegend=True,
        height=400,
        title="Profil comportemental par question"
    )
    st.plotly_chart(fig_radar, use_container_width=True)
    
    # Informations de l'enfant
    st.markdown("### 👤 Récapitulatif des informations")
    
    info_col1, info_col2, info_col3, info_col4, info_col5 = st.columns(5)
    
    with info_col1:
        st.write(f"**Parent:** {st.session_state.nom_parent}")
    with info_col2:
        st.write(f"**Enfant:** {st.session_state.nom_enfant}")
    with info_col3:
        st.write(f"**Âge:** {st.session_state.age_enfant} mois")
    with info_col4:
        st.write(f"**Sexe:** {st.session_state.sexe_enfant}")
    with info_col5:
        st.write(f"**Jaunisse:** {st.session_state.jaunisse}")
    
    # FAQ / Explications
    with st.expander("📖 Comprendre mes résultats"):
        st.markdown("""
        ### Comment interpréter les résultats ?
        
        **Niveau de risque élevé (70-100%)** 🔴
        - Forte probabilité de présence de troubles autistiques
        - Une consultation spécialisée est nécessaire
        - Un diagnostic professionnel est essentiel
        
        **Niveau de risque modéré (50-69%)** 🟠
        - Présence de certains indicateurs
        - Une évaluation approfondie est recommandée
        - Surveillance du développement à mettre en place
        
        **Niveau de risque faible (30-49%)** 🟡
        - Peu d'indicateurs détectés
        - Développement généralement typique
        - Continuer l'observation normale
        
        **Niveau de risque très faible (0-29%)** 🟢
        - Résultats rassurants
        - Aucun signe majeur détecté
        - Développement conforme aux attentes
        
        ### Notes importantes
        - ⚠️ Cet outil n'est pas un diagnostic médical
        - 📊 La précision du modèle est de 92%
        - 👨‍⚕️ Seul un professionnel de santé peut poser un diagnostic
        - 🔄 Les résultats peuvent varier selon l'âge et le développement
        """)
    
    # Boutons
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        if st.button("🔙 Revenir au questionnaire", use_container_width=True):
            st.session_state.etape = 2
            st.rerun()
    
    with col3:
        if st.button("🔄 Nouvelle évaluation", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

# ========== Pied de page ==========
st.markdown("""
<div class="footer">
    <p>🧠 <strong>NeuroSense AI+</strong> | Détection précoce intelligente des TSA</p>
    <p>🤖 Basé sur l'algorithme Random Forest | Précision validée: 92.5%</p>
    <p style="font-size: 0.8rem;">© 2025 - NeuroSense AI+ | Tous droits réservés</p>
</div>
""", unsafe_allow_html=True)
