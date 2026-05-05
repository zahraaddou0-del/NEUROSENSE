import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import random
import plotly.express as px
import plotly.graph_objects as go

# ========== PAGE CONFIGURATION ==========
st.set_page_config(
    page_title="NeuroSense AI+ - Détection précoce de l'autisme",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ========== STYLE CSS PERSONNALISÉ ==========
st.markdown("""
<style>
    /* Style général */
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #e4e8f0 100%);
    }
    
    /* Cartes de résultat */
    .result-card {
        border-radius: 20px;
        padding: 25px;
        margin: 20px 0;
        text-align: center;
        animation: fadeIn 0.5s ease-in;
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
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* En-tête */
    .header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        border-radius: 20px;
        padding: 30px;
        text-align: center;
        color: white;
        margin-bottom: 30px;
    }
    
    /* Barre de progression */
    .progress-bar {
        background: linear-gradient(90deg, #667eea, #764ba2);
        border-radius: 10px;
        height: 8px;
        transition: width 0.3s ease;
    }
    
    /* Pied de page */
    .footer {
        text-align: center;
        padding: 20px;
        color: #666;
        font-size: 0.8rem;
        border-top: 1px solid #ddd;
        margin-top: 40px;
    }
</style>
""", unsafe_allow_html=True)

# ========== INITIALISATION SESSION STATE ==========
if 'etape' not in st.session_state:
    st.session_state.etape = 1
if 'type_utilisateur' not in st.session_state:
    st.session_state.type_utilisateur = ""
if 'nom_parent' not in st.session_state:
    st.session_state.nom_parent = ""
if 'age_parent' not in st.session_state:
    st.session_state.age_parent = 0
if 'nom_enfant' not in st.session_state:
    st.session_state.nom_enfant = ""
if 'age_enfant' not in st.session_state:
    st.session_state.age_enfant = 0
if 'sexe_enfant' not in st.session_state:
    st.session_state.sexe_enfant = ""
if 'historique_familial' not in st.session_state:
    st.session_state.historique_familial = ""
if 'score_questionnaire' not in st.session_state:
    st.session_state.score_questionnaire = None
if 'score_audio' not in st.session_state:
    st.session_state.score_audio = None
if 'score_vision' not in st.session_state:
    st.session_state.score_vision = None
if 'score_global' not in st.session_state:
    st.session_state.score_global = None
if 'pourcentage' not in st.session_state:
    st.session_state.pourcentage = None
if 'niveau' not in st.session_state:
    st.session_state.niveau = ""
if 'recommandation' not in st.session_state:
    st.session_state.recommandation = ""
if 'reponses' not in st.session_state:
    st.session_state.reponses = []

# ========== EN-TÊTE ==========
st.markdown("""
<div class="header">
    <h1 style="font-size: 3rem; margin:0;">🧠 NeuroSense AI+</h1>
    <p style="font-size: 1.2rem; margin-top:10px;">
        Détection précoce intelligente de l'autisme par questionnaire, analyse vocale et vision
    </p>
</div>
""", unsafe_allow_html=True)

# ========== AFFICHAGE DE LA PROGRESSION ==========
if st.session_state.etape > 1 and st.session_state.etape < 6:
    progression = (st.session_state.etape - 1) / 5 * 100
    st.markdown(f"""
    <div style="margin-bottom: 20px;">
        <p style="margin-bottom: 5px;">📊 Progression de l'évaluation</p>
        <div style="background: #e0e0e0; border-radius: 10px;">
            <div class="progress-bar" style="width: {progression}%; border-radius: 10px;"></div>
        </div>
        <p style="text-align: right; font-size: 0.8rem; margin-top: 5px;">{int(progression)}%</p>
    </div>
    """, unsafe_allow_html=True)

# ========== ÉTAPE 1: CHOIX DU PROFIL ==========
if st.session_state.etape == 1:
    st.markdown("## 📋 Étape 1: Choix du profil")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    border-radius: 20px; padding: 2rem; text-align: center; color: white;">
            <div style="font-size: 4rem;">👨‍👩‍👧</div>
            <h3>Mode Parent</h3>
            <p>Pour les familles qui souhaitent évaluer leur enfant</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("📝 Choisir Mode Parent", key="btn_parent", use_container_width=True):
            st.session_state.type_utilisateur = "Parent"
            st.session_state.etape = 2
            st.rerun()
    
    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                    border-radius: 20px; padding: 2rem; text-align: center; color: white;">
            <div style="font-size: 4rem;">🏥</div>
            <h3>Mode Professionnel</h3>
            <p>Pour les médecins et spécialistes</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("👨‍⚕️ Choisir Mode Professionnel", key="btn_pro", use_container_width=True):
            st.session_state.type_utilisateur = "Professionnel"
            st.session_state.etape = 2
            st.rerun()

# ========== ÉTAPE 2: INFORMATIONS ==========
elif st.session_state.etape == 2:
    st.markdown("## 👤 Étape 2: Vos informations")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.session_state.nom_parent = st.text_input("📝 Votre nom complet", placeholder="Ex: Marie Dupont")
        st.session_state.age_parent = st.number_input("🎂 Votre âge", min_value=18, max_value=100, step=1)
    
    with col2:
        st.session_state.nom_enfant = st.text_input("👶 Nom de l'enfant", placeholder="Ex: Lucas")
        st.session_state.age_enfant = st.number_input("📅 Âge de l'enfant (en mois)", min_value=0, max_value=72, step=1)
        st.session_state.sexe_enfant = st.selectbox("⚥ Sexe", ["", "Masculin", "Féminin"])
    
    st.session_state.historique_familial = st.radio(
        "🧬 Antécédents familiaux d'autisme",
        ["", "Oui, diagnostiqué", "Oui, suspicion", "Non", "Je ne sais pas"]
    )
    
    col1, col2, col3 = st.columns([1,1,1])
    with col2:
        if st.button("➡️ Suivant", use_container_width=True):
            if st.session_state.nom_parent and st.session_state.age_parent > 0:
                st.session_state.etape = 3
                st.rerun()
            else:
                st.error("Veuillez remplir tous les champs obligatoires")

# ========== ÉTAPE 3: QUESTIONNAIRE ==========
elif st.session_state.etape == 3:
    st.markdown("## 📝 Étape 3: Questionnaire d'évaluation")
    st.markdown("---")
    st.info("Veuillez répondre aux questions suivantes en fonction du comportement de votre enfant.")
    
    questions = [
        "👁️ Votre enfant regarde-t-il dans les yeux?",
        "🔊 Votre enfant réagit-il quand on appelle son nom?",
        "👉 Votre enfant pointe-t-il du doigt pour montrer quelque chose?",
        "🧸 Votre enfant joue-t-il à faire semblant?",
        "🚫 Votre enfant évite-t-il le contact visuel?",
        "🔄 Votre enfant a-t-il des comportements répétitifs?",
        "😊 Votre enfant partage-t-il son plaisir?",
        "😢 Votre enfant semble-t-il insensible à la douleur?",
        "👂 Votre enfant a-t-il des sensibilités aux bruits?",
        "🗣️ Votre enfant utilise-t-il des mots ou phrases?"
    ]
    
    reponses = []
    for i, q in enumerate(questions):
        reponse = st.radio(
            q,
            ["Toujours", "Souvent", "Parfois", "Rarement", "Jamais"],
            key=f"q{i}",
            horizontal=True,
            label_visibility="collapsed"
        )
        reponses.append(reponse)
        st.session_state.reponses = reponses
    
    col1, col2, col3 = st.columns([1,1,1])
    with col2:
        if st.button("📊 Calculer le score", use_container_width=True):
            score_map = {"Toujours": 4, "Souvent": 3, "Parfois": 2, "Rarement": 1, "Jamais": 0}
            score = sum(score_map[r] for r in reponses)
            st.session_state.score_questionnaire = score
            
            st.session_state.etape = 4
            st.rerun()

# ========== ÉTAPE 4: ANALYSE VOCALE (SIMULATION) ==========
elif st.session_state.etape == 4:
    st.markdown("## 🎤 Étape 4: Analyse vocale")
    st.markdown("---")
    
    st.info("Simulation d'analyse vocale - Dans la version réelle, l'enfant serait invité à parler.")
    
    scores_audio = [random.randint(60, 100) for _ in range(4)]
    score_audio_total = int(np.mean(scores_audio))
    st.session_state.score_audio = score_audio_total
    
    metrics = {
        "Prosodie (intonation)": scores_audio[0],
        "Clarté articulatoire": scores_audio[1],
        "Variabilité vocale": scores_audio[2],
        "Réponse sonore": scores_audio[3]
    }
    
    for metric, score in metrics.items():
        st.progress(score/100, text=f"{metric}: {score}%")
    
    if st.button("➡️ Continuer vers l'analyse visuelle", use_container_width=True):
        st.session_state.etape = 5
        st.rerun()

# ========== ÉTAPE 5: ANALYSE VISION (SIMULATION) ==========
elif st.session_state.etape == 5:
    st.markdown("## 👁️ Étape 5: Analyse par Vision")
    st.markdown("---")
    
    st.info("Simulation d'analyse du regard - Dans la version réelle, une caméra suivrait les mouvements oculaires.")
    
    metrics_vision = {
        "👁️ Fixation sur les yeux": random.randint(30, 90),
        "🎯 Attention conjointe": random.randint(40, 95),
        "🔄 Poursuite visuelle": random.randint(50, 100),
        "😊 Reconnaissance émotions": random.randint(35, 85)
    }
    
    for metric, score in metrics_vision.items():
        st.progress(score/100, text=f"{metric}: {score}%")
    
    # Graphique radar
    fig = go.Figure(data=go.Scatterpolar(
        r=list(metrics_vision.values()),
        theta=list(metrics_vision.keys()),
        fill='toself',
        marker=dict(color='rgba(102, 126, 234, 0.8)'),
        line=dict(color='rgba(102, 126, 234, 1)', width=2)
    ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=False,
        height=350,
        margin=dict(l=40, r=40, t=20, b=20)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    score_vision_total = int(np.mean(list(metrics_vision.values())))
    st.session_state.score_vision = score_vision_total
    
    col1, col2, col3 = st.columns([1,1,1])
    with col2:
        if st.button("🔮 Générer le rapport final", use_container_width=True):
            scores = [st.session_state.score_questionnaire, st.session_state.score_audio, st.session_state.score_vision]
            st.session_state.score_global = int(np.mean(scores))
            st.session_state.pourcentage = (st.session_state.score_global / 20) * 100
            
            # Déterminer le risque
            if st.session_state.pourcentage >= 70:
                st.session_state.niveau = "🔴 Risque Élevé"
                st.session_state.recommandation = "Consultation avec un spécialiste recommandée dès que possible."
            elif st.session_state.pourcentage >= 50:
                st.session_state.niveau = "🟠 Risque Modéré"
                st.session_state.recommandation = "Surveillance attentive et consultation recommandée."
            elif st.session_state.pourcentage >= 30:
                st.session_state.niveau = "🟡 Risque Faible"
                st.session_state.recommandation = "Continuer à observer le développement normal."
            else:
                st.session_state.niveau = "🟢 Risque Très Faible"
                st.session_state.recommandation = "Développement typique, continuez le suivi normal."
            
            st.session_state.etape = 6
            st.rerun()

# ========== ÉTAPE 6: RAPPORT FINAL ==========
elif st.session_state.etape == 6:
    st.markdown("## 📊 Résultat NeuroSense AI+")
    st.markdown("---")
    
    # Calcul du pourcentage
    pourcentage_final = (st.session_state.score_global / 20) * 100
    
    # Déterminer la classe CSS
    if pourcentage_final >= 70:
        risk_class = "risk-high"
    elif pourcentage_final >= 50:
        risk_class = "risk-moderate"
    elif pourcentage_final >= 30:
        risk_class = "risk-low"
    else:
        risk_class = "risk-very-low"
    
    # Carte de résultat
    st.markdown(f"""
    <div class="result-card {risk_class}">
        <div style="font-size: 4rem;">{'⚠️' if pourcentage_final >= 50 else '✅'}</div>
        <h2 style="margin:10px 0;">{st.session_state.niveau}</h2>
        <div style="font-size: 3rem; font-weight: bold; margin:20px 0;">
            {pourcentage_final:.1f}%
        </div>
        <div style="background: rgba(255,255,255,0.2); border-radius: 20px; padding: 15px; margin-top: 10px;">
            <p style="margin:5px 0;"><strong>Score questionnaire:</strong> {st.session_state.score_questionnaire}/20</p>
            <p style="margin:5px 0;"><strong>Score analyse vocale:</strong> {st.session_state.score_audio}/100</p>
            <p style="margin:5px 0;"><strong>Score analyse vision:</strong> {st.session_state.score_vision}/100</p>
            <p style="margin:5px 0;"><strong>Score global:</strong> {st.session_state.score_global}/20</p>
        </div>
        <p style="margin-top:20px; font-size:1.1rem;">
            📌 {st.session_state.recommandation}
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Résumé visuel
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📈 Scores détaillés")
        scores_df = pd.DataFrame({
            'Catégorie': ['Questionnaire', 'Analyse vocale', 'Analyse vision'],
            'Score (%)': [
                (st.session_state.score_questionnaire / 20) * 100,
                st.session_state.score_audio,
                st.session_state.score_vision
            ]
        })
        
        fig = px.bar(scores_df, x='Catégorie', y='Score (%)', 
                     color='Catégorie',
                     color_discrete_sequence=['#667eea', '#f093fb', '#4ecdc4'],
                     title="Comparaison des scores")
        fig.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 📋 Recommandations")
        if pourcentage_final >= 70:
            st.error("""
            **🔴 Action immédiate:**
            - Consultez un pédiatre spécialisé
            - Programme d'intervention précoce
            """)
        elif pourcentage_final >= 40:
            st.warning("""
            **🟠 Surveillance active:**
            - Consultez un médecin
            - Stimulation du développement
            """)
        else:
            st.success("""
            **🟢 Développement typique:**
            - Continuez les activités stimulantes
            - Suivi normal
            """)
    
    # Boutons
    col1, col2, col3 = st.columns([1,1,1])
    with col2:
        if st.button("🔄 Nouvelle évaluation", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

# ========== PIED DE PAGE ==========
st.markdown("""
<div class="footer">
    <p>🧠 NeuroSense AI+ | 🤖 IA pour la détection précoce</p>
    <p>⚠️ Ceci est un outil d'aide à la décision, pas un diagnostic médical</p>
</div>
""", unsafe_allow_html=True)
