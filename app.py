import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import random
import plotly.express as px
import plotly.graph_objects as go
import audio_recorder_streamlit as ar
from PIL import Image
import cv2

# ========== PAGE CONFIGURATION ==========
st.set_page_config(
    page_title="NeuroSense AI+ - Détection précoce de l'autisme",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ========== STYLE CSS ==========
st.markdown("""
<style>
    .main { background: linear-gradient(135deg, #f5f7fa 0%, #e4e8f0 100%); }
    .result-card {
        border-radius: 20px;
        padding: 25px;
        margin: 20px 0;
        text-align: center;
        animation: fadeIn 0.5s ease-in;
    }
    .risk-high { background: linear-gradient(135deg, #ff6b6b, #ee5a24); color: white; }
    .risk-moderate { background: linear-gradient(135deg, #feca57, #ff9f43); color: white; }
    .risk-low { background: linear-gradient(135deg, #48dbfb, #0abde3); color: white; }
    .risk-very-low { background: linear-gradient(135deg, #10ac84, #1dd1a1); color: white; }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        border-radius: 20px;
        padding: 30px;
        text-align: center;
        color: white;
        margin-bottom: 30px;
    }
    .progress-bar {
        background: linear-gradient(90deg, #667eea, #764ba2);
        border-radius: 10px;
        height: 8px;
        transition: width 0.3s ease;
    }
    .footer {
        text-align: center;
        padding: 20px;
        color: #666;
        font-size: 0.8rem;
        border-top: 1px solid #ddd;
        margin-top: 40px;
    }
    .video-placeholder {
        background: #1a1a2e;
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        color: white;
        border: 2px dashed #667eea;
    }
</style>
""", unsafe_allow_html=True)

# ========== INITIALISATION ==========
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
if 'antecedents' not in st.session_state:
    st.session_state.antecedents = ""
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
if 'audio_uploaded' not in st.session_state:
    st.session_state.audio_uploaded = False
if 'video_uploaded' not in st.session_state:
    st.session_state.video_uploaded = False

# ========== EN-TÊTE ==========
st.markdown("""
<div class="header">
    <h1 style="font-size: 3rem; margin:0;">🧠 NeuroSense AI+</h1>
    <p style="font-size: 1.2rem; margin-top:10px;">
        Détection précoce intelligente de l'autisme par questionnaire, analyse vocale et vision
    </p>
</div>
""", unsafe_allow_html=True)

# ========== PROGRESSION ==========
if st.session_state.etape > 1 and st.session_state.etape < 7:
    progression = (st.session_state.etape - 1) / 6 * 100
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

# ========== ÉTAPE 2: INFORMATIONS (MODIFIÉE) ==========
elif st.session_state.etape == 2:
    st.markdown("## 👤 Étape 2: Vos informations")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.session_state.nom_parent = st.text_input("📝 Votre nom complet", placeholder="Ex: Marie Dupont")
        st.session_state.age_parent = st.number_input("🎂 Votre âge", min_value=18, max_value=100, step=1)
        st.session_state.nom_enfant = st.text_input("👶 Nom de l'enfant", placeholder="Ex: Lucas")
    
    with col2:
        st.session_state.age_enfant = st.number_input("📅 Âge de l'enfant (en mois)", min_value=0, max_value=72, step=1,
                                                        help="Pour les enfants de 0 à 6 ans (72 mois)")
        st.session_state.sexe_enfant = st.selectbox("⚥ Sexe de l'enfant", ["", "Masculin", "Féminin"])
    
    # Antécédents familiaux
    st.markdown("---")
    st.markdown("### 🧬 Antécédents familiaux")
    
    antecedents_options = st.multiselect(
        "Sélectionnez les antécédents familiaux présents:",
        ["Autisme (diagnostiqué)", "Autisme (suspicion)", "TDAH", "Trouble du langage", 
         "Retard de développement", "Épilepsie", "Trouble anxieux", "Aucun antécédent connu"]
    )
    
    autre_antecedent = st.text_input("Autre (précisez):", placeholder="Ex: Syndrome de Asperger")
    
    if autre_antecedent:
        antecedents_options.append(autre_antecedent)
    
    st.session_state.antecedents = ", ".join(antecedents_options) if antecedents_options else "Aucun signalé"
    
    col1, col2, col3 = st.columns([1,1,1])
    with col2:
        if st.button("➡️ Suivant", use_container_width=True):
            if st.session_state.nom_parent and st.session_state.age_parent > 0 and st.session_state.nom_enfant:
                st.session_state.etape = 3
                st.rerun()
            else:
                st.error("Veuillez remplir tous les champs obligatoires (Nom parent, Âge, Nom enfant)")

# ========== ÉTAPE 3: QUESTIONNAIRE ==========
elif st.session_state.etape == 3:
    st.markdown("## 📝 Étape 3: Questionnaire d'évaluation")
    st.markdown("---")
    st.info("Veuillez répondre aux questions suivantes en fonction du comportement de votre enfant au cours des 3 derniers mois.")
    
    questions = [
        "👁️ Votre enfant regarde-t-il dans les yeux?",
        "🔊 Votre enfant réagit-il quand on appelle son nom?",
        "👉 Votre enfant pointe-t-il du doigt pour montrer quelque chose?",
        "🧸 Votre enfant joue-t-il à faire semblant? (ex: donner à manger à une poupée)",
        "🚫 Votre enfant évite-t-il le contact visuel?",
        "🔄 Votre enfant a-t-il des comportements répétitifs? (ex: se balance, tourne en rond)",
        "😊 Votre enfant partage-t-il son plaisir avec vous? (ex: vous montre un jouet)",
        "😢 Votre enfant semble-t-il insensible à la douleur?",
        "👂 Votre enfant a-t-il des sensibilités aux bruits ou textures?",
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

# ========== ÉTAPE 4: ANALYSE AUDIO & VIDÉO (AJOUTÉE) ==========
elif st.session_state.etape == 4:
    st.markdown("## 🎤 Étape 4: Analyse Audio et Vidéo")
    st.markdown("---")
    
    st.info("""
    ### 📋 Instructions:
    1. **Analyse Audio** : Enregistrez ou téléchargez la voix de votre enfant
    2. **Analyse Vidéo** : Enregistrez ou téléchargez une vidéo de votre enfant jouant ou interagissant
    """)
    
    col1, col2 = st.columns(2)
    
    # ========== COLONNE 1: ANALYSE AUDIO ==========
    with col1:
        st.markdown("### 🎙️ Analyse Audio")
        
        tab1, tab2 = st.tabs(["🎤 Enregistrer", "📁 Télécharger"])
        
        with tab1:
            st.markdown("**Enregistrement vocal:**")
            audio_bytes = ar.audio_recorder(
                text="Cliquez pour enregistrer",
                recording_color="#ff0000",
                neutral_color="#667eea",
                icon_size="2x"
            )
            
            if audio_bytes:
                st.audio(audio_bytes, format="audio/wav")
                st.success("✅ Audio enregistré avec succès!")
                st.session_state.audio_uploaded = True
        
        with tab2:
            st.markdown("**Télécharger un fichier audio:**")
            audio_file = st.file_uploader("Choisir un fichier", type=["wav", "mp3", "m4a"], key="audio")
            if audio_file:
                st.audio(audio_file)
                st.success(f"✅ Fichier chargé: {audio_file.name}")
                st.session_state.audio_uploaded = True
        
        # Simulation du score audio
        if st.session_state.audio_uploaded:
            st.markdown("---")
            st.markdown("**📊 Analyse en temps réel:**")
            
            with st.spinner("Analyse de la voix en cours..."):
                import time
                time.sleep(1.5)
            
            # Simulation des scores
            scores_audio = {
                "Prosodie (intonation)": random.randint(40, 95),
                "Clarté articulatoire": random.randint(45, 98),
                "Variabilité vocale": random.randint(50, 100),
                "Réponse sonore": random.randint(35, 90)
            }
            
            for metric, score in scores_audio.items():
                st.progress(score/100, text=f"{metric}: {score}%")
                time.sleep(0.3)
            
            score_audio_total = int(np.mean(list(scores_audio.values())))
            st.session_state.score_audio = score_audio_total
            
            st.metric("🎵 Score vocal global", f"{score_audio_total}%")
    
    # ========== COLONNE 2: ANALYSE VIDÉO ==========
    with col2:
        st.markdown("### 🎥 Analyse Vidéo")
        
        tab3, tab4 = st.tabs(["🎥 Enregistrer", "📁 Télécharger"])
        
        with tab3:
            st.markdown("**Enregistrement vidéo (Eye Tracking):**")
            
            # Simuler une capture vidéo (Webcam)
            use_webcam = st.checkbox("Utiliser la webcam", key="use_webcam")
            
            if use_webcam:
                st.warning("⚠️ Dans un environnement réel, la webcam s'activerait pour suivre les mouvements oculaires.")
                
                # Placeholder pour la webcam
                st.markdown("""
                <div class="video-placeholder">
                    <div style="font-size: 3rem;">🎥</div>
                    <p>📷 Webcam activée - Tracking des yeux en cours...</p>
                    <p style="font-size: 0.8rem;">👁️ Points de fixation: 12 | 👀 Temps de regard: 3.2s</p>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button("📸 Capturer et analyser", key="capture_btn"):
                    with st.spinner("Analyse du regard en cours..."):
                        import time
                        time.sleep(2)
                    st.success("Analyse terminée!")
                    st.session_state.video_uploaded = True
        
        with tab4:
            st.markdown("**Télécharger une vidéo:**")
            video_file = st.file_uploader("Choisir une vidéo", type=["mp4", "avi", "mov"], key="video")
            if video_file:
                st.video(video_file)
                st.success(f"✅ Vidéo chargée: {video_file.name}")
                
                if st.button("🔍 Analyser la vidéo", key="analyze_video"):
                    with st.spinner("Analyse des mouvements et du regard en cours..."):
                        import time
                        time.sleep(2)
                    st.success("Analyse terminée!")
                    st.session_state.video_uploaded = True
        
        # Simulation du score vidéo
        if st.session_state.video_uploaded:
            st.markdown("---")
            st.markdown("**📊 Analyse Eye Tracking:**")
            
            with st.spinner("Analyse des mouvements oculaires en cours..."):
                import time
                time.sleep(1.5)
            
            scores_vision = {
                "👁️ Fixation sur les yeux": random.randint(30, 90),
                "🎯 Attention conjointe": random.randint(40, 95),
                "🔄 Poursuite visuelle": random.randint(50, 100),
                "😊 Reconnaissance émotions": random.randint(35, 85)
            }
            
            # Graphique radar
            fig = go.Figure(data=go.Scatterpolar(
                r=list(scores_vision.values()),
                theta=list(scores_vision.keys()),
                fill='toself',
                marker=dict(color='rgba(102, 126, 234, 0.8)'),
                line=dict(color='rgba(102, 126, 234, 1)', width=2)
            ))
            
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                showlegend=False,
                height=300,
                margin=dict(l=40, r=40, t=20, b=20)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            for metric, score in scores_vision.items():
                st.progress(score/100, text=f"{metric}: {score}%")
                time.sleep(0.3)
            
            score_vision_total = int(np.mean(list(scores_vision.values())))
            st.session_state.score_vision = score_vision_total
            
            st.metric("👁️ Score Eye Tracking global", f"{score_vision_total}%")
    
    # Bouton suivant
    st.markdown("---")
    col1, col2, col3 = st.columns([1,1,1])
    with col2:
        if st.button("➡️ Générer le rapport final", use_container_width=True):
            if st.session_state.audio_uploaded and st.session_state.video_uploaded:
                # Calcul du score global
                if st.session_state.score_audio and st.session_state.score_vision and st.session_state.score_questionnaire:
                    scores = [st.session_state.score_questionnaire * 5, st.session_state.score_audio, st.session_state.score_vision]
                    st.session_state.score_global = int(np.mean(scores))
                    st.session_state.pourcentage = st.session_state.score_global
                    
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
                    
                    st.session_state.etape = 5
                    st.rerun()
            else:
                st.error("Veuillez compléter l'analyse audio ET l'analyse vidéo avant de continuer")

# ========== ÉTAPE 5: RAPPORT FINAL ==========
elif st.session_state.etape == 5:
    st.markdown("## 📊 Résultat NeuroSense AI+")
    st.markdown("---")
    
    # Déterminer la classe CSS
    if st.session_state.pourcentage >= 70:
        risk_class = "risk-high"
    elif st.session_state.pourcentage >= 50:
        risk_class = "risk-moderate"
    elif st.session_state.pourcentage >= 30:
        risk_class = "risk-low"
    else:
        risk_class = "risk-very-low"
    
    # Carte de résultat
    st.markdown(f"""
    <div class="result-card {risk_class}">
        <div style="font-size: 4rem;">{'⚠️' if st.session_state.pourcentage >= 50 else '✅'}</div>
        <h2 style="margin:10px 0;">{st.session_state.niveau}</h2>
        <div style="font-size: 3rem; font-weight: bold; margin:20px 0;">
            {st.session_state.pourcentage:.1f}%
        </div>
        <div style="background: rgba(255,255,255,0.2); border-radius: 20px; padding: 15px; margin-top: 10px;">
            <p style="margin:5px 0;"><strong>Score questionnaire:</strong> {st.session_state.score_questionnaire}/20</p>
            <p style="margin:5px 0;"><strong>Score analyse vocale:</strong> {st.session_state.score_audio}%</p>
            <p style="margin:5px 0;"><strong>Score analyse vision:</strong> {st.session_state.score_vision}%</p>
            <p style="margin:5px 0;"><strong>Score global:</strong> {st.session_state.score_global}%</p>
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
            'Catégorie': ['Questionnaire', 'Analyse vocale', 'Eye Tracking'],
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
        st.markdown("### 📋 Détails de l'évaluation")
        st.write(f"**Enfant:** {st.session_state.nom_enfant}")
        st.write(f"**Âge:** {st.session_state.age_enfant} mois")
        st.write(f"**Sexe:** {st.session_state.sexe_enfant}")
        st.write(f"**Antécédents:** {st.session_state.antecedents}")
    
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
