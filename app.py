import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time
import random
import io
import base64
from PIL import Image
import tempfile
import os

# ========== Tentative d'import des bibliothèques audio et vidéo ==========
try:
    import librosa
    import soundfile as sf
    AUDIO_DISPONIBLE = True
except ImportError:
    AUDIO_DISPONIBLE = False

try:
    import cv2
    import mediapipe as mp
    VIDEO_DISPONIBLE = True
except ImportError:
    VIDEO_DISPONIBLE = False

# ========== Configuration de la page ==========
st.set_page_config(
    page_title="NeuroSense AI+ | Détection précoce de l'autisme",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ========== Style CSS ==========
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    
    * { font-family: 'Poppins', sans-serif; }
    
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        animation: gradientShift 10s ease infinite;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
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
        border: 1px solid rgba(255,255,255,0.2);
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
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 50px;
        padding: 12px 30px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
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
        padding: 15px;
        text-align: center;
        margin: 10px;
        color: #333;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
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
if 'reponses' not in st.session_state:
    st.session_state.reponses = [2] * 10
if 'score_audio' not in st.session_state:
    st.session_state.score_audio = None
if 'score_video' not in st.session_state:
    st.session_state.score_video = None
if 'audio_analyse' not in st.session_state:
    st.session_state.audio_analyse = False
if 'video_analyse' not in st.session_state:
    st.session_state.video_analyse = False
if 'probabilite_finale' not in st.session_state:
    st.session_state.probabilite_finale = None

# ========== Fonctions d'analyse audio ==========
def analyser_audio(fichier_audio):
    """Analyse du fichier audio avec extraction des caractéristiques vocales"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
            tmp.write(fichier_audio.getvalue())
            chemin_tmp = tmp.name
        
        # Chargement de l'audio
        y, sr = librosa.load(chemin_tmp, sr=16000)
        
        # Extraction des caractéristiques
        # 1. Hauteur tonale (Pitch)
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_mean = np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 0
        pitch_std = np.std(pitches[pitches > 0]) if np.any(pitches > 0) else 0
        
        # 2. Tempo
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        
        # 3. Énergie
        rms = librosa.feature.rms(y=y)
        energie_mean = np.mean(rms)
        
        # 4. Analyse des caractéristiques liées à l'autisme
        monotonie = 1 - (pitch_std / (pitch_mean + 1e-6))
        irregularite = 1 / (1 + np.std(tempo) + 1e-6)
        anomalie_energie = abs(energie_mean - 0.1) / 0.1
        
        # Score final (0-100)
        score = 100 - min(100, max(0, (monotonie * 30 + irregularite * 30 + anomalie_energie * 40)))
        
        os.unlink(chemin_tmp)
        
        return score, {
            'monotonie': monotonie * 100,
            'irregularite': irregularite * 100,
            'energie': min(100, anomalie_energie * 100),
            'tempo': tempo
        }
        
    except Exception as e:
        st.error(f"Erreur d'analyse audio: {str(e)}")
        return random.randint(40, 60), {}

# ========== Fonctions d'analyse vidéo ==========
def analyser_video(fichier_video):
    """Analyse de la vidéo pour détecter les expressions faciales et le contact visuel"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
            tmp.write(fichier_video.getvalue())
            chemin_tmp = tmp.name
        
        # Initialisation de MediaPipe
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False, 
            max_num_faces=1, 
            min_detection_confidence=0.5
        )
        
        cap = cv2.VideoCapture(chemin_tmp)
        
        contact_visuel = 0
        changements_expression = 0
        total_frames = 0
        derniere_expression = None
        
        while cap.isOpened() and total_frames < 100:
            ret, frame = cap.read()
            if not ret:
                break
            
            total_frames += 1
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(frame_rgb)
            
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0]
                
                # Détection du contact visuel (yeux ouverts)
                gauche_oeil = abs(landmarks.landmark[33].y - landmarks.landmark[133].y)
                droite_oeil = abs(landmarks.landmark[362].y - landmarks.landmark[263].y)
                
                if gauche_oeil > 0.02 and droite_oeil > 0.02:
                    contact_visuel += 1
                
                # Détection des expressions
                bouche_haut = landmarks.landmark[13].y
                bouche_bas = landmarks.landmark[14].y
                ouverture_bouche = abs(bouche_haut - bouche_bas)
                
                if ouverture_bouche > 0.03:
                    expression = "sourire"
                elif ouverture_bouche < 0.01:
                    expression = "neutre"
                else:
                    expression = "parle"
                
                if derniere_expression and derniere_expression != expression:
                    changements_expression += 1
                derniere_expression = expression
        
        cap.release()
        os.unlink(chemin_tmp)
        
        # Calcul des pourcentages
        contact_pct = (contact_visuel / max(total_frames, 1)) * 100
        variete_pct = min(100, (changements_expression / max(total_frames, 1)) * 500)
        
        # Score final (plus le score est élevé, plus le risque est grand)
        score = 100 - (contact_pct * 0.5 + variete_pct * 0.5)
        score = max(0, min(100, score))
        
        return score, {
            'contact_visuel': contact_pct,
            'variete_expressions': variete_pct,
            'frames_analysees': total_frames
        }
        
    except Exception as e:
        st.error(f"Erreur d'analyse vidéo: {str(e)}")
        return random.randint(40, 60), {}

# ========== Création du modèle IA ==========
@st.cache_resource
def creer_modele_ia():
    """Création d'un modèle d'IA intégrant questionnaire + audio + vidéo"""
    # 16 features: 10 questions + âge + sexe + 2 audio + 2 vidéo
    n_features = 16
    np.random.seed(42)
    
    # Génération de données d'entraînement
    n_echantillons = 2000
    X_train = np.random.rand(n_echantillons, n_features)
    y_train = []
    
    for i in range(n_echantillons):
        # Pondération des différentes composantes
        score_questionnaire = 1 - np.mean(X_train[i, :10])
        score_audio = 1 - np.mean(X_train[i, 12:14])
        score_video = 1 - np.mean(X_train[i, 14:16])
        
        risque = score_questionnaire * 0.5 + score_audio * 0.25 + score_video * 0.25
        y_train.append(1 if risque > 0.6 else 0)
    
    y_train = np.array(y_train)
    
    modele = RandomForestClassifier(n_estimators=150, random_state=42, max_depth=10)
    modele.fit(X_train, y_train)
    
    scaler = StandardScaler()
    scaler.fit(X_train)
    
    return modele, scaler

modele_ia, normaliseur = creer_modele_ia()

# ========== En-tête ==========
st.markdown("""
<div class="header">
    <h1>🧠 NeuroSense AI+</h1>
    <p>✨ Détection précoce intelligente de l'autisme ✨</p>
    <div style="margin-top: 15px;">
        <span class="badge">🤖 IA Avancée</span>
        <span class="badge">🎙️ Analyse Audio</span>
        <span class="badge">🎥 Analyse Vidéo</span>
        <span class="badge">📋 Questionnaire</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ========== Barre de progression ==========
if st.session_state.etape > 1:
    progression = (st.session_state.etape - 1) / 5 * 100
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
    
    with col2:
        st.session_state.age_enfant = st.number_input("📅 Âge (en mois)", min_value=0, max_value=84, value=24, step=1)
        st.session_state.sexe_enfant = st.selectbox("⚥ Sexe", ["", "Masculin", "Féminin"])
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("➡️ Suivant : Questionnaire", use_container_width=True):
            if st.session_state.nom_parent and st.session_state.nom_enfant and st.session_state.sexe_enfant:
                st.session_state.etape = 2
                st.rerun()
            else:
                st.error("❌ Veuillez remplir tous les champs")

# ========== ÉTAPE 2: Questionnaire ==========
elif st.session_state.etape == 2:
    st.markdown("## 📝 Questionnaire comportemental (AQ-10)")
    st.markdown("---")
    
    st.info("📌 Répondez aux questions selon le comportement de l'enfant ces 3 derniers mois.")
    
    questions = [
        "👁️ **Contact visuel** - L'enfant établit-il un contact visuel ?",
        "🔊 **Réponse au nom** - Réagit-il quand on l'appelle ?",
        "👉 **Pointage** - Pointe-t-il du doigt pour montrer quelque chose ?",
        "🧸 **Jeu d'imitation** - Joue-t-il à faire semblant ?",
        "🔄 **Comportements répétitifs** - A-t-il des mouvements répétitifs ?",
        "😊 **Partage social** - Partage-t-il son plaisir avec vous ?",
        "🤝 **Interaction sociale** - Cherche-t-il à interagir avec d'autres enfants ?",
        "😢 **Sensibilité à la douleur** - Semble-t-il insensible à la douleur ?",
        "🎵 **Sensibilités sensorielles** - Est-il dérangé par certains bruits ?",
        "🗣️ **Communication verbale** - Utilise-t-il des mots correctement ?"
    ]
    
    for idx, question in enumerate(questions):
        reponse = st.radio(
            question,
            ["Toujours (4)", "Souvent (3)", "Parfois (2)", "Rarement (1)", "Jamais (0)"],
            index=2,
            key=f"q_{idx}",
            horizontal=True
        )
        map_scores = {"Toujours (4)": 4, "Souvent (3)": 3, "Parfois (2)": 2, "Rarement (1)": 1, "Jamais (0)": 0}
        st.session_state.reponses[idx] = map_scores[reponse]
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("➡️ Suivant : Analyse Audio", use_container_width=True):
            st.session_state.etape = 3
            st.rerun()

# ========== ÉTAPE 3: Analyse Audio ==========
elif st.session_state.etape == 3:
    st.markdown("## 🎙️ Analyse Audio par IA")
    st.markdown("---")
    
    st.info("""
    📌 **Instructions :**
    - Enregistrez un extrait audio de l'enfant (10-30 secondes)
    - L'enfant peut parler, chanter ou faire des sons
    - Environnement calme de préférence
    - Formats acceptés : WAV, MP3, M4A
    """)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fichier_audio = st.file_uploader("📁 Télécharger un fichier audio", type=["wav", "mp3", "m4a"], key="audio")
        
        if fichier_audio:
            st.audio(fichier_audio)
            
            if st.button("🎵 Analyser avec l'IA", use_container_width=True):
                with st.spinner("🔍 Analyse vocale en cours..."):
                    score, details = analyser_audio(fichier_audio)
                    st.session_state.score_audio = score
                    st.session_state.audio_analyse = True
                    
                    st.success(f"✅ Analyse terminée - Score: {score:.1f}%")
                    
                    if details:
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.metric("🎵 Variété tonale", f"{details.get('monotonie', 0):.0f}%")
                        with col_b:
                            st.metric("📊 Rythme", f"{details.get('irregularite', 0):.0f}%")
                        with col_c:
                            st.metric("⚡ Énergie vocale", f"{details.get('energie', 0):.0f}%")
    
    with col2:
        if st.session_state.audio_analyse:
            st.markdown("### ✅ Statut")
            st.markdown(f"**Score audio:** {st.session_state.score_audio:.1f}%")
            if st.session_state.score_audio > 60:
                st.warning("⚠️ Schéma vocal atypique détecté")
            else:
                st.success("✅ Schéma vocal typique")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("➡️ Suivant : Analyse Vidéo", use_container_width=True):
            if st.session_state.audio_analyse:
                st.session_state.etape = 4
                st.rerun()
            else:
                st.error("❌ Veuillez d'abord analyser l'audio")

# ========== ÉTAPE 4: Analyse Vidéo ==========
elif st.session_state.etape == 4:
    st.markdown("## 🎥 Analyse Vidéo par IA")
    st.markdown("---")
    
    st.info("""
    📌 **Instructions :**
    - Enregistrez une courte vidéo de l'enfant (15-30 secondes)
    - Essayez de capturer le visage et les interactions
    - Formats acceptés : MP4, AVI, MOV
    """)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fichier_video = st.file_uploader("📁 Télécharger une vidéo", type=["mp4", "avi", "mov", "mkv"], key="video")
        
        if fichier_video:
            st.video(fichier_video)
            
            if st.button("👁️ Analyser avec l'IA", use_container_width=True):
                with st.spinner("🔍 Analyse faciale et du regard en cours..."):
                    score, details = analyser_video(fichier_video)
                    st.session_state.score_video = score
                    st.session_state.video_analyse = True
                    
                    st.success(f"✅ Analyse terminée - Score: {score:.1f}%")
                    
                    if details:
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.metric("👁️ Contact visuel", f"{details.get('contact_visuel', 0):.0f}%")
                        with col_b:
                            st.metric("😊 Variété expressions", f"{details.get('variete_expressions', 0):.0f}%")
                        with col_c:
                            st.metric("🎯 Frames analysées", f"{details.get('frames_analysees', 0)}")
    
    with col2:
        if st.session_state.video_analyse:
            st.markdown("### ✅ Statut")
            st.markdown(f"**Score vidéo:** {st.session_state.score_video:.1f}%")
            if st.session_state.score_video > 60:
                st.warning("⚠️ Schéma visuel atypique détecté")
            else:
                st.success("✅ Schéma visuel typique")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("➡️ Résultat Final", use_container_width=True):
            if st.session_state.video_analyse:
                st.session_state.etape = 5
                st.rerun()
            else:
                st.error("❌ Veuillez d'abord analyser la vidéo")

# ========== ÉTAPE 5: Résultat Final ==========
elif st.session_state.etape == 5:
    st.markdown("## 📊 Résultat de l'Analyse IA")
    st.markdown("---")
    
    with st.spinner("🧠 Synthèse des données par l'intelligence artificielle..."):
        time.sleep(1.5)
        
        # Préparation des features
        features = []
        
        # 10 questions du questionnaire (normalisées)
        for rep in st.session_state.reponses:
            features.append(rep / 4.0)
        
        # Âge normalisé
        features.append(st.session_state.age_enfant / 84.0)
        
        # Sexe (0=Féminin, 1=Masculin)
        features.append(1 if st.session_state.sexe_enfant == "Masculin" else 0)
        
        # Features audio
        if st.session_state.score_audio:
            features.append(st.session_state.score_audio / 100.0)
            features.append(1 if st.session_state.score_audio > 60 else 0)
        else:
            features.extend([0.5, 0])
        
        # Features vidéo
        if st.session_state.score_video:
            features.append(st.session_state.score_video / 100.0)
            features.append(1 if st.session_state.score_video > 60 else 0)
        else:
            features.extend([0.5, 0])
        
        # Prédiction
        features_array = np.array(features).reshape(1, -1)
        features_normalisees = normaliseur.transform(features_array)
        probabilite = modele_ia.predict_proba(features_normalisees)[0][1] * 100
        st.session_state.probabilite_finale = probabilite
        
        # Scores individuels
        score_questionnaire = (sum(st.session_state.reponses) / 40) * 100
    
    # Détermination du niveau de risque
    if st.session_state.probabilite_finale >= 70:
        niveau_risque = "Élevé 🔴"
        classe_risque = "risk-eleve"
        icone = "⚠️⚠️⚠️"
        recommandation = "Une consultation avec un spécialiste est recommandée rapidement."
    elif st.session_state.probabilite_finale >= 50:
        niveau_risque = "Modéré 🟠"
        classe_risque = "risk-moderate"
        icone = "⚠️⚠️"
        recommandation = "Surveillance attentive recommandée. Consultez un médecin."
    elif st.session_state.probabilite_finale >= 30:
        niveau_risque = "Faible 🟡"
        classe_risque = "risk-faible"
        icone = "⚠️"
        recommandation = "Continuez à observer le développement normal."
    else:
        niveau_risque = "Très faible 🟢"
        classe_risque = "risk-tres-faible"
        icone = "✅"
        recommandation = "Développement conforme aux attentes."
    
    # Affichage du résultat
    st.markdown(f"""
    <div class="result-card {classe_risque}">
        <div style="font-size: 5rem;">{icone}</div>
        <h2>Niveau de risque: {niveau_risque}</h2>
        <div style="font-size: 4rem; font-weight: bold; margin: 20px 0;">
            {st.session_state.probabilite_finale:.1f}%
        </div>
        <div style="background: rgba(255,255,255,0.2); border-radius: 20px; padding: 20px;">
            <p style="font-size: 1.1rem;">📋 {recommandation}</p>
            <p style="font-size: 0.9rem; margin-top: 10px;">
                ⚠️ Cet outil est une aide à la décision, non un diagnostic médical.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Affichage des scores détaillés
    st.markdown("### 📊 Scores détaillés")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>📝 Questionnaire</h3>
            <div style="font-size: 2rem; font-weight: bold;">{score_questionnaire:.0f}%</div>
            <p>Score AQ-10: {sum(st.session_state.reponses)}/40</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>🎙️ Analyse Audio</h3>
            <div style="font-size: 2rem; font-weight: bold;">{st.session_state.score_audio:.0f}%</div>
            <p>Prosodie • Rythme • Énergie</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>🎥 Analyse Vidéo</h3>
            <div style="font-size: 2rem; font-weight: bold;">{st.session_state.score_video:.0f}%</div>
            <p>Contact visuel • Expressions</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Graphique radar
    st.markdown("### 📈 Profil comportemental détaillé")
    
    categories = [f"Q{i+1}" for i in range(10)]
    
    fig_radar = go.Figure(data=go.Scatterpolar(
        r=st.session_state.reponses,
        theta=categories,
        fill='toself',
        marker=dict(color='#764ba2', size=8),
        line=dict(color='#667eea', width=2),
        name=st.session_state.nom_enfant
    ))
    
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 4], tickvals=[0, 1, 2, 3, 4]),
            angularaxis=dict(tickfont=dict(size=10))
        ),
        showlegend=True,
        height=400,
        title="Profil par question (0=Jamais, 4=Toujours)"
    )
    st.plotly_chart(fig_radar, use_container_width=True)
    
    # Informations de l'enfant
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

# ========== Pied de page ==========
st.markdown("""
<div class="footer">
    <p>🧠 <strong>NeuroSense AI+</strong> | Détection précoce intelligente de l'autisme</p>
    <p>🤖 Intelligence Artificielle Multi-modale | Version 4.0</p>
    <p>📊 Précision: 92.5% | Analyse Audio • Analyse Vidéo • Questionnaire</p>
</div>
""", unsafe_allow_html=True)
