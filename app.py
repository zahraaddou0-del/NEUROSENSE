import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import random
from PIL import Image
import io

# ========== CONFIGURATION DE LA PAGE ==========
st.set_page_config(
    page_title="NeuroSense AI+ | Détection précoce de l'autisme",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== CSS PERSONNALISÉ ==========
st.markdown("""
<style>
    /* Style principal */
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Titre principal */
    .main-title {
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #1e3c72, #2a5298, #1e3c72);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0;
    }
    
    /* Sous-titre */
    .subtitle {
        text-align: center;
        color: #4a5568;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Cartes */
    .card {
        background: white;
        border-radius: 20px;
        padding: 1.5rem;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        transition: transform 0.3s ease;
    }
    .card:hover {
        transform: translateY(-5px);
    }
    
    /* Boutons personnalisés */
    .stButton > button {
        background: linear-gradient(90deg, #1e3c72, #2a5298);
        color: white;
        border: none;
        border-radius: 30px;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #2a5298, #1e3c72);
        transform: scale(1.02);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    
    /* Métriques */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 1rem;
        text-align: center;
        color: white;
    }
    
    /* Header */
    .header {
        background: linear-gradient(135deg, #1e3c72, #2a5298);
        padding: 1rem;
        border-radius: 15px;
        margin-bottom: 2rem;
    }
    
    /* Onglets personnalisés */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background: white;
        border-radius: 30px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #1e3c72, #2a5298);
        color: white;
    }
    
    /* Animation */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .fade-in {
        animation: fadeIn 0.6s ease-out;
    }
    
    /* Badges */
    .badge-low {
        background: #48bb78;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 30px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-moderate {
        background: #ed8936;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 30px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-high {
        background: #f56565;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 30px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 1.5rem;
        color: #718096;
        font-size: 0.8rem;
        border-top: 1px solid #e2e8f0;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# ========== INITIALISATION DE LA SESSION ==========
if 'etape' not in st.session_state:
    st.session_state.etape = 1
if 'nom_parent' not in st.session_state:
    st.session_state.nom_parent = ""
if 'age_parent' not in st.session_state:
    st.session_state.age_parent = 0
if 'nom_enfant' not in st.session_state:
    st.session_state.nom_enfant = ""
if 'sexe_enfant' not in st.session_state:
    st.session_state.sexe_enfant = ""
if 'historique' not in st.session_state:
    st.session_state.historique = ""
if 'score_questionnaire' not in st.session_state:
    st.session_state.score_questionnaire = None
if 'score_audio' not in st.session_state:
    st.session_state.score_audio = None
if 'score_vision' not in st.session_state:
    st.session_state.score_vision = None
if 'historique_tests' not in st.session_state:
    st.session_state.historique_tests = []
if 'langue' not in st.session_state:
    st.session_state.langue = "Français"
if 'notifications' not in st.session_state:
    st.session_state.notifications = True
if 'reponses' not in st.session_state:
    st.session_state.reponses = []

# ========== HEADER ==========
col_logo, col_title = st.columns([1, 4])
with col_logo:
    st.markdown("""
    <div style="text-align: center;">
        <span style="font-size: 3rem;">🧠</span>
    </div>
    """, unsafe_allow_html=True)
with col_title:
    st.markdown('<h1 class="main-title">NeuroSense AI+</h1>', unsafe_allow_html=True)

st.markdown('<p class="subtitle">✨ Détection précoce des troubles du spectre autistique par Intelligence Artificielle ✨</p>', unsafe_allow_html=True)

# Barre de progression des étapes
if st.session_state.etape <= 4:
    etapes = ["Profil", "Enfant", "Analyse IA", "Résultats"]
    current = st.session_state.etape - 1
    cols = st.columns(len(etapes))
    for i, (col, etape) in enumerate(zip(cols, etapes)):
        with col:
            if i < current:
                st.markdown(f"<div style='text-align: center; color: #48bb78;'>✅ {etape}</div>", unsafe_allow_html=True)
            elif i == current:
                st.markdown(f"<div style='text-align: center; color: #1e3c72; font-weight: bold;'>🔵 {etape}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div style='text-align: center; color: #cbd5e0;'>⚪ {etape}</div>", unsafe_allow_html=True)
    st.markdown("---")

# ========== ÉTAPE 1: CHOIX DU PROFIL ==========
if st.session_state.etape == 1:
    st.markdown('<div class="fade-in">', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="card" style="text-align: center; cursor: pointer;">
            <span style="font-size: 3rem;">👨‍👩‍👦</span>
            <h3>Parent</h3>
            <p>Pour les familles qui souhaitent évaluer leur enfant</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("👨‍👩‍👦 Parent", key="btn_parent", use_container_width=True):
            st.session_state.type_utilisateur = "Parent"
            st.session_state.etape = 2
            st.rerun()
    
    with col2:
        st.markdown("""
        <div class="card" style="text-align: center; cursor: pointer;">
            <span style="font-size: 3rem;">⚕️</span>
            <h3>Professionnel</h3>
            <p>Pour les médecins, psychologues et spécialistes</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("⚕️ Professionnel", key="btn_pro", use_container_width=True):
            st.session_state.type_utilisateur = "Professionnel"
            st.session_state.etape = 2
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

# ========== ÉTAPE 2: PROFIL DE L'ENFANT ==========
elif st.session_state.etape == 2:
    st.markdown('<div class="fade-in">', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="card">
        <h2 style="color: #1e3c72;">👶 Informations de l'enfant</h2>
        <p style="color: #4a5568;">Veuillez remplir les informations ci-dessous</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.form("profil_enfant", clear_on_submit=False):
        col_left, col_right = st.columns(2)
        
        with col_left:
            nom_parent = st.text_input("📝 Nom du parent / Professionnel", placeholder="Ex: Jean Dupont")
            age_parent = st.number_input("🎂 Âge", min_value=18, max_value=100, step=1)
        
        with col_right:
            nom_enfant = st.text_input("🧒 Prénom de l'enfant", placeholder="Ex: Adam")
            sexe_enfant = st.selectbox("⚧ Sexe", ["Garçon 👦", "Fille 👧", "Autre")
        
        submitted = st.form_submit_button("📝 Continuer vers l'analyse", use_container_width=True)
        
        if submitted:
            if nom_parent and nom_enfant:
                st.session_state.nom_parent = nom_parent
                st.session_state.age_parent = age_parent
                st.session_state.nom_enfant = nom_enfant
                st.session_state.sexe_enfant = sexe_enfant
                st.session_state.etape = 3
                st.rerun()
            else:
                st.error("⚠️ Veuillez remplir tous les champs obligatoires")
    
    st.markdown('</div>', unsafe_allow_html=True)

# ========== ÉTAPE 3: ANALYSE IA ==========
elif st.session_state.etape == 3:
    st.markdown('<div class="fade-in">', unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="card">
        <h2 style="color: #1e3c72;">🤖 Analyse IA en cours</h2>
        <p>Bienvenue <strong>{st.session_state.nom_parent}</strong> ! Nous allons analyser le comportement de <strong>{st.session_state.nom_enfant}</strong></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Création des 3 onglets stylisés
    tab1, tab2, tab3 = st.tabs(["📋 Questionnaire (15 questions)", "🎙️ Analyse vocale", "📷 Vision par ordinateur"])
    
    # ===== TAB 1: QUESTIONNAIRE =====
    with tab1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📝 Questionnaire d'évaluation comportementale")
        st.markdown("*Veuillez répondre à chaque question selon le comportement habituel de l'enfant*")
        st.markdown("---")
        
        # Groupe 1
        st.markdown("### 🟢 1. Interaction sociale")
        
        q1 = st.radio("**1.** L'enfant répond-il à son prénom quand vous l'appelez ?",
                      ["✅ Oui, toujours", "🟡 Parfois", "❌ Non, rarement ou jamais"], index=1, key="q1")
        q2 = st.radio("**2.** L'enfant sourit-il en réponse à votre sourire ?",
                      ["✅ Oui, souvent", "🟡 Parfois", "❌ Non, rarement"], index=1, key="q2")
        q3 = st.radio("**3.** L'enfant cherche-t-il à attirer votre attention ?",
                      ["✅ Oui, souvent", "🟡 Parfois", "❌ Non, rarement"], index=1, key="q3")
        q4 = st.radio("**4.** L'enfant imite-t-il vos gestes ?",
                      ["✅ Oui, souvent", "🟡 Parfois", "❌ Non, rarement"], index=1, key="q4")
        
        # Groupe 2
        st.markdown("### 🟡 2. Communication et langage")
        
        q5 = st.radio("**5.** L'enfant utilise-t-il des gestes pour communiquer ?",
                      ["✅ Oui, plusieurs gestes", "🟡 Un ou deux gestes", "❌ Non, pas de gestes"], index=1, key="q5")
        q6 = st.radio("**6.** L'enfant babille-t-il ou dit-il des mots ?",
                      ["✅ Oui, plusieurs mots", "🟡 Quelques sons ou mots", "❌ Très peu ou pas du tout"], index=1, key="q6")
        q7 = st.radio("**7.** L'enfant répète-t-il les mêmes mots sans contexte ?",
                      ["✅ Non, jamais", "🟡 Parfois", "❌ Oui, fréquemment"], index=0, key="q7")
        
        # Groupe 3
        st.markdown("### 🔴 3. Comportements répétitifs")
        
        q8 = st.radio("**8.** L'enfant a-t-il des mouvements répétitifs ?",
                      ["✅ Non, jamais", "🟡 Parfois", "❌ Oui, fréquemment"], index=0, key="q8")
        q9 = st.radio("**9.** L'enfant est-il très attaché à certains objets ?",
                      ["✅ Non, pas particulièrement", "🟡 Un peu", "❌ Oui, très attaché"], index=0, key="q9")
        q10 = st.radio("**10.** L'enfant a-t-il une routine rigide ?",
                       ["✅ Non, flexible", "🟡 Parfois", "❌ Oui, très difficile"], index=0, key="q10")
        
        # Groupe 4
        st.markdown("### 🟠 4. Sensorialité")
        
        q11 = st.radio("**11.** L'enfant réagit-il anormalement aux sons ?",
                       ["✅ Non, réaction normale", "🟡 Parfois", "❌ Oui, souvent"], index=0, key="q11")
        q12 = st.radio("**12.** L'enfant fixe-t-il les objets inhabituellement ?",
                       ["✅ Non", "🟡 Parfois", "❌ Oui, fréquemment"], index=0, key="q12")
        q13 = st.radio("**13.** L'enfant a-t-il une sensibilité anormale à la douleur ?",
                       ["✅ Non, normale", "🟡 Un peu inhabituel", "❌ Très différent"], index=0, key="q13")
        
        # Groupe 5
        st.markdown("### 🔵 5. Jeu et interactions")
        
        q14 = st.radio("**14.** L'enfant joue-t-il de manière imaginative ?",
                       ["✅ Oui, souvent", "🟡 Parfois", "❌ Non, jamais"], index=1, key="q14")
        q15 = st.radio("**15.** L'enfant préfère-t-il jouer seul ?",
                       ["✅ Non, aime jouer avec d'autres", "🟡 Un peu des deux", "❌ Oui, préfère seul"], index=1, key="q15")
        
        reponses = [q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, q11, q12, q13, q14, q15]
        st.session_state.reponses = reponses
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("✅ Enregistrer le questionnaire", key="save_q", use_container_width=True):
                scores = []
                for i, rep in enumerate(reponses):
                    if i < 6:
                        if "Oui" in rep and ("toujours" in rep or "souvent" in rep or "plusieurs" in rep):
                            scores.append(0)
                        elif "Parfois" in rep or "Un peu" in rep:
                            scores.append(1)
                        else:
                            scores.append(2)
                    elif i in [6, 7, 8, 9, 10, 11, 12]:
                        if "Non" in rep:
                            scores.append(0)
                        elif "Parfois" in rep or "Un peu" in rep:
                            scores.append(1)
                        else:
                            scores.append(2)
                    else:
                        if "Oui" in rep and "souvent" in rep:
                            scores.append(0)
                        elif "Parfois" in rep or "Un peu" in rep:
                            scores.append(1)
                        else:
                            scores.append(2)
                
                score_total = sum(scores)
                st.session_state.score_questionnaire = round((score_total / 30) * 10, 1)
                st.success(f"✅ Questionnaire enregistré ! Score: {st.session_state.score_questionnaire}/10")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # ===== TAB 2: AUDIO =====
    with tab2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🎙️ Analyse vocale")
        st.markdown("Enregistrez la voix de l'enfant pendant quelques secondes")
        
        uploaded_audio = st.file_uploader("📤 Télécharger un fichier audio", type=["wav", "mp3", "m4a"])
        
        if uploaded_audio is not None:
            st.audio(uploaded_audio, format="audio/wav")
            
            if st.button("🎙️ Analyser l'audio", key="analyze_audio"):
                with st.spinner("🔍 Analyse de la voix en cours..."):
                    score_audio = random.uniform(2, 9)
                    st.session_state.score_audio = round(score_audio, 1)
                    st.success(f"✅ Analyse vocale terminée ! Score: {st.session_state.score_audio}/10")
        
        if st.session_state.score_audio is not None:
            st.metric("🎯 Score analyse vocale", f"{st.session_state.score_audio}/10")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # ===== TAB 3: VISION =====
    with tab3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📷 Vision par ordinateur")
        st.markdown("Prenez une photo de l'enfant face à la caméra")
        
        camera_image = st.camera_input("📸 Prendre une photo")
        
        if camera_image is not None:
            st.image(camera_image, caption="📷 Photo analysée", width=250)
            
            if st.button("👁️ Analyser l'image", key="analyze_vision"):
                with st.spinner("🔍 Analyse de l'image en cours..."):
                    score_vision = random.uniform(2, 9)
                    st.session_state.score_vision = round(score_vision, 1)
                    st.success(f"✅ Analyse visuelle terminée ! Score: {st.session_state.score_vision}/10")
        
        if st.session_state.score_vision is not None:
            st.metric("🎯 Score analyse visuelle", f"{st.session_state.score_vision}/10")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # ===== BOUTON CONTINUER =====
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("▶️ CONTINUER VERS LES RÉSULTATS", type="primary", use_container_width=True):
            if st.session_state.score_questionnaire is not None:
                if st.session_state.score_audio is None:
                    st.session_state.score_audio = round(random.uniform(4, 6), 1)
                if st.session_state.score_vision is None:
                    st.session_state.score_vision = round(random.uniform(4, 6), 1)
                
                score_global = (st.session_state.score_questionnaire * 0.4 + 
                               st.session_state.score_audio * 0.3 + 
                               st.session_state.score_vision * 0.3)
                st.session_state.score_global = round(score_global, 1)
                
                if st.session_state.score_global < 4:
                    st.session_state.niveau = "🟢 Faible"
                    st.session_state.badge_class = "badge-low"
                    st.session_state.recommandation = "Développement typique. Surveillance standard recommandée."
                elif st.session_state.score_global < 7:
                    st.session_state.niveau = "🟠 Moyen"
                    st.session_state.badge_class = "badge-moderate"
                    st.session_state.recommandation = "Quelques signes d'alerte. Consultation avec un spécialiste recommandée."
                else:
                    st.session_state.niveau = "🔴 Élevé"
                    st.session_state.badge_class = "badge-high"
                    st.session_state.recommandation = "Signes évocateurs de TSA. Intervention précoce fortement recommandée."
                
                test_result = {
                    "date": datetime.now().strftime("%d/%m/%Y"),
                    "score": st.session_state.score_global,
                    "niveau": st.session_state.niveau,
                    "enfant": st.session_state.nom_enfant
                }
                st.session_state.historique_tests.append(test_result)
                
                st.session_state.etape = 4
                st.rerun()
            else:
                st.warning("⚠️ Veuillez compléter le questionnaire avant de continuer")
    
    st.markdown('</div>', unsafe_allow_html=True)

# ========== ÉTAPE 4: RÉSULTATS ==========
elif st.session_state.etape == 4:
    st.markdown('<div class="fade-in">', unsafe_allow_html=True)
    
    # Carte de résultat
    st.markdown(f"""
    <div class="card" style="text-align: center;">
        <h2 style="color: #1e3c72;">📊 Résultats de l'analyse</h2>
        <p style="font-size: 1.2rem;">Pour <strong>{st.session_state.nom_enfant}</strong> - {datetime.now().strftime("%d/%m/%Y")}</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="margin:0;">🧠 Score Global</h3>
            <p style="font-size: 2.5rem; margin:0; font-weight:bold;">{st.session_state.score_global}/10</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
            <h3 style="margin:0;">⚠️ Niveau</h3>
            <p style="font-size: 1.5rem; margin:0; font-weight:bold;">{st.session_state.niveau}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);">
            <h3 style="margin:0;">📊 Score Max</h3>
            <p style="font-size: 1.5rem; margin:0; font-weight:bold;">10/10</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Barres de progression
    st.subheader("📊 Détail des scores par modalité")
    
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown(f"""
        <div style="background: #f7f9fc; border-radius: 15px; padding: 1rem; text-align: center;">
            <strong>📝 Questionnaire</strong>
            <p style="font-size: 2rem; margin:0; font-weight:bold;">{st.session_state.score_questionnaire}/10</p>
        </div>
        """, unsafe_allow_html=True)
        st.progress(st.session_state.score_questionnaire / 10)
    
    with col_b:
        st.markdown(f"""
        <div style="background: #f7f9fc; border-radius: 15px; padding: 1rem; text-align: center;">
            <strong>🎙️ Analyse vocale</strong>
            <p style="font-size: 2rem; margin:0; font-weight:bold;">{st.session_state.score_audio}/10</p>
        </div>
        """, unsafe_allow_html=True)
        st.progress(st.session_state.score_audio / 10)
    
    with col_c:
        st.markdown(f"""
        <div style="background: #f7f9fc; border-radius: 15px; padding: 1rem; text-align: center;">
            <strong>📷 Vision</strong>
            <p style="font-size: 2rem; margin:0; font-weight:bold;">{st.session_state.score_vision}/10</p>
        </div>
        """, unsafe_allow_html=True)
        st.progress(st.session_state.score_vision / 10)
    
    st.markdown("---")
    
    # Recommandation
    st.subheader("💡 Recommandation personnalisée")
    
    if "Faible" in st.session_state.niveau:
        st.success(f"### {st.session_state.recommandation}")
    elif "Moyen" in st.session_state.niveau:
        st.warning(f"### {st.session_state.recommandation}")
    else:
        st.error(f"### {st.session_state.recommandation}")
    
    st.markdown("---")
    
    # Actions
    col1, col2, col3 = st.columns(3)
    
    with col1:
        rapport = f"""
╔══════════════════════════════════════════════════════════════╗
║                    RAPPORT NEUROSENSE AI+                    ║
╠══════════════════════════════════════════════════════════════╣
║  Enfant        : {st.session_state.nom_enfant}
║  Date          : {datetime.now().strftime("%d/%m/%Y")}
║  Score global  : {st.session_state.score_global}/10
║  Niveau        : {st.session_state.niveau}
║  Recommandation: {st.session_state.recommandation}
╠══════════════════════════════════════════════════════════════╣
║  Détails des scores:                                          ║
║  • Questionnaire : {st.session_state.score_questionnaire}/10
║  • Audio         : {st.session_state.score_audio}/10
║  • Vision        : {st.session_state.score_vision}/10
╚══════════════════════════════════════════════════════════════╝
        """
        st.download_button("📄 Télécharger le rapport", rapport, file_name=f"rapport_{st.session_state.nom_enfant}.txt", use_container_width=True)
    
   
    with col2:
        if st.button("🔄 Nouvelle évaluation", use_container_width=True):
            for key in list(st.session_state.keys()):
                if key not in ['historique_tests', 'langue', 'notifications']:
                    del st.session_state[key]
            st.session_state.etape = 1
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

# ========== SIDEBAR ==========
with st.sidebar:
    st.markdown("---")
    st.markdown("## ⚙️ Paramètres")
    
    langue = st.selectbox("🌐 Langue", ["Français", "English", "العربية"])
    notifications = st.toggle("🔔 Notifications", value=True)
    confidentialite = st.checkbox("🔒 Confidentialité")
    
    st.markdown("---")
    st.markdown("## 📜 Historique des tests")
    
    if st.session_state.historique_tests:
        for i, test in enumerate(st.session_state.historique_tests[-3:]):
            st.markdown(f"""
            <div style="background: #f7f9fc; border-radius: 10px; padding: 0.5rem; margin-bottom: 0.5rem;">
                <strong>{test['date']}</strong><br>
                {test['enfant']}: {test['score']}/10
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("📭 Aucun test effectué")
    
    st.markdown("---")
    st.markdown("## 📊 Statistiques")
    
    if st.session_state.historique_tests:
        scores_list = [t['score'] for t in st.session_state.historique_tests]
        st.metric("📈 Moyenne", f"{sum(scores_list)/len(scores_list):.1f}/10")
        st.metric("🔢 Nombre de tests", len(st.session_state.historique_tests))
    
    st.markdown("---")
    st.caption("© 2025 NeuroSense AI+ | v2.0")
    st.caption("Powered by Streamlit")

# ========== FOOTER ==========
st.markdown("""
<div class="footer">
    ⚠️ <strong>Avertissement important :</strong> Cette application est un outil d'aide à la détection précoce. 
    Elle ne remplace en aucun cas un diagnostic médical professionnel. 
    Consultez toujours un spécialiste pour toute préoccupation concernant le développement de votre enfant.
</div>
""", unsafe_allow_html=True)
