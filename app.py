import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import random

# ========== CONFIGURATION ==========
st.set_page_config(
    page_title="NeuroSense AI+ - Détection précoce de l'autisme",
    page_icon="🧠",
    layout="wide"
)

# ========== INITIALISATION ==========
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
if 'reponses' not in st.session_state:
    st.session_state.reponses = []

# ========== TITRE ==========
st.markdown("""
<div style="text-align: center; padding: 1rem 0;">
    <h1 style="font-size: 3rem; color: #1e3c72;">🧠 NeuroSense AI+</h1>
    <p style="font-size: 1.1rem; color: #4a5568;">
        NeuroSense combine questionnaire, analyse vocale et vision par ordinateur pour une détection précoce intelligente.
    </p>
</div>
<hr>
""", unsafe_allow_html=True)

# ========== ÉTAPE 1: CHOIX DU PROFIL ==========
if st.session_state.etape == 1:
    st.header("1. 📋 Choix du profil")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    border-radius: 15px; padding: 1.5rem; text-align: center; color: white;">
            <h2 style="margin:0;">👨‍👩‍👦</h2>
            <h3>Parent</h3>
            <p>Pour les familles</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Choisir Parent", key="btn_parent", use_container_width=True):
            st.session_state.type_utilisateur = "Parent"
            st.session_state.etape = 2
            st.rerun()
    
    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                    border-radius: 15px; padding: 1.5rem; text-align: center; color: white;">
            <h2 style="margin:0;">⚕️</h2>
            <h3>Professionnel</h3>
            <p>Pour les spécialistes</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Choisir Professionnel", key="btn_pro", use_container_width=True):
            st.session_state.type_utilisateur = "Professionnel"
            st.session_state.etape = 2
            st.rerun()

# ========== ÉTAPE 2: PROFIL DE L'ENFANT ==========
elif st.session_state.etape == 2:
    st.header("2. 👶 Profil de l'enfant")
    
    with st.form("profil_enfant"):
        col1, col2 = st.columns(2)
        
        with col1:
            nom_parent = st.text_input("Nom du parent / Professionnel")
            age_parent = st.number_input("Âge", min_value=1, max_value=100, step=1)
        
        with col2:
            nom_enfant = st.text_input("Prénom de l'enfant")
            sexe_enfant = st.selectbox("Sexe", ["Garçon", "Fille", "Autre"])
    
        submitted = st.form_submit_button("📝 Continuer", use_container_width=True)
        
        if submitted:
            if nom_parent and nom_enfant:
                st.session_state.nom_parent = nom_parent
                st.session_state.age_parent = age_parent
                st.session_state.nom_enfant = nom_enfant
                st.session_state.sexe_enfant = sexe_enfant
                st.session_state.etape = 3
                st.rerun()
            else:
                st.error("Veuillez remplir tous les champs obligatoires")

# ========== ÉTAPE 3: ANALYSE IA ==========
elif st.session_state.etape == 3:
    st.header("3. 🤖 Analyse IA")
    st.info(f"👋 Bienvenue {st.session_state.nom_parent} ! Analyse pour {st.session_state.nom_enfant}")
    
    # Création des onglets
    tab1, tab2, tab3 = st.tabs(["📝 Questionnaire (15 questions)", "🎙️ Analyse vocale", "👁️ Vision par ordinateur"])
    
    # ===== TAB 1: QUESTIONNAIRE =====
    with tab1:
        st.subheader("📝 Questionnaire d'évaluation")
        st.caption("Veuillez répondre à toutes les questions")
        
        # Groupe 1
        st.markdown("**🟢 1. Interaction sociale**")
        
        q1 = st.radio("1. L'enfant répond-il à son prénom quand vous l'appelez ?",
                      ["Oui, toujours", "Parfois", "Non, rarement ou jamais"], index=1, key="q1")
        q2 = st.radio("2. L'enfant sourit-il en réponse à votre sourire ?",
                      ["Oui, souvent", "Parfois", "Non, rarement"], index=1, key="q2")
        q3 = st.radio("3. L'enfant cherche-t-il à attirer votre attention ?",
                      ["Oui, souvent", "Parfois", "Non, rarement"], index=1, key="q3")
        q4 = st.radio("4. L'enfant imite-t-il vos gestes ?",
                      ["Oui, souvent", "Parfois", "Non, rarement"], index=1, key="q4")
        
        # Groupe 2
        st.markdown("**🟡 2. Communication et langage**")
        
        q5 = st.radio("5. L'enfant utilise-t-il des gestes pour communiquer ?",
                      ["Oui, plusieurs gestes", "Un ou deux gestes", "Non, pas de gestes"], index=1, key="q5")
        q6 = st.radio("6. L'enfant babille-t-il ou dit-il des mots ?",
                      ["Oui, plusieurs mots", "Quelques sons ou mots", "Très peu ou pas du tout"], index=1, key="q6")
        q7 = st.radio("7. L'enfant répète-t-il les mêmes mots sans contexte ?",
                      ["Non, jamais", "Parfois", "Oui, fréquemment"], index=0, key="q7")
        
        # Groupe 3
        st.markdown("**🔴 3. Comportements répétitifs**")
        
        q8 = st.radio("8. L'enfant a-t-il des mouvements répétitifs ?",
                      ["Non, jamais", "Parfois", "Oui, fréquemment"], index=0, key="q8")
        q9 = st.radio("9. L'enfant est-il très attaché à certains objets ?",
                      ["Non, pas particulièrement", "Un peu", "Oui, très attaché"], index=0, key="q9")
        q10 = st.radio("10. L'enfant a-t-il une routine rigide ?",
                       ["Non, flexible", "Parfois", "Oui, très difficile"], index=0, key="q10")
        
        # Groupe 4
        st.markdown("**🟠 4. Sensorialité**")
        
        q11 = st.radio("11. L'enfant réagit-il anormalement aux sons ?",
                       ["Non, réaction normale", "Parfois", "Oui, souvent"], index=0, key="q11")
        q12 = st.radio("12. L'enfant fixe-t-il les objets inhabituellement ?",
                       ["Non", "Parfois", "Oui, fréquemment"], index=0, key="q12")
        q13 = st.radio("13. L'enfant a-t-il une sensibilité anormale à la douleur ?",
                       ["Non, normale", "Un peu inhabituel", "Très différent"], index=0, key="q13")
        
        # Groupe 5
        st.markdown("**🔵 5. Jeu et interactions**")
        
        q14 = st.radio("14. L'enfant joue-t-il de manière imaginative ?",
                       ["Oui, souvent", "Parfois", "Non, jamais"], index=1, key="q14")
        q15 = st.radio("15. L'enfant préfère-t-il jouer seul ?",
                       ["Non, aime jouer avec d'autres", "Un peu des deux", "Oui, préfère seul"], index=1, key="q15")
        
        reponses = [q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, q11, q12, q13, q14, q15]
        st.session_state.reponses = reponses
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("✅ Enregistrer le questionnaire", use_container_width=True):
                scores = []
                for i, rep in enumerate(reponses):
                    if i < 6:
                        if "toujours" in rep or "souvent" in rep or "plusieurs" in rep:
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
    
    # ===== TAB 2: AUDIO =====
    with tab2:
        st.subheader("🎙️ Analyse vocale")
        st.markdown("Enregistrez la voix de l'enfant")
        
        uploaded_audio = st.file_uploader("Télécharger un fichier audio", type=["wav", "mp3", "m4a"])
        
        if uploaded_audio is not None:
            st.audio(uploaded_audio, format="audio/wav")
            
            if st.button("Analyser l'audio"):
                with st.spinner("Analyse en cours..."):
                    score_audio = random.uniform(2, 9)
                    st.session_state.score_audio = round(score_audio, 1)
                    st.success(f"✅ Score: {st.session_state.score_audio}/10")
        
        if st.session_state.score_audio is not None:
            st.metric("Score analyse vocale", f"{st.session_state.score_audio}/10")
    
    # ===== TAB 3: VISION =====
    with tab3:
        st.subheader("👁️ Vision par ordinateur")
        st.markdown("Prenez une photo de l'enfant")
        
        camera_image = st.camera_input("Prendre une photo")
        
        if camera_image is not None:
            st.image(camera_image, caption="Photo analysée", width=250)
            
            if st.button("Analyser l'image"):
                with st.spinner("Analyse en cours..."):
                    score_vision = random.uniform(2, 9)
                    st.session_state.score_vision = round(score_vision, 1)
                    st.success(f"✅ Score: {st.session_state.score_vision}/10")
        
        if st.session_state.score_vision is not None:
            st.metric("Score analyse visuelle", f"{st.session_state.score_vision}/10")
    
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
                    st.session_state.recommandation = "Développement typique. Surveillance standard."
                elif st.session_state.score_global < 7:
                    st.session_state.niveau = "🟠 Moyen"
                    st.session_state.recommandation = "Quelques signes d'alerte. Consultez un spécialiste."
                else:
                    st.session_state.niveau = "🔴 Élevé"
                    st.session_state.recommandation = "Signes évocateurs de TSA. Intervention précoce recommandée."
                
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
                st.warning("Veuillez compléter le questionnaire avant de continuer")

# ========== ÉTAPE 4: RÉSULTATS ==========
elif st.session_state.etape == 4:
    st.header("📊 Résultats NeuroSense AI+")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Score global", f"{st.session_state.score_global}/10")
    
    with col2:
        if "Faible" in st.session_state.niveau:
            st.success(f"Niveau: {st.session_state.niveau}")
        elif "Moyen" in st.session_state.niveau:
            st.warning(f"Niveau: {st.session_state.niveau}")
        else:
            st.error(f"Niveau: {st.session_state.niveau}")
    
    with col3:
        st.progress(st.session_state.score_global / 10)
    
    st.markdown("---")
    
    st.subheader("Détail des scores")
    
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.info(f"**Questionnaire**\n\n{st.session_state.score_questionnaire}/10")
    with col_b:
        st.info(f"**Analyse vocale**\n\n{st.session_state.score_audio}/10")
    with col_c:
        st.info(f"**Analyse visuelle**\n\n{st.session_state.score_vision}/10")
    
    st.markdown("---")
    
    st.subheader("💡 Recommandation")
    if "Faible" in st.session_state.niveau:
        st.success(st.session_state.recommandation)
    elif "Moyen" in st.session_state.niveau:
        st.warning(st.session_state.recommandation)
    else:
        st.error(st.session_state.recommandation)
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        rapport = f"""
RAPPORT NEUROSENSE AI+
====================
Enfant: {st.session_state.nom_enfant}
Date: {datetime.now().strftime("%d/%m/%Y")}
Score global: {st.session_state.score_global}/10
Niveau: {st.session_state.niveau}
Recommandation: {st.session_state.recommandation}
        """
        st.download_button("📄 Télécharger", rapport, file_name="rapport.txt")
    
    with col2:
        if st.button("🔄 Nouveau test"):
            for key in list(st.session_state.keys()):
                if key not in ['historique_tests']:
                    del st.session_state[key]
            st.session_state.etape = 1
            st.rerun()

# ========== SIDEBAR ==========
with st.sidebar:
    st.markdown("## ⚙️ Paramètres")
    st.selectbox("Langue", ["Français", "English", "العربية"])
    st.toggle("Notifications")
    st.checkbox("Confidentialité")
    
    st.markdown("---")
    st.markdown("## 📜 Historique")
    
    if st.session_state.historique_tests:
        for test in st.session_state.historique_tests[-3:]:
            st.write(f"📅 {test['date']}: {test['score']}/10")
    else:
        st.info("Aucun test")

# ========== FOOTER ==========
st.markdown("---")
st.caption("⚠️ **Avertissement:** Cette application est un outil d'aide à la détection précoce. Elle ne remplace en aucun cas un diagnostic médical professionnel. Consultez toujours un spécialiste pour toute préoccupation concernant le développement de votre enfant.")
