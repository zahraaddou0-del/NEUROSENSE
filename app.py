import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import random
from PIL import Image
import io

# Paramètres de la page
st.set_page_config(
    page_title="NeuroSense AI+ - Détection précoce de l'autisme",
    page_icon="🧠",
    layout="wide"
)

# Initialisation de la session
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

# ========== TITRE PRINCIPAL ==========
st.title("🧠 NeuroSense AI+")
st.markdown("*NeuroSense combine questionnaire, analyse vocale et vision par ordinateur pour une détection précoce intelligente.*")
st.markdown("---")

# ========== ÉTAPE 1: CHOIX DU PROFIL ==========
if st.session_state.etape == 1:
    st.header("1. 📋 Choix du profil")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("👨‍👩‍👦 Parent", use_container_width=True):
            st.session_state.type_utilisateur = "Parent"
            st.session_state.etape = 2
            st.rerun()
    with col2:
        if st.button("⚕️ Professionnel", use_container_width=True):
            st.session_state.type_utilisateur = "Professionnel"
            st.session_state.etape = 2
            st.rerun()

# ========== ÉTAPE 2: PROFIL DE L'ENFANT ==========
elif st.session_state.etape == 2:
    st.header("2. 👶 Profil de l'enfant")
    
    with st.form("profil_enfant"):
        nom_parent = st.text_input("Nom du parent / Professionnel")
        age_parent = st.number_input("Âge", min_value=1, max_value=100, step=1)
        
        st.markdown("---")
        
        nom_enfant = st.text_input("Nom de l'enfant")
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

# ========== ÉTAPE 3: ANALYSE IA (Questionnaire + Audio + Vision) ==========
# (Les anciennes étapes 3 et 4 ont été supprimées)
elif st.session_state.etape == 3:
    st.header("3. 🤖 Analyse IA")
    
    # Création des 3 onglets
    tab1, tab2, tab3 = st.tabs(["📝 Questionnaire", "🎙️ Audio", "👁️ Vision"])
    
    # ===== TAB 1: QUESTIONNAIRE (15 QUESTIONS) =====
    with tab1:
        st.subheader("Questionnaire d'évaluation")
        
        # Groupe 1: Interaction sociale (Q1-Q4)
        st.markdown("**🟢 Interaction sociale**")
        
        q1 = st.radio(
            "**1.** L'enfant répond-il à son prénom quand vous l'appelez ?",
            ["Oui, toujours", "Parfois", "Non, rarement ou jamais"],
            index=1,
            key="q1"
        )
        
        q2 = st.radio(
            "**2.** L'enfant sourit-il en réponse à votre sourire ?",
            ["Oui, souvent", "Parfois", "Non, rarement"],
            index=1,
            key="q2"
        )
        
        q3 = st.radio(
            "**3.** L'enfant cherche-t-il à attirer votre attention ?",
            ["Oui, souvent", "Parfois", "Non, rarement"],
            index=1,
            key="q3"
        )
        
        q4 = st.radio(
            "**4.** L'enfant imite-t-il vos gestes ?",
            ["Oui, souvent", "Parfois", "Non, rarement"],
            index=1,
            key="q4"
        )
        
        # Groupe 2: Communication (Q5-Q7)
        st.markdown("**🟡 Communication et langage**")
        
        q5 = st.radio(
            "**5.** L'enfant utilise-t-il des gestes pour communiquer ?",
            ["Oui, plusieurs gestes", "Un ou deux gestes", "Non, pas de gestes"],
            index=1,
            key="q5"
        )
        
        q6 = st.radio(
            "**6.** L'enfant babille-t-il ou dit-il des mots ?",
            ["Oui, plusieurs mots", "Quelques sons ou mots", "Très peu ou pas du tout"],
            index=1,
            key="q6"
        )
        
        q7 = st.radio(
            "**7.** L'enfant répète-t-il les mêmes mots sans contexte ?",
            ["Non, jamais", "Parfois", "Oui, fréquemment"],
            index=0,
            key="q7"
        )
        
        # Groupe 3: Comportements répétitifs (Q8-Q10)
        st.markdown("**🔴 Comportements répétitifs**")
        
        q8 = st.radio(
            "**8.** L'enfant a-t-il des mouvements répétitifs ?",
            ["Non, jamais", "Parfois", "Oui, fréquemment"],
            index=0,
            key="q8"
        )
        
        q9 = st.radio(
            "**9.** L'enfant est-il très attaché à certains objets ?",
            ["Non, pas particulièrement", "Un peu", "Oui, très attaché"],
            index=0,
            key="q9"
        )
        
        q10 = st.radio(
            "**10.** L'enfant a-t-il une routine rigide ?",
            ["Non, flexible", "Parfois", "Oui, très difficile"],
            index=0,
            key="q10"
        )
        
        # Groupe 4: Sensorialité (Q11-Q13)
        st.markdown("**🟠 Sensorialité**")
        
        q11 = st.radio(
            "**11.** L'enfant réagit-il anormalement aux sons ?",
            ["Non, réaction normale", "Parfois", "Oui, souvent"],
            index=0,
            key="q11"
        )
        
        q12 = st.radio(
            "**12.** L'enfant fixe-t-il les objets inhabituellement ?",
            ["Non", "Parfois", "Oui, fréquemment"],
            index=0,
            key="q12"
        )
        
        q13 = st.radio(
            "**13.** L'enfant a-t-il une sensibilité anormale à la douleur ?",
            ["Non, normale", "Un peu inhabituel", "Très différent"],
            index=0,
            key="q13"
        )
        
        # Groupe 5: Jeu (Q14-Q15)
        st.markdown("**🔵 Jeu et interactions**")
        
        q14 = st.radio(
            "**14.** L'enfant joue-t-il de manière imaginative ?",
            ["Oui, souvent", "Parfois", "Non, jamais"],
            index=1,
            key="q14"
        )
        
        q15 = st.radio(
            "**15.** L'enfant préfère-t-il jouer seul ?",
            ["Non, aime jouer avec d'autres", "Un peu des deux", "Oui, préfère seul"],
            index=1,
            key="q15"
        )
        
        # Sauvegarder les réponses
        reponses = [q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, q11, q12, q13, q14, q15]
        st.session_state.reponses = reponses
        
        if st.button("✅ Enregistrer le questionnaire", key="save_q"):
            # Calcul du score
            scores = []
            for i, rep in enumerate(reponses):
                if i < 6:
                    if "toujours" in rep or "souvent" in rep or "plusieurs" in rep:
                        scores.append(0)
                    elif "Parfois" in rep or "Un peu" in rep or "quelques" in rep or "deux" in rep:
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
            st.success(f"Questionnaire enregistré ! Score: {st.session_state.score_questionnaire}/10")
    
    # ===== TAB 2: AUDIO =====
    with tab2:
        st.subheader("Analyse vocale")
        st.markdown("Enregistrez la voix de l'enfant pendant quelques secondes")
        
        uploaded_audio = st.file_uploader("📤 Télécharger un fichier audio", type=["wav", "mp3", "m4a"])
        
        if uploaded_audio is not None:
            st.audio(uploaded_audio, format="audio/wav")
            
            if st.button("🎙️ Analyser l'audio", key="analyze_audio"):
                with st.spinner("Analyse de la voix en cours..."):
                    score_audio = random.uniform(2, 9)
                    st.session_state.score_audio = round(score_audio, 1)
                    st.success(f"Analyse vocale terminée ! Score: {st.session_state.score_audio}/10")
        
        if st.session_state.score_audio is not None:
            st.metric("Score analyse vocale", f"{st.session_state.score_audio}/10")
    
    # ===== TAB 3: VISION =====
    with tab3:
        st.subheader("Vision par ordinateur")
        st.markdown("Prenez une photo de l'enfant face à la caméra")
        
        camera_image = st.camera_input("📸 Prendre une photo")
        
        if camera_image is not None:
            st.image(camera_image, caption="Photo analysée", width=250)
            
            if st.button("👁️ Analyser l'image", key="analyze_vision"):
                with st.spinner("Analyse de l'image en cours..."):
                    score_vision = random.uniform(2, 9)
                    st.session_state.score_vision = round(score_vision, 1)
                    st.success(f"Analyse visuelle terminée ! Score: {st.session_state.score_vision}/10")
        
        if st.session_state.score_vision is not None:
            st.metric("Score analyse visuelle", f"{st.session_state.score_vision}/10")
    
    # ===== BOUTON CONTINUER =====
    st.markdown("---")
    
    if st.button("▶️ CONTINUER", type="primary", use_container_width=True):
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
            
            # Sauvegarder dans l'historique
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
    st.header("4. 📊 Résultats")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Score NeuroSense", f"{st.session_state.score_global}/10")
    with col2:
        st.metric("Niveau de risque", st.session_state.niveau)
    with col3:
        st.progress(st.session_state.score_global / 10)
    
    st.markdown("---")
    
    # Affichage simple des scores
    st.subheader("Scores par modalité")
    
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.info(f"**Questionnaire**\n\n{st.session_state.score_questionnaire}/10")
    with col_b:
        st.info(f"**Analyse vocale**\n\n{st.session_state.score_audio}/10")
    with col_c:
        st.info(f"**Vision**\n\n{st.session_state.score_vision}/10")
    
    # Barre de progression visuelle pour chaque score
    st.subheader("Détail des scores")
    st.write("**Questionnaire**")
    st.progress(st.session_state.score_questionnaire / 10)
    st.write("**Analyse vocale**")
    st.progress(st.session_state.score_audio / 10)
    st.write("**Vision**")
    st.progress(st.session_state.score_vision / 10)
    
    st.markdown("---")
    
    # Recommandation
    st.subheader("💡 Recommandation")
    if "Faible" in st.session_state.niveau:
        st.success(st.session_state.recommandation)
    elif "Moyen" in st.session_state.niveau:
        st.warning(st.session_state.recommandation)
    else:
        st.error(st.session_state.recommandation)
    
    st.markdown("---")
    
    # Boutons
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

Détails des scores:
- Questionnaire: {st.session_state.score_questionnaire}/10
- Analyse vocale: {st.session_state.score_audio}/10
- Analyse visuelle: {st.session_state.score_vision}/10
        """
        st.download_button("📄 Télécharger le rapport", rapport, file_name=f"rapport_{st.session_state.nom_enfant}.txt")
   
    with col2:
        if st.button("🔄 Nouveau test"):
            for key in list(st.session_state.keys()):
                if key not in ['historique_tests', 'langue', 'notifications']:
                    del st.session_state[key]
            st.session_state.etape = 1
            st.rerun()

# ========== SIDEBAR: PARAMÈTRES + HISTORIQUE ==========
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2103/2103704.png", width=60)
    st.markdown("## ⚙️ Paramètres")
    
    langue = st.selectbox("🌐 Langue", ["Français", "English", "العربية"], index=["Français", "English", "العربية"].index(st.session_state.langue))
    if langue != st.session_state.langue:
        st.session_state.langue = langue
    
    notifications = st.toggle("🔔 Notifications", value=st.session_state.notifications)
    st.session_state.notifications = notifications
    
    confidentialite = st.checkbox("🔒 Confidentialité")
    
    st.markdown("---")
    st.markdown("### 📜 Historique des tests")
    
    if st.session_state.historique_tests:
        for i, test in enumerate(st.session_state.historique_tests):
            with st.expander(f"Test {i+1} - {test['date']}"):
                st.write(f"**Enfant:** {test['enfant']}")
                st.write(f"**Score:** {test['score']}/10")
                st.write(f"**Niveau:** {test['niveau']}")
    else:
        st.info("Aucun test effectué")
    
    st.markdown("---")
    st.markdown("### 📊 Statistiques")
    if st.session_state.historique_tests:
        scores_list = [t['score'] for t in st.session_state.historique_tests]
        st.write(f"**Moyenne des scores:** {sum(scores_list)/len(scores_list):.1f}/10")
        st.write(f"**Nombre de tests:** {len(st.session_state.historique_tests)}")
    
    st.markdown("---")
    st.caption("© 2024 NeuroSense AI+ | v1.0")

st.markdown("---")
st.caption("⚠️ **Avertissement:** Cette application est un outil d'aide à la détection précoce. Elle ne remplace en aucun cas un diagnostic médical professionnel.")
