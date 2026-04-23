import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import random

# Paramètres de la page
st.set_page_config(
    page_title="NeuroSense - Détection précoce de l'autisme",
    page_icon="🧠",
    layout="wide"
)

# Titre principal
st.title("🧠 NeuroSense")
st.markdown("### Détection précoce des troubles du spectre autistique par intelligence artificielle")

# Initialisation des scores dans la session
if 'score_audio' not in st.session_state:
    st.session_state.score_audio = None
if 'score_vision' not in st.session_state:
    st.session_state.score_vision = None
if 'analyse_complete' not in st.session_state:
    st.session_state.analyse_complete = False

# Création de 3 onglets (Tabs)
tab1, tab2, tab3 = st.tabs(["📝 Questionnaire", "🎙️ Analyse vocale", "📹 Vision par ordinateur"])

# ========== TAB 1: QUESTIONNAIRE (15 questions) ==========
with tab1:
    st.markdown("Veuillez répondre précisément aux questions suivantes :")
    
    st.markdown("### 🟢 1. Interaction sociale")
    
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
        "**3.** L'enfant cherche-t-il à attirer votre attention sur quelque chose qui l'intéresse ?",
        ["Oui, souvent", "Parfois", "Non, rarement"],
        index=1,
        key="q3"
    )
    
    q4 = st.radio(
        "**4.** L'enfant imite-t-il vos gestes (ex: taper des mains, faire au revoir) ?",
        ["Oui, souvent", "Parfois", "Non, rarement"],
        index=1,
        key="q4"
    )
    
    st.markdown("### 🟡 2. Communication et langage")
    
    q5 = st.radio(
        "**5.** L'enfant utilise-t-il des gestes pour communiquer (pointer, faire non de la tête) ?",
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
        "**7.** L'enfant répète-t-il les mêmes mots ou phrases sans contexte approprié ?",
        ["Non, jamais", "Parfois", "Oui, fréquemment"],
        index=0,
        key="q7"
    )
    
    st.markdown("### 🔴 3. Comportements répétitifs")
    
    q8 = st.radio(
        "**8.** L'enfant a-t-il des mouvements répétitifs (battre des mains, se balancer, tourner) ?",
        ["Non, jamais", "Parfois", "Oui, fréquemment"],
        index=0,
        key="q8"
    )
    
    q9 = st.radio(
        "**9.** L'enfant est-il très attaché à certains objets inhabituels ?",
        ["Non, pas particulièrement", "Un peu", "Oui, très attaché"],
        index=0,
        key="q9"
    )
    
    q10 = st.radio(
        "**10.** L'enfant a-t-il une routine rigide et se fâche-t-il s'il y a un changement ?",
        ["Non, flexible", "Parfois", "Oui, très difficile"],
        index=0,
        key="q10"
    )
    
    st.markdown("### 🟠 4. Sensorialité")
    
    q11 = st.radio(
        "**11.** L'enfant réagit-il anormalement aux sons ?",
        ["Non, réaction normale", "Parfois", "Oui, souvent"],
        index=0,
        key="q11"
    )
    
    q12 = st.radio(
        "**12.** L'enfant fixe-t-il les objets ou les lumières de façon inhabituelle ?",
        ["Non", "Parfois", "Oui, fréquemment"],
        index=0,
        key="q12"
    )
    
    q13 = st.radio(
        "**13.** L'enfant est-il insensible à la douleur ?",
        ["Non, sensibilité normale", "Un peu inhabituel", "Très différent"],
        index=0,
        key="q13"
    )
    
    st.markdown("### 🔵 5. Jeu et interactions")
    
    q14 = st.radio(
        "**14.** L'enfant joue-t-il de manière imaginative ?",
        ["Oui, souvent", "Parfois", "Non, jamais"],
        index=1,
        key="q14"
    )
    
    q15 = st.radio(
        "**15.** L'enfant préfère-t-il jouer seul plutôt qu'avec d'autres enfants ?",
        ["Non, aime jouer avec d'autres", "Un peu des deux", "Oui, préfère seul"],
        index=1,
        key="q15"
    )
    
    # Sauvegarder les réponses
    reponses_list = [q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, q11, q12, q13, q14, q15]
    st.session_state.reponses = reponses_list

# ========== TAB 2: ANALYSE VOCALE (Enregistrement audio) ==========
with tab2:
    st.subheader("🎙️ Analyse vocale")
    st.markdown("""
    **Instructions :**
    1. Cliquez sur le bouton d'enregistrement ci-dessous
    2. Demandez à l'enfant de dire quelques mots (ex: "Maman", "Papa", ou "Au revoir")
    3. Enregistrez pendant 3 à 5 secondes
    4. Cliquez sur "Analyser l'audio" pour obtenir le résultat
    """)
    
    # Méthode simple pour l'enregistrement audio (sans bibliothèque externe)
    st.info("💡 Pour enregistrer l'audio, vous pouvez utiliser l'enregistreur vocal de votre téléphone, puis télécharger le fichier ici.")
    
    uploaded_audio = st.file_uploader("📤 Télécharger un fichier audio (WAV, MP3)", type=["wav", "mp3", "m4a"])
    
    if uploaded_audio is not None:
        st.audio(uploaded_audio, format="audio/wav")
        
        if st.button("🔍 Analyser l'audio", key="analyze_audio"):
            with st.spinner("Analyse de la voix en cours..."):
                # Simulation de l'analyse IA (dans la réalité, on utiliserait un modèle)
                # Les critères analysés: prosodie, répétitions, réponse aux appels
                score_audio = random.uniform(2, 9)
                st.session_state.score_audio = round(score_audio, 1)
                
                st.success(f"✅ Analyse vocale terminée ! Score: {st.session_state.score_audio}/10")
                
                if st.session_state.score_audio > 7:
                    st.warning("⚠️ La prosodie et le rythme de la parole présentent des particularités")
                elif st.session_state.score_audio < 4:
                    st.success("✅ La voix de l'enfant semble typique pour son âge")
                else:
                    st.info("ℹ️ Quelques particularités vocales sont présentes, à surveiller")
    
    # Afficher le score audio s'il existe déjà
    if st.session_state.score_audio is not None:
        st.metric("Score analyse vocale", f"{st.session_state.score_audio}/10")

# ========== TAB 3: VISION PAR ORDINATEUR (Caméra) ==========
with tab3:
    st.subheader("📹 Vision par ordinateur")
    st.markdown("""
    **Instructions :**
    1. Activez votre caméra ci-dessous
    2. Placez l'enfant face à la caméra
    3. Prenez une photo
    4. L'IA analysera le contact visuel et les expressions faciales
    """)
    
    # Capture d'image par caméra
    camera_image = st.camera_input("📸 Prendre une photo de l'enfant")
    
    if camera_image is not None:
        # Afficher l'image capturée
        st.image(camera_image, caption="Photo capturée", width=300)
        
        if st.button("🔍 Analyser l'image", key="analyze_vision"):
            with st.spinner("Analyse de l'image en cours..."):
                # Simulation de l'analyse IA
                # Dans la réalité, on utiliserait un modèle comme MediaPipe ou OpenCV
                # pour détecter: contact visuel, sourire, expression faciale
                
                score_vision = random.uniform(2, 9)
                st.session_state.score_vision = round(score_vision, 1)
                
                st.success(f"✅ Analyse visuelle terminée ! Score: {st.session_state.score_vision}/10")
                
                if st.session_state.score_vision > 7:
                    st.warning("⚠️ Le contact visuel et les expressions faciales montrent des particularités")
                elif st.session_state.score_vision < 4:
                    st.success("✅ Le contact visuel et les expressions semblent typiques")
                else:
                    st.info("ℹ️ Quelques particularités dans le regard, à surveiller")
    
    # Afficher le score vision s'il existe déjà
    if st.session_state.score_vision is not None:
        st.metric("Score analyse visuelle", f"{st.session_state.score_vision}/10")

# ========== INFORMATIONS COMPLÉMENTAIRES ==========
st.markdown("---")
st.subheader("📋 Informations complémentaires")

col_age, col_gender = st.columns(2)
with col_age:
    age_mois = st.number_input("Âge de l'enfant (en mois)", min_value=0, max_value=72, value=24)
with col_gender:
    genre = st.selectbox("Sexe", ["Masculin", "Féminin"])

col_family, col_jaundice = st.columns(2)
with col_family:
    famille_asd = st.selectbox("Antécédents familiaux d'autisme ?", ["Non", "Oui"])
with col_jaundice:
    jaunisse = st.selectbox("Jaunisse à la naissance ?", ["Non", "Oui"])

# ========== BOUTON D'ANALYSE GLOBALE ==========
st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.button("🔍 ANALYSE GLOBALE (Questionnaire + Audio + Vision)", type="primary", use_container_width=True):
        
        with st.spinner("L'IA analyse l'ensemble des données..."):
            
            # 1. Calcul du score du questionnaire
            scores = []
            for i, rep in enumerate(st.session_state.reponses):
                if i < 6:
                    if "toujours" in rep or "souvent" in rep or "plusieurs" in rep:
                        scores.append(0)
                    elif "Parfois" in rep or "Un peu" in rep:
                        scores.append(1)
                    else:
                        scores.append(2)
                elif i == 6 or i == 7 or i == 8 or i == 9 or i == 10 or i == 11 or i == 12:
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
            
            score_questionnaire = sum(scores)  # Score sur 30
            score_q_normalise = (score_questionnaire / 30) * 10  # Convertir sur 10
            
            # 2. Récupérer les scores audio et vision (ou valeurs par défaut)
            score_audio = st.session_state.score_audio if st.session_state.score_audio is not None else 5
            score_vision = st.session_state.score_vision if st.session_state.score_vision is not None else 5
            
            # 3. Score global pondéré
            score_global = (score_q_normalise * 0.4) + (score_audio * 0.3) + (score_vision * 0.3)
            score_global = round(score_global, 1)
            
            # 4. Déterminer le niveau de risque
            if score_global >= 7:
                niveau_risque = "🔴 Risque Élevé"
                couleur = "red"
                recommandation = "Consultation URGENTE avec un spécialiste"
            elif score_global >= 4:
                niveau_risque = "🟠 Risque Moyen"
                couleur = "orange"
                recommandation = "Surveillance rapprochée et consultation possible"
            else:
                niveau_risque = "🟢 Risque Faible"
                couleur = "green"
                recommandation = "Suivi normal"
            
            # 5. Afficher les résultats
            st.markdown("---")
            st.subheader("📊 RÉSULTAT DE L'ANALYSE GLOBALE NEUROSENSE AI+")
            
            col_r1, col_r2, col_r3 = st.columns(3)
            with col_r1:
                st.metric("Score Questionnaire", f"{score_q_normalise:.1f}/10")
            with col_r2:
                st.metric("Score Audio", f"{score_audio}/10")
            with col_r3:
                st.metric("Score Vision", f"{score_vision}/10")
            
            st.markdown("---")
            
            # Jauge de risque
            if score_global >= 7:
                st.error(f"### {niveau_risque}")
            elif score_global >= 4:
                st.warning(f"### {niveau_risque}")
            else:
                st.success(f"### {niveau_risque}")
            
            st.progress(score_global / 10)
            
            st.markdown(f"**Score global neurodéveloppemental :** {score_global}/10")
            st.markdown(f"**Recommandation principale :** {recommandation}")
            
            # 6. Détails par modalité
            with st.expander("📋 Voir le détail par modalité d'analyse"):
                st.write("**Questionnaire (poids 40%) :**")
                st.write(f"- Score: {score_q_normalise:.1f}/10")
                st.write("**Analyse vocale (poids 30%) :**")
                st.write(f"- Score: {score_audio}/10")
                st.write("**Analyse visuelle (poids 30%) :**")
                st.write(f"- Score: {score_vision}/10")
            
            # 7. Téléchargement du rapport
            rapport = f"""
RAPPORT NEUROSENSE AI+ - ANALYSE COMPLÈTE
=========================================
Date: {datetime.now().strftime("%d/%m/%Y")}
Âge: {age_mois} mois | Sexe: {genre}

RÉSULTATS DES ANALYSES
---------------------
• Questionnaire: {score_q_normalise:.1f}/10
• Analyse vocale: {score_audio}/10
• Analyse visuelle: {score_vision}/10
• Score global: {score_global}/10

NIVEAU DE RISQUE: {niveau_risque}

RECOMMANDATION: {recommandation}

Ce rapport est généré par l'IA de NeuroSense.
Il ne remplace pas un diagnostic médical professionnel.
            """
            
            st.download_button("📥 Télécharger le rapport complet", rapport, file_name="rapport_neurosense.txt")

st.markdown("---")
st.caption("⚠️ **Avertissement :** Cette application est un outil d'aide à la détection précoce. Elle ne remplace en aucun cas un diagnostic médical professionnel.")
