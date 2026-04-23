import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

# Paramètres de la page
st.set_page_config(
    page_title="NeuroSense - Détection précoce de l'autisme",
    page_icon="🧠",
    layout="wide"
)

# Titre principal
st.title("🧠 NeuroSense AI+")
st.markdown("### Détection précoce des troubles du spectre autistique par intelligence artificielle")

# Initialisation de la session
if 'step' not in st.session_state:
    st.session_state.step = 1
if 'responses' not in st.session_state:
    st.session_state.responses = {}
if 'age_mois' not in st.session_state:
    st.session_state.age_mois = 24
if 'genre' not in st.session_state:
    st.session_state.genre = "Masculin"

# Barre de progression
progress_bar = st.progress(0)

# ========== ÉTAPE 1: INFORMATIONS DE BASE ==========
if st.session_state.step == 1:
    progress_bar.progress(10)
    
    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        st.info("""
        **🧠 Qu'est-ce que NeuroSense ?**
        
        NeuroSense est un système intelligent qui aide à détecter 
        précocement les signes de l'autisme chez les enfants.
        
        **📋 Comment ça fonctionne ?**
        1. Renseignez les informations de base
        2. Répondez au questionnaire (15 questions)
        3. Obtenez une évaluation initiale
        4. Recevez des recommandations personnalisées
        
        ⏱️ **Durée :** 5-10 minutes
        """)
    
    with col2:
        st.subheader("📋 Informations de base")
        
        nom_enfant = st.text_input("Prénom de l'enfant (optionnel)", placeholder="Ex: Adam")
        
        col_age, col_gender = st.columns(2)
        with col_age:
            age_mois = st.number_input("Âge de l'enfant (en mois)", min_value=0, max_value=72, value=24, 
                                       help="0 à 72 mois = 0 à 6 ans")
            st.session_state.age_mois = age_mois
        with col_gender:
            genre = st.selectbox("Sexe", ["Masculin", "Féminin"])
            st.session_state.genre = genre
        
        col_family, col_jaundice = st.columns(2)
        with col_family:
            famille_asd = st.selectbox("Antécédents familiaux d'autisme ?", ["Non", "Oui", "Je ne sais pas"])
        with col_jaundice:
            jaunisse = st.selectbox("Jaunisse à la naissance ?", ["Non", "Oui", "Je ne sais pas"])
        
        if st.button("📝 Commencer le questionnaire", type="primary", use_container_width=True):
            st.session_state.nom_enfant = nom_enfant if nom_enfant else "Mon enfant"
            st.session_state.famille_asd = famille_asd
            st.session_state.jaunisse = jaunisse
            st.session_state.step = 2
            st.rerun()

# ========== ÉTAPE 2: QUESTIONNAIRE (15 Questions) ==========
elif st.session_state.step == 2:
    progress_bar.progress(30)
    
    st.header("📝 Questionnaire d'évaluation M-CHAT adapté")
    st.markdown("Veuillez répondre à chaque question selon le comportement habituel de l'enfant.")
    
    # Questions organisées par catégorie
    st.subheader("🟢 1. Interaction sociale")
    
    q1 = st.radio(
        "**Q1:** Votre enfant répond-il à son prénom quand vous l'appelez ?",
        ["Oui, toujours", "Parfois", "Non, rarement ou jamais"],
        index=1,
        key="q1"
    )
    
    q2 = st.radio(
        "**Q2:** Votre enfant sourit-il en réponse à votre sourire ?",
        ["Oui, souvent", "Parfois", "Non, rarement"],
        index=1,
        key="q2"
    )
    
    q3 = st.radio(
        "**Q3:** Votre enfant cherche-t-il à attirer votre attention sur quelque chose qui l'intéresse ?",
        ["Oui, souvent", "Parfois", "Non, rarement"],
        index=1,
        key="q3"
    )
    
    q4 = st.radio(
        "**Q4:** Votre enfant imite-t-il vos gestes (ex: taper des mains, faire au revoir) ?",
        ["Oui, souvent", "Parfois", "Non, rarement"],
        index=1,
        key="q4"
    )
    
    st.subheader("🟡 2. Communication et langage")
    
    q5 = st.radio(
        "**Q5:** Votre enfant utilise-t-il des gestes pour communiquer (pointer, faire non de la tête) ?",
        ["Oui, plusieurs gestes", "Un ou deux gestes", "Non, pas de gestes"],
        index=1,
        key="q5"
    )
    
    q6 = st.radio(
        "**Q6:** Votre enfant babille-t-il ou dit-il des mots ?",
        ["Oui, plusieurs mots", "Quelques sons ou mots", "Très peu ou pas du tout"],
        index=1,
        key="q6"
    )
    
    q7 = st.radio(
        "**Q7:** Votre enfant répète-t-il les mêmes mots ou phrases sans contexte approprié ?",
        ["Non, jamais", "Parfois", "Oui, fréquemment"],
        index=0,
        key="q7"
    )
    
    st.subheader("🔴 3. Comportements répétitifs et intérêts restreints")
    
    q8 = st.radio(
        "**Q8:** Votre enfant a-t-il des mouvements répétitifs (battre des mains, se balancer, tourner) ?",
        ["Non, jamais", "Parfois", "Oui, fréquemment"],
        index=0,
        key="q8"
    )
    
    q9 = st.radio(
        "**Q9:** Votre enfant est-il très attaché à certains objets inhabituels ?",
        ["Non, pas particulièrement", "Un peu", "Oui, très attaché"],
        index=0,
        key="q9"
    )
    
    q10 = st.radio(
        "**Q10:** Votre enfant a-t-il une routine rigide et se fâche-t-il s'il y a un changement ?",
        ["Non, flexible", "Parfois", "Oui, très difficile"],
        index=0,
        key="q10"
    )
    
    st.subheader("🟠 4. Sensorialité et réactions inhabituelles")
    
    q11 = st.radio(
        "**Q11:** Votre enfant réagit-il anormalement aux sons (ignore ou panique face à certains bruits) ?",
        ["Non, réaction normale", "Parfois", "Oui, souvent"],
        index=0,
        key="q11"
    )
    
    q12 = st.radio(
        "**Q12:** Votre enfant fixe-t-il les objets ou les lumières de façon inhabituelle ?",
        ["Non", "Parfois", "Oui, fréquemment"],
        index=0,
        key="q12"
    )
    
    q13 = st.radio(
        "**Q13:** Votre enfant est-il insensible à la douleur ou au contraire hypersensible ?",
        ["Non, sensibilité normale", "Un peu inhabituel", "Très différent des autres"],
        index=0,
        key="q13"
    )
    
    st.subheader("🔵 5. Jeu et interactions")
    
    q14 = st.radio(
        "**Q14:** Votre enfant joue-t-il de manière imaginative (ex: faire semblant de manger, donner à manger à une poupée) ?",
        ["Oui, souvent", "Parfois", "Non, jamais"],
        index=1,
        key="q14"
    )
    
    q15 = st.radio(
        "**Q15:** Votre enfant préfère-t-il jouer seul plutôt qu'avec d'autres enfants ?",
        ["Non, aime jouer avec d'autres", "Un peu des deux", "Oui, préfère seul"],
        index=1,
        key="q15"
    )
    
    # Boutons de navigation
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("✅ Analyser les résultats", type="primary", use_container_width=True):
            # Stocker les réponses
            st.session_state.responses = {
                'q1': q1, 'q2': q2, 'q3': q3, 'q4': q4,
                'q5': q5, 'q6': q6, 'q7': q7, 'q8': q8,
                'q9': q9, 'q10': q10, 'q11': q11, 'q12': q12,
                'q13': q13, 'q14': q14, 'q15': q15
            }
            st.session_state.step = 3
            st.rerun()

# ========== ÉTAPE 3: ANALYSE ET RÉSULTATS ==========
elif st.session_state.step == 3:
    progress_bar.progress(80)
    
    with st.spinner("🔄 Analyse des données par l'intelligence artificielle en cours..."):
        # Calcul du score
        score = 0
        reponses_detail = []
        
        # Fonction pour convertir les réponses en score
        def get_score_3(reponse):
            if reponse == "Oui, toujours" or reponse == "Oui, souvent" or reponse == "Oui, plusieurs mots" or reponse == "Oui, plusieurs gestes":
                return 0
            elif reponse == "Parfois" or reponse == "Un peu" or reponse == "Quelques sons ou mots" or reponse == "Un ou deux gestes" or reponse == "Un peu inhabituel" or reponse == "Un peu des deux":
                return 1
            else:
                return 2
        
        def get_score_2(reponse):
            if reponse == "Non, jamais" or reponse == "Non, pas particulièrement" or reponse == "Non, réaction normale" or reponse == "Non, sensibilité normale" or reponse == "Non, flexible":
                return 0
            elif reponse == "Parfois" or reponse == "Un peu" or reponse == "Quelques sons ou mots":
                return 1
            else:
                return 2
        
        # Calculer le score total
        scores_q = {
            'q1': get_score_3(st.session_state.responses['q1']),
            'q2': get_score_3(st.session_state.responses['q2']),
            'q3': get_score_3(st.session_state.responses['q3']),
            'q4': get_score_3(st.session_state.responses['q4']),
            'q5': get_score_3(st.session_state.responses['q5']),
            'q6': get_score_3(st.session_state.responses['q6']),
            'q7': 0 if st.session_state.responses['q7'] == "Non, jamais" else (1 if st.session_state.responses['q7'] == "Parfois" else 2),
            'q8': 0 if st.session_state.responses['q8'] == "Non, jamais" else (1 if st.session_state.responses['q8'] == "Parfois" else 2),
            'q9': 0 if st.session_state.responses['q9'] == "Non, pas particulièrement" else (1 if st.session_state.responses['q9'] == "Un peu" else 2),
            'q10': 0 if st.session_state.responses['q10'] == "Non, flexible" else (1 if st.session_state.responses['q10'] == "Parfois" else 2),
            'q11': 0 if st.session_state.responses['q11'] == "Non, réaction normale" else (1 if st.session_state.responses['q11'] == "Parfois" else 2),
            'q12': 0 if st.session_state.responses['q12'] == "Non" else (1 if st.session_state.responses['q12'] == "Parfois" else 2),
            'q13': 0 if st.session_state.responses['q13'] == "Non, sensibilité normale" else (1 if st.session_state.responses['q13'] == "Un peu inhabituel" else 2),
            'q14': 2 if st.session_state.responses['q14'] == "Non, jamais" else (1 if st.session_state.responses['q14'] == "Parfois" else 0),
            'q15': 2 if st.session_state.responses['q15'] == "Oui, préfère seul" else (1 if st.session_state.responses['q15'] == "Un peu des deux" else 0)
        }
        
        score_total = sum(scores_q.values())
        score_pourcentage = (score_total / 30) * 100
        
        # Déterminer le niveau de risque
        if score_total >= 18:
            prediction = 1
            niveau_risque = "🔴 Risque Élevé"
            couleur = "red"
            message_action = "consultation URGENTE avec un spécialiste"
        elif score_total >= 10:
            prediction = 0
            niveau_risque = "🟠 Risque Moyen"
            couleur = "orange"
            message_action = "surveillance rapprochée et consultation possible"
        else:
            prediction = 0
            niveau_risque = "🟢 Risque Faible"
            couleur = "green"
            message_action = "suivi normal"
        
        # Recommandations personnalisées
        recommendations = []
        if scores_q['q1'] >= 1:
            recommendations.append("🔹 Difficulté à répondre au prénom → stimuler l'appel par des jeux")
        if scores_q['q2'] >= 1:
            recommendations.append("🔹 Contact visuel limité → pratiquer des jeux de regards")
        if scores_q['q8'] >= 1:
            recommendations.append("🔹 Mouvements répétitifs → proposer des activités alternatives")
        if scores_q['q11'] >= 1:
            recommendations.append("🔹 Hypersensibilité sensorielle → consulter un ergothérapeute")
        if scores_q['q14'] >= 1:
            recommendations.append("🔹 Difficultés dans le jeu imaginaire → modéliser des jeux de faire semblant")
        
        # Facteurs de risque supplémentaires
        facteurs_risque = []
        if st.session_state.famille_asd == "Oui":
            facteurs_risque.append("🔸 Antécédents familiaux d'autisme")
        if st.session_state.jaunisse == "Oui":
            facteurs_risque.append("🔸 Antécédent de jaunisse néonatale")
        
        import time
        time.sleep(1.5)
    
    progress_bar.progress(100)
    
    # Affichage des résultats
    st.header("📊 Résultats de l'analyse NeuroSense AI+")
    st.markdown("---")
    
    # Métriques principales
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Score total", f"{score_total}/30")
        st.caption(f"Soit {score_pourcentage:.0f}%")
    
    with col2:
        st.metric("Niveau de risque", niveau_risque.split(" ")[1], delta=niveau_risque.split(" ")[0])
    
    with col3:
        st.metric("Âge de l'enfant", f"{st.session_state.age_mois} mois")
    
    st.markdown("---")
    
    # Message principal
    if prediction == 1:
        st.error(f"""
        ### ⚠️ {niveau_risque}
        
        D'après l'analyse de l'intelligence artificielle, votre enfant présente plusieurs indicateurs 
        qui méritent une attention particulière.
        
        **Recommandation principale :** {message_action}
        """)
    elif score_total >= 10:
        st.warning(f"""
        ### 🟡 {niveau_risque}
        
        Certains indicateurs sont présents, mais ils ne sont pas tous concluants.
        
        **Recommandation principale :** {message_action}
        """)
    else:
        st.success(f"""
        ### ✅ {niveau_risque}
        
        Les indicateurs observés sont dans les limites normales pour l'âge de l'enfant.
        
        **Recommandation principale :** {message_action}
        """)
    
    st.markdown("---")
    
    # Résultats détaillés par catégorie
    with st.expander("📋 Voir les résultats détaillés par catégorie"):
        
        # Catégorie 1: Interaction sociale
        st.subheader("Interaction sociale")
        scores_social = [scores_q['q1'], scores_q['q2'], scores_q['q3'], scores_q['q4']]
        score_social = sum(scores_social)
        st.write(f"**Score partiel :** {score_social}/8")
        if score_social >= 4:
            st.warning("⚠️ Des difficultés sont observées dans l'interaction sociale")
        else:
            st.success("✅ L'interaction sociale semble adaptée")
        
        # Catégorie 2: Communication
        st.subheader("Communication et langage")
        scores_comm = [scores_q['q5'], scores_q['q6'], scores_q['q7']]
        score_comm = sum(scores_comm)
        st.write(f"**Score partiel :** {score_comm}/6")
        if score_comm >= 3:
            st.warning("⚠️ Des retards ou particularités sont observés dans la communication")
        else:
            st.success("✅ La communication semble adaptée à l'âge")
        
        # Catégorie 3: Comportements répétitifs
        st.subheader("Comportements répétitifs")
        scores_rep = [scores_q['q8'], scores_q['q9'], scores_q['q10']]
        score_rep = sum(scores_rep)
        st.write(f"**Score partiel :** {score_rep}/6")
        if score_rep >= 3:
            st.warning("⚠️ Des comportements répétitifs ou des intérêts restreints sont observés")
        else:
            st.success("✅ Les comportements sont typiques")
    
    # Facteurs de risque
    if facteurs_risque:
        with st.expander("⚠️ Facteurs de risque identifiés"):
            for facteur in facteurs_risque:
                st.write(facteur)
    
    # Recommandations personnalisées
    with st.expander("💡 Recommandations personnalisées"):
        if recommendations:
            for rec in recommendations:
                st.write(rec)
        else:
            st.write("✅ Aucune recommandation spécifique pour le moment")
        
        st.divider()
        st.write("**Recommandations générales :**")
        st.write("- 📍 Stimulez l'interaction sociale par des jeux et des chansons")
        st.write("- 📍 Lisez des histoires adaptées à l'âge de l'enfant")
        st.write("- 📍 Consultez régulièrement le pédiatre pour le suivi")
    
    # Téléchargement du rapport
    with st.expander("📄 Générer et télécharger le rapport complet"):
        
        date_rapport = datetime.now().strftime("%d/%m/%Y")
        
        rapport = f"""
RAPPORT D'ÉVALUATION NEUROSENSE AI+
=====================================

Date : {date_rapport}
Enfant : {st.session_state.nom_enfant}
Âge : {st.session_state.age_mois} mois
Sexe : {st.session_state.genre}

RÉSULTATS DE L'ANALYSE
-----------------------
• Score total : {score_total}/30 ({score_pourcentage:.0f}%)
• Niveau de risque : {niveau_risque}

DÉTAIL DES SCORES PAR CATÉGORIE
-------------------------------
• Interaction sociale : {score_social}/8
• Communication : {score_comm}/6
• Comportements répétitifs : {score_rep}/6

FACTEURS DE RISQUE
------------------
• Antécédents familiaux : {st.session_state.famille_asd}
• Jaunisse néonatale : {st.session_state.jaunisse}

RECOMMANDATIONS
---------------
Recommandation principale : {message_action}

Recommandations spécifiques :
{chr(10).join(recommendations) if recommendations else '• Aucune recommandation spécifique pour le moment'}

RAPPEL IMPORTANT
----------------
Ce rapport est une aide à la décision basée sur l'intelligence artificielle.
Il ne constitue PAS un diagnostic médical définitif.
Pour un diagnostic officiel, veuillez consulter un médecin spécialiste.

© NeuroSense AI+ - Détection précoce de l'autisme
        """
        
        st.download_button(
            label="📥 Télécharger le rapport (PDF/TXT)",
            data=rapport,
            file_name=f"rapport_neurosense_{st.session_state.nom_enfant}_{date_rapport}.txt",
            mime="text/plain",
            use_container_width=True
        )
    
    st.markdown("---")
    st.caption("⚠️ **Avertissement :** Cette application est un outil d'aide à la détection précoce. Elle ne remplace en aucun cas un diagnostic médical professionnel. Consultez toujours un spécialiste pour toute préoccupation concernant le développement de votre enfant.")
    
    # Bouton pour recommencer
    if st.button("🔄 Nouvelle évaluation", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# Pied de page
st.markdown("---")
st.caption("© 2024 NeuroSense AI+ - Détection précoce des troubles du spectre autistique | Version 2.0")
