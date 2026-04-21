import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from utils import save_assessment_to_db, get_model, predict_asd

# Paramètres de la page
st.set_page_config(
    page_title="NeuroSense - Détection précoce de l'autisme",
    page_icon="🧠",
    layout="wide"
)

# Titre principal
st.title("🧠 NeuroSense")
st.markdown("### Détection précoce des troubles du spectre autistique par intelligence artificielle")

# Deux colonnes : une pour l'explication, l'autre pour le questionnaire
col1, col2 = st.columns([1, 2])

with col1:
    st.info("""
    **Qu'est-ce que NeuroSense ?**
    
    NeuroSense est un système intelligent qui aide à détecter précocement 
    les signes de l'autisme chez les enfants en utilisant des technologies d'intelligence artificielle.
    
    **Comment ça fonctionne ?**
    1. Répondez aux questions du questionnaire
    2. Le système analyse vos réponses
    3. Vous obtenez une évaluation initiale et un indicateur de risque
    """)

with col2:
    st.subheader("📝 Questionnaire d'évaluation")
    st.markdown("Veuillez répondre précisément aux questions suivantes :")

    # Questions Q-Chat-10 (10 questions principales)
    questions = {
        "a1": "L'enfant fixe-t-il les objets sans but apparent ?",
        "a2": "L'enfant préfère-t-il jouer seul ?",
        "a3": "L'enfant évite-t-il le contact visuel ?",
        "a4": "L'enfant répète-t-il les mêmes mouvements de façon répétitive ?",
        "a5": "L'enfant réagit-il anormalement aux sons ?",
        "a6": "L'enfant a-t-il du mal à comprendre les émotions des autres ?",
        "a7": "L'enfant range-t-il les objets d'une manière spécifique et se fâche-t-il si cela change ?",
        "a8": "L'enfant semble-t-il peu intéressé par les interactions avec les autres ?",
        "a9": "L'enfant a-t-il un retard dans le développement du langage ?",
        "a10": "L'enfant préfère-t-il une routine fixe et a-t-il du mal à s'adapter aux changements ?"
    }

    responses = {}
    for key, question in questions.items():
        responses[key] = st.radio(
            question, 
            [0, 1], 
            format_func=lambda x: "Oui ✅" if x == 1 else "Non ❌",
            horizontal=True,
            key=key
        )
    
    # Informations complémentaires
    st.subheader("📋 Informations complémentaires")
    
    col_age, col_gender = st.columns(2)
    with col_age:
        age_months = st.number_input("Âge de l'enfant (en mois)", min_value=0, max_value=72, value=24)
    with col_gender:
        gender = st.selectbox("Sexe", ["Masculin", "Féminin"])
        gender_val = 1 if gender == "Masculin" else 0
    
    col_family, col_jaundice = st.columns(2)
    with col_family:
        family_asd = st.selectbox("Y a-t-il des antécédents familiaux d'autisme ?", ["Non", "Oui"])
        family_val = 1 if family_asd == "Oui" else 0
    with col_jaundice:
        jaundice = st.selectbox("L'enfant a-t-il souffert de jaunisse à la naissance ?", ["Non", "Oui"])
        jaundice_val = 1 if jaundice == "Oui" else 0
    
    # Bouton de prédiction
    if st.button("🔍 Analyser les résultats et prédire", type="primary", use_container_width=True):
        with st.spinner("Analyse des données en cours..."):
            # Calcul du score total
            total_score = sum(responses.values())
            
            # Préparation des données pour sauvegarde
            assessment_data = {
                "timestamp": datetime.now(),
                "age_months": age_months,
                "gender": gender_val,
                "family_asd": family_val,
                "jaundice": jaundice_val,
                "responses": responses,
                "total_score": total_score,
                "prediction": None
            }
            
            # Prédiction basée sur le score
            if total_score >= 6:
                prediction = 1
                risk_level = "Élevé"
                message = """
                ⚠️ **Résultat de l'évaluation : Probabilité élevée**
                
                D'après vos réponses, certains indicateurs méritent une consultation avec un spécialiste.
                Ce résultat n'est pas un diagnostic définitif, mais un guide pour un suivi approprié.
                
                **Recommandations :**
                - Consultez un spécialiste du développement et du comportement de l'enfant
                - Continuez à observer le développement de votre enfant
                - Notez vos observations à partager avec le médecin
                """
            elif total_score >= 4:
                prediction = 0
                risk_level = "Moyen"
                message = """
                🟡 **Résultat de l'évaluation : Probabilité moyenne**
                
                Certains indicateurs sont présents, mais ils ne sont pas concluants.
                
                **Recommandations :**
                - Suivez le développement de votre enfant pendant 3 à 6 mois
                - Si vous observez une évolution des symptômes, consultez un médecin
                """
            else:
                prediction = 0
                risk_level = "Faible"
                message = """
                🟢 **Résultat de l'évaluation : Probabilité faible**
                
                Les indicateurs que vous avez rapportés sont dans les limites normales.
                
                **Recommandations :**
                - Continuez à stimuler l'interaction avec votre enfant
                - Effectuez les contrôles pédiatriques réguliers
                """
            
            assessment_data["prediction"] = prediction
            assessment_data["risk_level"] = risk_level
            
            # Sauvegarde dans la base de données
            try:
                saved_id = save_assessment_to_db(assessment_data)
                st.success(f"✅ Évaluation sauvegardée avec succès ! (ID: {saved_id})")
            except Exception as e:
                st.error(f"Erreur lors de la sauvegarde des données : {str(e)}")
            
            # Affichage du résultat
            st.markdown("---")
            st.subheader("📊 Résultat de l'évaluation")
            st.markdown(message)
            
            # Affichage des détails
            with st.expander("Voir le détail des réponses"):
                st.write(f"**Score total :** {total_score} sur 10")
                st.write("**Détail des réponses :**")
                for key, question in questions.items():
                    status = "✅ Oui" if responses[key] == 1 else "❌ Non"
                    st.write(f"- {question}: {status}") 
