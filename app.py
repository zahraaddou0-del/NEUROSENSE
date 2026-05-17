# -*- coding: utf-8 -*-
"""
Application de prédiction de l'autisme - Version Wizard (page par page)
Affiche une question par écran avec boutons Suivant/Précédent
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc

# Configuration de la page
st.set_page_config(
    page_title="Prédiction de l'autisme",
    page_icon="🧩",
    layout="wide"
)

# Initialisation des variables de session
if 'page' not in st.session_state:
    st.session_state.page = 0
if 'reponses' not in st.session_state:
    st.session_state.reponses = {}
if 'model_entraine' not in st.session_state:
    st.session_state.model_entraine = False
if 'model' not in st.session_state:
    st.session_state.model = None
if 'le_dict' not in st.session_state:
    st.session_state.le_dict = {}
if 'donnees_chargees' not in st.session_state:
    st.session_state.donnees_chargees = False
if 'df' not in st.session_state:
    st.session_state.df = None
if 'accuracy' not in st.session_state:
    st.session_state.accuracy = 0

# ==================== FONCTIONS ====================

@st.cache_data
def load_data():
    """Chargement des données - CSV ou données fictives"""
    try:
        df = pd.read_csv('autism_screening.csv')
        st.sidebar.success("✅ Fichier CSV chargé!")
        return df, True
    except FileNotFoundError:
        st.sidebar.warning("⚠️ CSV non trouvé - utilisation de données fictives")
        np.random.seed(42)
        n_samples = 500
        data = {
            'A1_Score': np.random.randint(0, 2, n_samples),
            'A2_Score': np.random.randint(0, 2, n_samples),
            'A3_Score': np.random.randint(0, 2, n_samples),
            'A4_Score': np.random.randint(0, 2, n_samples),
            'A5_Score': np.random.randint(0, 2, n_samples),
            'A6_Score': np.random.randint(0, 2, n_samples),
            'A7_Score': np.random.randint(0, 2, n_samples),
            'A8_Score': np.random.randint(0, 2, n_samples),
            'A9_Score': np.random.randint(0, 2, n_samples),
            'A10_Score': np.random.randint(0, 2, n_samples),
            'age': np.random.randint(2, 60, n_samples),
            'gender': np.random.choice(['m', 'f'], n_samples),
            'ethnicity': np.random.choice(['Blanc', 'Asiatique', 'Noir', 'Hispanique', 'Autre'], n_samples),
            'jaundice': np.random.choice([0, 1], n_samples),
            'family_member_with_ASD': np.random.choice([0, 1], n_samples),
            'Class_ASD': np.random.choice([0, 1], n_samples, p=[0.65, 0.35])
        }
        df = pd.DataFrame(data)
        return df, False

def entrainer_modele(df):
    """Entraînement du modèle Random Forest"""
    target_col = 'Class_ASD'
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    categorical_cols = X.select_dtypes(include=['object']).columns
    le_dict = {}
    
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        le_dict[col] = le
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, le_dict, X_train, X_test, y_train, y_test, y_pred, accuracy

# ==================== SIDEBAR (toujours visible) ====================

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2917/2917995.png", width=70)
    st.title("📊 Navigation")
    
    # Indicateur de progression
    if st.session_state.model_entraine:
        st.success(f"✅ Modèle entraîné - Précision: {st.session_state.accuracy:.1%}")
    else:
        st.info("🤖 Cliquez sur 'Entraîner' ci-dessous")
    
    st.markdown("---")
    
    # Bouton d'entraînement (toujours disponible)
    if st.button("🚀 Entraîner le modèle", type="primary", use_container_width=True):
        if st.session_state.df is not None:
            with st.spinner("Entraînement en cours..."):
                model, le_dict, X_train, X_test, y_train, y_test, y_pred, accuracy = entrainer_modele(st.session_state.df)
                st.session_state.model = model
                st.session_state.le_dict = le_dict
                st.session_state.accuracy = accuracy
                st.session_state.model_entraine = True
                st.session_state.X_train = X_train
                st.session_state.X_test = X_test
                st.session_state.y_train = y_train
                st.session_state.y_test = y_test
                st.session_state.y_pred = y_pred
                st.rerun()
        else:
            st.error("❌ Chargez les données d'abord!")
    
    # Bouton réinitialiser
    if st.button("🔄 Réinitialiser les réponses", use_container_width=True):
        st.session_state.page = 0
        st.session_state.reponses = {}
        st.rerun()
    
    st.markdown("---")
    st.caption("💡 Répondez aux questions page par page")

# ==================== TITRE PRINCIPAL ====================

st.title("🧩 Détection précoce des troubles du spectre autistique")
st.markdown("---")

# ==================== CHARGEMENT DES DONNÉES ====================

if not st.session_state.donnees_chargees:
    with st.spinner("Chargement des données..."):
        st.session_state.df, _ = load_data()
        st.session_state.donnees_chargees = True
        st.rerun()

# ==================== AFFICHAGE DES STATISTIQUES (si modèle entraîné) ====================

if st.session_state.model_entraine:
    with st.expander("📈 Voir les statistiques et graphiques du modèle", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Matrice de confusion")
            cm = confusion_matrix(st.session_state.y_test, st.session_state.y_pred)
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel('Prédiction')
            ax.set_ylabel('Réalité')
            ax.set_title('Matrice de confusion')
            st.pyplot(fig)
        
        with col2:
            st.subheader("📈 Rapport de classification")
            report = classification_report(st.session_state.y_test, st.session_state.y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.round(3), use_container_width=True)
        
        # Courbe ROC
        st.subheader("📉 Courbe ROC")
        y_pred_proba = st.session_state.model.predict_proba(st.session_state.X_test)[:, 1]
        fpr, tpr, _ = roc_curve(st.session_state.y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc:.2f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax.set_xlabel('Taux de faux positifs')
        ax.set_ylabel('Taux de vrais positifs')
        ax.set_title('Courbe ROC - Modèle Random Forest')
        ax.legend(loc="lower right")
        st.pyplot(fig)
        
        # Importance des caractéristiques
        st.subheader("🎯 Importance des caractéristiques")
        feature_names = st.session_state.X_train.columns.tolist()
        importances = st.session_state.model.feature_importances_
        
        fig, ax = plt.subplots(figsize=(10, 6))
        indices = np.argsort(importances)[-10:]
        ax.barh(range(len(indices)), importances[indices])
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels([feature_names[i] for i in indices])
        ax.set_xlabel('Importance')
        ax.set_title('Top 10 des caractéristiques les plus importantes')
        st.pyplot(fig)

# ==================== QUESTIONNAIRE PAGE PAR PAGE ====================

st.markdown("### 📝 Questionnaire d'évaluation")

# Barre de progression
progress = (st.session_state.page + 1) / 13
st.progress(progress)
st.caption(f"Question {st.session_state.page + 1} sur 13")

st.markdown("---")

# Liste de toutes les questions
questions = [
    {
        "id": "q1",
        "titre": "🧠 Question 1/13 - Interactions sociales",
        "question": "Avez-vous des difficultés à comprendre les expressions faciales ou le ton de la voix des autres ?",
        "type": "score",
        "field": "A1_Score"
    },
    {
        "id": "q2",
        "titre": "🗣️ Question 2/13 - Communication",
        "question": "Avez-vous du mal à maintenir une conversation (ex: savoir quand parler ou s'arrêter) ?",
        "type": "score",
        "field": "A2_Score"
    },
    {
        "id": "q3",
        "titre": "🔄 Question 3/13 - Comportements répétitifs",
        "question": "Avez-vous des comportements répétitifs (ex: se balancer, tourner en rond, trier des objets) ?",
        "type": "score",
        "field": "A3_Score"
    },
    {
        "id": "q4",
        "titre": "🎨 Question 4/13 - Intérêts restreints",
        "question": "Avez-vous des intérêts très spécifiques et intenses (ex: toujours parler du même sujet) ?",
        "type": "score",
        "field": "A4_Score"
    },
    {
        "id": "q5",
        "titre": "😐 Question 5/13 - Expression émotionnelle",
        "question": "Est-ce que les autres vous disent que vous semblez distant ou sans émotion ?",
        "type": "score",
        "field": "A5_Score"
    },
    {
        "id": "q6",
        "titre": "🔊 Question 6/13 - Sensibilité sensorielle",
        "question": "Êtes-vous gêné par certains bruits, lumières ou textures (ex: aspirateur, étiquettes de vêtements) ?",
        "type": "score",
        "field": "A6_Score"
    },
    {
        "id": "q7",
        "titre": "🤝 Question 7/13 - Relations sociales",
        "question": "Préférez-vous être seul(e) plutôt qu'avec d'autres personnes ?",
        "type": "score",
        "field": "A7_Score"
    },
    {
        "id": "q8",
        "titre": "💬 Question 8/13 - Langage",
        "question": "Avez-vous un langage très littéral (difficulté à comprendre les blagues ou l'ironie) ?",
        "type": "score",
        "field": "A8_Score"
    },
    {
        "id": "q9",
        "titre": "👀 Question 9/13 - Contact visuel",
        "question": "Évitez-vous le contact visuel avec les autres ?",
        "type": "score",
        "field": "A9_Score"
    },
    {
        "id": "q10",
        "titre": "📅 Question 10/13 - Routine",
        "question": "Êtes-vous très attaché(e) à vos routines (vous énervez quand quelque chose change) ?",
        "type": "score",
        "field": "A10_Score"
    },
    {
        "id": "q11",
        "titre": "👤 Question 11/13 - Informations personnelles",
        "question": "Quel est votre âge ?",
        "type": "age",
        "field": "age"
    },
    {
        "id": "q12",
        "titre": "⚥ Question 12/13 - Genre",
        "question": "Quel est votre genre ?",
        "type": "gender",
        "field": "gender"
    },
    {
        "id": "q13",
        "titre": "👨‍👩‍👧‍👦 Question 13/13 - Antécédents familiaux",
        "question": "Avez-vous un membre de votre famille (parent, frère, sœur) diagnostiqué autiste ?",
        "type": "family",
        "field": "family_asd"
    }
]

# Affichage de la question actuelle
q = questions[st.session_state.page]

# Conteneur pour la question
with st.container():
    st.subheader(q["titre"])
    st.write(f"**{q['question']}**")
    st.write("")
    
    # Réponse selon le type de question
    if q["type"] == "score":
        reponse = st.radio(
            "Sélectionnez une réponse :",
            options=[0, 1],
            format_func=lambda x: "✅ Oui" if x == 1 else "❌ Non",
            key=f"radio_{q['id']}",
            index=st.session_state.reponses.get(q['field'], None),
            horizontal=True
        )
    
    elif q["type"] == "age":
        reponse = st.number_input(
            "Âge en années :",
            min_value=1,
            max_value=100,
            value=st.session_state.reponses.get(q['field'], 25),
            step=1
        )
    
    elif q["type"] == "gender":
        reponse = st.radio(
            "Sélectionnez votre genre :",
            options=["m", "f"],
            format_func=lambda x: "👨 Homme" if x == "m" else "👩 Femme",
            key=f"radio_{q['id']}",
            index=0 if st.session_state.reponses.get(q['field'], "m") == "m" else 1,
            horizontal=True
        )
    
    elif q["type"] == "family":
        reponse = st.radio(
            "Sélectionnez une réponse :",
            options=[0, 1],
            format_func=lambda x: "❌ Non" if x == 0 else "✅ Oui",
            key=f"radio_{q['id']}",
            index=st.session_state.reponses.get(q['field'], 0),
            horizontal=True
        )
    
    # Sauvegarder la réponse
    st.session_state.reponses[q['field']] = reponse

st.markdown("---")

# ==================== BOUTONS SUIVANT / PRÉCÉDENT ====================

col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    if st.session_state.page > 0:
        if st.button("⬅️ Précédent", use_container_width=True):
            st.session_state.page -= 1
            st.rerun()

with col3:
    if st.session_state.page < len(questions) - 1:
        if st.button("Suivant ➡️", type="primary", use_container_width=True):
            st.session_state.page += 1
            st.rerun()
    else:
        # Dernière page - bouton de prédiction
        if st.button("🔍 Voir le résultat", type="primary", use_container_width=True):
            st.session_state.page = 0
            st.rerun()

# ==================== AFFICHAGE DU RÉSULTAT (après la dernière question) ====================

if len(st.session_state.reponses) == len(questions) and st.session_state.model_entraine:
    st.markdown("---")
    st.subheader("🔮 Résultat de l'évaluation")
    
    # Construction des données pour la prédiction
    input_data = []
    
    # Scores A1 à A10
    for i in range(1, 11):
        input_data.append(st.session_state.reponses.get(f'A{i}_Score', 0))
    
    # Âge
    input_data.append(st.session_state.reponses.get('age', 25))
    
    # Genre (encodage)
    gender_val = st.session_state.reponses.get('gender', 'm')
    input_data.append(0 if gender_val == 'm' else 1)
    
    # Origine ethnique (par défaut)
    input_data.append(0)  # Blanc par défaut
    
    # Jaundice (par défaut 0)
    input_data.append(0)
    
    # Antécédents familiaux
    input_data.append(st.session_state.reponses.get('family_asd', 0))
    
    # Prédiction
    try:
        input_array = np.array(input_data).reshape(1, -1)
        prediction = st.session_state.model.predict(input_array)[0]
        probabilities = st.session_state.model.predict_proba(input_array)[0]
        
        col_resultat, col_proba = st.columns(2)
        
        with col_resultat:
            if prediction == 1:
                st.error("🚨 **Résultat : Risque élevé de troubles du spectre autistique (TSA)**")
                st.warning("⚠️ Nous vous recommandons de consulter un professionnel de santé")
            else:
                st.success("✅ **Résultat : Risque faible de troubles du spectre autistique (TSA)**")
                st.info("ℹ️ Restez attentif aux signes de développement")
        
        with col_proba:
            st.write(f"**Probabilité non-autiste :** {probabilities[0]:.1%}")
            st.write(f"**Probabilité autiste :** {probabilities[1]:.1%}")
            st.progress(probabilities[1])
        
        # Affichage du score total
        total_score = sum([st.session_state.reponses.get(f'A{i}_Score', 0) for i in range(1, 11)])
        st.markdown("---")
        col_score, col_conseil = st.columns(2)
        with col_score:
            st.metric("📊 Score total A1-A10", f"{total_score} / 10")
        with col_conseil:
            if total_score >= 6:
                st.caption("⚡ Score élevé - Une évaluation approfondie est recommandée")
            else:
                st.caption("✅ Score bas - Le risque semble limité")
    
    except Exception as e:
        st.error(f"Erreur lors de la prédiction: {str(e)}")
        st.info("Assurez-vous que le modèle est entraîné (cliquez sur 'Entraîner le modèle' dans la barre latérale)")

elif len(st.session_state.reponses) == len(questions) and not st.session_state.model_entraine:
    st.warning("⚠️ Veuillez d'abord entraîner le modèle en cliquant sur 'Entraîner le modèle' dans la barre latérale")

# ==================== PIED DE PAGE ====================
st.markdown("---")
st.caption("🧩 Application de détection précoce de l'autisme - Développée avec Streamlit | ⚠️ Ceci est un outil d'aide non-diagnostique")
