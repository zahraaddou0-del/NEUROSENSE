# -*- coding: utf-8 -*-
"""
Application de prédiction de l'autisme - Autism Prediction App
Fonctionne sur Streamlit Cloud sans erreurs
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Configuration de la page
st.set_page_config(
    page_title="Prédiction de l'autisme",
    page_icon="🧩",
    layout="wide"
)

# Titre de l'application
st.title("🧩 Système de prédiction des troubles du spectre autistique (TSA)")
st.markdown("---")

@st.cache_data
def load_data():
    """
    Chargement des données - essaie de lire un fichier CSV s'il existe, sinon utilise des données fictives
    """
    try:
        df = pd.read_csv('autism_screening.csv')
        st.success("✅ Fichier de données chargé avec succès!")
        return df, True
    except FileNotFoundError:
        st.warning("⚠️ Fichier CSV non trouvé. Utilisation de données de démonstration.")
        
        # Création de données de démonstration réalistes
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

@st.cache_resource
def train_model(df):
    """
    Entraînement du modèle Random Forest
    """
    target_col = 'Class_ASD'
    
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    # Encodage des variables catégorielles
    categorical_cols = X.select_dtypes(include=['object']).columns
    le_dict = {}
    
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        le_dict[col] = le
    
    # Division des données
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Entraînement
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Évaluation
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, X_train, X_test, y_train, y_test, y_pred, accuracy, le_dict, categorical_cols

# Barre latérale
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2917/2917995.png", width=80)
    st.header("📊 Informations")
    st.markdown("""
    Cette application utilise **Random Forest Classifier** pour prédire les troubles du spectre autistique.
    
    **Caractéristiques utilisées :**
    - Scores A1 à A10
    - Âge
    - Genre
    - Origine ethnique
    - Ictère à la naissance
    - Antécédents familiaux d'autisme
    """)
    
    st.markdown("---")
    st.caption("👩‍💻 Développé avec Streamlit")

# Chargement des données
df, has_file = load_data()

# Affichage des données
col1, col2 = st.columns(2)

with col1:
    st.subheader("📋 Aperçu des données")
    st.dataframe(df.head(10), use_container_width=True)
    
with col2:
    st.subheader("📈 Statistiques")
    st.write(f"- Nombre d'échantillons: **{df.shape[0]}**")
    st.write(f"- Nombre de caractéristiques: **{df.shape[1] - 1}**")
    
    if 'Class_ASD' in df.columns:
        asd_count = df['Class_ASD'].sum()
        no_asd_count = len(df) - asd_count
        st.write(f"- Cas d'autisme: **{asd_count}**")
        st.write(f"- Cas non-autistes: **{no_asd_count}**")
        
        # Graphique de distribution
        fig, ax = plt.subplots()
        ax.bar(['Non-autiste', 'Autiste'], [no_asd_count, asd_count], color=['green', 'red'])
        ax.set_ylabel('Nombre')
        ax.set_title('Distribution des classes')
        st.pyplot(fig)

st.markdown("---")

# Entraînement du modèle
st.subheader("🤖 Entraînement du modèle")

if st.button("🚀 Démarrer l'entraînement", type="primary"):
    with st.spinner("Entraînement du modèle en cours... ⏳"):
        try:
            model, X_train, X_test, y_train, y_test, y_pred, accuracy, le_dict, categorical_cols = train_model(df)
            
            st.success(f"✅ Entraînement réussi! Précision du modèle: **{accuracy:.2%}**")
            
            # Affichage des résultats
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Matrice de confusion")
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                ax.set_xlabel('Prédiction')
                ax.set_ylabel('Réalité')
                ax.set_title('Matrice de confusion')
                st.pyplot(fig)
            
            with col2:
                st.subheader("📈 Rapport de classification")
                report = classification_report(y_test, y_pred, output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df.round(3), use_container_width=True)
            
            st.session_state['model'] = model
            st.session_state['le_dict'] = le_dict
            st.session_state['categorical_cols'] = categorical_cols
            st.session_state['trained'] = True
            
        except Exception as e:
            st.error(f"❌ Erreur lors de l'entraînement: {str(e)}")
else:
    st.info("👆 Cliquez sur 'Démarrer l'entraînement' pour entraîner le modèle")

st.markdown("---")

# Section de prédiction
st.subheader("🔮 Prédiction pour un nouveau cas")

with st.form("prediction_form"):
    st.write("Entrez les données de la personne à évaluer:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Âge (en années)", min_value=1, max_value=100, value=25)
        gender = st.selectbox("Genre", ["m", "f"])
        ethnicity = st.selectbox("Origine ethnique", ["Blanc", "Asiatique", "Noir", "Hispanique", "Autre"])
    
    with col2:
        jaundice = st.selectbox("Ictère à la naissance ?", [0, 1], format_func=lambda x: "Oui" if x == 1 else "Non")
        family_asd = st.selectbox("Antécédents familiaux d'autisme ?", [0, 1], format_func=lambda x: "Oui" if x == 1 else "Non")
        
        st.write("---")
        st.write("**Résultats du test A1-A10:**")
        a_scores = []
        for i in range(1, 11):
            score = st.selectbox(f"A{i}", [0, 1], key=f"A{i}", label_visibility="collapsed")
            a_scores.append(score)
    
    with col3:
        st.write("---")
        st.write("**Résumé des scores:**")
        total_a_score = sum(a_scores)
        st.metric("Score total A1-A10", f"{total_a_score} / 10")
        
        if total_a_score >= 6:
            st.warning("⚠️ Score élevé - Peut indiquer un risque d'autisme")
        else:
            st.info("ℹ️ Score bas - Risque plus faible")
    
    submit_button = st.form_submit_button("🔍 Prédire", type="primary")

# Traitement de la prédiction
if submit_button:
    if 'trained' in st.session_state and st.session_state['trained']:
        try:
            input_data = []
            input_data.extend(a_scores)
            input_data.append(age)
            
            gender_encoded = 0 if gender == 'm' else 1
            input_data.append(gender_encoded)
            
            ethnicity_map = {'Blanc': 0, 'Asiatique': 1, 'Noir': 2, 'Hispanique': 3, 'Autre': 4}
            ethnicity_encoded = ethnicity_map[ethnicity]
            input_data.append(ethnicity_encoded)
            
            input_data.append(jaundice)
            input_data.append(family_asd)
            
            model = st.session_state['model']
            input_array = np.array(input_data).reshape(1, -1)
            prediction = model.predict(input_array)[0]
            probability = model.predict_proba(input_array)[0]
            
            st.markdown("---")
            st.subheader("📋 Résultat de l'évaluation")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if prediction == 1:
                    st.error("🚨 **Résultat: Risque élevé de troubles du spectre autistique**")
                else:
                    st.success("✅ **Résultat: Risque faible de troubles du spectre autistique**")
            
            with col2:
                st.write(f"**Probabilité non-autiste:** {probability[0]:.2%}")
                st.write(f"**Probabilité autiste:** {probability[1]:.2%}")
                st.progress(float(probability[1]))
            
            st.info("💡 Remarque: Cette prédiction est basée sur un modèle d'apprentissage automatique et doit être utilisée à titre indicatif uniquement. Consultez un spécialiste pour un diagnostic définitif.")
            
        except Exception as e:
            st.error(f"❌ Erreur lors de la prédiction: {str(e)}")
    else:
        st.warning("⚠️ Veuillez d'abord entraîner le modèle avant d'effectuer une prédiction!")

# Pied de page
st.markdown("---")
st.caption("🧩 Application de prédiction de l'autisme - Développée avec Streamlit")
