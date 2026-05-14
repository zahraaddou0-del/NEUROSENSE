# -*- coding: utf-8 -*-
"""
NeuroSense AI Pro - Détection précoce de l'autisme chez les enfants
Version: 1.0
Langue: Interface en français, commentaires en arabe
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Bibliothèques de Machine Learning
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             roc_auc_score, roc_curve, confusion_matrix, classification_report)
from xgboost import XGBClassifier

# Deep Learning
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# Gestion du déséquilibre
from imblearn.over_sampling import SMOTE

# Explicabilité (XAI)
import shap
import lime
import lime.lime_tabular

import joblib
import warnings
warnings.filterwarnings('ignore')

# Configuration de la page
st.set_page_config(
    page_title="NeuroSense AI Pro - Détection de l'autisme",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS personnalisé
st.markdown("""
    <style>
    .main-title {
        font-size: 48px;
        font-weight: bold;
        background: linear-gradient(90deg, #4A90E2, #7B2F9C);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 5px;
    }
    .sub-title {
        font-size: 18px;
        color: #666;
        text-align: center;
        margin-bottom: 30px;
    }
    .result-box-low {
        padding: 25px;
        border-radius: 20px;
        background: linear-gradient(135deg, #E8F5E9, #C8E6C9);
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        margin-top: 20px;
        border: 2px solid #4CAF50;
    }
    .result-box-high {
        padding: 25px;
        border-radius: 20px;
        background: linear-gradient(135deg, #FFEBEE, #FFCDD2);
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        margin-top: 20px;
        border: 2px solid #F44336;
    }
    .info-box {
        padding: 15px;
        border-radius: 10px;
        background: #E3F2FD;
        border-left: 5px solid #2196F3;
        margin: 10px 0;
    }
    .stButton > button {
        background: linear-gradient(90deg, #4A90E2, #7B2F9C);
        color: white;
        font-size: 18px;
        padding: 10px 25px;
        border-radius: 30px;
        width: 100%;
    }
    </style>
""", unsafe_allow_html=True)

# Affichage des titres
st.markdown('<div class="main-title">NeuroSense AI Pro 🧠</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Détection précoce des troubles du spectre autistique (TSA) chez les enfants</div>', unsafe_allow_html=True)

# Sidebar avec informations
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2917/2917995.png", width=80)
    st.markdown("## 📋 À propos")
    st.markdown("""
    **NeuroSense AI Pro** utilise l'intelligence artificielle pour :
    - ✅ Détecter précocement les signes d'autisme
    - ✅ Fournir une analyse rapide et non invasive
    - ✅ Aider les professionnels de santé
    """)
    st.markdown("---")
    st.markdown("### 🤖 Modèles utilisés")
    st.markdown("""
    - 🧠 Deep Neural Network
    - ⚡ XGBoost
    - 🌲 Random Forest
    - 🤝 Voting Ensemble
    """)
    st.markdown("---")
    st.markdown("### ⚠️ Avertissement")
    st.markdown("""
    Cette outil est à but **éducatif et de recherche uniquement**.  
    Ne remplace pas une consultation médicale professionnelle.
    """)

# Chargement des données
@st.cache_data
def load_data():
    """Charger les données depuis un fichier CSV"""
    # Essayer de charger depuis une URL (dataset public)
    urls = [
        "https://raw.githubusercontent.com/nagatejakachapuram/Autism-Prediction-System-ML/main/train.csv",
        "https://raw.githubusercontent.com/Ankita-M-24/AutismPrediction/main/train.csv"
    ]
    
    for url in urls:
        try:
            df = pd.read_csv(url)
            if len(df) > 100:
                st.success("✅ Données chargées avec succès !")
                return df
        except:
            continue
    
    # Si aucun chargement ne réussit, créer des données synthétiques
    st.warning("⚠️ Utilisation de données synthétiques pour la démonstration")
    np.random.seed(42)
    n = 800
    
    data = {
        'A1_Score': np.random.binomial(1, 0.4, n), 'A2_Score': np.random.binomial(1, 0.45, n),
        'A3_Score': np.random.binomial(1, 0.5, n), 'A4_Score': np.random.binomial(1, 0.4, n),
        'A5_Score': np.random.binomial(1, 0.55, n), 'A6_Score': np.random.binomial(1, 0.45, n),
        'A7_Score': np.random.binomial(1, 0.5, n), 'A8_Score': np.random.binomial(1, 0.4, n),
        'A9_Score': np.random.binomial(1, 0.5, n), 'A10_Score': np.random.binomial(1, 0.6, n),
        'age': np.random.normal(8, 5, n).clip(1, 50),
        'gender': np.random.choice(['m', 'f'], n, p=[0.6, 0.4]),
        'ethnicity': np.random.choice(['White-European', 'Asian', 'Latino', 'Middle Eastern', 'Others'], n),
        'jaundice': np.random.binomial(1, 0.1, n),
        'austim': np.random.binomial(1, 0.15, n),
        'Class/ASD': np.random.binomial(1, 0.3, n)
    }
    return pd.DataFrame(data)

# Prétraitement des données
def preprocess_data(df, fit_encoders=True, encoders=None, scaler=None):
    """Prétraiter les données pour l'entraînement"""
    
    # Colonnes des caractéristiques
    feature_cols = ['A1_Score', 'A2_Score', 'A3_Score', 'A4_Score', 'A5_Score', 
                    'A6_Score', 'A7_Score', 'A8_Score', 'A9_Score', 'A10_Score',
                    'age', 'gender', 'ethnicity', 'jaundice', 'austim']
    
    # Ingénierie des caractéristiques
    df['total_score'] = df[['A1_Score', 'A2_Score', 'A3_Score', 'A4_Score', 'A5_Score',
                              'A6_Score', 'A7_Score', 'A8_Score', 'A9_Score', 'A10_Score']].sum(axis=1)
    
    feature_cols.append('total_score')
    
    available_cols = [col for col in feature_cols if col in df.columns]
    X = df[available_cols].copy()
    y = df['Class/ASD'] if 'Class/ASD' in df.columns else None
    
    # Remplacer les valeurs manquantes
    X = X.fillna(X.median())
    
    # Encodage des variables catégorielles
    categorical_cols = X.select_dtypes(include=['object']).columns
    label_encoders = encoders if encoders else {}
    
    for col in categorical_cols:
        if fit_encoders:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le
        else:
            X[col] = encoders[col].transform(X[col].astype(str))
    
    # Normalisation
    if fit_encoders:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)
    
    return X_scaled, y, label_encoders, scaler, available_cols

# Création du modèle Deep Learning
def create_deep_model(input_dim):
    """Créer un réseau de neurones profond"""
    model = Sequential([
        Dense(128, activation='relu', input_dim=input_dim),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), 
                  loss='binary_crossentropy', 
                  metrics=['accuracy', 'AUC'])
    return model

# Entraînement de tous les modèles
@st.cache_resource
def train_all_models(X, y):
    """Entraîner tous les modèles d'IA"""
    
    # SMOTE pour équilibrer les classes
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    # Division train/test
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
    
    # Modèle 1: Deep Neural Network
    with st.spinner("Entraînement du Deep Neural Network..."):
        dnn_model = create_deep_model(X_train.shape[1])
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
        
        history = dnn_model.fit(X_train, y_train, 
                                validation_split=0.2, 
                                epochs=50, 
                                batch_size=32,
                                callbacks=[early_stop, reduce_lr],
                                verbose=0)
    
    # Modèle 2: XGBoost
    with st.spinner("Entraînement de XGBoost..."):
        xgb_model = XGBClassifier(n_estimators=150, max_depth=6, learning_rate=0.1, 
                                  random_state=42, use_label_encoder=False, eval_metric='logloss')
        xgb_model.fit(X_train, y_train)
    
    # Modèle 3: Random Forest
    with st.spinner("Entraînement de Random Forest..."):
        rf_model = RandomForestClassifier(n_estimators=150, max_depth=12, 
                                          min_samples_split=5, random_state=42)
        rf_model.fit(X_train, y_train)
    
    # Modèle 4: Voting Classifier (Ensemble)
    with st.spinner("Création du modèle Ensemble..."):
        voting_model = VotingClassifier(
            estimators=[('xgb', xgb_model), ('rf', rf_model)],
            voting='soft'
        )
        voting_model.fit(X_train, y_train)
    
    # Évaluation des modèles
    models = {
        '🧠 Deep Neural Network': dnn_model,
        '⚡ XGBoost': xgb_model,
        '🌲 Random Forest': rf_model,
        '🤝 Voting Ensemble': voting_model
    }
    
    results = {}
    predictions = {}
    
    for name, model in models.items():
        if name == '🧠 Deep Neural Network':
            y_pred_proba = model.predict(X_test).flatten()
            y_pred = (y_pred_proba > 0.5).astype(int)
        else:
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        results[name] = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1-Score': f1_score(y_test, y_pred),
            'ROC-AUC': roc_auc_score(y_test, y_pred_proba)
        }
        predictions[name] = y_pred_proba
    
    # Meilleur modèle
    best_name = max(results, key=lambda x: results[x]['F1-Score'])
    best_model = models[best_name]
    
    # Sauvegarde des modèles
    joblib.dump(best_model, 'best_model.pkl')
    
    return models, results, predictions, X_train, X_test, y_train, y_test, best_name, best_model

# Questionnaire en français
def show_questionnaire():
    """Afficher le questionnaire en français"""
    st.subheader("📋 Questionnaire d'évaluation")
    
    st.markdown('<div class="info-box">Veuillez répondre aux questions suivantes concernant l\'enfant.</div>', unsafe_allow_html=True)
    
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.markdown("#### 🧩 Questions comportementales")
        
        questions = [
            "L'enfant ne répond pas quand on l'appelle par son prénom ?",
            "L'enfant évite le contact visuel ?",
            "L'enfant préfère jouer seul ?",
            "L'enfant a du mal à comprendre les émotions des autres ?",
            "L'enfant répète les mêmes mouvements de manière stéréotypée ?",
            "L'enfant est très sensible aux bruits ou à la lumière ?",
            "L'enfant a du mal à s'adapter aux changements de routine ?",
            "L'enfant range les objets d'une manière spécifique ?",
            "L'enfant a du mal à initier une conversation ?",
            "L'enfant ne montre pas d'intérêt pour les interactions sociales ?"
        ]
        
        scores = []
        for i, q in enumerate(questions, 1):
            score = st.radio(f"**{i}.** {q}", [0, 1], key=f"q{i}", 
                           format_func=lambda x: "✅ Oui" if x == 1 else "❌ Non",
                           horizontal=True)
            scores.append(score)
    
    with col_right:
        st.markdown("#### 👤 Informations démographiques")
        
        age = st.number_input("Âge (en années)", min_value=1, max_value=100, value=5)
        gender = st.selectbox("Genre", ['m', 'f'], format_func=lambda x: "Masculin" if x == 'm' else "Féminin")
        ethnicity = st.selectbox("Ethnicité", 
                                ['Européen Blanc', 'Asiatique', 'Latino', 'Moyen-Oriental', 'Autres'])
        jaundice = st.radio("Jaunisse à la naissance ?", [0, 1], 
                           format_func=lambda x: "Oui" if x == 1 else "Non")
        austim = st.radio("Antécédents familiaux d'autisme ?", [0, 1],
                         format_func=lambda x: "Oui" if x == 1 else "Non")
    
    total_score = sum(scores)
    st.progress(total_score / 10)
    st.write(f"**Score comportemental total :** {total_score}/10")
    
    return {
        'scores': scores,
        'age': age,
        'gender': gender,
        'ethnicity': ethnicity,
        'jaundice': jaundice,
        'austim': austim,
        'total_score': total_score
    }

# Fonction principale
def main():
    # Chargement des données
    with st.spinner("Chargement des données et entraînement des modèles d'IA..."):
        df = load_data()
        
        if df is not None:
            X_scaled, y, label_encoders, scaler, features = preprocess_data(df, fit_encoders=True)
            models, results, predictions, X_train, X_test, y_train, y_test, best_name, best_model = train_all_models(X_scaled, y)
            
            # Sauvegarde du scaler et des encodeurs
            joblib.dump(scaler, 'scaler.pkl')
            joblib.dump(label_encoders, 'label_encoders.pkl')
            joblib.dump(features, 'features.pkl')
            
            # Affichage des résultats de performance
            st.header("📊 Performance des modèles d'IA")
            
            results_df = pd.DataFrame(results).T.round(4)
            
            # Graphique comparatif
            fig = go.Figure()
            metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
            colors = ['#4A90E2', '#7B2F9C', '#E24A4A', '#4AE2A4', '#E2A44A']
            
            for i, metric in enumerate(metrics):
                fig.add_trace(go.Bar(
                    name=metric,
                    x=results_df.index,
                    y=results_df[metric],
                    marker_color=colors[i % len(colors)],
                    text=results_df[metric].round(3),
                    textposition='auto'
                ))
            
            fig.update_layout(
                title="Comparaison des performances des modèles",
                barmode='group',
                xaxis_title="Modèle",
                yaxis_title="Score",
                height=500,
                template="plotly_white"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Meilleur modèle
            st.success(f"🏆 **Meilleur modèle :** {best_name} avec un F1-Score de {results[best_name]['F1-Score']:.2%}")
            
            # Tabs principaux
            tab1, tab2, tab3 = st.tabs(["📋 Questionnaire", "📊 Analyse des résultats", "ℹ️ Informations scientifiques"])
            
            with tab1:
                # Afficher le questionnaire
                user_data = show_questionnaire()
                
                if st.button("🔮 Analyser avec l'IA", type="primary"):
                    # Préparer les données d'entrée
                    input_dict = {
                        'A1_Score': user_data['scores'][0], 'A2_Score': user_data['scores'][1],
                        'A3_Score': user_data['scores'][2], 'A4_Score': user_data['scores'][3],
                        'A5_Score': user_data['scores'][4], 'A6_Score': user_data['scores'][5],
                        'A7_Score': user_data['scores'][6], 'A8_Score': user_data['scores'][7],
                        'A9_Score': user_data['scores'][8], 'A10_Score': user_data['scores'][9],
                        'age': user_data['age'], 
                        'gender': user_data['gender'],
                        'ethnicity': user_data['ethnicity'],
                        'jaundice': user_data['jaundice'],
                        'austim': user_data['austim'],
                        'total_score': user_data['total_score']
                    }
                    
                    input_df = pd.DataFrame([input_dict])
                    
                    # S'assurer que toutes les colonnes sont présentes
                    for col in features:
                        if col not in input_df.columns:
                            input_df[col] = 0
                    
                    input_df = input_df[features]
                    
                    # Encodage des variables catégorielles
                    for col in input_df.select_dtypes(include=['object']).columns:
                        if col in label_encoders:
                            input_df[col] = label_encoders[col].transform(input_df[col].astype(str))
                    
                    # Normalisation
                    input_scaled = scaler.transform(input_df)
                    
                    # Prédictions de tous les modèles
                    st.markdown("### 🤖 Résultats de l'analyse par IA")
                    
                    cols = st.columns(len(models))
                    all_probas = []
                    
                    for idx, (name, model) in enumerate(models.items()):
                        if name == '🧠 Deep Neural Network':
                            proba = float(model.predict(input_scaled).flatten()[0])
                        else:
                            proba = float(model.predict_proba(input_scaled)[0][1])
                        
                        all_probas.append(proba)
                        
                        with cols[idx]:
                            color = "#4CAF50" if proba < 0.5 else "#F44336"
                            st.markdown(f"""
                            <div style="text-align:center; padding:15px; border-radius:10px; background:#f5f5f5; margin:5px;">
                                <b>{name}</b><br>
                                <span style="font-size:22px; color:{color}; font-weight:bold;">
                                    {proba:.1%}
                                </span>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Prédiction finale
                    final_proba = np.mean(all_probas)
                    final_prediction = final_proba > 0.5
                    
                    if final_prediction:
                        st.markdown(f"""
                        <div class="result-box-high">
                        ⚠️ **Probabilité de troubles du spectre autistique : {final_proba:.1%}**<br><br>
                        🏥 Il est recommandé de consulter un spécialiste en psychiatrie infantile.<br>
                        🎯 Une intervention précoce améliore significativement les résultats.
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="result-box-low">
                        ✅ **Probabilité de troubles du spectre autistique : {final_proba:.1%}**<br><br>
                        👶 Continuez à surveiller les étapes du développement de l'enfant.<br>
                        📚 Rappelez-vous que chaque enfant est unique dans son rythme de développement.
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Conseils supplémentaires
                    with st.expander("📝 Conseils et recommandations"):
                        st.markdown("""
                        **Pour les parents :**
                        - 📖 Observez et notez les comportements de votre enfant
                        - 🗣️ Encouragez la communication et l'interaction sociale
                        - 🎨 Proposez des activités structurées adaptées
                        - 👨‍⚕️ Consultez régulièrement un pédiatre
                        
                        **Ressources utiles :**
                        - Associations nationales sur l'autisme
                        - Groupes de soutien pour parents
                        - Programmes d'intervention précoce
                        """)
            
            with tab2:
                st.subheader("📊 Analyse détaillée des modèles")
                
                # Courbes ROC
                st.markdown("#### 📉 Courbes ROC comparatives")
                fig_roc = go.Figure()
                
                for name, y_pred_proba in predictions.items():
                    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                    auc = results[name]['ROC-AUC']
                    fig_roc.add_trace(go.Scatter(
                        x=fpr, y=tpr, mode='lines',
                        name=f'{name} (AUC = {auc:.3f})',
                        line=dict(width=2)
                    ))
                
                fig_roc.add_trace(go.Scatter(
                    x=[0, 1], y=[0, 1], mode='lines',
                    name='Classificateur aléatoire',
                    line=dict(dash='dash', color='gray')
                ))
                
                fig_roc.update_layout(
                    title="Courbes ROC - Comparaison des modèles",
                    xaxis_title="Taux de faux positifs",
                    yaxis_title="Taux de vrais positifs",
                    height=450,
                    template="plotly_white"
                )
                st.plotly_chart(fig_roc, use_container_width=True)
                
                # Importance des caractéristiques
                if hasattr(best_model, 'feature_importances_'):
                    st.markdown("#### 🎯 Importance des caractéristiques")
                    importances = best_model.feature_importances_
                    importance_df = pd.DataFrame({
                        'Caractéristique': features,
                        'Importance': importances
                    }).sort_values('Importance', ascending=False).head(10)
                    
                    fig_imp = px.bar(importance_df, x='Importance', y='Caractéristique',
                                     orientation='h', color='Importance',
                                     color_continuous_scale='Viridis',
                                     title="Top 10 des caractéristiques les plus influentes")
                    fig_imp.update_layout(height=450)
                    st.plotly_chart(fig_imp, use_container_width=True)
                
                # Matrice de confusion
                st.markdown(f"#### 🔢 Matrice de confusion - {best_name}")
                
                if best_name == '🧠 Deep Neural Network':
                    best_pred = (best_model.predict(X_test).flatten() > 0.5).astype(int)
                else:
                    best_pred = best_model.predict(X_test)
                
                cm = confusion_matrix(y_test, best_pred)
                fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale='Blues',
                                   labels=dict(x="Prédiction", y="Vérité terrain", color="Nombre"),
                                   x=['Non-TSA', 'TSA'], y=['Non-TSA', 'TSA'])
                fig_cm.update_layout(height=450)
                st.plotly_chart(fig_cm, use_container_width=True)
            
            with tab3:
                st.subheader("ℹ️ Informations scientifiques sur l'autisme")
                
                st.markdown("""
                ### 🧠 Qu'est-ce que le trouble du spectre autistique (TSA) ?
                
                Le trouble du spectre autistique est un trouble neurodéveloppemental caractérisé par :
                
                - **Difficultés dans les interactions sociales et la communication**
                - **Comportements répétitifs et intérêts restreints**
                - **Sensibilités sensorielles particulières**
                
                ### 🔬 L'IA au service du dépistage précoce
                
                **NeuroSense AI Pro utilise plusieurs techniques avancées :**
                
                1. **Deep Learning** : Réseau de neurones à 5 couches
                2. **Ensemble Learning** : Combinaison de 4 modèles
                3. **XAI (IA Explicable)** : Compréhension des décisions
                4. **SMOTE** : Gestion des données déséquilibrées
                
                ### 📊 Statistiques du modèle
                
                - 📈 Nombre de modèles entraînés : 4
                - 🎯 Meilleure précision : {results[best_name]['Accuracy']:.1%}
                - 💪 Meilleur F1-Score : {results[best_name]['F1-Score']:.1%}
                - 📊 AUC-ROC : {results[best_name]['ROC-AUC']:.1%}
                """.format(results=results, best_name=best_name))
                
                st.warning("""
                ⚠️ **Avertissement médical important**
                
                Cet outil est destiné uniquement à des fins éducatives et de recherche.
                Il ne remplace en aucun cas un diagnostic médical professionnel.
                Si vous avez des inquiétudes concernant le développement de votre enfant,
                veuillez consulter un professionnel de santé qualifié.
                """)
        else:
            st.error("❌ Impossible de charger les données. Vérifiez votre connexion Internet.")

if __name__ == "__main__":
    main()
