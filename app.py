import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import random
import os
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
import joblib

# ========== PAGE CONFIGURATION ==========
st.set_page_config(
    page_title="NeuroSense AI+ - Détection précoce de l'autisme",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ========== STYLE CSS PERSONNALISÉ ==========
st.markdown("""
<style>
    /* Style général */
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #e4e8f0 100%);
    }
    
    /* Cartes de résultat */
    .result-card {
        border-radius: 20px;
        padding: 25px;
        margin: 20px 0;
        text-align: center;
        animation: fadeIn 0.5s ease-in;
    }
    
    .risk-high {
        background: linear-gradient(135deg, #ff6b6b, #ee5a24);
        color: white;
    }
    
    .risk-moderate {
        background: linear-gradient(135deg, #feca57, #ff9f43);
        color: white;
    }
    
    .risk-low {
        background: linear-gradient(135deg, #48dbfb, #0abde3);
        color: white;
    }
    
    .risk-very-low {
        background: linear-gradient(135deg, #10ac84, #1dd1a1);
        color: white;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* En-tête */
    .header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        border-radius: 20px;
        padding: 30px;
        text-align: center;
        color: white;
        margin-bottom: 30px;
    }
    
    /* Barre de progression */
    .progress-bar {
        background: linear-gradient(90deg, #667eea, #764ba2);
        border-radius: 10px;
        height: 8px;
        transition: width 0.3s ease;
    }
    
    /* Pied de page */
    .footer {
        text-align: center;
        padding: 20px;
        color: #666;
        font-size: 0.8rem;
        border-top: 1px solid #ddd;
        margin-top: 40px;
    }
    
    /* Boutons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 25px;
        font-weight: bold;
        transition: transform 0.2s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

# ========== FICHIER EXCEL ==========
EXCEL_FILE = "neurosense_data.xlsx"

def init_excel_file():
    """Initialiser le fichier Excel avec les colonnes nécessaires"""
    if not os.path.exists(EXCEL_FILE):
        df = pd.DataFrame(columns=[
            "Date", "ID_Unique", "Type_Utilisateur", "Nom_Parent", "Age_Parent",
            "Nom_Enfant", "Age_Enfant", "Sexe_Enfant", "Historique_Familial",
            "Score_Questionnaire", "Score_Max_Questionnaire", "Score_Audio", 
            "Score_Vision", "Score_Global", "Pourcentage_Global",
            "Niveau_Risque", "Recommandation", "Reponses_Detaillees"
        ])
        df.to_excel(EXCEL_FILE, index=False)

def sauvegarder_dans_excel(data):
    """Sauvegarder les données dans le fichier Excel"""
    init_excel_file()
    df_existant = pd.read_excel(EXCEL_FILE)
    df_nouveau = pd.DataFrame([data])
    df_combine = pd.concat([df_existant, df_nouveau], ignore_index=True)
    df_combine.to_excel(EXCEL_FILE, index=False)
    
    # Afficher le message de succès
    st.success("✅ Données sauvegardées avec succès dans le fichier Excel!")

# ========== INITIALISATION SESSION STATE ==========
if 'etape' not in st.session_state:
    st.session_state.etape = 1
if 'type_utilisateur' not in st.session_state:
    st.session_state.type_utilisateur = ""
if 'nom_parent' not in st.session_state:
    st.session_state.nom_parent = ""
if 'age_parent' not in st.session_state:
    st.session_state.age_parent = 0
if 'nom_enfant' not in st.session_state:
    st.session_state.nom_enfant = ""
if 'age_enfant' not in st.session_state:
    st.session_state.age_enfant = 0
if 'sexe_enfant' not in st.session_state:
    st.session_state.sexe_enfant = ""
if 'historique_familial' not in st.session_state:
    st.session_state.historique_familial = ""
if 'score_questionnaire' not in st.session_state:
    st.session_state.score_questionnaire = None
if 'score_audio' not in st.session_state:
    st.session_state.score_audio = None
if 'score_vision' not in st.session_state:
    st.session_state.score_vision = None
if 'score_global' not in st.session_state:
    st.session_state.score_global = None
if 'pourcentage' not in st.session_state:
    st.session_state.pourcentage = None
if 'niveau' not in st.session_state:
    st.session_state.niveau = ""
if 'recommandation' not in st.session_state:
    st.session_state.recommandation = ""
if 'reponses' not in st.session_state:
    st.session_state.reponses = []

# ========== EN-TÊTE ==========
st.markdown("""
<div class="header">
    <h1 style="font-size: 3rem; margin:0;">🧠 NeuroSense AI+</h1>
    <p style="font-size: 1.2rem; margin-top:10px;">
        Détection précoce intelligente de l'autisme par questionnaire, analyse vocale et vision
    </p>
    <p style="font-size: 0.9rem; opacity:0.9;">
        🤖 Alimenté par Intelligence Artificielle | 🔒 Données confidentielles | 📊 Suivi personnalisé
    </p>
</div>
""", unsafe_allow_html=True)

# ========== AFFICHAGE DE LA PROGRESSION ==========
if st.session_state.etape > 1:
    progression = (st.session_state.etape - 1) / 6 * 100
    st.markdown(f"""
    <div style="margin-bottom: 20px;">
        <p style="margin-bottom: 5px;">📊 Progression de l'évaluation</p>
        <div style="background: #e0e0e0; border-radius: 10px;">
            <div class="progress-bar" style="width: {progression}%; border-radius: 10px;"></div>
        </div>
        <p style="text-align: right; font-size: 0.8rem; margin-top: 5px;">{int(progression)}%</p>
    </div>
    """, unsafe_allow_html=True)

# ========== ÉTAPE 1: CHOIX DU PROFIL ==========
if st.session_state.etape == 1:
    st.markdown("## 📋 Étape 1: Choix du profil")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    border-radius: 20px; padding: 2rem; text-align: center; color: white; cursor: pointer;">
            <div style="font-size: 4rem;">👨‍👩‍👧</div>
            <h2 style="margin:0;">Mode Parent</h2>
            <p>Pour les familles qui souhaitent évaluer leur enfant</p>
            <p style="font-size: 0.8rem; opacity:0.9;">✅ Questionnaire complet | ✅ Analyse vocale | ✅ Vision par IA</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("📝 Choisir Mode Parent", key="btn_parent", use_container_width=True):
            st.session_state.type_utilisateur = "Parent"
            st.session_state.etape = 2
            st.rerun()
    
    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                    border-radius: 20px; padding: 2rem; text-align: center; color: white;">
            <div style="font-size: 4rem;">🏥</div>
            <h2 style="margin:0;">Mode Professionnel</h2>
            <p>Pour les médecins et spécialistes</p>
            <p style="font-size: 0.8rem; opacity:0.9;">✅ Analyse détaillée | ✅ Rapport PDF | ✅ Suivi patients</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("👨‍⚕️ Choisir Mode Professionnel", key="btn_pro", use_container_width=True):
            st.session_state.type_utilisateur = "Professionnel"
            st.session_state.etape = 2
            st.rerun()

# ========== ÉTAPE 2: INFORMATIONS PERSONNELLES ==========
elif st.session_state.etape == 2:
    st.markdown("## 👤 Étape 2: Vos informations")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.session_state.nom_parent = st.text_input("📝 Votre nom complet", placeholder="Ex: Marie Dupont")
        st.session_state.age_parent = st.number_input("🎂 Votre âge", min_value=18, max_value=100, step=1)
    
    with col2:
        st.session_state.nom_enfant = st.text_input("👶 Nom de l'enfant", placeholder="Ex: Lucas Dupont")
        st.session_state.age_enfant = st.number_input("📅 Âge de l'enfant (en mois)", min_value=0, max_value=72, step=1,
                                                       help="Pour les enfants de 0 à 6 ans (72 mois)")
        st.session_state.sexe_enfant = st.selectbox("⚥ Sexe de l'enfant", ["", "Masculin", "Féminin"])
    
    st.session_state.historique_familial = st.radio(
        "🧬 Antécédents familiaux d'autisme",
        ["", "Oui, diagnostiqué", "Oui, suspicion", "Non", "Je ne sais pas"]
    )
    
    col1, col2, col3 = st.columns([1,1,1])
    with col2:
        if st.button("➡️ Suivant", use_container_width=True):
            if st.session_state.nom_parent and st.session_state.age_parent > 0:
                st.session_state.etape = 3
                st.rerun()
            else:
                st.error("Veuillez remplir tous les champs obligatoires")

# ========== ÉTAPE 3: QUESTIONNAIRE COMPLET ==========
elif st.session_state.etape == 3:
    st.markdown("## 📝 Étape 3: Questionnaire d'évaluation")
    st.markdown("---")
    st.info("Veuillez répondre aux questions suivantes en fonction du comportement de votre enfant au cours des 3 derniers mois.")
    
    questions = [
        "👁️ Votre enfant regarde-t-il dans les yeux?",
        "🔊 Votre enfant réagit-il quand on appelle son nom?",
        "👉 Votre enfant pointe-t-il du doigt pour montrer quelque chose?",
        "🧸 Votre enfant joue-t-il à faire semblant? (ex: donner à manger à une poupée)",
        "🚫 Votre enfant évite-t-il le contact visuel?",
        "🔄 Votre enfant a-t-il des comportements répétitifs? (ex: se balance, tourne en rond)",
        "😊 Votre enfant partage-t-il son plaisir avec vous? (ex: vous montre un jouet)",
        "😢 Votre enfant semble-t-il insensible à la douleur?",
        "👂 Votre enfant a-t-il des sensibilités aux bruits ou textures?",
        "🗣️ Votre enfant utilise-t-il des mots ou phrases?"
    ]
    
    reponses = []
    for i, q in enumerate(questions):
        col1, col2 = st.columns([4,1])
        with col1:
            reponse = st.radio(
                q,
                ["Toujours", "Souvent", "Parfois", "Rarement", "Jamais"],
                key=f"q{i}",
                horizontal=True,
                label_visibility="collapsed"
            )
        with col2:
            st.markdown("---")
        reponses.append(reponse)
        
        st.session_state.reponses = reponses
    
    col1, col2, col3 = st.columns([1,1,1])
    with col2:
        if st.button("📊 Calculer le score", use_container_width=True):
            # Calcul du score (0-20)
            score_map = {"Toujours": 4, "Souvent": 3, "Parfois": 2, "Rarement": 1, "Jamais": 0}
            score = sum(score_map[r] for r in reponses)
            st.session_state.score_questionnaire = score
            
            # Déterminer le risque
            if score >= 16:
                st.session_state.niveau = "🔴 Risque Élevé"
                st.session_state.recommandation = "Une consultation avec un spécialiste est recommandée dès que possible."
            elif score >= 11:
                st.session_state.niveau = "🟠 Risque Modéré"
                st.session_state.recommandation = "Surveillance attentive et consultation recommandée."
            elif score >= 6:
                st.session_state.niveau = "🟡 Risque Faible"
                st.session_state.recommandation = "Continuer à observer le développement normal."
            else:
                st.session_state.niveau = "🟢 Risque Très Faible"
                st.session_state.recommandation = "Développement typique, continuez le suivi normal."
            
            st.session_state.etape = 4
            st.rerun()

# ========== ÉTAPE 4: ANALYSE VOCALE ==========
elif st.session_state.etape == 4:
    st.markdown("## 🎤 Étape 4: Analyse vocale")
    st.markdown("---")
    
    st.info("Simulation d'analyse vocale - Dans la version réelle, l'enfant serait invité à parler ou à produire des sons.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🗣️ Instructions")
        st.write("""
        1. Placez-vous dans un endroit calme
        2. Cliquez sur 'Démarrer l'enregistrement'
        3. Demandez à votre enfant de répéter: **"Aaaah"** et **"Mama"**
        4. Enregistrez environ 5-10 secondes
        """)
    
    with col2:
        st.markdown("### 📊 Résultats de simulation")
        
        # Simulation des scores audio
        scores_audio = [random.randint(60, 100) for _ in range(4)]
        
        metrics = {
            "Prosodie (intonation)": scores_audio[0],
            "Clarté articulatoire": scores_audio[1],
            "Variabilité vocale": scores_audio[2],
            "Réponse sonore": scores_audio[3]
        }
        
        for metric, score in metrics.items():
            st.progress(score/100, text=f"{metric}: {score}%")
        
        score_audio_total = int(np.mean(scores_audio))
        st.session_state.score_audio = score_audio_total
    
    if st.button("➡️ Continuer vers l'analyse visuelle", use_container_width=True):
        st.session_state.etape = 5
        st.rerun()

# ========== ÉTAPE 5: ANALYSE VISION ==========
elif st.session_state.etape == 5:
    st.markdown("## 👁️ Étape 5: Analyse par Vision (Eye Tracking)")
    st.markdown("---")
    
    st.info("Simulation d'analyse du regard - Dans la version réelle, une caméra suivrait les mouvements oculaires.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 👀 Instructions")
        st.write("""
        1. Placez l'enfant face à l'écran
        2. Cliquez sur 'Démarrer l'analyse'
        3. L'IA suivra les mouvements oculaires
        4. L'analyse dure environ 30 secondes
        """)
        
        if st.button("🎯 Démarrer l'analyse", use_container_width=True):
            with st.spinner("Analyse en cours..."):
                import time
                time.sleep(2)
            st.success("Analyse terminée!")
    
    with col2:
        st.markdown("### 📊 Résultats de simulation")
        
        # Simulation des scores de vision
        metrics_vision = {
            "👁️ Fixation sur les yeux": random.randint(30, 90),
            "🎯 Attention conjointe": random.randint(40, 95),
            "🔄 Poursuite visuelle": random.randint(50, 100),
            "😊 Reconnaissance émotions": random.randint(35, 85)
        }
        
        for metric, score in metrics_vision.items():
            st.progress(score/100, text=f"{metric}: {score}%")
        
        # Graphique radar
        fig = go.Figure(data=go.Scatterpolar(
            r=list(metrics_vision.values()),
            theta=list(metrics_vision.keys()),
            fill='toself',
            marker=dict(color='rgba(102, 126, 234, 0.8)'),
            line=dict(color='rgba(102, 126, 234, 1)', width=2)
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100]),
                angularaxis=dict(tickfont=dict(size=10))
            ),
            showlegend=False,
            height=350,
            margin=dict(l=40, r=40, t=20, b=20)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        score_vision_total = int(np.mean(list(metrics_vision.values())))
        st.session_state.score_vision = score_vision_total
    
    col1, col2, col3 = st.columns([1,1,1])
    with col2:
        if st.button("🔮 Générer le rapport final", use_container_width=True):
            # Calcul du score global
            scores = [s for s in [st.session_state.score_questionnaire, st.session_state.score_audio, st.session_state.score_vision] if s is not None]
            if scores:
                st.session_state.score_global = int(np.mean(scores))
                st.session_state.pourcentage = st.session_state.score_global
            
                # Ajustement du niveau de risque
                if st.session_state.pourcentage >= 16:
                    st.session_state.niveau = "🔴 Risque Élevé"
                elif st.session_state.pourcentage >= 11:
                    st.session_state.niveau = "🟠 Risque Modéré"
                elif st.session_state.pourcentage >= 6:
                    st.session_state.niveau = "🟡 Risque Très Faible"
                else:
                    st.session_state.niveau = "🟢 Développement Typique"
            
            st.session_state.etape = 6
            st.rerun()

# ========== ÉTAPE 6: RAPPORT FINAL ==========
elif st.session_state.etape == 6:
    st.markdown("## 📊 Étape 6: Rapport d'évaluation NeuroSense AI+")
    st.markdown("---")
    
    # Calcul du pourcentage final
    if st.session_state.score_global:
        pourcentage_final = (st.session_state.score_global / 20) * 100
        
        # Déterminer la classe CSS
        if pourcentage_final >= 70:
            risk_class = "risk-high"
        elif pourcentage_final >= 50:
            risk_class = "risk-moderate"
        elif pourcentage_final >= 30:
            risk_class = "risk-low"
        else:
            risk_class = "risk-very-low"
        
        # Carte de résultat
        st.markdown(f"""
        <div class="result-card {risk_class}">
            <div style="font-size: 4rem;">{'⚠️' if pourcentage_final >= 50 else '✅'}</div>
            <h2 style="margin:10px 0;">{st.session_state.niveau}</h2>
            <div style="font-size: 3rem; font-weight: bold; margin:20px 0;">
                {pourcentage_final:.1f}%
            </div>
            <div style="background: rgba(255,255,255,0.2); border-radius: 20px; padding: 15px; margin-top: 10px;">
                <p style="margin:5px 0;"><strong>Score questionnaire:</strong> {st.session_state.score_questionnaire}/20</p>
                <p style="margin:5px 0;"><strong>Score analyse vocale:</strong> {st.session_state.score_audio}/100</p>
                <p style="margin:5px 0;"><strong>Score analyse vision:</strong> {st.session_state.score_vision}/100</p>
                <p style="margin:5px 0;"><strong>Score global:</strong> {st.session_state.score_global}/20</p>
            </div>
            <p style="margin-top:20px; font-size:1.1rem;">
                📌 {st.session_state.recommandation}
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Boutons d'action
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("💾 Sauvegarder les données", use_container_width=True):
            # Préparer les données pour Excel
            data_to_save = {
                "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "ID_Unique": f"NS-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                "Type_Utilisateur": st.session_state.type_utilisateur,
                "Nom_Parent": st.session_state.nom_parent,
                "Age_Parent": st.session_state.age_parent,
                "Nom_Enfant": st.session_state.nom_enfant,
                "Age_Enfant": st.session_state.age_enfant,
                "Sexe_Enfant": st.session_state.sexe_enfant,
                "Historique_Familial": st.session_state.historique_familial,
                "Score_Questionnaire": st.session_state.score_questionnaire,
                "Score_Max_Questionnaire": 20,
                "Score_Audio": st.session_state.score_audio,
                "Score_Vision": st.session_state.score_vision,
                "Score_Global": st.session_state.score_global,
                "Pourcentage_Global": pourcentage_final,
                "Niveau_Risque": st.session_state.niveau,
                "Recommandation": st.session_state.recommandation,
                "Reponses_Detaillees": str(st.session_state.reponses)
            }
            sauvegarder_dans_excel(data_to_save)
    
    with col2:
        if st.button("📄 Exporter en PDF", use_container_width=True):
            st.info("📝 Fonctionnalité d'export PDF à venir dans la prochaine version!")
    
    with col3:
        if st.button("🔄 Nouvelle évaluation", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    
    # Résumé visuel avec Plotly
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📈 Scores détaillés")
        scores_df = pd.DataFrame({
            'Catégorie': ['Questionnaire', 'Analyse vocale', 'Analyse vision'],
            'Score (%)': [
                (st.session_state.score_questionnaire / 20) * 100,
                st.session_state.score_audio,
                st.session_state.score_vision
            ]
        })
        
        fig = px.bar(scores_df, x='Catégorie', y='Score (%)', 
                     color='Catégorie',
                     color_discrete_sequence=['#667eea', '#f093fb', '#4ecdc4'],
                     title="Comparaison des scores par modalité")
        fig.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 📋 Recommandations")
        
        if pourcentage_final >= 70:
            st.error("""
            **🔴 Action immédiate recommandée:**
            - Consultez un pédiatre spécialisé
            - Programme d'intervention précoce
            - Suivi régulier
            """)
        elif pourcentage_final >= 40:
            st.warning("""
            **🟠 Surveillance active:**
            - Consultez un médecin généraliste
            - Stimulation du développement
            - Re-test dans 3 mois
            """)
        else:
            st.success("""
            **🟢 Développement typique:**
            - Continuez les activités stimulantes
            - Suivi normal avec pédiatre
            - Restez attentif aux évolutions
            """)

# ========== PIED DE PAGE ==========
st.markdown("""
<div class="footer">
    <p>🧠 NeuroSense AI+ v2.0 | 🤖 Intelligence Artificielle pour la détection précoce</p>
    <p>🔒 Toutes les données sont confidentielles et stockées localement | ⚠️ Ceci est un outil d'aide à la décision, pas un diagnostic médical</p>
    <p>📧 Contact: support@neurosense.ai | 📱 Application mobile disponible prochainement</p>
</div>
""", unsafe_allow_html=True)
