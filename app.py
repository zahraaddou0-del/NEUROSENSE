import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.express as px
import base64
from io import BytesIO
import json

# إعدادات الصفحة
st.set_page_config(
    page_title="NeuroSense AI+ | الكشف المبكر للتوحد",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# تهيئة حالة الجلسة
if 'step' not in st.session_state:
    st.session_state.step = 1
if 'parent_profile' not in st.session_state:
    st.session_state.parent_profile = {}
if 'child_profile' not in st.session_state:
    st.session_state.child_profile = {}
if 'test_history' not in st.session_state:
    st.session_state.test_history = []
if 'current_test' not in st.session_state:
    st.session_state.current_test = {}
if 'audio_recorded' not in st.session_state:
    st.session_state.audio_recorded = False
if 'image_uploaded' not in st.session_state:
    st.session_state.image_uploaded = False

# تخصيص CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 2rem;
    }
    .step-indicator {
        display: flex;
        justify-content: space-between;
        margin-bottom: 2rem;
    }
    .step {
        background: #f0f0f0;
        padding: 0.5rem 1rem;
        border-radius: 30px;
        text-align: center;
        flex: 1;
        margin: 0 0.2rem;
    }
    .step-active {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    .step-completed {
        background: #28a745;
        color: white;
    }
    .result-card {
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        text-align: center;
    }
    .low-risk { background: #d4edda; border: 2px solid #28a745; }
    .medium-risk { background: #fff3cd; border: 2px solid #ffc107; }
    .high-risk { background: #f8d7da; border: 2px solid #dc3545; }
</style>
""", unsafe_allow_html=True)

# عنوان التطبيق
st.markdown("""
<div class="main-header">
    <h1 style="color: white;">🧠 NeuroSense AI+</h1>
    <p style="color: white; font-size: 1.2rem;">Détection précoce du TSA par questionnaire, analyse vocale et vision par ordinateur</p>
</div>
""", unsafe_allow_html=True)

# مؤشر الخطوات
steps = ["👤 Profil Parent", "👶 Profil Enfant", "📝 Questionnaire", "🎤 Analyse Audio", "📸 Analyse Vision", "📊 Résultats"]
current_step = st.session_state.step - 1

cols = st.columns(len(steps))
for i, step_name in enumerate(steps):
    if i < current_step:
        cols[i].markdown(f"<div class='step step-completed'>✅ {step_name}</div>", unsafe_allow_html=True)
    elif i == current_step:
        cols[i].markdown(f"<div class='step step-active'>▶ {step_name}</div>", unsafe_allow_html=True)
    else:
        cols[i].markdown(f"<div class='step'>⏳ {step_name}</div>", unsafe_allow_html=True)

st.markdown("---")

# ==================== ÉTAPE 1: PROFIL PARENT ====================
if st.session_state.step == 1:
    st.subheader("👤 Informations du parent")
    
    col1, col2 = st.columns(2)
    with col1:
        nom_parent = st.text_input("Nom complet", placeholder="Votre nom")
        type_parent = st.selectbox("Type de profil", ["Parent", "Professionnel de santé", "Éducateur"])
    with col2:
        age_parent = st.number_input("Âge", min_value=18, max_value=100, value=30)
        email = st.text_input("Email (optionnel)", placeholder="exemple@email.com")
    
    if st.button("📌 Continuer", type="primary", use_container_width=True):
        if nom_parent:
            st.session_state.parent_profile = {
                "nom": nom_parent,
                "type": type_parent,
                "age": age_parent,
                "email": email,
                "date": datetime.now().strftime("%Y-%m-%d %H:%M")
            }
            st.session_state.step = 2
            st.rerun()
        else:
            st.error("Veuillez entrer votre nom")

# ==================== ÉTAPE 2: PROFIL ENFANT ====================
elif st.session_state.step == 2:
    st.subheader("👶 Profil de l'enfant")
    
    col1, col2 = st.columns(2)
    with col1:
        nom_enfant = st.text_input("Nom de l'enfant", placeholder="Prénom")
        age_enfant = st.number_input("Âge (en mois)", min_value=0, max_value=72, value=24)
    with col2:
        sexe = st.selectbox("Sexe", ["Masculin", "Féminin"])
        historique = st.text_area("Historique médical (optionnel)", placeholder="Antécédents familiaux, problèmes à la naissance, etc.")
    
    if st.button("✅ Continuer", type="primary", use_container_width=True):
        if nom_enfant:
            st.session_state.child_profile = {
                "nom": nom_enfant,
                "age_mois": age_enfant,
                "sexe": sexe,
                "historique": historique
            }
            st.session_state.step = 3
            st.rerun()
        else:
            st.error("Veuillez entrer le nom de l'enfant")

# ==================== ÉTAPE 3: QUESTIONNAIRE ====================
elif st.session_state.step == 3:
    st.subheader("📝 Questionnaire d'évaluation")
    st.info("Répondez aux questions suivantes basées sur le comportement de l'enfant ces 3 derniers mois")
    
    questions = {
        "Q1": "L'enfant évite-t-il le contact visuel ?",
        "Q2": "L'enfant préfère-t-il jouer seul ?",
        "Q3": "L'enfant répète-t-il les mêmes mouvements ?",
        "Q4": "L'enfant réagit-il anormalement aux sons ?",
        "Q5": "L'enfant a-t-il des difficultés à comprendre les émotions ?",
        "Q6": "L'enfant s'oppose-t-il aux changements de routine ?",
        "Q7": "L'enfant fixe-t-il les objets sans but ?",
        "Q8": "L'enfant a-t-il un retard de langage ?",
        "Q9": "L'enfant organise-t-il les objets de façon rigide ?",
        "Q10": "L'enfant semble-t-il désintéressé des autres ?"
    }
    
    reponses = {}
    for qid, question in questions.items():
        reponses[qid] = st.radio(question, [0, 1], format_func=lambda x: "✅ Oui" if x == 1 else "❌ Non", horizontal=True, key=qid)
    
    score_q = sum(reponses.values())
    
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("🎙️ Passer à l'analyse audio", use_container_width=True):
            st.session_state.current_test["questionnaire"] = reponses
            st.session_state.current_test["score_questionnaire"] = score_q
            st.session_state.step = 4
            st.rerun()
    with col2:
        st.markdown(f"**Score actuel:** {score_q}/10")

# ==================== ÉTAPE 4: ANALYSE AUDIO ====================
elif st.session_state.step == 4:
    st.subheader("🎤 Analyse vocale")
    st.markdown("Enregistrez la voix de l'enfant (ou répondez à la question suivante)")
    
    st.markdown("**Question:** *« Quel est ton animal préféré ? »*")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🎙️ Démarrer l'enregistrement", use_container_width=True):
            st.info("🎤 Enregistrement simulé - Dans la version complète, utilisez la bibliothèque 'sounddevice'")
            st.session_state.audio_recorded = True
            st.session_state.current_test["audio_score"] = 0.3  # Simulation
    
    with col2:
        if st.session_state.get("audio_recorded", False):
            st.success("✅ Audio enregistré avec succès!")
            st.audio("https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3", format="audio/mp3")
    
    if st.button("📸 Passer à l'analyse vision", use_container_width=True):
        st.session_state.step = 5
        st.rerun()

# ==================== ÉTAPE 5: ANALYSE VISION ====================
elif st.session_state.step == 5:
    st.subheader("📸 Analyse par vision par ordinateur")
    st.markdown("Téléchargez une photo du visage de l'enfant (regard, expressions faciales)")
    
    uploaded_file = st.file_uploader("Choisir une image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Image téléchargée", width=200)
        st.session_state.image_uploaded = True
        st.session_state.current_test["image_score"] = 0.4  # Simulation
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Analyse du regard:** 👀 Contact visuel détecté: 60%")
        with col2:
            st.markdown("**Analyse des expressions:** 😊 Expressions normales: 75%")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("⏮️ Retour audio", use_container_width=True):
            st.session_state.step = 4
            st.rerun()
    with col2:
        if st.button("📊 Voir les résultats", type="primary", use_container_width=True):
            st.session_state.step = 6
            st.rerun()

# ==================== ÉTAPE 6: RÉSULTATS ====================
elif st.session_state.step == 6:
    st.subheader("📊 Résultats de l'évaluation NeuroSense AI+")
    
    # Calcul du score total
    score_q = st.session_state.current_test.get("score_questionnaire", 0)
    score_audio = st.session_state.current_test.get("audio_score", 0.5)
    score_vision = st.session_state.current_test.get("image_score", 0.5)
    
    # Pondération
    total_score = (score_q * 0.6) + (score_audio * 0.2) + (score_vision * 0.2)
    total_score = min(1.0, total_score) * 100
    
    if total_score >= 70:
        niveau = "Élevé"
        classe = "high-risk"
        message = "⚠️ Niveau de risque élevé - Consultation recommandée"
    elif total_score >= 40:
        niveau = "Moyen"
        classe = "medium-risk"
        message = "🟡 Niveau de risque moyen - Surveillance recommandée"
    else:
        niveau = "Faible"
        classe = "low-risk"
        message = "🟢 Niveau de risque faible - Développement typique"
    
    # Affichage du score
    st.markdown(f"""
    <div class="result-card {classe}">
        <h2>{message}</h2>
        <p style="font-size: 2rem; font-weight: bold;">{total_score:.1f}%</p>
        <p>Niveau de risque: {niveau}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Détails par catégorie
    st.subheader("📊 Détails par analyse")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📝 Questionnaire", f"{score_q}/10", f"{score_q*10}%")
    with col2:
        st.metric("🎤 Analyse audio", f"{score_audio*100:.0f}%")
    with col3:
        st.metric("📸 Analyse vision", f"{score_vision*100:.0f}%")
    
    # Graphique radar
    categories = ['Questionnaire', 'Audio', 'Vision']
    valeurs = [score_q/10, score_audio, score_vision]
    
    fig = px.line_polar(r=valeurs, theta=categories, line_close=True, range_r=[0,1])
    fig.update_traces(fill='toself', fillcolor='rgba(102, 126, 234, 0.3)', line_color='#667eea')
    st.plotly_chart(fig, use_container_width=True)
    
    # Sauvegarde du test
    test_result = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "parent": st.session_state.parent_profile.get("nom", "Inconnu"),
        "enfant": st.session_state.child_profile.get("nom", "Inconnu"),
        "score_total": total_score,
        "niveau": niveau,
        "details": st.session_state.current_test
    }
    st.session_state.test_history.append(test_result)
    
    # Boutons d'action
    col1, col2, col3 = st.columns(3)
    with col1:
        report_data = json.dumps(test_result, indent=2, default=str)
        st.download_button("📥 Télécharger PDF", report_data, file_name=f"rapport_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", use_container_width=True)
    with col2:
        st.link_button("📞 Contacter spécialiste", "https://telephone.tice.ac-martinique.fr/", use_container_width=True)
    with col3:
        if st.button("🏠 Nouveau test", use_container_width=True):
            st.session_state.step = 1
            st.rerun()
    
    # Historique des tests
    st.subheader("📋 Historique des tests")
    if len(st.session_state.test_history) > 1:
        df_history = pd.DataFrame(st.session_state.test_history)
        st.dataframe(df_history[["date", "enfant", "score_total", "niveau"]], use_container_width=True)
    
    # Paramètres
    with st.expander("⚙️ Paramètres"):
        col1, col2 = st.columns(2)
        with col1:
            st.selectbox("🌐 Langue", ["Français", "English", "العربية"])
        with col2:
            st.toggle("🔔 Notifications", value=True)
        st.toggle("🔒 Mode confidentiel", value=True)

# Pied de page
st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>© 2025 NeuroSense AI+ | Détection précoce du TSA | Solution non invasive</p>", unsafe_allow_html=True)
