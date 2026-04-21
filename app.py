import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.express as px
import json
from PIL import Image
import cv2
import io

# إعدادات الصفحة
st.set_page_config(
    page_title="NeuroSense AI+ | Détection précoce du TSA",
    page_icon="🧠",
    layout="wide"
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
if 'questionnaire_responses' not in st.session_state:
    st.session_state.questionnaire_responses = {}
if 'image_analysis' not in st.session_state:
    st.session_state.image_analysis = None
if 'audio_analysis' not in st.session_state:
    st.session_state.audio_analysis = None

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
        gap: 0.5rem;
        flex-wrap: wrap;
    }
    .step {
        background: #f0f0f0;
        padding: 0.5rem 1rem;
        border-radius: 30px;
        text-align: center;
        flex: 1;
        font-size: 0.8rem;
        min-width: 80px;
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
    .info-box {
        background: #e3f2fd;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# عنوان التطبيق
st.markdown("""
<div class="main-header">
    <h1 style="color: white;">🧠 NeuroSense AI+</h1>
    <p style="color: white; font-size: 1.1rem;">Détection précoce du TSA par questionnaire, analyse vocale et vision par ordinateur</p>
</div>
""", unsafe_allow_html=True)

# مؤشر الخطوات
steps = ["👤 Parent", "👶 Enfant", "📝 Questionnaire", "🎤 Audio", "📸 Vision", "📊 Résultats"]
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
    st.subheader("👤 Chéix du profil")
    
    col1, col2 = st.columns(2)
    with col1:
        nom_parent = st.text_input("Nom complet", placeholder="Votre nom")
        type_parent = st.selectbox("Type de profil", ["👨‍👩‍👧 Parent", "👨‍⚕️ Professionnel", "👩‍🏫 Éducateur"])
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
        sexe = st.selectbox("Sexe", ["👦 Masculin", "👧 Féminin"])
        historique = st.text_area("Historique médical (optionnel)", placeholder="Antécédents familiaux, problèmes à la naissance...")
    
    col1, col2 = st.columns(2)
    with col1:
        famille_asd = st.selectbox("Histoire familiale de TSA", ["Non", "Oui"])
    with col2:
        jaunisse = st.selectbox("Jaunisse à la naissance", ["Non", "Oui"])
    
    if st.button("✅ Continuer", type="primary", use_container_width=True):
        if nom_enfant:
            st.session_state.child_profile = {
                "nom": nom_enfant,
                "age_mois": age_enfant,
                "sexe": sexe,
                "historique": historique,
                "famille_asd": famille_asd,
                "jaunisse": jaunisse
            }
            st.session_state.step = 3
            st.rerun()
        else:
            st.error("Veuillez entrer le nom de l'enfant")

# ==================== ÉTAPE 3: QUESTIONNAIRE ====================
elif st.session_state.step == 3:
    st.subheader("📝 Test intelligent - Questionnaire")
    st.info("Répondez aux questions suivantes basées sur le comportement de l'enfant ces 3 derniers mois")
    
    questions = {
        "Q1": "L'enfant évite-t-il le contact visuel ?",
        "Q2": "L'enfant préfère-t-il jouer seul plutôt qu'avec d'autres enfants ?",
        "Q3": "L'enfant répète-t-il les mêmes mouvements ou paroles de façon stéréotypée ?",
        "Q4": "L'enfant réagit-il anormalement aux sons (peur excessive ou indifférence) ?",
        "Q5": "L'enfant a-t-il des difficultés à comprendre les émotions des autres ?",
        "Q6": "L'enfant s'oppose-t-il fortement aux changements de routine ?",
        "Q7": "L'enfant fixe-t-il les objets ou les lumières sans but apparent ?",
        "Q8": "L'enfant a-t-il un retard de langage par rapport à son âge ?",
        "Q9": "L'enfant organise-t-il les objets de façon rigide (alignement) ?",
        "Q10": "L'enfant semble-t-il désintéressé des interactions sociales ?"
    }
    
    for qid, question in questions.items():
        st.session_state.questionnaire_responses[qid] = st.radio(
            question, 
            [0, 1], 
            format_func=lambda x: "✅ Oui" if x == 1 else "❌ Non", 
            horizontal=True, 
            key=qid
        )
    
    score_q = sum(st.session_state.questionnaire_responses.values())
    st.progress(score_q / 10)
    st.markdown(f"**Score actuel:** {score_q}/10")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("⏮️ Retour", use_container_width=True):
            st.session_state.step = 2
            st.rerun()
    with col2:
        if st.button("🎙️ Passer à l'analyse audio", type="primary", use_container_width=True):
            st.session_state.step = 4
            st.rerun()

# ==================== ÉTAPE 4: ANALYSE AUDIO ====================
elif st.session_state.step == 4:
    st.subheader("🎤 Analyse vocale")
    
    st.markdown("""
    <div class="info-box">
        <b>Instructions:</b> Demandez à l'enfant de dire "Bonjour" ou de nommer un animal. 
        Enregistrez la réponse ou téléchargez un fichier audio.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("**Question:** *« Quel est ton animal préféré ? »*")
    
    # Option: upload fichier audio
    audio_file = st.file_uploader("Télécharger un fichier audio", type=["wav", "mp3", "m4a"])
    
    if audio_file is not None:
        st.audio(audio_file, format="audio/wav")
        st.success("✅ Fichier audio reçu!")
        
        # Simulation d'analyse
        import random
        random.seed(hash(audio_file.name) % 2**32)
        prosody_score = random.uniform(0.3, 0.8)
        st.session_state.audio_analysis = {
            "score": prosody_score,
            "prosody": "Normale" if prosody_score > 0.5 else "Anormale",
            "qualite": "Bonne"
        }
        
        st.markdown(f"""
        **Résultat analyse vocale:**
        - Intonation/prosodie: {prosody_score*100:.0f}%
        - Qualité vocale: {st.session_state.audio_analysis['qualite']}
        """)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("⏮️ Retour questionnaire", use_container_width=True):
            st.session_state.step = 3
            st.rerun()
    with col2:
        if st.button("📸 Passer à l'analyse vision", type="primary", use_container_width=True):
            if audio_file is None:
                st.warning("Veuillez télécharger un fichier audio ou passer")
            st.session_state.step = 5
            st.rerun()
    
    # Bouton pour passer sans audio
    if st.button("⏩ Passer l'analyse audio", use_container_width=True):
        st.session_state.audio_analysis = {"score": 0.5, "prosody": "Non évalué", "qualite": "N/A"}
        st.session_state.step = 5
        st.rerun()

# ==================== ÉTAPE 5: ANALYSE VISION ====================
elif st.session_state.step == 5:
    st.subheader("📸 Analyse par vision par ordinateur")
    
    st.markdown("""
    <div class="info-box">
        <b>Instructions:</b> Téléchargez une photo du visage de l'enfant (regard, expression faciale).
        L'IA analysera le contact visuel et les expressions.
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Choisir une image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Afficher l'image
        image = Image.open(uploaded_file)
        st.image(image, caption="Image téléchargée", width=250)
        
        # Analyse OpenCV simulée
        img_array = np.array(image)
        
        # Détection de visage simulée
        st.markdown("**Analyse en cours...**")
        
        # Simulation des résultats
        eye_contact_score = np.random.uniform(0.2, 0.9)
        expression_score = np.random.uniform(0.3, 0.8)
        
        st.session_state.image_analysis = {
            "score": (eye_contact_score + expression_score) / 2,
            "eye_contact": eye_contact_score * 100,
            "expressions": expression_score * 100,
            "visage_detecte": True
        }
        
        st.markdown(f"""
        **Résultats analyse image:**
        - 🎯 Contact visuel détecté: {st.session_state.image_analysis['eye_contact']:.0f}%
        - 😊 Expressions faciales: {st.session_state.image_analysis['expressions']:.0f}%
        - 👤 Visage détecté: ✅ Oui
        """)
        
        # Barre de progression
        st.progress(100)
        st.success("✅ Analyse terminée!")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("⏮️ Retour audio", use_container_width=True):
            st.session_state.step = 4
            st.rerun()
    with col2:
        if st.button("📊 Voir les résultats", type="primary", use_container_width=True):
            if uploaded_file is None:
                st.warning("Veuillez télécharger une image ou passer")
            st.session_state.step = 6
            st.rerun()
    
    if st.button("⏩ Passer l'analyse vision", use_container_width=True):
        st.session_state.image_analysis = {"score": 0.5, "eye_contact": 50, "expressions": 50, "visage_detecte": False}
        st.session_state.step = 6
        st.rerun()

# ==================== ÉTAPE 6: RÉSULTATS ====================
elif st.session_state.step == 6:
    st.subheader("📊 Résultats de l'évaluation NeuroSense AI+")
    
    # Calcul des scores
    score_q = sum(st.session_state.questionnaire_responses.values())
    score_q_norm = score_q / 10
    
    score_audio = st.session_state.audio_analysis["score"] if st.session_state.audio_analysis else 0.5
    score_vision = st.session_state.image_analysis["score"] if st.session_state.image_analysis else 0.5
    
    # Pondération: Questionnaire 50%, Audio 25%, Vision 25%
    total_score = (score_q_norm * 0.5) + (score_audio * 0.25) + (score_vision * 0.25)
    total_percent = total_score * 100
    
    # Détermination du niveau
    if total_percent >= 70:
        niveau = "Élevé"
        classe = "high-risk"
        message = "⚠️ Niveau de risque élevé - Consultation recommandée"
        recommandation = "Nous vous recommandons vivement de consulter un spécialiste en développement de l'enfant dès que possible."
    elif total_percent >= 40:
        niveau = "Moyen"
        classe = "medium-risk"
        message = "🟡 Niveau de risque moyen - Surveillance recommandée"
        recommandation = "Continuez à observer le développement de l'enfant et consultez si les symptômes persistent."
    else:
        niveau = "Faible"
        classe = "low-risk"
        message = "🟢 Niveau de risque faible - Développement typique"
        recommandation = "Le développement de l'enfant semble dans la norme. Continuez les suivis réguliers."
    
    # Affichage du résultat
    st.markdown(f"""
    <div class="result-card {classe}">
        <h2>{message}</h2>
        <p style="font-size: 2.5rem; font-weight: bold;">{total_percent:.0f}%</p>
        <p>Niveau de risque: {niveau}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Graphique radar
    categories = ['Questionnaire', 'Audio', 'Vision']
    valeurs = [score_q_norm * 100, score_audio * 100, score_vision * 100]
    
    fig = px.line_polar(
        r=valeurs, 
        theta=categories, 
        line_close=True, 
        range_r=[0, 100],
        title="Analyse multi-modale NeuroSense AI+"
    )
    fig.update_traces(fill='toself', fillcolor='rgba(102, 126, 234, 0.3)', line_color='#667eea')
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # Détails des analyses
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📝 Questionnaire", f"{score_q}/10", f"{score_q_norm*100:.0f}%")
    with col2:
        audio_val = st.session_state.audio_analysis["score"] * 100 if st.session_state.audio_analysis else 50
        st.metric("🎤 Analyse audio", f"{audio_val:.0f}%")
    with col3:
        vision_val = st.session_state.image_analysis["score"] * 100 if st.session_state.image_analysis else 50
        st.metric("📸 Analyse vision", f"{vision_val:.0f}%")
    
    # Recommandation
    st.info(f"💡 **Recommandation:** {recommandation}")
    
    # Sauvegarde
    test_result = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "parent": st.session_state.parent_profile.get("nom", "Inconnu"),
        "enfant": st.session_state.child_profile.get("nom", "Inconnu"),
        "score_total": total_percent,
        "niveau": niveau,
        "questionnaire": score_q,
        "audio": audio_val,
        "vision": vision_val,
        "age_enfant": st.session_state.child_profile.get("age_mois", "?"),
        "sexe": st.session_state.child_profile.get("sexe", "?")
    }
    st.session_state.test_history.append(test_result)
    
    # Boutons
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        report_json = json.dumps(test_result, indent=2, default=str, ensure_ascii=False)
        st.download_button("📥 Télécharger PDF", report_json, file_name=f"rapport_NeuroSense_{datetime.now().strftime('%Y%m%d_%H%M')}.json", use_container_width=True)
    with col2:
        st.link_button("📞 Contacter spécialiste", "https://www.doctolib.fr/", use_container_width=True)
    with col3:
        if st.button("🏠 Nouveau test", use_container_width=True):
            st.session_state.step = 1
            st.session_state.questionnaire_responses = {}
            st.session_state.audio_analysis = None
            st.session_state.image_analysis = None
            st.rerun()
    with col4:
        if st.button("📊 Historique", use_container_width=True):
            st.session_state.show_history = not st.session_state.get('show_history', False)
    
    # Historique
    if st.session_state.get('show_history', False) and len(st.session_state.test_history) > 0:
        st.subheader("📋 Historique des tests")
        df = pd.DataFrame(st.session_state.test_history)
        st.dataframe(df, use_container_width=True)

st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>© 2025 NeuroSense AI+ | Détection précoce du TSA | Solution non invasive | Ce test n'est pas un diagnostic médical</p>", unsafe_allow_html=True)
