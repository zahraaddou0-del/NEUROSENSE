import streamlit as st
import numpy as np
import random
from datetime import datetime

# إعدادات الصفحة
st.set_page_config(page_title="NeuroSense AI+", page_icon="🧠", layout="wide")

# تهيئة الجلسة
if 'step' not in st.session_state:
    st.session_state.step = 1

# العنوان الرئيسي
st.title("🧠 NeuroSense AI+")
st.markdown("*Détection précoce de l'autisme par Intelligence Artificielle*")
st.divider()

# ========== الخطوة 1 ==========
if st.session_state.step == 1:
    st.header("1. 📋 Choix du profil")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("👨‍👩‍👦 Parent", use_container_width=True):
            st.session_state.user_type = "Parent"
            st.session_state.step = 2
            st.rerun()
    with col2:
        if st.button("⚕️ Professionnel", use_container_width=True):
            st.session_state.user_type = "Professionnel"
            st.session_state.step = 2
            st.rerun()

# ========== الخطوة 2 ==========
elif st.session_state.step == 2:
    st.header("2. 👶 Profil de l'enfant")
    
    with st.form("profil_form"):
        nom_parent = st.text_input("Nom du parent")
        nom_enfant = st.text_input("Nom de l'enfant")
        age_enfant = st.number_input("Âge de l'enfant (mois)", 0, 72, 24)
        
        if st.form_submit_button("✅ Continuer"):
            if nom_parent and nom_enfant:
                st.session_state.nom_parent = nom_parent
                st.session_state.nom_enfant = nom_enfant
                st.session_state.age_enfant = age_enfant
                st.session_state.step = 3
                st.rerun()
            else:
                st.error("Veuillez remplir tous les champs")

# ========== الخطوة 3 ==========
elif st.session_state.step == 3:
    st.header("3. 🧪 Test intelligent")
    
    st.info("Cliquez sur le bouton ci-dessous pour lancer l'analyse IA")
    
    if st.button("🚀 LANCER L'ANALYSE", type="primary", use_container_width=True):
        st.session_state.step = 4
        st.rerun()

# ========== الخطوة 4: Questionnaire ==========
elif st.session_state.step == 4:
    st.header("4. 📝 Questionnaire d'évaluation")
    
    st.write("Veuillez répondre aux questions suivantes (0 = jamais, 10 = toujours)")
    
    q1 = st.slider("L'enfant évite-t-il le contact visuel ?", 0, 10, 5)
    q2 = st.slider("L'enfant répond-il à son prénom ?", 0, 10, 5)
    q3 = st.slider("L'enfant a-t-il des comportements répétitifs ?", 0, 10, 5)
    
    if st.button("✅ Enregistrer et continuer", type="primary"):
        # Calcul du score
        score_q = (q1 + q2 + q3) / 3
        st.session_state.score_q = score_q
        
        # Score audio simulé
        st.session_state.score_audio = random.uniform(3, 8)
        
        # Score vision simulé
        st.session_state.score_vision = random.uniform(2, 9)
        
        # Score global
        score_global = (st.session_state.score_q + st.session_state.score_audio + st.session_state.score_vision) / 3
        
        if score_global < 4:
            niveau = "🟢 Faible risque"
            recommandation = "Développement typique"
        elif score_global < 7:
            niveau = "🟠 Risque modéré"
            recommandation = "Consultez un spécialiste"
        else:
            niveau = "🔴 Risque élevé"
            recommandation = "Intervention précoce recommandée"
        
        st.session_state.score_global = round(score_global, 1)
        st.session_state.niveau = niveau
        st.session_state.recommandation = recommandation
        
        st.session_state.step = 5
        st.rerun()

# ========== الخطوة 5: Résultats ==========
elif st.session_state.step == 5:
    st.header("5. 📊 Résultats de l'analyse IA")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Score global", f"{st.session_state.score_global}/10")
    with col2:
        st.metric("Niveau", st.session_state.niveau)
    with col3:
        st.progress(st.session_state.score_global / 10)
    
    st.divider()
    
    # Détails des scores
    st.subheader("Détails par modalité")
    
    scores_data = {
        "Modalité": ["Questionnaire", "Analyse vocale", "Vision"],
        "Score": [
            round(st.session_state.score_q, 1),
            round(st.session_state.score_audio, 1),
            round(st.session_state.score_vision, 1)
        ]
    }
    
    st.dataframe(scores_data, use_container_width=True)
    
    # Recommandation
    st.subheader("💡 Recommandation")
    if "Faible" in st.session_state.niveau:
        st.success(st.session_state.recommandation)
    elif "modéré" in st.session_state.niveau:
        st.warning(st.session_state.recommandation)
    else:
        st.error(st.session_state.recommandation)
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        rapport = f"""
        RAPPORT NEUROSENSE
        ==================
        Enfant: {st.session_state.nom_enfant}
        Score: {st.session_state.score_global}/10
        Niveau: {st.session_state.niveau}
        Recommandation: {st.session_state.recommandation}
        """
        st.download_button("📄 Télécharger rapport", rapport, file_name="rapport.txt")
    
    with col2:
        if st.button("🔄 Nouveau test"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

# شريط جانبي
with st.sidebar:
    st.markdown("## ⚙️ Paramètres")
    langue = st.selectbox("Langue", ["Français", "English", "العربية"])
    notifications = st.toggle("Notifications")
    st.divider()
    st.markdown("### 📜 Historique")
    if 'score_global' in st.session_state:
        st.write(f"Dernier score: {st.session_state.score_global}/10")
    else:
        st.write("Aucun test effectué")
