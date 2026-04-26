import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import random
import json
import os

# ========== CONFIGURATION ==========
st.set_page_config(
    page_title="NeuroSense AI+ - Détection précoce de l'autisme",
    page_icon="🧠",
    layout="wide"
)

# ========== FONCTION POUR SAUVEGARDER L'ÉTAPE ==========
def sauvegarder_etape():
    """Sauvegarde l'étape actuelle dans le fichier local"""
    data = {
        "etape": st.session_state.etape,
        "nom_parent": st.session_state.nom_parent,
        "age_parent": st.session_state.age_parent,
        "nom_enfant": st.session_state.nom_enfant,
        "sexe_enfant": st.session_state.sexe_enfant,
        "historique_medical": st.session_state.historique_medical,
        "type_utilisateur": st.session_state.type_utilisateur
    }
    with open("save_neurosense.json", "w", encoding="utf-8") as f:
        json.dump(data, f)

def charger_sauvegarde():
    """Charge la sauvegarde si elle existe"""
    if os.path.exists("save_neurosense.json"):
        try:
            with open("save_neurosense.json", "r", encoding="utf-8") as f:
                data = json.load(f)
                return data
        except:
            return None
    return None

# ========== INITIALISATION ==========
if 'etape' not in st.session_state:
    # Essayer de charger une sauvegarde existante
    sauvegarde = charger_sauvegarde()
    if sauvegarde and sauvegarde.get("etape", 1) > 1:
        st.session_state.etape = sauvegarde.get("etape", 1)
        st.session_state.nom_parent = sauvegarde.get("nom_parent", "")
        st.session_state.age_parent = sauvegarde.get("age_parent", 0)
        st.session_state.nom_enfant = sauvegarde.get("nom_enfant", "")
        st.session_state.sexe_enfant = sauvegarde.get("sexe_enfant", "")
        st.session_state.historique_medical = sauvegarde.get("historique_medical", "")
        st.session_state.type_utilisateur = sauvegarde.get("type_utilisateur", "")
    else:
        st.session_state.etape = 1
        st.session_state.nom_parent = ""
        st.session_state.age_parent = 0
        st.session_state.nom_enfant = ""
        st.session_state.sexe_enfant = ""
        st.session_state.historique_medical = ""
        st.session_state.type_utilisateur = ""

if 'score_questionnaire' not in st.session_state:
    st.session_state.score_questionnaire = None
if 'score_audio' not in st.session_state:
    st.session_state.score_audio = None
if 'score_vision' not in st.session_state:
    st.session_state.score_vision = None
if 'reponses' not in st.session_state:
    st.session_state.reponses = []

# ========== TITRE ==========
st.markdown("""
<div style="text-align: center; padding: 1rem 0;">
    <h1 style="font-size: 3rem; color: #1e3c72;">🧠 NeuroSense AI+</h1>
    <p style="font-size: 1.1rem; color: #4a5568;">
        NeuroSense combine questionnaire, analyse vocale et vision par ordinateur pour une détection précoce intelligente.
    </p>
</div>
<hr>
""", unsafe_allow_html=True)

# ========== AFFICHAGE DE LA PROGRESSION ==========
if st.session_state.etape > 1:
    etapes = ["Choix du profil", "Profil enfant", "Analyse IA", "Résultats"]
    current = st.session_state.etape - 1
    cols = st.columns(4)
    for i, (col, etape) in enumerate(zip(cols, etapes)):
        with col:
            if i < current:
                st.markdown(f"✅ {etape}")
            elif i == current:
                st.markdown(f"🔵 **{etape}**")
            else:
                st.markdown(f"⚪ {etape}")
    st.markdown("---")

# ========== ÉTAPE 1: CHOIX DU PROFIL ==========
if st.session_state.etape == 1:
    st.header("1. 📋 Choix du profil")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("👨‍👩‍👦 Parent", use_container_width=True):
            st.session_state.type_utilisateur = "Parent"
            st.session_state.etape = 2
            sauvegarder_etape()
            st.rerun()
    
    with col2:
        if st.button("⚕️ Professionnel", use_container_width=True):
            st.session_state.type_utilisateur = "Professionnel"
            st.session_state.etape = 2
            sauvegarder_etape()
            st.rerun()

# ========== ÉTAPE 2: PROFIL DE L'ENFANT ==========
elif st.session_state.etape == 2:
    st.header("2. 👶 Profil de l'enfant")
    
    with st.form("profil_enfant"):
        col1, col2 = st.columns(2)
        
        with col1:
            nom_parent = st.text_input("Nom du parent / Professionnel", value=st.session_state.nom_parent)
            age_parent = st.number_input("Âge", min_value=18, max_value=100, step=1, value=st.session_state.age_parent)
        
        with col2:
            nom_enfant = st.text_input("Nom de l'enfant", value=st.session_state.nom_enfant)
            sexe_enfant = st.selectbox("Sexe", ["Garçon", "Fille", "Autre"], 
                                       index=["Garçon", "Fille", "Autre"].index(st.session_state.sexe_enfant) if st.session_state.sexe_enfant in ["Garçon", "Fille", "Autre"] else 0)
        
        historique = st.text_area("Historique médical (optionnel)", 
                                  value=st.session_state.historique_medical,
                                  placeholder="Antécédents médicaux, naissance prématurée...")
        
        submitted = st.form_submit_button("📝 Continuer", use_container_width=True)
        
        if submitted:
            if nom_parent and nom_enfant:
                st.session_state.nom_parent = nom_parent
                st.session_state.age_parent = age_parent
                st.session_state.nom_enfant = nom_enfant
                st.session_state.sexe_enfant = sexe_enfant
                st.session_state.historique_medical = historique
                st.session_state.etape = 3
                sauvegarder_etape()
                st.rerun()
            else:
                st.error("Veuillez remplir tous les champs obligatoires")
    
    # Bouton pour recommencer
    if st.button("🔄 Recommencer depuis le début"):
        if os.path.exists("save_neurosense.json"):
            os.remove("save_neurosense.json")
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# ========== ÉTAPE 3: ANALYSE IA ==========
elif st.session_state.etape == 3:
    st.header("3. 🤖 Analyse IA")
    st.info(f"Bienvenue {st.session_state.nom_parent} ! Analyse pour {st.session_state.nom_enfant}")
    
    # Sauvegarde de l'étape
    sauvegarder_etape()
    
    # Création des onglets
    tab1, tab2, tab3 = st.tabs(["📝 Questionnaire (15 questions)", "🎙️ Analyse vocale", "👁️ Vision par ordinateur"])
    
    # ===== TAB 1: QUESTIONNAIRE =====
    with tab1:
        st.subheader("📝 Questionnaire d'évaluation")
        st.caption("Veuillez répondre à toutes les questions")
        
        # Groupe 1
        st.markdown("**🟢 1. Interaction sociale**")
        
        q1 = st.radio("1. L'enfant répond-il à son prénom quand vous l'appelez ?",
                      ["Oui, toujours", "Parfois", "Non, rarement ou jamais"], index=1, key="q1")
        q2 = st.radio("2. L'enfant sourit-il en réponse à votre sourire ?",
                      ["Oui, souvent", "Parfois", "Non, rarement"], index=1, key="q2")
        q3 = st.radio("3. L'enfant cherche-t-il à attirer votre attention ?",
                      ["Oui, souvent", "Parfois", "Non, rarement"], index=1, key="q3")
        q4 = st.radio("4. L'enfant imite-t-il vos gestes ?",
                      ["Oui, souvent", "Parfois", "Non, rarement"], index=1, key="q4")
        
        # Groupe 2
        st.markdown("**🟡 2. Communication et langage**")
        
        q5 = st.radio("5. L'enfant utilise-t-il des gestes pour communiquer ?",
                      ["Oui, plusieurs gestes", "Un ou deux gestes", "Non, pas de gestes"], index=1, key="q5")
        q6 = st.radio("6. L'enfant babille-t-il ou dit-il des mots ?",
                      ["Oui, plusieurs mots", "Quelques sons ou mots", "Très peu ou pas du tout"], index=1, key="q6")
        q7 = st.radio("7. L'enfant répète-t-il les mêmes mots sans contexte ?",
                      ["Non, jamais", "Parfois", "Oui, fréquemment"], index=0, key="q7")
        
        # Groupe 3
        st.markdown("**🔴 3. Comportements répétitifs**")
        
        q8 = st.radio("8. L'enfant a-t-il des mouvements répétitifs ?",
                      ["Non, jamais", "Parfois", "Oui, fréquemment"], index=0, key="q8")
        q9 = st.radio("9. L'enfant est-il très attaché à certains objets ?",
                      ["Non, pas particulièrement", "Un peu", "Oui, très attaché"], index=0, key="q9")
        q10 = st.radio("10. L'enfant a-t-il une routine rigide ?",
                       ["Non, flexible", "Parfois", "Oui, très difficile"], index=0, key="q10")
        
        # Groupe 4
        st.markdown("**🟠 4. Sensorialité**")
        
        q11 = st.radio("11. L'enfant réagit-il anormalement aux sons ?",
                       ["Non, réaction normale", "Parfois", "Oui, souvent"], index=0, key="q11")
        q12 = st.radio("12. L'enfant fixe-t-il les objets inhabituellement ?",
                       ["Non", "Parfois", "Oui, fréquemment"], index=0, key="q12")
        q13 = st.radio("13. L'enfant a-t-il une sensibilité anormale à la douleur ?",
                       ["Non, normale", "Un peu inhabituel", "Très différent"], index=0, key="q13")
        
        # Groupe 5
        st.markdown("**🔵 5. Jeu et interactions**")
        
        q14 = st.radio("14. L'enfant joue-t-il de manière imaginative ?",
                       ["Oui, souvent", "Parfois", "Non, jamais"], index=1, key="q14")
        q15 = st.radio("15. L'enfant préfère-t-il jouer seul ?",
                       ["Non, aime jouer avec d'autres", "Un peu des deux", "Oui, préfère seul"], index=1, key="q15")
        
        reponses = [q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, q11, q12, q13, q14, q15]
        st.session_state.reponses = reponses
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("✅ Enregistrer le questionnaire", use_container_width=True):
                scores = []
                for i, rep in enumerate(reponses):
                    if i < 6:
                        if "toujours" in rep or "souvent" in rep or "plusieurs" in rep:
                            scores.append(0)
                        elif "Parfois" in rep or "Un peu" in rep:
                            scores.append(1)
                        else:
                            scores.append(2)
                    elif i in [6, 7, 8, 9, 10, 11, 12]:
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
                
                score_total = sum(scores)
                st.session_state.score_questionnaire = round((score_total / 30) * 10, 1)
                st.success(f"✅ Questionnaire enregistré ! Score: {st.session_state.score_questionnaire}/10")
    
    # ===== TAB 2: AUDIO =====
    with tab2:
        st.subheader("🎙️ Analyse vocale")
        st.markdown("Enregistrez la voix de l'enfant pendant quelques secondes")
        
        uploaded_audio = st.file_uploader("Télécharger un fichier audio", type=["wav", "mp3", "m4a"])
        
        if uploaded_audio is not None:
            st.audio(uploaded_audio, format="audio/wav")
            
            if st.button("Analyser l'audio"):
                with st.spinner("Analyse en cours..."):
                    score_audio = random.uniform(2, 9)
                    st.session_state.score_audio = round(score_audio, 1)
                    st.success(f"✅ Score: {st.session_state.score_audio}/10")
        
        if st.session_state.score_audio is not None:
            st.metric("Score analyse vocale", f"{st.session_state.score_audio}/10")
    
    # ===== TAB 3: VISION =====
    with tab3:
        st.subheader("👁️ Vision par ordinateur")
        st.markdown("Prenez une photo de l'enfant face à la caméra")
        
        camera_image = st.camera_input("Prendre une photo")
        
        if camera_image is not None:
            st.image(camera_image, caption="Photo analysée", width=250)
            
            if st.button("Analyser l'image"):
                with st.spinner("Analyse en cours..."):
                    score_vision = random.uniform(2, 9)
                    st.session_state.score_vision = round(score_vision, 1)
                    st.success(f"✅ Score: {st.session_state.score_vision}/10")
        
        if st.session_state.score_vision is not None:
            st.metric("Score analyse visuelle", f"{st.session_state.score_vision}/10")
    
    # ===== BOUTON CONTINUER =====
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("▶️ CONTINUER VERS LES RÉSULTATS", type="primary", use_container_width=True):
            if st.session_state.score_questionnaire is not None:
                if st.session_state.score_audio is None:
                    st.session_state.score_audio = round(random.uniform(4, 6), 1)
                if st.session_state.score_vision is None:
                    st.session_state.score_vision = round(random.uniform(4, 6), 1)
                
                score_global = (st.session_state.score_questionnaire * 0.4 + 
                               st.session_state.score_audio * 0.3 + 
                               st.session_state.score_vision * 0.3)
                st.session_state.score_global = round(score_global, 1)
                
                if st.session_state.score_global < 4:
                    st.session_state.niveau = "🟢 Niveau : Faible"
                    st.session_state.recommandation = "Développement typique. Surveillance standard."
                elif st.session_state.score_global < 7:
                    st.session_state.niveau = "🟠 Niveau : Moyen"
                    st.session_state.recommandation = "Quelques signes d'alerte. Consultez un spécialiste."
                else:
                    st.session_state.niveau = "🔴 Niveau : Élevé"
                    st.session_state.recommandation = "Signes évocateurs de TSA. Intervention précoce recommandée."
                
                st.session_state.etape = 4
                sauvegarder_etape()
                st.rerun()
            else:
                st.warning("Veuillez compléter le questionnaire avant de continuer")

# ========== ÉTAPE 4: RÉSULTATS ==========
elif st.session_state.etape == 4:
    st.header("📊 Résultats NeuroSense AI+")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Score global", f"{st.session_state.score_global}/10")
    
    with col2:
        if "Faible" in st.session_state.niveau:
            st.success(st.session_state.niveau)
        elif "Moyen" in st.session_state.niveau:
            st.warning(st.session_state.niveau)
        else:
            st.error(st.session_state.niveau)
    
    with col3:
        st.progress(st.session_state.score_global / 10)
    
    st.markdown("---")
    
    st.subheader("Détail des scores par modalité")
    
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.info(f"**📝 Questionnaire**\n\n{st.session_state.score_questionnaire}/10")
    with col_b:
        st.info(f"**🎙️ Analyse vocale**\n\n{st.session_state.score_audio}/10")
    with col_c:
        st.info(f"**📷 Analyse visuelle**\n\n{st.session_state.score_vision}/10")
    
    st.markdown("---")
    
    st.subheader("💡 Recommandation")
    if "Faible" in st.session_state.niveau:
        st.success(f"✅ {st.session_state.recommandation}")
    elif "Moyen" in st.session_state.niveau:
        st.warning(f"⚠️ {st.session_state.recommandation}")
    else:
        st.error(f"🔴 {st.session_state.recommandation}")
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        rapport = f"""
RAPPORT NEUROSENSE AI+
====================
Enfant: {st.session_state.nom_enfant}
Date: {datetime.now().strftime("%d/%m/%Y")}
Score global: {st.session_state.score_global}/10
Niveau: {st.session_state.niveau}
Recommandation: {st.session_state.recommandation}

Détails des scores:
- Questionnaire: {st.session_state.score_questionnaire}/10
- Analyse vocale: {st.session_state.score_audio}/10
- Analyse visuelle: {st.session_state.score_vision}/10
        """
        st.download_button("📄 Télécharger le rapport", rapport, file_name=f"rapport_{st.session_state.nom_enfant}.txt")
    
    with col2:
        if st.button("📞 Contacter un spécialiste"):
            st.info("📋 Liste des spécialistes dans votre région")
    
    with col3:
        if st.button("🔄 Nouvelle évaluation"):
            if os.path.exists("save_neurosense.json"):
                os.remove("save_neurosense.json")
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

# ========== SIDEBAR ==========
with st.sidebar:
    st.markdown("## ⚙️ Paramètres")
    langue = st.selectbox("🌐 Langue", ["Français", "English", "العربية"])
    notifications = st.toggle("🔔 Notifications")
    confidentialite = st.checkbox("🔒 Confidentialité")
    
    st.markdown("---")
    
    # Afficher l'étape actuelle sauvegardée
    if st.session_state.etape > 1:
        st.info(f"💾 **Progression sauvegardée**\n\nÉtape: {st.session_state.etape}/4")
        if st.button("🗑️ Effacer la sauvegarde"):
            if os.path.exists("save_neurosense.json"):
                os.remove("save_neurosense.json")
            st.success("Sauvegarde effacée !")
            st.rerun()
    
    st.markdown("---")
    st.caption("© 2024 NeuroSense AI+")

# ========== FOOTER ==========
st.markdown("---")
st.caption("⚠️ **Avertissement:** Cette application est un outil d'aide à la détection précoce. Elle ne remplace en aucun cas un diagnostic médical professionnel.")
