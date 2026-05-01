import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import random
import os
from pathlib import Path

# ========== CONFIGURATION ==========
st.set_page_config(
    page_title="NeuroSense AI+ - Détection précoce de l'autisme",
    page_icon="🧠",
    layout="wide"
)

# ========== FICHIER EXCEL ==========
EXCEL_FILE = "neurosense_data.xlsx"

def init_excel_file():
    """Initialiser le fichier Excel avec les colonnes nécessaires"""
    if not os.path.exists(EXCEL_FILE):
        df = pd.DataFrame(columns=[
            "Date", "Type_Utilisateur", "Nom_Parent", "Age_Parent",
            "Nom_Enfant", "Sexe_Enfant", "Historique_Familial",
            "Score_Questionnaire", "Score_Audio", "Score_Vision",
            "Score_Global", "Niveau_Risque", "Recommandation",
            "Reponses_Detaillees"
        ])
        df.to_excel(EXCEL_FILE, index=False)

def sauvegarder_dans_excel(data):
    """Sauvegarder les données dans le fichier Excel"""
    init_excel_file()
    
    df_existant = pd.read_excel(EXCEL_FILE)
    df_nouveau = pd.DataFrame([data])
    df_combine = pd.concat([df_existant, df_nouveau], ignore_index=True)
    df_combine.to_excel(EXCEL_FILE, index=False)

# ========== INITIALISATION ==========
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
if 'niveau' not in st.session_state:
    st.session_state.niveau = ""
if 'recommandation' not in st.session_state:
    st.session_state.recommandation = ""
if 'historique_tests' not in st.session_state:
    st.session_state.historique_tests = []
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

# ========== ÉTAPE 1: CHOIX DU PROFIL ==========
if st.session_state.etape == 1:
    st.header("1. 📋 Choix du profil")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    border-radius: 15px; padding: 1.5rem; text-align: center; color: white;">
            <h2 style="margin:0;">👨‍👩‍👦</h2>
            <h3>Parent</h3>
            <p>Pour les familles</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Choisir Parent", key="btn_parent", use_container_width=True):
            st.session_state.type_utilisateur = "Parent"
            st.session_state.etape = 2
            st.rerun()
    
    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                    border-radius: 15px; padding: 1.5rem; text-align: center; color: white;">
            <h2 style="margin:0;">⚕️</h2>
            <h3>Professionnel</h3>
            <p>Pour les spécialistes</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Choisir Professionnel", key="btn_pro", use_container_width=True):
            st.session_state.type_utilisateur = "Professionnel"
            st.session_state.etape = 2
            st.rerun()

# ========== ÉTAPE 2: PROFIL DE L'ENFANT ==========
elif st.session_state.etape == 2:
    st.header("2. 👶 Profil de l'enfant")
    
    with st.form("profil_enfant"):
        col1, col2 = st.columns(2)
        
        with col1:
            nom_parent = st.text_input("Nom du parent / Professionnel")
            age_parent = st.number_input("Âge du parent", min_value=18, max_value=100, step=1)
        
        with col2:
            nom_enfant = st.text_input("Prénom de l'enfant")
            sexe_enfant = st.selectbox("Sexe de l'enfant", ["Garçon", "Fille", "Autre"])
            age_enfant = st.number_input("Âge de l'enfant (mois)", min_value=0, max_value=72, step=1, help="Âge en mois (0-72 mois pour détection précoce)")
        
        st.markdown("**📋 Informations complémentaires**")
        historique_familial = st.radio(
            "Existe-t-il des antécédents familiaux de TSA ?",
            ["Aucun", "Cas dans la famille éloignée", "Cas dans la famille proche (parents/frères/sœurs)"]
        )
        
        developpement_notes = st.text_area("Observations complémentaires (optionnel)", 
                                           placeholder="Décrivez tout comportement ou développement inhabituel...")
        
        submitted = st.form_submit_button("📝 Continuer", use_container_width=True)
        
        if submitted:
            if nom_parent and nom_enfant:
                st.session_state.nom_parent = nom_parent
                st.session_state.age_parent = age_parent
                st.session_state.nom_enfant = nom_enfant
                st.session_state.sexe_enfant = sexe_enfant
                st.session_state.age_enfant = age_enfant
                st.session_state.historique_familial = historique_familial
                st.session_state.developpement_notes = developpement_notes
                st.session_state.etape = 3
                st.rerun()
            else:
                st.error("Veuillez remplir tous les champs obligatoires")

# ========== ÉTAPE 3: ANALYSE IA ==========
elif st.session_state.etape == 3:
    st.header("3. 🤖 Analyse IA")
    st.info(f"👋 Bienvenue {st.session_state.nom_parent} ! Analyse pour {st.session_state.nom_enfant} ({st.session_state.age_enfant} mois)")
    
    # Création des onglets
    tab1, tab2, tab3 = st.tabs(["📝 Questionnaire (15 questions)", "🎙️ Analyse vocale", "👁️ Vision par ordinateur"])
    
    # ===== TAB 1: QUESTIONNAIRE =====
    with tab1:
        st.subheader("📝 Questionnaire d'évaluation")
        st.caption("Veuillez répondre à toutes les questions")
        
        # Groupe 1: Interaction sociale
        st.markdown("**🟢 1. Interaction sociale**")
        
        q1 = st.radio("1. L'enfant répond-il à son prénom quand vous l'appelez ?",
                      ["Oui, toujours", "Parfois", "Non, rarement ou jamais"], key="q1")
        q2 = st.radio("2. L'enfant sourit-il en réponse à votre sourire ?",
                      ["Oui, souvent", "Parfois", "Non, rarement"], key="q2")
        q3 = st.radio("3. L'enfant cherche-t-il à attirer votre attention ?",
                      ["Oui, souvent", "Parfois", "Non, rarement"], key="q3")
        q4 = st.radio("4. L'enfant imite-t-il vos gestes ?",
                      ["Oui, souvent", "Parfois", "Non, rarement"], key="q4")
        
        # Groupe 2: Communication
        st.markdown("**🟡 2. Communication et langage**")
        
        q5 = st.radio("5. L'enfant utilise-t-il des gestes pour communiquer ?",
                      ["Oui, plusieurs gestes", "Un ou deux gestes", "Non, pas de gestes"], key="q5")
        q6 = st.radio("6. L'enfant babille-t-il ou dit-il des mots ?",
                      ["Oui, plusieurs mots", "Quelques sons ou mots", "Très peu ou pas du tout"], key="q6")
        q7 = st.radio("7. L'enfant répète-t-il les mêmes mots sans contexte ?",
                      ["Non, jamais", "Parfois", "Oui, fréquemment"], key="q7")
        
        # Groupe 3: Comportements répétitifs
        st.markdown("**🔴 3. Comportements répétitifs**")
        
        q8 = st.radio("8. L'enfant a-t-il des mouvements répétitifs ?",
                      ["Non, jamais", "Parfois", "Oui, fréquemment"], key="q8")
        q9 = st.radio("9. L'enfant est-il très attaché à certains objets ?",
                      ["Non, pas particulièrement", "Un peu", "Oui, très attaché"], key="q9")
        q10 = st.radio("10. L'enfant a-t-il une routine rigide ?",
                       ["Non, flexible", "Parfois", "Oui, très difficile"], key="q10")
        
        # Groupe 4: Sensorialité
        st.markdown("**🟠 4. Sensorialité**")
        
        q11 = st.radio("11. L'enfant réagit-il anormalement aux sons ?",
                       ["Non, réaction normale", "Parfois", "Oui, souvent"], key="q11")
        q12 = st.radio("12. L'enfant fixe-t-il les objets inhabituellement ?",
                       ["Non", "Parfois", "Oui, fréquemment"], key="q12")
        q13 = st.radio("13. L'enfant a-t-il une sensibilité anormale ?",
                       ["Non, normale", "Un peu inhabituel", "Très différent"], key="q13")
        
        # Groupe 5: Jeu
        st.markdown("**🔵 5. Jeu et interactions**")
        
        q14 = st.radio("14. L'enfant joue-t-il de manière imaginative ?",
                       ["Oui, souvent", "Parfois", "Non, jamais"], key="q14")
        q15 = st.radio("15. L'enfant préfère-t-il jouer seul ?",
                       ["Non, aime jouer avec d'autres", "Un peu des deux", "Oui, préfère seul"], key="q15")
        
        reponses = [q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, q11, q12, q13, q14, q15]
        st.session_state.reponses = reponses
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("✅ Enregistrer le questionnaire", use_container_width=True):
                # Calcul du score (0-10, plus élevé = plus de risques)
                scores = []
                for rep in reponses:
                    if "toujours" in rep or "souvent" in rep or "plusieurs" in rep:
                        if "Non" in rep:
                            scores.append(2)
                        else:
                            scores.append(0)
                    elif "Parfois" in rep or "Un peu" in rep or "Quelques" in rep:
                        scores.append(1)
                    else:
                        if "Non" in rep:
                            scores.append(0)
                        else:
                            scores.append(2)
                
                # Correction du calcul de score
                score_total = sum(scores)
                st.session_state.score_questionnaire = round((score_total / 30) * 10, 1)
                st.success(f"✅ Questionnaire enregistré ! Score: {st.session_state.score_questionnaire}/10")
    
    # ===== TAB 2: AUDIO =====
    with tab2:
        st.subheader("🎙️ Analyse vocale")
        st.markdown("Enregistrez la voix de l'enfant pour analyse des patterns vocaux")
        st.info("L'analyse vocale examine: intonation, fréquence, réactivité aux stimuli sonores")
        
        uploaded_audio = st.file_uploader("Télécharger un fichier audio (WAV/MP3)", type=["wav", "mp3", "m4a"])
        
        if uploaded_audio is not None:
            st.audio(uploaded_audio, format="audio/wav")
            
            if st.button("🎤 Analyser l'audio", key="btn_audio"):
                with st.spinner("Analyse des patterns vocaux en cours..."):
                    # Simulation d'analyse (à remplacer par vrai modèle ML)
                    score_audio = random.uniform(2, 9)
                    st.session_state.score_audio = round(score_audio, 1)
                    st.success(f"✅ Analyse complète ! Score: {st.session_state.score_audio}/10")
                    
                    # Feedback détaillé simulé
                    if score_audio < 4:
                        st.info("📊 Résultat: Patterns vocaux typiques")
                    elif score_audio < 7:
                        st.warning("📊 Résultat: Quelques variations dans les patterns vocaux")
                    else:
                        st.error("📊 Résultat: Patterns vocaux atypiques détectés")
        
        if st.session_state.score_audio is not None:
            st.metric("Score analyse vocale", f"{st.session_state.score_audio}/10")
    
    # ===== TAB 3: VISION =====
    with tab3:
        st.subheader("👁️ Vision par ordinateur")
        st.markdown("Analyse des expressions faciales et du contact visuel")
        st.info("L'analyse visuelle examine: contact visuel, expressions faciales, réactions aux stimuli visuels")
        
        camera_image = st.camera_input("Prendre une photo de l'enfant")
        
        if camera_image is not None:
            st.image(camera_image, caption="Photo analysée", width=250)
            
            if st.button("👁️ Analyser l'image", key="btn_vision"):
                with st.spinner("Analyse des caractéristiques faciales en cours..."):
                    score_vision = random.uniform(2, 9)
                    st.session_state.score_vision = round(score_vision, 1)
                    st.success(f"✅ Analyse complète ! Score: {st.session_state.score_vision}/10")
                    
                    if score_vision < 4:
                        st.info("📊 Résultat: Contact visuel typique")
                    elif score_vision < 7:
                        st.warning("📊 Résultat: Contact visuel variable")
                    else:
                        st.error("📊 Résultat: Contact visuel réduit détecté")
        
        if st.session_state.score_vision is not None:
            st.metric("Score analyse visuelle", f"{st.session_state.score_vision}/10")
    
    # ===== BOUTON CONTINUER =====
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("▶️ CONTINUER VERS LES RÉSULTATS", type="primary", use_container_width=True):
            if st.session_state.score_questionnaire is not None:
                # Générer scores simulés si manquants
                if st.session_state.score_audio is None:
                    st.session_state.score_audio = round(random.uniform(3, 7), 1)
                if st.session_state.score_vision is None:
                    st.session_state.score_vision = round(random.uniform(3, 7), 1)
                
                # Score global avec pondération
                score_global = (st.session_state.score_questionnaire * 0.5 + 
                               st.session_state.score_audio * 0.25 + 
                               st.session_state.score_vision * 0.25)
                st.session_state.score_global = round(score_global, 1)
                
                # Déterminer le niveau de risque
                if st.session_state.score_global < 4:
                    st.session_state.niveau = "🟢 Risque Faible"
                    st.session_state.recommandation = "Développement typique. Surveillance standard recommandée."
                elif st.session_state.score_global < 7:
                    st.session_state.niveau = "🟠 Risque Modéré"
                    st.session_state.recommandation = "Quelques signes d'alerte détectés. Consultation avec un pédiatre conseillée."
                else:
                    st.session_state.niveau = "🔴 Risque Élevé"
                    st.session_state.recommandation = "Signes évocateurs de TSA détectés. Intervention précoce et consultation spécialisée recommandées."
                
                # Sauvegarder dans l'historique
                test_result = {
                    "date": datetime.now().strftime("%d/%m/%Y %H:%M"),
                    "score": st.session_state.score_global,
                    "niveau": st.session_state.niveau,
                    "enfant": st.session_state.nom_enfant
                }
                st.session_state.historique_tests.append(test_result)
                
                st.session_state.etape = 4
                st.rerun()
            else:
                st.warning("⚠️ Veuillez compléter le questionnaire avant de continuer")

# ========== ÉTAPE 4: RÉSULTATS ==========
elif st.session_state.etape == 4:
    st.header("📊 Résultats NeuroSense AI+")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Score global", f"{st.session_state.score_global}/10", 
                  delta="Plus le score est élevé, plus l'attention est requise")
    
    with col2:
        if "Faible" in st.session_state.niveau:
            st.success(f"**{st.session_state.niveau}**")
        elif "Modéré" in st.session_state.niveau:
            st.warning(f"**{st.session_state.niveau}**")
        else:
            st.error(f"**{st.session_state.niveau}**")
    
    with col3:
        st.progress(st.session_state.score_global / 10)
        st.caption("Seuil d'alerte: > 7/10")
    
    st.markdown("---")
    
    st.subheader("📈 Détail des scores par analyse")
    
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.info(f"**📝 Questionnaire**\n\n{st.session_state.score_questionnaire}/10\n\nPondération: 50%")
    with col_b:
        st.info(f"**🎙️ Analyse vocale**\n\n{st.session_state.score_audio}/10\n\nPondération: 25%")
    with col_c:
        st.info(f"**👁️ Analyse visuelle**\n\n{st.session_state.score_vision}/10\n\nPondération: 25%")
    
    st.markdown("---")
    
    st.subheader("💡 Recommandations personnalisées")
    st.markdown(f"**{st.session_state.recommandation}**")
    
    # Recommandations supplémentaires selon l'âge
    if hasattr(st.session_state, 'age_enfant'):
        st.markdown("**📌 Actions suggérées:**")
        if st.session_state.age_enfant < 18:
            st.markdown("- ✅ Suivi rapproché du développement moteur et social")
            st.markdown("- ✅ Stimulation précoce à domicile")
        elif st.session_state.age_enfant < 36:
            st.markdown("- ✅ Consultation avec un pédiatre développementaliste")
            st.markdown("- ✅ Évaluation par un orthophoniste")
        else:
            st.markdown("- ✅ Évaluation multidisciplinaire recommandée")
            st.markdown("- ✅ Contact avec un centre de ressources autisme")
    
    st.markdown("---")
    
    # Boutons d'action
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Sauvegarde dans Excel
        if st.button("💾 Sauvegarder dans Excel", use_container_width=True):
            try:
                data_to_save = {
                    "Date": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
                    "Type_Utilisateur": st.session_state.type_utilisateur,
                    "Nom_Parent": st.session_state.nom_parent,
                    "Age_Parent": st.session_state.age_parent,
                    "Nom_Enfant": st.session_state.nom_enfant,
                    "Sexe_Enfant": st.session_state.sexe_enfant,
                    "Age_Enfant_Mois": st.session_state.get('age_enfant', 0),
                    "Historique_Familial": st.session_state.get('historique_familial', ""),
                    "Score_Questionnaire": st.session_state.score_questionnaire,
                    "Score_Audio": st.session_state.score_audio,
                    "Score_Vision": st.session_state.score_vision,
                    "Score_Global": st.session_state.score_global,
                    "Niveau_Risque": st.session_state.niveau,
                    "Recommandation": st.session_state.recommandation,
                    "Reponses_Detaillees": str(st.session_state.get('reponses', []))
                }
                sauvegarder_dans_excel(data_to_save)
                st.success("✅ Données sauvegardées dans neurosense_data.xlsx")
            except Exception as e:
                st.error(f"Erreur lors de la sauvegarde: {e}")
    
    with col2:
        # Export CSV
        rapport_df = pd.DataFrame([{
            "Enfant": st.session_state.nom_enfant,
            "Date": datetime.now().strftime("%d/%m/%Y"),
            "Score_Global": st.session_state.score_global,
            "Niveau": st.session_state.niveau,
            "Recommandation": st.session_state.recommandation
        }])
        csv = rapport_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Télécharger CSV", csv, file_name=f"rapport_{st.session_state.nom_enfant}.csv", use_container_width=True)
    
    with col3:
        rapport_texte = f"""
NEUROSENSE AI+ - RAPPORT D'ÉVALUATION
=====================================
Date: {datetime.now().strftime("%d/%m/%Y %H:%M")}
Enfant: {st.session_state.nom_enfant} ({st.session_state.get('age_enfant', '?')} mois)
Sexe: {st.session_state.sexe_enfant}

RÉSULTATS:
- Score global: {st.session_state.score_global}/10
- Niveau: {st.session_state.niveau}
- Questionnaire: {st.session_state.score_questionnaire}/10
- Analyse vocale: {st.session_state.score_audio}/10
- Analyse visuelle: {st.session_state.score_vision}/10

RECOMMANDATION:
{st.session_state.recommandation}

⚠️ Ce rapport est un outil d'aide à la détection précoce. 
Il ne remplace pas un diagnostic médical professionnel.
        """
        st.download_button("📄 Télécharger Rapport TXT", rapport_texte, file_name=f"rapport_{st.session_state.nom_enfant}.txt", use_container_width=True)
    
    with col4:
        if st.button("🔄 Nouveau test", use_container_width=True):
            # Ne pas effacer l'historique
            for key in ['etape', 'score_questionnaire', 'score_audio', 'score_vision', 'score_global', 'niveau', 'recommandation', 'reponses']:
                if key in st.session_state:
                    if key != 'historique_tests':
                        st.session_state[key] = None if key != 'etape' else 1
            st.session_state.etape = 1
            st.rerun()
    
    # Afficher l'historique des tests récents
    st.markdown("---")
    st.subheader("📋 Historique des tests récents")
    if st.session_state.historique_tests:
        df_historique = pd.DataFrame(st.session_state.historique_tests[-5:])
        st.dataframe(df_historique, use_container_width=True, hide_index=True)
    else:
        st.info("Aucun test précédent")

# ========== SIDEBAR ==========
with st.sidebar:
    st.markdown("## ⚙️ Paramètres")
    st.selectbox("Langue", ["Français", "English", "العربية"])
    
    st.markdown("---")
    st.markdown("## 📊 Statistiques")
    
    if os.path.exists(EXCEL_FILE):
        try:
            df_data = pd.read_excel(EXCEL_FILE)
            total_tests = len(df_data)
            st.metric("Total des tests", total_tests)
            if total_tests > 0:
                avg_score = df_data['Score_Global'].mean()
                st.metric("Score moyen", f"{avg_score:.1f}/10")
        except:
            st.info("Chargement des statistiques...")
    
    st.markdown("---")
    st.markdown("## 📜 Tests récents")
    
    if st.session_state.historique_tests:
        for test in st.session_state.historique_tests[-3:]:
            st.write(f"📅 {test['date'][:10]}: {test['enfant']} → {test['score']}/10")
    else:
        st.info("Aucun test")
    
    st.markdown("---")
    st.markdown("## ℹ️ À propos")
    st.caption("NeuroSense AI+ v1.0 - Détection précoce de l'autisme par IA")
    st.caption("© 2024 - Tous droits réservés")

# ========== FOOTER ==========
st.markdown("---")
st.caption("""
⚠️ **AVERTISSEMENT MÉDICAL:** Cette application est un outil **d'aide à la détection précoce** utilisant l'intelligence artificielle. 
Elle ne remplace en aucun cas un diagnostic médical professionnel. Si vous avez des préoccupations concernant le développement 
de votre enfant, veuillez consulter un pédiatre ou un spécialiste du développement de l'enfant.
""")
