import streamlit as st
import numpy as np
import joblib

# =========================
# Charger modèle
# =========================

model = joblib.load("autism_model.pkl")
scaler = joblib.load("scaler.pkl")

# =========================
# Page
# =========================

st.set_page_config(
    page_title="NeuroSense",
    page_icon="🧠"
)

st.title("🧠 NeuroSense")
st.subheader("Détection précoce de l'autisme")

st.write(
    "Répondez au questionnaire suivant."
)

# =========================
# Questions
# =========================

questions = [
    "Contact visuel faible",
    "Difficultés sociales",
    "Mouvements répétitifs",
    "Réponse faible au prénom",
    "Préfère jouer seul",
    "Comportement répétitif",
    "Retard de langage",
    "Évite interactions",
    "Sensibilité sonore",
    "Faible expression émotionnelle"
]

answers = []

for q in questions:

    ans = st.radio(
        q,
        [0, 1],
        horizontal=True,
        format_func=lambda x:
        "Oui" if x == 1 else "Non"
    )

    answers.append(ans)

# Age
age = st.slider(
    "Âge",
    2,
    12,
    5
)

# =========================
# Prediction
# =========================

if st.button("Analyser"):

    input_data = np.array(
        [answers + [age]]
    )

    input_scaled = scaler.transform(
        input_data
    )

    prediction = model.predict(
        input_scaled
    )[0]

    probability = model.predict_proba(
        input_scaled
    )[0][1]

    st.write("---")

    if prediction == 1:

        st.error(
            "⚠️ Risque potentiel détecté"
        )

    else:

        st.success(
            "✅ Faible risque"
        )

    st.metric(
        "Probabilité",
        f"{probability * 100:.2f}%"
    )
