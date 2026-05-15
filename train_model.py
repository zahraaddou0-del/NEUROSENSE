import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# =========================
# تحميل البيانات
# =========================

df = pd.read_csv("autism_data.csv")

# =========================
# Features / Target
# =========================

X = df.drop("Result", axis=1)
y = df["Result"]

# =========================
# تقسيم البيانات
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================
# Normalisation
# =========================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =========================
# IA Model
# =========================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Training
model.fit(X_train_scaled, y_train)

# Prediction
y_pred = model.predict(X_test_scaled)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy : {accuracy * 100:.2f}%")

# =========================
# Sauvegarde
# =========================

joblib.dump(model, "autism_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Model saved successfully")
