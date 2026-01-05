import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Heart Disease Prediction", layout="centered")

MODEL_PATH = os.getenv("MODEL_PATH", "models/model_pipeline.joblib")

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(
            f"Model not found at {MODEL_PATH}.\n\n"
            "Please run training first (docker compose trainer or build step)."
        )
        st.stop()
    return joblib.load(MODEL_PATH)

model = load_model()

st.title("❤️ Heart Disease Prediction")

st.markdown("Enter patient details below:")

# ---- Input fields (match training features) ----
age = st.number_input("Age", 1, 120, 50)
sex = st.selectbox("Sex", ["Male", "Female"])
cp = st.selectbox("Chest Pain Type", ["typical angina", "atypical angina", "non-anginal pain", "asymptomatic"])
trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)
chol = st.number_input("Cholesterol", 100, 600, 200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["FALSE", "TRUE"])
restecg = st.selectbox("Rest ECG", ["normal", "st-t abnormality", "lv hypertrophy"])
thalach = st.number_input("Max Heart Rate", 60, 220, 150)
exang = st.selectbox("Exercise Induced Angina", ["FALSE", "TRUE"])
oldpeak = st.number_input("Oldpeak (ST depression)", 0.0, 6.0, 1.0)
slope = st.selectbox("Slope", ["upsloping", "flat", "downsloping"])
ca = st.number_input("Number of Major Vessels", 0, 4, 0)
thal = st.selectbox("Thal", ["normal", "fixed defect", "reversible defect"])

# ---- Build input dataframe (column names must match training) ----
input_df = pd.DataFrame([{
    "age": age,
    "sex": sex,
    "cp": cp,
    "trestbps": trestbps,
    "chol": chol,
    "fbs": fbs,
    "restecg": restecg,
    "thalach": thalach,
    "exang": exang,
    "oldpeak": oldpeak,
    "slope": slope,
    "ca": ca,
    "thal": thal,
}])

if st.button("Predict"):
    pred = model.predict(input_df)[0]
    proba = None

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(input_df)[0][1]

    st.subheader("Prediction Result")
    if pred == 1:
        st.error("⚠️ Heart Disease Detected")
    else:
        st.success("✅ No Heart Disease Detected")

    if proba is not None:
        st.write(f"Risk Probability: **{proba:.2f}**")
