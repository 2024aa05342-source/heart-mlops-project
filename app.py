import streamlit as st
import numpy as np
import pickle
import json

# Load trained model, scaler, and feature order
with open("models/logistic_regression.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("models/columns.json", "r") as f:
    columns = json.load(f)


st.title("💓 Heart Disease Prediction App")
st.write("Enter patient details below and click *Predict*")

# Input Fields (18 features)
# You can later add dropdowns, sliders etc. for better UI
age = st.number_input("Age", 20, 100, 50)
trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)
chol = st.number_input("Cholesterol", 100, 600, 250)
thalch = st.number_input("Max Heart Rate", 60, 250, 150)
oldpeak = st.number_input("Oldpeak", 0.0, 6.0, 1.0, step=0.1)
ca = st.selectbox("Number of Major Vessels (0–3)", [0,1,2,3])

# One-hot encoded categorical fields
sex_Male = st.selectbox("Sex", ["Female", "Male"])
sex_Male = 1 if sex_Male=="Male" else 0

cp = st.selectbox("Chest Pain Type", ["atypical angina","non-anginal","typical angina"])
cp_atypical = 1 if cp=="atypical angina" else 0
cp_non_anginal = 1 if cp=="non-anginal" else 0
cp_typical = 1 if cp=="typical angina" else 0

fbs_True = st.selectbox("Fasting Blood Sugar >120 mg/dl", ["No","Yes"])
fbs_True = 1 if fbs_True=="Yes" else 0

restecg = st.selectbox("Resting ECG Result", ["normal","st-t abnormality"])
restecg_normal = 1 if restecg=="normal" else 0
restecg_st = 1 if restecg=="st-t abnormality" else 0

exang_True = st.selectbox("Exercise Induced Angina", ["No","Yes"])
exang_True = 1 if exang_True=="Yes" else 0

slope = st.selectbox("Slope of ST segment", ["flat","upsloping"])
slope_flat = 1 if slope=="flat" else 0
slope_up = 1 if slope=="upsloping" else 0

thal = st.selectbox("Thalassemia", ["normal","reversible defect"])
thal_normal = 1 if thal=="normal" else 0
thal_reversible = 1 if thal=="reversible defect" else 0

# Create input list in same order as training
features = np.array([
    age, trestbps, chol, thalch, oldpeak, ca,
    sex_Male, cp_atypical, cp_non_anginal, cp_typical,
    fbs_True, restecg_normal, restecg_st,
    exang_True, slope_flat, slope_up,
    thal_normal, thal_reversible
]).reshape(1,-1)

import requests
import json

API_URL = "http://localhost:8000/predict"  # FastAPI endpoint

import pandas as pd
import plotly.express as px
if st.button("Predict"):
    # Scale input like training
    scaled = scaler.transform(features)

    prediction = model.predict(scaled)[0]
    prob = model.predict_proba(scaled)[0]

    st.subheader("🔍 Result:")
    if prediction == 0:
        st.success("🟢 No Significant Risk of Heart Disease")
    else:
        st.error("🔴 Potential Heart Disease Risk")

    st.write("### 📊 Probability Distribution")
    st.bar_chart(prob)
