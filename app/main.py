from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle

# -----------------------------
# Load Model
# -----------------------------
with open("models/random_forest.pkl", "rb") as f:
    model = pickle.load(f)

# -----------------------------
# FastAPI App
# -----------------------------
app = FastAPI(title="Heart Disease Prediction API")

# -----------------------------
# Input Schema (must match feature order)
# -----------------------------
class HeartData(BaseModel):
    age: float
    trestbps: float
    chol: float
    thalch: float
    oldpeak: float
    ca: float
    sex_Male: int
    cp_atypical_angina: int
    cp_non_anginal: int
    cp_typical_angina: int
    fbs_True: int
    restecg_normal: int
    restecg_st_t_abnormality: int
    exang_True: int
    slope_flat: int
    slope_upsloping: int
    thal_normal: int
    thal_reversable_defect: int


# -----------------------------
# Routes
# -----------------------------
@app.get("/")
def home():
    return {"message": "Heart Disease Prediction API is Running 🚀"}

@app.post("/predict")
def predict(data: HeartData):

    features = np.array([
        data.age, data.trestbps, data.chol, data.thalch, data.oldpeak, data.ca,
        data.sex_Male, data.cp_atypical_angina, data.cp_non_anginal, data.cp_typical_angina,
        data.fbs_True, data.restecg_normal, data.restecg_st_t_abnormality,
        data.exang_True, data.slope_flat, data.slope_upsloping,
        data.thal_normal, data.thal_reversable_defect
    ]).reshape(1,-1)

    pred = model.predict(features)[0]
    prob = model.predict_proba(features)[0].tolist()

    return {
        "prediction": int(pred),
        "probability": prob
    }

