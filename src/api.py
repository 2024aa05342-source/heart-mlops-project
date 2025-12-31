from fastapi import FastAPI
import pickle
import numpy as np
import uvicorn
from pydantic import BaseModel     # <-- add

app = FastAPI()

# Load model (pick one model for inference)
with open("models/logistic_regression.pkl", "rb") as f:
    model = pickle.load(f)

@app.get("/")
def home():
    return {"message": "Heart Disease Prediction API is running"}

# ------------- CHANGE DONE BELOW -----------------

class InputData(BaseModel):
    features: list   # Accepts JSON {"features":[...]}

@app.post("/predict")
def predict(data: InputData):
    X = np.array(data.features).reshape(1, -1)
    prediction = model.predict(X)[0]
    prob = model.predict_proba(X)[0].tolist()

    return {"prediction": int(prediction), "probability": prob}

# --------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
