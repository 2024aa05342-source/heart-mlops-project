import logging
import time
import json
import pickle
from typing import List, Any

import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from prometheus_fastapi_instrumentator import Instrumentator


# -------------------- Logging Setup --------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("heart-api")

# -------------------- App --------------------
app = FastAPI(title="Heart Disease Prediction API")

# -------------------- Prometheus Metrics --------------------
# Exposes /metrics automatically
Instrumentator().instrument(app).expose(app, endpoint="/metrics")


# -------------------- Load Artifacts --------------------
# Model
with open("models/logistic_regression.pkl", "rb") as f:
    model = pickle.load(f)

# Scaler
with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Feature column order used during training (should be length 18 for your LR)
with open("models/columns.json", "r") as f:
    FEATURE_COLUMNS = json.load(f)

# Raw row schema (based on your curl sample)
# Example row you sent:
# [1,63,"Male","Cleveland","typical angina",145,233,"TRUE","lv hypertrophy",150,"FALSE",2.3,"downsloping",0,"fixed defect",0]
RAW_COLUMNS = [
    "id", "age", "sex", "dataset", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalch", "exang", "oldpeak", "slope", "ca", "thal", "num"
]


# -------------------- Request Logging Middleware --------------------
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration_ms = (time.time() - start) * 1000
    logger.info(
        "%s %s -> %s (%.2f ms)",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms
    )
    return response


@app.get("/")
def home():
    return {"message": "Heart Disease Prediction API is running"}


class InputData(BaseModel):
    features: List[Any]  # can include strings like "Male", "TRUE", etc.


def _to_bool(v):
    """Convert common boolean encodings used in the dataset."""
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)) and v in (0, 1):
        return bool(v)
    if isinstance(v, str):
        vv = v.strip().lower()
        if vv in ("true", "t", "yes", "y", "1"):
            return True
        if vv in ("false", "f", "no", "n", "0"):
            return False
    return v


def preprocess_row(features: List[Any]) -> np.ndarray:
    """
    Convert incoming row-like features into model-ready numpy array
    using the same preprocessing logic as training:
    - build DF using RAW_COLUMNS
    - drop id, dataset, num
    - one-hot encode categoricals
    - align to FEATURE_COLUMNS
    - scale using scaler
    """
    if len(features) != len(RAW_COLUMNS):
        raise ValueError(f"Expected {len(RAW_COLUMNS)} values (heart row length), got {len(features)}")

    row = dict(zip(RAW_COLUMNS, features))

    # Normalize boolean-like fields (fbs, exang often appear as TRUE/FALSE)
    row["fbs"] = _to_bool(row.get("fbs"))
    row["exang"] = _to_bool(row.get("exang"))

    df = pd.DataFrame([row])

    # Drop columns not used for prediction
    df = df.drop(columns=["id", "dataset", "num"], errors="ignore")

    # One-hot encode (categorical -> dummies)
    df_enc = pd.get_dummies(df, drop_first=False)

    # Align to training columns
    for col in FEATURE_COLUMNS:
        if col not in df_enc.columns:
            df_enc[col] = 0

    # Keep exact order
    df_enc = df_enc[FEATURE_COLUMNS]

    # Scale
    X_scaled = scaler.transform(df_enc.values)
    return X_scaled


@app.post("/predict")
def predict(data: InputData):
    try:
        X = preprocess_row(data.features)
        pred = model.predict(X)[0]
        proba = model.predict_proba(X)[0].tolist()
        return {"prediction": int(pred), "probability": proba}
    except Exception as e:
        # Return clean error to client instead of crashing ASGI
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
