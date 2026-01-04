import json
import logging
import time
from pathlib import Path
from typing import Any, List, Optional

import joblib
import pandas as pd
import uvicorn
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from prometheus_fastapi_instrumentator import Instrumentator


# -------------------- Logging Setup --------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("heart-api")

MODELS_DIR = Path("models")
PIPELINE_PATH = MODELS_DIR / "model_pipeline.joblib"
META_PATH = MODELS_DIR / "model_meta.json"

app = FastAPI(title="Heart Disease Prediction API")


# -------------------- Middleware --------------------
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration_ms = (time.time() - start) * 1000
    logger.info("%s %s -> %s (%.2f ms)", request.method, request.url.path, response.status_code, duration_ms)
    return response


# -------------------- Load model pipeline & metadata --------------------
pipeline = None
INPUT_FEATURES: Optional[List[str]] = None

def _load_assets():
    global pipeline, INPUT_FEATURES
    if not PIPELINE_PATH.exists():
        raise RuntimeError(f"Missing model pipeline at {PIPELINE_PATH}. Train first: python -m src.train")

    pipeline = joblib.load(PIPELINE_PATH)

    if META_PATH.exists():
        meta = json.loads(META_PATH.read_text(encoding="utf-8"))
        INPUT_FEATURES = meta.get("input_features")
    else:
        INPUT_FEATURES = None

_load_assets()

Instrumentator().instrument(app).expose(app)


@app.get("/")
def home():
    return {"message": "Heart Disease Prediction API is running"}


class InputData(BaseModel):
    # Keep your existing contract: ordered list of feature values
    features: List[Any]


def _validate_features(values: List[Any]) -> List[Any]:
    if INPUT_FEATURES is None:
        return values
    if len(values) != len(INPUT_FEATURES):
        raise HTTPException(
            status_code=400,
            detail=f"Expected {len(INPUT_FEATURES)} features in order {INPUT_FEATURES}, got {len(values)}",
        )
    return values


@app.post("/predict")
def predict(data: InputData):
    if pipeline is None:
        raise HTTPException(status_code=500, detail="Model pipeline not loaded")

    values = _validate_features(data.features)

    # Convert list -> single-row DataFrame with correct column names
    if INPUT_FEATURES is None:
        # Fallback: accept list without schema (best effort)
        raise HTTPException(status_code=500, detail="Model metadata missing: input feature schema unknown")
    df = pd.DataFrame([values], columns=INPUT_FEATURES)

    try:
        pred = int(pipeline.predict(df)[0])
        proba = None
        if hasattr(pipeline, "predict_proba"):
            try:
                proba = float(pipeline.predict_proba(df)[0, 1])
            except Exception:
                proba = None
        return {"prediction": pred, "probability": proba}
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")


if __name__ == "__main__":
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=False)
