import json
import logging
import os
from pathlib import Path
from typing import Any, Optional

import joblib
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel

from src.preprocess import DEFAULT_SPEC

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("heart_api")

app = FastAPI(title="Heart Disease Prediction API")

Instrumentator().instrument(app).expose(app)

MODEL_DIR = Path(os.getenv("MODEL_DIR", "models"))
PIPELINE_PATH = MODEL_DIR / "model_pipeline.joblib"
META_PATH = MODEL_DIR / "model_meta.json"

_pipeline = None
_meta: Optional[dict[str, Any]] = None


class PredictRequest(BaseModel):
    # Accept arbitrary key/value; we will align to DEFAULT_SPEC.all_features
    payload: dict[str, Any]


def _load_assets() -> None:
    global _pipeline, _meta

    if not PIPELINE_PATH.exists():
        raise RuntimeError(
            f"Missing model pipeline at {PIPELINE_PATH}. "
            f"Run training first (trainer/init step): python -m src.train"
        )

    _pipeline = joblib.load(PIPELINE_PATH)

    if META_PATH.exists():
        _meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    else:
        _meta = {"input_features": DEFAULT_SPEC.all_features}


@app.on_event("startup")
def startup_event():
    # Load once when app starts (cleaner than import-time)
    _load_assets()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def root():
    return {
        "message": "Heart Disease Prediction API",
        "model_loaded": _pipeline is not None,
    }


@app.post("/predict")
def predict(req: PredictRequest, request: Request):
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Align inputs to stable feature order
        features = (_meta or {}).get("input_features", DEFAULT_SPEC.all_features)
        row = {k: req.payload.get(k, None) for k in features}
        df = pd.DataFrame([row], columns=features)

        pred = int(_pipeline.predict(df)[0])

        proba = None
        if hasattr(_pipeline, "predict_proba"):
            try:
                proba = float(_pipeline.predict_proba(df)[0, 1])
            except Exception:
                proba = None

        return {"prediction": pred, "probability": proba}
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")


if __name__ == "__main__":
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=False)
