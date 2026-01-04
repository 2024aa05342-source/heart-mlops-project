from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Tuple, Any

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score

from src.data_loader import load_data
from src.preprocess import prepare_xy, build_model_pipeline, DEFAULT_SPEC


MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Keep MLflow local & repo-relative (works in CI too)
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("heart-disease-exp")


def _evaluate(model_pipeline, X_test, y_test) -> Dict[str, float]:
    preds = model_pipeline.predict(X_test)
    proba = None
    if hasattr(model_pipeline, "predict_proba"):
        try:
            proba = model_pipeline.predict_proba(X_test)[:, 1]
        except Exception:
            proba = None

    metrics = {
        "test_accuracy": float(accuracy_score(y_test, preds)),
        "test_precision": float(precision_score(y_test, preds, zero_division=0)),
        "test_recall": float(recall_score(y_test, preds, zero_division=0)),
        "test_f1": float(f1_score(y_test, preds)),
    }
    if proba is not None:
        metrics["test_roc_auc"] = float(roc_auc_score(y_test, proba))
    return metrics


def tune_pipeline(pipeline, param_grid: Dict[str, Any], X_train, y_train, run_name: str) -> Tuple[Any, Dict[str, Any]]:
    """
    Task 2: CV + hyperparameter tuning (GridSearchCV) on the FULL pipeline.
    Logs CV mean/std for f1 and best params into MLflow.
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="f1",
        refit=True,
        cv=cv,
        n_jobs=-1,
        verbose=0,
        return_train_score=False,
    )

    with mlflow.start_run(run_name=run_name):
        search.fit(X_train, y_train)

        best = search.best_estimator_
        best_params = search.best_params_

        # CV summary for the best index
        idx = int(search.best_index_)
        mean_f1 = float(search.cv_results_["mean_test_score"][idx])
        std_f1 = float(search.cv_results_["std_test_score"][idx])

        mlflow.log_params(best_params)
        mlflow.log_metrics({
            "cv_f1_mean": mean_f1,
            "cv_f1_std": std_f1,
        })

        # Store the entire cv_results_ (lightweight JSON)
        cv_results_path = MODELS_DIR / f"{run_name}_cv_results.json"
        with open(cv_results_path, "w", encoding="utf-8") as f:
            json.dump(
                {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in search.cv_results_.items()},
                f,
                indent=2,
            )
        mlflow.log_artifact(str(cv_results_path))

        # Log tuned pipeline as MLflow model artifact
        mlflow.sklearn.log_model(best, artifact_path="model_pipeline")

    return best, {"cv_f1_mean": mean_f1, "cv_f1_std": std_f1, **best_params}


def main():
    df = load_data()
    X, y = prepare_xy(df, DEFAULT_SPEC)

    # Split once for final held-out test evaluation (CV happens on train portion)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Candidate 1: Logistic Regression
    lr_pipe = build_model_pipeline(LogisticRegression(max_iter=2000), DEFAULT_SPEC)
    lr_grid = {
        "model__C": [0.01, 0.1, 1, 10, 100],
        "model__solver": ["liblinear", "lbfgs"],
    }
    best_lr, lr_meta = tune_pipeline(lr_pipe, lr_grid, X_train, y_train, run_name="LR_CV_TUNE")

    # Candidate 2: Random Forest
    rf_pipe = build_model_pipeline(RandomForestClassifier(random_state=42), DEFAULT_SPEC)
    rf_grid = {
        "model__n_estimators": [100, 200, 400],
        "model__max_depth": [None, 3, 5, 8, 12],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
    }
    best_rf, rf_meta = tune_pipeline(rf_pipe, rf_grid, X_train, y_train, run_name="RF_CV_TUNE")

    # Pick best by CV f1 mean
    chosen_name = "logistic_regression" if lr_meta["cv_f1_mean"] >= rf_meta["cv_f1_mean"] else "random_forest"
    chosen = best_lr if chosen_name == "logistic_regression" else best_rf
    chosen_meta = lr_meta if chosen_name == "logistic_regression" else rf_meta

    # Final evaluation on held-out test
    test_metrics = _evaluate(chosen, X_test, y_test)

    # Persist single pipeline artifact (Task 4)
    pipeline_path = MODELS_DIR / "model_pipeline.joblib"
    joblib.dump(chosen, pipeline_path)

    # Persist model metadata (input schema + chosen params/metrics)
    meta = {
        "chosen_model": chosen_name,
        "input_features": DEFAULT_SPEC.all_features,
        "cv": {
            "f1_mean": float(chosen_meta["cv_f1_mean"]),
            "f1_std": float(chosen_meta["cv_f1_std"]),
        },
        "best_params": {k: v for k, v in chosen_meta.items() if k not in ("cv_f1_mean", "cv_f1_std")},
        "test_metrics": test_metrics,
    }
    meta_path = MODELS_DIR / "model_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("\nTraining complete.")
    print(f"Saved pipeline: {pipeline_path}")
    print(f"Saved metadata: {meta_path}")
    print(f"Chosen: {chosen_name}")
    print("Test metrics:", test_metrics)


if __name__ == "__main__":
    main()
