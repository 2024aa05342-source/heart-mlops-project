from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    RocCurveDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

from src.preprocess import DEFAULT_SPEC, build_model_pipeline, load_dataset, prepare_xy


def _evaluate(pipe, X, y) -> Dict[str, float]:
    preds = pipe.predict(X)
    metrics = {
        "accuracy": float(accuracy_score(y, preds)),
        "f1": float(f1_score(y, preds)),
        "precision": float(precision_score(y, preds, zero_division=0)),
        "recall": float(recall_score(y, preds, zero_division=0)),
    }
    if hasattr(pipe, "predict_proba"):
        try:
            proba = pipe.predict_proba(X)[:, 1]
            metrics["roc_auc"] = float(roc_auc_score(y, proba))
        except Exception:
            metrics["roc_auc"] = float("nan")
    else:
        metrics["roc_auc"] = float("nan")
    return metrics


def _save_confusion_matrix_png(y_true, y_pred, out_path: Path, title: str) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig = plt.figure(figsize=(5, 4))
    plt.imshow(cm)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _save_roc_curve_png(pipe, X, y, out_path: Path, title: str) -> None:
    # Only if proba exists
    if not hasattr(pipe, "predict_proba"):
        return
    fig = plt.figure(figsize=(6, 4.5))
    RocCurveDisplay.from_estimator(pipe, X, y)
    plt.title(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _run_gridsearch(
    base_model,
    param_grid: Dict[str, Any],
    X_train,
    y_train,
    run_name: str,
    experiment_name: str,
) -> Tuple[Any, Dict[str, Any]]:
    """Task 2 + Task 3: CV + tuning on full pipeline + MLflow logging."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    pipeline = build_model_pipeline(base_model, DEFAULT_SPEC)

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

    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name):
        search.fit(X_train, y_train)

        best_pipe = search.best_estimator_
        best_params = search.best_params_
        best_cv_f1 = float(search.best_score_)

        # --- Part 3: track params + metrics ---
        mlflow.log_params(best_params)
        mlflow.log_metric("cv_best_f1", best_cv_f1)

        # --- Part 3: log CV results artifact ---
        cv_summary = {
            "best_params": best_params,
            "best_cv_f1": best_cv_f1,
            "mean_test_score": [
                float(x) for x in search.cv_results_["mean_test_score"]
            ],
            "std_test_score": [float(x) for x in search.cv_results_["std_test_score"]],
            "params": [dict(p) for p in search.cv_results_["params"]],
        }
        tmp = Path("models") / f"{run_name}_cv_results.json"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(cv_summary, indent=2), encoding="utf-8")
        mlflow.log_artifact(str(tmp), artifact_path="cv")

        # --- Part 3: log tuned model artifact to MLflow ---
        mlflow.sklearn.log_model(best_pipe, artifact_path="tuned_model")

    return best_pipe, {"best_params": best_params, "best_cv_f1": best_cv_f1}


def main():
    # Make MLflow tracking explicit & demo-friendly
    # local file-based tracking (works in docker too)
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
    experiment_name = os.getenv("MLFLOW_EXPERIMENT", "heart-disease-exp")
    mlflow.set_tracking_uri(tracking_uri)

    # Paths configurable for Docker/K8s init style
    model_dir = Path(os.getenv("MODEL_DIR", "models"))
    model_dir.mkdir(parents=True, exist_ok=True)

    # Data path (your code already uses this convention)
    data_path = os.getenv("DATA_PATH", "data/heart.csv")

    df = load_dataset(data_path)
    X, y = prepare_xy(df, DEFAULT_SPEC)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # --- Logistic Regression ---
    lr = LogisticRegression(max_iter=2000)
    lr_grid = {
        "model__C": [0.01, 0.1, 1, 10, 100],
        "model__solver": ["liblinear", "lbfgs"],
    }
    best_lr, lr_meta = _run_gridsearch(
        lr, lr_grid, X_train, y_train, "LR_CV_TUNE", experiment_name
    )

    # --- Random Forest ---
    rf = RandomForestClassifier(random_state=42)
    rf_grid = {
        "model__n_estimators": [100, 200, 400],
        "model__max_depth": [None, 3, 5, 8, 12],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
    }
    best_rf, rf_meta = _run_gridsearch(
        rf, rf_grid, X_train, y_train, "RF_CV_TUNE", experiment_name
    )

    # Choose model by CV f1 (Task 2)
    chosen_name = (
        "logistic_regression"
        if lr_meta["best_cv_f1"] >= rf_meta["best_cv_f1"]
        else "random_forest"
    )
    chosen = best_lr if chosen_name == "logistic_regression" else best_rf
    chosen_meta = lr_meta if chosen_name == "logistic_regression" else rf_meta

    # Final evaluation on held-out test
    test_metrics = _evaluate(chosen, X_test, y_test)

    # Persist single pipeline artifact (Task 4)
    pipeline_path = model_dir / "model_pipeline.joblib"
    joblib.dump(chosen, pipeline_path)

    meta = {
        "chosen_model": chosen_name,
        "chosen_model_meta": chosen_meta,
        "test_metrics": test_metrics,
        "tracking_uri": tracking_uri,
        "experiment": experiment_name,
        "data_path": data_path,
        "feature_spec": (
            DEFAULT_SPEC.model_dump()
            if hasattr(DEFAULT_SPEC, "model_dump")
            else str(DEFAULT_SPEC)
        ),
    }
    meta_path = model_dir / "model_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # --- Part 3: FINAL “selected model” run in MLflow ---
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name="FINAL_SELECTED_MODEL"):
        mlflow.log_param("chosen_model", chosen_name)
        mlflow.log_params(
            {f"chosen_{k}": v for k, v in chosen_meta.get("best_params", {}).items()}
        )
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})

        # Save plots (confusion matrix + ROC) and log them
        fig_dir = Path("reports/figures")
        cm_path = fig_dir / "final_confusion_matrix.png"
        roc_path = fig_dir / "final_roc_curve.png"

        y_pred = chosen.predict(X_test)
        _save_confusion_matrix_png(
            y_test, y_pred, cm_path, title=f"Confusion Matrix ({chosen_name})"
        )
        _save_roc_curve_png(
            chosen, X_test, y_test, roc_path, title=f"ROC Curve ({chosen_name})"
        )

        if cm_path.exists():
            mlflow.log_artifact(str(cm_path), artifact_path="plots")
        if roc_path.exists():
            mlflow.log_artifact(str(roc_path), artifact_path="plots")

        # Log final packaged model and meta
        mlflow.log_artifact(str(meta_path), artifact_path="package")
        mlflow.log_artifact(str(pipeline_path), artifact_path="package")
        mlflow.sklearn.log_model(chosen, artifact_path="final_model")

    print("\nTraining complete.")
    print(f"Saved pipeline: {pipeline_path}")
    print(f"Saved metadata: {meta_path}")
    print(f"Chosen: {chosen_name}")
    print("Test metrics:", test_metrics)
    print(f"MLflow tracking URI: {tracking_uri}")
    print("To view runs: mlflow ui --backend-store-uri ./mlruns")


if __name__ == "__main__":
    main()
