import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import mlflow
from mlflow.tracking import MlflowClient
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

from src.preprocess import preprocess
from src.data_loader import load_data

# ----------------- FIX FOR CI & ARTIFACT ISSUE -----------------
mlflow.set_tracking_uri("file:./mlruns")     # Store experiment inside repo folder
client = MlflowClient()

os.makedirs("mlruns", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Always write experiments inside repository for CI
experiment_name = "heart-disease-exp"

# Try to get experiment, else create new one
try:
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        exp_id = client.create_experiment(name=experiment_name, artifact_location="file:./mlruns")
    else:
        exp_id = experiment.experiment_id
except:
    exp_id = client.create_experiment(name=experiment_name, artifact_location="file:./mlruns")

mlflow.set_experiment(experiment_name)

# ---------------------------------------------------------------


def train_and_log(model, X_train, X_test, y_train, y_test, model_name):
    import os
    IN_CI = os.getenv("GITHUB_ACTIONS") == "true"

    with mlflow.start_run(run_name=model_name):   # Run logs safely handled

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        acc = accuracy_score(y_test, y_pred)
        pre = precision_score(y_test, y_pred, average='macro', zero_division=0)
        rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
        # For binary classification AUC
        if y_proba.shape[1] == 2:
            auc = roc_auc_score(y_test, y_proba[:,1])
        else:
            auc = roc_auc_score(y_test, y_proba, multi_class='ovr')  # fallback if >2 classes


        print(f"\nModel: {model_name}")
        print(f"Accuracy: {acc}")
        print(f"Precision: {pre}")
        print(f"Recall: {rec}")
        print(f"AUC: {auc}")

        mlflow.log_params(model.get_params())
        mlflow.log_metrics({"accuracy": acc, "precision": pre, "recall": rec, "auc": auc})

        file_path = f"models/{model_name}.pkl"
        with open(file_path, "wb") as f:
            pickle.dump(model, f)

        # prevent writing to /Users path inside CI
        if not IN_CI:
            mlflow.log_artifact(file_path, artifact_path="models")

import json

if __name__ == "__main__":
    df = load_data()
    (X_train, X_test, y_train, y_test), scaler, feature_columns = preprocess(df)

    # Save scaler
    with open("models/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    # Save feature order
    with open("models/columns.json", "w") as f:
        json.dump(feature_columns, f)

    # Train models
    train_and_log(LogisticRegression(max_iter=500), X_train, X_test, y_train, y_test, "logistic_regression")
    train_and_log(RandomForestClassifier(), X_train, X_test, y_train, y_test, "random_forest")

    print("\nTraining Complete — models & artifacts saved.")
    print("Files saved in models/:")
    print(" → logistic_regression.pkl")
    print(" → random_forest.pkl")
    print(" → scaler.pkl")
    print(" → columns.json")

