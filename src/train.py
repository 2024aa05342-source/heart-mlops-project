import mlflow
import pickle
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

from src.preprocess import preprocess
from src.data_loader import load_data

# ------------ FIX FOR CI ------------
mlflow.set_tracking_uri("file:./mlruns")  # ensures it works in GitHub Actions
mlflow.set_experiment("heart-disease-exp")
os.makedirs("models", exist_ok=True)      # ensures folder exists
# -------------------------------------


def train_and_log(model, X_train, X_test, y_train, y_test, model_name):

    with mlflow.start_run(run_name=model_name):   # safer context manager

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        acc = accuracy_score(y_test, y_pred)
        pre = precision_score(y_test, y_pred, average='macro', zero_division=0)
        rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
        auc = roc_auc_score(y_test, y_proba, multi_class='ovr')

        print(f"\nModel: {model_name}")
        print(f"Accuracy: {acc}")
        print(f"Precision: {pre}")
        print(f"Recall: {rec}")
        print(f"AUC: {auc}")

        mlflow.log_params(model.get_params())
        mlflow.log_metrics({"accuracy": acc, "precision": pre, "recall": rec, "auc": auc})

        # ---- Save model locally ----
        file_path = f"models/{model_name}.pkl"
        with open(file_path, "wb") as f:
            pickle.dump(model, f)

        # ---- Upload artifact (CI compatible) ----
        mlflow.log_artifact(file_path)


if __name__ == "__main__":
    df = load_data()
    (X_train, X_test, y_train, y_test), scaler = preprocess(df)

    train_and_log(LogisticRegression(max_iter=500), X_train, X_test, y_train, y_test, "logistic_regression")
    train_and_log(RandomForestClassifier(), X_train, X_test, y_train, y_test, "random_forest")

    print("\nTraining Complete. Models saved in /models and mlflow logs stored in mlruns/")
