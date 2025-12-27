import mlflow
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

from src.preprocess import preprocess
from src.data_loader import load_data

mlflow.set_tracking_uri("mlruns")  # stores experiment logs locally
mlflow.set_experiment("heart-disease-exp")

def train_and_log(model, X_train, X_test, y_train, y_test, model_name):
    mlflow.start_run(run_name=model_name)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)   # <-- required for AUC

    acc = accuracy_score(y_test, y_pred)
    pre = precision_score(y_test, y_pred, average='macro')
    rec = recall_score(y_test, y_pred, average='macro')
    auc = roc_auc_score(y_test, y_proba, multi_class='ovr')

    print(f"\nModel: {model_name}")
    print(f"Accuracy: {acc}")
    print(f"Precision: {pre}")
    print(f"Recall: {rec}")
    print(f"AUC: {auc}")

    mlflow.log_params(model.get_params())
    mlflow.log_metrics({"accuracy": acc, "precision": pre, "recall": rec, "auc": auc})

    # Save model file
    with open(f"models/{model_name}.pkl", "wb") as f:
        pickle.dump(model, f)
    mlflow.log_artifact(f"models/{model_name}.pkl")

    mlflow.end_run()


if __name__ == "__main__":
    df = load_data()
    (X_train, X_test, y_train, y_test), scaler = preprocess(df)

    # Train two models
    train_and_log(LogisticRegression(max_iter=500), X_train, X_test, y_train, y_test, "logistic_regression")
    train_and_log(RandomForestClassifier(), X_train, X_test, y_train, y_test, "random_forest")

    print("\nTraining Complete. Models saved in /models")
