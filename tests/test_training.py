from src.preprocess import DEFAULT_SPEC, build_model_pipeline, prepare_xy
from src.data_loader import load_data
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def test_model_pipeline_training_runs():
    df = load_data()
    X, y = prepare_xy(df, DEFAULT_SPEC)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe = build_model_pipeline(LogisticRegression(max_iter=500), DEFAULT_SPEC)
    pipe.fit(X_train, y_train)

    preds = pipe.predict(X_test)

    assert len(preds) == len(y_test)
