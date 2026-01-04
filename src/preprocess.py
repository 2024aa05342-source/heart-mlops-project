from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET_COL = "num"

# UCI Heart Disease commonly uses these columns (your data_loader reads heart.csv)
# If your CSV has extra columns, they are dropped in `prepare_xy`.
NUMERIC_FEATURES: List[str] = [
    "age", "trestbps", "chol", "thalach", "oldpeak"
]

CATEGORICAL_FEATURES: List[str] = [
    "sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"
]


@dataclass(frozen=True)
class FeatureSpec:
    numeric: List[str]
    categorical: List[str]

    @property
    def all_features(self) -> List[str]:
        return self.numeric + self.categorical


DEFAULT_SPEC = FeatureSpec(numeric=NUMERIC_FEATURES, categorical=CATEGORICAL_FEATURES)


def prepare_xy(df: pd.DataFrame, spec: FeatureSpec = DEFAULT_SPEC) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare X (raw feature dataframe) and y (binary target).
    - Drops non-useful columns if present.
    - Converts target `num` into binary: 1 if num > 0 else 0.
    """
    df = df.copy()
    df = df.drop(columns=["id", "dataset"], errors="ignore")

    if TARGET_COL not in df.columns:
        raise ValueError(f"Expected target column '{TARGET_COL}' in dataframe. Found: {list(df.columns)}")

    # Ensure all expected feature columns exist
    missing = [c for c in spec.all_features if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected feature columns: {missing}. Found: {list(df.columns)}")

    y = (df[TARGET_COL] > 0).astype(int)
    X = df[spec.all_features].copy()
    return X, y


def build_preprocessor(spec: FeatureSpec = DEFAULT_SPEC) -> ColumnTransformer:
    """
    Build a reproducible preprocessing transformer:
    - Numeric: median impute + standard scale
    - Categorical: most_frequent impute + one-hot encode (ignore unknowns)
    """
    numeric_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, spec.numeric),
            ("cat", categorical_pipe, spec.categorical),
        ],
        remainder="drop",
    )
    return preprocessor


def build_model_pipeline(model, spec: FeatureSpec = DEFAULT_SPEC) -> Pipeline:
    """
    Full pipeline = preprocess + model.
    This is what you persist for inference (Task 4).
    """
    return Pipeline(steps=[
        ("preprocess", build_preprocessor(spec)),
        ("model", model),
    ])
