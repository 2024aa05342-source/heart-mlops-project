from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass(frozen=True)
class FeatureSpec:
    numeric_features: List[str]
    categorical_features: List[str]
    target_candidates: List[str]

    @property
    def all_features(self) -> List[str]:
        # Preserve a stable input schema for the API (Task 4)
        return list(self.numeric_features) + list(self.categorical_features)


# Heart dataset (common Kaggle/UCI CSV form)
DEFAULT_SPEC = FeatureSpec(
    numeric_features=["age", "trestbps", "chol", "thalach", "oldpeak"],
    categorical_features=[
        "sex",
        "cp",
        "fbs",
        "restecg",
        "exang",
        "slope",
        "ca",
        "thal",
    ],
    # Some heart datasets use 'target', some use 'num'
    target_candidates=["target", "num"],
)


def detect_target_column(df: pd.DataFrame, spec: FeatureSpec = DEFAULT_SPEC) -> str:
    for col in spec.target_candidates:
        if col in df.columns:
            return col
    raise ValueError(
        f"Could not detect target column. Expected one of {spec.target_candidates}. "
        f"Found: {list(df.columns)}"
    )


def prepare_xy(
    df: pd.DataFrame, spec: FeatureSpec = DEFAULT_SPEC
) -> Tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) with a stable feature order and binary target.

    - Drops extra columns not in spec + target
    - Ensures X columns match spec.all_features order
    - Supports target column named 'target' (0/1) or 'num' (0-4 -> binarized)
    """
    target_col = detect_target_column(df, spec)

    # needed = set(spec.all_features + [target_col])
    missing_features = [c for c in spec.all_features if c not in df.columns]
    if missing_features:
        raise ValueError(
            f"Missing expected feature columns: {missing_features}. "
            f"Found: {list(df.columns)}"
        )

    df2 = df[[*spec.all_features, target_col]].copy()

    y = df2[target_col]
    # Normalize target to binary 0/1
    try:
        y_int = y.astype(int)
        if target_col == "num":
            y = (y_int > 0).astype(int)
        else:
            # target usually already 0/1; but keep safe
            y = (y_int > 0).astype(int)
    except Exception:
        raise ValueError(
            f"Target column '{target_col}' must be numeric/binary. Got dtype={y.dtype}"
        )

    X = df2.drop(columns=[target_col])
    # Ensure stable order
    X = X[spec.all_features]
    return X, y


def load_dataset(path: str):
    """
    Load dataset from CSV.
    Kept minimal for reproducibility and MLOps clarity.
    """
    return pd.read_csv(path)


def build_preprocessor(spec: FeatureSpec = DEFAULT_SPEC) -> ColumnTransformer:
    """Preprocessing for Task 4 (reproducible inference):
    - Numeric: median impute + standard scale
    - Categorical: most_frequent impute + one-hot (ignore unknown)
    """
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, spec.numeric_features),
            ("cat", categorical_pipe, spec.categorical_features),
        ],
        remainder="drop",
    )


def build_model_pipeline(model, spec: FeatureSpec = DEFAULT_SPEC) -> Pipeline:
    """Full pipeline = preprocess + model (persist this single artifact)."""
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor(spec)),
            ("model", model),
        ]
    )
