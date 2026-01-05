from src.preprocess import DEFAULT_SPEC, build_preprocessor, prepare_xy
from src.data_loader import load_data
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def test_prepare_xy_shapes_and_target():
    df = load_data()
    X, y = prepare_xy(df, DEFAULT_SPEC)

    assert len(X) == len(y)
    assert set(y.unique()).issubset({0, 1})
    assert list(X.columns) == DEFAULT_SPEC.all_features


def test_preprocessor_removes_missing_after_fit_transform():
    df = load_data()
    X, y = prepare_xy(df, DEFAULT_SPEC)

    pre = build_preprocessor(DEFAULT_SPEC)
    Xt = pre.fit_transform(X, y)

    # Xt is numpy or sparse; ensure no NaNs
    import numpy as np

    if hasattr(Xt, "toarray"):
        Xt = Xt.toarray()
    assert np.isnan(Xt).sum() == 0
