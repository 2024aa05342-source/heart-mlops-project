import pandas as pd
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader import load_data
from src.preprocess import preprocess

def test_preprocess_output_shapes():
    df = load_data()
    (X_train, X_test, y_train, y_test), scaler, cols = preprocess(df)

    # Ensure shapes look valid
    assert len(X_train) > 0
    assert len(X_test) > 0
    assert len(y_train) > 0
    assert len(y_test) > 0
    assert len(cols) == X_train.shape[1]  # feature-count consistency

def test_no_missing_values_after_preprocess():
    df = load_data()
    (X_train, X_test, y_train, y_test), scaler, cols = preprocess(df)

    # Ensure no missing values remain
    assert pd.DataFrame(X_train).isna().sum().sum() == 0
    assert pd.DataFrame(X_test).isna().sum().sum() == 0
