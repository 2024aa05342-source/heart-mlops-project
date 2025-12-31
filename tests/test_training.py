from src.data_loader import load_data
from src.preprocess import preprocess
from sklearn.linear_model import LogisticRegression

def test_model_training_runs():
    df = load_data()
    # UPDATED — unpack 3 returns from preprocess
    (X_train, X_test, y_train, y_test), scaler, cols = preprocess(df)

    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    # Model must produce predictions equal to test-set length
    assert len(preds) == len(y_test)

    # Basic sanity check — score should give a valid float value
    assert model.score(X_test, y_test) >= 0
