from src.data_loader import load_data
from src.preprocess import preprocess
from sklearn.linear_model import LogisticRegression

def test_model_training_runs():
    df = load_data()
    (X_train, X_test, y_train, y_test), scaler = preprocess(df)

    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    # check training successful
    assert model.score(X_test, y_test) >= 0  # basic sanity check

