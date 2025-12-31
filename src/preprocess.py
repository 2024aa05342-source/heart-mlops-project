import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

def preprocess(df):
    # Remove non-useful columns
    df = df.drop(columns=['id', 'dataset'], errors='ignore')

    # Convert target to binary
    y = (df['num'] > 0).astype(int)

    # Split features
    X = df.drop('num', axis=1)

    # One-hot encode
    X = pd.get_dummies(X, drop_first=True)

    # Save column names (order needed for inference)
    cols = list(X.columns)

    # Handle missing values
    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X)

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # RETURN **3 VALUES**
    return (X_train, X_test, y_train, y_test), scaler, cols


if __name__ == "__main__":
    from src.data_loader import load_data

    df = load_data()
    (_, _, _, _), scaler, cols = preprocess(df)

    print("Features:", len(cols))
    print(cols)
