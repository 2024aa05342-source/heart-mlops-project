import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

def preprocess(df):
    # Remove non-useful columns
    df = df.drop(columns=['id', 'dataset'], errors='ignore')

    # Convert target to binary (very important)
    y = (df['num'] > 0).astype(int)

    # Split features
    X = df.drop('num', axis=1)

    # One-hot encode categorical features
    X = pd.get_dummies(X, drop_first=True)

    # Handle missing values
    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X)

    # Scale numeric features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # Return scaler + **feature column names**
    return (X_train, X_test, y_train, y_test), scaler, list(X.columns)


if __name__ == "__main__":
    from src.data_loader import load_data

    df = load_data()
    (_, _, _, _), scaler, cols = preprocess(df)

    print("Preprocessing successful.")
    print("Total features after encoding:", len(cols))
    print("\nFeature Order for Prediction:")
    print(cols)
