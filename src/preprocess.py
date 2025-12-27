from src.data_loader import load_data
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


def preprocess(df):
    # Remove non-useful columns
    df = df.drop(columns=['id','dataset'], errors='ignore')

    # Split features and target
    X = df.drop('num', axis=1)   # num is target
    y = df['num']

    # Convert categorical features to numeric
    X = pd.get_dummies(X, drop_first=True)
    
    # Handle missing values
    imputer = SimpleImputer(strategy="median")   # median better for medical data
    X_imputed = imputer.fit_transform(X)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    return train_test_split(X_scaled, y, test_size=0.2, random_state=42), scaler

if __name__ == "__main__":
    from src.data_loader import load_data
    df = load_data()
    (X_train, X_test, y_train, y_test), scaler = preprocess(df)
    print(X_train.shape, X_test.shape)
