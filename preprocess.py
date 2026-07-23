"""
preprocess.py
--------------
Handles dataset generation (if no dataset is present), cleaning,
encoding, and scaling for the Loan Approval Assistant.
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
DATA_PATH = os.path.join(os.path.dirname(__file__), "loan_dataset.csv")

CATEGORICAL_COLUMNS = [
    "Gender", "Married", "Dependents", "Education",
    "Self_Employed", "Property_Area"
]

NUMERIC_COLUMNS = [
    "ApplicantIncome", "CoapplicantIncome", "LoanAmount",
    "Loan_Amount_Term", "Credit_History", "Existing_Loans"
]


def generate_synthetic_dataset(n_samples=800, save=True, path=DATA_PATH):
    """
    Creates a synthetic but realistic loan-application dataset.
    Used automatically if no CSV dataset is supplied by the user.
    """
    rng = np.random.default_rng(RANDOM_STATE)

    genders = rng.choice(["Male", "Female"], n_samples, p=[0.6, 0.4])
    married = rng.choice(["Yes", "No"], n_samples, p=[0.65, 0.35])
    dependents = rng.choice(["0", "1", "2", "3+"], n_samples, p=[0.5, 0.2, 0.2, 0.1])
    education = rng.choice(["Graduate", "Not Graduate"], n_samples, p=[0.78, 0.22])
    self_employed = rng.choice(["Yes", "No"], n_samples, p=[0.14, 0.86])
    property_area = rng.choice(["Urban", "Semiurban", "Rural"], n_samples)

    applicant_income = rng.gamma(shape=5, scale=1200, size=n_samples).round(2)
    coapplicant_income = rng.gamma(shape=2, scale=800, size=n_samples).round(2)
    coapplicant_income[rng.random(n_samples) < 0.35] = 0.0

    loan_amount = (
        (applicant_income + coapplicant_income) * rng.uniform(0.02, 0.12, n_samples)
    ).round(1)

    loan_amount_term = rng.choice(
        [360, 180, 120, 60, 300, 84], n_samples, p=[0.6, 0.15, 0.1, 0.05, 0.05, 0.05]
    )
    credit_history = rng.choice([1, 0], n_samples, p=[0.84, 0.16]).astype(float)
    existing_loans = rng.choice([0, 1, 2, 3], n_samples, p=[0.55, 0.25, 0.13, 0.07])

    applicant_id = [f"APP{1000 + i}" for i in range(n_samples)]

    # ---- Approval logic (drives a realistic, learnable target) ----
    total_income = applicant_income + coapplicant_income
    debt_to_income = loan_amount / (total_income + 1)

    score = (
        2.6 * credit_history
        - 3.0 * debt_to_income
        + 0.00025 * total_income
        - 0.35 * existing_loans
        + np.where(education == "Graduate", 0.25, -0.1)
        + rng.normal(0, 0.6, n_samples)
    )
    approval_prob = 1 / (1 + np.exp(-score))
    loan_status = np.where(approval_prob > 0.5, "Y", "N")

    df = pd.DataFrame({
        "Applicant_ID": applicant_id,
        "Gender": genders,
        "Married": married,
        "Dependents": dependents,
        "Education": education,
        "Self_Employed": self_employed,
        "ApplicantIncome": applicant_income,
        "CoapplicantIncome": coapplicant_income,
        "LoanAmount": loan_amount,
        "Loan_Amount_Term": loan_amount_term,
        "Credit_History": credit_history,
        "Property_Area": property_area,
        "Existing_Loans": existing_loans,
        "Loan_Status": loan_status,
    })

    # Inject a few missing values to exercise the cleaning pipeline
    for col in ["LoanAmount", "Loan_Amount_Term", "Credit_History", "Self_Employed"]:
        idx = rng.choice(n_samples, size=max(1, n_samples // 50), replace=False)
        df.loc[idx, col] = np.nan

    if save:
        df.to_csv(path, index=False)

    return df


def load_dataset(path=DATA_PATH):
    """Loads the dataset from disk, generating one if it doesn't exist."""
    if not os.path.exists(path):
        return generate_synthetic_dataset(save=True, path=path)
    return pd.read_csv(path)


def clean_data(df):
    """Handles missing values for categorical and numeric columns."""
    df = df.copy()

    for col in CATEGORICAL_COLUMNS:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode(dropna=True)[0])

    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    return df


def encode_features(df, encoders=None):
    """
    Label-encodes categorical columns.
    If `encoders` (dict of fitted LabelEncoders) is given, reuse them
    (needed at prediction time so categories map consistently).
    """
    df = df.copy()
    fitted = encoders is not None
    encoders = encoders or {}

    for col in CATEGORICAL_COLUMNS:
        if col not in df.columns:
            continue
        if fitted:
            le = encoders[col]
            df[col] = df[col].map(lambda x, le=le: x if x in le.classes_ else le.classes_[0])
            df[col] = le.transform(df[col])
        else:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le

    return df, encoders


def scale_features(df, feature_cols, scaler=None):
    """Standardizes numeric feature columns."""
    df = df.copy()
    if scaler is None:
        scaler = StandardScaler()
        df[feature_cols] = scaler.fit_transform(df[feature_cols])
    else:
        df[feature_cols] = scaler.transform(df[feature_cols])
    return df, scaler


def full_preprocess_pipeline(df, target_col="Loan_Status", test_size=0.2):
    """
    Runs the entire preprocessing pipeline end-to-end and returns
    train/test splits plus the fitted encoders/scaler for reuse at
    prediction time.
    """
    df = clean_data(df)

    y = df[target_col].map({"Y": 1, "N": 0}).astype(int)
    X = df.drop(columns=[target_col, "Applicant_ID"], errors="ignore")

    X_encoded, encoders = encode_features(X)

    feature_cols = list(X_encoded.columns)
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )

    X_train_scaled, scaler = scale_features(X_train, feature_cols)
    X_test_scaled, _ = scale_features(X_test, feature_cols, scaler=scaler)

    return {
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
        "y_train": y_train,
        "y_test": y_test,
        "encoders": encoders,
        "scaler": scaler,
        "feature_cols": feature_cols,
    }


if __name__ == "__main__":
    data = load_dataset()
    print(f"Dataset loaded/generated with shape: {data.shape}")
    print(data.head())
