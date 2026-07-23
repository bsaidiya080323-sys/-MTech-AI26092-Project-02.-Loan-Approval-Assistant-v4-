"""
predictor.py
-------------
Loads the saved model bundle and provides a simple interface for
predicting loan approval for a single applicant (as used by the GUI).
"""

import os
import joblib
import pandas as pd

BASE_DIR = os.path.dirname(__file__)
MODEL_BUNDLE_PATH = os.path.join(BASE_DIR, "model.pkl")


class LoanPredictor:
    def __init__(self, bundle_path=MODEL_BUNDLE_PATH):
        if not os.path.exists(bundle_path):
            raise FileNotFoundError(
                "No trained model found. Run 'python train_model.py' first."
            )
        bundle = joblib.load(bundle_path)
        self.model = bundle["model"]
        self.encoders = bundle["encoders"]
        self.scaler = bundle["scaler"]
        self.feature_cols = bundle["feature_cols"]

    def predict(self, applicant: dict):
        """
        applicant: dict with keys matching the raw feature columns, e.g.
            {
              "Gender": "Male", "Married": "Yes", "Dependents": "0",
              "Education": "Graduate", "Self_Employed": "No",
              "ApplicantIncome": 5000, "CoapplicantIncome": 0,
              "LoanAmount": 150, "Loan_Amount_Term": 360,
              "Credit_History": 1, "Property_Area": "Urban",
              "Existing_Loans": 0
            }

        Returns: dict with prediction label, probability (%), and confidence.
        """
        df = pd.DataFrame([applicant])
        df = df[self.feature_cols]

        # Encode categorical columns using the fitted encoders
        for col, le in self.encoders.items():
            if col in df.columns:
                df[col] = df[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
                df[col] = le.transform(df[col])

        df_scaled = df.copy()
        df_scaled[self.feature_cols] = self.scaler.transform(df[self.feature_cols])

        proba = self.model.predict_proba(df_scaled)[0]
        approval_prob = proba[1] * 100
        prediction = "Approved" if approval_prob >= 50 else "Rejected"
        confidence = max(proba) * 100

        return {
            "prediction": prediction,
            "approval_probability": round(approval_prob, 2),
            "confidence": round(confidence, 2),
        }


if __name__ == "__main__":
    predictor = LoanPredictor()
    sample = {
        "Gender": "Male", "Married": "Yes", "Dependents": "0",
        "Education": "Graduate", "Self_Employed": "No",
        "ApplicantIncome": 6000, "CoapplicantIncome": 1500,
        "LoanAmount": 120, "Loan_Amount_Term": 360,
        "Credit_History": 1, "Property_Area": "Urban",
        "Existing_Loans": 0,
    }
    print(predictor.predict(sample))
