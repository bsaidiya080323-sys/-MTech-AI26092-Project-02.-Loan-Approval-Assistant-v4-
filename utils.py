"""
utils.py
---------
Shared helper functions: input validation, prediction history
persistence, and Excel export.
"""

import os
import datetime
import pandas as pd

BASE_DIR = os.path.dirname(__file__)
HISTORY_DIR = os.path.join(BASE_DIR, "history")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
HISTORY_FILE = os.path.join(HISTORY_DIR, "predictions.xlsx")

os.makedirs(HISTORY_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

HISTORY_COLUMNS = [
    "Timestamp", "Applicant_ID", "Gender", "Married", "Dependents",
    "Education", "Self_Employed", "ApplicantIncome", "CoapplicantIncome",
    "LoanAmount", "Loan_Amount_Term", "Credit_History", "Property_Area",
    "Existing_Loans", "Prediction", "Approval_Probability", "Confidence",
]


def is_valid_number(value, allow_zero=True, allow_float=True):
    """Validates that a string represents a usable numeric value."""
    try:
        num = float(value) if allow_float else int(value)
    except (ValueError, TypeError):
        return False
    if not allow_zero and num == 0:
        return False
    return num >= 0


def validate_applicant_form(fields: dict):
    """
    Validates the raw form fields collected from the GUI.
    Returns (is_valid: bool, errors: list[str]).
    """
    errors = []

    required_text = ["Gender", "Married", "Dependents", "Education",
                      "Self_Employed", "Property_Area"]
    for field in required_text:
        if not fields.get(field):
            errors.append(f"'{field.replace('_', ' ')}' is required.")

    numeric_fields = {
        "ApplicantIncome": False,
        "CoapplicantIncome": True,
        "LoanAmount": False,
        "Loan_Amount_Term": False,
        "Existing_Loans": True,
    }
    for field, allow_zero in numeric_fields.items():
        value = fields.get(field, "")
        if not is_valid_number(value, allow_zero=allow_zero):
            errors.append(f"'{field.replace('_', ' ')}' must be a valid non-negative number.")

    credit_history = fields.get("Credit_History")
    if credit_history not in ("0", "1", 0, 1):
        errors.append("'Credit History' must be 0 or 1.")

    return (len(errors) == 0), errors


def save_prediction_to_history(applicant: dict, result: dict):
    """Appends a prediction record to the history Excel file."""
    record = {
        "Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Applicant_ID": applicant.get("Applicant_ID", "N/A"),
        **{k: applicant.get(k, "") for k in HISTORY_COLUMNS if k in applicant},
        "Prediction": result["prediction"],
        "Approval_Probability": result["approval_probability"],
        "Confidence": result["confidence"],
    }

    if os.path.exists(HISTORY_FILE):
        history_df = pd.read_excel(HISTORY_FILE)
        history_df = pd.concat([history_df, pd.DataFrame([record])], ignore_index=True)
    else:
        history_df = pd.DataFrame([record], columns=HISTORY_COLUMNS)

    history_df.to_excel(HISTORY_FILE, index=False)
    return history_df


def load_history():
    """Loads prediction history, returning an empty DataFrame if none exists."""
    if os.path.exists(HISTORY_FILE):
        return pd.read_excel(HISTORY_FILE)
    return pd.DataFrame(columns=HISTORY_COLUMNS)


def search_history(query: str):
    """Case-insensitive search across all history columns."""
    df = load_history()
    if not query:
        return df
    mask = df.astype(str).apply(lambda col: col.str.contains(query, case=False, na=False))
    return df[mask.any(axis=1)]


def export_history_to_excel(dest_path=None):
    """Exports the current prediction history to an Excel report."""
    df = load_history()
    if dest_path is None:
        dest_path = os.path.join(
            REPORTS_DIR,
            f"analytics_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        )
    df.to_excel(dest_path, index=False)
    return dest_path
