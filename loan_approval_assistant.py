"""
loan_approval_assistant.py
============================
Loan Approval Assistant (v4) - SINGLE-FILE EDITION

Everything from the multi-file project (dataset generation,
preprocessing, model training, prediction, history/export utilities,
and the Tkinter GUI) is combined into this one script.

Run:
    python loan_approval_assistant.py

On first run it will:
  1. Generate a synthetic loan_dataset.csv (if none exists in this folder)
  2. Preprocess the data and train Logistic Regression (+ comparison models)
  3. Save model.pkl / models/metrics.json
  4. Launch the Tkinter GUI

To use your OWN data: drop a loan_dataset.csv (same columns, see README
section at the bottom of this file) into the same folder before running.

Requirements:
    pip install pandas numpy scikit-learn matplotlib joblib openpyxl
"""

import os
import json
import datetime
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

import matplotlib
matplotlib.use("Agg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


# ==========================================================================
# PATHS / CONSTANTS
# ==========================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
HISTORY_DIR = os.path.join(BASE_DIR, "history")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(HISTORY_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

DATA_PATH = os.path.join(BASE_DIR, "loan_dataset.csv")
MODEL_BUNDLE_PATH = os.path.join(BASE_DIR, "model.pkl")
LOGREG_PATH = os.path.join(MODELS_DIR, "logistic_regression.pkl")
METRICS_PATH = os.path.join(MODELS_DIR, "metrics.json")
HISTORY_FILE = os.path.join(HISTORY_DIR, "predictions.xlsx")

RANDOM_STATE = 42

CATEGORICAL_COLUMNS = [
    "Gender", "Married", "Dependents", "Education",
    "Self_Employed", "Property_Area"
]
NUMERIC_COLUMNS = [
    "ApplicantIncome", "CoapplicantIncome", "LoanAmount",
    "Loan_Amount_Term", "Credit_History", "Existing_Loans"
]
HISTORY_COLUMNS = [
    "Timestamp", "Applicant_ID", "Gender", "Married", "Dependents",
    "Education", "Self_Employed", "ApplicantIncome", "CoapplicantIncome",
    "LoanAmount", "Loan_Amount_Term", "Credit_History", "Property_Area",
    "Existing_Loans", "Prediction", "Approval_Probability", "Confidence",
]

CANDIDATE_MODELS_FACTORY = lambda: {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE, max_depth=6),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
    "SVM": SVC(probability=True, random_state=RANDOM_STATE),
    "KNN": KNeighborsClassifier(n_neighbors=7),
}

# GUI theme
APP_BG = "#0f1b2d"
PANEL_BG = "#16233a"
ACCENT = "#3fa7ff"
ACCENT_GREEN = "#33cc7a"
ACCENT_RED = "#ff5c5c"
TEXT_LIGHT = "#eaf1fb"
TEXT_MUTED = "#93a3bb"
FONT_TITLE = ("Segoe UI", 20, "bold")
FONT_HEADER = ("Segoe UI", 14, "bold")
FONT_BODY = ("Segoe UI", 10)
FONT_BODY_BOLD = ("Segoe UI", 10, "bold")


# ==========================================================================
# SECTION 1: DATASET GENERATION + PREPROCESSING
# ==========================================================================
def generate_synthetic_dataset(n_samples=800, save=True, path=DATA_PATH):
    """Creates a synthetic but realistic loan-application dataset."""
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

    for col in ["LoanAmount", "Loan_Amount_Term", "Credit_History", "Self_Employed"]:
        idx = rng.choice(n_samples, size=max(1, n_samples // 50), replace=False)
        df.loc[idx, col] = np.nan

    if save:
        df.to_csv(path, index=False)

    return df


def load_dataset(path=DATA_PATH):
    if not os.path.exists(path):
        return generate_synthetic_dataset(save=True, path=path)
    return pd.read_csv(path)


def clean_data(df):
    df = df.copy()
    for col in CATEGORICAL_COLUMNS:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode(dropna=True)[0])
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    return df


def encode_features(df, encoders=None):
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
    df = df.copy()
    if scaler is None:
        scaler = StandardScaler()
        df[feature_cols] = scaler.fit_transform(df[feature_cols])
    else:
        df[feature_cols] = scaler.transform(df[feature_cols])
    return df, scaler


def full_preprocess_pipeline(df, target_col="Loan_Status", test_size=0.2):
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
        "X_train": X_train_scaled, "X_test": X_test_scaled,
        "y_train": y_train, "y_test": y_test,
        "encoders": encoders, "scaler": scaler, "feature_cols": feature_cols,
    }


# ==========================================================================
# SECTION 2: MODEL TRAINING
# ==========================================================================
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1_score": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }


def train_all_models(compare=True):
    df = load_dataset()
    prep = full_preprocess_pipeline(df)
    X_train, X_test = prep["X_train"], prep["X_test"]
    y_train, y_test = prep["y_train"], prep["y_test"]

    all_candidates = CANDIDATE_MODELS_FACTORY()
    models_to_run = all_candidates if compare else {
        "Logistic Regression": all_candidates["Logistic Regression"]
    }

    results = {}
    for name, model in models_to_run.items():
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)
        results[name] = (model, metrics)
        print(f"[{name}] accuracy={metrics['accuracy']}  f1={metrics['f1_score']}  "
              f"roc_auc={metrics['roc_auc']}")

    primary_model = results["Logistic Regression"][0]
    bundle = {
        "model": primary_model,
        "encoders": prep["encoders"],
        "scaler": prep["scaler"],
        "feature_cols": prep["feature_cols"],
    }
    joblib.dump(bundle, MODEL_BUNDLE_PATH)
    joblib.dump(primary_model, LOGREG_PATH)

    all_metrics = {name: res[1] for name, res in results.items()}
    with open(METRICS_PATH, "w") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"\nSaved primary model bundle -> {MODEL_BUNDLE_PATH}")
    print(f"Saved metrics -> {METRICS_PATH}")
    return results


# ==========================================================================
# SECTION 3: PREDICTOR
# ==========================================================================
class LoanPredictor:
    def __init__(self, bundle_path=MODEL_BUNDLE_PATH):
        if not os.path.exists(bundle_path):
            raise FileNotFoundError("No trained model found. Train first.")
        bundle = joblib.load(bundle_path)
        self.model = bundle["model"]
        self.encoders = bundle["encoders"]
        self.scaler = bundle["scaler"]
        self.feature_cols = bundle["feature_cols"]

    def predict(self, applicant: dict):
        df = pd.DataFrame([applicant])
        df = df[self.feature_cols]

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


# ==========================================================================
# SECTION 4: UTILITIES (validation, history, export)
# ==========================================================================
def is_valid_number(value, allow_zero=True, allow_float=True):
    try:
        num = float(value) if allow_float else int(value)
    except (ValueError, TypeError):
        return False
    if not allow_zero and num == 0:
        return False
    return num >= 0


def validate_applicant_form(fields: dict):
    errors = []
    required_text = ["Gender", "Married", "Dependents", "Education",
                      "Self_Employed", "Property_Area"]
    for field in required_text:
        if not fields.get(field):
            errors.append(f"'{field.replace('_', ' ')}' is required.")

    numeric_fields = {
        "ApplicantIncome": False, "CoapplicantIncome": True,
        "LoanAmount": False, "Loan_Amount_Term": False, "Existing_Loans": True,
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
    if os.path.exists(HISTORY_FILE):
        return pd.read_excel(HISTORY_FILE)
    return pd.DataFrame(columns=HISTORY_COLUMNS)


def search_history(query: str):
    df = load_history()
    if not query:
        return df
    mask = df.astype(str).apply(lambda col: col.str.contains(query, case=False, na=False))
    return df[mask.any(axis=1)]


def export_history_to_excel(dest_path=None):
    df = load_history()
    if dest_path is None:
        dest_path = os.path.join(
            REPORTS_DIR,
            f"analytics_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        )
    df.to_excel(dest_path, index=False)
    return dest_path


# ==========================================================================
# SECTION 5: TKINTER GUI
# ==========================================================================
class LoanApprovalApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Loan Approval Assistant")
        self.geometry("1180x720")
        self.minsize(1000, 650)
        self.configure(bg=APP_BG)

        self.predictor = None
        self._load_predictor_safe()

        self._build_layout()
        self.show_frame("Dashboard")

    def _load_predictor_safe(self):
        try:
            self.predictor = LoanPredictor()
        except FileNotFoundError:
            self.predictor = None

    def _build_layout(self):
        nav = tk.Frame(self, bg=PANEL_BG, width=210)
        nav.pack(side="left", fill="y")
        nav.pack_propagate(False)

        tk.Label(
            nav, text="Loan Approval\nAssistant", bg=PANEL_BG, fg=TEXT_LIGHT,
            font=FONT_HEADER, justify="left", anchor="w", padx=18, pady=24
        ).pack(fill="x")

        self.nav_buttons = {}
        for name in ["Dashboard", "Applicant Info", "Prediction",
                     "Analytics", "Model Performance", "History"]:
            btn = tk.Button(
                nav, text=name, bg=PANEL_BG, fg=TEXT_MUTED, bd=0,
                font=FONT_BODY_BOLD, anchor="w", padx=20, pady=12,
                activebackground=APP_BG, activeforeground=ACCENT,
                command=lambda n=name: self.show_frame(n)
            )
            btn.pack(fill="x")
            self.nav_buttons[name] = btn

        self.container = tk.Frame(self, bg=APP_BG)
        self.container.pack(side="right", fill="both", expand=True)

        self.frames = {}
        for FrameClass, name in [
            (DashboardFrame, "Dashboard"),
            (ApplicantInfoFrame, "Applicant Info"),
            (PredictionFrame, "Prediction"),
            (AnalyticsFrame, "Analytics"),
            (ModelPerformanceFrame, "Model Performance"),
            (HistoryFrame, "History"),
        ]:
            frame = FrameClass(self.container, self)
            self.frames[name] = frame
            frame.place(relx=0, rely=0, relwidth=1, relheight=1)

    def show_frame(self, name):
        for n, btn in self.nav_buttons.items():
            btn.configure(fg=ACCENT if n == name else TEXT_MUTED,
                          bg=APP_BG if n == name else PANEL_BG)
        frame = self.frames[name]
        frame.tkraise()
        if hasattr(frame, "on_show"):
            frame.on_show()

    def get_applicant_from_form(self):
        return self.frames["Applicant Info"].get_form_data()


class DashboardFrame(tk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, bg=APP_BG)
        self.app = app

        tk.Label(self, text="Loan Approval Assistant", font=FONT_TITLE,
                  bg=APP_BG, fg=TEXT_LIGHT).pack(anchor="w", padx=30, pady=(30, 4))
        tk.Label(self, text="ML-powered screening dashboard for loan officers",
                  font=FONT_BODY, bg=APP_BG, fg=TEXT_MUTED).pack(anchor="w", padx=30)

        self.stats_frame = tk.Frame(self, bg=APP_BG)
        self.stats_frame.pack(fill="x", padx=30, pady=30)

        self.cards = {}
        for key in ["Total Applications", "Approval Rate", "Avg Loan Amount", "Avg Credit History"]:
            card = tk.Frame(self.stats_frame, bg=PANEL_BG, width=250, height=110)
            card.pack(side="left", padx=10, fill="both", expand=True)
            card.pack_propagate(False)
            tk.Label(card, text=key, bg=PANEL_BG, fg=TEXT_MUTED, font=FONT_BODY).pack(
                anchor="w", padx=16, pady=(16, 4))
            value_lbl = tk.Label(card, text="--", bg=PANEL_BG, fg=ACCENT, font=("Segoe UI", 22, "bold"))
            value_lbl.pack(anchor="w", padx=16)
            self.cards[key] = value_lbl

        info = tk.Frame(self, bg=PANEL_BG)
        info.pack(fill="both", expand=True, padx=30, pady=(0, 30))
        tk.Label(info, text="Quick Guide", bg=PANEL_BG, fg=TEXT_LIGHT, font=FONT_HEADER).pack(
            anchor="w", padx=20, pady=(16, 6))
        guide_text = (
            "1. Go to 'Applicant Info' and enter the applicant's details.\n"
            "2. Move to 'Prediction' to run the model and see the approval probability.\n"
            "3. Check 'Analytics' for portfolio-level trends and 'Model Performance' for accuracy metrics.\n"
            "4. Every prediction is saved automatically under 'History', where it can be searched or exported to Excel."
        )
        tk.Label(info, text=guide_text, bg=PANEL_BG, fg=TEXT_MUTED, font=FONT_BODY,
                 justify="left").pack(anchor="w", padx=20, pady=(0, 20))

    def on_show(self):
        try:
            df = load_history()
            dataset = load_dataset()
            total = len(df)
            approval_rate = (df["Prediction"].eq("Approved").mean() * 100) if total else 0
            avg_loan = dataset["LoanAmount"].mean()
            avg_credit = dataset["Credit_History"].mean()

            self.cards["Total Applications"].configure(text=str(total))
            self.cards["Approval Rate"].configure(text=f"{approval_rate:.1f}%")
            self.cards["Avg Loan Amount"].configure(text=f"{avg_loan:,.0f}")
            self.cards["Avg Credit History"].configure(text=f"{avg_credit:.2f}")
        except Exception:
            pass


class ApplicantInfoFrame(tk.Frame):
    FIELD_DEFS = [
        ("Applicant_ID", "text", None),
        ("Gender", "combo", ["Male", "Female"]),
        ("Married", "combo", ["Yes", "No"]),
        ("Dependents", "combo", ["0", "1", "2", "3+"]),
        ("Education", "combo", ["Graduate", "Not Graduate"]),
        ("Self_Employed", "combo", ["Yes", "No"]),
        ("ApplicantIncome", "text", None),
        ("CoapplicantIncome", "text", None),
        ("LoanAmount", "text", None),
        ("Loan_Amount_Term", "text", None),
        ("Credit_History", "combo", ["1", "0"]),
        ("Property_Area", "combo", ["Urban", "Semiurban", "Rural"]),
        ("Existing_Loans", "text", None),
    ]

    def __init__(self, parent, app):
        super().__init__(parent, bg=APP_BG)
        self.app = app
        self.widgets = {}

        tk.Label(self, text="Applicant Information", font=FONT_TITLE, bg=APP_BG,
                  fg=TEXT_LIGHT).pack(anchor="w", padx=30, pady=(30, 4))
        tk.Label(self, text="Enter applicant details, then head to the Prediction tab.",
                  font=FONT_BODY, bg=APP_BG, fg=TEXT_MUTED).pack(anchor="w", padx=30, pady=(0, 20))

        form = tk.Frame(self, bg=PANEL_BG)
        form.pack(fill="both", expand=True, padx=30, pady=(0, 20))

        cols = 2
        for i, (field, kind, options) in enumerate(self.FIELD_DEFS):
            row, col = divmod(i, cols)
            cell = tk.Frame(form, bg=PANEL_BG)
            cell.grid(row=row, column=col, sticky="ew", padx=20, pady=12)
            form.grid_columnconfigure(col, weight=1)

            label = field.replace("_", " ")
            tk.Label(cell, text=label, bg=PANEL_BG, fg=TEXT_MUTED, font=FONT_BODY).pack(anchor="w")

            if kind == "combo":
                var = tk.StringVar(value=options[0])
                widget = ttk.Combobox(cell, textvariable=var, values=options, state="readonly")
            else:
                var = tk.StringVar()
                widget = tk.Entry(cell, textvariable=var, bg="#0d1826", fg=TEXT_LIGHT,
                                   insertbackground=TEXT_LIGHT, relief="flat")
            widget.pack(fill="x", ipady=4, pady=(4, 0))
            self.widgets[field] = var

        btn_row = tk.Frame(self, bg=APP_BG)
        btn_row.pack(fill="x", padx=30, pady=(0, 30))
        tk.Button(btn_row, text="Clear Form", command=self.clear_form, bg=PANEL_BG,
                  fg=TEXT_LIGHT, bd=0, padx=16, pady=8).pack(side="left")
        tk.Button(btn_row, text="Go to Prediction  →", command=lambda: app.show_frame("Prediction"),
                  bg=ACCENT, fg="#04121f", bd=0, padx=16, pady=8, font=FONT_BODY_BOLD).pack(side="right")

    def get_form_data(self):
        return {field: var.get().strip() for field, var in self.widgets.items()}

    def clear_form(self):
        for field, var in self.widgets.items():
            if isinstance(var, tk.StringVar):
                var.set("")


class PredictionFrame(tk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, bg=APP_BG)
        self.app = app

        tk.Label(self, text="Loan Prediction", font=FONT_TITLE, bg=APP_BG,
                  fg=TEXT_LIGHT).pack(anchor="w", padx=30, pady=(30, 4))
        tk.Label(self, text="Runs the trained Logistic Regression model on the current applicant.",
                  font=FONT_BODY, bg=APP_BG, fg=TEXT_MUTED).pack(anchor="w", padx=30, pady=(0, 20))

        tk.Button(self, text="Predict Loan Approval", command=self.run_prediction,
                  bg=ACCENT, fg="#04121f", font=FONT_BODY_BOLD, bd=0, padx=20, pady=12
                  ).pack(anchor="w", padx=30)

        self.result_panel = tk.Frame(self, bg=PANEL_BG)
        self.result_panel.pack(fill="both", expand=True, padx=30, pady=20)

        self.status_lbl = tk.Label(self.result_panel, text="No prediction yet.",
                                    font=("Segoe UI", 26, "bold"), bg=PANEL_BG, fg=TEXT_MUTED)
        self.status_lbl.pack(pady=(40, 10))

        self.prob_lbl = tk.Label(self.result_panel, text="", font=("Segoe UI", 16),
                                  bg=PANEL_BG, fg=TEXT_LIGHT)
        self.prob_lbl.pack()

        self.conf_lbl = tk.Label(self.result_panel, text="", font=("Segoe UI", 13),
                                  bg=PANEL_BG, fg=TEXT_MUTED)
        self.conf_lbl.pack(pady=(4, 20))

    def run_prediction(self):
        if self.app.predictor is None:
            messagebox.showerror(
                "Model Not Found",
                "No trained model found. Please restart the app so it can train one."
            )
            return

        applicant = self.app.get_applicant_from_form()
        is_valid, errors = validate_applicant_form(applicant)
        if not is_valid:
            messagebox.showwarning("Invalid Input", "\n".join(errors))
            return

        payload = dict(applicant)
        for numeric_field in ["ApplicantIncome", "CoapplicantIncome", "LoanAmount",
                               "Loan_Amount_Term", "Existing_Loans"]:
            payload[numeric_field] = float(payload[numeric_field])
        payload["Credit_History"] = float(payload["Credit_History"])

        try:
            result = self.app.predictor.predict(payload)
        except Exception as e:
            messagebox.showerror("Prediction Error", str(e))
            return

        approved = result["prediction"] == "Approved"
        color = ACCENT_GREEN if approved else ACCENT_RED
        self.status_lbl.configure(text=result["prediction"], fg=color)
        self.prob_lbl.configure(text=f"Approval Probability: {result['approval_probability']}%")
        self.conf_lbl.configure(text=f"Model Confidence: {result['confidence']}%")

        save_prediction_to_history(payload, result)
        messagebox.showinfo("Saved", "Prediction saved to history.")


class AnalyticsFrame(tk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, bg=APP_BG)
        self.app = app
        self.canvas_widget = None

        tk.Label(self, text="Analytics Dashboard", font=FONT_TITLE, bg=APP_BG,
                  fg=TEXT_LIGHT).pack(anchor="w", padx=30, pady=(30, 4))
        tk.Label(self, text="Approval rates, income distribution, credit history, and loan amounts.",
                  font=FONT_BODY, bg=APP_BG, fg=TEXT_MUTED).pack(anchor="w", padx=30, pady=(0, 10))

        self.chart_area = tk.Frame(self, bg=APP_BG)
        self.chart_area.pack(fill="both", expand=True, padx=20, pady=10)

    def on_show(self):
        for widget in self.chart_area.winfo_children():
            widget.destroy()

        df = load_dataset()
        fig = Figure(figsize=(11, 6), dpi=100, facecolor=APP_BG)

        ax1 = fig.add_subplot(221)
        approval_counts = df["Loan_Status"].value_counts()
        ax1.bar(approval_counts.index.map({"Y": "Approved", "N": "Rejected"}),
                approval_counts.values, color=[ACCENT_GREEN, ACCENT_RED])
        ax1.set_title("Approval Rate", color=TEXT_LIGHT, fontsize=10)
        self._style_axis(ax1)

        ax2 = fig.add_subplot(222)
        ax2.hist(df["ApplicantIncome"].dropna(), bins=20, color=ACCENT)
        ax2.set_title("Applicant Income Distribution", color=TEXT_LIGHT, fontsize=10)
        self._style_axis(ax2)

        ax3 = fig.add_subplot(223)
        credit_counts = df["Credit_History"].value_counts()
        ax3.pie(credit_counts.values, labels=["Good (1)", "Poor (0)"] if len(credit_counts) > 1 else ["Good (1)"],
                autopct="%1.0f%%", colors=[ACCENT_GREEN, ACCENT_RED],
                textprops={"color": TEXT_LIGHT, "fontsize": 8})
        ax3.set_title("Credit History Split", color=TEXT_LIGHT, fontsize=10)

        ax4 = fig.add_subplot(224)
        ax4.boxplot(df["LoanAmount"].dropna(), vert=False, patch_artist=True,
                    boxprops=dict(facecolor=ACCENT))
        ax4.set_title("Loan Amount Statistics", color=TEXT_LIGHT, fontsize=10)
        self._style_axis(ax4)

        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=self.chart_area)
        canvas.draw()
        self.canvas_widget = canvas.get_tk_widget()
        self.canvas_widget.pack(fill="both", expand=True)

    @staticmethod
    def _style_axis(ax):
        ax.set_facecolor(PANEL_BG)
        ax.tick_params(colors=TEXT_MUTED, labelsize=8)
        for spine in ax.spines.values():
            spine.set_color(TEXT_MUTED)


class ModelPerformanceFrame(tk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, bg=APP_BG)
        self.app = app

        tk.Label(self, text="Model Performance", font=FONT_TITLE, bg=APP_BG,
                  fg=TEXT_LIGHT).pack(anchor="w", padx=30, pady=(30, 4))
        tk.Label(self, text="Metrics from the most recent training run.",
                  font=FONT_BODY, bg=APP_BG, fg=TEXT_MUTED).pack(anchor="w", padx=30, pady=(0, 20))

        self.table_frame = tk.Frame(self, bg=PANEL_BG)
        self.table_frame.pack(fill="both", expand=True, padx=30, pady=(0, 20))

    def on_show(self):
        for widget in self.table_frame.winfo_children():
            widget.destroy()

        if not os.path.exists(METRICS_PATH):
            tk.Label(self.table_frame, text="No metrics found yet.",
                      bg=PANEL_BG, fg=TEXT_MUTED, font=FONT_BODY).pack(padx=20, pady=20)
            return

        with open(METRICS_PATH) as f:
            metrics = json.load(f)

        cols = ["Model", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
        tree = ttk.Treeview(self.table_frame, columns=cols, show="headings", height=6)
        for c in cols:
            tree.heading(c, text=c)
            tree.column(c, anchor="center", width=140)
        tree.pack(fill="x", padx=20, pady=20)

        for name, m in metrics.items():
            tree.insert("", "end", values=(
                name, m["accuracy"], m["precision"], m["recall"], m["f1_score"], m["roc_auc"]
            ))

        logreg_metrics = metrics.get("Logistic Regression", {})
        cm = logreg_metrics.get("confusion_matrix")
        if cm:
            cm_frame = tk.Frame(self.table_frame, bg=PANEL_BG)
            cm_frame.pack(padx=20, pady=(0, 20), anchor="w")
            tk.Label(cm_frame, text="Confusion Matrix (Logistic Regression)",
                      bg=PANEL_BG, fg=TEXT_LIGHT, font=FONT_BODY_BOLD).grid(
                row=0, column=0, columnspan=2, sticky="w", pady=(0, 8))
            labels = [["TN", "FP"], ["FN", "TP"]]
            for r in range(2):
                for c in range(2):
                    cell = tk.Label(
                        cm_frame, text=f"{labels[r][c]}: {cm[r][c]}",
                        bg="#0d1826", fg=TEXT_LIGHT, font=FONT_BODY, width=14, height=2
                    )
                    cell.grid(row=r + 1, column=c, padx=4, pady=4)


class HistoryFrame(tk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, bg=APP_BG)
        self.app = app

        tk.Label(self, text="Prediction History", font=FONT_TITLE, bg=APP_BG,
                  fg=TEXT_LIGHT).pack(anchor="w", padx=30, pady=(30, 4))
        tk.Label(self, text="Search past applications or export results to Excel.",
                  font=FONT_BODY, bg=APP_BG, fg=TEXT_MUTED).pack(anchor="w", padx=30, pady=(0, 10))

        controls = tk.Frame(self, bg=APP_BG)
        controls.pack(fill="x", padx=30)

        self.search_var = tk.StringVar()
        entry = tk.Entry(controls, textvariable=self.search_var, bg="#0d1826", fg=TEXT_LIGHT,
                          insertbackground=TEXT_LIGHT, relief="flat", width=40)
        entry.pack(side="left", ipady=6, padx=(0, 10))
        entry.bind("<Return>", lambda e: self.refresh_table())

        tk.Button(controls, text="Search", command=self.refresh_table, bg=ACCENT,
                  fg="#04121f", bd=0, padx=14, pady=6, font=FONT_BODY_BOLD).pack(side="left")
        tk.Button(controls, text="Clear", command=self.clear_search, bg=PANEL_BG,
                  fg=TEXT_LIGHT, bd=0, padx=14, pady=6).pack(side="left", padx=6)
        tk.Button(controls, text="Export to Excel", command=self.export_history, bg=ACCENT_GREEN,
                  fg="#04121f", bd=0, padx=14, pady=6, font=FONT_BODY_BOLD).pack(side="right")

        table_wrap = tk.Frame(self, bg=PANEL_BG)
        table_wrap.pack(fill="both", expand=True, padx=30, pady=20)

        self.tree = ttk.Treeview(table_wrap, show="headings")
        vsb = ttk.Scrollbar(table_wrap, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        self.tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

    def on_show(self):
        self.refresh_table()

    def refresh_table(self):
        query = self.search_var.get().strip()
        df = search_history(query) if query else load_history()

        self.tree.delete(*self.tree.get_children())
        self.tree["columns"] = list(df.columns)
        for col in df.columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=110, anchor="center")

        for _, row in df.iterrows():
            self.tree.insert("", "end", values=list(row))

    def clear_search(self):
        self.search_var.set("")
        self.refresh_table()

    def export_history(self):
        dest = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx")],
            initialfile="analytics_report.xlsx",
        )
        if not dest:
            return
        path = export_history_to_excel(dest)
        messagebox.showinfo("Exported", f"History exported to:\n{path}")


# ==========================================================================
# ENTRY POINT
# ==========================================================================
if __name__ == "__main__":
    if not os.path.exists(MODEL_BUNDLE_PATH):
        print("No trained model found — training one now with default synthetic data...")
        train_all_models(compare=True)

    app = LoanApprovalApp()
    app.mainloop()
