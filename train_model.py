"""
train_model.py
---------------
Trains the primary Logistic Regression model (plus optional comparison
models), evaluates them, and saves the trained artifacts with joblib.
"""

import os
import json
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

from preprocess import load_dataset, full_preprocess_pipeline

BASE_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

MODEL_BUNDLE_PATH = os.path.join(BASE_DIR, "model.pkl")
LOGREG_PATH = os.path.join(MODELS_DIR, "logistic_regression.pkl")
METRICS_PATH = os.path.join(MODELS_DIR, "metrics.json")

CANDIDATE_MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=6),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "SVM": SVC(probability=True, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=7),
}


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
    """
    Trains the primary Logistic Regression model, and optionally the
    comparison models, returning a dict of {name: (model, metrics)}.
    """
    df = load_dataset()
    prep = full_preprocess_pipeline(df)

    X_train, X_test = prep["X_train"], prep["X_test"]
    y_train, y_test = prep["y_train"], prep["y_test"]

    models_to_run = CANDIDATE_MODELS if compare else {
        "Logistic Regression": CANDIDATE_MODELS["Logistic Regression"]
    }

    results = {}
    for name, model in models_to_run.items():
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)
        results[name] = (model, metrics)
        print(f"[{name}] accuracy={metrics['accuracy']}  f1={metrics['f1_score']}  "
              f"roc_auc={metrics['roc_auc']}")

    # Persist the primary model bundle (model + encoders + scaler + feature order)
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


if __name__ == "__main__":
    train_all_models(compare=True)
