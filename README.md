# Loan Approval Assistant (v4)

A desktop ML application (Python + Tkinter + Scikit-learn) that predicts
loan approval likelihood and shows an approval probability score.

## Setup

```bash
pip install -r requirements.txt
```

`tkinter` ships with most Python installs. If missing:
- **Windows/macOS:** included with the standard python.org installer.
- **Linux (Debian/Ubuntu):** `sudo apt-get install python3-tk`

## 1. Train the model

```bash
python train_model.py
```

This will:
- Generate `loan_dataset.csv` (a realistic synthetic dataset) if none exists yet.
  **To use your own data**, just drop a `loan_dataset.csv` with the same
  column names into the project root before running this step.
- Clean, encode, and scale the data.
- Train Logistic Regression (primary) plus Decision Tree, Random Forest,
  SVM, and KNN for comparison.
- Save `model.pkl` (full bundle: model + encoders + scaler) and
  `models/logistic_regression.pkl`.
- Save evaluation metrics to `models/metrics.json`.

## 2. Launch the GUI

```bash
python app.py
```

If no model exists yet, `app.py` will train one automatically on first run.

## App Modules

| Module | Description |
|---|---|
| Dashboard | Overview stats: total applications, approval rate, averages |
| Applicant Info | Form to enter applicant details, with input validation |
| Prediction | Runs the model, shows Approved/Rejected, probability %, confidence |
| Analytics | Approval rate, income distribution, credit history split, loan amount stats |
| Model Performance | Accuracy, precision, recall, F1, ROC-AUC, confusion matrix (all trained models) |
| History | Every prediction is saved; search by any field, export to Excel |

## Project Structure

```
Loan_Approval_Assistant/
├── app.py              # Tkinter GUI (entry point)
├── train_model.py       # Training pipeline
├── predictor.py          # Single-applicant prediction wrapper
├── preprocess.py         # Dataset generation + cleaning/encoding/scaling
├── utils.py              # Validation, history storage, Excel export
├── requirements.txt
├── loan_dataset.csv      # Created on first run
├── model.pkl              # Created on first run
├── models/
│   ├── logistic_regression.pkl
│   └── metrics.json
├── history/
│   └── predictions.xlsx
└── reports/
    └── analytics_report_*.xlsx
```

## Using Your Own Dataset

Replace `loan_dataset.csv` with your own file using these columns before
running `train_model.py`:

```
Applicant_ID, Gender, Married, Dependents, Education, Self_Employed,
ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term,
Credit_History, Property_Area, Existing_Loans, Loan_Status
```

`Loan_Status` should be `Y` (approved) or `N` (rejected) — this is the
training label and won't be present at prediction time in the GUI.

## Notes

- The bundled dataset generator (`preprocess.generate_synthetic_dataset`)
  creates data with a realistic, learnable approval pattern so the demo
  works out of the box — swap in real data for production use.
- All predictions are logged to `history/predictions.xlsx` automatically.
