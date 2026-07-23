"""
app.py
-------
Loan Approval Assistant - Tkinter Desktop Application.

Run:
    python train_model.py   # one-time: generates data + trains model
    python app.py            # launches the GUI
"""

import os
import json
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # backend switched to TkAgg once embedded below
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from preprocess import load_dataset
from predictor import LoanPredictor
from utils import (
    validate_applicant_form, save_prediction_to_history,
    load_history, search_history, export_history_to_excel, HISTORY_FILE
)

BASE_DIR = os.path.dirname(__file__)
METRICS_PATH = os.path.join(BASE_DIR, "models", "metrics.json")
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")

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

    # ------------------------------------------------------------------
    # Layout scaffolding
    # ------------------------------------------------------------------
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

    # Shared helper so child frames can grab the latest applicant form data
    def get_applicant_from_form(self):
        return self.frames["Applicant Info"].get_form_data()


# ==========================================================================
# 1. Dashboard
# ==========================================================================
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


# ==========================================================================
# 2. Applicant Information
# ==========================================================================
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


# ==========================================================================
# 3. Prediction Module
# ==========================================================================
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
                "No trained model found. Please run 'python train_model.py' first."
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


# ==========================================================================
# 4. Analytics Dashboard
# ==========================================================================
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


# ==========================================================================
# 5. Model Performance
# ==========================================================================
class ModelPerformanceFrame(tk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, bg=APP_BG)
        self.app = app

        tk.Label(self, text="Model Performance", font=FONT_TITLE, bg=APP_BG,
                  fg=TEXT_LIGHT).pack(anchor="w", padx=30, pady=(30, 4))
        tk.Label(self, text="Metrics from the most recent training run (train_model.py).",
                  font=FONT_BODY, bg=APP_BG, fg=TEXT_MUTED).pack(anchor="w", padx=30, pady=(0, 20))

        self.table_frame = tk.Frame(self, bg=PANEL_BG)
        self.table_frame.pack(fill="both", expand=True, padx=30, pady=(0, 20))

    def on_show(self):
        for widget in self.table_frame.winfo_children():
            widget.destroy()

        if not os.path.exists(METRICS_PATH):
            tk.Label(self.table_frame, text="No metrics found. Run 'python train_model.py' first.",
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


# ==========================================================================
# 6. History Module
# ==========================================================================
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


if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        print("No trained model found — training one now with default synthetic data...")
        from train_model import train_all_models
        train_all_models(compare=True)

    app = LoanApprovalApp()
    app.mainloop()
