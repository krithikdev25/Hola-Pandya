"""Train the supply-chain adverse-outcome probability model.

Model: logistic regression on standardized operational/financial features.
Why logistic regression?  It gives a genuine probability model with an
explicit equation p(y=1|x)=sigmoid(b0 + sum(w_i z_i)), and every coefficient
can be inspected.  The *decision threshold is NOT hard-coded*: it is selected
on a validation split by minimizing a documented expected-cost function, then
saved with the model artifacts and loaded by the runtime agent.
"""
import json, os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, roc_auc_score, average_precision_score,
                             brier_score_loss, confusion_matrix, classification_report,
                             precision_score, recall_score, f1_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "historical_financing_outcomes.csv")
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

FEATURES = [
    "buyer_risk_score", "buyer_payment_history", "supplier_reliability",
    "production_delay_days", "shipment_delay_days", "payment_delay_days",
    "inventory_age_days", "deterioration", "invoice_dispute_flag",
    "duplicate_financing_flag", "source_confidence", "stage_progress",
    "advance_rate_requested", "financing_utilization", "demand_volatility",
]
TARGET = "adverse_outcome"

# Business policy, not a model parameter: missing a genuinely risky exposure is
# deliberately more costly than reviewing a false positive.
FALSE_NEGATIVE_COST = 5.0
FALSE_POSITIVE_COST = 1.0


def choose_threshold(y, p):
    candidates = np.linspace(0.02, 0.98, 193)
    best = None
    for t in candidates:
        pred = (p >= t).astype(int)
        fn = int(((y == 1) & (pred == 0)).sum())
        fp = int(((y == 0) & (pred == 1)).sum())
        cost = FALSE_NEGATIVE_COST * fn + FALSE_POSITIVE_COST * fp
        if best is None or cost < best["expected_cost"]:
            best = {"threshold": float(t), "expected_cost": float(cost), "false_negatives": fn, "false_positives": fp}
    return best


def main():
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURES].astype(float).values
    y = df[TARGET].astype(int).values

    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.40, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.50, random_state=43, stratify=y_tmp
    )

    scaler = StandardScaler().fit(X_train)
    Xtr, Xv, Xte = scaler.transform(X_train), scaler.transform(X_val), scaler.transform(X_test)

    model = LogisticRegression(max_iter=2500, random_state=42)
    model.fit(Xtr, y_train)

    p_val = model.predict_proba(Xv)[:, 1]
    threshold_info = choose_threshold(y_val, p_val)
    threshold = threshold_info["threshold"]

    p_test = model.predict_proba(Xte)[:, 1]
    pred_test = (p_test >= threshold).astype(int)
    report = classification_report(y_test, pred_test, output_dict=True, zero_division=0)

    # Risk-band thresholds are derived from validation probabilities.  These
    # are descriptive bands; the operating threshold above is the action gate.
    low_cut = float(np.quantile(p_val, 0.50))
    high_cut = float(np.quantile(p_val, 0.90))
    low_cut = min(low_cut, threshold)
    high_cut = max(high_cut, threshold)

    metrics = {
        "model": "logistic_regression",
        "dataset_rows": int(len(df)),
        "n_train": int(len(y_train)), "n_validation": int(len(y_val)), "n_test": int(len(y_test)),
        "test_auc": float(roc_auc_score(y_test, p_test)),
        "test_average_precision": float(average_precision_score(y_test, p_test)),
        "test_brier_score": float(brier_score_loss(y_test, p_test)),
        "test_accuracy_at_learned_threshold": float(accuracy_score(y_test, pred_test)),
        "precision_risky": float(precision_score(y_test, pred_test, zero_division=0)),
        "recall_risky": float(recall_score(y_test, pred_test, zero_division=0)),
        "f1_risky": float(f1_score(y_test, pred_test, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, pred_test).tolist(),
        "classification_report": report,
        "operating_threshold": threshold_info,
        "risk_band_thresholds": {"low_cut": low_cut, "high_cut": high_cut},
        "threshold_method": "validation expected-cost minimization",
        "false_negative_cost": FALSE_NEGATIVE_COST,
        "false_positive_cost": FALSE_POSITIVE_COST,
        "coefficients": {f: float(c) for f, c in zip(FEATURES, model.coef_[0])},
        "intercept": float(model.intercept_[0]),
        "feature_order": FEATURES,
        "formula": "p(adverse)=1/(1+exp(-(intercept + sum(coef_i * standardized_feature_i))))",
    }

    joblib.dump(model, os.path.join(MODEL_DIR, "risk_model.joblib"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "risk_scaler.joblib"))
    with open(os.path.join(MODEL_DIR, "model_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Trained logistic regression on {len(df):,} rows")
    print(f"Holdout AUC: {metrics['test_auc']:.3f} | PR-AUC: {metrics['test_average_precision']:.3f} | Brier: {metrics['test_brier_score']:.3f}")
    print(f"LEARNED OPERATING THRESHOLD: {threshold:.3f} (validation expected cost={threshold_info['expected_cost']:.0f})")
    print(f"Risk bands derived from validation: LOW < {low_cut:.3f}, MODERATE < {high_cut:.3f}, HIGH >= {high_cut:.3f}")
    print("Saved risk_model.joblib, risk_scaler.joblib, model_metrics.json")

if __name__ == "__main__":
    main()
