"""Common evaluation functions used by training, tests, and the Streamlit app."""

from typing import Any, Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def score_model(model: Any, features, target) -> Dict[str, Any]:
    """Score a fitted pipeline using Churn=Yes as the positive class."""
    predictions = model.predict(features)
    scores = np.asarray(model.predict_proba(features))[:, 1] if hasattr(model, "predict_proba") else model.decision_function(features)
    auc_value = None
    auc_reason = None
    if len(np.unique(target)) < 2:
        auc_reason = "ROC AUC is undefined because the evaluation data has one target class."
    else:
        auc_value = float(roc_auc_score(target, scores))
    return {
        "Accuracy": float(accuracy_score(target, predictions)),
        "AUC": auc_value,
        "Precision": float(precision_score(target, predictions, zero_division=0)),
        "Recall": float(recall_score(target, predictions, zero_division=0)),
        "F1": float(f1_score(target, predictions, zero_division=0)),
        "MCC": float(matthews_corrcoef(target, predictions)),
        "confusion_matrix": confusion_matrix(target, predictions, labels=[0, 1]).tolist(),
        "classification_report": classification_report(target, predictions, labels=[0, 1], target_names=["No", "Yes"], output_dict=True, zero_division=0),
        "auc_reason": auc_reason,
    }


def comparison_row(model_name: str, result: Dict[str, Any]) -> Dict[str, Any]:
    """Select the required comparison columns in their prescribed order."""
    return {"ML Model Name": model_name, "Accuracy": result["Accuracy"], "AUC": result["AUC"], "Precision": result["Precision"], "Recall": result["Recall"], "F1": result["F1"], "MCC": result["MCC"]}