"""Evaluation metrics shared by every experiment."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize


def compute_metrics(y, prob, classes):
    C = len(classes)
    prob = np.clip(prob, 1e-9, 1.0)
    prob = prob / prob.sum(1, keepdims=True)
    pred = prob.argmax(1)
    out = {
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "precision_macro": precision_score(y, pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y, pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y, pred, average="macro", zero_division=0),
        "precision_weighted": precision_score(y, pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y, pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y, pred, average="weighted", zero_division=0),
        "mcc": matthews_corrcoef(y, pred),
        "kappa": cohen_kappa_score(y, pred),
        "log_loss": log_loss(y, prob, labels=list(range(C))),
    }
    if C == 2:
        out["roc_auc"] = roc_auc_score(y, prob[:, 1])
        out["pr_auc"] = average_precision_score(y, prob[:, 1])
    else:
        out["roc_auc"] = roc_auc_score(y, prob, multi_class="ovr", average="macro")
        out["pr_auc"] = average_precision_score(label_binarize(y, classes=list(range(C))), prob, average="macro")
    # Collapse to benign-vs-malware to report detection quality for every task.
    b = classes.index("Benign")
    mal_true, mal_pred = y != b, pred != b
    out["malware_detection_rate"] = float(mal_pred[mal_true].mean())
    out["false_positive_rate"] = float(mal_pred[~mal_true].mean())
    return {k: float(v) for k, v in out.items()}
