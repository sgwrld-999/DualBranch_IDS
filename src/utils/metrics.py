from __future__ import annotations

import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve, auc


def pr_curve(y_true: np.ndarray, y_prob: np.ndarray):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    return precision, recall, auc(recall, precision)


def roc_curve_data(y_true: np.ndarray, y_prob: np.ndarray):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return fpr, tpr, auc(fpr, tpr)
