from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix


@dataclass
class EvalResults:
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float | None
    conf_mat: np.ndarray


def evaluate(model, dataloader, device: torch.device) -> EvalResults:
    model.eval()
    ys, ps, probs = [], [], []
    with torch.no_grad():
        for batch in dataloader:
            x_cnn = batch["x_cnn"].to(device)
            x_lstm = batch["x_lstm"].to(device)
            y = batch["y"].to(device)
            logits, _, _, _ = model(x_cnn, x_lstm)
            prob = torch.sigmoid(logits)
            p = (prob >= 0.5).long()

            ys.append(y.cpu().numpy())
            ps.append(p.cpu().numpy())
            probs.append(prob.cpu().numpy())

    y_true = np.concatenate(ys)
    y_pred = np.concatenate(ps)
    y_prob = np.concatenate(probs)

    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = None
    cm = confusion_matrix(y_true, y_pred)

    return EvalResults(acc, precision, recall, f1, auc, cm)
