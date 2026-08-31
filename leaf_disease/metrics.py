"""Classification metrics without sklearn, so the eval script stays lightweight."""

from __future__ import annotations

import numpy as np


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true.tolist(), y_pred.tolist()):
        cm[int(t), int(p)] += 1
    return cm


def per_class_prf(cm: np.ndarray) -> dict[str, np.ndarray]:
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp), where=(precision + recall) > 0)
    return {"precision": precision, "recall": recall, "f1": f1}


def overall_accuracy(cm: np.ndarray) -> float:
    return float(cm.trace() / max(cm.sum(), 1))


def summarize(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]) -> dict:
    cm = confusion_matrix(y_true, y_pred, num_classes=len(class_names))
    prf = per_class_prf(cm)
    return {
        "accuracy": overall_accuracy(cm),
        "macro_precision": float(prf["precision"].mean()),
        "macro_recall": float(prf["recall"].mean()),
        "macro_f1": float(prf["f1"].mean()),
        "per_class": {
            name: {
                "precision": float(prf["precision"][i]),
                "recall": float(prf["recall"][i]),
                "f1": float(prf["f1"][i]),
                "support": int(cm[i].sum()),
            }
            for i, name in enumerate(class_names)
        },
        "confusion": cm.tolist(),
    }
