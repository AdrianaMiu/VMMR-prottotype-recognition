"""Closed-set classification metrics from top-1 predictions (PROTOCOL.md §13.1).

Every metric is computed over all eligible classes passed in, not only the classes that
appear in the predictions. Classwise precision and F1 use ``zero_division=0`` when a class
has no predicted positives, and the number of such classes is reported. Weighted averages
weight each class by its number of true queries. Balanced accuracy is the mean recall over
the classes that occur in ``y_true`` (the scikit-learn definition), since recall is undefined
for a class with no queries. The confusion matrix is normalised by true class; a class with
no queries gets an all-zero row.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ClassificationMetrics:
    class_ids: list[str]
    query_count: int
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    weighted_precision: float
    weighted_recall: float
    weighted_f1: float
    balanced_accuracy: float
    classes_without_predicted_positives: int
    per_class_recall: dict[str, float]
    per_class_f1: dict[str, float]
    confusion_normalised: np.ndarray


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Elementwise division with ``zero_division=0``."""
    out = np.zeros_like(numerator, dtype=np.float64)
    np.divide(numerator, denominator, out=out, where=denominator != 0)
    return out


def classification_metrics(
    y_true: Sequence[str], y_pred: Sequence[str], class_ids: Sequence[str]
) -> ClassificationMetrics:
    if len(y_true) != len(y_pred):
        raise ValueError(f"{len(y_true)} true labels but {len(y_pred)} predictions")
    if len(set(class_ids)) != len(class_ids):
        raise ValueError("class ids must be unique")
    index = {c: i for i, c in enumerate(class_ids)}
    for name, labels in (("y_true", y_true), ("y_pred", y_pred)):
        unknown = sorted({c for c in labels if c not in index})
        if unknown:
            raise ValueError(f"{name} contains labels outside the eligible classes: {unknown[:5]}")

    c = len(class_ids)
    t = np.array([index[v] for v in y_true], dtype=np.int64)
    p = np.array([index[v] for v in y_pred], dtype=np.int64)
    confusion = np.zeros((c, c), dtype=np.int64)
    np.add.at(confusion, (t, p), 1)

    tp = np.diag(confusion).astype(np.float64)
    support = confusion.sum(axis=1).astype(np.float64)  # true count per class
    predicted = confusion.sum(axis=0).astype(np.float64)  # predicted count per class
    precision = _safe_divide(tp, predicted)
    recall = _safe_divide(tp, support)
    f1 = _safe_divide(2 * precision * recall, precision + recall)

    present = support > 0
    weights = support / support.sum() if support.sum() else np.zeros(c)
    return ClassificationMetrics(
        class_ids=list(class_ids),
        query_count=int(len(y_true)),
        accuracy=float(tp.sum() / len(y_true)) if len(y_true) else 0.0,
        macro_precision=float(precision.mean()),
        macro_recall=float(recall.mean()),
        macro_f1=float(f1.mean()),
        weighted_precision=float(precision @ weights),
        weighted_recall=float(recall @ weights),
        weighted_f1=float(f1 @ weights),
        balanced_accuracy=float(recall[present].mean()) if present.any() else 0.0,
        classes_without_predicted_positives=int((predicted == 0).sum()),
        per_class_recall={cid: float(recall[i]) for i, cid in enumerate(class_ids)},
        per_class_f1={cid: float(f1[i]) for i, cid in enumerate(class_ids)},
        confusion_normalised=_safe_divide(confusion.astype(np.float64), support[:, None]),
    )
