"""Closed-set classification metrics (PROTOCOL.md §13.1, §23 "Metrics")."""

import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from avr.evaluation.classification import classification_metrics

FIXTURE = Path(__file__).parents[1] / "fixtures" / "ranking" / "toy_problem.json"


@pytest.fixture(scope="module")
def toy() -> dict:
    return json.loads(FIXTURE.read_text())


def test_hand_calculated_classification_metrics(toy):
    expected = toy["expected"]
    metrics = classification_metrics(
        toy["true_class_ids"], expected["top1_predictions"], toy["class_ids"]
    )
    assert metrics.accuracy == pytest.approx(expected["accuracy"])
    assert metrics.macro_precision == pytest.approx(expected["macro_precision"])
    assert metrics.macro_recall == pytest.approx(expected["macro_recall"])
    assert metrics.macro_f1 == pytest.approx(expected["macro_f1"])
    assert metrics.weighted_precision == pytest.approx(expected["weighted_precision"])
    assert metrics.weighted_recall == pytest.approx(expected["weighted_recall"])
    assert metrics.weighted_f1 == pytest.approx(expected["weighted_f1"])
    assert metrics.balanced_accuracy == pytest.approx(expected["balanced_accuracy"])
    assert metrics.classes_without_predicted_positives == 1
    assert metrics.per_class_recall == pytest.approx(expected["per_class_recall"])
    assert metrics.per_class_f1 == pytest.approx(expected["per_class_f1"])
    np.testing.assert_allclose(metrics.confusion_normalised, expected["confusion_normalised"])
    assert metrics.class_ids == ["a", "b", "c", "d"]


def test_matches_scikit_learn_on_fixture(toy):
    y_true, y_pred = toy["true_class_ids"], toy["expected"]["top1_predictions"]
    labels = list("abcd")
    metrics = classification_metrics(y_true, y_pred, labels)
    for average in ("macro", "weighted"):
        p, r, f, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, average=average, zero_division=0
        )
        assert getattr(metrics, f"{average}_precision") == pytest.approx(p)
        assert getattr(metrics, f"{average}_recall") == pytest.approx(r)
        assert getattr(metrics, f"{average}_f1") == pytest.approx(f)
    assert metrics.balanced_accuracy == pytest.approx(balanced_accuracy_score(y_true, y_pred))
    np.testing.assert_allclose(
        metrics.confusion_normalised,
        confusion_matrix(y_true, y_pred, labels=labels, normalize="true"),
    )


def test_macro_over_all_eligible_classes_including_unpredicted_and_absent_ones():
    # Class "e" is eligible but has no queries and no predictions: it still enters the macro
    # average (recall 0, precision 0 with zero_division=0), as §13.1 requires, and its
    # confusion row is all zeros rather than NaN.
    y_true = ["a", "a", "b"]
    y_pred = ["a", "a", "a"]
    labels = ["a", "b", "e"]
    metrics = classification_metrics(y_true, y_pred, labels)
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    assert metrics.macro_precision == pytest.approx(p) == pytest.approx(2 / 9)  # (2/3+0+0)/3
    assert metrics.macro_recall == pytest.approx(r) == pytest.approx(1 / 3)
    assert metrics.macro_f1 == pytest.approx(f)
    assert metrics.classes_without_predicted_positives == 2
    assert metrics.confusion_normalised[2].tolist() == [0.0, 0.0, 0.0]
    # Balanced accuracy averages recall over classes present in y_true only (a: 1, b: 0),
    # matching scikit-learn rather than dividing by every eligible class.
    assert metrics.balanced_accuracy == pytest.approx(balanced_accuracy_score(y_true, y_pred))
    assert metrics.balanced_accuracy == pytest.approx(0.5)


def test_random_labels_match_scikit_learn():
    rng = np.random.default_rng(7)
    labels = [f"k{i}" for i in range(12)]
    y_true = list(rng.choice(labels[:10], size=300))
    y_pred = list(rng.choice(labels[:9], size=300))
    metrics = classification_metrics(y_true, y_pred, labels)
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    assert [metrics.per_class_recall[c] for c in labels] == pytest.approx(r.tolist())
    assert [metrics.per_class_f1[c] for c in labels] == pytest.approx(f.tolist())
    assert metrics.macro_f1 == pytest.approx(f.mean())
    assert metrics.classes_without_predicted_positives == 3
    assert metrics.balanced_accuracy == pytest.approx(balanced_accuracy_score(y_true, y_pred))


def test_labels_outside_eligible_classes_raise():
    with pytest.raises(ValueError):
        classification_metrics(["a", "zz"], ["a", "a"], ["a", "b"])
    with pytest.raises(ValueError):
        classification_metrics(["a", "a"], ["a", "zz"], ["a", "b"])
