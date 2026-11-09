"""Hierarchy-level metrics derived from a class ranking (PROTOCOL.md §13.1, §13.2).

Expected values are read off the hand-built rank matrix in the comments.
"""

import numpy as np
import pytest

from avr.evaluation.hierarchy import (
    error_decomposition,
    parent_ranking_metrics,
    rank_of_true_parent,
)

# classes c0..c3; parents (e.g. makes): c0,c1 -> A; c2 -> B; c3 -> C
CLASS_IDS = ["c0", "c1", "c2", "c3"]
PARENT_OF = {"c0": "A", "c1": "A", "c2": "B", "c3": "C"}
# query 0 ranks c2, c0, c1, c3 ; true parent A -> first A at rank 2
# query 1 ranks c3, c1, c2, c0 ; true parent C -> rank 1
# query 2 ranks c0, c1, c3, c2 ; true parent B -> rank 4
RANKS = np.array([[2, 0, 1, 3], [3, 1, 2, 0], [0, 1, 3, 2]])
TRUE_PARENTS = ["A", "C", "B"]


def test_rank_of_true_parent_is_the_first_prototype_of_that_parent():
    ranks = rank_of_true_parent(RANKS, TRUE_PARENTS, CLASS_IDS, PARENT_OF)
    np.testing.assert_array_equal(ranks, [2, 1, 4])


def test_parent_ranking_metrics_count_distinct_parents():
    metrics = parent_ranking_metrics(RANKS, TRUE_PARENTS, CLASS_IDS, PARENT_OF, ks=(1, 2))
    assert metrics.class_count == 3  # A, B, C
    assert metrics.query_count == 3
    assert metrics.top_k == {1: pytest.approx(1 / 3), 2: pytest.approx(2 / 3)}
    assert metrics.mrr == pytest.approx((1 / 2 + 1 + 1 / 4) / 3)
    assert metrics.mean_rank == pytest.approx(7 / 3)
    # normalised rank with C = 3 parents: (2-1)/2, 0, (4-1)/2 -> mean 2/3
    assert metrics.mean_normalised_rank == pytest.approx(2 / 3)


def test_error_decomposition_for_three_level_labels():
    # labels are (make, model, class); predictions are top-1 classes
    true = ["1/10/2010", "1/10/2010", "1/10/2010", "1/10/2010", "2/20/2012"]
    pred = ["1/10/2010", "1/10/2011", "1/11/2010", "2/20/2012", "2/20/2012"]

    def levels(class_id: str) -> tuple[str, str]:
        make, model, _ = class_id.split("/")
        return make, f"{make}/{model}"

    d = error_decomposition(true, pred, levels)
    assert d.query_count == 5
    assert d.exact_correct == 2
    assert d.wrong_generation_same_model == 1
    assert d.wrong_model_same_make == 1
    assert d.wrong_make == 1
    assert d.model_correct == 3
    assert d.make_correct == 4
    assert d.exact_accuracy == pytest.approx(0.4)
    assert d.model_accuracy == pytest.approx(0.6)
    assert d.make_accuracy == pytest.approx(0.8)
    # generation accuracy given the model is right: 2 of 3
    assert d.generation_accuracy_given_model == pytest.approx(2 / 3)
    assert d.as_dict()["wrong_generation_same_model_fraction"] == pytest.approx(0.2)
