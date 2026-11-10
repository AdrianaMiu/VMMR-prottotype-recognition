"""Prototype ranking metrics (PROTOCOL.md §13.2, §23 "Metrics")."""

import json
from pathlib import Path

import numpy as np
import pytest

from avr.evaluation.ranking import (
    average_precision_single_relevant,
    rank_of_true_class,
    ranking_metrics,
    recall_at_k_single_relevant,
)

FIXTURE = Path(__file__).parents[1] / "fixtures" / "ranking" / "toy_problem.json"


@pytest.fixture(scope="module")
def toy() -> dict:
    return json.loads(FIXTURE.read_text())


@pytest.fixture(scope="module")
def toy_ranks(toy) -> np.ndarray:
    index = {c: i for i, c in enumerate(toy["class_ids"])}
    return np.array([[index[c] for c in row] for row in toy["expected"]["rank_order"]])


def test_rank_of_true_class_is_one_based(toy, toy_ranks):
    ranks = rank_of_true_class(toy_ranks, toy["true_class_ids"], toy["class_ids"])
    assert ranks.tolist() == toy["expected"]["rank_of_true"]


def test_hand_calculated_ranking_metrics(toy, toy_ranks):
    expected = toy["expected"]
    metrics = ranking_metrics(
        toy_ranks, toy["true_class_ids"], toy["class_ids"], tied_queries=expected["tied_queries"]
    )
    assert metrics.query_count == 6
    assert metrics.class_count == 4
    assert metrics.tied_queries == 1
    assert metrics.top_k == {k: pytest.approx(expected["top_k"][str(k)]) for k in (1, 5, 10, 20)}
    assert metrics.mrr == pytest.approx(expected["mrr"])
    assert metrics.mean_rank == pytest.approx(expected["mean_rank"])
    assert metrics.median_rank == pytest.approx(expected["median_rank"])
    assert metrics.rank_p75 == pytest.approx(expected["rank_p75"])
    assert metrics.rank_p90 == pytest.approx(expected["rank_p90"])
    assert metrics.rank_p95 == pytest.approx(expected["rank_p95"])
    assert metrics.mean_normalised_rank == pytest.approx(expected["mean_normalised_rank"])


def test_custom_k_values(toy, toy_ranks):
    metrics = ranking_metrics(toy_ranks, toy["true_class_ids"], toy["class_ids"], ks=(2, 3))
    assert metrics.top_k[2] == pytest.approx(0.5)
    assert metrics.top_k[3] == pytest.approx(5 / 6)


def test_single_relevant_ap_and_recall_on_a_five_query_example():
    # Five queries whose true prototype sits at rank 1, 2, 4, 1 and 3. With one relevant
    # item per query (§13.2), AP is the precision at that hit: 1/1, 1/2, 1/4, 1/1, 1/3; and
    # Recall@k is the fraction of queries whose hit is within the first k.
    true_ranks = np.array([1, 2, 4, 1, 3])

    ap = average_precision_single_relevant(true_ranks)

    np.testing.assert_allclose(ap, [1.0, 0.5, 0.25, 1.0, 0.333333333], rtol=1e-8)
    assert ap.mean() == pytest.approx(0.616666667)
    assert recall_at_k_single_relevant(true_ranks, 1) == pytest.approx(0.4)
    assert recall_at_k_single_relevant(true_ranks, 2) == pytest.approx(0.6)
    assert recall_at_k_single_relevant(true_ranks, 3) == pytest.approx(0.8)
    assert recall_at_k_single_relevant(true_ranks, 4) == 1.0
    assert recall_at_k_single_relevant(true_ranks, 10) == 1.0


def test_unknown_true_class_raises(toy, toy_ranks):
    with pytest.raises(ValueError, match="not among"):
        rank_of_true_class(toy_ranks, ["a", "b", "c", "d", "a", "zz"], toy["class_ids"])
