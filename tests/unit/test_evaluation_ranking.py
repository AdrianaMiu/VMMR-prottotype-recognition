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


def test_single_relevant_identities(toy, toy_ranks):
    # With exactly one relevant prototype per query, AP == reciprocal rank and
    # Recall@k == Hit@k (§13.2); both are asserted on the fixture ranks [1,1,1,3,3,4].
    ranks = rank_of_true_class(toy_ranks, toy["true_class_ids"], toy["class_ids"])
    ap = average_precision_single_relevant(ranks)
    np.testing.assert_allclose(ap, 1 / ranks)
    assert ap.mean() == pytest.approx(toy["expected"]["mrr"])
    for k in (1, 2, 3, 4):
        hit_at_k = (ranks <= k).mean()
        assert recall_at_k_single_relevant(ranks, k) == pytest.approx(hit_at_k)
    assert recall_at_k_single_relevant(ranks, 3) == pytest.approx(5 / 6)


def test_unknown_true_class_raises(toy, toy_ranks):
    with pytest.raises(ValueError, match="not among"):
        rank_of_true_class(toy_ranks, ["a", "b", "c", "d", "a", "zz"], toy["class_ids"])
