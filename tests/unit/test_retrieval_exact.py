"""Exact cosine search (PROTOCOL.md §12 "Scientific evaluation", §23 "Prototypes and ranking")."""

import json
from pathlib import Path

import numpy as np
import pytest

from avr.retrieval.exact import rank_by_cosine, rank_by_cosine_faiss

FIXTURE = Path(__file__).parents[1] / "fixtures" / "ranking" / "toy_problem.json"
BACKENDS = [rank_by_cosine, rank_by_cosine_faiss]


def normalise(rows) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.float32)
    return rows / np.linalg.norm(rows, axis=1, keepdims=True)


@pytest.fixture(scope="module")
def toy() -> dict:
    return json.loads(FIXTURE.read_text())


@pytest.mark.parametrize("rank", BACKENDS)
def test_hand_calculated_fixture_matches_cosine_ranking(toy, rank):
    result = rank(normalise(toy["queries"]), normalise(toy["prototypes"]), toy["class_ids"])
    ranked_ids = [[toy["class_ids"][j] for j in row] for row in result.ranks]
    assert ranked_ids == toy["expected"]["rank_order"]
    assert result.ranks.shape == (6, 4)
    assert result.tied_queries == toy["expected"]["tied_queries"]


@pytest.mark.parametrize("rank", BACKENDS)
def test_ties_break_by_class_id_not_by_prototype_position(toy, rank):
    # Present the prototypes in reverse order: ranks are indices into that reversed array, but
    # q5's b/c tie must still put "b" first because the rule is lexicographic class id.
    class_ids = toy["class_ids"][::-1]
    prototypes = normalise(toy["prototypes"])[::-1]
    result = rank(normalise(toy["queries"]), prototypes, class_ids)
    ranked_ids = [[class_ids[j] for j in row] for row in result.ranks]
    assert ranked_ids == toy["expected"]["rank_order"]


@pytest.mark.parametrize("rank", BACKENDS)
def test_all_scores_tied_orders_classes_lexicographically(rank):
    prototypes = np.eye(3, dtype=np.float32)
    query = normalise([[1, 1, 1]])
    result = rank(query, prototypes, ["m", "k", "z"])
    assert [["m", "k", "z"][j] for j in result.ranks[0]] == ["k", "m", "z"]
    assert result.tied_queries == 1
    assert result.tied_queries_at(1) == 1


def test_tied_query_count_depends_on_k():
    # Scores (1.0, 0.5, 0.5, 0.2): the tie sits at positions 2-3, so it touches top-1 only
    # through the boundary rule (a tie between positions k and k+1 counts for k).
    prototypes = np.eye(4, dtype=np.float32)
    query = np.array([[1.0, 0.5, 0.5, 0.2]], dtype=np.float32)
    result = rank_by_cosine(query, prototypes, ["a", "b", "c", "d"])
    assert result.tied_queries_at(1) == 0
    assert result.tied_queries_at(2) == 1
    assert result.tied_queries_at(3) == 1
    assert result.tied_queries == 1


def test_wrong_shapes_raise():
    with pytest.raises(ValueError):
        rank_by_cosine(np.ones((2, 3), np.float32), np.ones((4, 2), np.float32), list("abcd"))
    with pytest.raises(ValueError):
        rank_by_cosine(np.ones((2, 3), np.float32), np.ones((4, 3), np.float32), list("abc"))


def test_faiss_and_matmul_ranks_are_identical_on_random_vectors():
    # 1,000 random L2-normalised float32 queries against 50 random prototypes in 64-d. Both
    # backends round scores to 6 decimals before sorting, so BLAS versus FAISS summation-order
    # noise (a few float32 ulp, ~1e-7) cannot flip the order of two distinct scores.
    rng = np.random.default_rng(20261106)
    queries = normalise(rng.standard_normal((1000, 64)))
    prototypes = normalise(rng.standard_normal((50, 64)))
    class_ids = [f"c{i:02d}" for i in range(50)]
    matmul = rank_by_cosine(queries, prototypes, class_ids)
    faiss = rank_by_cosine_faiss(queries, prototypes, class_ids)
    np.testing.assert_array_equal(matmul.ranks, faiss.ranks)
    assert matmul.tied_queries == faiss.tied_queries
    assert matmul.ranks.shape == (1000, 50)
    assert all(sorted(row) == list(range(50)) for row in matmul.ranks)


def test_backend_disagreement_is_bounded_by_rounding_noise():
    # Rounding cannot be perfect: when two raw scores differ by less than the ~2e-7 backend
    # noise and straddle a 1e-6 rounding boundary, one backend sees a tie and the other does
    # not. Across seeds this happens for well under 1% of queries, and whenever it does the
    # two backends' sorted score vectors still agree to within one rounding step.
    class_ids = [f"c{i:02d}" for i in range(50)]
    differing = 0
    for seed in range(10):
        rng = np.random.default_rng(seed)
        queries = normalise(rng.standard_normal((1000, 64)))
        prototypes = normalise(rng.standard_normal((50, 64)))
        matmul = rank_by_cosine(queries, prototypes, class_ids)
        faiss = rank_by_cosine_faiss(queries, prototypes, class_ids)
        differing += int((matmul.ranks != faiss.ranks).any(axis=1).sum())
        np.testing.assert_allclose(matmul.scores, faiss.scores, atol=1.5e-6, rtol=0)
    assert differing <= 10
