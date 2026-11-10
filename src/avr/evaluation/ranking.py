"""Prototype ranking metrics with one correct class per query (PROTOCOL.md §13.2).

Ranks are one-based. ``ranks[q, r]`` is the prototype index at rank ``r + 1`` as returned by
``avr.retrieval.exact``; the rank of the true class is the position of its prototype plus
one. Top-k with ``k >= C`` is 1 by definition. Percentiles use numpy's default linear
interpolation. Normalised rank is ``(rank - 1) / (C - 1)`` with zero as best (zero for every
query when ``C == 1``).

With a single relevant prototype, average precision equals the reciprocal rank and
Recall@k equals Hit@k, so only MRR and Top-k are reported; the two helper functions exist
to assert those identities in tests, not to add redundant rows to metric tables.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

DEFAULT_KS = (1, 5, 10, 20)


@dataclass(frozen=True)
class RankingMetrics:
    query_count: int
    class_count: int
    tied_queries: int | None
    top_k: dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    mean_rank: float = 0.0
    median_rank: float = 0.0
    rank_p75: float = 0.0
    rank_p90: float = 0.0
    rank_p95: float = 0.0
    mean_normalised_rank: float = 0.0


def rank_of_true_class(
    ranks: np.ndarray, true_class_ids: Sequence[str], class_ids: Sequence[str]
) -> np.ndarray:
    """One-based rank of the true class prototype for every query."""
    ranks = np.asarray(ranks)
    if ranks.ndim != 2 or ranks.shape[1] != len(class_ids):
        raise ValueError(f"ranks must be [q, C={len(class_ids)}], got shape {ranks.shape}")
    if len(true_class_ids) != ranks.shape[0]:
        raise ValueError(f"{ranks.shape[0]} rank rows but {len(true_class_ids)} true class ids")
    index = {c: i for i, c in enumerate(class_ids)}
    unknown = sorted({c for c in true_class_ids if c not in index})
    if unknown:
        raise ValueError(f"true class ids not among the ranked classes: {unknown[:5]}")
    true_index = np.array([index[c] for c in true_class_ids])
    positions = np.argmax(ranks == true_index[:, None], axis=1)
    return positions + 1


def average_precision_single_relevant(true_ranks: np.ndarray) -> np.ndarray:
    """AP per query with one relevant item: precision at the single hit, i.e. 1 / rank."""
    true_ranks = np.asarray(true_ranks, dtype=np.float64)
    return 1.0 / true_ranks


def recall_at_k_single_relevant(true_ranks: np.ndarray, k: int) -> float:
    """Recall@k with one relevant item: the item is either retrieved (1) or not (0)."""
    return float((np.asarray(true_ranks) <= k).mean())


def summarise_true_ranks(
    true_ranks: np.ndarray,
    class_count: int,
    ks: Sequence[int] = DEFAULT_KS,
    tied_queries: int | None = None,
) -> RankingMetrics:
    """Top-k, MRR, mean/median/percentile rank and normalised rank from one-based true ranks."""
    true_ranks = np.asarray(true_ranks)
    normalised = (
        (true_ranks - 1) / (class_count - 1)
        if class_count > 1
        else np.zeros_like(true_ranks, dtype=np.float64)
    )
    return RankingMetrics(
        query_count=int(true_ranks.shape[0]),
        class_count=class_count,
        tied_queries=tied_queries,
        top_k={int(k): float((true_ranks <= k).mean()) for k in ks},
        mrr=float((1.0 / true_ranks).mean()),
        mean_rank=float(true_ranks.mean()),
        median_rank=float(np.median(true_ranks)),
        rank_p75=float(np.percentile(true_ranks, 75)),
        rank_p90=float(np.percentile(true_ranks, 90)),
        rank_p95=float(np.percentile(true_ranks, 95)),
        mean_normalised_rank=float(np.mean(normalised)),
    )


def ranking_metrics(
    ranks: np.ndarray,
    true_class_ids: Sequence[str],
    class_ids: Sequence[str],
    ks: Sequence[int] = DEFAULT_KS,
    tied_queries: int | None = None,
) -> RankingMetrics:
    """Top-k, MRR, mean/median/percentile rank and normalised rank from a rank matrix."""
    true_ranks = rank_of_true_class(ranks, true_class_ids, class_ids)
    return summarise_true_ranks(true_ranks, len(class_ids), ks, tied_queries)
