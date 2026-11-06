"""Exact cosine search over class prototypes (PROTOCOL.md §12 "Scientific evaluation").

Two backends compute the same ranking: a float32 matrix multiplication and FAISS
``IndexFlatIP``. Both feed their scores through ``_rank_scores`` so the ranking rule is
shared:

* scores are rounded to ``SCORE_DECIMALS`` decimals before sorting. The two backends sum
  the same float32 products in different orders (BLAS blocking versus FAISS SIMD), which
  perturbs a score by a few float32 ulp (about 1e-7). Rounding at 1e-6 absorbs that noise
  so the backends return identical ranks (§23 "FAISS exact and matrix search return
  identical ranks") and so an exact tie is detected identically by both;
* equal rounded scores are ordered by lexicographic class id (ADR-0002), through a stable
  sort over prototypes pre-sorted by class id, so the rule does not depend on the order in
  which prototypes were passed;
* the number of tied queries is reported. A query counts as tied at ``k`` when two
  adjacent ranked scores among positions ``1..k+1`` are equal, i.e. when the tie changes
  or makes ambiguous the composition of its top-k list.

Inputs are expected to be L2-normalised, so inner product equals cosine similarity.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import faiss
import numpy as np

SCORE_DECIMALS = 6


@dataclass(frozen=True)
class Ranking:
    """Full ranking of every prototype for every query.

    ``ranks[q, r]`` is the index (into the prototypes passed in) of the prototype at rank
    ``r + 1`` for query ``q``; ``scores[q, r]`` is its rounded cosine score.
    """

    ranks: np.ndarray
    scores: np.ndarray
    class_ids: list[str]

    @property
    def tied_queries(self) -> int:
        """Queries with an exact tie anywhere in their ranking."""
        return self.tied_queries_at(self.ranks.shape[1])

    def tied_queries_at(self, k: int) -> int:
        """Queries whose top-k list is affected by an exact tie (see module docstring)."""
        last = min(k, self.scores.shape[1] - 1)
        if last < 1:
            return 0
        adjacent_equal = self.scores[:, :last] == self.scores[:, 1 : last + 1]
        return int(adjacent_equal.any(axis=1).sum())


def _check(queries: np.ndarray, prototypes: np.ndarray, class_ids: Sequence[str]) -> None:
    if queries.ndim != 2 or prototypes.ndim != 2:
        raise ValueError("queries and prototypes must be 2-d arrays")
    if queries.shape[1] != prototypes.shape[1]:
        raise ValueError(
            f"query dimension {queries.shape[1]} != prototype dimension {prototypes.shape[1]}"
        )
    if len(class_ids) != prototypes.shape[0]:
        raise ValueError(f"{prototypes.shape[0]} prototypes but {len(class_ids)} class ids")
    if len(set(class_ids)) != len(class_ids):
        raise ValueError("class ids must be unique")


def _rank_scores(scores: np.ndarray, class_ids: Sequence[str]) -> Ranking:
    """Shared ranking rule: round, then stable-sort by descending score within class-id order."""
    rounded = np.round(scores.astype(np.float32), SCORE_DECIMALS)
    by_class_id = np.argsort(np.asarray(list(class_ids), dtype=object), kind="stable")
    order_within = np.argsort(-rounded[:, by_class_id], axis=1, kind="stable")
    ranks = by_class_id[order_within]
    return Ranking(
        ranks=ranks,
        scores=np.take_along_axis(rounded, ranks, axis=1),
        class_ids=list(class_ids),
    )


def rank_by_cosine(
    queries: np.ndarray, prototypes: np.ndarray, class_ids: Sequence[str]
) -> Ranking:
    """Rank every prototype for every query by float32 matrix multiplication."""
    queries = np.ascontiguousarray(queries, dtype=np.float32)
    prototypes = np.ascontiguousarray(prototypes, dtype=np.float32)
    _check(queries, prototypes, class_ids)
    return _rank_scores(queries @ prototypes.T, class_ids)


def rank_by_cosine_faiss(
    queries: np.ndarray, prototypes: np.ndarray, class_ids: Sequence[str]
) -> Ranking:
    """Rank every prototype for every query through FAISS ``IndexFlatIP`` (exact search)."""
    queries = np.ascontiguousarray(queries, dtype=np.float32)
    prototypes = np.ascontiguousarray(prototypes, dtype=np.float32)
    _check(queries, prototypes, class_ids)
    index = faiss.IndexFlatIP(prototypes.shape[1])
    index.add(prototypes)
    faiss_scores, faiss_ids = index.search(queries, prototypes.shape[0])
    # Rebuild the full [q, C] score matrix in prototype order so the shared rule (rounding,
    # class-id ties) decides the ranking rather than FAISS's own tie order.
    scores = np.empty_like(faiss_scores)
    np.put_along_axis(scores, faiss_ids, faiss_scores, axis=1)
    return _rank_scores(scores, class_ids)
