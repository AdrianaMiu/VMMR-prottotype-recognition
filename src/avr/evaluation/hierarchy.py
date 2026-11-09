"""Hierarchy-level metrics derived from a class ranking (PROTOCOL.md §13.1, §13.2).

A ranking over fine classes (make-model, or make-model-generation) induces a ranking over
every coarser level: the rank of a parent (a make, a model) is the position of the first
prototype that belongs to it. :func:`parent_ranking_metrics` reports the §13.2 summary at
that level, with ``class_count`` the number of distinct parents. :func:`error_decomposition`
is the §13.1 hierarchical exact accuracy, the conditional generation accuracy given a
correct model and the wrong-generation/same-model, wrong-model/same-make, wrong-make split
of the top-1 errors.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from avr.evaluation.ranking import DEFAULT_KS, RankingMetrics, summarise_true_ranks


def rank_of_true_parent(
    ranks: np.ndarray,
    true_parents: Sequence[str],
    class_ids: Sequence[str],
    parent_of: Mapping[str, str],
) -> np.ndarray:
    """One-based rank of the first prototype whose parent is the query's true parent."""
    ranks = np.asarray(ranks)
    if ranks.ndim != 2 or ranks.shape[1] != len(class_ids):
        raise ValueError(f"ranks must be [q, C={len(class_ids)}], got shape {ranks.shape}")
    if len(true_parents) != ranks.shape[0]:
        raise ValueError(f"{ranks.shape[0]} rank rows but {len(true_parents)} true parents")
    parents = sorted({parent_of[c] for c in class_ids})
    index = {p: i for i, p in enumerate(parents)}
    unknown = sorted({p for p in true_parents if p not in index})
    if unknown:
        raise ValueError(f"true parents without any ranked class: {unknown[:5]}")
    parent_index_of_class = np.array([index[parent_of[c]] for c in class_ids])
    ranked_parents = parent_index_of_class[ranks]
    true_index = np.array([index[p] for p in true_parents])
    return np.argmax(ranked_parents == true_index[:, None], axis=1) + 1


def parent_ranking_metrics(
    ranks: np.ndarray,
    true_parents: Sequence[str],
    class_ids: Sequence[str],
    parent_of: Mapping[str, str],
    ks: Sequence[int] = DEFAULT_KS,
    tied_queries: int | None = None,
) -> RankingMetrics:
    true_ranks = rank_of_true_parent(ranks, true_parents, class_ids, parent_of)
    parent_count = len({parent_of[c] for c in class_ids})
    return summarise_true_ranks(true_ranks, parent_count, ks, tied_queries)


@dataclass(frozen=True)
class ErrorDecomposition:
    query_count: int
    exact_correct: int
    model_correct: int
    make_correct: int
    wrong_generation_same_model: int
    wrong_model_same_make: int
    wrong_make: int

    @property
    def exact_accuracy(self) -> float:
        return self.exact_correct / self.query_count if self.query_count else 0.0

    @property
    def model_accuracy(self) -> float:
        return self.model_correct / self.query_count if self.query_count else 0.0

    @property
    def make_accuracy(self) -> float:
        return self.make_correct / self.query_count if self.query_count else 0.0

    @property
    def generation_accuracy_given_model(self) -> float | None:
        return self.exact_correct / self.model_correct if self.model_correct else None

    def as_dict(self) -> dict[str, float | int | None]:
        n = self.query_count or 1
        return {
            "query_count": self.query_count,
            "exact_correct": self.exact_correct,
            "exact_accuracy": self.exact_accuracy,
            "model_accuracy": self.model_accuracy,
            "make_accuracy": self.make_accuracy,
            "generation_accuracy_given_model": self.generation_accuracy_given_model,
            "wrong_generation_same_model": self.wrong_generation_same_model,
            "wrong_generation_same_model_fraction": self.wrong_generation_same_model / n,
            "wrong_model_same_make": self.wrong_model_same_make,
            "wrong_model_same_make_fraction": self.wrong_model_same_make / n,
            "wrong_make": self.wrong_make,
            "wrong_make_fraction": self.wrong_make / n,
        }


def error_decomposition(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    levels: Callable[[str], tuple[str, str]],
) -> ErrorDecomposition:
    """Split top-1 outcomes by hierarchy level; ``levels`` maps a class to (make, model).

    When the fine class is the model itself (Track A) every error is a model or make error
    and ``wrong_generation_same_model`` is zero by construction.
    """
    if len(y_true) != len(y_pred):
        raise ValueError(f"{len(y_true)} true labels but {len(y_pred)} predictions")
    exact = model = make = gen_err = model_err = make_err = 0
    for t, p in zip(y_true, y_pred, strict=True):
        t_make, t_model = levels(t)
        p_make, p_model = levels(p)
        if t == p:
            exact += 1
            model += 1
            make += 1
        elif t_model == p_model:
            model += 1
            make += 1
            gen_err += 1
        elif t_make == p_make:
            make += 1
            model_err += 1
        else:
            make_err += 1
    return ErrorDecomposition(
        query_count=len(y_true),
        exact_correct=exact,
        model_correct=model,
        make_correct=make,
        wrong_generation_same_model=gen_err,
        wrong_model_same_make=model_err,
        wrong_make=make_err,
    )
