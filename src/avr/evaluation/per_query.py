"""Per-query rank table, per-class summary and CMC curve (issue M2-6; PROTOCOL.md §13.2).

:func:`build_per_query` ranks every prototype for every query with the exact search of
:mod:`avr.retrieval.exact` (same rounding, same class-id tie rule) and keeps, for each query
image, where its correct class landed: ``rank_true`` (1-based), the score of the true class,
the top-1 prediction, the margin between the first two scores, the top-10 classes and scores,
hit flags at k = 1, 5, 10, 20, an exact-tie flag, and the first rank at which a prototype of
the right make (``rank_make``) or the right make and model (``rank_model``) appears.

Tie flag: ``tied`` is true when two adjacent ranked scores among positions 1..11 are equal
(the rule of ``Ranking.tied_queries_at(10)``), i.e. when an exact tie changes or makes
ambiguous the composition of the top-10 list.

Class id conventions (shared with the gate): Track A ``"<make:03d>_<model:04d>"`` is already a
make-model class, so its model-level key is the class id itself; Track B
``"<make_id>/<model_id>/<year>"`` is a provisional model-year class.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from avr.retrieval.exact import rank_by_cosine, rank_by_cosine_faiss

HIT_KS = (1, 5, 10, 20)
TOP_LIST = 10
CMC_MAX_K = 50
DEFAULT_CHUNK = 4096

META_COLUMNS = ("image_id", "true_class_id", "make_id", "model_id", "year", "viewpoint")


def hierarchy_keys(track: str, class_id: str) -> tuple[str, str]:
    """``(make key, make-model key)`` of a fine class id."""
    if track == "track_a":
        return class_id.split("_")[0], class_id
    parts = class_id.split("/")
    return parts[0], "/".join(parts[:2])


def _first_rank(ranks: np.ndarray, key_of_prototype: np.ndarray, wanted: np.ndarray) -> np.ndarray:
    """1-based first position at which a prototype with the wanted key appears."""
    return np.argmax(key_of_prototype[ranks] == wanted[:, None], axis=1) + 1


def _codes(values: Sequence[str], universe: list[str]) -> np.ndarray:
    index = {v: i for i, v in enumerate(universe)}
    return np.array([index[v] for v in values], dtype=np.int64)


def build_per_query(
    queries: np.ndarray,
    prototypes: np.ndarray,
    class_ids: list[str],
    meta: pd.DataFrame,
    *,
    track: str,
    encoder: str,
    variant: str,
    partition: str,
    split_label: str,
    mapping_status: str,
    chunk_size: int = DEFAULT_CHUNK,
    check_faiss: bool = False,
) -> pd.DataFrame:
    """One row per query, in the row order of ``queries``/``meta`` (see module docstring)."""
    if len(meta) != queries.shape[0]:
        raise ValueError(f"{queries.shape[0]} query embeddings but {len(meta)} metadata rows")
    missing = sorted(set(meta.true_class_id) - set(class_ids))
    if missing:
        raise ValueError(f"{len(missing)} true class(es) have no prototype: {missing[:5]}")
    n_classes = len(class_ids)
    k_list = min(TOP_LIST, n_classes)
    ids_array = np.asarray(class_ids, dtype=object)
    keys = [hierarchy_keys(track, c) for c in class_ids]
    makes = sorted({m for m, _ in keys})
    models = sorted({mm for _, mm in keys})
    proto_make = _codes([m for m, _ in keys], makes)
    proto_model = _codes([mm for _, mm in keys], models)
    true_index = _codes(list(meta.true_class_id), class_ids)
    true_make = proto_make[true_index]
    true_model = proto_model[true_index]

    parts: list[dict[str, np.ndarray]] = []
    for start in range(0, queries.shape[0], chunk_size):
        stop = min(start + chunk_size, queries.shape[0])
        ranking = rank_by_cosine(queries[start:stop], prototypes, class_ids)
        if check_faiss:
            faiss_ranking = rank_by_cosine_faiss(queries[start:stop], prototypes, class_ids)
            if not np.array_equal(ranking.ranks, faiss_ranking.ranks):
                raise RuntimeError("FAISS exact search and matrix search disagree on the ranking")
        ranks, scores = ranking.ranks, ranking.scores
        rows = np.arange(stop - start)
        position = np.argmax(ranks == true_index[start:stop, None], axis=1)
        last = min(TOP_LIST, n_classes - 1)
        tied = (
            (scores[:, :last] == scores[:, 1 : last + 1]).any(axis=1)
            if last >= 1
            else np.zeros(stop - start, dtype=bool)
        )
        second = scores[:, 1] if n_classes > 1 else np.zeros(stop - start, dtype=np.float32)
        parts.append(
            {
                "rank_true": position + 1,
                "score_true": scores[rows, position],
                "top1_class_id": ids_array[ranks[:, 0]],
                "top1_score": scores[:, 0],
                "margin_top1_top2": scores[:, 0] - second,
                "top10_class_ids": ids_array[ranks[:, :k_list]],
                "top10_scores": scores[:, :k_list],
                "tied": tied,
                "rank_make": _first_rank(ranks, proto_make, true_make[start:stop]),
                "rank_model": _first_rank(ranks, proto_model, true_model[start:stop]),
            }
        )

    def joined(name: str) -> np.ndarray:
        return np.concatenate([p[name] for p in parts]) if parts else np.empty(0)

    rank_true = joined("rank_true").astype(np.int32)
    table = pd.DataFrame(
        {
            "image_id": meta.image_id.to_numpy(),
            "track": track,
            "encoder": encoder,
            "variant": variant,
            "partition": partition,
            "split_label": split_label,
            "mapping_status": mapping_status,
            "true_class_id": meta.true_class_id.to_numpy(),
            "make_id": meta.make_id.astype("int32").to_numpy(),
            "model_id": meta.model_id.astype("int32").to_numpy(),
            "year": meta.year.astype("Int16").to_numpy(),
            "viewpoint": meta.viewpoint.astype("int8").to_numpy(),
            "rank_true": rank_true,
            "score_true": joined("score_true").astype(np.float32),
            "top1_class_id": joined("top1_class_id"),
            "top1_score": joined("top1_score").astype(np.float32),
            "margin_top1_top2": joined("margin_top1_top2").astype(np.float32),
            "top10_class_ids": list(joined("top10_class_ids")),
            "top10_scores": list(joined("top10_scores").astype(np.float32)),
            **{f"hit_at_{k}": rank_true <= k for k in HIT_KS},
            "tied": joined("tied").astype(bool),
            "rank_make": joined("rank_make").astype(np.int32),
            "rank_model": joined("rank_model").astype(np.int32),
        }
    )
    return table


def per_class_summary(per_query: pd.DataFrame, class_names: pd.DataFrame) -> pd.DataFrame:
    """One row per true class: hits, hit rates, rank statistics and the commonest wrong top-1.

    ``class_names`` has ``class_id``, ``make_name``, ``model_name`` and ``year``. The most
    frequent wrong top-1 class breaks ties by the smaller class id; a class never mistaken has
    ``top_wrong_class_id`` null and ``top_wrong_count`` 0.
    """
    rows = []
    wrong = per_query[per_query.top1_class_id != per_query.true_class_id]
    for class_id, group in per_query.groupby("true_class_id", sort=True):
        ranks = group.rank_true.to_numpy()
        mistaken = wrong[wrong.true_class_id == class_id].top1_class_id.value_counts()
        if len(mistaken):
            top = mistaken[mistaken == mistaken.max()].sort_index().index[0]
            top_count = int(mistaken[top])
        else:
            top, top_count = None, 0
        row = {"class_id": class_id, "queries": len(group)}
        for k in HIT_KS:
            row[f"hits_at_{k}"] = int((ranks <= k).sum())
        for k in HIT_KS:
            row[f"hit_rate_at_{k}"] = float((ranks <= k).mean())
        row.update(
            mean_rank=float(ranks.mean()),
            median_rank=float(np.median(ranks)),
            mrr=float((1.0 / ranks).mean()),
            top_wrong_class_id=top,
            top_wrong_count=top_count,
        )
        rows.append(row)
    summary = pd.DataFrame(rows)
    names = class_names[["class_id", "make_name", "model_name", "year"]]
    summary = names.merge(summary, on="class_id", how="inner", validate="one_to_one")
    return summary.sort_values("class_id").reset_index(drop=True)


def cmc_table(per_query: pd.DataFrame, max_k: int = CMC_MAX_K) -> pd.DataFrame:
    """Cumulative hit rate for k = 1..max_k of one track, encoder and variant."""
    for column in ("track", "encoder", "variant", "partition"):
        if per_query[column].nunique() != 1:
            raise ValueError(f"cmc_table expects one {column} per table")
    ranks = per_query.rank_true.to_numpy()
    ks = np.arange(1, max_k + 1)
    return pd.DataFrame(
        {
            "track": per_query.track.iloc[0],
            "encoder": per_query.encoder.iloc[0],
            "variant": per_query.variant.iloc[0],
            "partition": per_query.partition.iloc[0],
            "k": ks,
            "queries": len(ranks),
            "hit_rate": [(ranks <= k).mean() for k in ks],
        }
    )
