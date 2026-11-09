"""Per-query rank table (issue M2-6; PROTOCOL.md §13.2). Expected values are hand-computed.

Toy problem, four prototypes that are the unit axes of a 4-d space, in class-id order:

    track_b  A = "1/1/2010" (e0)   B = "1/1/2011" (e1)   C = "1/2/2012" (e2)   D = "2/1/2013" (e3)

Queries and their cosine scores (query vectors are normalised, so the score of a prototype is
its coordinate divided by the query norm):

    q1 = (.8, .6, 0, 0), true B : A .8, B .6, C 0, D 0   -> order A B C D (C/D tie at 0 -> class id)
    q2 = (0, 0, .6, .8), true C : D .8, C .6, A 0, B 0   -> order D C A B
    q3 = (.1, .2, .3, .4), true A : norm sqrt(.3) = .547723; D .730297, C .547723, B .365148,
         A .182574                                       -> order D C B A
    q4 = (.5, .5, .5, .5), true D : every score .5       -> order A B C D (all tied)
"""

import numpy as np
import pandas as pd
import pytest

from avr.evaluation.per_query import (
    CMC_MAX_K,
    build_per_query,
    cmc_table,
    hierarchy_keys,
    per_class_summary,
)

CLASSES_B = ["1/1/2010", "1/1/2011", "1/2/2012", "2/1/2013"]
PROTOTYPES = np.eye(4, dtype=np.float32)
QUERIES = np.array(
    [[0.8, 0.6, 0, 0], [0, 0, 0.6, 0.8], [0.1, 0.2, 0.3, 0.4], [0.5, 0.5, 0.5, 0.5]],
    dtype=np.float32,
)
QUERIES = QUERIES / np.linalg.norm(QUERIES, axis=1, keepdims=True)


def meta_b() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "image_id": ["q1", "q2", "q3", "q4"],
            "true_class_id": ["1/1/2011", "1/2/2012", "1/1/2010", "2/1/2013"],
            "make_id": pd.array([1, 1, 1, 2], dtype="int32"),
            "model_id": pd.array([1, 2, 1, 1], dtype="int32"),
            "year": pd.array([2011, 2012, 2010, 2013], dtype="Int16"),
            "viewpoint": pd.array([1, 2, 3, 4], dtype="int8"),
        }
    )


def build_b() -> pd.DataFrame:
    return build_per_query(
        QUERIES,
        PROTOTYPES,
        CLASSES_B,
        meta_b(),
        track="track_b",
        encoder="toy",
        variant="p1",
        partition="test",
        split_label="toy_split",
        mapping_status="provisional",
    )


def test_hierarchy_keys_track_b_and_a():
    assert hierarchy_keys("track_b", "152/407/2012") == ("152", "152/407")
    assert hierarchy_keys("track_a", "118_0976") == ("118", "118_0976")


def test_ranks_scores_and_hits_match_hand_calculation():
    table = build_b()
    assert list(table.image_id) == ["q1", "q2", "q3", "q4"]
    assert list(table.rank_true) == [2, 2, 4, 4]
    assert list(table.rank_make) == [1, 2, 2, 4]
    assert list(table.rank_model) == [1, 2, 3, 4]
    assert list(table.top1_class_id) == ["1/1/2010", "2/1/2013", "2/1/2013", "1/1/2010"]
    np.testing.assert_allclose(table.score_true, [0.6, 0.6, 0.182574, 0.5], atol=1e-6)
    np.testing.assert_allclose(table.top1_score, [0.8, 0.8, 0.730297, 0.5], atol=1e-6)
    np.testing.assert_allclose(table.margin_top1_top2, [0.2, 0.2, 0.182574, 0.0], atol=1e-6)
    assert list(table.hit_at_1) == [False] * 4
    assert list(table.hit_at_5) == [True] * 4  # only four classes, so k >= 4 always hits
    assert list(table.hit_at_10) == [True] * 4
    assert list(table.hit_at_20) == [True] * 4
    assert list(table.tied) == [True, True, False, True]
    assert list(table.top10_class_ids[0]) == CLASSES_B  # C/D tie broken by class id
    assert list(table.top10_class_ids[1]) == ["2/1/2013", "1/2/2012", "1/1/2010", "1/1/2011"]
    np.testing.assert_allclose(table.top10_scores[2], [0.730297, 0.547723, 0.365148, 0.182574])


def test_identity_columns_and_labels():
    table = build_b()
    assert set(table.track) == {"track_b"}
    assert set(table.encoder) == {"toy"}
    assert set(table.variant) == {"p1"}
    assert set(table.partition) == {"test"}
    assert set(table.mapping_status) == {"provisional"}
    assert set(table.split_label) == {"toy_split"}
    assert list(table.true_class_id) == ["1/1/2011", "1/2/2012", "1/1/2010", "2/1/2013"]
    assert list(table.year) == [2011, 2012, 2010, 2013]


def test_chunking_does_not_change_the_table():
    whole = build_b()
    chunked = build_per_query(
        QUERIES,
        PROTOTYPES,
        CLASSES_B,
        meta_b(),
        track="track_b",
        encoder="toy",
        variant="p1",
        partition="test",
        split_label="toy_split",
        mapping_status="provisional",
        chunk_size=1,
    )
    pd.testing.assert_frame_equal(whole, chunked)


def test_track_a_class_is_make_model_so_model_rank_equals_class_rank():
    classes = ["001_0001", "001_0002", "002_0001"]
    meta = pd.DataFrame(
        {
            "image_id": ["a"],
            "true_class_id": ["001_0002"],
            "make_id": pd.array([1], dtype="int32"),
            "model_id": pd.array([2], dtype="int32"),
            "year": pd.array([pd.NA], dtype="Int16"),
            "viewpoint": pd.array([1], dtype="int8"),
        }
    )
    query = np.array([[0.0, 0.6, 0.8]], dtype=np.float32)  # order 002_0001, 001_0002, 001_0001
    table = build_per_query(
        query,
        np.eye(3, dtype=np.float32),
        classes,
        meta,
        track="track_a",
        encoder="toy",
        variant="p1",
        partition="query",
        split_label="official",
        mapping_status="not_applicable",
    )
    assert table.rank_true[0] == table.rank_model[0] == 2
    assert table.rank_make[0] == 2  # 002_0001 is first, the first make-1 prototype is second
    assert pd.isna(table.year[0])


def test_unknown_true_class_is_rejected():
    meta = meta_b()
    meta.loc[0, "true_class_id"] = "9/9/9999"
    with pytest.raises(ValueError, match="no prototype"):
        build_per_query(
            QUERIES,
            PROTOTYPES,
            CLASSES_B,
            meta,
            track="track_b",
            encoder="toy",
            variant="p1",
            partition="test",
            split_label="s",
            mapping_status="provisional",
        )


def test_per_class_summary_hand_calculated():
    table = build_b()
    # two more queries of class B (true), ranks 1 and 3, to give class B three queries
    extra = table.iloc[[0, 0]].copy()
    extra["true_class_id"] = "1/1/2011"
    extra["rank_true"] = [1, 3]
    extra["hit_at_1"] = [True, False]
    extra["top1_class_id"] = ["1/1/2011", "2/1/2013"]
    table = pd.concat([table, extra], ignore_index=True)
    names = pd.DataFrame(
        {
            "class_id": CLASSES_B,
            "make_name": ["M1", "M1", "M1", "M2"],
            "model_name": ["X", "X", "Y", "Z"],
            "year": pd.array([2010, 2011, 2012, 2013], dtype="Int16"),
        }
    )
    summary = per_class_summary(table, names).set_index("class_id")
    b = summary.loc["1/1/2011"]
    # B queries: q1 (rank 2, top1 A), + ranks 1 (top1 B) and 3 (top1 D)
    assert b.queries == 3
    assert (b.hits_at_1, b.hits_at_5, b.hits_at_10, b.hits_at_20) == (1, 3, 3, 3)
    assert b.hit_rate_at_1 == pytest.approx(1 / 3)
    assert b.mean_rank == pytest.approx(2.0)
    assert b.median_rank == pytest.approx(2.0)
    assert b.mrr == pytest.approx((1 / 2 + 1 + 1 / 3) / 3)
    # wrong top-1 classes: A once, D once -> tie broken by smaller class id
    assert (b.top_wrong_class_id, b.top_wrong_count) == ("1/1/2010", 1)
    assert (b.make_name, b.model_name) == ("M1", "X")
    d = summary.loc["2/1/2013"]
    assert d.queries == 1 and d.top_wrong_class_id == "1/1/2010" and d.top_wrong_count == 1
    assert "1/2/2012" in summary.index and summary.loc["1/2/2012"].queries == 1


def test_per_class_summary_class_without_wrong_top1():
    table = build_b().iloc[[0]].copy()
    table["top1_class_id"] = table["true_class_id"]
    table["hit_at_1"] = True
    table["rank_true"] = 1
    names = pd.DataFrame(
        {
            "class_id": CLASSES_B,
            "make_name": "M",
            "model_name": "X",
            "year": pd.array([1, 2, 3, 4], dtype="Int16"),
        }
    )
    row = per_class_summary(table, names).iloc[0]
    assert pd.isna(row.top_wrong_class_id) and row.top_wrong_count == 0


def test_cmc_table_hand_calculated():
    table = build_b()  # true ranks 2, 2, 4, 4
    cmc = cmc_table(table)
    assert list(cmc.k) == list(range(1, CMC_MAX_K + 1))
    rates = dict(zip(cmc.k, cmc.hit_rate, strict=True))
    assert rates[1] == 0.0
    assert rates[2] == 0.5
    assert rates[3] == 0.5
    assert rates[4] == 1.0
    assert rates[50] == 1.0
    assert set(cmc.queries) == {4}
    assert set(cmc.track) == {"track_b"} and set(cmc.variant) == {"p1"}


def test_aggregates_from_table_equal_ranking_metrics_on_the_toy_problem():
    # the gate's metric code, fed the same toy ranking, must agree with the per-query table
    from avr.evaluation.ranking import ranking_metrics
    from avr.retrieval.exact import rank_by_cosine

    table = build_b()
    ranking = rank_by_cosine(QUERIES, PROTOTYPES, CLASSES_B)
    gate = ranking_metrics(ranking.ranks, list(table.true_class_id), CLASSES_B)
    assert gate.mean_rank == table.rank_true.mean() == 3.0  # (2 + 2 + 4 + 4) / 4
    assert gate.mrr == (1.0 / table.rank_true).mean() == 0.375  # (.5 + .5 + .25 + .25) / 4
    assert gate.top_k[1] == table.hit_at_1.mean() == 0.0
    assert gate.top_k[5] == table.hit_at_5.mean() == 1.0
