"""Track A split manifest (PROTOCOL.md §5 Track A, issue M1-9).

The synthetic manifest is small enough that every expected count and id below is read
off the fixture by hand.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from avr.data.split_track_a import (
    OfficialSplit,
    join_official_split,
    partition_validation,
    read_official_split,
)

# image_id, class (make, model), duplicate group, validity. Groups: g1 = {s1, q1} spans
# the official support/query boundary; g2 = {s2, s3} stays inside support.
MANIFEST_ROWS = [
    ("s1", 1, 10, "s1", "valid"),
    ("s2", 1, 10, "s2", "valid"),
    ("s3", 1, 10, "s2", "valid"),
    ("s4", 2, 20, "s4", "excluded"),
    ("q1", 1, 10, "s1", "valid"),
    ("q2", 2, 20, "q2", "valid"),
    ("u1", 2, 20, "u1", "valid"),
]


@pytest.fixture
def manifest() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "image_id": i,
                "sha256": f"sha-{i}",
                "make_id": make,
                "model_id": model,
                "duplicate_group_id": group,
                "validity_status": validity,
                # the manifest reserves the column and leaves it null (PROTOCOL.md §7)
                "split_track_a": None,
            }
            for i, make, model, group, validity in MANIFEST_ROWS
        ]
    )


@pytest.fixture
def paths() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "image_id": [r[0] for r in MANIFEST_ROWS],
            "relative_path": [f"image/{r[1]}/{r[2]}/2010/{r[0]}.jpg" for r in MANIFEST_ROWS],
        }
    )


@pytest.fixture
def official() -> OfficialSplit:
    return OfficialSplit(
        support=["1/10/2010/s1.jpg", "1/10/2010/s2.jpg", "1/10/2010/s3.jpg", "2/20/2010/s4.jpg"],
        query=["1/10/2010/q1.jpg", "2/20/2010/q2.jpg", "2/20/2010/gone.jpg"],
    )


def test_read_official_split_reads_train_as_support_and_test_as_query(tmp_path: Path):
    folder = tmp_path / "train_test_split" / "classification"
    folder.mkdir(parents=True)
    (folder / "train.txt").write_text("78/1/2010/a.jpg\n78/1/2010/b.jpg\n")
    (folder / "test.txt").write_text("37/1993/2014/c.jpg\n")

    split = read_official_split(tmp_path)

    assert split.support == ["78/1/2010/a.jpg", "78/1/2010/b.jpg"]
    assert split.query == ["37/1993/2014/c.jpg"]


def test_join_labels_exactly_the_listed_images(manifest, paths, official):
    result = join_official_split(manifest, paths, official)

    labels = dict(zip(result.rows.image_id, result.rows.split_track_a, strict=True))
    assert labels == {
        "s1": "support",
        "s2": "support",
        "s3": "support",
        "s4": "support",
        "q1": "query",
        "q2": "query",
    }
    assert list(result.rows.image_id) == sorted(result.rows.image_id)
    assert list(result.rows.columns) == [
        "image_id",
        "sha256",
        "make_id",
        "model_id",
        "duplicate_group_id",
        "validity_status",
        "split_track_a",
    ]


def test_join_reports_missing_unlisted_and_cross_split_groups(manifest, paths, official):
    result = join_official_split(manifest, paths, official)

    assert result.missing == {"support": [], "query": ["2/20/2010/gone.jpg"]}
    assert result.unlisted_image_ids == ["u1"]
    assert result.cross_split_group_ids == ["s1"]


def test_join_rejects_a_path_listed_on_both_sides(manifest, paths):
    both = OfficialSplit(support=["1/10/2010/s1.jpg"], query=["1/10/2010/s1.jpg"])

    with pytest.raises(ValueError, match="both"):
        join_official_split(manifest, paths, both)


def support_rows(spec: list[tuple[str, int, int, str, str]]) -> pd.DataFrame:
    """Support-side rows from (image_id, make_id, model_id, duplicate_group_id, validity)."""
    return pd.DataFrame(
        [
            {
                "image_id": i,
                "sha256": f"sha-{i}",
                "make_id": make,
                "model_id": model,
                "duplicate_group_id": group,
                "validity_status": validity,
                "split_track_a": "support",
            }
            for i, make, model, group, validity in spec
        ]
    )


def test_partition_leaves_query_and_excluded_images_unpartitioned(manifest, paths, official):
    rows = join_official_split(manifest, paths, official).rows

    labels = partition_validation(rows, fraction=0.2, seed=73)

    by_id = dict(zip(rows.image_id, labels, strict=True))
    assert by_id["q1"] is None and by_id["q2"] is None
    assert by_id["s4"] is None  # excluded images are not fit-eligible
    assert {by_id["s1"], by_id["s2"], by_id["s3"]} <= {"fit", "validation"}


def test_partition_takes_about_the_fraction_per_class_and_at_least_one_when_it_can():
    # class (1, 10): ten singleton groups -> 2 in validation at 20 %
    # class (2, 20): two groups -> rounding would give 0, the floor of one applies
    # class (3, 30): one group -> nothing can be held out
    spec = [(f"a{i}", 1, 10, f"a{i}", "valid") for i in range(10)]
    spec += [("b1", 2, 20, "b1", "valid"), ("b2", 2, 20, "b2", "valid")]
    spec += [("c1", 3, 30, "c1", "valid"), ("c2", 3, 30, "c1", "valid")]
    rows = support_rows(spec)

    labels = partition_validation(rows, fraction=0.2, seed=73)

    rows = rows.assign(track_a_validation=labels)
    counts = rows.groupby("model_id").track_a_validation.apply(lambda s: (s == "validation").sum())
    assert counts.to_dict() == {10: 2, 20: 1, 30: 0}
    assert list(rows.track_a_validation[rows.model_id == 30]) == ["fit", "fit"]


def test_partition_never_splits_a_duplicate_group_even_across_classes():
    # g spans classes (1, 10) and (2, 20); whichever class decides g's label, every
    # member of g, including b1 on the other side, must carry it.
    spec = [("a1", 1, 10, "g", "valid"), ("a2", 1, 10, "g", "valid"), ("a3", 1, 10, "a3", "valid")]
    spec += [("b1", 2, 20, "g", "valid"), ("b2", 2, 20, "b2", "valid")]
    rows = support_rows(spec)

    for seed in (1, 2, 3, 73):
        labels = partition_validation(rows, fraction=0.5, seed=seed)

        by_id = dict(zip(rows.image_id, labels, strict=True))
        assert by_id["a1"] == by_id["a2"] == by_id["b1"], seed
        assert {by_id["a3"], by_id["a1"]} == {"fit", "validation"}, seed


def test_partition_always_keeps_one_fit_unit_per_class():
    # even at fraction 1.0 a class keeps a unit to build its prototype from
    spec = [(f"a{i}", 1, 10, f"a{i}", "valid") for i in (1, 2, 3)]
    rows = support_rows(spec)

    labels = partition_validation(rows, fraction=1.0, seed=73)

    assert sorted(labels) == ["fit", "validation", "validation"]


def test_partition_is_deterministic_for_a_seed_and_moves_with_it():
    spec = [(f"a{i:02d}", 1, 10, f"a{i:02d}", "valid") for i in range(40)]
    rows = support_rows(spec)

    first = partition_validation(rows, fraction=0.2, seed=73)
    again = partition_validation(rows, fraction=0.2, seed=73)
    other = partition_validation(rows, fraction=0.2, seed=13)

    assert list(first) == list(again)
    assert (first == "validation").sum() == 8
    assert list(first) != list(other)
