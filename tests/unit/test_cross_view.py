"""Track B cross-view variants (ADR-0004, issue M1-18).

A variant for held-out view ``v`` filters the stored Track B split: support keeps the
other known viewpoints, validation and test keep viewpoint ``v`` only, and no row ever
changes partition. Expected rows in the hand-written cases are read off the fixture by
hand.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from avr.data.cross_view import (
    CrossViewThresholds,
    audit_cross_view,
    build_cross_view_splits,
    cross_view_variant,
)
from avr.data.split_track_b import OUTPUT_SCHEMA


def split_rows(spec: list[tuple[str, str, int, str]]) -> pd.DataFrame:
    """Rows as (image_id, class_id, viewpoint, partition) in the stored seed_<s> schema."""
    frame = pd.DataFrame(spec, columns=["image_id", "class_id", "viewpoint", "partition"])
    frame["sha256"] = frame["image_id"].map(lambda i: hashlib.sha256(i.encode()).hexdigest())
    frame["make_id"] = 1
    frame["model_id"] = frame["class_id"].map(lambda c: int(c.split("/")[1]))
    frame["mapping_status"] = "provisional"
    return frame


def rich_class(class_id: str) -> list[tuple[str, str, int, str]]:
    """Five support images per view 1..5, one validation and one test image per view."""
    rows = []
    for view in range(1, 6):
        rows += [(f"{class_id}_s{view}_{i}", class_id, view, "support") for i in range(5)]
        rows.append((f"{class_id}_v{view}", class_id, view, "validation"))
        rows.append((f"{class_id}_t{view}", class_id, view, "test"))
    return rows


def test_variant_keeps_other_views_in_support_and_the_held_out_view_in_queries():
    split = split_rows(rich_class("1/10/2011"))

    variant = cross_view_variant(split, view=3)

    support = variant[variant["partition"] == "support"]
    assert sorted(support["viewpoint"].unique()) == [1, 2, 4, 5]
    assert len(support) == 20
    for partition in ("validation", "test"):
        rows = variant[variant["partition"] == partition]
        assert rows["image_id"].tolist() == [f"1/10/2011_{partition[0]}3"]
    assert list(variant.columns) == list(split.columns)


def test_uncertain_viewpoint_rows_are_absent_from_every_partition():
    split = split_rows(
        rich_class("1/10/2011")
        + [
            ("u_s", "1/10/2011", -1, "support"),
            ("u_v", "1/10/2011", -1, "validation"),
            ("u_t", "1/10/2011", -1, "test"),
        ]
    )

    for view in range(1, 6):
        variant = cross_view_variant(split, view=view)
        assert -1 not in set(variant["viewpoint"])
        assert not variant["image_id"].str.startswith("u_").any()


def test_class_is_dropped_when_it_keeps_fewer_than_five_support_images():
    # class 1/20/2011: 6 support images in total, 5 of them of view 2. Holding out view 2
    # leaves one support image, so the class is dropped from every partition of that
    # variant; holding out view 1 leaves five, so it is kept.
    narrow = [(f"n_s2_{i}", "1/20/2011", 2, "support") for i in range(5)]
    narrow += [("n_s1", "1/20/2011", 1, "support"), ("n_t1", "1/20/2011", 1, "test")]
    narrow += [("n_t2", "1/20/2011", 2, "test"), ("n_v2", "1/20/2011", 2, "validation")]
    split = split_rows(rich_class("1/10/2011") + narrow)

    held_out_2 = cross_view_variant(split, view=2)
    held_out_1 = cross_view_variant(split, view=1)

    assert set(held_out_2["class_id"]) == {"1/10/2011"}
    assert set(held_out_1["class_id"]) == {"1/10/2011", "1/20/2011"}
    kept = held_out_1[held_out_1["class_id"] == "1/20/2011"]
    assert sorted(kept["image_id"]) == ["n_s2_0", "n_s2_1", "n_s2_2", "n_s2_3", "n_s2_4", "n_t1"]


def test_class_is_dropped_when_it_has_no_test_image_of_the_held_out_view():
    # class 1/30/2011 has plenty of support of every view but test images of view 4 only.
    no_test = [(f"m_s{v}_{i}", "1/30/2011", v, "support") for v in range(1, 6) for i in range(5)]
    no_test += [("m_t4", "1/30/2011", 4, "test"), ("m_v5", "1/30/2011", 5, "validation")]
    split = split_rows(rich_class("1/10/2011") + no_test)

    assert set(cross_view_variant(split, view=4)["class_id"]) == {"1/10/2011", "1/30/2011"}
    # a validation image of view 5 is not a test image: still dropped
    assert set(cross_view_variant(split, view=5)["class_id"]) == {"1/10/2011"}
    assert set(cross_view_variant(split, view=1)["class_id"]) == {"1/10/2011"}


def test_no_image_changes_partition_and_no_image_is_invented():
    split = split_rows(rich_class("1/10/2011") + rich_class("1/20/2012"))
    parent = split.set_index("image_id")["partition"]

    for view in range(1, 6):
        variant = cross_view_variant(split, view=view)
        assert variant["image_id"].is_unique
        assert set(variant["image_id"]) <= set(parent.index)
        expected = parent.loc[variant["image_id"]].tolist()
        assert variant["partition"].tolist() == expected


def test_audit_charges_each_dropped_class_to_the_first_rule_it_fails_and_counts_images():
    narrow = [(f"n_s2_{i}", "1/20/2011", 2, "support") for i in range(5)]
    narrow += [("n_t2", "1/20/2011", 2, "test")]  # held-out 2: 0 support -> too few support
    no_test = [(f"m_s{v}_{i}", "1/30/2011", v, "support") for v in range(1, 6) for i in range(5)]
    no_test += [("m_t4", "1/30/2011", 4, "test")]  # held-out 2: 20 support, 0 test
    uncertain = [("u_s", "1/10/2011", -1, "support"), ("u_t", "1/10/2011", -1, "test")]
    split = split_rows(rich_class("1/10/2011") + narrow + no_test + uncertain)

    audit = audit_cross_view(split, view=2)

    assert (audit.classes_before, audit.classes_after) == (3, 1)
    assert audit.eligible_class_ids == ["1/10/2011"]
    assert audit.dropped_too_few_support == ["1/20/2011"]
    assert audit.dropped_no_test == ["1/30/2011"]
    # rich class: 25 support, 5 validation, 5 test; narrow: 5 + 1; no_test: 25 + 1; -1: 1 + 1
    assert audit.images_before == {"support": 56, "validation": 5, "test": 8}
    assert audit.images_uncertain == {"support": 1, "validation": 0, "test": 1}
    assert audit.images_after == {"support": 20, "validation": 1, "test": 1}


def test_build_writes_the_eligibility_report_then_one_pinned_table_per_view(tmp_path: Path):
    narrow = [(f"n_s2_{i}", "1/20/2011", 2, "support") for i in range(5)]
    narrow += [("n_t2", "1/20/2011", 2, "test"), ("n_t1", "1/20/2011", 1, "test")]
    split = split_rows(rich_class("1/10/2011") + narrow)
    report = tmp_path / "reports" / "seed_7.md"
    out_dir = tmp_path / "splits" / "seed_7"

    result = build_cross_view_splits(
        split,
        seed=7,
        views=[1, 2],
        thresholds=CrossViewThresholds(),
        report_path=report,
        out_dir=out_dir,
    )

    text = report.read_text()
    assert "`1/20/2011`" in text  # dropped for too few support under held-out view 2
    assert "seed 7" in text
    assert sorted(result.audits) == [1, 2]
    assert result.audits[2].dropped_too_few_support == ["1/20/2011"]
    assert result.audits[1].classes_after == 2
    for view in (1, 2):
        out = result.outputs[view]
        assert out.manifest == out_dir / f"view_{view}.parquet"
        digest = hashlib.sha256(out.manifest.read_bytes()).hexdigest()
        assert out.sha256.read_text() == f"{digest}  view_{view}.parquet\n"
        table = pq.read_table(out.manifest)
        assert table.schema.equals(OUTPUT_SCHEMA)
        expected = cross_view_variant(split, view)
        assert table.column("image_id").to_pylist() == expected["image_id"].tolist()
    assert result.class_counts == {1: 2, 2: 1}
    assert result.partition_counts[2] == {"support": 20, "validation": 1, "test": 1}
