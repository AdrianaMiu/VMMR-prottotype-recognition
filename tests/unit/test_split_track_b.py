"""Track B eligibility and group-aware split (PROTOCOL.md §5 Track B, issue M1-10).

Expected counts in the hand-written cases are read off the fixture tables by hand; the
random fixtures only state invariants the protocol demands (no group crosses a partition,
every class in the test partition has support).
"""

from __future__ import annotations

import hashlib
import random

import pandas as pd

from avr.data.split_track_b import (
    ClassStats,
    EligibilityThresholds,
    assign_partitions,
    audit_eligibility,
    class_stats_from_manifest,
    with_provisional_classes,
)

THRESHOLDS = EligibilityThresholds(
    min_images=20,
    min_viewpoints=2,
    min_query_eligible=5,
    require_approved_mapping=True,
    provisional_fallback=True,
)


def stats(class_id: str, **overrides) -> ClassStats:
    base = dict(
        class_id=class_id,
        make_id=1,
        model_id=1,
        generation_id=class_id,
        mapping_status="provisional",
        image_count=30,
        deduplicated_count=30,
        viewpoint_count=3,
        query_eligible_count=30,
        has_unresolved_overlap=False,
    )
    base.update(overrides)
    return ClassStats(**base)


def test_audit_counts_classes_before_and_after_every_rule_in_protocol_order():
    classes = [
        stats("ok"),
        stats("few_unique", deduplicated_count=19),  # 30 valid images, 19 after dedup
        stats("one_view", viewpoint_count=1),
        stats("no_queries", query_eligible_count=4, image_count=25, deduplicated_count=25),
        stats("ambiguous", mapping_status="ambiguous"),
        stats("overlap", has_unresolved_overlap=True),
        stats("two_rules", deduplicated_count=5, viewpoint_count=1, image_count=5),
    ]

    audit = audit_eligibility(classes, THRESHOLDS)

    assert audit.classes_before == 7
    assert audit.images_before == 30 * 5 + 25 + 5
    assert [r.rule for r in audit.rules] == [
        "min_images",
        "min_viewpoints",
        "min_query_eligible",
        "mapping_status",
        "overlap",
    ]
    # sequential: a class is charged to the first rule it fails
    assert [(r.classes_before, r.classes_after) for r in audit.rules] == [
        (7, 5),
        (5, 4),
        (4, 3),
        (3, 2),
        (2, 1),
    ]
    assert [r.images_dropped for r in audit.rules] == [35, 30, 25, 30, 30]
    assert audit.eligible_class_ids == ["ok"]
    assert audit.classes_after == 1
    assert audit.images_after == 30


def test_provisional_classes_pass_only_with_the_fallback_on():
    classes = [stats("p", mapping_status="provisional"), stats("a", mapping_status="approved")]

    with_fallback = audit_eligibility(classes, THRESHOLDS)
    strict = audit_eligibility(
        classes, THRESHOLDS.model_copy(update={"provisional_fallback": False})
    )

    assert with_fallback.eligible_class_ids == ["a", "p"]
    assert strict.eligible_class_ids == ["a"]
    assert strict.rules[3].classes_dropped == 1


def test_audit_bins_classes_by_image_count_before_any_rule_is_applied():
    classes = [
        stats(f"c{i}", image_count=n, deduplicated_count=n)
        for i, n in enumerate([1, 4, 5, 19, 20, 49, 50, 99, 100, 500])
    ]

    audit = audit_eligibility(classes, THRESHOLDS)

    assert audit.frequency_histogram == {
        "1-4": 2,
        "5-9": 1,
        "10-19": 1,
        "20-49": 2,
        "50-99": 2,
        "100+": 2,
    }


def manifest_rows(spec: list[tuple[str, int, int, int | None, int, str, str]]) -> pd.DataFrame:
    """Rows as (image_id, make_id, model_id, year, viewpoint, duplicate_group_id, status)."""
    frame = pd.DataFrame(
        spec,
        columns=[
            "image_id",
            "make_id",
            "model_id",
            "year",
            "viewpoint",
            "duplicate_group_id",
            "validity_status",
        ],
    )
    frame["sha256"] = frame["image_id"].map(lambda i: hashlib.sha256(i.encode()).hexdigest())
    frame["source_group_id"] = frame["duplicate_group_id"]
    frame["generation_id"] = None
    return frame


def test_class_stats_fall_back_to_the_model_year_class_and_count_known_viewpoints_only():
    rows = manifest_rows(
        [
            ("a1", 1, 10, 2011, 1, "a1", "valid"),
            ("a2", 1, 10, 2011, 1, "a1", "valid"),  # exact duplicate of a1
            ("a3", 1, 10, 2011, -1, "a3", "valid"),  # uncertain viewpoint: not query-eligible
            ("a4", 1, 10, 2011, 4, "a4", "valid"),
            ("a5", 1, 10, 2011, 4, "a5", "excluded"),  # excluded rows never count
            ("b1", 1, 10, 2014, 2, "b1", "valid"),  # another year = another provisional class
            ("c1", 2, 20, None, 3, "c1", "excluded"),  # unknown year, excluded
        ]
    )

    classes = class_stats_from_manifest(rows)

    assert [c.class_id for c in classes] == ["1/10/2011", "1/10/2014"]
    a = classes[0]
    assert (a.make_id, a.model_id, a.generation_id, a.mapping_status) == (
        1,
        10,
        "1/10/2011",
        "provisional",
    )
    assert (a.image_count, a.deduplicated_count, a.viewpoint_count, a.query_eligible_count) == (
        4,
        3,
        2,
        3,
    )
    assert classes[1].image_count == 1


FRACTIONS = {"support": 0.6, "validation": 0.2, "test": 0.2}


def random_manifest(rng: random.Random) -> pd.DataFrame:
    """A synthetic manifest: 2-6 classes, 20-80 images each, duplicate groups of 1-6 images."""
    spec = []
    for class_index in range(rng.randint(2, 6)):
        make_id, model_id, year = 1 + class_index // 3, 10 + class_index, 2010 + class_index % 2
        n = rng.randint(20, 80)
        image_index = 0
        while image_index < n:
            group_size = min(rng.choice([1, 1, 1, 2, 3, 6]), n - image_index)
            group = f"c{class_index}_g{image_index}"
            for _ in range(group_size):
                viewpoint = rng.choice([1, 2, 3, 4, 5, -1])
                image_id = f"c{class_index}_i{image_index}"
                spec.append((image_id, make_id, model_id, year, viewpoint, group, "valid"))
                image_index += 1
    rng.shuffle(spec)
    return manifest_rows(spec)


def test_no_duplicate_group_crosses_a_partition_and_every_class_keeps_support_100_times():
    for case in range(100):
        rng = random.Random(case)
        valid = with_provisional_classes(random_manifest(rng))

        split = assign_partitions(valid, seed=case, fractions=FRACTIONS)

        assert len(split) == len(valid)
        assert set(split["partition"]) <= set(FRACTIONS)
        per_group = split.groupby("source_group_id")["partition"].nunique()
        assert (per_group == 1).all(), f"case {case}: a group spans partitions"
        with_test = set(split.loc[split["partition"] == "test", "class_id"])
        with_support = set(split.loc[split["partition"] == "support", "class_id"])
        assert with_test <= with_support, f"case {case}: a test class has no support"


def test_partitions_are_approximately_60_20_20_per_class_and_stratified_by_viewpoint():
    # one class, 100 singleton images, viewpoints 1..5 with 20 images each
    spec = [(f"i{k}", 1, 10, 2011, 1 + k % 5, f"i{k}", "valid") for k in range(100)]
    valid = with_provisional_classes(manifest_rows(spec))

    split = assign_partitions(valid, seed=73, fractions=FRACTIONS)

    counts = split["partition"].value_counts().to_dict()
    assert counts == {"support": 60, "validation": 20, "test": 20}
    by_view = split.groupby(["partition", "viewpoint"]).size().unstack()
    assert (by_view.loc["support"] == 12).all()
    assert (by_view.loc["validation"] == 4).all()
    assert (by_view.loc["test"] == 4).all()


def test_assignment_is_deterministic_per_seed_and_differs_between_seeds():
    valid = with_provisional_classes(random_manifest(random.Random(7)))

    first = assign_partitions(valid, seed=73, fractions=FRACTIONS)
    second = assign_partitions(valid.sample(frac=1, random_state=1), seed=73, fractions=FRACTIONS)
    other = assign_partitions(valid, seed=13, fractions=FRACTIONS)

    assert first.sort_values("image_id").to_dict("records") == second.sort_values(
        "image_id"
    ).to_dict("records")
    assert (
        first.sort_values("image_id")["partition"].tolist()
        != other.sort_values("image_id")["partition"].tolist()
    )


def test_audit_markdown_lists_every_rule_with_classes_before_and_after_and_images_dropped(
    tmp_path,
):
    from avr.data.split_track_b import write_eligibility_audit

    classes = [
        stats("ok"),
        stats("few_unique", deduplicated_count=19),
        stats("one_view", viewpoint_count=1),
    ]
    audit = audit_eligibility(classes, THRESHOLDS)

    path = write_eligibility_audit(audit, tmp_path / "eligibility_audit.md")

    text = path.read_text()
    assert "| min_images | deduplicated_count >= 20 | 3 | 2 | 1 | 30 |" in text
    assert "| min_viewpoints | viewpoint_count >= 2 | 2 | 1 | 1 | 30 |" in text
    assert "| 20-49 | 3 |" in text  # frequency histogram before any rule
    assert "mapping_status = provisional" in text
    assert "few_unique" in text and "one_view" in text  # dropped classes are named


def eligible_manifest(classes: int = 2, images: int = 30) -> pd.DataFrame:
    spec = []
    for c in range(classes):
        for k in range(images):
            spec.append((f"c{c}_i{k}", 1, 10 + c, 2011, 1 + k % 3, f"c{c}_i{k}", "valid"))
    # one class below min_images: must be dropped, after the audit is written
    spec += [(f"small_{k}", 2, 20, 2011, 1 + k % 2, f"small_{k}", "valid") for k in range(6)]
    return manifest_rows(spec)


def test_build_writes_the_audit_before_dropping_anything(tmp_path, monkeypatch):
    import avr.data.split_track_b as mod

    def boom(*args, **kwargs):
        raise RuntimeError("split failed")

    monkeypatch.setattr(mod, "assign_partitions", boom)
    audit_path = tmp_path / "reports" / "eligibility_audit.md"

    try:
        mod.build_track_b_splits(
            eligible_manifest(),
            THRESHOLDS,
            FRACTIONS,
            seeds=[73],
            audit_path=audit_path,
            out_dir=tmp_path / "splits",
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("the patched split should have failed")

    assert audit_path.is_file()
    assert "| min_images | deduplicated_count >= 20 | 3 | 2 | 1 | 6 |" in audit_path.read_text()
    assert not (tmp_path / "splits").exists() or not list((tmp_path / "splits").iterdir())


def test_build_writes_one_public_table_and_sidecar_per_seed_with_no_paths(tmp_path):
    import hashlib

    import pyarrow.parquet as pq

    from avr.data.split_track_b import build_track_b_splits

    result = build_track_b_splits(
        eligible_manifest(),
        THRESHOLDS,
        FRACTIONS,
        seeds=[73, 13],
        audit_path=tmp_path / "audit.md",
        out_dir=tmp_path / "splits",
    )

    assert result.audit.eligible_class_ids == ["1/10/2011", "1/11/2011"]
    assert sorted(p.name for p in (tmp_path / "splits").iterdir()) == [
        "seed_13.parquet",
        "seed_13.sha256",
        "seed_73.parquet",
        "seed_73.sha256",
    ]
    table = pq.read_table(tmp_path / "splits" / "seed_73.parquet")
    assert table.column_names == [
        "image_id",
        "sha256",
        "class_id",
        "make_id",
        "model_id",
        "viewpoint",
        "partition",
        "mapping_status",
    ]
    frame = table.to_pandas()
    assert len(frame) == 60 and not frame["image_id"].str.startswith("small").any()
    assert frame["image_id"].is_monotonic_increasing
    assert set(frame["mapping_status"]) == {"provisional"}
    # 30 singletons per class, 10 per viewpoint: 6/2/2 per viewpoint, 18/6/6 per class
    assert frame["partition"].value_counts().to_dict() == {
        "support": 36,
        "validation": 12,
        "test": 12,
    }
    per_class_view = frame.groupby(["class_id", "viewpoint", "partition"]).size()
    assert set(per_class_view.xs("support", level="partition")) == {6}
    assert set(per_class_view.xs("test", level="partition")) == {2}
    digest = hashlib.sha256((tmp_path / "splits" / "seed_73.parquet").read_bytes()).hexdigest()
    assert (tmp_path / "splits" / "seed_73.sha256").read_text() == f"{digest}  seed_73.parquet\n"
    assert result.partition_counts[73] == {"support": 36, "validation": 12, "test": 12}


def test_cross_class_duplicate_groups_are_excluded_before_eligibility_and_audited(tmp_path):
    """ADR-0006: a duplicate group spanning two classes leaves Track B with all its images."""
    import pyarrow.parquet as pq

    from avr.data.split_track_b import build_track_b_splits, exclude_cross_class_duplicates

    spec = [
        # group xg: one photo filed under two model-year classes
        ("x1", 1, 10, 2011, 1, "xg", "valid"),
        ("x2", 1, 11, 2011, 1, "xg", "valid"),
        # group yg: three images, two classes
        ("y1", 1, 10, 2011, 2, "yg", "valid"),
        ("y2", 1, 10, 2011, 2, "yg", "valid"),
        ("y3", 1, 11, 2011, 2, "yg", "valid"),
        # group zg: two images, one class: stays
        ("z1", 1, 10, 2011, 3, "zg", "valid"),
        ("z2", 1, 10, 2011, 3, "zg", "valid"),
        # an excluded row never makes its group cross-class
        ("w1", 2, 20, 2011, 3, "zg", "excluded"),
    ]
    manifest = pd.concat([eligible_manifest(), manifest_rows(spec)], ignore_index=True)

    pool, excluded = exclude_cross_class_duplicates(with_provisional_classes(manifest))
    assert excluded.reason == "cross_class_duplicate"
    assert (excluded.groups, excluded.images, excluded.group_ids) == (2, 5, ["xg", "yg"])
    assert sorted(set(pool["image_id"]) & {"x1", "x2", "y1", "y2", "y3", "z1", "z2", "w1"}) == [
        "z1",
        "z2",
    ]

    result = build_track_b_splits(
        manifest,
        THRESHOLDS,
        FRACTIONS,
        seeds=[73],
        audit_path=tmp_path / "audit.md",
        out_dir=tmp_path / "splits",
    )

    assert result.audit.cross_class.groups == 2 and result.audit.cross_class.images == 5
    assert result.audit.images_before == 60 + 6 + 2  # 68 valid rows, the 5 excluded ones gone
    frame = pq.read_table(tmp_path / "splits" / "seed_73.parquet").to_pandas()
    assert len(frame) == 62
    assert not frame["image_id"].isin(["x1", "x2", "y1", "y2", "y3"]).any()
    text = (tmp_path / "audit.md").read_text()
    assert "| cross_class_duplicate | 2 | 5 |" in text
    assert text.index("cross_class_duplicate") < text.index("## Rules in protocol order")
    assert "`xg`, `yg`" in text
