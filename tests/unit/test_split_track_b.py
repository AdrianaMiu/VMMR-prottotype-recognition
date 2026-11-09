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
