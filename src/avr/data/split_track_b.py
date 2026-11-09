"""Track B class eligibility and group-aware 60/20/20 split (PROTOCOL.md §5 Track B).

Eligibility is a pure function over per-class statistics with thresholds from
configs/splits/track_b.yaml. The rules are applied in protocol order and every class is
charged to the first rule it fails, so the audit shows how many classes and images each
rule removes; the audit is computed and written before any class is dropped.

No approved generation-mapping rows exist yet. With ``provisional_fallback`` on, the
model-year class ``make_id/model_id/year`` stands in for ``generation_id`` and every
output row is marked ``mapping_status = provisional``.

Before eligibility, every image whose ``duplicate_group_id`` spans more than one class is
excluded with reason ``cross_class_duplicate`` (ADR-0006): a group-aware split that assigns
groups class by class could not keep such a group in one partition. The exclusion is
counted in the audit before the eligibility rules.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Literal

import pandas as pd
import pyarrow as pa
from pydantic import BaseModel

from avr.data.splits_io import SplitManifestOutputs, write_split_manifest
from avr.generation_mapping.schema import MappingStatus

Rule = Literal["min_images", "min_viewpoints", "min_query_eligible", "mapping_status", "overlap"]
RULES: tuple[Rule, ...] = (
    "min_images",
    "min_viewpoints",
    "min_query_eligible",
    "mapping_status",
    "overlap",
)
KNOWN_VIEWPOINTS = frozenset({1, 2, 3, 4, 5})
PROVISIONAL_STATUS: MappingStatus = "provisional"
FREQUENCY_BINS: tuple[tuple[str, int, int], ...] = (
    ("1-4", 1, 4),
    ("5-9", 5, 9),
    ("10-19", 10, 19),
    ("20-49", 20, 49),
    ("50-99", 50, 99),
    ("100+", 100, 10**9),
)


class EligibilityThresholds(BaseModel):
    min_images: int
    min_viewpoints: int
    min_query_eligible: int
    require_approved_mapping: bool
    provisional_fallback: bool


class ClassStats(BaseModel):
    """Per-class counts the eligibility rules read; computed once from the manifest."""

    class_id: str
    make_id: int
    model_id: int
    generation_id: str
    mapping_status: MappingStatus
    image_count: int
    deduplicated_count: int
    viewpoint_count: int
    query_eligible_count: int
    has_unresolved_overlap: bool = False


class RuleOutcome(BaseModel):
    rule: Rule
    threshold: str
    classes_before: int
    classes_after: int
    classes_dropped: int
    images_dropped: int
    dropped_class_ids: list[str]


class CrossClassExclusion(BaseModel):
    """Images excluded before eligibility because their duplicate group spans classes."""

    reason: Literal["cross_class_duplicate"] = "cross_class_duplicate"
    groups: int = 0
    images: int = 0
    group_ids: list[str] = []


class EligibilityAudit(BaseModel):
    thresholds: EligibilityThresholds
    cross_class: CrossClassExclusion = CrossClassExclusion()
    classes_before: int
    images_before: int
    frequency_histogram: dict[str, int]
    rules: list[RuleOutcome]
    classes_after: int
    images_after: int
    eligible_class_ids: list[str]


def _passes(stats: ClassStats, rule: Rule, t: EligibilityThresholds) -> bool:
    if rule == "min_images":
        return stats.deduplicated_count >= t.min_images
    if rule == "min_viewpoints":
        return stats.viewpoint_count >= t.min_viewpoints
    if rule == "min_query_eligible":
        return stats.query_eligible_count >= t.min_query_eligible
    if rule == "mapping_status":
        if not t.require_approved_mapping:
            return True
        allowed = {"approved", "provisional"} if t.provisional_fallback else {"approved"}
        return stats.mapping_status in allowed
    return not stats.has_unresolved_overlap


def _threshold_text(rule: Rule, t: EligibilityThresholds) -> str:
    if rule == "min_images":
        return f"deduplicated_count >= {t.min_images}"
    if rule == "min_viewpoints":
        return f"viewpoint_count >= {t.min_viewpoints}"
    if rule == "min_query_eligible":
        return f"query_eligible_count >= {t.min_query_eligible}"
    if rule == "mapping_status":
        if not t.require_approved_mapping:
            return "not required"
        return "approved or provisional" if t.provisional_fallback else "approved"
    return "no unresolved overlap"


def frequency_histogram(classes: list[ClassStats]) -> dict[str, int]:
    return {
        label: sum(lo <= c.image_count <= hi for c in classes) for label, lo, hi in FREQUENCY_BINS
    }


def audit_eligibility(
    classes: list[ClassStats],
    t: EligibilityThresholds,
    cross_class: CrossClassExclusion | None = None,
) -> EligibilityAudit:
    """Apply the rules in protocol order and record what each one removes; drops nothing.

    ``classes`` are the stats of the pool after the cross-class exclusion, which
    ``cross_class`` records; the counts before the first rule exclude those images.
    """
    remaining = sorted(classes, key=lambda c: c.class_id)
    outcomes = []
    for rule in RULES:
        kept = [c for c in remaining if _passes(c, rule, t)]
        dropped = [c for c in remaining if not _passes(c, rule, t)]
        outcomes.append(
            RuleOutcome(
                rule=rule,
                threshold=_threshold_text(rule, t),
                classes_before=len(remaining),
                classes_after=len(kept),
                classes_dropped=len(dropped),
                images_dropped=sum(c.image_count for c in dropped),
                dropped_class_ids=[c.class_id for c in dropped],
            )
        )
        remaining = kept
    return EligibilityAudit(
        thresholds=t,
        cross_class=cross_class or CrossClassExclusion(),
        classes_before=len(classes),
        images_before=sum(c.image_count for c in classes),
        frequency_histogram=frequency_histogram(classes),
        rules=outcomes,
        classes_after=len(remaining),
        images_after=sum(c.image_count for c in remaining),
        eligible_class_ids=[c.class_id for c in remaining],
    )


def provisional_class_id(make_id: int, model_id: int, year: int) -> str:
    """Stand-in ``generation_id`` while no approved mapping exists: the model-year class."""
    return f"{make_id}/{model_id}/{year}"


def with_provisional_classes(manifest: pd.DataFrame) -> pd.DataFrame:
    """Valid rows with ``class_id``/``generation_id`` set to the model-year stand-in."""
    valid = manifest[manifest["validity_status"] == "valid"].copy()
    valid["class_id"] = [
        provisional_class_id(int(m), int(mo), int(y))
        for m, mo, y in zip(valid["make_id"], valid["model_id"], valid["year"], strict=True)
    ]
    valid["generation_id"] = valid["class_id"]
    valid["mapping_status"] = PROVISIONAL_STATUS
    return valid


def exclude_cross_class_duplicates(
    valid: pd.DataFrame,
) -> tuple[pd.DataFrame, CrossClassExclusion]:
    """Drop every image whose ``duplicate_group_id`` holds more than one ``class_id``.

    Returns the remaining rows and the count of groups and images removed (ADR-0006).
    """
    classes_per_group = valid.groupby("duplicate_group_id")["class_id"].nunique()
    crossing = sorted(str(g) for g in classes_per_group.index[classes_per_group > 1])
    in_crossing = valid["duplicate_group_id"].isin(crossing)
    exclusion = CrossClassExclusion(
        groups=len(crossing), images=int(in_crossing.sum()), group_ids=crossing
    )
    return valid[~in_crossing], exclusion


def class_stats(pool: pd.DataFrame) -> list[ClassStats]:
    """One ``ClassStats`` per provisional class of ``pool``, sorted by ``class_id``.

    ``deduplicated_count`` is the number of distinct duplicate groups, ``viewpoint_count``
    the number of distinct known viewpoints (``-1`` is uncertain and not counted) and
    ``query_eligible_count`` the number of valid images with a known viewpoint.
    """
    classes = []
    for class_id, rows in pool.groupby("class_id", sort=True):
        known = rows[rows["viewpoint"].isin(KNOWN_VIEWPOINTS)]
        classes.append(
            ClassStats(
                class_id=str(class_id),
                make_id=int(rows["make_id"].iloc[0]),
                model_id=int(rows["model_id"].iloc[0]),
                generation_id=str(class_id),
                mapping_status=PROVISIONAL_STATUS,
                image_count=len(rows),
                deduplicated_count=rows["duplicate_group_id"].nunique(),
                viewpoint_count=known["viewpoint"].nunique(),
                query_eligible_count=len(known),
            )
        )
    return classes


def class_stats_from_manifest(manifest: pd.DataFrame) -> list[ClassStats]:
    """``class_stats`` of the valid rows after the cross-class exclusion."""
    pool, _ = exclude_cross_class_duplicates(with_provisional_classes(manifest))
    return class_stats(pool)


PARTITIONS: tuple[str, ...] = ("support", "validation", "test")


def _class_rng(seed: int, class_id: str) -> random.Random:
    """One generator per (seed, class): str seeding is stable across processes."""
    return random.Random(f"{seed}:{class_id}")


def _assign_class(rows: pd.DataFrame, seed: int, fractions: dict[str, float]) -> pd.Series:
    """Greedy group-aware assignment of one class, approximately stratified by viewpoint.

    Whole ``source_group_id`` groups are placed, largest first after a seeded shuffle,
    into the partition with the largest remaining deficit, measured in images as the
    class-level deficit (``fraction * class size`` minus assigned) plus the per-viewpoint
    deficit summed over the group's images. The class term keeps every class close to
    60/20/20; the viewpoint term makes the split approximately viewpoint-stratified.
    Ties go to the earlier partition, so the first group of every class lands in
    ``support``.
    """
    class_id = str(rows["class_id"].iloc[0])
    partitions = list(fractions)
    view_totals = rows["viewpoint"].value_counts()
    target = {p: {v: fractions[p] * n for v, n in view_totals.items()} for p in partitions}
    assigned = {p: dict.fromkeys(view_totals.index, 0) for p in partitions}
    class_target = {p: fractions[p] * len(rows) for p in partitions}
    class_assigned = dict.fromkeys(partitions, 0)

    groups = {str(g): sub["viewpoint"].tolist() for g, sub in rows.groupby("source_group_id")}
    order = sorted(groups)
    _class_rng(seed, class_id).shuffle(order)
    order.sort(key=lambda g: -len(groups[g]))

    choice: dict[str, str] = {}
    for group_id in order:
        views = groups[group_id]
        best = max(
            partitions,
            key=lambda p: (
                class_target[p]
                - class_assigned[p]
                + sum(target[p][v] - assigned[p][v] for v in views),
                -partitions.index(p),
            ),
        )
        for v in views:
            assigned[best][v] += 1
        class_assigned[best] += len(views)
        choice[group_id] = best
    return rows["source_group_id"].astype(str).map(choice)


def assign_partitions(valid: pd.DataFrame, seed: int, fractions: dict[str, float]) -> pd.DataFrame:
    """Add ``partition`` to every row; no ``source_group_id`` group crosses a partition.

    Each class is split on its own so the result is stratified by class; the result does
    not depend on the row order of ``valid``.
    """
    out = valid.copy()
    out["partition"] = ""
    for _, rows in out.groupby("class_id", sort=True):
        out.loc[rows.index, "partition"] = _assign_class(rows, seed, fractions)
    return out


OUTPUT_SCHEMA = pa.schema(
    [
        ("image_id", pa.string()),
        ("sha256", pa.string()),
        ("class_id", pa.string()),
        ("make_id", pa.int32()),
        ("model_id", pa.int32()),
        ("viewpoint", pa.int8()),
        ("partition", pa.string()),
        ("mapping_status", pa.string()),
    ]
)
OUTPUT_COLUMNS = tuple(OUTPUT_SCHEMA.names)


def _table(rows: list[list[object]]) -> str:
    return "\n".join("| " + " | ".join(str(c) for c in row) + " |" for row in rows)


def write_eligibility_audit(audit: EligibilityAudit, path: Path) -> Path:
    """Render the class-frequency audit as Markdown: thresholds, histogram, per-rule drops."""
    t = audit.thresholds
    lines = [
        "# Track B class eligibility audit",
        "",
        "Generated by `scripts/build_track_b_splits.py` (PROTOCOL.md §5 Track B); do not edit",
        "by hand. Classes are the provisional model-year stand-ins `make_id/model_id/year`",
        "(no approved generation-mapping rows exist yet, so every output row carries",
        "`mapping_status = provisional`). Rules are applied in protocol order and a class is",
        "charged to the first rule it fails. This report is written before any class is dropped.",
        "Before any rule, every image whose `duplicate_group_id` spans more than one class is",
        "excluded with reason `cross_class_duplicate` (ADR-0006); the counts below start after",
        "that exclusion.",
        "",
        "## Thresholds (configs/splits/track_b.yaml)",
        "",
        f"- min_images: {t.min_images} (valid images after deduplication)",
        f"- min_viewpoints: {t.min_viewpoints} (known CompCars viewpoints; -1 not counted)",
        f"- min_query_eligible: {t.min_query_eligible} (valid images with a known viewpoint)",
        f"- require_approved_mapping: {str(t.require_approved_mapping).lower()}",
        f"- provisional_fallback: {str(t.provisional_fallback).lower()}",
        "",
        "## Excluded before eligibility (ADR-0006)",
        "",
        _table(
            [
                ["reason", "groups", "images"],
                ["---"] * 3,
                [audit.cross_class.reason, audit.cross_class.groups, audit.cross_class.images],
            ]
        ),
        "",
        "Groups: " + (", ".join(f"`{g}`" for g in audit.cross_class.group_ids) or "none") + ".",
        "",
        "## Class-frequency histogram before any rule (valid images per class)",
        "",
        _table(
            [
                ["images per class", "classes"],
                ["---", "---"],
                *[[label, n] for label, n in audit.frequency_histogram.items()],
            ]
        ),
        "",
        f"Classes before: {audit.classes_before}; valid images before: {audit.images_before}.",
        "",
        "## Rules in protocol order",
        "",
        _table(
            [
                [
                    "rule",
                    "threshold",
                    "classes before",
                    "classes after",
                    "classes dropped",
                    "images dropped",
                ],
                ["---"] * 6,
                *[
                    [
                        r.rule,
                        r.threshold,
                        r.classes_before,
                        r.classes_after,
                        r.classes_dropped,
                        r.images_dropped,
                    ]
                    for r in audit.rules
                ],
            ]
        ),
        "",
        f"Classes after: {audit.classes_after}; valid images after: {audit.images_after}.",
        "",
        "## Dropped classes per rule",
        "",
    ]
    for r in audit.rules:
        lines.append(f"### {r.rule} ({r.classes_dropped} classes)")
        lines.append("")
        lines.append(", ".join(f"`{c}`" for c in r.dropped_class_ids) or "none")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    return path


class TrackBSplitResult(BaseModel):
    audit: EligibilityAudit
    audit_path: Path
    outputs: dict[int, SplitManifestOutputs]
    partition_counts: dict[int, dict[str, int]]


def split_table(split: pd.DataFrame) -> pa.Table:
    """The public per-seed table, sorted by ``image_id``: ids, hashes and labels, no paths."""
    ordered = split.sort_values("image_id")[list(OUTPUT_COLUMNS)].reset_index(drop=True)
    return pa.Table.from_pandas(ordered, schema=OUTPUT_SCHEMA, preserve_index=False)


def build_track_b_splits(
    manifest: pd.DataFrame,
    thresholds: EligibilityThresholds,
    fractions: dict[str, float],
    seeds: list[int],
    audit_path: Path,
    out_dir: Path,
) -> TrackBSplitResult:
    """Exclude cross-class groups, audit eligibility and write it, then split per seed."""
    pool, cross_class = exclude_cross_class_duplicates(with_provisional_classes(manifest))
    audit = audit_eligibility(class_stats(pool), thresholds, cross_class)
    write_eligibility_audit(audit, audit_path)

    eligible = pool[pool["class_id"].isin(audit.eligible_class_ids)]
    outputs, counts = {}, {}
    for seed in seeds:
        split = assign_partitions(eligible, seed=seed, fractions=fractions)
        outputs[seed] = write_split_manifest(split_table(split), out_dir / f"seed_{seed}.parquet")
        counts[seed] = {p: int((split["partition"] == p).sum()) for p in fractions}
    return TrackBSplitResult(
        audit=audit, audit_path=audit_path, outputs=outputs, partition_counts=counts
    )
