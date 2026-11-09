"""Track B class eligibility and group-aware 60/20/20 split (PROTOCOL.md §5 Track B).

Eligibility is a pure function over per-class statistics with thresholds from
configs/splits/track_b.yaml. The rules are applied in protocol order and every class is
charged to the first rule it fails, so the audit shows how many classes and images each
rule removes; the audit is computed and written before any class is dropped.

No approved generation-mapping rows exist yet. With ``provisional_fallback`` on, the
model-year class ``make_id/model_id/year`` stands in for ``generation_id`` and every
output row is marked ``mapping_status = provisional``.
"""

from __future__ import annotations

import random
from typing import Literal

import pandas as pd
from pydantic import BaseModel

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


class EligibilityAudit(BaseModel):
    thresholds: EligibilityThresholds
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


def audit_eligibility(classes: list[ClassStats], t: EligibilityThresholds) -> EligibilityAudit:
    """Apply the rules in protocol order and record what each one removes; drops nothing."""
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


def class_stats_from_manifest(manifest: pd.DataFrame) -> list[ClassStats]:
    """One ``ClassStats`` per provisional class, sorted by ``class_id``.

    ``deduplicated_count`` is the number of distinct duplicate groups, ``viewpoint_count``
    the number of distinct known viewpoints (``-1`` is uncertain and not counted) and
    ``query_eligible_count`` the number of valid images with a known viewpoint.
    """
    valid = with_provisional_classes(manifest)
    classes = []
    for class_id, rows in valid.groupby("class_id", sort=True):
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


PARTITIONS: tuple[str, ...] = ("support", "validation", "test")


def _class_rng(seed: int, class_id: str) -> random.Random:
    """One generator per (seed, class): str seeding is stable across processes."""
    return random.Random(f"{seed}:{class_id}")


def _assign_class(rows: pd.DataFrame, seed: int, fractions: dict[str, float]) -> pd.Series:
    """Greedy group-aware assignment of one class, approximately stratified by viewpoint.

    Whole ``source_group_id`` groups are placed, largest first after a seeded shuffle,
    into the partition whose remaining per-viewpoint deficit (target minus assigned,
    summed over the group's images) is largest; ties go to the earlier partition, so the
    first group of every class lands in ``support``.
    """
    class_id = str(rows["class_id"].iloc[0])
    partitions = list(fractions)
    view_totals = rows["viewpoint"].value_counts()
    target = {p: {v: fractions[p] * n for v, n in view_totals.items()} for p in partitions}
    assigned = {p: dict.fromkeys(view_totals.index, 0) for p in partitions}

    groups = {str(g): sub["viewpoint"].tolist() for g, sub in rows.groupby("source_group_id")}
    order = sorted(groups)
    _class_rng(seed, class_id).shuffle(order)
    order.sort(key=lambda g: -len(groups[g]))

    choice: dict[str, str] = {}
    for group_id in order:
        views = groups[group_id]
        best = max(
            partitions,
            key=lambda p: (sum(target[p][v] - assigned[p][v] for v in views), -partitions.index(p)),
        )
        for v in views:
            assigned[best][v] += 1
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
