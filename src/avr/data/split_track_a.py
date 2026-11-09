"""Track A split manifest: the official CompCars split plus a nested validation partition.

PROTOCOL.md §5 Track A: prototypes come from the official training half (``support``),
queries from the official test half (``query``), and hyper-parameters are chosen on a
deterministic, class-stratified validation partition carved out of the support half.
The official files list ``make_id/model_id/year/file.jpg``; they are matched against the
private path table (``image/`` + path) and never published (ADR-0002). Images listed but
absent from the manifest, and images present but unlisted, are reported, never dropped
silently. A duplicate group that spans support and query is a property of the official
split: it is reported and left as distributed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
from pydantic import BaseModel

from avr.data.manifest import IMAGE_DIR
from avr.data.split_identity import OFFICIAL_SPLIT_FILES

SUPPORT, QUERY = "support", "query"
FIT, VALIDATION = "fit", "validation"
SIDES = (SUPPORT, QUERY)
SPLIT_NAME = "official_cvpr2015"
PUBLIC_SCHEMA = pa.schema(
    [
        ("image_id", pa.string()),
        ("sha256", pa.string()),
        ("make_id", pa.int32()),
        ("model_id", pa.int32()),
        ("split_track_a", pa.string()),
        ("track_a_validation", pa.string()),
    ]
)
JOIN_COLUMNS = (
    "image_id",
    "sha256",
    "make_id",
    "model_id",
    "duplicate_group_id",
    "validity_status",
    "split_track_a",
)


class OfficialSplit(BaseModel):
    """Relative image paths (without ``image/``) of the two official classification files."""

    support: list[str]
    query: list[str]


class JoinResult(BaseModel, arbitrary_types_allowed=True):
    rows: pd.DataFrame
    missing: dict[str, list[str]]
    unlisted_image_ids: list[str]
    cross_split_group_ids: list[str]


def _read_paths(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def read_official_split(root: Path) -> OfficialSplit:
    """Read ``train.txt`` as the support half and ``test.txt`` as the query half."""
    train, test = (root / rel for rel in OFFICIAL_SPLIT_FILES)
    return OfficialSplit(support=_read_paths(train), query=_read_paths(test))


def join_official_split(
    manifest: pd.DataFrame, paths: pd.DataFrame, official: OfficialSplit
) -> JoinResult:
    """Label every listed image with its official side; report what does not line up."""
    overlap = set(official.support) & set(official.query)
    if overlap:
        raise ValueError(f"{len(overlap)} paths are listed on both sides of the official split")
    side_of_path = {f"{IMAGE_DIR}/{p}": SUPPORT for p in official.support}
    side_of_path.update({f"{IMAGE_DIR}/{p}": QUERY for p in official.query})

    labelled = paths.assign(split_track_a=paths.relative_path.map(side_of_path))
    found = set(labelled.relative_path[labelled.split_track_a.notna()])
    missing = {
        side: [p for p in listed if f"{IMAGE_DIR}/{p}" not in found]
        for side, listed in ((SUPPORT, official.support), (QUERY, official.query))
    }

    # the manifest reserves ``split_track_a`` as a null column; the join fills it
    joined = manifest.drop(columns="split_track_a", errors="ignore").merge(
        labelled[["image_id", "split_track_a"]], on="image_id", how="left"
    )
    unlisted = sorted(joined.image_id[joined.split_track_a.isna()])
    rows = joined[joined.split_track_a.notna()].sort_values("image_id").reset_index(drop=True)
    sides_per_group = rows.groupby("duplicate_group_id").split_track_a.nunique()
    cross = sorted(sides_per_group.index[sides_per_group > 1])
    return JoinResult(
        rows=rows[list(JOIN_COLUMNS)],
        missing=missing,
        unlisted_image_ids=unlisted,
        cross_split_group_ids=cross,
    )


def _validation_count(unit_count: int, fraction: float) -> int:
    """Units held out of a class: the rounded fraction, at least one when two exist."""
    if unit_count < 2:
        return 0
    return min(unit_count - 1, max(1, round(fraction * unit_count)))


def partition_validation(rows: pd.DataFrame, fraction: float, seed: int) -> pd.Series:
    """``fit``/``validation`` for every fit-eligible support image, ``None`` elsewhere.

    The unit of assignment is the duplicate group, so a group never crosses the boundary.
    Each unit belongs to the smallest (make_id, model_id) class among its members; within
    a class the units are shuffled with ``seed`` and the first ``_validation_count`` go to
    validation. Classes are processed in sorted order from one generator, so the result is
    a pure function of (rows, fraction, seed).
    """
    eligible = rows[(rows.split_track_a == SUPPORT) & (rows.validity_status == "valid")]
    unit_class = (
        eligible.groupby("duplicate_group_id")[["make_id", "model_id"]].min().apply(tuple, axis=1)
    )
    rng = np.random.default_rng(seed)
    validation_units: set[str] = set()
    for _, units in sorted(unit_class.groupby(unit_class).groups.items()):
        ordered = sorted(units)
        order = rng.permutation(len(ordered))
        chosen = order[: _validation_count(len(ordered), fraction)]
        validation_units.update(ordered[i] for i in chosen)
    labels = pd.Series([None] * len(rows), index=rows.index, dtype=object)
    labels[eligible.index] = [
        VALIDATION if g in validation_units else FIT for g in eligible.duplicate_group_id
    ]
    return labels


class TrackAReport(BaseModel):
    """Counts published with the split manifest; no path and no image id appears here."""

    listed: dict[str, int]
    matched: dict[str, int]
    missing: dict[str, int]
    unlisted_images: int
    classes: dict[str, int]
    excluded: dict[str, int]
    cross_split_groups: int
    cross_split_images: int
    validation_fraction: float
    validation_seed: int
    partition: dict[str, int]
    classes_with_validation: int


class TrackABuild(BaseModel, arbitrary_types_allowed=True):
    table: pa.Table
    report: TrackAReport
    join: JoinResult


def build_track_a(
    manifest: pd.DataFrame,
    paths: pd.DataFrame,
    official: OfficialSplit,
    fraction: float,
    seed: int,
) -> TrackABuild:
    """Join the official split, partition the support half and shape the public table."""
    join = join_official_split(manifest, paths, official)
    rows = join.rows.assign(track_a_validation=partition_validation(join.rows, fraction, seed))
    by_side = {side: rows[rows.split_track_a == side] for side in SIDES}
    partitioned = rows[rows.track_a_validation.notna()]
    held_out = partitioned.groupby(["make_id", "model_id"]).track_a_validation.apply(
        lambda s: (s == VALIDATION).any()
    )
    report = TrackAReport(
        listed={side: len(getattr(official, side)) for side in SIDES},
        matched={side: len(by_side[side]) for side in SIDES},
        missing={side: len(join.missing[side]) for side in SIDES},
        unlisted_images=len(join.unlisted_image_ids),
        classes={
            side: len(by_side[side][["make_id", "model_id"]].drop_duplicates()) for side in SIDES
        },
        excluded={side: int((by_side[side].validity_status != "valid").sum()) for side in SIDES},
        cross_split_groups=len(join.cross_split_group_ids),
        cross_split_images=int(rows.duplicate_group_id.isin(join.cross_split_group_ids).sum()),
        validation_fraction=fraction,
        validation_seed=seed,
        partition={
            label: int((partitioned.track_a_validation == label).sum())
            for label in (FIT, VALIDATION)
        },
        classes_with_validation=int(held_out.sum()),
    )
    table = pa.Table.from_pandas(
        rows[PUBLIC_SCHEMA.names], schema=PUBLIC_SCHEMA, preserve_index=False
    )
    return TrackABuild(table=table, report=report, join=join)


def render_readme(report: TrackAReport) -> str:
    """The Markdown report published next to the split manifest."""
    side_rows = "\n".join(
        f"| {side} | {report.listed[side]} | {report.matched[side]} | {report.missing[side]} | "
        f"{report.classes[side]} | {report.excluded[side]} |"
        for side in SIDES
    )
    return f"""# Track A split: {SPLIT_NAME}

Generated by `scripts/build_track_a_split.py` (PROTOCOL.md §5 Track A). The official
CompCars `train_test_split/classification/{{train,test}}.txt` files label the support
(`train`) and query (`test`) halves; the public manifest
`data/splits/track_a/{SPLIT_NAME}.parquet` carries `image_id`, `sha256`, `make_id`,
`model_id`, `split_track_a` and `track_a_validation` only (ADR-0002). The split-file
identity is recorded in `data/manifests/track_a_split_identity.json`.

## Official split against the manifest

| side | listed | matched | missing | classes | excluded |
|------|--------|---------|---------|---------|----------|
{side_rows}

- listed: lines in the official file; matched: manifest rows labelled with that side;
  missing: listed paths with no image in the manifest (counted, never published).
- excluded: matched rows whose `validity_status` is `excluded` (PROTOCOL.md §7); they keep
  their split label but take no part in prototypes, queries or the validation partition.
- {report.unlisted_images} manifest images are listed on neither side: Part-I is a subset
  of the web-nature release.
- {report.cross_split_groups} duplicate group(s) span support and query, covering
  {report.cross_split_images} images. This is a property of the official split; it is
  reported and left as distributed.

## Nested validation partition (support half only)

- fraction {report.validation_fraction}, seed {report.validation_seed}
  (`configs/splits/track_a.yaml`); class-stratified on (`make_id`, `model_id`),
  group-aware on `duplicate_group_id`, at least one unit held out and one kept per class.
- fit: {report.partition[FIT]}, validation: {report.partition[VALIDATION]}
  (valid support images); {report.classes_with_validation} classes have a validation unit.
- Final prototypes are rebuilt from the complete support half once choices are frozen.
"""
