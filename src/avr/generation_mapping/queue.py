"""Generation-mapping curation queue built from the image manifest (PROTOCOL.md §6).

One queue row per distinct valid ``(make_id, model_id, year)`` model-year class in the
manifest. Every row is ``provisional`` and carries only what CompCars itself states:
ids, names and the folder year. Generation identifiers, intervals, facelift phases,
sources and reviewers are left empty for the human reviewers; nothing is inferred here.

The queue is written as the 23 §6 columns (``generation_mapping_queue_v<ver>.csv``) plus a
separate counts table (``..._counts.csv``) with the valid images per row, so the queue file
stays a plain mapping table. The reviewer sheet is one row per model with the year list.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from pydantic import BaseModel

from avr.generation_mapping.schema import COLUMNS

KEY_COLUMNS = ("make_id", "model_id", "compcars_year")
IMAGE_COUNT = "image_count"
QUEUE_NAME = "generation_mapping_queue"
REVIEWER_SHEET_NAME = "reviewer_sheet"

REVIEWER_SHEET_COLUMNS = (
    "make_id",
    "make_name",
    "model_id",
    "model_name",
    "years",
    "year_count",
    "image_count",
    "generation_id",
    "generation_name",
    "generation_start_year",
    "generation_end_year",
    "facelift_phase",
    "market_region",
    "source_url_1",
    "source_url_2",
    "reviewer_1_initials",
    "reviewer_2_initials",
    "review_notes",
)


class QueueValidationError(ValueError):
    """The queue breaks a rule it must hold before it reaches the reviewers."""


class QueueOutputs(BaseModel):
    queue: Path
    counts: Path


def build_queue(manifest: pd.DataFrame) -> pd.DataFrame:
    """Queue rows (§6 columns + ``image_count``) from the valid rows of the manifest."""
    valid = manifest[manifest["validity_status"] == "valid"]
    grouped = (
        valid.groupby(["make_id", "make_name", "model_id", "model_name", "year"], sort=True)
        .size()
        .reset_index(name=IMAGE_COUNT)
        .rename(columns={"year": "compcars_year"})
    )
    queue = pd.DataFrame({column: "" for column in COLUMNS}, index=grouped.index)
    for column in ("make_id", "make_name", "model_id", "model_name", "compcars_year"):
        queue[column] = grouped[column]
    for column in ("make_id", "model_id", "compcars_year"):
        queue[column] = queue[column].astype("int64")
    queue["canonical_model_name"] = grouped["model_name"]
    queue["mapping_status"] = "provisional"
    queue[IMAGE_COUNT] = grouped[IMAGE_COUNT].astype("int64")
    queue = queue.sort_values(list(KEY_COLUMNS), ignore_index=True)
    return queue


def validate_queue(queue: pd.DataFrame) -> pd.DataFrame:
    """Check the queue's columns, key uniqueness and that no row is ``approved``."""
    expected = [*COLUMNS, IMAGE_COUNT]
    if list(queue.columns) != expected:
        raise QueueValidationError(f"queue columns {list(queue.columns)} != {expected}")
    duplicates = queue.duplicated(list(KEY_COLUMNS))
    if duplicates.any():
        keys = queue.loc[duplicates, list(KEY_COLUMNS)].values.tolist()
        raise QueueValidationError(f"duplicate model-year keys: {keys}")
    statuses = set(queue["mapping_status"])
    if "approved" in statuses:
        raise QueueValidationError("queue must not contain approved rows")
    if statuses - {"provisional", "ambiguous"}:
        raise QueueValidationError(f"unknown mapping_status values: {sorted(statuses)}")
    return queue


def write_queue(queue: pd.DataFrame, mappings_dir: Path, version: str) -> QueueOutputs:
    """Write the 23-column queue CSV and the separate per-row image-count CSV."""
    validate_queue(queue)
    mappings_dir.mkdir(parents=True, exist_ok=True)
    out = QueueOutputs(
        queue=mappings_dir / f"{QUEUE_NAME}_v{version}.csv",
        counts=mappings_dir / f"{QUEUE_NAME}_v{version}_counts.csv",
    )
    queue[list(COLUMNS)].to_csv(out.queue, index=False)
    queue[[*KEY_COLUMNS, IMAGE_COUNT]].to_csv(out.counts, index=False)
    return out


def build_reviewer_sheet(queue: pd.DataFrame) -> pd.DataFrame:
    """One row per model with its year list and image count, sorted by make then model."""
    per_model = (
        queue.groupby(["make_id", "make_name", "model_id", "model_name"], sort=False)
        .agg(
            years=("compcars_year", lambda s: ";".join(str(y) for y in sorted(s))),
            year_count=("compcars_year", "size"),
            image_count=(IMAGE_COUNT, "sum"),
        )
        .reset_index()
        .sort_values(["make_name", "model_name", "make_id", "model_id"], ignore_index=True)
    )
    for column in REVIEWER_SHEET_COLUMNS:
        if column not in per_model.columns:
            per_model[column] = ""
    return per_model[list(REVIEWER_SHEET_COLUMNS)]
