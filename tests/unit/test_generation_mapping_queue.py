"""Curation queue built from the image manifest (PROTOCOL.md §6, ticket M1-11).

The synthetic manifest below is small enough that every expected value is computed by
hand in the comments next to it.
"""

from pathlib import Path

import pandas as pd
import pytest

from avr.generation_mapping.queue import (
    QueueValidationError,
    build_queue,
    build_reviewer_sheet,
    validate_queue,
    write_queue,
)
from avr.generation_mapping.schema import COLUMNS


def manifest_row(make_id, make_name, model_id, model_name, year, status="valid"):
    return dict(
        make_id=make_id,
        make_name=make_name,
        model_id=model_id,
        model_name=model_name,
        year=year,
        validity_status=status,
        exclusion_reason=None if status == "valid" else "unknown_year",
    )


@pytest.fixture
def manifest() -> pd.DataFrame:
    # Valid images per (make, model, year):
    #   Dacia(2)/Logan(20): 2008 x2, 2010 x1
    #   Dacia(2)/Duster(21): 2012 x1
    #   Audi(1)/A3(10): 2014 x3
    # plus one excluded Logan image with no year, and one excluded Audi A3 2014 image.
    rows = [
        manifest_row(2, "Dacia", 20, "Logan", 2008),
        manifest_row(2, "Dacia", 20, "Logan", 2010),
        manifest_row(2, "Dacia", 20, "Logan", 2008),
        manifest_row(2, "Dacia", 21, "Duster", 2012),
        manifest_row(1, "Audi", 10, "A3", 2014),
        manifest_row(1, "Audi", 10, "A3", 2014),
        manifest_row(1, "Audi", 10, "A3", 2014),
        manifest_row(2, "Dacia", 20, "Logan", None, status="excluded"),
        manifest_row(1, "Audi", 10, "A3", 2014, status="excluded"),
    ]
    frame = pd.DataFrame(rows)
    frame["year"] = frame["year"].astype("float64")  # as pandas reads a nullable int32
    return frame


def test_one_queue_row_per_valid_model_year_sorted_by_ids(manifest):
    queue = build_queue(manifest)

    assert list(queue.columns) == [*COLUMNS, "image_count"]
    keys = queue[["make_id", "model_id", "compcars_year", "image_count"]].values.tolist()
    assert keys == [[1, 10, 2014, 3], [2, 20, 2008, 2], [2, 20, 2010, 1], [2, 21, 2012, 1]]
    assert queue["compcars_year"].dtype.kind == "i"


def test_queue_rows_are_provisional_with_nothing_invented(manifest):
    queue = build_queue(manifest)
    logan_2010 = queue[(queue.model_id == 20) & (queue.compcars_year == 2010)].iloc[0]

    assert logan_2010["make_name"] == "Dacia"
    assert logan_2010["model_name"] == "Logan"
    assert logan_2010["canonical_model_name"] == "Logan"
    assert logan_2010["mapping_status"] == "provisional"
    empty = [
        "generation_id",
        "generation_name",
        "generation_start_year",
        "generation_end_year",
        "facelift_phase",
        "body_style_or_variant",
        "market_region",
        "source_url_1",
        "source_url_2",
        "source_title_1",
        "source_title_2",
        "accessed_at",
        "reviewer_1",
        "reviewer_2",
        "review_notes",
        "ambiguity_code",
    ]
    assert all(logan_2010[c] == "" for c in empty), logan_2010[empty].to_dict()


def test_validate_queue_accepts_the_built_queue_and_rejects_approved_rows(manifest):
    queue = build_queue(manifest)
    assert validate_queue(queue) is queue

    approved = queue.copy()
    approved.loc[0, "mapping_status"] = "approved"
    with pytest.raises(QueueValidationError, match="approved"):
        validate_queue(approved)

    duplicated = pd.concat([queue, queue.iloc[[0]]], ignore_index=True)
    with pytest.raises(QueueValidationError, match="duplicate"):
        validate_queue(duplicated)

    renamed = queue.rename(columns={"compcars_year": "year"})
    with pytest.raises(QueueValidationError, match="columns"):
        validate_queue(renamed)


def test_write_queue_splits_schema_columns_from_counts(manifest, tmp_path: Path):
    out = write_queue(build_queue(manifest), tmp_path, version="0.1")

    queue_csv = tmp_path / "generation_mapping_queue_v0.1.csv"
    counts_csv = tmp_path / "generation_mapping_queue_v0.1_counts.csv"
    assert out.queue == queue_csv and out.counts == counts_csv
    queue_lines = queue_csv.read_text().splitlines()
    assert queue_lines[0] == ",".join(COLUMNS)
    assert queue_lines[1] == "1,Audi,10,A3,2014,A3,,,,,,,,,,,,,provisional,,,,"
    assert len(queue_lines) == 5
    assert counts_csv.read_text().splitlines() == [
        "make_id,model_id,compcars_year,image_count",
        "1,10,2014,3",
        "2,20,2008,2",
        "2,20,2010,1",
        "2,21,2012,1",
    ]


def test_reviewer_sheet_has_one_row_per_model_sorted_by_make_then_model(manifest):
    sheet = build_reviewer_sheet(build_queue(manifest))

    assert sheet[["make_name", "model_name", "years", "image_count"]].values.tolist() == [
        ["Audi", "A3", "2014", 3],
        ["Dacia", "Duster", "2012", 1],
        ["Dacia", "Logan", "2008;2010", 3],
    ]
    for column in (
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
    ):
        assert (sheet[column] == "").all(), column
