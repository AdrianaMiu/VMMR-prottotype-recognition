"""Coverage report for the curation queue (PROTOCOL.md §6 "Mapping quality report")."""

import hashlib
from pathlib import Path

import pandas as pd
import pytest

from avr.generation_mapping.coverage import (
    coverage_stats,
    flag_models,
    render_report,
)
from avr.generation_mapping.queue import build_queue, write_queue


def manifest_row(make_id, make_name, model_id, model_name, year, status="valid", reason=None):
    return dict(
        make_id=make_id,
        make_name=make_name,
        model_id=model_id,
        model_name=model_name,
        year=year,
        validity_status=status,
        exclusion_reason=reason,
    )


@pytest.fixture
def manifest() -> pd.DataFrame:
    # Valid model-years (images):
    #   Dacia/Logan(20):  2004 (1), 2006 (2), 2010 (1)   -> span 7 inclusive, gap 4
    #   Dacia/Duster(21): 2010 (1), 2011 (1)             -> span 2, gap 1
    #   Audi/A3(10):      2008 (1), 2012 (1)             -> span 5, gap 4
    #   Audi/A4(11):      2009..2015, one image each (7) -> span 7, gap 1
    # Excluded: one Logan unknown_year, one A3 degenerate_bbox, and Audi/A6(12) whose only
    # image has an unknown year (so the model never reaches the queue).
    rows = [
        manifest_row(2, "Dacia", 20, "Logan", 2004),
        manifest_row(2, "Dacia", 20, "Logan", 2006),
        manifest_row(2, "Dacia", 20, "Logan", 2006),
        manifest_row(2, "Dacia", 20, "Logan", 2010),
        manifest_row(2, "Dacia", 21, "Duster", 2010),
        manifest_row(2, "Dacia", 21, "Duster", 2011),
        manifest_row(1, "Audi", 10, "A3", 2008),
        manifest_row(1, "Audi", 10, "A3", 2012),
        *[manifest_row(1, "Audi", 11, "A4", y) for y in range(2009, 2016)],
        manifest_row(2, "Dacia", 20, "Logan", None, "excluded", "unknown_year"),
        manifest_row(1, "Audi", 10, "A3", 2012, "excluded", "degenerate_bbox"),
        manifest_row(1, "Audi", 12, "A6", None, "excluded", "unknown_year"),
    ]
    frame = pd.DataFrame(rows)
    frame["year"] = frame["year"].astype("float64")
    return frame


def test_flag_models_by_span_and_gap(manifest):
    flagged = flag_models(build_queue(manifest), span_min=7, gap_min=3)

    assert flagged[["model_id", "years", "span_years", "max_gap", "reasons"]].values.tolist() == [
        [10, "2008;2012", 5, 4, "gap"],
        [11, "2009;2010;2011;2012;2013;2014;2015", 7, 1, "span"],
        [20, "2004;2006;2010", 7, 4, "span;gap"],
    ]


def test_coverage_stats_counts_records_images_statuses_and_hypotheses(manifest):
    stats = coverage_stats(manifest, build_queue(manifest))

    assert stats.model_year_records == {"total": 14, "mapped": 0, "unmapped": 14}
    assert stats.images == {"total": 18, "retained": 15, "excluded": 3}
    assert stats.exclusion_reason_counts == {"degenerate_bbox": 1, "unknown_year": 2}
    assert stats.status_counts == {"provisional": 14}
    assert stats.model_count == 4
    assert stats.models_without_valid_model_year == 1
    assert stats.make_count == 2
    assert stats.classes_if_every_year_is_a_generation == 14
    assert stats.classes_if_all_years_merged == 4
    assert [m.model_id for m in stats.flagged_models] == [10, 11, 20]
    assert [m.model_id for m in stats.top_models] == [11, 20, 10, 21]
    assert [(m.model_id, m.image_count, m.years) for m in stats.top_models[:2]] == [
        (11, 7, "2009;2010;2011;2012;2013;2014;2015"),
        (20, 4, "2004;2006;2010"),
    ]


def test_render_report_states_version_checksum_and_no_invented_intervals(manifest, tmp_path: Path):
    queue = build_queue(manifest)
    out = write_queue(queue, tmp_path, version="0.1")
    expected_digest = hashlib.sha256(out.queue.read_bytes()).hexdigest()

    text = render_report(coverage_stats(manifest, queue), out.queue, version="0.1.0")

    assert "mapping version: 0.1.0" in text
    assert f"sha256: {expected_digest}  {out.queue.name}" in text
    assert "| provisional | 14 |" in text
    assert "no generation interval" in text.lower()
    assert "| Dacia | Logan | 20 | 2004;2006;2010 | 7 | 4 | span;gap |" in text
