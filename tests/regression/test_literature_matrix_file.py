"""CI check for ``reports/literature/matrix.csv`` (PROTOCOL.md §21, issue M6-1).

Fails when any row lacks a source URL, when a row is marked directly comparable on a split
other than Track A, or when the committed README drifts from the CSV.
"""

from pathlib import Path

from avr.reporting.literature import (
    TRACK_A_SPLIT_ID,
    comparable_rows,
    load_matrix,
    render_readme,
)

ROOT = Path(__file__).resolve().parents[2]
MATRIX = ROOT / "reports" / "literature" / "matrix.csv"
README = ROOT / "reports" / "literature" / "README.md"


def test_matrix_validates_and_every_row_has_a_url():
    rows = load_matrix(MATRIX)
    assert 20 <= len(rows) <= 45
    assert all(row.source_url.startswith("https://") for row in rows)


def test_comparable_rows_are_on_the_track_a_split_only():
    rows = load_matrix(MATRIX)
    for row in comparable_rows(rows):
        assert row.split == TRACK_A_SPLIT_ID
        assert row.label_level == "make-model"
    # Hand count from the CSV: Yang 2015 (half/half all-view), Watkins 2018, Liu 2024.
    assert len(comparable_rows(rows)) == 3


def test_every_70_30_row_carries_the_registered_reason():
    rows = load_matrix(MATRIX)
    seventy_thirty = [r for r in rows if r.split == "compcars-official-arxiv-v2-70-30-52083"]
    assert seventy_thirty, "the 70/30 setting must be represented"
    for row in seventy_thirty:
        assert row.directly_comparable is False
        assert row.reason_if_not_comparable == "different official split version"


def test_readme_matches_rendering_of_the_csv():
    assert README.read_text(encoding="utf-8") == render_readme(load_matrix(MATRIX))
