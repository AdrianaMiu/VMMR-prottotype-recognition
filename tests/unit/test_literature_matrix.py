"""Literature comparison matrix: row schema, comparability rules, Markdown rendering (M6-1)."""

from datetime import date

import pytest
from pydantic import ValidationError

from avr.reporting.literature import (
    COLUMNS,
    TRACK_A_LABEL_LEVEL,
    TRACK_A_METRIC_FAMILY,
    TRACK_A_SPLIT_ID,
    LiteratureRow,
    render_markdown,
)

PROTOCOL_21_COLUMNS = (
    "paper",
    "year",
    "dataset",
    "dataset_subset",
    "split",
    "number_of_images",
    "number_of_classes",
    "label_level",
    "training_regime",
    "backbone",
    "input_resolution",
    "preprocessing",
    "metric_definition",
    "reported_result",
    "directly_comparable",
    "reason_if_not_comparable",
)


def comparable_row(**overrides) -> dict:
    row = {
        "paper": "Yang et al., A Large-Scale Car Dataset (CVPR 2015)",
        "year": 2015,
        "dataset": "CompCars",
        "dataset_subset": "web-nature Part-I, 431 models",
        "split": TRACK_A_SPLIT_ID,
        "number_of_images": "30,955 (14,939 test)",
        "number_of_classes": "431",
        "label_level": TRACK_A_LABEL_LEVEL,
        "training_regime": "fine-tuned",
        "backbone": "Overfeat",
        "input_resolution": "not stated",
        "preprocessing": "not stated",
        "metric_definition": "top-1 accuracy, all viewpoints",
        "reported_result": "0.767",
        "directly_comparable": True,
        "reason_if_not_comparable": "",
        "source_url": "https://arxiv.org/abs/1506.08959",
        "accessed_at": date(2026, 11, 9),
    }
    row.update(overrides)
    return row


def test_columns_are_protocol_21_plus_provenance():
    assert COLUMNS == PROTOCOL_21_COLUMNS + ("source_url", "accessed_at")


def test_comparable_row_is_valid():
    row = LiteratureRow(**comparable_row())
    assert row.directly_comparable is True
    assert TRACK_A_METRIC_FAMILY == "top-1 accuracy"


def test_row_without_source_url_is_invalid():
    with pytest.raises(ValidationError):
        LiteratureRow(**comparable_row(source_url=""))
    data = comparable_row()
    del data["source_url"]
    with pytest.raises(ValidationError):
        LiteratureRow(**data)


def test_source_url_must_be_http():
    with pytest.raises(ValidationError):
        LiteratureRow(**comparable_row(source_url="arxiv.org/abs/1506.08959"))


def test_directly_comparable_requires_track_a_split():
    with pytest.raises(ValidationError, match="split"):
        LiteratureRow(**comparable_row(split="compcars-official-arxiv-v2-70-30-52083"))


def test_directly_comparable_requires_make_model_label():
    with pytest.raises(ValidationError, match="label_level"):
        LiteratureRow(**comparable_row(label_level="make-model-year"))


def test_directly_comparable_requires_top1_metric_family():
    with pytest.raises(ValidationError, match="metric"):
        LiteratureRow(**comparable_row(metric_definition="macro F1 over novel classes"))


def test_directly_comparable_row_carries_no_reason():
    with pytest.raises(ValidationError, match="reason"):
        LiteratureRow(**comparable_row(reason_if_not_comparable="some reason"))


def test_not_comparable_row_requires_reason():
    with pytest.raises(ValidationError, match="reason"):
        LiteratureRow(**comparable_row(directly_comparable=False))
    row = LiteratureRow(
        **comparable_row(
            split="compcars-official-arxiv-v2-70-30-52083",
            directly_comparable=False,
            reason_if_not_comparable="different official split version",
        )
    )
    assert row.directly_comparable is False


def test_unknown_column_is_rejected():
    with pytest.raises(ValidationError):
        LiteratureRow(**comparable_row(venue="CVPR"))


def test_render_markdown_has_header_and_one_row_per_entry():
    rows = [
        LiteratureRow(**comparable_row()),
        LiteratureRow(
            **comparable_row(
                paper="Yu et al., EP-CNN",
                year=2022,
                split="compcars-official-arxiv-v2-70-30-52083",
                reported_result="98.6 % | note with a pipe",
                directly_comparable=False,
                reason_if_not_comparable="different official split version",
            )
        ),
    ]
    text = render_markdown(rows)
    lines = text.strip().splitlines()
    assert lines[0].startswith("| paper | year | dataset |")
    assert lines[0].endswith("| source_url | accessed_at |")
    assert lines[1].startswith("|---|")
    assert len(lines) == 2 + len(rows)
    assert "| yes |" in lines[2]
    assert "| no | different official split version |" in lines[3]
    assert "98.6 % \\| note with a pipe" in lines[3]
    assert "2026-11-09" in lines[2]


def _write_csv(path, rows):
    import csv

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def test_load_matrix_reads_yes_no_and_dates(tmp_path):
    from avr.reporting.literature import load_matrix

    path = tmp_path / "matrix.csv"
    _write_csv(
        path,
        [comparable_row(directly_comparable="yes", accessed_at="2026-11-09")],
    )
    rows = load_matrix(path)
    assert len(rows) == 1
    assert rows[0].directly_comparable is True
    assert rows[0].accessed_at == date(2026, 11, 9)


def test_load_matrix_fails_on_row_without_url_and_names_the_line(tmp_path):
    from avr.reporting.literature import load_matrix

    path = tmp_path / "matrix.csv"
    _write_csv(path, [comparable_row(), comparable_row(source_url="")])
    with pytest.raises(ValueError, match="line 3"):
        load_matrix(path)


def test_load_matrix_fails_on_comparable_row_with_other_split(tmp_path):
    from avr.reporting.literature import load_matrix

    path = tmp_path / "matrix.csv"
    _write_csv(path, [comparable_row(split="compcars-official-arxiv-v2-70-30-52083")])
    with pytest.raises(ValueError, match="split"):
        load_matrix(path)


def test_load_matrix_rejects_wrong_header(tmp_path):
    from avr.reporting.literature import load_matrix

    path = tmp_path / "matrix.csv"
    path.write_text("paper,year\nx,2015\n", encoding="utf-8")
    with pytest.raises(ValueError, match="header"):
        load_matrix(path)
