from datetime import date
from pathlib import Path

import pytest
from pydantic import ValidationError

from avr.generation_mapping.schema import (
    GenerationMappingRow,
    MappingTableError,
    read_csv,
    read_parquet,
    validate_table,
    write_csv,
    write_parquet,
)

PROTOCOL_COLUMNS = [
    "make_id",
    "make_name",
    "model_id",
    "model_name",
    "compcars_year",
    "canonical_model_name",
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
    "mapping_status",
    "reviewer_1",
    "reviewer_2",
    "review_notes",
    "ambiguity_code",
]


def make_row(**overrides) -> GenerationMappingRow:
    base = dict(
        make_id=999,
        make_name="Dacia",
        model_id=9991,
        model_name="Logan",
        compcars_year=2010,
        canonical_model_name="Dacia Logan",
        generation_id="dacia_logan_g1",
        generation_name="Logan I",
        generation_start_year=2004,
        generation_end_year=2012,
    )
    base.update(overrides)
    return GenerationMappingRow(**base)


def test_row_carries_exactly_the_protocol_section_6_columns_in_order():
    assert list(GenerationMappingRow.model_fields) == PROTOCOL_COLUMNS


def test_minimal_row_defaults_to_provisional_with_no_facelift_phase():
    row = make_row()

    assert row.mapping_status == "provisional"
    assert row.facelift_phase == "none"
    assert row.reviewer_1 is None
    assert row.reviewer_2 is None


@pytest.mark.parametrize(
    "bad_id",
    ["Dacia_Logan_g2", "dacia-logan-g2", "dacia_logan_2", "dacia_logan_g0", "dacia_logan_g", "g2"],
)
def test_generation_id_must_be_lowercase_snake_make_model_g_n(bad_id: str):
    with pytest.raises(ValidationError, match="generation_id"):
        make_row(generation_id=bad_id)


def test_generation_id_accepts_the_protocol_example():
    assert make_row(generation_id="dacia_logan_g2").generation_id == "dacia_logan_g2"


def test_generation_start_year_after_end_year_is_rejected():
    with pytest.raises(ValidationError, match="generation_start_year"):
        make_row(generation_start_year=2013, generation_end_year=2012, compcars_year=2012)


def test_single_year_generation_is_allowed():
    row = make_row(generation_start_year=2010, generation_end_year=2010)

    assert row.generation_start_year == row.generation_end_year == 2010


@pytest.mark.parametrize("year", [2003, 2013])
def test_compcars_year_outside_generation_interval_is_rejected(year: int):
    with pytest.raises(ValidationError, match="compcars_year"):
        make_row(compcars_year=year)


APPROVED = dict(
    mapping_status="approved",
    source_url_1="https://example.org/archive/logan-i",
    source_url_2="https://example.org/catalogue/logan-2004",
    source_title_1="Example archive: Logan I",
    source_title_2="Example catalogue 2004",
    accessed_at="2026-11-06",
    reviewer_1="reviewer_a",
    reviewer_2="reviewer_b",
)


def test_fully_sourced_and_dual_reviewed_row_can_be_approved():
    row = make_row(**APPROVED)

    assert row.mapping_status == "approved"
    assert row.accessed_at.isoformat() == "2026-11-06"


def test_approved_without_reviewer_2_is_invalid():
    with pytest.raises(ValidationError, match="reviewer_2"):
        make_row(**{**APPROVED, "reviewer_2": None})


def test_approved_with_the_same_person_as_both_reviewers_is_invalid():
    with pytest.raises(ValidationError, match="distinct"):
        make_row(**{**APPROVED, "reviewer_2": "reviewer_a"})


@pytest.mark.parametrize("missing", ["source_url_1", "source_url_2", "reviewer_1", "accessed_at"])
def test_approved_without_each_required_provenance_field_is_invalid(missing: str):
    with pytest.raises(ValidationError, match=missing):
        make_row(**{**APPROVED, missing: None})


def test_provisional_row_does_not_need_sources_or_reviewers():
    row = make_row(mapping_status="provisional")

    assert row.source_url_1 is None and row.accessed_at is None


def test_empty_strings_in_optional_text_fields_read_as_absent():
    row = make_row(review_notes="", reviewer_2="")

    assert row.review_notes is None
    assert row.reviewer_2 is None


G1 = dict(generation_id="dacia_logan_g1", generation_start_year=2004, generation_end_year=2012)
G2 = dict(generation_id="dacia_logan_g2", generation_start_year=2013, generation_end_year=2020)
G2_EARLY = {**G2, "generation_start_year": 2012}


def test_disjoint_generations_within_a_model_pass_table_validation():
    rows = [make_row(**G1, compcars_year=2010), make_row(**G2, compcars_year=2014)]

    assert validate_table(rows) == rows


def test_overlap_between_provisional_generations_is_rejected():
    rows = [make_row(**G1, compcars_year=2010), make_row(**G2_EARLY, compcars_year=2014)]

    with pytest.raises(MappingTableError, match="dacia_logan_g1.*dacia_logan_g2"):
        validate_table(rows)


def test_overlap_between_approved_generations_is_rejected():
    rows = [
        make_row(**G1, **APPROVED, compcars_year=2010),
        make_row(**G2_EARLY, **APPROVED, compcars_year=2014),
    ]

    with pytest.raises(MappingTableError, match="2012"):
        validate_table(rows)


def test_overlap_between_an_approved_and_an_ambiguous_generation_is_rejected():
    rows = [
        make_row(**G1, **APPROVED, compcars_year=2010),
        make_row(**G2_EARLY, compcars_year=2014, mapping_status="ambiguous"),
    ]

    with pytest.raises(MappingTableError, match="approved"):
        validate_table(rows)


def test_overlap_is_tolerated_when_both_rows_are_marked_ambiguous():
    rows = [
        make_row(**G1, compcars_year=2010, mapping_status="ambiguous"),
        make_row(**G2_EARLY, compcars_year=2014, mapping_status="ambiguous"),
    ]

    assert validate_table(rows) == rows


def test_overlap_between_ambiguous_and_provisional_rows_is_rejected():
    rows = [
        make_row(**G1, compcars_year=2010, mapping_status="ambiguous"),
        make_row(**G2_EARLY, compcars_year=2014),
    ]

    with pytest.raises(MappingTableError):
        validate_table(rows)


def test_overlapping_intervals_of_different_models_are_independent():
    rows = [
        make_row(**G1, compcars_year=2010),
        make_row(**G2_EARLY, compcars_year=2014, model_id=9992, model_name="Sandero"),
    ]

    assert validate_table(rows) == rows


def test_rows_of_the_same_generation_do_not_count_as_overlapping():
    rows = [make_row(**G1, compcars_year=2010), make_row(**G1, compcars_year=2011)]

    assert validate_table(rows) == rows


def test_rows_of_one_generation_must_agree_on_its_interval():
    rows = [
        make_row(**G1, compcars_year=2010),
        make_row(**{**G1, "generation_end_year": 2011}, compcars_year=2010),
    ]

    with pytest.raises(MappingTableError, match="interval"):
        validate_table(rows)


FIXTURE = Path(__file__).parents[1] / "fixtures" / "generation_mapping" / "three_rows.csv"


def test_fixture_csv_reads_into_three_validated_rows():
    rows = read_csv(FIXTURE)

    assert [r.generation_id for r in rows] == [
        "dacia_logan_g1",
        "dacia_logan_g2",
        "dacia_sandero_g2",
    ]
    assert rows[0].facelift_phase == "pre_facelift"
    assert rows[0].source_url_1 is None
    assert rows[1].accessed_at == date(2026, 11, 6)
    assert rows[1].review_notes == "second source still missing; see notes, pending"
    assert rows[2].mapping_status == "ambiguous"
    assert rows[2].market_region is None
    assert rows[2].ambiguity_code == "cross_market_year_conflict"


def test_csv_round_trip_is_lossless(tmp_path: Path):
    rows = read_csv(FIXTURE)
    out = tmp_path / "rows.csv"

    write_csv(rows, out)

    assert read_csv(out) == rows
    assert out.read_text().splitlines()[0] == FIXTURE.read_text().splitlines()[0]


def test_parquet_round_trip_is_lossless(tmp_path: Path):
    rows = read_csv(FIXTURE)
    out = tmp_path / "rows.parquet"

    write_parquet(rows, out)

    assert read_parquet(out) == rows


def test_reading_a_table_applies_the_cross_row_rules(tmp_path: Path):
    rows = [make_row(**G1, compcars_year=2010), make_row(**G2_EARLY, compcars_year=2014)]
    out = tmp_path / "rows.csv"
    write_csv(rows, out)

    with pytest.raises(MappingTableError):
        read_csv(out)
