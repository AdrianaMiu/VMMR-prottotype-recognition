"""misc/make_model_name.mat -> 1-based name tables (PROTOCOL.md §7, issue M1-7)."""

from pathlib import Path

import pytest

from avr.data.names import NameTable, load_name_tables
from conftest import write_name_mat


@pytest.fixture
def tiny_mat(tmp_path: Path) -> Path:
    return write_name_mat(
        tmp_path / "make_model_name.mat",
        makes=["ABT", "Lamorghini ", "Hyundai"],
        models=["Audi A3 hatchback", None, " Hector"],
    )


def test_ids_are_one_based_dictionary_keys(tiny_mat: Path):
    tables = load_name_tables(tiny_mat)

    assert list(tables.makes) == [1, 2, 3]
    assert list(tables.models) == [1, 3]
    assert tables.makes[1] == NameTable(name_raw="ABT", name="ABT")
    assert tables.models[3].name == "Hector"


def test_whitespace_is_kept_in_name_raw_and_stripped_in_name(tiny_mat: Path):
    tables = load_name_tables(tiny_mat)

    assert tables.makes[2].name_raw == "Lamorghini "
    assert tables.makes[2].name == "Lamorghini"


def test_empty_slot_is_recorded_as_missing(tiny_mat: Path):
    tables = load_name_tables(tiny_mat)

    assert 2 not in tables.models
    assert tables.missing_model_ids == (2,)
    assert tables.model_slots == 3
