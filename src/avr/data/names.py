"""Make and model names from ``misc/make_model_name.mat`` (PROTOCOL.md §7).

The .mat file is the only source of truth for names. Both arrays are MATLAB cell columns
whose position ``i`` holds the name of id ``i + 1``; 288 model slots are empty (ids that
have no images). Ids are therefore exposed as 1-based dictionary keys, never as array
positions, and empty slots are recorded as missing instead of as empty strings.

Names are stored exactly as distributed in ``name_raw`` (the dataset writes
``"Lamorghini "`` and ``" Hector"``) and whitespace-trimmed in ``name``; spellings are
never corrected.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.io
from pydantic import BaseModel


class NameTable(BaseModel):
    """One name slot of the .mat file."""

    name_raw: str
    name: str


class NameTables(BaseModel):
    """1-based ``make_id -> name`` and ``model_id -> name`` with the empty slots recorded."""

    makes: dict[int, NameTable]
    models: dict[int, NameTable]
    make_slots: int
    model_slots: int
    missing_make_ids: tuple[int, ...]
    missing_model_ids: tuple[int, ...]


def _read_column(cells: np.ndarray) -> tuple[dict[int, NameTable], int, tuple[int, ...]]:
    names: dict[int, NameTable] = {}
    missing: list[int] = []
    for index, cell in enumerate(cells.ravel()):
        slot_id = index + 1
        if cell.size == 0:
            missing.append(slot_id)
            continue
        raw = str(cell.ravel()[0])
        names[slot_id] = NameTable(name_raw=raw, name=raw.strip())
    return names, cells.size, tuple(missing)


def load_name_tables(path: Path) -> NameTables:
    """Load ``make_names`` and ``model_names`` from a CompCars ``make_model_name.mat``."""
    mat = scipy.io.loadmat(path)
    makes, make_slots, missing_makes = _read_column(mat["make_names"])
    models, model_slots, missing_models = _read_column(mat["model_names"])
    return NameTables(
        makes=makes,
        models=models,
        make_slots=make_slots,
        model_slots=model_slots,
        missing_make_ids=missing_makes,
        missing_model_ids=missing_models,
    )
