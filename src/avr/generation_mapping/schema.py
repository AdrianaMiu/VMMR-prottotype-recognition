"""Generation-mapping row schema and validation rules (PROTOCOL.md §6).

One row maps a CompCars model-year class ``(make_id, model_id, compcars_year)`` to a
generation. The facelift phase is its own field (§6 rule 6); ``generation_id`` is a
stable identifier independent of display names (rule 7); uncertain rows are
``ambiguous``, never ``approved`` (rule 8); an approved row carries two sources and two
distinct human reviewers (rules 2 and 10). Reviewers are not yet named, so every row
defaults to ``provisional``.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import date
from itertools import combinations
from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field, field_validator, model_validator

MappingStatus = Literal["provisional", "ambiguous", "approved"]
FaceliftPhase = Literal["pre_facelift", "facelift", "none"]

GENERATION_ID_PATTERN = r"^[a-z0-9]+(?:_[a-z0-9]+)*_g[1-9][0-9]*$"

OPTIONAL_TEXT_FIELDS = (
    "body_style_or_variant",
    "market_region",
    "source_url_1",
    "source_url_2",
    "source_title_1",
    "source_title_2",
    "reviewer_1",
    "reviewer_2",
    "review_notes",
    "ambiguity_code",
)


class GenerationMappingRow(BaseModel):
    """One generation-mapping row; field order is the §6 column order."""

    make_id: int
    make_name: str
    model_id: int
    model_name: str
    compcars_year: int
    canonical_model_name: str
    generation_id: str = Field(pattern=GENERATION_ID_PATTERN)
    generation_name: str
    generation_start_year: int
    generation_end_year: int
    facelift_phase: FaceliftPhase = "none"
    body_style_or_variant: str | None = None
    market_region: str | None = None
    source_url_1: str | None = None
    source_url_2: str | None = None
    source_title_1: str | None = None
    source_title_2: str | None = None
    accessed_at: date | None = None
    mapping_status: MappingStatus = "provisional"
    reviewer_1: str | None = None
    reviewer_2: str | None = None
    review_notes: str | None = None
    ambiguity_code: str | None = None

    @field_validator(*OPTIONAL_TEXT_FIELDS, "accessed_at", mode="before")
    @classmethod
    def _empty_text_is_absent(cls, value: object) -> object:
        if isinstance(value, str) and value.strip() == "":
            return None
        return value

    @model_validator(mode="after")
    def _years_are_ordered(self) -> GenerationMappingRow:
        if self.generation_start_year > self.generation_end_year:
            raise ValueError(
                f"generation_start_year {self.generation_start_year} is after "
                f"generation_end_year {self.generation_end_year}"
            )
        if not self.generation_start_year <= self.compcars_year <= self.generation_end_year:
            raise ValueError(
                f"compcars_year {self.compcars_year} lies outside generation interval "
                f"[{self.generation_start_year}, {self.generation_end_year}]"
            )
        return self

    @model_validator(mode="after")
    def _approved_rows_carry_provenance_and_two_reviewers(self) -> GenerationMappingRow:
        if self.mapping_status != "approved":
            return self
        required = ("source_url_1", "source_url_2", "accessed_at", "reviewer_1", "reviewer_2")
        missing = [name for name in required if getattr(self, name) is None]
        if missing:
            raise ValueError(f"approved row is missing {', '.join(missing)}")
        if self.reviewer_1 == self.reviewer_2:
            raise ValueError("approved row needs two distinct reviewers")
        return self


class MappingTableError(ValueError):
    """A set of rows breaks a rule that only holds across rows (§6 rule 9)."""


def validate_table(rows: list[GenerationMappingRow]) -> list[GenerationMappingRow]:
    """Check cross-row rules and return ``rows`` unchanged.

    Within one ``(make_id, model_id)``, every generation has one interval, and two
    generations may overlap in years only when both are ``ambiguous``; an ``approved``
    generation never overlaps another.
    """
    by_model: dict[tuple[int, int], dict[str, list[GenerationMappingRow]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        by_model[(row.make_id, row.model_id)][row.generation_id].append(row)

    for (make_id, model_id), generations in by_model.items():
        for generation_id, members in generations.items():
            intervals = {(m.generation_start_year, m.generation_end_year) for m in members}
            if len(intervals) > 1:
                raise MappingTableError(
                    f"{generation_id} (make {make_id}, model {model_id}) has more than one "
                    f"interval: {sorted(intervals)}"
                )
        for (id_a, rows_a), (id_b, rows_b) in combinations(generations.items(), 2):
            overlap = _overlap_years(rows_a[0], rows_b[0])
            if overlap is None:
                continue
            statuses = {r.mapping_status for r in rows_a + rows_b}
            if statuses == {"ambiguous"}:
                continue
            why = "approved rows must not overlap" if "approved" in statuses else "not ambiguous"
            raise MappingTableError(
                f"{id_a} and {id_b} (make {make_id}, model {model_id}) overlap in years "
                f"{overlap[0]}-{overlap[1]}: {why}"
            )
    return rows


def _overlap_years(a: GenerationMappingRow, b: GenerationMappingRow) -> tuple[int, int] | None:
    start = max(a.generation_start_year, b.generation_start_year)
    end = min(a.generation_end_year, b.generation_end_year)
    return (start, end) if start <= end else None


COLUMNS = tuple(GenerationMappingRow.model_fields)


def rows_to_frame(rows: list[GenerationMappingRow]) -> pd.DataFrame:
    """Rows as a DataFrame in §6 column order; dates as ISO strings, absent text as null."""
    records = [row.model_dump(mode="json") for row in rows]
    return pd.DataFrame.from_records(records, columns=list(COLUMNS))


def frame_to_rows(frame: pd.DataFrame) -> list[GenerationMappingRow]:
    """Validate every row, then the table; raises on the first broken rule."""
    clean = frame.astype(object).where(frame.notna(), None)
    rows = [GenerationMappingRow(**record) for record in clean.to_dict(orient="records")]
    return validate_table(rows)


def write_csv(rows: list[GenerationMappingRow], path: Path) -> None:
    rows_to_frame(rows).to_csv(path, index=False)


def read_csv(path: Path) -> list[GenerationMappingRow]:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    return frame_to_rows(frame)


def write_parquet(rows: list[GenerationMappingRow], path: Path) -> None:
    rows_to_frame(rows).to_parquet(path, index=False)


def read_parquet(path: Path) -> list[GenerationMappingRow]:
    return frame_to_rows(pd.read_parquet(path))
