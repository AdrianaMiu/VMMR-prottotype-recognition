"""Literature comparison matrix with comparability flags (PROTOCOL.md §21, issue M6-1).

The matrix lives in ``reports/literature/matrix.csv``. Every row is validated by
:class:`LiteratureRow`; the columns are exactly the §21 columns plus two provenance columns
(``source_url``, ``accessed_at``). A row may be marked ``directly_comparable`` to Track A only
when it reports on the same split (:data:`TRACK_A_SPLIT_ID`), the same class definition
(:data:`TRACK_A_LABEL_LEVEL`) and the same metric family (:data:`TRACK_A_METRIC_FAMILY`),
following §21 rules 1, 2 and 6. Any other row must carry a reason.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable, Sequence
from datetime import date
from pathlib import Path

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

#: Identifier of the Track A setting (docs/protocols/track-a-split-identity.md).
TRACK_A_SPLIT_ID = "compcars-official-cvpr2015-half-half-30955"
#: Identifier of the other official CompCars setting (arXiv v2, 52,083 images, 70/30).
ARXIV_V2_SPLIT_ID = "compcars-official-arxiv-v2-70-30-52083"
#: Track A class definition (GLOSSARY.md, Track A).
TRACK_A_LABEL_LEVEL = "make-model"
#: Track A primary outcome metric family; ``metric_definition`` must start with it.
TRACK_A_METRIC_FAMILY = "top-1 accuracy"
#: Registered reason for rows on the arXiv v2 70/30 setting.
REASON_DIFFERENT_SPLIT_VERSION = "different official split version"

PROTOCOL_COLUMNS: tuple[str, ...] = (
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
PROVENANCE_COLUMNS: tuple[str, ...] = ("source_url", "accessed_at")
COLUMNS: tuple[str, ...] = PROTOCOL_COLUMNS + PROVENANCE_COLUMNS

_TEXT_FIELDS = (
    "paper",
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
)


class LiteratureRow(BaseModel):
    """One literature result with its comparability verdict against Track A."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    paper: str
    year: int
    dataset: str
    dataset_subset: str
    split: str
    number_of_images: str
    number_of_classes: str
    label_level: str
    training_regime: str
    backbone: str
    input_resolution: str
    preprocessing: str
    metric_definition: str
    reported_result: str
    directly_comparable: bool
    reason_if_not_comparable: str = ""
    source_url: str
    accessed_at: date

    @field_validator(*_TEXT_FIELDS)
    @classmethod
    def _non_empty(cls, value: str) -> str:
        if not value:
            raise ValueError("field must not be empty (write UNVERIFIED or not stated)")
        return value

    @field_validator("source_url")
    @classmethod
    def _url(cls, value: str) -> str:
        if not value.startswith(("http://", "https://")):
            raise ValueError("source_url must be an http(s) URL; a row without a URL is invalid")
        return value

    @model_validator(mode="after")
    def _comparability(self) -> LiteratureRow:
        if self.directly_comparable:
            problems = []
            if self.split != TRACK_A_SPLIT_ID:
                problems.append(f"split must be {TRACK_A_SPLIT_ID!r}, got {self.split!r}")
            if self.label_level != TRACK_A_LABEL_LEVEL:
                problems.append(
                    f"label_level must be {TRACK_A_LABEL_LEVEL!r}, got {self.label_level!r}"
                )
            if not self.metric_definition.lower().startswith(TRACK_A_METRIC_FAMILY):
                problems.append(
                    f"metric_definition must start with {TRACK_A_METRIC_FAMILY!r}, "
                    f"got {self.metric_definition!r}"
                )
            if self.reason_if_not_comparable:
                problems.append("reason_if_not_comparable must be empty when comparable")
            if problems:
                raise ValueError("directly_comparable=true but " + "; ".join(problems))
        elif not self.reason_if_not_comparable:
            raise ValueError("reason_if_not_comparable is required when not comparable")
        return self


def _parse_bool(value: str) -> bool:
    key = value.strip().lower()
    if key in {"true", "yes", "1"}:
        return True
    if key in {"false", "no", "0"}:
        return False
    raise ValueError(f"directly_comparable must be yes/no, got {value!r}")


def load_matrix(path: Path) -> list[LiteratureRow]:
    """Read and validate ``matrix.csv``; the header must be exactly :data:`COLUMNS`."""
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != COLUMNS:
            raise ValueError(f"{path}: header must be {list(COLUMNS)}, got {reader.fieldnames}")
        rows = []
        for index, record in enumerate(reader, start=2):
            record["directly_comparable"] = _parse_bool(record["directly_comparable"])
            try:
                rows.append(LiteratureRow(**record))
            except ValueError as exc:
                raise ValueError(f"{path} line {index}: {exc}") from exc
    return rows


def _cell(value: object) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    return str(value).replace("|", "\\|").replace("\n", " ")


def render_markdown(rows: Iterable[LiteratureRow]) -> str:
    """Render the rows as one Markdown table with the :data:`COLUMNS` header."""
    lines = ["| " + " | ".join(COLUMNS) + " |", "|" + "---|" * len(COLUMNS)]
    for row in rows:
        values = row.model_dump()
        lines.append("| " + " | ".join(_cell(values[column]) for column in COLUMNS) + " |")
    return "\n".join(lines) + "\n"


def comparable_rows(rows: Sequence[LiteratureRow]) -> list[LiteratureRow]:
    return [row for row in rows if row.directly_comparable]


CONSULTED_WITHOUT_ROW: tuple[tuple[str, str, str], ...] = (
    (
        "Gayen et al. Two decades of vehicle make and model recognition (JKSU-CIS 36(1), 2024)",
        "survey; no result of its own",
        "https://doi.org/10.1016/j.jksuci.2023.101885",
    ),
    (
        "Lima et al. Toward Unified Fine-Grained Vehicle Classification and ALPR "
        "(arXiv:2604.05271)",
        "UFPR-VeSV dataset paper; the frozen-encoder numbers on it are the Delazeri et al. row",
        "https://arxiv.org/abs/2604.05271",
    ),
    (
        "Kerdvibulvech. Multimodal AI model for zero-shot vehicle brand identification (MTAP 2025)",
        "make-only task; abstract and body not retrievable (login wall), content UNVERIFIED",
        "https://doi.org/10.1007/s11042-024-20559-3",
    ),
    (
        "Chen, Hsieh, Yan, Chen. VMMR using sparse representation and symmetrical SURFs (PR 2015)",
        "NTOU-MMR dataset paper; its 91.10 % figure was only reachable through Jamil et al. 2020",
        "https://doi.org/10.1016/j.patcog.2014.12.018",
    ),
    (
        "Picek et al. FungiTastic (arXiv:2408.13632)",
        "not a vehicle dataset; cited only as evidence that frozen-encoder centroid prototypes "
        "are an established baseline",
        "https://arxiv.org/abs/2408.13632",
    ),
    (
        "Goswami et al. Project and Mix (arXiv:2603.24528)",
        "Stanford Cars image-prototype numbers appear only in figures; the Li et al. row "
        "carries the tabulated nearest-class-mean baseline",
        "https://arxiv.org/abs/2603.24528",
    ),
    (
        "Miu (Pavel), Ionescu. Can we recognize Cars we've never seen? (AI4MM 2025)",
        "survey plus proposed pipeline; no results table",
        "https://aimultimedialab.ro/ai4mm-workshop-2/6_Miu_Pavel_Adriana_AIM_Workshop_2025.pdf",
    ),
    (
        "VehicleDINO model card (wms2537)",
        "not peer reviewed; LoRA adapters, so the encoder is not frozen; numbers UNVERIFIED",
        "https://huggingface.co/wms2537/VehicleDINO",
    ),
    (
        "Kumar, Vykunth, Jayanthi. Real-time smart vehicle surveillance system (arXiv:2111.12289)",
        "repeats the Wille et al. 2018 MobileNet numbers on the 70/30 setting; not independent",
        "https://arxiv.org/abs/2111.12289",
    ),
)


def render_readme(rows: Sequence[LiteratureRow]) -> str:
    """Render ``reports/literature/README.md`` entirely from the validated rows."""
    comparable = comparable_rows(rows)
    seventy_thirty = [r for r in rows if r.split == ARXIV_V2_SPLIT_ID]
    unverified = [r for r in rows if "UNVERIFIED" in r.reported_result]
    lines = [
        "# Literature comparison matrix (PROTOCOL.md §21)",
        "",
        "Rendered from `matrix.csv` by `avr.reporting.literature`; do not edit by hand. "
        "`tests/regression/test_literature_matrix_file.py` validates every row and fails when "
        "this file drifts from the CSV, when a row has no source URL, or when a row is marked "
        "directly comparable on any split other than Track A.",
        "",
        f"Rows: {len(rows)}. Directly comparable to Track A: {len(comparable)}. "
        f"Rows on the arXiv v2 70/30 setting: {len(seventy_thirty)}. "
        f"Rows whose reported result is UNVERIFIED: {len(unverified)}.",
        "",
        "## How to read `directly_comparable`",
        "",
        "Track A is the official CVPR 2015 half/half classification split of CompCars "
        "(30,955 images, 431 make-model classes; `docs/protocols/track-a-split-identity.md`), "
        f"identified in the `split` column as `{TRACK_A_SPLIT_ID}`. A row is `yes` only when "
        "all three hold (§21 rules 1, 2 and 6):",
        "",
        f"1. `split` is exactly `{TRACK_A_SPLIT_ID}`;",
        f"2. `label_level` is `{TRACK_A_LABEL_LEVEL}` (the 431-model class definition);",
        f"3. `metric_definition` starts with `{TRACK_A_METRIC_FAMILY}`, "
        "the Track A primary outcome.",
        "",
        "Every other row is `no` and carries `reason_if_not_comparable`. CompCars results on "
        f"the arXiv v2 setting (52,083 images, 70/30; `{ARXIV_V2_SPLIT_ID}`) are flagged "
        f"`{REASON_DIFFERENT_SPLIT_VERSION}`; they are the majority of published 431-model "
        "numbers (EP-CNN and most later work) and must never be placed in the same column as a "
        "Track A number. Buzzelli & Segantin use their own feature-distance split over all "
        "136,725 images at make-model-year granularity and are never directly comparable; their "
        "leakage finding is why the duplicate audit is reported next to every Track A number. "
        "Track B (generation label) has no external comparator; the generation, few-shot and "
        "cross-view rows are descriptive context only (§21 rule 3).",
        "",
        "`training_regime` separates frozen prototypes, frozen linear probes, fine-tuned models "
        "and bespoke supervised architectures (§21 rule 4); comparability does not depend on it, "
        "so a fine-tuned result on the Track A split is `yes` and must be read together with "
        "that column.",
        "",
        "Every number comes from the cited paper (abstract or tables) read on `accessed_at`; "
        "`UNVERIFIED` marks fields the full text did not confirm on that date, and `not stated` "
        "marks fields the paper does not give.",
        "",
        "## Directly comparable rows",
        "",
    ]
    for r in comparable:
        lines.append(f"- {r.paper} ({r.year}): {r.reported_result}")
    lines += ["", "## Matrix", "", render_markdown(rows).rstrip("\n"), ""]
    lines += [
        "## Sources consulted without a row",
        "",
        "Seed sources from `docs/research/generation-recognition-gap.md` that report no vehicle "
        "result usable in the matrix, or that could not be verified against the paper itself.",
        "",
        "| source | why no row | url |",
        "|---|---|---|",
    ]
    for source, why, url in CONSULTED_WITHOUT_ROW:
        lines.append(f"| {_cell(source)} | {_cell(why)} | {url} |")
    lines.append("")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m avr.reporting.literature [matrix.csv] [README.md]``: validate and render."""
    args = list(argv) if argv is not None else []
    matrix = Path(args[0]) if args else Path("reports/literature/matrix.csv")
    readme = Path(args[1]) if len(args) > 1 else matrix.with_name("README.md")
    rows = load_matrix(matrix)
    readme.write_text(render_readme(rows), encoding="utf-8")
    print(f"{len(rows)} rows, {len(comparable_rows(rows))} directly comparable -> {readme}")
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(main(sys.argv[1:]))
