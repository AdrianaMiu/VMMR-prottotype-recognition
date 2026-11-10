"""Mapping coverage report (PROTOCOL.md §6 "Mapping quality report").

Computed from the image manifest and the curation queue: mapped/unmapped model-year
records, images retained/excluded, mapping-status distribution, the candidate class count
under two bracketing hypotheses (every CompCars year is its own generation; all years of a
model form one generation), the models whose year list suggests more than one generation
(span >= ``SPAN_MIN`` calendar years inclusive, or a gap >= ``GAP_MIN`` years between
consecutive recorded years), the largest models by image count, and the queue checksum
with the mapping's semantic version. The flags are review hints only; no generation
interval is derived here.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd
from pydantic import BaseModel

from avr.generation_mapping.queue import IMAGE_COUNT
from avr.io import sha256_of_file

SPAN_MIN = 7
GAP_MIN = 3
TOP_MODELS = 30
MODEL_KEYS = ["make_id", "make_name", "model_id", "model_name"]


class ModelSummary(BaseModel):
    make_id: int
    make_name: str
    model_id: int
    model_name: str
    years: str
    year_count: int
    image_count: int
    span_years: int
    max_gap: int
    reasons: str


class CoverageStats(BaseModel):
    model_year_records: dict[str, int]
    images: dict[str, int]
    exclusion_reason_counts: dict[str, int]
    status_counts: dict[str, int]
    make_count: int
    model_count: int
    models_without_valid_model_year: int
    classes_if_every_year_is_a_generation: int
    classes_if_all_years_merged: int
    span_min: int
    gap_min: int
    flagged_models: list[ModelSummary]
    top_models: list[ModelSummary]


def _per_model(queue: pd.DataFrame) -> pd.DataFrame:
    """One row per model: sorted year list, inclusive span, largest gap between years."""

    def summarise(group: pd.DataFrame) -> pd.Series:
        years = sorted(set(group["compcars_year"].astype(int)))
        gaps = [b - a for a, b in zip(years[:-1], years[1:], strict=True)]
        return pd.Series(
            {
                "years": ";".join(str(y) for y in years),
                "year_count": len(years),
                IMAGE_COUNT: int(group[IMAGE_COUNT].sum()),
                "span_years": years[-1] - years[0] + 1,
                "max_gap": max(gaps, default=0),
            }
        )

    if queue.empty:
        return pd.DataFrame(
            columns=[*MODEL_KEYS, "years", "year_count", IMAGE_COUNT, "span_years", "max_gap"]
        )
    per_model = (
        queue.groupby(MODEL_KEYS, sort=True).apply(summarise, include_groups=False).reset_index()
    )
    for column in ("year_count", IMAGE_COUNT, "span_years", "max_gap"):
        per_model[column] = per_model[column].astype("int64")
    return per_model.sort_values(["make_id", "model_id"], ignore_index=True)


def flag_models(
    queue: pd.DataFrame, span_min: int = SPAN_MIN, gap_min: int = GAP_MIN
) -> pd.DataFrame:
    """Models whose year list suggests more than one generation, with the reason(s)."""
    per_model = _per_model(queue)
    span = per_model["span_years"] >= span_min
    gap = per_model["max_gap"] >= gap_min
    per_model["reasons"] = [
        ";".join(r for r, hit in (("span", s), ("gap", g)) if hit)
        for s, g in zip(span, gap, strict=True)
    ]
    return per_model[span | gap].reset_index(drop=True)


def coverage_stats(
    manifest: pd.DataFrame,
    queue: pd.DataFrame,
    span_min: int = SPAN_MIN,
    gap_min: int = GAP_MIN,
    top_n: int = TOP_MODELS,
) -> CoverageStats:
    mapped = int((queue["generation_id"].astype(str).str.strip() != "").sum())
    retained = int((manifest["validity_status"] == "valid").sum())
    reasons = Counter(r for r in manifest["exclusion_reason"] if isinstance(r, str))
    per_model = _per_model(queue)
    flagged = flag_models(queue, span_min, gap_min)
    top = per_model.sort_values(
        [IMAGE_COUNT, "make_id", "model_id"], ascending=[False, True, True]
    ).head(top_n)
    top["reasons"] = ""
    return CoverageStats(
        model_year_records={"total": len(queue), "mapped": mapped, "unmapped": len(queue) - mapped},
        images={"total": len(manifest), "retained": retained, "excluded": len(manifest) - retained},
        exclusion_reason_counts=dict(sorted(reasons.items())),
        status_counts=dict(sorted(Counter(queue["mapping_status"]).items())),
        make_count=int(queue["make_id"].nunique()),
        model_count=len(per_model),
        models_without_valid_model_year=int(manifest["model_id"].nunique()) - len(per_model),
        classes_if_every_year_is_a_generation=len(queue),
        classes_if_all_years_merged=len(per_model),
        span_min=span_min,
        gap_min=gap_min,
        flagged_models=[ModelSummary(**r) for r in flagged.to_dict(orient="records")],
        top_models=[ModelSummary(**r) for r in top.to_dict(orient="records")],
    )


def _table(header: list[str], rows: list[list[object]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return lines


def _model_rows(models: list[ModelSummary], with_reasons: bool) -> list[list[object]]:
    rows = []
    for m in models:
        row = [m.make_name, m.model_name, m.model_id, m.years, m.span_years, m.max_gap]
        if with_reasons:
            row.append(m.reasons)
        else:
            row.insert(3, m.image_count)
        rows.append(row)
    return rows


def render_report(stats: CoverageStats, queue_csv: Path, version: str) -> str:
    """The coverage report as Markdown; the checksum is the SHA-256 of ``queue_csv``."""
    digest = sha256_of_file(queue_csv)
    r, i = stats.model_year_records, stats.images
    lines = [
        f"# Generation mapping coverage report v{version}",
        "",
        f"- mapping version: {version}",
        f"- queue file: `{queue_csv.name}`",
        f"- sha256: {digest}  {queue_csv.name}",
        "",
        "Every queue row is `provisional` and carries only what the CompCars folder "
        "structure states (ids, names, folder year). No generation interval, identifier, "
        "facelift phase, source or reviewer has been filled in, by hand or by any automated "
        "tool: those fields stay empty until the named human reviewers approve them "
        "(PROTOCOL.md §6 rules 8 and 10).",
        "",
        "## Model-year records",
        "",
        *_table(["total", "mapped", "unmapped"], [[r["total"], r["mapped"], r["unmapped"]]]),
        "",
        "## Images",
        "",
        *_table(["total", "retained", "excluded"], [[i["total"], i["retained"], i["excluded"]]]),
        "",
        *_table(["exclusion reason", "images"], sorted(stats.exclusion_reason_counts.items())),
        "",
        "## Mapping status distribution",
        "",
        *_table(["mapping_status", "rows"], sorted(stats.status_counts.items())),
        "",
        "## Candidate class counts",
        "",
        f"Makes: {stats.make_count}. Models with at least one valid model-year: "
        f"{stats.model_count}; models in the manifest whose every image is excluded "
        f"(no valid year): {stats.models_without_valid_model_year}.",
        "",
        *_table(
            ["hypothesis", "classes"],
            [
                [
                    "every CompCars year is its own generation",
                    stats.classes_if_every_year_is_a_generation,
                ],
                ["all years of a model form one generation", stats.classes_if_all_years_merged],
            ],
        ),
        "",
        "The approved mapping will land between these two bounds.",
        "",
        "## Models whose year list suggests more than one generation",
        "",
        f"Flag `span`: inclusive span of recorded years >= {stats.span_min}. Flag `gap`: "
        f"a gap >= {stats.gap_min} years between consecutive recorded years. "
        f"{len(stats.flagged_models)} models flagged; these are review hints, not labels.",
        "",
        *_table(
            ["make", "model", "model_id", "years", "span", "max gap", "flags"],
            _model_rows(stats.flagged_models, with_reasons=True),
        ),
        "",
        f"## Top {len(stats.top_models)} models by retained image count",
        "",
        *_table(
            ["make", "model", "model_id", "images", "years", "span", "max gap"],
            _model_rows(stats.top_models, with_reasons=False),
        ),
        "",
    ]
    return "\n".join(lines)
