"""Render ``reports/m1_gate/README.md`` from the gate result files (issue M1-16).

The README is derived entirely from ``reports/m1_gate/<track>_<variant>.json`` so the tables
can never drift from the recorded numbers. Every table is internal validation only.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Sequence
from pathlib import Path

from pydantic import BaseModel

from avr.data.split_identity import TRACK_A_LABEL

EXPECTED_RUNS = (("track_a", "p1"), ("track_a", "p3"), ("track_b", "p1"), ("track_b", "p3"))

# --- checklist context: every number comes from a committed record, none is typed here ----

LEAKAGE_DIR = Path("reports") / "leakage"
MANIFEST_SUMMARY = Path("data") / "manifests" / "compcars_web_images.summary.json"
MASK_SUMMARY = Path("data") / "manifests" / "p3_mask_metadata.summary.json"
TRACK_B_SUMMARY = Path("data") / "splits" / "track_b" / "summary.json"
SEGMENTATION_REPORT = Path("reports") / "segmentation" / "README.md"
EMBEDDINGS_REPORT = Path("reports") / "embeddings" / "README.md"
CROSS_VIEW_AUDITS = Path("reports") / "splits" / "track_b" / "cross_view"
WAIVER_23 = (
    "§23 'all Track B generation rows approved': WAIVED, protocol provisional (issue #17); "
    "every Track B class is a model-year stand-in with `mapping_status = provisional`"
)
_VERDICT = re.compile(r"Verdict: \*\*(PASS|FAIL)\*\*")
_CROSSING = re.compile(
    r"\| no_duplicate_group_id_crosses_support_query \|[^|]*\|[^|]*\|[^|]*crossing_groups=(\d+)"
)


class LeakageSummary(BaseModel):
    report: str
    verdict: str
    official_crossing_groups: int | None = None


class ChecklistContext(BaseModel):
    """Counts and verdicts the checklist cites, read from the committed records."""

    manifest_rows: int
    manifest_valid: int
    duplicate_groups: int
    mask_rows: int
    mask_failure_rate: float
    track_b_eligible_classes: int
    track_b_eligible_images: int
    leakage: list[LeakageSummary]
    cross_view_audits: list[str]


def leakage_summary(path: Path, repo_root: Path) -> LeakageSummary:
    text = path.read_text()
    verdict = _VERDICT.search(text)
    if verdict is None:
        raise ValueError(f"{path}: no verdict line")
    crossing = _CROSSING.search(text)
    return LeakageSummary(
        report=path.relative_to(repo_root).as_posix(),
        verdict=verdict.group(1),
        official_crossing_groups=int(crossing.group(1)) if crossing else None,
    )


def load_checklist_context(repo_root: Path) -> ChecklistContext:
    manifest = json.loads((repo_root / MANIFEST_SUMMARY).read_text())
    masks = json.loads((repo_root / MASK_SUMMARY).read_text())
    track_b = json.loads((repo_root / TRACK_B_SUMMARY).read_text())
    return ChecklistContext(
        manifest_rows=manifest["row_count"],
        manifest_valid=manifest["valid_count"],
        duplicate_groups=manifest["duplicate_group_count"],
        mask_rows=masks["row_count"],
        mask_failure_rate=masks["mask_failure_rate"],
        track_b_eligible_classes=track_b["eligible_classes"],
        track_b_eligible_images=track_b["eligible_images"],
        leakage=[
            leakage_summary(p, repo_root) for p in sorted((repo_root / LEAKAGE_DIR).glob("*.md"))
        ],
        cross_view_audits=[
            p.relative_to(repo_root).as_posix()
            for p in sorted((repo_root / CROSS_VIEW_AUDITS).glob("seed_*.md"))
        ],
    )


def _leakage_cell(context: ChecklistContext) -> str:
    parts = []
    for item in context.leakage:
        note = ""
        if item.official_crossing_groups is not None:
            note = (
                f"; {item.official_crossing_groups} official-split crossing groups reported, "
                "not altered"
            )
        parts.append(f"`{item.report}` ({item.verdict}{note})")
    return ", ".join(parts) if parts else "no leakage report found"


def checklist(context: ChecklistContext) -> list[tuple[str, str]]:
    cross_view = ", ".join(f"`{p}`" for p in context.cross_view_audits) or "none"
    return [
        (
            "validated manifest",
            "`data/manifests/compcars_web_images.sha256`, `compcars_web_images.summary.json` "
            f"({context.manifest_rows:,} rows, {context.manifest_valid:,} valid; the Parquet is "
            "DVC-tracked, rebuilt by `scripts/build_image_manifest.py` and verified against the "
            "sidecar by every gate run)",
        ),
        (
            "deduplication report",
            "`reports/dedup/README.md`, `reports/dedup/AUDIT.md` (ADR-0005, pHash distance 2; "
            f"{context.duplicate_groups:,} duplicate groups in the manifest summary)",
        ),
        (
            "split manifests",
            "`data/splits/track_a/official_cvpr2015.parquet` + `.sha256`, "
            "`reports/splits/track_a/README.md`; `data/splits/track_b/seed_*.sha256`, "
            f"`summary.json` ({context.track_b_eligible_classes:,} eligible classes, "
            f"{context.track_b_eligible_images:,} images), "
            "`reports/splits/track_b/eligibility_audit.md`; cross-view eligibility "
            f"(ADR-0004, ADR-0006): {cross_view}",
        ),
        (
            "P3 masks",
            f"`{SEGMENTATION_REPORT.as_posix()}` ({context.mask_rows:,} images, mask failure "
            f"rate {context.mask_failure_rate:.2%}; metadata sha256 recorded per P3 run below)",
        ),
        ("embedding extraction", f"`{EMBEDDINGS_REPORT.as_posix()}`"),
        (
            "initial mapping coverage report",
            "`reports/mapping/v0.1/coverage.md` (queue v0.1, every row `provisional`)",
        ),
        ("leakage audit on the manifests used", _leakage_cell(context)),
        ("generation mapping approval", WAIVER_23),
        (
            "deterministic run",
            "this report: two repeats per track and variant, prototypes and rank matrices "
            "byte-identical, embeddings identical (tables below)",
        ),
    ]


def _yes(flag: bool) -> str:
    return "yes" if flag else "NO"


def _f(value: float | None, digits: int = 4) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _ranking_table(levels: dict) -> list[str]:
    rows = [
        "| level | classes | top-1 | top-5 | top-10 | top-20 | MRR | median rank | P90 rank |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, m in levels.items():
        r = m["ranking"]
        top = r["top_k"]
        rows.append(
            f"| {name} | {r['class_count']} | {_f(top['1'])} | {_f(top['5'])} | "
            f"{_f(top['10'])} | {_f(top['20'])} | {_f(r['mrr'])} | "
            f"{r['median_rank']:.1f} | {r['rank_p90']:.1f} |"
        )
    return rows


def _classification_table(levels: dict) -> list[str]:
    rows = [
        "| level | accuracy | macro P | macro R | macro F1 | weighted F1 | balanced acc. | "
        "classes without predicted positives |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, m in levels.items():
        c = m["classification"]
        rows.append(
            f"| {name} | {_f(c['accuracy'])} | {_f(c['macro_precision'])} | "
            f"{_f(c['macro_recall'])} | {_f(c['macro_f1'])} | {_f(c['weighted_f1'])} | "
            f"{_f(c['balanced_accuracy'])} | {c['classes_without_predicted_positives']} |"
        )
    return rows


def _decomposition(d: dict) -> list[str]:
    return [
        "| top-1 outcome | fraction of queries |",
        "|---|---:|",
        f"| wrong generation, same model | {_f(d['wrong_generation_same_model_fraction'])} |",
        f"| wrong model, same make | {_f(d['wrong_model_same_make_fraction'])} |",
        f"| wrong make | {_f(d['wrong_make_fraction'])} |",
        f"| generation accuracy given correct model | {_f(d['generation_accuracy_given_model'])} |",
    ]


def _run_section(r: dict) -> list[str]:
    track, variant = r["track"], r["variant"]
    lines = [f"## {track} {variant}", ""]
    if track == "track_a":
        lines.append(f"Split label: **{r['split_label']}**")
    else:
        lines.append(
            f"Split: `{r['protocol_id']}` (primary seed 73); classes are the provisional "
            "model-year stand-ins, every row `mapping_status = provisional`."
        )
    store = r["embedding_store"]
    lines += [
        "",
        f"Embedding store: {store['rows']:,} rows in {store['shards']} shards, "
        f"dimension {store['dimension']}. Result file: `{track}_{variant}.json`. "
        f"P3 mask metadata sha256: `{r.get('mask_metadata_sha256', 'not recorded')}`.",
        "",
    ]
    for name, ev in r["evaluations"].items():
        missing = ev["images_without_embedding"]
        lines += [
            f"### {name}",
            "",
            f"- support images {ev['support_images']:,}, query images {ev['query_images']:,}, "
            f"classes {ev['class_count']}; images without an embedding: support "
            f"{missing['support']}, query {missing['query']}; queries not eligible "
            f"(unknown viewpoint) {ev['queries_not_eligible_unknown_viewpoint']}; "
            f"queries tied at rank 1: {ev['tied_queries_top1']}",
            "",
            "Ranking (PROTOCOL.md §13.2)",
            "",
            *_ranking_table(ev["levels"]),
            "",
            "Classification (PROTOCOL.md §13.1)",
            "",
            *_classification_table(ev["levels"]),
            "",
            "Error decomposition (top-1)",
            "",
            *_decomposition(ev["error_decomposition"]),
            "",
        ]
    p, q = r["prototypes"], r["qdrant"]
    parity = q["parity"]
    lines += [
        "### Persisted prototypes and Qdrant parity",
        "",
        f"- Parquet `{p['parquet']}` ({p['class_count']} prototypes, support count min "
        f"{p['support_counts']['min']}, max {p['support_counts']['max']}), sha256 "
        f"`{p['parquet_sha256']}`",
        f"- Qdrant collection `{q['collection']}` ({q['point_count']} points, embedded local mode)",
        f"- Parity on {parity['queries']} queries, top-{parity['k']}: ordered-equal "
        f"{parity['ordered_equal']}, equal within score tolerance {parity['score_tolerance']} "
        f"{parity['equal_within_tolerance']}, passed: {_yes(parity['passed'])}",
        "",
    ]
    rerun = r.get("rerun")
    if rerun:
        lines += ["### Rerun diff", ""]
        lines += [
            "| repeat | prototypes byte-identical | ranks byte-identical | embeddings max abs "
            "diff | Parquet sha256 equal | metrics equal | runtime s | peak RSS MB |",
            "|---|---|---|---:|---|---|---|---|",
        ]
        for i, c in enumerate(rerun["comparisons"], start=2):
            ranks = ", ".join(f"{k}: {_yes(v)}" for k, v in c["ranks_byte_identical"].items())
            lines.append(
                f"| 1 vs {i} | {_yes(c['prototypes_byte_identical'])} | {ranks} | "
                f"{c['embeddings_max_abs_difference']:.1e} | "
                f"{_yes(c['prototype_parquet_sha256_equal'])} | {_yes(c['metrics_equal'])} | "
                f"{c['runtime_s'][0]} / {c['runtime_s'][1]} | "
                f"{c['peak_rss_mb'][0]} / {c['peak_rss_mb'][1]} |"
            )
        lines.append("")
    m = r["run_manifest"]
    lines += [
        f"Run manifest `{m['run_id']}` at commit `{m['git_commit']}` "
        f"(dirty worktree: {_yes(m['dirty_worktree_flag'])}); full manifest in the JSON and "
        "in the local MLflow store.",
        "",
    ]
    return lines


def render_readme(results: Iterable[dict], context: ChecklistContext) -> str:
    results = sorted(results, key=lambda r: (r["track"], r["variant"]))
    present = {(r["track"], r["variant"]) for r in results}
    lines = [
        "# M1 acceptance gate (issue M1-16, PROTOCOL.md §25 M1)",
        "",
        "DINOv3 ViT-B/16 (`configs/encoders/dinov3_vitb16.yaml`), P1 and P3 preprocessing, "
        "M0 global-mean prototypes built from support rows only, exact cosine search "
        "(matrix multiplication asserted equal to FAISS `IndexFlatIP`), over every Track A "
        "class on the official split and the "
        f"{context.track_b_eligible_classes:,} currently eligible Track B classes on "
        "seed 73. Built by `scripts/run_m1_gate.py`; this file is rendered from the result "
        "JSON files by `scripts/render_m1_gate_readme.py`.",
        "",
        "**These are internal validation numbers for the M1 gate. They are not article "
        "results and must not appear in any article table.** Track B uses the provisional "
        "model-year stand-in classes (no approved generation mapping exists yet), so every "
        "Track B output is `mapping_status = provisional`. Every Track A output carries the "
        f"label: *{TRACK_A_LABEL}*",
        "",
        "Prototype Parquet tables live under `data/prototypes/<encoder>/<variant>/` and the "
        "embedded Qdrant store under `data/qdrant/`; both are git-ignored and only the "
        "`.sha256` sidecars are committed. In embedded local mode Qdrant searches by brute "
        "force and ignores payload indexes; the keyword indexes are declared so the same "
        "export works against a server.",
        "",
        "## PROTOCOL.md §25 M1 checklist",
        "",
        "| output | artifact |",
        "|---|---|",
    ]
    lines += [f"| {name} | {where} |" for name, where in checklist(context)]
    lines.append("")
    for r in results:
        lines += _run_section(r)
    missing = [f"- {t} {v}" for t, v in EXPECTED_RUNS if (t, v) not in present]
    if missing:
        lines += ["## Not yet run", "", *missing, ""]
    return "\n".join(lines).rstrip("\n") + "\n"


def render_from_files(paths: Sequence[Path], repo_root: Path) -> str:
    return render_readme(
        (json.loads(p.read_text()) for p in paths), load_checklist_context(repo_root)
    )
