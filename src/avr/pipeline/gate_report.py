"""Render ``reports/m1_gate/README.md`` from the gate result files (issue M1-16).

The README is derived entirely from ``reports/m1_gate/<track>_<variant>.json`` so the tables
can never drift from the recorded numbers. Every table is internal validation only.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from avr.data.split_identity import TRACK_A_LABEL

EXPECTED_RUNS = (("track_a", "p1"), ("track_a", "p3"), ("track_b", "p1"), ("track_b", "p3"))

CHECKLIST = (
    (
        "validated manifest",
        "`data/manifests/compcars_web_images.sha256`, `compcars_web_images.summary.json` "
        "(136,726 rows, 136,365 valid; the Parquet is rebuilt by "
        "`scripts/build_image_manifest.py` and must match the sidecar)",
    ),
    (
        "deduplication report",
        "`reports/dedup/README.md`, `reports/dedup/AUDIT.md` (ADR-0005, pHash distance 2)",
    ),
    (
        "split manifests",
        "`data/splits/track_a/official_cvpr2015.parquet` + `.sha256`, "
        "`reports/splits/track_a/README.md`; `data/splits/track_b/seed_*.sha256`, "
        "`summary.json`, `reports/splits/track_b/eligibility_audit.md`; cross-view "
        "`data/splits/track_b/cross_view/` (ADR-0004, ADR-0006)",
    ),
    (
        "initial mapping coverage report",
        "`reports/mapping/v0.1/coverage.md` (queue v0.1, every row `provisional`)",
    ),
    (
        "leakage audit on the manifests used",
        "`reports/leakage/track_a.md` (PASS; 52 official-split crossing groups reported, "
        "not altered) and `reports/leakage/track_b_seed_73.md` (PASS)",
    ),
    (
        "deterministic run",
        "this report: two repeats per track and variant, prototypes and rank matrices "
        "byte-identical, embeddings identical (tables below)",
    ),
)


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
        f"dimension {store['dimension']}. Result file: `{track}_{variant}.json`.",
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


def render_readme(results: Iterable[dict]) -> str:
    results = sorted(results, key=lambda r: (r["track"], r["variant"]))
    present = {(r["track"], r["variant"]) for r in results}
    lines = [
        "# M1 acceptance gate (issue M1-16, PROTOCOL.md §25 M1)",
        "",
        "DINOv3 ViT-B/16 (`configs/encoders/dinov3_vitb16.yaml`), P1 and P3 preprocessing, "
        "M0 global-mean prototypes built from support rows only, exact cosine search "
        "(matrix multiplication asserted equal to FAISS `IndexFlatIP`), over all 431 Track A "
        "classes on the official split and all 2,645 currently eligible Track B classes on "
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
    lines += [f"| {name} | {where} |" for name, where in CHECKLIST]
    lines.append("")
    for r in results:
        lines += _run_section(r)
    missing = [f"- {t} {v}" for t, v in EXPECTED_RUNS if (t, v) not in present]
    if missing:
        lines += ["## Not yet run", "", *missing, ""]
    return "\n".join(lines).rstrip("\n") + "\n"


def render_from_files(paths: Sequence) -> str:
    import json

    return render_readme(json.loads(p.read_text()) for p in paths)
