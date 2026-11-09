"""Run the M1 acceptance gate for one track and variant, twice, and diff the runs (M1-16).

    uv run python scripts/run_m1_gate.py track_a p1 experiment.embeddings_dir=/abs/embeddings

Positional arguments are the track (``track_a`` | ``track_b``) and the preprocessing variant
(``p1`` | ``p3``); every further argument is a Hydra override (``experiments@experiment=m1_gate``
is selected here). The run is repeated ``experiment.repeats`` times; prototypes and rank
matrices must be byte-identical (``numpy.array_equal``) and embeddings within 1e-6 between
repeats, and the verdicts are written into reports/m1_gate/<track>_<variant>.json under
``rerun``. Every number is internal validation only, not an article result.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from avr.config import CONFIG_DIR, load_config
from avr.data.manifest import MANIFEST_NAME
from avr.pipeline.gate import GateInputs, GateResult, run_track
from avr.tracking.mlflow_run import tracking_dir_from_config

REPO_ROOT = CONFIG_DIR.parent
EMBEDDING_TOLERANCE = 1e-6


def _resolve(path: str) -> Path:
    p = Path(path).expanduser()
    return p if p.is_absolute() else REPO_ROOT / p


def compare_runs(first: GateResult, second: GateResult) -> dict:
    """Byte-identity of prototypes and ranks, embedding tolerance, metric equality."""
    embedding_max_abs = max(
        float(np.max(np.abs(first.embeddings[k] - second.embeddings[k])))
        if first.embeddings[k].shape == second.embeddings[k].shape
        else float("inf")
        for k in first.embeddings
    )
    return {
        "prototypes_byte_identical": bool(
            first.class_ids == second.class_ids
            and np.array_equal(first.prototypes, second.prototypes)
            and first.prototypes.tobytes() == second.prototypes.tobytes()
        ),
        "ranks_byte_identical": {
            k: bool(
                np.array_equal(first.ranks[k], second.ranks[k])
                and first.ranks[k].tobytes() == second.ranks[k].tobytes()
            )
            for k in first.ranks
        },
        "embeddings_max_abs_difference": embedding_max_abs,
        "embeddings_within_tolerance": embedding_max_abs <= EMBEDDING_TOLERANCE,
        "prototype_parquet_sha256_equal": (
            first.results["prototypes"]["parquet_sha256"]
            == second.results["prototypes"]["parquet_sha256"]
        ),
        "metrics_equal": first.results["evaluations"] == second.results["evaluations"],
        "determinism_hashes_equal": first.results["determinism"] == second.results["determinism"],
        "runtime_s": [first.results["runtime_s"], second.results["runtime_s"]],
        "peak_rss_mb": [first.results["peak_rss_mb"], second.results["peak_rss_mb"]],
    }


def main(argv: list[str]) -> dict:
    track, variant, *overrides = argv
    cfg = load_config(overrides=["experiments@experiment=m1_gate", *overrides])
    exp = cfg.experiment
    inputs = GateInputs(
        track=track,
        variant=variant,
        encoder=str(cfg.encoder.name),
        embeddings_dir=_resolve(str(exp.embeddings_dir)),
        manifest_path=REPO_ROOT / "data" / "manifests" / f"{MANIFEST_NAME}.parquet",
        split_path=_resolve(str(exp.splits[track])),
        mapping_queue_path=_resolve(str(exp.mapping_queue)) if track == "track_b" else None,
        mask_metadata_path=_resolve(str(exp.mask_metadata)) if variant == "p3" else None,
        output_dir=_resolve(str(exp.output_dir)),
        prototypes_dir=_resolve(str(exp.prototypes_dir)),
        qdrant_dir=_resolve(str(exp.qdrant_dir)),
        tracking_dir=tracking_dir_from_config(cfg.tracking),
        seed=int(cfg.seed),
        parity_queries=int(exp.parity_queries),
    )
    runs = []
    for repeat in range(int(exp.repeats)):
        runs.append(run_track(inputs))
        elapsed = runs[-1].results["runtime_s"]
        print(f"{track} {variant} repeat {repeat + 1}: {elapsed} s", flush=True)
    results = runs[0].results
    results["rerun"] = {
        "repeats": len(runs),
        "comparisons": [compare_runs(runs[0], later) for later in runs[1:]],
    }
    out = inputs.output_dir / f"{track}_{variant}.json"
    out.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps({"output": str(out), "rerun": results["rerun"]}, indent=2))
    return results


if __name__ == "__main__":
    main(sys.argv[1:])
