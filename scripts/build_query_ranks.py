"""Build the per-query rank tables for one preprocessing variant (M2-6).

    uv run python scripts/build_query_ranks.py p1 experiment.embeddings_dir=/abs/embeddings

The positional argument is the variant (``p1`` | ``p3``); further arguments are Hydra
overrides (``experiments@experiment=query_ranks`` is selected here, ``encoder=...`` picks the
encoder). For Track A (official query half) and Track B (seed 73, test) it writes

* ``data/results/query_ranks/<track>/<encoder>/<variant>.parquet`` (public columns only; DVC
  tracked, the ``.parquet.dvc`` pointer is what gets committed),
* ``reports/query_ranks/<track>_<variant>_per_class.csv``, and
* ``reports/query_ranks/cmc.csv`` (recomputed from every per-query table present),

after computing every table ``experiment.repeats`` times and checking the Parquet bytes are
identical. A run manifest is recorded through ``tracked_run`` for each track.
"""

from __future__ import annotations

import hashlib
import io
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from avr.config import CONFIG_DIR, load_config
from avr.evaluation.per_query import cmc_table, per_class_summary
from avr.pipeline.query_ranks import QueryRanksInputs, QueryRanksResult, run_query_ranks
from avr.tracking.manifest import RunManifest, capture_environment
from avr.tracking.mlflow_run import REPO_ROOT, tracked_run, tracking_dir_from_config

assert REPO_ROOT == CONFIG_DIR.parent
PROTOTYPE_METHOD = "m0_global_mean"
TRACKS = ("track_a", "track_b")


def _resolve(path: str) -> Path:
    p = Path(path).expanduser()
    return p if p.is_absolute() else REPO_ROOT / p


def parquet_bytes(table: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    pq.write_table(pa.Table.from_pandas(table, preserve_index=False), buffer, compression="zstd")
    return buffer.getvalue()


def rebuild_cmc(results_dir: Path, path: Path) -> None:
    """Recompute cmc.csv from every per-query table under ``results_dir``.

    The CSV is never read back: parsing floats from text would not round-trip exactly, so the
    file is always a pure function of the Parquet tables.
    """
    tables = sorted(results_dir.glob("*/*/*.parquet"))
    frames = [cmc_table(pd.read_parquet(t)) for t in tables]
    keys = ["track", "encoder", "variant", "k"]
    pd.concat(frames, ignore_index=True).sort_values(keys).to_csv(path, index=False)


def main(argv: list[str]) -> dict:
    variant, *overrides = argv
    cfg = load_config(overrides=["experiments@experiment=query_ranks", *overrides])
    exp = cfg.experiment
    encoder = str(cfg.encoder.name)
    results_dir, report_dir = _resolve(str(exp.results_dir)), _resolve(str(exp.report_dir))
    report_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for track in TRACKS:
        inputs = QueryRanksInputs(
            track=track,
            variant=variant,
            encoder=encoder,
            embeddings_dir=_resolve(str(exp.embeddings_dir)),
            manifest_path=_resolve(str(exp.manifest)),
            split_path=_resolve(str(exp.splits[track])),
        )
        runs: list[QueryRanksResult] = [run_query_ranks(inputs) for _ in range(int(exp.repeats))]
        blobs = [parquet_bytes(r.per_query) for r in runs]
        identical = len({hashlib.sha256(b).hexdigest() for b in blobs}) == 1
        if not identical:
            raise RuntimeError(f"{track} {variant}: reruns produced different Parquet bytes")
        result = runs[0]
        table = result.per_query
        out = results_dir / track / encoder / f"{variant}.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(blobs[0])
        per_class = per_class_summary(table, result.class_names)
        per_class.to_csv(report_dir / f"{track}_{variant}_per_class.csv", index=False)
        rebuild_cmc(results_dir, report_dir / "cmc.csv")
        _record_run(cfg, inputs, result, table)
        summary[track] = {
            "rows": len(table),
            "classes_with_queries": len(per_class),
            "prototypes": result.prototype_count,
            "support_images": result.support_images,
            "without_embedding": result.without_embedding,
            "queries_not_eligible_unknown_viewpoint": result.queries_not_eligible,
            "parquet": str(out),
            "parquet_sha256": hashlib.sha256(blobs[0]).hexdigest(),
            "reruns_byte_identical": identical,
            "top1": float(table.hit_at_1.mean()),
            "top5": float(table.hit_at_5.mean()),
            "mrr": float((1.0 / table.rank_true).mean()),
            "mean_rank": float(table.rank_true.mean()),
        }
        print(track, variant, summary[track], flush=True)
    return summary


def _record_run(cfg, inputs: QueryRanksInputs, result: QueryRanksResult, table: pd.DataFrame):
    exp = cfg.experiment
    mapping = (
        hashlib.sha256(_resolve(str(exp.mapping_queue)).read_bytes()).hexdigest()
        if inputs.track == "track_b"
        else "not_applicable"
    )
    identity = result.identity
    manifest = RunManifest(
        run_id=f"query_ranks_{inputs.track}_{inputs.variant}_{inputs.encoder}",
        dataset_protocol=f"compcars_web:{inputs.track}_{inputs.split_path.stem}",
        dataset_manifest_hash=result.manifest_sha256,
        split_manifest_hash=result.split_sha256,
        generation_mapping_hash=mapping,
        encoder_id=identity.encoder,
        encoder_revision=identity.revision,
        encoder_checksum=identity.checkpoint_sha256,
        preprocessing_id=f"{identity.preprocessing_id}+{identity.variant}",
        dimension_transform_id="native",
        prototype_method=PROTOTYPE_METHOD,
        support_k="full",
        seed=int(cfg.seed),
        **capture_environment(REPO_ROOT),
    )
    with tracked_run(manifest, tracking_dir=tracking_dir_from_config(cfg.tracking)) as run:
        run.log_metric("top1", float(table.hit_at_1.mean()))
        run.log_metric("top5", float(table.hit_at_5.mean()))
        run.log_metric("mrr", float((1.0 / table.rank_true).mean()))
        run.log_metric("queries", float(len(table)))


if __name__ == "__main__":
    main(sys.argv[1:])
