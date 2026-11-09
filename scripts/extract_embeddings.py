"""Extract frozen-encoder embeddings for the manifest images (PROTOCOL.md §9, issue M1-13).

Hydra overrides select the encoder and the extraction settings, e.g.
    uv run python scripts/extract_embeddings.py encoders@encoder=dinov3_vitb16 \\
        extraction.variant=p3 extraction.limit=2000 extraction.masks_dir=/abs/path/masks
Shards land under data/embeddings/<encoder>/<variant>/ (git-ignored); rerunning resumes at
the first unfinished shard. For P3 the images whose mask is not ``ok`` (or whose PNG is
absent) are written to missing.json and excluded; they are never replaced by the P1 crop.
Throughput and peak GPU memory are logged to the local MLflow store under a run manifest.
Fields that only make sense for an evaluation run (split, generation mapping, prototype
method, support_k) are filled with ``not_applicable`` / 0 so the manifest is complete.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import mlflow

from avr.config import CONFIG_DIR, load_config
from avr.encoders.adapter import FrozenEncoder
from avr.encoders.extract import (
    ExtractionStats,
    load_jobs,
    run_extraction,
    select_p3_jobs,
    write_missing,
)
from avr.encoders.registry import EncoderSpec
from avr.encoders.store import EmbeddingStore
from avr.tracking.manifest import RunManifest, capture_environment
from avr.tracking.mlflow_run import tracked_run, tracking_dir_from_config

REPO_ROOT = CONFIG_DIR.parent
MANIFESTS_DIR = REPO_ROOT / "data" / "manifests"
PATHS_DIR = REPO_ROOT / "data" / "paths"


def _resolve(path: str) -> Path:
    p = Path(path).expanduser()
    return p if p.is_absolute() else REPO_ROOT / p


def main(overrides: list[str]) -> ExtractionStats:
    import torch
    from omegaconf import OmegaConf

    cfg = load_config(overrides=overrides)
    ext = cfg.extraction
    variant = str(ext.variant)
    root = (REPO_ROOT / cfg.dataset.root).resolve()
    spec = EncoderSpec.model_validate(OmegaConf.to_container(cfg.encoder, resolve=True))

    manifest_path = MANIFESTS_DIR / "compcars_web_images.parquet"
    jobs = load_jobs(manifest_path, PATHS_DIR / "compcars_web_image_paths.parquet", ext.limit)
    masks_dir = _resolve(str(ext.masks_dir))
    missing = []
    if variant == "p3":
        jobs, missing = select_p3_jobs(jobs, _resolve(str(ext.mask_metadata)), masks_dir)

    encoder = FrozenEncoder.from_spec(spec, device=ext.device)
    store = EmbeddingStore(_resolve(str(ext.embeddings_dir)), encoder.identity(variant))
    if missing:
        write_missing(store, missing)
    print(
        f"{spec.name} {variant}: {len(jobs)} images ({len(missing)} excluded for p3), "
        f"{len(store.finished_shards())} shards already finished, device {encoder.device} "
        f"{encoder.dtype}",
        flush=True,
    )

    manifest = RunManifest(
        run_id=f"extract_{spec.name}_{variant}_{datetime.now(UTC):%Y%m%dT%H%M%SZ}",
        dataset_protocol=str(cfg.dataset.name),
        dataset_manifest_hash=(MANIFESTS_DIR / "compcars_web_images.sha256").read_text().split()[0],
        split_manifest_hash="not_applicable",
        generation_mapping_hash="not_applicable",
        encoder_id=spec.name,
        encoder_revision=encoder.revision,
        encoder_checksum=encoder.checkpoint_sha256,
        preprocessing_id=f"{spec.preprocessing.id}+{variant}",
        dimension_transform_id="native",
        prototype_method="not_applicable",
        support_k=0,
        seed=int(cfg.seed),
        **capture_environment(REPO_ROOT),
    )

    use_cuda = encoder.device.startswith("cuda")
    if use_cuda:
        torch.cuda.reset_peak_memory_stats()

    def on_shard(index: int, n_shards: int, stats: ExtractionStats) -> None:
        rate = stats.images_embedded / stats.elapsed_s if stats.elapsed_s else 0.0
        print(f"shard {index + 1}/{n_shards}: {stats.images_embedded} images, {rate:.1f} img/s")

    with tracked_run(
        manifest, tracking_dir_from_config(cfg.tracking), str(cfg.tracking.experiment_name)
    ) as run:
        started = time.perf_counter()
        stats = run_extraction(
            root,
            jobs,
            encoder,
            store,
            variant,
            batch_size=int(ext.batch_size),
            shard_size=int(ext.shard_size),
            masks_dir=masks_dir,
            expand=float(cfg.preprocessing.bbox_expand_fraction),
            on_shard=on_shard,
        )
        elapsed = time.perf_counter() - started
        peak_mb = torch.cuda.max_memory_allocated() / 2**20 if use_cuda else 0.0
        rate = stats.images_embedded / elapsed if stats.images_embedded else 0.0
        run.log_metric("images_embedded", stats.images_embedded)
        run.log_metric("images_per_second", rate)
        run.log_metric("peak_gpu_memory_mb", peak_mb)
        run.log_metric("p3_missing_count", len(missing))
        ids, matrix = store.load_all()
        run.log_metric("store_rows", len(ids))
        if missing:
            mlflow.log_artifact(str(store.directory / "missing.json"), run_id=run.mlflow_run_id)
        summary = {
            "encoder": spec.name,
            "variant": variant,
            "images_embedded": stats.images_embedded,
            "shards_written": stats.shards_written,
            "shards_skipped": stats.shards_skipped,
            "elapsed_s": round(elapsed, 1),
            "images_per_second": round(rate, 2),
            "peak_gpu_memory_mb": round(peak_mb, 1),
            "store_shape": list(matrix.shape),
            "p3_missing": len(missing),
        }
        print(json.dumps(summary, indent=2))
    return stats


if __name__ == "__main__":
    main(sys.argv[1:])
