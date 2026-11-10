"""Build the committed E0 ten-class fixture once (PROTOCOL.md §14 "Stage E0", issue M1-15).

    uv run python scripts/build_e0_fixture.py [hydra overrides]

Selects the fixture split from the public manifest (``avr.pipeline.e0.select_fixture_split``,
seed ``cfg.seed``), embeds the P1 crops of its images with the configured frozen encoder
(``encoders@encoder``, default DINOv3 ViT-B/16, fp16 on CUDA) and writes
``tests/fixtures/e0_10class/`` with the manifest slice, embeddings, split, recorded metrics
and README. The fixture is a few hundred KB and is committed; CI replays it on CPU through
``tests/integration/test_e0_fixture.py``. Needs the ``gpu`` extra, the built manifest and
the private path table (ADR-0002). This is the only step of E0 that touches the GPU.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from avr.config import CONFIG_DIR, load_config
from avr.encoders.adapter import FrozenEncoder
from avr.encoders.registry import EncoderSpec
from avr.pipeline.e0 import build_fixture
from avr.tracking.manifest import capture_git_commit

REPO_ROOT = CONFIG_DIR.parent
MANIFESTS_DIR = REPO_ROOT / "data" / "manifests"
PATHS_DIR = REPO_ROOT / "data" / "paths"
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "e0_10class"


def main(overrides: list[str]) -> dict[str, float]:
    from omegaconf import OmegaConf

    cfg = load_config(overrides=overrides)
    spec = EncoderSpec.model_validate(OmegaConf.to_container(cfg.encoder, resolve=True))
    encoder = FrozenEncoder.from_spec(spec, device=cfg.extraction.device)
    print(f"{spec.name} on {encoder.device} {encoder.dtype}", flush=True)

    metrics = build_fixture(
        FIXTURE_DIR,
        manifest_path=MANIFESTS_DIR / "compcars_web_images.parquet",
        paths_path=PATHS_DIR / "compcars_web_image_paths.parquet",
        root=(REPO_ROOT / cfg.dataset.root).resolve(),
        embedder=encoder,
        identity=encoder.identity("p1"),
        git_commit=capture_git_commit(REPO_ROOT),
        seed=int(cfg.seed),
        batch_size=int(cfg.extraction.batch_size),
        expand=float(cfg.preprocessing.bbox_expand_fraction),
    )
    print(json.dumps(metrics, indent=2))
    sizes = {p.name: p.stat().st_size for p in sorted(Path(FIXTURE_DIR).iterdir())}
    print(json.dumps(sizes, indent=2))
    return metrics


if __name__ == "__main__":
    main(sys.argv[1:])
