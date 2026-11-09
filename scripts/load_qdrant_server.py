"""Load the prototype tables into the Qdrant server and print its collections (issue M5-1).

    docker compose -f docker/docker-compose.yml up -d
    uv run python scripts/load_qdrant_server.py [--prototypes-dir data/prototypes] [--url ...]

The target defaults to ``configs/service/qdrant.yaml`` (``AVR_QDRANT_URL`` and
``AVR_QDRANT_API_KEY`` override it); ``--path DIR`` loads an embedded local store instead.
Every ``<encoder>/<variant>/m0_<protocol_id>.parquet`` under the prototypes directory is
checked against its sidecar and exported under its PROTOCOL.md §12 collection name.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from avr.config import CONFIG_DIR, load_config
from avr.retrieval.qdrant_load import load_prototype_tables
from avr.retrieval.qdrant_store import QdrantTarget, list_collections, resolve_target

REPO_ROOT = CONFIG_DIR.parent


def target_from_config(cfg) -> QdrantTarget:
    """The server target from ``cfg.service.qdrant`` with the environment overrides applied."""
    return resolve_target(
        {
            "url": os.environ.get("AVR_QDRANT_URL", cfg.url),
            "api_key": os.environ.get("AVR_QDRANT_API_KEY", cfg.api_key),
            "timeout_s": int(cfg.timeout_s),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--prototypes-dir", default=None, help="default: experiment.prototypes_dir")
    parser.add_argument("--url", default=None, help="Qdrant server URL (default: config / env)")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--path", default=None, help="embedded local store instead of a server")
    args = parser.parse_args()

    cfg = load_config(overrides=["experiments@experiment=m1_gate"])
    prototypes_dir = Path(args.prototypes_dir or cfg.experiment.prototypes_dir).expanduser()
    if not prototypes_dir.is_absolute():
        prototypes_dir = REPO_ROOT / prototypes_dir
    if args.path:
        target = resolve_target({"path": args.path})
    elif args.url:
        target = resolve_target({"url": args.url, "api_key": args.api_key})
    else:
        target = target_from_config(cfg.service.qdrant)

    summaries = load_prototype_tables(prototypes_dir, target)
    for summary in summaries:
        print(
            f"loaded {summary.point_count:>6} points  dim {summary.dimension}  {summary.collection}"
        )
    print("collections on target:")
    for name in list_collections(target):
        print(f"  {name}")


if __name__ == "__main__":
    main()
