"""Build the CompCars image manifest (PROTOCOL.md §7) from the local release.

Writes data/manifests/compcars_web_images.{parquet,sha256,summary.json} (public) and
data/paths/compcars_web_image_paths.parquet (private, git-ignored). The dataset root comes
from configs/datasets (COMPCARS_ROOT overrides it).
Usage: uv run python scripts/build_image_manifest.py
"""

import time

from avr.config import CONFIG_DIR, load_config
from avr.data.manifest import build_manifest, write_manifest

REPO_ROOT = CONFIG_DIR.parent
MANIFESTS_DIR = REPO_ROOT / "data" / "manifests"
PATHS_DIR = REPO_ROOT / "data" / "paths"


def main() -> None:
    root = (REPO_ROOT / load_config().dataset.root).resolve()
    started = time.perf_counter()
    rows = build_manifest(root)
    out = write_manifest(rows, manifests_dir=MANIFESTS_DIR, paths_dir=PATHS_DIR)
    elapsed = time.perf_counter() - started
    print(f"wrote {out.manifest.relative_to(REPO_ROOT)} ({len(rows)} rows) in {elapsed:.0f}s")
    print(out.summary.read_text())


if __name__ == "__main__":
    main()
