"""Build the CompCars image manifest with duplicate groups (PROTOCOL.md §7).

Writes data/manifests/compcars_web_images.{parquet,sha256,summary.json} (public, with
``phash``, ``duplicate_group_id`` and ``source_group_id`` filled), the private
data/paths/compcars_web_image_paths.parquet (git-ignored) and the duplicate report under
reports/dedup/ (README.md and near_threshold_pairs.csv committed, pairs/ git-ignored).
The dataset root comes from configs/datasets (COMPCARS_ROOT overrides it) and the pHash
threshold from configs/dedup.
Usage: uv run python scripts/build_image_manifest.py
"""

import time

from avr.config import CONFIG_DIR, load_config
from avr.data.dedup import deduplicate, write_report
from avr.data.manifest import build_manifest, write_manifest

REPO_ROOT = CONFIG_DIR.parent
MANIFESTS_DIR = REPO_ROOT / "data" / "manifests"
PATHS_DIR = REPO_ROOT / "data" / "paths"
REPORT_DIR = REPO_ROOT / "reports" / "dedup"
AUDIT_PAIRS = 50


def main() -> None:
    cfg = load_config()
    root = (REPO_ROOT / cfg.dataset.root).resolve()
    started = time.perf_counter()
    rows = build_manifest(root)
    built = time.perf_counter() - started
    result = deduplicate(
        rows, root, threshold=cfg.dedup.phash_hamming_threshold, hash_size=cfg.dedup.phash_hash_size
    )
    elapsed = time.perf_counter() - started
    out = write_manifest(result.rows, manifests_dir=MANIFESTS_DIR, paths_dir=PATHS_DIR)
    image_paths = {r.image_id: root / r.relative_path for r in result.rows}
    write_report(result, image_paths, REPORT_DIR, max_pairs=AUDIT_PAIRS, runtime_seconds=elapsed)
    print(
        f"wrote {out.manifest.relative_to(REPO_ROOT)} ({len(rows)} rows) in {elapsed:.0f}s "
        f"(manifest {built:.0f}s, dedup {elapsed - built:.0f}s)"
    )
    print(out.summary.read_text())
    print(result.stats.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
