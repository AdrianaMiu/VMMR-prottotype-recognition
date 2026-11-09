"""Build the public Track A split manifest (PROTOCOL.md §5 Track A).

Joins the official ``train_test_split/classification/{train,test}.txt`` files onto the
image manifest through the private path table, partitions the support half into
``fit``/``validation`` and writes data/splits/track_a/official_cvpr2015.{parquet,sha256}
plus reports/splits/track_a/README.md. The dataset root comes from configs/datasets
(COMPCARS_ROOT overrides it); fraction and seed from configs/splits/track_a.yaml.
Usage: uv run python scripts/build_track_a_split.py
"""

import pandas as pd

from avr.config import CONFIG_DIR, load_config
from avr.data.manifest import MANIFEST_NAME, PATHS_NAME
from avr.data.split_track_a import build_track_a, read_official_split, write_track_a_split

REPO_ROOT = CONFIG_DIR.parent
MANIFEST = REPO_ROOT / "data" / "manifests" / f"{MANIFEST_NAME}.parquet"
PATHS = REPO_ROOT / "data" / "paths" / f"{PATHS_NAME}.parquet"
SPLITS_DIR = REPO_ROOT / "data" / "splits" / "track_a"
REPORT_DIR = REPO_ROOT / "reports" / "splits" / "track_a"


def main() -> None:
    cfg = load_config()
    split_cfg = load_config("splits/track_a").splits
    root = (REPO_ROOT / cfg.dataset.root).resolve()
    build = build_track_a(
        pd.read_parquet(MANIFEST),
        pd.read_parquet(PATHS),
        read_official_split(root),
        fraction=split_cfg.validation_fraction,
        seed=split_cfg.validation_seed,
    )
    out = write_track_a_split(build, SPLITS_DIR, REPORT_DIR)
    print(f"wrote {out.manifest.relative_to(REPO_ROOT)} ({build.table.num_rows} rows)")
    print(build.report.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
