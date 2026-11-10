"""Build the Track B cross-view variants (ADR-0004): held-out query viewpoint.

Reads every stored parent split data/splits/track_b/seed_<s>.parquet (built by
scripts/build_track_b_splits.py; each sha256 sidecar must match), writes the per-view
eligibility table reports/splits/track_b/cross_view/seed_<s>.md before any variant, then
one data/splits/track_b/cross_view/seed_<s>/view_<v>.parquet (+ .sha256 sidecar) per
held-out view and summary.json with the class and partition counts. Views and thresholds
come from configs/splits/track_b.yaml (cross_view).
Usage: uv run python scripts/build_cross_view_splits.py
"""

import json

import pandas as pd

from avr.config import CONFIG_DIR, load_config
from avr.data.cross_view import CrossViewThresholds, build_cross_view_splits
from avr.io import read_sidecar_sha256, sha256_of_file

REPO_ROOT = CONFIG_DIR.parent
SPLITS = REPO_ROOT / "data" / "splits" / "track_b"
OUT_DIR = SPLITS / "cross_view"
REPORT_DIR = REPO_ROOT / "reports" / "splits" / "track_b" / "cross_view"


def main() -> None:
    cfg = load_config().split_track_b
    thresholds = CrossViewThresholds(
        min_support=cfg.cross_view.min_support, min_test=cfg.cross_view.min_test
    )
    views = list(cfg.cross_view.views)
    summary: dict = {"parent_sha256": {}, "class_counts": {}, "partition_counts": {}}
    for seed in [cfg.primary_seed, *cfg.sensitivity_seeds]:
        parent = SPLITS / f"seed_{seed}.parquet"
        digest = sha256_of_file(parent)
        if digest != read_sidecar_sha256(parent.with_suffix(".sha256")):
            raise SystemExit(f"{parent} does not match its .sha256 sidecar; rebuild it first")
        result = build_cross_view_splits(
            pd.read_parquet(parent),
            seed=seed,
            views=views,
            thresholds=thresholds,
            report_path=REPORT_DIR / f"seed_{seed}.md",
            out_dir=OUT_DIR / f"seed_{seed}",
        )
        summary["parent_sha256"][str(seed)] = digest
        summary["class_counts"][str(seed)] = {str(v): n for v, n in result.class_counts.items()}
        summary["partition_counts"][str(seed)] = {
            str(v): c for v, c in result.partition_counts.items()
        }
        print(f"wrote {result.report_path.relative_to(REPO_ROOT)}")
        for v, out in result.outputs.items():
            print(
                f"  view {v}: {result.class_counts[v]} classes, "
                f"{result.partition_counts[v]} -> {out.manifest.relative_to(REPO_ROOT)} "
                f"({out.manifest.stat().st_size} bytes)"
            )
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {(OUT_DIR / 'summary.json').relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
