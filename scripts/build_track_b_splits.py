"""Build the Track B eligibility audit and the five stored split manifests.

Reads the public image manifest data/manifests/compcars_web_images.parquet (built by
scripts/build_image_manifest.py; its sha256 sidecar must match), writes
reports/splits/track_b/eligibility_audit.md before dropping any class, then one
data/splits/track_b/seed_<s>.parquet (+ .sha256 sidecar) per seed and summary.json with
the partition sizes. Thresholds, fractions and seeds come from configs/splits/track_b.yaml.
Usage: uv run python scripts/build_track_b_splits.py
"""

import json

import pandas as pd

from avr.config import CONFIG_DIR, load_config
from avr.data.manifest import MANIFEST_NAME
from avr.data.split_track_b import EligibilityThresholds, build_track_b_splits
from avr.io import read_sidecar_sha256, sha256_of_file

REPO_ROOT = CONFIG_DIR.parent
MANIFEST = REPO_ROOT / "data" / "manifests" / f"{MANIFEST_NAME}.parquet"
AUDIT = REPO_ROOT / "reports" / "splits" / "track_b" / "eligibility_audit.md"
OUT_DIR = REPO_ROOT / "data" / "splits" / "track_b"


def main() -> None:
    cfg = load_config().split_track_b
    manifest_digest = sha256_of_file(MANIFEST)
    if manifest_digest != read_sidecar_sha256(MANIFEST.with_suffix(".sha256")):
        raise SystemExit(f"{MANIFEST} does not match its .sha256 sidecar; rebuild it first")

    result = build_track_b_splits(
        pd.read_parquet(MANIFEST),
        EligibilityThresholds(**{k: cfg[k] for k in EligibilityThresholds.model_fields}),
        dict(cfg.fractions),
        seeds=[cfg.primary_seed, *cfg.sensitivity_seeds],
        audit_path=AUDIT,
        out_dir=OUT_DIR,
    )
    summary = {
        "manifest_sha256": manifest_digest,
        "eligible_classes": result.audit.classes_after,
        "eligible_images": result.audit.images_after,
        "partition_counts": {str(s): c for s, c in result.partition_counts.items()},
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {AUDIT.relative_to(REPO_ROOT)}")
    for rule in result.audit.rules:
        print(
            f"  {rule.rule}: {rule.classes_before} -> {rule.classes_after} classes, "
            f"{rule.images_dropped} images dropped"
        )
    print(json.dumps(summary, indent=2))
    for out in result.outputs.values():
        print(f"wrote {out.manifest.relative_to(REPO_ROOT)} ({out.manifest.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
