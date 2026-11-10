"""The image manifest must rebuild byte-identically from the release (PROTOCOL.md §7).

A fresh build, including the pHash and duplicate-group columns at the configs/dedup
threshold, is written to a temporary directory and compared with the published
``data/manifests/compcars_web_images.sha256`` and summary; this runs only where the
CompCars release is present (dataset root from configs/datasets, COMPCARS_ROOT overrides)
and takes a few minutes. The summary checks run everywhere.

Parquet bytes depend on the writer, so the published hash is tied to the pyarrow pinned in
uv.lock; a pyarrow upgrade means rebuilding with scripts/build_image_manifest.py and
recording the new hash.
"""

import hashlib
import json
from pathlib import Path

import pytest

from avr.config import CONFIG_DIR, load_config
from avr.data.dedup import deduplicate
from avr.data.manifest import MANIFEST_NAME, build_manifest, write_manifest

REPO_ROOT = CONFIG_DIR.parent
MANIFESTS = REPO_ROOT / "data" / "manifests"
CFG = load_config()
ROOT = (REPO_ROOT / CFG.dataset.root).resolve()

# From the format notes: 136,726 label files, 17 ``unknown`` year folders (343 images),
# one typo folder ``5008`` (8 images), 10 degenerate boxes; the duplicate counts are the
# recorded outcome of the dedup run at the configured threshold (ADR-0005) and change
# only with the threshold or hash.
EXPECTED_SUMMARY = {
    "row_count": 136726,
    "valid_count": 136365,
    "exclusion_reason_counts": {
        "degenerate_bbox": 10,
        "invalid_year_folder": 8,
        "unknown_year": 343,
    },
    "viewpoint_counts": {"-1": 780, "1": 18431, "2": 13513, "3": 23551, "4": 49301, "5": 31150},
    "make_count": 163,
    "model_count": 1716,
    # groups of size >= 2 at phash_hamming_threshold 2 (ADR-0005, reports/dedup/README.md)
    "duplicate_group_count": 998,
    "duplicate_image_count": 2065,
}


def published_sha256() -> str:
    return (MANIFESTS / f"{MANIFEST_NAME}.sha256").read_text().split()[0]


@pytest.mark.skipif(not ROOT.is_dir(), reason="CompCars release not present on this machine")
def test_rebuilt_manifest_matches_the_published_hash(tmp_path: Path):
    result = deduplicate(
        build_manifest(ROOT),
        ROOT,
        threshold=CFG.dedup.phash_hamming_threshold,
        hash_size=CFG.dedup.phash_hash_size,
    )
    out = write_manifest(result.rows, manifests_dir=tmp_path, paths_dir=tmp_path / "p")

    assert hashlib.sha256(out.manifest.read_bytes()).hexdigest() == published_sha256()
    assert json.loads(out.summary.read_text()) == EXPECTED_SUMMARY


def test_published_summary_matches_the_format_notes():
    summary = json.loads((MANIFESTS / f"{MANIFEST_NAME}.summary.json").read_text())

    assert summary == EXPECTED_SUMMARY
    assert len(published_sha256()) == 64
