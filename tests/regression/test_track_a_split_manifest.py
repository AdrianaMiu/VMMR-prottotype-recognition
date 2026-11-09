"""The Track A split manifest must rebuild byte-identically (PROTOCOL.md §5 Track A).

A fresh build from the published image manifest, the private path table and the official
split files is compared with the published ``data/splits/track_a/official_cvpr2015.sha256``;
this runs only where the CompCars release and the path table are present (dataset root from
configs/datasets, COMPCARS_ROOT overrides). The checks on the published file run everywhere.

Expected counts are the official split's own: 16,016 train and 14,939 test lines over 431
make-model classes (data/manifests/track_a_split_identity.json, PROTOCOL.md §5).
"""

import hashlib

import pandas as pd
import pyarrow.parquet as pq
import pytest

from avr.config import CONFIG_DIR, load_config
from avr.data.manifest import MANIFEST_NAME, PATHS_NAME
from avr.data.split_track_a import SPLIT_NAME, build_track_a, read_official_split
from avr.data.splits_io import read_sidecar_sha256

REPO_ROOT = CONFIG_DIR.parent
MANIFEST = REPO_ROOT / "data" / "manifests" / f"{MANIFEST_NAME}.parquet"
PATHS = REPO_ROOT / "data" / "paths" / f"{PATHS_NAME}.parquet"
PUBLISHED = REPO_ROOT / "data" / "splits" / "track_a" / f"{SPLIT_NAME}.parquet"
SIDECAR = PUBLISHED.with_suffix(".sha256")
ROOT = (REPO_ROOT / load_config().dataset.root).resolve()
SPLIT_CFG = load_config("splits/track_a").splits

PUBLIC_COLUMNS = [
    "image_id",
    "sha256",
    "make_id",
    "model_id",
    "split_track_a",
    "track_a_validation",
]


@pytest.mark.skipif(
    not (ROOT.is_dir() and MANIFEST.is_file() and PATHS.is_file()),
    reason="CompCars release or the local manifest is not present on this machine",
)
def test_rebuilt_split_manifest_matches_the_published_hash(tmp_path):
    build = build_track_a(
        pd.read_parquet(MANIFEST),
        pd.read_parquet(PATHS),
        read_official_split(ROOT),
        fraction=SPLIT_CFG.validation_fraction,
        seed=SPLIT_CFG.validation_seed,
    )
    pq.write_table(build.table, tmp_path / "rebuilt.parquet", compression="zstd")

    assert hashlib.sha256((tmp_path / "rebuilt.parquet").read_bytes()).hexdigest() == (
        read_sidecar_sha256(SIDECAR)
    )


def test_published_manifest_matches_its_sidecar_and_carries_no_path():
    assert hashlib.sha256(PUBLISHED.read_bytes()).hexdigest() == read_sidecar_sha256(SIDECAR)
    assert pq.read_schema(PUBLISHED).names == PUBLIC_COLUMNS


def test_published_split_is_the_official_half_half_over_431_classes():
    frame = pd.read_parquet(PUBLISHED)

    assert frame.split_track_a.value_counts().to_dict() == {"support": 16016, "query": 14939}
    classes = frame.groupby("split_track_a")[["make_id", "model_id"]].nunique()
    per_side = frame.groupby("split_track_a").apply(
        lambda d: len(d[["make_id", "model_id"]].drop_duplicates()), include_groups=False
    )
    assert per_side.to_dict() == {"support": 431, "query": 431}
    assert classes.model_id.to_dict() == {"support": 431, "query": 431}
    assert frame.track_a_validation[frame.split_track_a == "query"].isna().all()
    assert set(frame.track_a_validation.dropna()) == {"fit", "validation"}
    assert list(frame.image_id) == sorted(frame.image_id)
