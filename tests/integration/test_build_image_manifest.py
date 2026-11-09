"""Full manifest build on the synthetic CompCars tree (PROTOCOL.md §7, issue M1-7).

Expected values are hand-written from tests/conftest.py; only the SHA-256 of each JPEG is
computed here (with hashlib, independently of the code under test) because Pillow decides
the bytes.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from avr.data.dedup import deduplicate
from avr.data.manifest import (
    PRIVATE_COLUMNS,
    PUBLIC_COLUMNS,
    build_manifest,
    write_manifest,
)
from conftest import SYNTHETIC_IMAGES


def sha_of(root: Path, stem: str) -> str:
    return hashlib.sha256((root / "image" / f"{stem}.jpg").read_bytes()).hexdigest()


@pytest.fixture
def built(synthetic_compcars: Path):
    rows = build_manifest(synthetic_compcars, workers=2)
    return synthetic_compcars, {row.relative_path: row for row in rows}


def test_one_row_per_image_with_hash_derived_id(built):
    root, by_path = built

    assert len(by_path) == len(SYNTHETIC_IMAGES)
    row = by_path["image/1/1/2010/aaa.jpg"]
    assert row.sha256 == sha_of(root, "1/1/2010/aaa")
    assert row.image_id == row.sha256[:16]
    assert (row.width, row.height) == (8, 8)
    assert (row.make_id, row.model_id, row.year) == (1, 1, 2010)
    assert (row.make_name, row.model_name) == ("ABT", "Audi A3 hatchback")
    assert row.viewpoint == 4
    assert row.bbox_xyxy == (1, 1, 6, 6)
    assert (row.validity_status, row.exclusion_reason) == ("valid", None)
    assert row.phash is None and row.generation_id is None and row.mapping_version is None
    assert row.split_track_a is None and row.duplicate_group_id is None


def test_unknown_year_folder_is_excluded_with_null_year(built):
    row = built[1]["image/1/1/unknown/bbb.jpg"]

    assert row.year is None
    assert (row.validity_status, row.exclusion_reason) == ("excluded", "unknown_year")
    assert row.viewpoint == 1  # CRLF on the viewpoint line


def test_typo_year_folder_is_excluded_as_invalid(built):
    row = built[1]["image/1/1/5008/ccc.jpg"]

    assert row.year is None
    assert (row.validity_status, row.exclusion_reason) == ("excluded", "invalid_year_folder")
    assert (row.width, row.height) == (10, 6)


def test_degenerate_bbox_is_excluded_and_names_are_trimmed(built):
    row = built[1]["image/2/3/2014/ddd.jpg"]

    assert (row.validity_status, row.exclusion_reason) == ("excluded", "degenerate_bbox")
    assert row.bbox_xyxy == (2, 2, 2, 2)
    assert row.viewpoint == -1
    assert (row.make_name, row.model_name) == ("Lamorghini", "Hector")


def test_empty_model_name_slot_is_excluded_with_null_name(built):
    row = built[1]["image/2/2/2012/eee.jpg"]

    assert row.model_name is None
    assert row.make_name == "Lamorghini"
    assert (row.validity_status, row.exclusion_reason) == ("excluded", "empty_model_name")
    assert (row.width, row.height) == (6, 10)


def test_write_splits_public_manifest_from_private_path_table(built, tmp_path: Path):
    root, by_path = built
    rows = list(by_path.values())
    out = write_manifest(rows, manifests_dir=tmp_path / "m", paths_dir=tmp_path / "p")

    public = pq.read_table(out.manifest)
    private = pq.read_table(out.paths)
    assert public.column_names == list(PUBLIC_COLUMNS)
    assert "relative_path" not in public.column_names
    assert private.column_names == list(PRIVATE_COLUMNS)
    ids = public.column("image_id").to_pylist()
    assert ids == sorted(ids)
    assert dict(zip(*(private.column(c).to_pylist() for c in PRIVATE_COLUMNS), strict=True)) == {
        row.image_id: row.relative_path for row in rows
    }
    manifest_sha = hashlib.sha256(out.manifest.read_bytes()).hexdigest()
    assert out.sha256.read_text().split()[0] == manifest_sha
    summary = json.loads(out.summary.read_text())
    assert summary["row_count"] == 5
    assert summary["valid_count"] == 1
    assert summary["exclusion_reason_counts"] == {
        "degenerate_bbox": 1,
        "empty_model_name": 1,
        "invalid_year_folder": 1,
        "unknown_year": 1,
    }
    assert summary["viewpoint_counts"] == {"-1": 1, "1": 1, "2": 1, "4": 1, "5": 1}


def test_two_builds_are_byte_identical(synthetic_compcars: Path, tmp_path: Path):
    outputs = []
    for i in range(2):
        rows = build_manifest(synthetic_compcars, workers=2)
        out = write_manifest(rows, manifests_dir=tmp_path / f"m{i}", paths_dir=tmp_path / f"p{i}")
        outputs.append(out)

    assert outputs[0].manifest.read_bytes() == outputs[1].manifest.read_bytes()
    assert outputs[0].paths.read_bytes() == outputs[1].paths.read_bytes()


def test_byte_identical_files_get_distinct_ids_from_the_same_hash(synthetic_compcars: Path):
    # the release holds 52 byte-identical pairs, 50 of them across two models
    source = synthetic_compcars / "image" / "1/1/2010/aaa.jpg"
    copy = synthetic_compcars / "image" / "2/3/2014/aaa.jpg"
    copy.write_bytes(source.read_bytes())
    (synthetic_compcars / "label" / "2/3/2014/aaa.txt").write_bytes(b"3\n1\n0 0 7 7\n")

    rows = {row.relative_path: row for row in build_manifest(synthetic_compcars, workers=2)}

    first, second = rows["image/1/1/2010/aaa.jpg"], rows["image/2/3/2014/aaa.jpg"]
    assert first.sha256 == second.sha256
    assert first.image_id == first.sha256[:16]
    assert second.image_id == first.sha256[:16] + "-1"
    assert len({row.image_id for row in rows.values()}) == 6


def test_deduplicate_fills_phash_from_the_release_tree(synthetic_compcars: Path):
    rows = build_manifest(synthetic_compcars, workers=2)

    result = deduplicate(rows, synthetic_compcars, threshold=6, hash_size=8, workers=2)

    assert [r.image_id for r in result.rows] == [r.image_id for r in rows]
    assert all(len(r.phash) == 16 and int(r.phash, 16) >= 0 for r in result.rows)
    # flat single-colour JPEGs all hash to the same pHash, so they form one group
    assert {r.duplicate_group_id for r in result.rows} == {min(r.image_id for r in rows)}
    assert result.stats.exact_duplicate_groups == 0
    assert result.stats.near_duplicate_images == 5
