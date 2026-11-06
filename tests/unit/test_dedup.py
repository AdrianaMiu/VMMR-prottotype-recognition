"""Duplicate grouping (PROTOCOL.md §7 leakage controls, issue M1-8).

Hamming distances in the hand-written cases are counted on paper from the hex digits;
the synthetic fixture only states which images are copies of which, and the grouping
is asserted at the threshold that configs/dedup records.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageEnhance

from avr.config import load_config
from avr.data.dedup import UnionFind, compute_phashes, group_images, hamming_pairs


def test_union_find_groups_are_transitive_without_a_direct_link():
    uf = UnionFind(5)
    uf.union(0, 1)
    uf.union(1, 2)
    uf.union(3, 4)

    assert uf.find(0) == uf.find(2)
    assert uf.find(3) == uf.find(4)
    assert uf.find(0) != uf.find(3)
    assert uf.groups() == [[0, 1, 2], [3, 4]]


def test_hamming_pairs_reports_each_unordered_pair_within_the_bound_once():
    # 0x0 vs 0x3 -> 2 bits; 0x3 vs 0xf -> 2 bits; 0x0 vs 0xf -> 4 bits; the last hash
    # has 15 leading one-bits set, so it is far from all of them.
    hashes = np.array([0x0, 0x3, 0xF, 0x7FFF_0000_0000_0000], dtype=np.uint64)

    pairs = hamming_pairs(hashes, max_distance=3, block=2)

    assert sorted(pairs) == [(0, 1, 2), (1, 2, 2)]
    assert sorted(hamming_pairs(hashes, max_distance=4)) == [(0, 1, 2), (0, 2, 4), (1, 2, 2)]


def textured_image(array: np.ndarray) -> Image.Image:
    arr = np.clip(array, 0, 255).astype(np.uint8)
    return Image.fromarray(np.stack([arr, arr // 2, 200 - arr // 2], axis=-1), "RGB")


@pytest.fixture
def duplicate_images(tmp_path: Path) -> dict[str, Path]:
    """original, a byte-identical copy, a 40 px resize, a brightened copy, an unrelated image."""
    y, x = np.mgrid[0:128, 0:128]
    noise = np.random.default_rng(1).integers(0, 40, (128, 128))
    original = textured_image((x * 2 + y * 1.5) % 200 + 20 + noise)
    unrelated = textured_image((np.sin(x / 5) * np.cos(y / 7)) * 100 + 120)
    names = ("original", "exact", "resized", "bright", "other")
    paths = {name: tmp_path / f"{name}.jpg" for name in names}
    original.save(paths["original"], format="JPEG", quality=90)
    paths["exact"].write_bytes(paths["original"].read_bytes())
    with Image.open(paths["original"]) as saved:
        saved.resize((40, 40)).save(paths["resized"], format="JPEG", quality=90)
        ImageEnhance.Brightness(saved).enhance(1.3).save(paths["bright"], format="JPEG", quality=90)
    unrelated.save(paths["other"], format="JPEG", quality=90)
    return paths


def grouping(paths: dict[str, Path], threshold: int) -> dict[str, str]:
    names = list(paths)
    sha256 = [hashlib.sha256(paths[n].read_bytes()).hexdigest() for n in names]
    phashes = compute_phashes([paths[n] for n in names], hash_size=8, workers=2)
    assert all(len(h) == 16 for h in phashes)
    return dict(zip(names, group_images(names, sha256, phashes, threshold), strict=True))


def test_exact_resized_and_brightened_copies_share_one_group_at_the_config_threshold(
    duplicate_images: dict[str, Path],
):
    threshold = load_config().dedup.phash_hamming_threshold

    groups = grouping(duplicate_images, threshold)

    assert groups["original"] == groups["exact"] == groups["resized"] == groups["bright"]
    assert groups["other"] != groups["original"]
    assert groups["original"] == "bright"  # smallest image_id of the group
    assert groups["other"] == "other"  # singleton keeps its own id


def test_threshold_zero_keeps_only_the_exact_copy_in_the_group(duplicate_images: dict[str, Path]):
    groups = grouping(duplicate_images, 0)

    assert groups["original"] == groups["exact"]
    assert groups["resized"] != groups["original"]
