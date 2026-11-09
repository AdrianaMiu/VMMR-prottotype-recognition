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
from avr.data.dedup import (
    PANEL_HEIGHT,
    UnionFind,
    assign_duplicate_groups,
    compute_phashes,
    group_images,
    hamming_pairs,
    write_report,
)
from avr.data.manifest import ImageRow, summarise


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


def row(image_id: str, sha256: str, phash: str, make_id: int, model_id: int) -> ImageRow:
    return ImageRow(
        image_id=image_id,
        relative_path=f"image/{make_id}/{model_id}/2010/{image_id}.jpg",
        sha256=sha256,
        phash=phash,
        width=8,
        height=8,
        make_id=make_id,
        make_name=f"make{make_id}",
        model_id=model_id,
        model_name=f"model{model_id}",
        year=2010,
        viewpoint=1,
        bbox_xyxy=(0, 0, 7, 7),
        validity_status="valid",
        exclusion_reason=None,
    )


# Hand-counted Hamming distances between the hex hashes: a-b 0 (same bytes), a-c 2,
# a-d 8, c-d 6, e is 56..64 bits from every other hash.
HAND_ROWS = [
    row("a", "s1", "0000000000000000", 1, 1),
    row("b", "s1", "0000000000000000", 1, 2),
    row("c", "s2", "0000000000000003", 1, 1),
    row("d", "s3", "00000000000000ff", 2, 3),
    row("e", "s4", "ffffffffffffffff", 2, 3),
]


def test_groups_close_transitively_and_count_cross_label_evidence():
    result = assign_duplicate_groups(HAND_ROWS, threshold=6)

    groups = {r.image_id: r.duplicate_group_id for r in result.rows}
    assert groups == {"a": "a", "b": "a", "c": "a", "d": "a", "e": "e"}
    assert all(r.source_group_id == r.duplicate_group_id for r in result.rows)
    assert result.stats.exact_duplicate_groups == 1
    assert result.stats.exact_duplicate_images == 2
    assert result.stats.near_duplicate_groups == 1
    assert result.stats.near_duplicate_images == 4
    assert result.stats.groups_crossing_model == 1
    assert result.stats.groups_crossing_make == 1
    assert result.stats.histogram == {0: 1, 2: 2, 6: 1, 8: 2}
    assert result.stats.phash_hamming_threshold == 6
    assert result.stats.group_size_histogram == {4: 1}
    # what the grouping would be at every lower threshold: (groups, images, largest, crossing)
    assert [
        (
            s.threshold,
            s.duplicate_groups,
            s.duplicate_images,
            s.largest_group_size,
            s.groups_crossing_model,
        )
        for s in result.stats.threshold_sweep
    ] == [
        (0, 1, 2, 2, 1),
        (1, 1, 2, 2, 1),
        (2, 1, 3, 3, 1),
        (3, 1, 3, 3, 1),
        (4, 1, 3, 3, 1),
        (5, 1, 3, 3, 1),
        (6, 1, 4, 4, 1),
    ]


def test_near_threshold_pairs_are_every_phash_pair_within_the_threshold():
    # ADR-0005: at threshold 2 the audit list holds the distance-0 and distance-2 pairs,
    # so a-b (distance 0, same bytes) is listed before the two distance-2 pairs.
    result = assign_duplicate_groups(HAND_ROWS, threshold=2)

    assert [(p.image_id_a, p.image_id_b, p.distance) for p in result.near_threshold_pairs] == [
        ("a", "b", 0),
        ("a", "c", 2),
        ("b", "c", 2),
    ]
    assert result.near_threshold_pairs[0].same_model is False
    assert result.near_threshold_pairs[1].same_model is True
    assert result.near_threshold_pairs[2].same_model is False
    # a-d (8) and c-d (6) lie above the threshold and are not listed
    assert all(p.distance <= 2 for p in result.near_threshold_pairs)


def test_manifest_summary_counts_multi_image_groups_once_ids_are_assigned():
    assert "duplicate_group_count" not in summarise(HAND_ROWS)

    summary = summarise(assign_duplicate_groups(HAND_ROWS, threshold=6).rows)

    assert summary["duplicate_group_count"] == 1
    assert summary["duplicate_image_count"] == 4


def test_report_writes_readme_csv_and_capped_side_by_side_panels(
    duplicate_images: dict[str, Path], tmp_path: Path
):
    result = assign_duplicate_groups(HAND_ROWS, threshold=2)
    image_paths = dict(zip("abcde", duplicate_images.values(), strict=True))
    out = tmp_path / "report"

    written = write_report(result, image_paths, out, max_pairs=1, runtime_seconds=12.5)

    readme = (out / "README.md").read_text()
    assert "phash_hamming_threshold: 2" in readme
    assert "| 2 | 2 |" in readme  # histogram row: distance 2, two pairs
    csv_lines = (out / "near_threshold_pairs.csv").read_text().splitlines()
    header = "panel,image_id_a,image_id_b,distance,same_model,make_model_a,make_model_b"
    assert csv_lines[0] == header
    assert csv_lines[1:] == [
        "00_a_b_d0.jpg,a,b,0,False,1/1,1/2",
        ",a,c,2,True,1/1,1/1",
        ",b,c,2,False,1/2,1/1",
    ]
    panels = sorted(p.name for p in (out / "pairs").iterdir())
    assert panels == ["00_a_b_d0.jpg"]
    with Image.open(out / "pairs" / panels[0]) as panel:
        assert panel.height == PANEL_HEIGHT
        assert panel.width > 2 * PANEL_HEIGHT * 0.9  # two 128 px squares, scaled, plus gap
    assert written == [out / "README.md", out / "near_threshold_pairs.csv"]
