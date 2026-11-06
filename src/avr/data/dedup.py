"""Duplicate groups for the image manifest (PROTOCOL.md §7 leakage controls).

Two images belong to one duplicate group when their file bytes are identical (same
``sha256``) or their 64-bit pHash values differ in at most ``phash_hamming_threshold``
bits (configs/dedup). Links are closed transitively with union-find, so a group is a
connected component of the "exact or near" relation. ``duplicate_group_id`` is the
smallest ``image_id`` of the component; singletons carry their own ``image_id``.
``source_group_id`` is set equal to ``duplicate_group_id`` because no same-source series
are known yet.

The near-duplicate search is an exact all-pairs scan: hashes are packed as ``uint64``
and compared block by block with a vectorised popcount, which takes about a minute for
the 136k-image release and needs no index structure.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import imagehash
import numpy as np
from PIL import Image
from pydantic import BaseModel

from avr.data.manifest import ImageRow

PHASH_HEX = 16
# Pair distances are tallied up to this bound so the report shows the tail past the threshold.
HISTOGRAM_MAX_DISTANCE = 16


class UnionFind:
    """Disjoint sets over ``0..n-1`` with path compression; ``groups`` is deterministic."""

    def __init__(self, n: int) -> None:
        self._parent = list(range(n))

    def find(self, i: int) -> int:
        root = i
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[i] != root:
            self._parent[i], i = root, self._parent[i]
        return root

    def union(self, i: int, j: int) -> None:
        ri, rj = self.find(i), self.find(j)
        if ri != rj:
            self._parent[max(ri, rj)] = min(ri, rj)

    def groups(self) -> list[list[int]]:
        """Members of every set, each sorted, ordered by smallest member."""
        members: dict[int, list[int]] = {}
        for i in range(len(self._parent)):
            members.setdefault(self.find(i), []).append(i)
        return [members[root] for root in sorted(members)]


def hamming_pairs(
    hashes: np.ndarray, max_distance: int, block: int = 512
) -> list[tuple[int, int, int]]:
    """Every unordered pair ``(i, j, distance)`` with ``i < j`` and distance <= bound."""
    hashes = np.ascontiguousarray(hashes, dtype=np.uint64)
    n = len(hashes)
    out: list[tuple[int, int, int]] = []
    for start in range(0, n, block):
        rows = hashes[start : start + block, None]
        distance = np.bitwise_count(rows ^ hashes[None, start:])
        local_i, local_j = np.nonzero(distance <= max_distance)
        keep = local_j > local_i
        i = local_i[keep] + start
        j = local_j[keep] + start
        d = distance[local_i[keep], local_j[keep]]
        out.extend(zip(i.tolist(), j.tolist(), d.tolist(), strict=True))
    return out


def _phash_worker(args: tuple[str, int]) -> str | None:
    path, hash_size = args
    try:
        with Image.open(path) as image:
            return str(imagehash.phash(image, hash_size=hash_size))
    except OSError:
        return None


def compute_phashes(
    paths: Sequence[Path], hash_size: int = 8, workers: int | None = None
) -> list[str | None]:
    """pHash of each full image as ``hash_size**2 / 4`` hex digits; None when unreadable."""
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_phash_worker, ((str(p), hash_size) for p in paths), chunksize=64))


def phash_to_uint64(phashes: Sequence[str]) -> np.ndarray:
    return np.array([int(h, 16) for h in phashes], dtype=np.uint64)


def near_pairs(
    phashes: Sequence[str | None], max_distance: int = HISTOGRAM_MAX_DISTANCE
) -> list[tuple[int, int, int]]:
    """``(i, j, distance)`` over the positions that carry a hash, distance <= bound."""
    hashed = [i for i, h in enumerate(phashes) if h is not None]
    if not hashed:
        return []
    values = phash_to_uint64([phashes[i] for i in hashed])  # type: ignore[misc]
    return [(hashed[a], hashed[b], d) for a, b, d in hamming_pairs(values, max_distance)]


def link_duplicates(
    sha256: Sequence[str], pairs: Sequence[tuple[int, int, int]], threshold: int
) -> UnionFind:
    """Union images with the same bytes and pairs whose distance is <= threshold."""
    uf = UnionFind(len(sha256))
    first_by_sha: dict[str, int] = {}
    for i, digest in enumerate(sha256):
        if digest in first_by_sha:
            uf.union(first_by_sha[digest], i)
        else:
            first_by_sha[digest] = i
    for a, b, d in pairs:
        if d <= threshold:
            uf.union(a, b)
    return uf


def group_images(
    image_ids: Sequence[str],
    sha256: Sequence[str],
    phashes: Sequence[str | None],
    threshold: int,
) -> list[str]:
    """``duplicate_group_id`` per image: the smallest ``image_id`` of its component."""
    uf = link_duplicates(sha256, near_pairs(phashes, threshold), threshold)
    group_id = [""] * len(image_ids)
    for members in uf.groups():
        smallest = min(image_ids[i] for i in members)
        for i in members:
            group_id[i] = smallest
    return group_id


class DedupStats(BaseModel):
    phash_hash_size: int
    phash_hamming_threshold: int
    image_count: int
    hashed_image_count: int
    exact_duplicate_groups: int
    exact_duplicate_images: int
    near_duplicate_groups: int
    near_duplicate_images: int
    duplicate_groups: int
    duplicate_images: int
    largest_group_size: int
    groups_crossing_model: int
    groups_crossing_make: int
    histogram: dict[int, int]


class NearThresholdPair(BaseModel):
    image_id_a: str
    image_id_b: str
    distance: int
    same_model: bool
    make_model_a: str
    make_model_b: str


class DedupResult(BaseModel):
    rows: list[ImageRow]
    stats: DedupStats
    near_threshold_pairs: list[NearThresholdPair]


def _make_model(row: ImageRow) -> str:
    return f"{row.make_id}/{row.model_id}"


def assign_duplicate_groups(
    rows: Sequence[ImageRow], threshold: int, hash_size: int = 8
) -> DedupResult:
    """Fill ``duplicate_group_id`` and ``source_group_id`` from ``sha256`` and ``phash``.

    ``phash`` must already be set (``compute_phashes``); rows without one only join
    exact groups. Rows come back in input order.
    """
    sha256 = [r.sha256 for r in rows]
    pairs = near_pairs([r.phash for r in rows])
    uf = link_duplicates(sha256, pairs, threshold)
    group_id = [""] * len(rows)
    groups = uf.groups()
    for members in groups:
        smallest = min(rows[i].image_id for i in members)
        for i in members:
            group_id[i] = smallest
    updated = [
        r.model_copy(update={"duplicate_group_id": g, "source_group_id": g})
        for r, g in zip(rows, group_id, strict=True)
    ]

    exact_sizes = Counter(sha256)
    exact_groups = [n for n in exact_sizes.values() if n > 1]
    multi = [m for m in groups if len(m) > 1]
    # a group is "near" when it holds more than one distinct byte content
    near = [m for m in multi if len({sha256[i] for i in m}) > 1]
    histogram = Counter(d for _, _, d in pairs)
    stats = DedupStats(
        phash_hash_size=hash_size,
        phash_hamming_threshold=threshold,
        image_count=len(rows),
        hashed_image_count=sum(r.phash is not None for r in rows),
        exact_duplicate_groups=len(exact_groups),
        exact_duplicate_images=sum(exact_groups),
        near_duplicate_groups=len(near),
        near_duplicate_images=sum(len(m) for m in near),
        duplicate_groups=len(multi),
        duplicate_images=sum(len(m) for m in multi),
        largest_group_size=max((len(m) for m in multi), default=1),
        groups_crossing_model=sum(len({_make_model(rows[i]) for i in m}) > 1 for m in multi),
        groups_crossing_make=sum(len({rows[i].make_id for i in m}) > 1 for m in multi),
        histogram={d: histogram[d] for d in sorted(histogram)},
    )
    near_threshold = sorted(
        (
            NearThresholdPair(
                image_id_a=min(rows[a].image_id, rows[b].image_id),
                image_id_b=max(rows[a].image_id, rows[b].image_id),
                distance=d,
                same_model=_make_model(rows[a]) == _make_model(rows[b]),
                make_model_a=_make_model(rows[min(a, b, key=lambda k: rows[k].image_id)]),
                make_model_b=_make_model(rows[max(a, b, key=lambda k: rows[k].image_id)]),
            )
            for a, b, d in pairs
            if d in (threshold - 1, threshold)
        ),
        key=lambda p: (p.distance, p.image_id_a, p.image_id_b),
    )
    return DedupResult(rows=updated, stats=stats, near_threshold_pairs=near_threshold)
