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

from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import imagehash
import numpy as np
from PIL import Image

PHASH_HEX = 16


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


def link_duplicates(
    sha256: Sequence[str], phashes: Sequence[str | None], threshold: int
) -> tuple[UnionFind, list[tuple[int, int, int]]]:
    """Union exact (same sha256) and near (Hamming <= threshold) pairs; return the near pairs."""
    uf = UnionFind(len(sha256))
    first_by_sha: dict[str, int] = {}
    for i, digest in enumerate(sha256):
        if digest in first_by_sha:
            uf.union(first_by_sha[digest], i)
        else:
            first_by_sha[digest] = i
    hashed = [i for i, h in enumerate(phashes) if h is not None]
    pairs = []
    if hashed:
        values = phash_to_uint64([phashes[i] for i in hashed])  # type: ignore[misc]
        for a, b, d in hamming_pairs(values, max_distance=threshold):
            uf.union(hashed[a], hashed[b])
            pairs.append((hashed[a], hashed[b], d))
    return uf, pairs


def group_images(
    image_ids: Sequence[str],
    sha256: Sequence[str],
    phashes: Sequence[str | None],
    threshold: int,
) -> list[str]:
    """``duplicate_group_id`` per image: the smallest ``image_id`` of its component."""
    uf, _ = link_duplicates(sha256, phashes, threshold)
    group_id = [""] * len(image_ids)
    for members in uf.groups():
        smallest = min(image_ids[i] for i in members)
        for i in members:
            group_id[i] = smallest
    return group_id
