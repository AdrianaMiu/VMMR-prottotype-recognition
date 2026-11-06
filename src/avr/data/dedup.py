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

import numpy as np


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
