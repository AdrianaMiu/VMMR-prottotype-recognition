"""Duplicate grouping (PROTOCOL.md §7 leakage controls, issue M1-8).

Hamming distances in the hand-written cases are counted on paper from the hex digits;
the synthetic fixture only states which images are copies of which, and the grouping
is asserted at the threshold that configs/dedup records.
"""

from __future__ import annotations

import numpy as np

from avr.data.dedup import UnionFind, hamming_pairs


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
