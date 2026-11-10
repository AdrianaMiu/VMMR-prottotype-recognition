# Visual audit of the pHash grouping (hand-written, first run)

Companion to the generated `README.md`; written after looking at the exported panels
under `pairs/` and at a seed-0 random sample of pairs at distances 0, 2 and 4 drawn from
the same pair list. Numbers refer to the first run (threshold 6, hash_size 8, 136,726
images, 227 s on 32 cores including the 35 s manifest build).

## What the panels show

- Distance 0, distinct bytes (563 pairs, 69 across make/model): every sampled pair is the
  same photograph re-encoded or re-saved. Cross-model pairs at distance 0 are therefore
  the same photo filed under two models, which is label noise (together with the 50 of
  52 byte-identical pairs that also sit in two models).
- Distance 2 (517 pairs, 29 across make/model): same-model pairs are true duplicates
  (identical shot, slightly different compression or crop); the sampled cross-model pairs
  are already different cars photographed with the same layout (side view against the
  same white corrugated wall used by one photo studio in the release).
- Distance 4 (932 pairs, 241 across make/model): same-model pairs are still mostly true
  duplicates; cross-model pairs are different cars in the same composition.
- Distance 6 (3,033 pairs, the exported panels): overwhelmingly different cars sharing a
  composition (profile view, plain background). This is where chaining happens: the
  largest group holds 695 images from 379 models and 96 makes, almost all viewpoint 3
  and 4, linked only through such composition matches.

Only even distances occur because a 64-bit pHash compares DCT coefficients with their
median, so every hash has exactly 32 set bits and two hashes always differ in an even
number of bits. The threshold 6 therefore behaves like "distance in {0, 2, 4, 6}" and
threshold 5 would be identical to 4.

## Reading of the threshold sweep

| threshold | groups | images | largest group | crossing make/model |
|---|---|---|---|---|
| 0 | 595 | 1,199 | 4 | 116 |
| 2 | 998 | 2,065 | 6 | 142 |
| 4 | 1,625 | 3,504 | 28 | 276 |
| 6 | 2,593 | 6,748 | 695 | 628 |

At 6 the grouping is safe for leakage (every true duplicate is linked) but it merges
unrelated cars into one component, so a group-aware split would move 695 images of
379 models together. At 2 the groups are almost all true duplicates and the largest is 6.

## Recommendation (from the first run; applied below)

Lower `phash_hamming_threshold` to 2 (or 4) in `configs/dedup/default.yaml`, or keep 6
and only link pairs that also share `make_id/model_id` for the near case while keeping
cross-model links for distance 0 and exact matches. Either change must be recorded as a
protocol decision because PROTOCOL.md §7 asks for a recorded threshold, and the manifest
hash in `data/manifests/compcars_web_images.sha256` and the summary counts in
`tests/regression/test_image_manifest_identity.py` must be re-recorded after a rebuild.

## Applied (ADR-0005)

`docs/adr/0005-phash-threshold-2.md` accepts the recommendation: `phash_hamming_threshold`
is 2 in `configs/dedup/default.yaml`, exact SHA-256 duplicates stay grouped regardless of
distance, and the manifest was rebuilt (209 s on 32 cores, 136,726 images, all hashed).
The grouping now matches the threshold-2 row of the sweep above:

| quantity | threshold 6 (first run) | threshold 2 (applied) |
|---|---|---|
| duplicate groups (size >= 2) | 2,593 | 998 |
| images in those groups | 6,748 | 2,065 |
| largest group | 695 | 6 |
| groups crossing make/model | 628 | 142 |
| groups crossing make | 470 | 26 |

Of the 998 groups, 52 are byte-identical pairs and 948 hold at least one pHash link between
distinct bytes. `near_threshold_pairs.csv` now lists every linked pHash pair, 615 at distance
0 and 517 at distance 2, and the 50 exported panels are spread over both distances. The
manifest hash in `data/manifests/compcars_web_images.sha256` and the summary counts in
`tests/regression/test_image_manifest_identity.py` were re-recorded.
