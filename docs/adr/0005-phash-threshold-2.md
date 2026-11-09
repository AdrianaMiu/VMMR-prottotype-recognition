---
status: accepted
date: 2026-11-09
---

# Near-duplicate grouping uses pHash Hamming distance 2

PROTOCOL.md §7 requires perceptual duplicate grouping with a recorded threshold and a visual audit near it. The first run used Hamming distance 6 on a 64-bit pHash. The audit (`reports/dedup/AUDIT.md`) found that distance 6 links different cars that merely share a composition, chaining 695 images from 379 models into one duplicate group, which would force them into one split partition and distort class stratification. At distance 2 the sampled pairs are the same photograph re-encoded or trivially re-cropped, the largest group has 6 images, and 998 groups cover 2,065 images; distance 4 already pairs different cars in the same studio layout.

Decision: `phash_hamming_threshold` is 2. Exact SHA-256 duplicates are always grouped regardless of distance. Because the hash has exactly 32 set bits, only even distances occur, so 2 means "distance 0 or 2". The remaining risk, true duplicates at distance 4 or more, is accepted and stays visible through the committed threshold sweep. Distance-0 pairs filed under two different models are label noise and are counted in the dedup report.
