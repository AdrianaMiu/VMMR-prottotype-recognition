# E0 ten-class fixture

Frozen inputs for the E0 pipeline validation (PROTOCOL.md §14 "Stage E0"):
10 make-model classes, 5 support and 3 query images per class, with their P1 embeddings. `avr.pipeline.e0.run_e0` replays manifest validation, M0 prototypes, exact search, ranking and classification metrics and the run manifest on CPU; `tests/integration/test_e0_fixture.py` asserts every metric equals `expected_metrics.json`.

## This is a fixture split, not the Track A official split

The Track A split (issue #9) was not available when the fixture was built. Classes and images were chosen by the rule in `split.json` (`rule`): fixture-only split, not the Track A official split: the classes with the most valid images, then per class a shuffle of the sorted image ids with numpy default_rng(seed), at most one image per duplicate_group_id, first n_support images as support and the next n_query as query.
Seed: 73. No duplicate group spans support and query.

## Files

- `manifest_slice.parquet`: public manifest columns only (no paths) for the fixture images; sha256 `f3daae1176b154132191d7a7177d3dffea7cdc4fa0459cdfb813baf5e9071aa6`.
- `embeddings.parquet`: `image_id` and L2-normalised float32 `embedding` rows.
- `split.json`: support/query image ids per class, selection rule, checksums and the embedding identity.
- `expected_metrics.json`: the metric values recorded when the fixture was built.

## Provenance

- encoder: dinov3_vitb16 (revision 5931719e67bbdb9737e363e781fb0c67687896bc, checkpoint sha256 9a21ac3df0c63839d62612dda6f454d816c25611cc7a52966ed5a5a94921dc8b)
- preprocessing: dinov3_vit16_224_v1, variant p1, compute dtype float16
- source manifest sha256: 24a158622d01d7df4ec699b54c1469524436846ed715b0886ccb4a95936bd532
- built at commit 3ac2bcccbb6ea8015257fd01b7632ef6708734e5 by `scripts/build_e0_fixture.py`

## Classes

- 054_0196
- 077_0127
- 078_0024
- 081_0068
- 081_0070
- 111_1706
- 122_0590
- 140_1181
- 157_1915
- 157_1917

## Recorded metrics

- query_count: 30
- class_count: 10
- tied_queries: 0
- top1: 0.7
- top5: 0.9666666666666667
- top10: 1.0
- top20: 1.0
- mrr: 0.8194444444444444
- mean_rank: 1.5666666666666667
- median_rank: 1.0
- rank_p75: 2.0
- rank_p90: 3.0
- rank_p95: 3.549999999999997
- mean_normalised_rank: 0.06296296296296297
- accuracy: 0.7
- macro_precision: 0.7416666666666666
- macro_recall: 0.7
- macro_f1: 0.6704761904761904
- weighted_precision: 0.7416666666666666
- weighted_recall: 0.7
- weighted_f1: 0.6704761904761906
- balanced_accuracy: 0.7
- classes_without_predicted_positives: 0
