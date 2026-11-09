# Track B split manifests (PROTOCOL.md §5 Track B)

Built by `scripts/build_track_b_splits.py` from `data/manifests/compcars_web_images.parquet`
with the thresholds, fractions and seeds in `configs/splits/track_b.yaml`; the eligibility
audit is `reports/splits/track_b/eligibility_audit.md`. One table per seed:
`seed_73.parquet` is the primary split, `seed_13`, `seed_37`, `seed_101` and `seed_137`
the sensitivity splits. Stored, never regenerated implicitly: the Parquet files are
larger than the 2 MB commit limit, so only the `.sha256` sidecars and `summary.json`
(partition sizes, manifest hash) are committed; `tests/regression/test_track_b_split_identity.py`
rebuilds every seed and compares it byte for byte with its sidecar.

Columns (opaque ids and hashes only, no paths; ADR-0002): `image_id`, `sha256`,
`class_id`, `make_id`, `model_id`, `viewpoint`, `partition` (`support`, `validation`,
`test`), `mapping_status`. `class_id` is the provisional model-year stand-in
`make_id/model_id/year` and `mapping_status` is `provisional` until approved
generation-mapping rows exist. Every `source_group_id` group of the image manifest lies
in one partition. Images with the uncertain viewpoint `-1` are split like the others but
are not query-eligible; filter on `viewpoint` for query sets and for the cross-view
condition (ADR-0004).

## Cross-view variants (ADR-0004)

Built by `scripts/build_cross_view_splits.py` from the stored seed tables; the per-seed
eligibility tables are `reports/splits/track_b/cross_view/seed_<s>.md`, written before any
variant. `cross_view/seed_<s>/view_<v>.parquet` is the variant with held-out viewpoint `v`
(1..5): support keeps the rows of the other known viewpoints, validation and test keep the
rows of viewpoint `v`. Rows are filtered, never re-assigned, so every row keeps the
partition it has in `seed_<s>.parquet` and the parent's group-aware guarantees carry over.
Rows with the uncertain viewpoint `-1` are excluded from every partition of every variant.
A class stays in a variant only with >= 5 support and >= 1 test rows after filtering
(`cross_view` in `configs/splits/track_b.yaml`). Same columns as the parent tables. As
with the parents, only the `.sha256` sidecars and `cross_view/summary.json` (parent
hashes, classes and images kept per view) are committed;
`tests/regression/test_cross_view_split_identity.py` rebuilds every variant and compares
it byte for byte with its sidecar.
