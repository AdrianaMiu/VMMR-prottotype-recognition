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
