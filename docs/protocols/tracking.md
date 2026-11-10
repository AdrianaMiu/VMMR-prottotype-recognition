# Experiment tracking: run manifest, MLflow and DVC

Implements PROTOCOL.md §17 and ADR-0002 (local MLflow file store, DVC-versioned manifests).
Rule from GLOSSARY.md: **no manifest, no metric**.

## Run manifest

`avr.tracking.manifest.RunManifest` is a Pydantic model whose fields are exactly the §17 list,
in protocol order:

```text
run_id, git_commit, dirty_worktree_flag, dataset_protocol, dataset_manifest_hash,
split_manifest_hash, generation_mapping_hash, encoder_id, encoder_revision, encoder_checksum,
preprocessing_id, dimension_transform_id, prototype_method, support_k, seed, hardware,
software_versions, start_time, end_time
```

Every field defaults to `None` so a manifest can be assembled incrementally; unknown fields are
rejected. `RunManifest.missing_fields()` lists the required fields that are still `None`
(everything except `end_time`, which is written when the run closes).

`capture_environment(repo)` fills the environment half of the manifest:

| Field | Source |
|---|---|
| `git_commit` | `git rev-parse HEAD` |
| `dirty_worktree_flag` | `git status --porcelain` non-empty |
| `hardware` | CPU model from `/proc/cpuinfo`; GPU name from `nvidia-smi --query-gpu=name`, `null` when absent |
| `software_versions` | `python`, `numpy`, `pandas`, `avr`; `torch` only when importable |
| `start_time` | UTC now |

The experiment half (`dataset_*`, `split_*`, `generation_mapping_hash`, `encoder_*`,
`preprocessing_id`, `dimension_transform_id`, `prototype_method`, `support_k`, `seed`) is
supplied by the experiment driver from the composed Hydra configuration and the manifest hashes.

## MLflow

`avr.tracking.mlflow_run.tracked_run(manifest, tracking_dir=..., experiment_name=...)` is a
context manager that:

1. points MLflow at a **local file store** (`file://<tracking_dir>`), creating the directory;
2. opens a run named after `manifest.run_id`;
3. logs every manifest field as a parameter (`None` is logged as the string `"None"`);
4. writes the manifest as the artifact `manifest.json` on entry and again on exit, once
   `end_time` is set;
5. yields a `TrackedRun` whose `log_metric(name, value, step=None)` **raises
   `IncompleteManifestError`** while any required field is `None`.

The store location comes from the `tracking` config group (`configs/tracking/local.yaml`,
selected in `configs/config.yaml`):

```yaml
backend: local_file
tracking_dir: mlruns        # relative to the repository root, git-ignored
experiment_name: avr
```

`tracking_dir_from_config(cfg.tracking)` resolves it. Browse runs with
`uv run mlflow ui --backend-store-uri file://$PWD/mlruns`.

Tests: `tests/unit/test_run_manifest.py`, `tests/unit/test_mlflow_run.py` (metric without a
complete manifest raises) and `tests/integration/test_mlflow_tracking.py` (a dummy run leaves
a FINISHED MLflow run whose `git_commit` parameter equals `git rev-parse HEAD` and whose
artifacts contain `manifest.json`). Both use a `tmp_path` store, never `mlruns/`.

## DVC

DVC is initialised in the repository (`.dvc/config`, `.dvcignore` are tracked; `.dvc/cache/`
and `.dvc/tmp/` are ignored). Settings:

- `core.analytics = false`
- cache: the DVC default `.dvc/cache` under the repository (git-ignored); no remote is configured

What is DVC-tracked and when:

- Small JSON records under `data/manifests/` (for example
  `data/manifests/track_a_split_identity.json`, the `*.summary.json` files) and every
  `.sha256` sidecar stay in git: they are a few kilobytes and must be reviewable in diffs.
- Parquet tables larger than the 2 MB commit limit are owned by DVC. `uv run dvc add <file>`
  writes the `<file>.dvc` pointer (md5 of the content) next to the file; the pointer is
  committed, the bytes go to `.dvc/cache`. DVC adds a per-directory `.gitignore` entry only
  for a file git does not already ignore; the root rule below ignores every Parquet, so none
  is written. Currently DVC-tracked:
  - `data/manifests/compcars_web_images.parquet` (public image manifest);
  - `data/manifests/p3_mask_metadata.parquet` (P3 mask metadata, full run);
  - `data/splits/track_b/seed_<s>.parquet` and `data/splits/track_b/cross_view/seed_<s>/view_<v>.parquet`;
  - `data/prototypes/<encoder>/<variant>/*.parquet` (M1 gate prototype tables).
  The private path table `data/paths/compcars_web_image_paths.parquet` is **never** added to
  DVC or git (ADR-0002); it is rebuilt from the dataset by `scripts/build_image_manifest.py`.
  `data/splits/track_a/official_cvpr2015.parquet` is under 2 MB and stays in git.
- The root `.gitignore` ignores `*.parquet` and re-allows only
  `data/splits/track_a/*.parquet` and `tests/fixtures/**`, so a DVC-owned Parquet is ignored
  by git while its `.dvc` pointer and `.sha256` sidecar are not (`.dvc/` itself is re-allowed
  by `!/.dvc/` after the `/.*/` rule). `tests/unit/test_gitignore_policy.py` asks git itself.
- Prototype tables are written by the M1 gate (`scripts/run_m1_gate.py`) and added with
  `uv run dvc add data/prototypes/<encoder>/<variant>/*.parquet` after each gate run; their
  `.sha256` sidecars stay in git.
- The `dataset_manifest_hash`, `split_manifest_hash` and `generation_mapping_hash` fields of
  the run manifest are the SHA-256 of the Parquet bytes (the `.sha256` sidecars), recorded
  independently of DVC's md5. The M1 gate verifies every table it reads against its sidecar.
- Image data, masks and embedding shards are never added to DVC or git (CompCars licence,
  PROTOCOL.md §17); embeddings are rebuilt by `scripts/extract_embeddings.py`.

To restore tracked files on a fresh clone with the same cache: `uv run dvc checkout`. A remote
(`dvc remote add`) will be configured when a second machine needs the files.

## Run manifest fields that do not apply

Every §17 field is required before a metric is logged, so a run that has no value for a
field records an explicit placeholder rather than `null`:

- `not_applicable` (string) for `split_manifest_hash`, `generation_mapping_hash` and
  `prototype_method` of an extraction run (`scripts/extract_embeddings.py`), and for
  `generation_mapping_hash` of a Track A gate run (Track A has no generation mapping);
- `support_k = 0` for an extraction run (no support set exists) and `support_k = "full"` for a
  gate run that builds prototypes from every support image;
- Inputs outside the §17 list (for example the P3 `mask_metadata_sha256`) are logged as extra
  MLflow parameters (`TrackedRun.log_param`) and written into the result JSON, never into the
  manifest model, whose fields are exactly the §17 list.
