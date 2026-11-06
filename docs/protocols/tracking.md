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
- cache: the DVC default `.dvc/cache` under the repository (git-ignored); no remote is configured yet

What is DVC-tracked and when:

- Small JSON records under `data/manifests/` (for example
  `data/manifests/track_a_split_identity.json`) stay in git: they are a few kilobytes and must
  be reviewable in diffs.
- Parquet manifests under `data/manifests/` (the dataset manifest, split manifests and the
  generation mapping once they exist, PROTOCOL.md §5-§6 and ADR-0002) are added with
  `uv run dvc add data/manifests/<name>.parquet`. That commits the `<name>.parquet.dvc`
  pointer file (md5 of the content) to git and places the bytes in `.dvc/cache`. The
  `dataset_manifest_hash`, `split_manifest_hash` and `generation_mapping_hash` fields of the run
  manifest are the SHA-256 of the Parquet bytes, recorded independently of DVC's md5.
- Derived artifacts (embedding caches, prototype tables) follow the same rule once they are
  produced.
- Image data is never added to DVC or git (CompCars licence, PROTOCOL.md §17).

To restore tracked files on a fresh clone with the same cache: `uv run dvc checkout`. A remote
(`dvc remote add`) will be configured when a second machine needs the files.
