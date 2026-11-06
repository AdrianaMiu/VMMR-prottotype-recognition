---
status: accepted
date: 2026-11-06
---

# Engineering defaults for the AVR package

The package lives at `phd_vmmr/prototype_app` as its own git repository (remote `AdrianaMiu/VMMR-prottotype-recognition`), managed by `uv` on Python 3.11 with torch 2.7 / CUDA 12.6 pinned in a lock file. Manifests are Parquet, schemas are Pydantic, configuration is Hydra, exact search is FAISS `IndexFlatIP` with matrix-multiplication parity tests, ties break by lexicographic class ID, experiment tracking is a local MLflow file store, and dataset manifests are versioned with DVC. Tickets are GitHub issues on the repo (one milestone per M0-M6, labels `ws:A1..A9`, `track:a`, `track:b`). Public manifests carry opaque image IDs and SHA-256 hashes only; the path table stays untracked under `data/`.

All extraction and evaluation runs on the single local RTX 3080 (10 GB) in fp16 with resumable per-shard embedding stores; there is no server GPU. Existing code in `car-reid` and `project/` is reference only: functions are ported into `src/avr` with tests, never imported, and Elasticsearch is not adopted.

These are the §26 engineer-owned defaults, recorded once so nobody re-litigates them; any of them can be swapped with a superseding ADR.
