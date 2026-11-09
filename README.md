# VMMR prototype recognition (AVR)

Working title: *Fine-Grained Vehicle Generation Recognition on CompCars with Frozen Visual Encoders and Viewpoint-Aware Prototypes*.

Fine-grained vehicle make, model and generation recognition on CompCars using frozen visual encoders and prototype-based retrieval.

- Protocol: `PROTOCOL.md` (v1.2, see `docs/adr/`)
- Vocabulary: `GLOSSARY.md`
- Decisions: `docs/adr/`
- Dataset licence and manifest publication policy: `docs/protocols/compcars-licence.md`
- Duplicate groups and the pHash audit: `reports/dedup/README.md` (built by `scripts/build_image_manifest.py`) and `reports/dedup/AUDIT.md`
- Track A official split and nested validation partition: `data/splits/track_a/` and `reports/splits/track_a/README.md` (built by `scripts/build_track_a_split.py`)
- Qdrant server, Prometheus and Grafana stack: `docker/README.md` (loaded by `scripts/load_qdrant_server.py`)
- Work tracking: GitHub issues, milestones M0 to M6

CompCars images are used under the dataset's non-commercial research licence and are never committed or redistributed; public manifests carry an opaque `image_id` and SHA-256 only (see `docs/protocols/compcars-licence.md`).

## Recognition service (PROTOCOL.md §16)

`src/avr/service/` serves per-vehicle ranked results on scene images: RF-DETR detection and
masks -> P1 or P3 crop per vehicle -> one batched frozen-encoder call -> prototype search ->
`POST /v1/recognize`. Profiles (`configs/service/profiles.yaml`) name the track, encoder,
variant and prototype method and map to the §12 collection; `index_version` is the SHA-256 of
the prototype table. Settings come from `AVR_*` environment variables (`avr.service.settings`);
run locally with `uv run uvicorn avr.service.app:create_app --factory` and build the image with
`docker/Dockerfile.service`. Cosine similarities are similarities, never probabilities;
`calibrated_confidence` and `viewpoint` are `null` until those components exist.
