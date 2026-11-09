# VMMR prototype recognition (AVR)

Working title: *Fine-Grained Vehicle Generation Recognition on CompCars with Frozen Visual Encoders and Viewpoint-Aware Prototypes*.

Fine-grained vehicle make, model and generation recognition on CompCars using frozen visual encoders and prototype-based retrieval.

- Protocol: `PROTOCOL.md` (v1.2, see `docs/adr/`)
- Vocabulary: `GLOSSARY.md`
- Decisions: `docs/adr/`
- Dataset licence and manifest publication policy: `docs/protocols/compcars-licence.md`
- Duplicate groups and the pHash audit: `reports/dedup/README.md` (built by `scripts/build_image_manifest.py`) and `reports/dedup/AUDIT.md`
- Track A official split and nested validation partition: `data/splits/track_a/` and `reports/splits/track_a/README.md` (built by `scripts/build_track_a_split.py`)
- Work tracking: GitHub issues, milestones M0 to M6

CompCars images are used under the dataset's non-commercial research licence and are never committed or redistributed; public manifests carry an opaque `image_id` and SHA-256 only (see `docs/protocols/compcars-licence.md`).
