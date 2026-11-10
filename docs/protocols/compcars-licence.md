# CompCars licence note and manifest publication policy

**Short form:** CompCars is licensed for non-commercial research only and may not be
redistributed. This repository therefore never contains images, and every manifest that can be
published carries an opaque `image_id` and a SHA-256 hash instead of a dataset path. The
`image_id -> relative_path` table lives under `data/paths/`, which is git-ignored.

Recorded 2026-11-06 (M0-2). Protocol references: `PROTOCOL.md` §7 (last bullet of the leakage
controls), §17 (reproducibility), §21 item 7; decision: `docs/adr/0002-engineering-defaults.md`.

## 1. Licence terms

Source: the official dataset page, <https://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/>, accessed
2026-11-06. The page carries the terms inline; there is no separate licence file or agreement
form. The relevant sentences, quoted verbatim:

> The CompCars database is available for non-commercial research purposes only.
>
> You agree not to reproduce, duplicate, copy, sell, trade, resell or exploit for any commercial
> purposes, any portion of the images and any portion of derived data.
>
> You agree not to further copy, publish or distribute any portion of the CompCars database.
> Except, for internal use at a single site within the same organization it is allowed to make
> copies of the database.
>
> All images of the CompCars database are obtained from the Internet which are not property of
> MMLAB, The Chinese University of Hong Kong. The MMLAB is not responsible for the content nor
> the meaning of these images.
>
> The MMLAB reserves the right to terminate your access to the database at any time.
>
> All submitted papers or any publicly available text using the CompCars database must cite the
> following paper: [...] A Large-Scale Car Dataset for Fine-Grained Categorization and
> Verification, Linjie Yang, Ping Luo, Chen Change Loy, Xiaoou Tang, CVPR 2015.

Reading applied in this project:

| Term | What it means for us |
|---|---|
| Non-commercial research only | The study, the prototype service and every artefact are research use. No commercial deployment of anything derived from the images. |
| No copying, publishing or distributing any portion of the database; no commercial exploitation of images or derived data | No image, crop, mask, bounding-box render or thumbnail of a CompCars image is committed, attached to an issue, published with the paper or served outside the single-site deployment. Embeddings, prototypes and indices built from the images are derived data and stay local as well. |
| Single-site internal copies allowed | The local copy under `COMPCARS_ROOT` (see `configs/datasets`) and per-run caches on the same machine are permitted. |
| Mandatory citation | Yang et al., CVPR 2015 is cited in every paper, report or public text produced by this project (`PROTOCOL.md` §28). |

The licence says nothing explicit about *metadata* such as file names or directory paths. The
dataset's relative paths encode make, model and year (`<make_id>/<model_id>/<year>/<file>.jpg`)
and are part of the distributed database, so `PROTOCOL.md` §7 asks whether they may be
published. We treat them as "a portion of the CompCars database" and do not publish them. This
is the conservative reading; it costs nothing because hashes identify the images just as well.

## 2. Publication policy (ADR-0002)

1. **No image redistribution.** Images, derived images (P1-P3 preprocessing variants,
   segmentation masks, crops) and feature caches stay under git-ignored directories
   (`data/external/`, `outputs/`, `mlruns/`). The CI never has access to the dataset.
2. **Public manifests carry `image_id` and `sha256` only.** Every manifest that is committed,
   versioned with DVC, attached to a paper or shared with another group identifies an image by
   an opaque `image_id` and the SHA-256 of the original file bytes. Labels, split membership,
   viewpoint, bounding boxes, duplicate groups and mapping versions (the §7 fields) may appear;
   `relative_path` may not.
3. **The path table is private.** The `image_id -> relative_path` mapping lives under
   `data/paths/`, which `.gitignore` excludes. It is regenerated locally from the dataset copy
   and is the only place where a path is joined to an identifier.
4. **Split and mapping metadata are publishable** under the same rule (`PROTOCOL.md` §21 item
   7): the Track A split identity in `data/manifests/track_a_split_identity.json` stores hashes
   of the distributed split files, never their contents; Track B split manifests will store
   `image_id` and `sha256` per row.
5. **Reproduction by a licensed third party** works by obtaining CompCars from the official
   page, hashing their copy and joining on `sha256`. The `image_id` is opaque by design: it
   carries no information about the path, and the hash is what ties it to a file.

Where the rules live:

| Rule | Enforced by |
|---|---|
| `data/paths/` and `data/external/` are never committed | `.gitignore`; `tests/unit/test_gitignore_policy.py` creates dummy files there and asserts `git check-ignore` rejects them |
| `data/manifests/*.json` and `data/manifests/**/*.parquet` are committable | same test, negative case |
| No path strings leak through split-identity records | `tests/unit/test_split_identity.py` asserts no file name appears in the serialised record |
| Public manifest schema has no `relative_path` column | `tests/integration/test_build_image_manifest.py` asserts the public Parquet columns equal `PUBLIC_COLUMNS` and that the path table is written separately under `data/paths/` |

## 3. Checklist before anything leaves the machine

- The artefact contains no image bytes or renders of CompCars images.
- Every row is keyed by `image_id` and `sha256`; no `relative_path` column, no file names.
- The citation to Yang et al. (CVPR 2015) is present in the accompanying text.
- The artefact is described as research output; nothing implies commercial use.
