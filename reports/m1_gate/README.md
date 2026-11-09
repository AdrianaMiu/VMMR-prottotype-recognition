# M1 acceptance gate (issue M1-16, PROTOCOL.md §25 M1)

DINOv3 ViT-B/16 (`configs/encoders/dinov3_vitb16.yaml`), P1 and P3 preprocessing, M0 global-mean prototypes built from support rows only, exact cosine search (matrix multiplication asserted equal to FAISS `IndexFlatIP`), over every Track A class on the official split and the 2,645 currently eligible Track B classes on seed 73. Built by `scripts/run_m1_gate.py`; this file is rendered from the result JSON files by `scripts/render_m1_gate_readme.py`.

**These are internal validation numbers for the M1 gate. They are not article results and must not appear in any article table.** Track B uses the provisional model-year stand-in classes (no approved generation mapping exists yet), so every Track B output is `mapping_status = provisional`. Every Track A output carries the label: *official CVPR 2015 half/half classification split (30,955 images, 431 models); release version unverified; hashes recorded*

Prototype Parquet tables live under `data/prototypes/<encoder>/<variant>/` and the embedded Qdrant store under `data/qdrant/`; both are git-ignored and only the `.sha256` sidecars are committed. In embedded local mode Qdrant searches by brute force and ignores payload indexes; the keyword indexes are declared so the same export works against a server.

## PROTOCOL.md §25 M1 checklist

| output | artifact |
|---|---|
| validated manifest | `data/manifests/compcars_web_images.sha256`, `compcars_web_images.summary.json` (136,726 rows, 136,365 valid; the Parquet is DVC-tracked, rebuilt by `scripts/build_image_manifest.py` and verified against the sidecar by every gate run) |
| deduplication report | `reports/dedup/README.md`, `reports/dedup/AUDIT.md` (ADR-0005, pHash distance 2; 998 duplicate groups in the manifest summary) |
| split manifests | `data/splits/track_a/official_cvpr2015.parquet` + `.sha256`, `reports/splits/track_a/README.md`; `data/splits/track_b/seed_*.sha256`, `summary.json` (2,645 eligible classes, 118,070 images), `reports/splits/track_b/eligibility_audit.md`; cross-view eligibility (ADR-0004, ADR-0006): `reports/splits/track_b/cross_view/seed_101.md`, `reports/splits/track_b/cross_view/seed_13.md`, `reports/splits/track_b/cross_view/seed_137.md`, `reports/splits/track_b/cross_view/seed_37.md`, `reports/splits/track_b/cross_view/seed_73.md` |
| P3 masks | `reports/segmentation/README.md` (136,365 images, mask failure rate 0.20%; metadata sha256 recorded per P3 run below) |
| embedding extraction | `reports/embeddings/README.md` |
| initial mapping coverage report | `reports/mapping/v0.1/coverage.md` (queue v0.1, every row `provisional`) |
| leakage audit on the manifests used | `reports/leakage/e0_fixture.md` (PASS), `reports/leakage/track_a.md` (PASS; 52 official-split crossing groups reported, not altered), `reports/leakage/track_b_seed_101.md` (PASS), `reports/leakage/track_b_seed_13.md` (PASS), `reports/leakage/track_b_seed_137.md` (PASS), `reports/leakage/track_b_seed_37.md` (PASS), `reports/leakage/track_b_seed_73.md` (PASS) |
| generation mapping approval | §23 'all Track B generation rows approved': WAIVED, protocol provisional (issue #17); every Track B class is a model-year stand-in with `mapping_status = provisional` |
| deterministic run | this report: two repeats per track and variant, prototypes and rank matrices byte-identical, embeddings identical (tables below) |

## track_a p1

Split label: **official CVPR 2015 half/half classification split (30,955 images, 431 models); release version unverified; hashes recorded**

Embedding store: 136,365 rows in 137 shards, dimension 768. Result file: `track_a_p1.json`. P3 mask metadata sha256: `not recorded`.

### official_full

- support images 16,003, query images 14,922, classes 431; images without an embedding: support 0, query 0; queries not eligible (unknown viewpoint) 0; queries tied at rank 1: 0

Ranking (PROTOCOL.md §13.2)

| level | classes | top-1 | top-5 | top-10 | top-20 | MRR | median rank | P90 rank |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| make_model | 431 | 0.8319 | 0.9739 | 0.9874 | 0.9938 | 0.8952 | 1.0 | 2.0 |
| make | 75 | 0.9485 | 0.9902 | 0.9962 | 0.9983 | 0.9675 | 1.0 | 1.0 |

Classification (PROTOCOL.md §13.1)

| level | accuracy | macro P | macro R | macro F1 | weighted F1 | balanced acc. | classes without predicted positives |
|---|---:|---:|---:|---:|---:|---:|---:|
| make_model | 0.8319 | 0.8396 | 0.8505 | 0.8287 | 0.8394 | 0.8505 | 0 |
| make | 0.9485 | 0.9284 | 0.9425 | 0.9320 | 0.9489 | 0.9425 | 0 |

Error decomposition (top-1)

| top-1 outcome | fraction of queries |
|---|---:|
| wrong generation, same model | 0.0000 |
| wrong model, same make | 0.1167 |
| wrong make | 0.0515 |
| generation accuracy given correct model | 1.0000 |

### validation_fit_only

- support images 12,814, query images 3,189, classes 431; images without an embedding: support 0, query 0; queries not eligible (unknown viewpoint) 0; queries tied at rank 1: 0

Ranking (PROTOCOL.md §13.2)

| level | classes | top-1 | top-5 | top-10 | top-20 | MRR | median rank | P90 rank |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| make_model | 431 | 0.7598 | 0.9445 | 0.9690 | 0.9865 | 0.8430 | 1.0 | 3.0 |
| make | 75 | 0.9178 | 0.9784 | 0.9890 | 0.9966 | 0.9455 | 1.0 | 1.0 |

Classification (PROTOCOL.md §13.1)

| level | accuracy | macro P | macro R | macro F1 | weighted F1 | balanced acc. | classes without predicted positives |
|---|---:|---:|---:|---:|---:|---:|---:|
| make_model | 0.7598 | 0.7908 | 0.7647 | 0.7524 | 0.7717 | 0.7647 | 0 |
| make | 0.9178 | 0.9051 | 0.9003 | 0.8985 | 0.9190 | 0.9003 | 0 |

Error decomposition (top-1)

| top-1 outcome | fraction of queries |
|---|---:|
| wrong generation, same model | 0.0000 |
| wrong model, same make | 0.1580 |
| wrong make | 0.0822 |
| generation accuracy given correct model | 1.0000 |

### Persisted prototypes and Qdrant parity

- Parquet `data/prototypes/dinov3_vitb16/p1/m0_track_a_official_cvpr2015.parquet` (431 prototypes, support count min 10, max 143), sha256 `df146806d74b0dbe61d516449f6d36f524fcf2f1b6686b5197cc82b937d4ffe1`
- Qdrant collection `track_a_official_cvpr2015__none__dinov3_vitb16__dinov3_vit16_224_v1+p1__768__m0_global_mean` (431 points, embedded local mode)
- Parity on 1000 queries, top-10: ordered-equal 999, equal within score tolerance 1e-05 1, passed: yes

### Rerun diff

| repeat | prototypes byte-identical | ranks byte-identical | embeddings max abs diff | Parquet sha256 equal | metrics equal | runtime s | peak RSS MB |
|---|---|---|---:|---|---|---|---|
| 1 vs 2 | yes | official_full: yes, validation_fit_only: yes | 0.0e+00 | yes | yes | 16.82 / 15.59 | 2091.8 / 2426.7 |

Run manifest `m1_gate_track_a_p1_dinov3_vitb16` at commit `e2627197e1f4be645a957a94fff27ad69ee22632` (dirty worktree: yes); full manifest in the JSON and in the local MLflow store.

## track_a p3

Split label: **official CVPR 2015 half/half classification split (30,955 images, 431 models); release version unverified; hashes recorded**

Embedding store: 136,097 rows in 137 shards, dimension 768. Result file: `track_a_p3.json`. P3 mask metadata sha256: `not recorded`.

### official_full

- support images 16,002, query images 14,922, classes 431; images without an embedding: support 1, query 0; queries not eligible (unknown viewpoint) 0; queries tied at rank 1: 0

Ranking (PROTOCOL.md §13.2)

| level | classes | top-1 | top-5 | top-10 | top-20 | MRR | median rank | P90 rank |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| make_model | 431 | 0.8283 | 0.9723 | 0.9871 | 0.9934 | 0.8928 | 1.0 | 2.0 |
| make | 75 | 0.9476 | 0.9899 | 0.9960 | 0.9983 | 0.9666 | 1.0 | 1.0 |

Classification (PROTOCOL.md §13.1)

| level | accuracy | macro P | macro R | macro F1 | weighted F1 | balanced acc. | classes without predicted positives |
|---|---:|---:|---:|---:|---:|---:|---:|
| make_model | 0.8283 | 0.8342 | 0.8463 | 0.8240 | 0.8358 | 0.8463 | 0 |
| make | 0.9476 | 0.9283 | 0.9422 | 0.9318 | 0.9479 | 0.9422 | 0 |

Error decomposition (top-1)

| top-1 outcome | fraction of queries |
|---|---:|
| wrong generation, same model | 0.0000 |
| wrong model, same make | 0.1193 |
| wrong make | 0.0524 |
| generation accuracy given correct model | 1.0000 |

### validation_fit_only

- support images 12,813, query images 3,189, classes 431; images without an embedding: support 1, query 0; queries not eligible (unknown viewpoint) 0; queries tied at rank 1: 0

Ranking (PROTOCOL.md §13.2)

| level | classes | top-1 | top-5 | top-10 | top-20 | MRR | median rank | P90 rank |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| make_model | 431 | 0.7535 | 0.9373 | 0.9690 | 0.9859 | 0.8372 | 1.0 | 3.0 |
| make | 75 | 0.9178 | 0.9752 | 0.9875 | 0.9956 | 0.9445 | 1.0 | 1.0 |

Classification (PROTOCOL.md §13.1)

| level | accuracy | macro P | macro R | macro F1 | weighted F1 | balanced acc. | classes without predicted positives |
|---|---:|---:|---:|---:|---:|---:|---:|
| make_model | 0.7535 | 0.7818 | 0.7566 | 0.7440 | 0.7660 | 0.7566 | 0 |
| make | 0.9178 | 0.8938 | 0.8959 | 0.8903 | 0.9189 | 0.8959 | 0 |

Error decomposition (top-1)

| top-1 outcome | fraction of queries |
|---|---:|
| wrong generation, same model | 0.0000 |
| wrong model, same make | 0.1643 |
| wrong make | 0.0822 |
| generation accuracy given correct model | 1.0000 |

### Persisted prototypes and Qdrant parity

- Parquet `data/prototypes/dinov3_vitb16/p3/m0_track_a_official_cvpr2015.parquet` (431 prototypes, support count min 10, max 143), sha256 `56f0d5d1a691b81f2a87058cddbd283f3ce6f75de1d5ce942d194c0529ac0754`
- Qdrant collection `track_a_official_cvpr2015__none__dinov3_vitb16__dinov3_vit16_224_v1+p3__768__m0_global_mean` (431 points, embedded local mode)
- Parity on 1000 queries, top-10: ordered-equal 1000, equal within score tolerance 1e-05 0, passed: yes

### Rerun diff

| repeat | prototypes byte-identical | ranks byte-identical | embeddings max abs diff | Parquet sha256 equal | metrics equal | runtime s | peak RSS MB |
|---|---|---|---:|---|---|---|---|
| 1 vs 2 | yes | official_full: yes, validation_fit_only: yes | 0.0e+00 | yes | yes | 15.23 / 12.54 | 2166.5 / 2504.5 |

Run manifest `m1_gate_track_a_p3_dinov3_vitb16` at commit `e2627197e1f4be645a957a94fff27ad69ee22632` (dirty worktree: yes); full manifest in the JSON and in the local MLflow store.

## track_b p1

Split: `track_b_seed_73` (primary seed 73); classes are the provisional model-year stand-ins, every row `mapping_status = provisional`.

Embedding store: 136,365 rows in 137 shards, dimension 768. Result file: `track_b_p1.json`. P3 mask metadata sha256: `not recorded`.

### test

- support images 70,880, query images 23,160, classes 2645; images without an embedding: support 0, query 0; queries not eligible (unknown viewpoint) 121; queries tied at rank 1: 0

Ranking (PROTOCOL.md §13.2)

| level | classes | top-1 | top-5 | top-10 | top-20 | MRR | median rank | P90 rank |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| class | 2645 | 0.6471 | 0.9329 | 0.9649 | 0.9830 | 0.7730 | 1.0 | 4.0 |
| model | 1390 | 0.8283 | 0.9650 | 0.9820 | 0.9912 | 0.8890 | 1.0 | 2.0 |
| make | 153 | 0.9402 | 0.9862 | 0.9931 | 0.9972 | 0.9609 | 1.0 | 1.0 |

Classification (PROTOCOL.md §13.1)

| level | accuracy | macro P | macro R | macro F1 | weighted F1 | balanced acc. | classes without predicted positives |
|---|---:|---:|---:|---:|---:|---:|---:|
| class | 0.6471 | 0.6561 | 0.6237 | 0.6128 | 0.6578 | 0.6237 | 36 |
| model | 0.8283 | 0.8547 | 0.8240 | 0.8270 | 0.8328 | 0.8240 | 1 |
| make | 0.9402 | 0.9249 | 0.9074 | 0.9138 | 0.9404 | 0.9074 | 0 |

Error decomposition (top-1)

| top-1 outcome | fraction of queries |
|---|---:|
| wrong generation, same model | 0.1812 |
| wrong model, same make | 0.1119 |
| wrong make | 0.0598 |
| generation accuracy given correct model | 0.7812 |

### validation

- support images 70,880, query images 23,785, classes 2645; images without an embedding: support 0, query 0; queries not eligible (unknown viewpoint) 124; queries tied at rank 1: 0

Ranking (PROTOCOL.md §13.2)

| level | classes | top-1 | top-5 | top-10 | top-20 | MRR | median rank | P90 rank |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| class | 2645 | 0.6423 | 0.9311 | 0.9642 | 0.9815 | 0.7693 | 1.0 | 4.0 |
| model | 1390 | 0.8269 | 0.9649 | 0.9812 | 0.9902 | 0.8878 | 1.0 | 2.0 |
| make | 153 | 0.9400 | 0.9865 | 0.9931 | 0.9967 | 0.9608 | 1.0 | 1.0 |

Classification (PROTOCOL.md §13.1)

| level | accuracy | macro P | macro R | macro F1 | weighted F1 | balanced acc. | classes without predicted positives |
|---|---:|---:|---:|---:|---:|---:|---:|
| class | 0.6423 | 0.6512 | 0.6180 | 0.6078 | 0.6529 | 0.6180 | 49 |
| model | 0.8269 | 0.8551 | 0.8239 | 0.8270 | 0.8313 | 0.8239 | 1 |
| make | 0.9400 | 0.9200 | 0.9027 | 0.9092 | 0.9403 | 0.9027 | 0 |

Error decomposition (top-1)

| top-1 outcome | fraction of queries |
|---|---:|
| wrong generation, same model | 0.1846 |
| wrong model, same make | 0.1132 |
| wrong make | 0.0600 |
| generation accuracy given correct model | 0.7768 |

### Persisted prototypes and Qdrant parity

- Parquet `data/prototypes/dinov3_vitb16/p1/m0_track_b_seed_73.parquet` (2645 prototypes, support count min 11, max 101), sha256 `e68f9f37c57ee289d2283ca2b2154ab7c4a93a8bc83e8d7ad7feff80e7adc2dd`
- Qdrant collection `track_b_seed_73__v0.1_provisional__dinov3_vitb16__dinov3_vit16_224_v1+p1__768__m0_global_mean` (2645 points, embedded local mode)
- Parity on 1000 queries, top-10: ordered-equal 1000, equal within score tolerance 1e-05 0, passed: yes

### Rerun diff

| repeat | prototypes byte-identical | ranks byte-identical | embeddings max abs diff | Parquet sha256 equal | metrics equal | runtime s | peak RSS MB |
|---|---|---|---:|---|---|---|---|
| 1 vs 2 | yes | test: yes, validation: yes | 0.0e+00 | yes | yes | 240.92 / 168.78 | 6416.2 / 7653.7 |

Run manifest `m1_gate_track_b_p1_dinov3_vitb16` at commit `e2627197e1f4be645a957a94fff27ad69ee22632` (dirty worktree: yes); full manifest in the JSON and in the local MLflow store.

## track_b p3

Split: `track_b_seed_73` (primary seed 73); classes are the provisional model-year stand-ins, every row `mapping_status = provisional`.

Embedding store: 136,097 rows in 137 shards, dimension 768. Result file: `track_b_p3.json`. P3 mask metadata sha256: `not recorded`.

### test

- support images 70,733, query images 23,143, classes 2645; images without an embedding: support 147, query 17; queries not eligible (unknown viewpoint) 121; queries tied at rank 1: 1

Ranking (PROTOCOL.md §13.2)

| level | classes | top-1 | top-5 | top-10 | top-20 | MRR | median rank | P90 rank |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| class | 2645 | 0.6326 | 0.9295 | 0.9628 | 0.9808 | 0.7628 | 1.0 | 4.0 |
| model | 1390 | 0.8205 | 0.9627 | 0.9809 | 0.9903 | 0.8835 | 1.0 | 2.0 |
| make | 153 | 0.9377 | 0.9857 | 0.9927 | 0.9968 | 0.9592 | 1.0 | 1.0 |

Classification (PROTOCOL.md §13.1)

| level | accuracy | macro P | macro R | macro F1 | weighted F1 | balanced acc. | classes without predicted positives |
|---|---:|---:|---:|---:|---:|---:|---:|
| class | 0.6326 | 0.6435 | 0.6088 | 0.5972 | 0.6434 | 0.6088 | 44 |
| model | 0.8205 | 0.8497 | 0.8174 | 0.8200 | 0.8251 | 0.8174 | 1 |
| make | 0.9377 | 0.9164 | 0.9015 | 0.9063 | 0.9378 | 0.9015 | 0 |

Error decomposition (top-1)

| top-1 outcome | fraction of queries |
|---|---:|
| wrong generation, same model | 0.1879 |
| wrong model, same make | 0.1172 |
| wrong make | 0.0623 |
| generation accuracy given correct model | 0.7710 |

### validation

- support images 70,733, query images 23,759, classes 2645; images without an embedding: support 147, query 26; queries not eligible (unknown viewpoint) 124; queries tied at rank 1: 0

Ranking (PROTOCOL.md §13.2)

| level | classes | top-1 | top-5 | top-10 | top-20 | MRR | median rank | P90 rank |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| class | 2645 | 0.6272 | 0.9263 | 0.9625 | 0.9805 | 0.7584 | 1.0 | 4.0 |
| model | 1390 | 0.8177 | 0.9625 | 0.9807 | 0.9901 | 0.8816 | 1.0 | 2.0 |
| make | 153 | 0.9367 | 0.9849 | 0.9931 | 0.9968 | 0.9586 | 1.0 | 1.0 |

Classification (PROTOCOL.md §13.1)

| level | accuracy | macro P | macro R | macro F1 | weighted F1 | balanced acc. | classes without predicted positives |
|---|---:|---:|---:|---:|---:|---:|---:|
| class | 0.6272 | 0.6382 | 0.6033 | 0.5923 | 0.6378 | 0.6033 | 47 |
| model | 0.8177 | 0.8476 | 0.8145 | 0.8177 | 0.8223 | 0.8145 | 2 |
| make | 0.9367 | 0.9117 | 0.8947 | 0.9010 | 0.9371 | 0.8947 | 1 |

Error decomposition (top-1)

| top-1 outcome | fraction of queries |
|---|---:|
| wrong generation, same model | 0.1905 |
| wrong model, same make | 0.1190 |
| wrong make | 0.0633 |
| generation accuracy given correct model | 0.7671 |

### Persisted prototypes and Qdrant parity

- Parquet `data/prototypes/dinov3_vitb16/p3/m0_track_b_seed_73.parquet` (2645 prototypes, support count min 7, max 101), sha256 `3807cf838d4adc2c1b0fc3a07a7bb1df1acb5ccf08155f2070679dc1be563c8a`
- Qdrant collection `track_b_seed_73__v0.1_provisional__dinov3_vitb16__dinov3_vit16_224_v1+p3__768__m0_global_mean` (2645 points, embedded local mode)
- Parity on 1000 queries, top-10: ordered-equal 999, equal within score tolerance 1e-05 1, passed: yes

### Rerun diff

| repeat | prototypes byte-identical | ranks byte-identical | embeddings max abs diff | Parquet sha256 equal | metrics equal | runtime s | peak RSS MB |
|---|---|---|---:|---|---|---|---|
| 1 vs 2 | yes | test: yes, validation: yes | 0.0e+00 | yes | yes | 96.46 / 101.62 | 6429.1 / 7677.7 |

Run manifest `m1_gate_track_b_p3_dinov3_vitb16` at commit `e2627197e1f4be645a957a94fff27ad69ee22632` (dirty worktree: yes); full manifest in the JSON and in the local MLflow store.
