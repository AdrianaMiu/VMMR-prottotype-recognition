# Advanced Vehicle Recognition (AVR)

A reproducible experiment and service for fine-grained vehicle make, model and generation recognition on CompCars, comparing frozen visual encoders through prototype-based retrieval.

## Method

**Prototype-based recognition**:
Recognition by ranking a query image embedding against class prototypes by cosine similarity. The encoder is frozen; labeled support images build the prototypes.
_Avoid_: zero-shot, few-shot classification (without qualification), training-free (without "at the encoder level")

**Frozen encoder**:
A pretrained visual encoder whose weights are never updated in this project.
_Avoid_: backbone (ambiguous with fine-tuned usage)

**Support image**:
A labeled image whose embedding contributes to a prototype, index, or probe. Never a query.
_Avoid_: training image, gallery image (that is a retrieval term, see below)

**Query image**:
An image whose embedding is ranked against prototypes or a gallery. Never used to build, tune, or calibrate anything.
_Avoid_: test image (unless naming the split explicitly)

**Prototype**:
An L2-normalized vector representing a class (or a class-viewpoint pair, or a within-class cluster), built from support embeddings only.
_Avoid_: centroid (fine informally, not in results), class embedding

**Embedding**:
The L2-normalized pooled output of an encoder for one image.
_Avoid_: feature vector, representation (in metric tables)

## Labels

**Make**:
The vehicle brand as identified by CompCars `make_id`.
_Avoid_: brand, manufacturer

**Model**:
A CompCars `model_id` under a make. Different years are the same model.
_Avoid_: series, nameplate

**Model-year class**:
A `make + model + year` triple as CompCars records it. It is NOT a generation.
_Avoid_: generation (for this concept), year class

**Generation**:
A curated interval or set of model years corresponding to a stable vehicle generation, assigned through the generation mapping. Facelift phases are recorded separately and merged in the primary label.
_Avoid_: model year, year, version, mark

**Facelift phase**:
A pre-facelift or post-facelift period inside one generation.
_Avoid_: minor generation, refresh

**Generation mapping**:
The versioned, provenance-backed, dual-reviewed table that maps each CompCars `(make_id, model_id, year)` to a generation. A research artifact, not a config file.
_Avoid_: year table, lookup CSV

**Mapping status**:
The review state of one generation-mapping row: `provisional`, `ambiguous`, or `approved`. Only `approved` rows enter the final Track B protocol.

**Viewpoint**:
One of the five CompCars viewpoint labels of a whole-car image.
_Avoid_: pose, angle, orientation

## Protocols

**Track A**:
The CompCars benchmark-comparable protocol: 431 make-model classes, official Part-I classification split, make-model label.
_Avoid_: the benchmark, the official experiment

**Track B**:
The curated generation protocol: make-model-generation classes on eligible web-nature images, group-aware 60/20/20 split, primary seed 73.
_Avoid_: the new dataset, generation benchmark

**Track C**:
Optional cross-domain stress test from web-nature prototypes to surveillance-nature queries. Not part of the primary benchmark.

**Official split**:
The CompCars `train_test_split/classification/{train,test}.txt` files as distributed. Results on any other partition must not be called official.
_Avoid_: standard split, paper split

**Duplicate group**:
A set of images that must stay in one partition: exact SHA-256 duplicates, perceptual near-duplicates, or a known same-source series.
_Avoid_: cluster (reserved for prototype clustering)

**Primary outcome**:
The one pre-registered metric per family (Track A Top-1 make-model accuracy; Track B macro F1 on exact generation; retrieval mAP at exact generation). Everything else is secondary.

## Preprocessing variants

**P0**: original image, encoder-canonical preprocessing.
**P1**: ground-truth bounding-box crop, 5% expansion, aspect preserved.
**P2**: predicted detector crop from the same detector the service uses.
**P3**: segmented vehicle on a fixed neutral background, projected into the P1 crop for the scientific ablation.

## Prototype methods

**M0**: global class mean.
**M1**: view-balanced class mean.
**M2**: view-conditioned prototypes, oracle or predicted viewpoint.
**M3**: multi-prototype clustering within a class.
**M4**: support-image k-NN baseline (no averaging).
**M5**: linear probe on frozen embeddings (supervised baseline, reported separately).

## Experiment stages

**E0** pipeline validation; **E1** primary encoder comparison; **E2** dimension ablation; **E3** viewpoint ablation; **E4** segmentation ablation; **E5** few-shot curves; **E6** image retrieval and verification; **E7** service validation.

## Metrics

**Top-k accuracy**:
Fraction of queries whose correct class is within the first k ranked prototypes. Equal to Hit@k and Recall@k when there is one relevant class.
_Avoid_: F10, F5, precision@k (for the single-relevant case)

**Exact search**:
Cosine ranking over all prototypes by matrix multiplication or FAISS IndexFlatIP. The only search allowed in reported metrics.
_Avoid_: ANN, approximate search (for article numbers)

**Run manifest**:
The record that makes a metric citable: dataset, split, mapping, encoder, preprocessing and config hashes plus environment. No manifest, no metric.
