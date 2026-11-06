# Advanced Vehicle Recognition - Project Context and Protocol

**Document status:** design specification for approval  
**Date:** 2026-11-06  
**Primary goal:** implement a reproducible scientific experiment and an end-to-end service for fine-grained vehicle make, model, and generation recognition using frozen visual encoders and prototype-based retrieval on CompCars.

---

## 1. Instructions to the implementation team

Treat this document as the authoritative project context. Do not silently change the dataset protocol, label hierarchy, split, metric definition, encoder checkpoint, preprocessing variant, or statistical procedure. If implementation evidence contradicts the document, stop and record the conflict in an ADR before changing behavior.

Before implementation:

1. Inspect the repository, dataset metadata, official CompCars split files, and available compute.
2. Produce a task plan with file paths, tests, dependencies, and acceptance criteria.
3. Separate independent work among workstreams only after interfaces and output schemas are fixed.
4. Use test-driven development for deterministic data, metric, prototype, and API logic.
5. Verify every completed phase with executable checks and stored evidence.
6. Never report a metric unless its run manifest, dataset hash, split hash, label-map hash, model checkpoint hash, and configuration are available.

The research lead owns the plan, shared schemas, integration, and final scientific claims. Workstreams may not redefine labels, splits, or metrics. Each workstream works in isolated files or an isolated worktree and returns a compact handoff containing changed files, commands run, results, unresolved issues, and artifact locations.

---

## 2. Research objective

Build and evaluate a frozen-encoder recognition system in which labeled support images are embedded, L2-normalized, grouped by vehicle class, and averaged into class prototypes. A query image is embedded and ranked against the prototypes by cosine similarity.

The project compares:

- DINOv3;
- SigLIP2;
- original OpenAI CLIP;
- a modern 1024-dimensional OpenCLIP model.

The principal class hierarchy is:

```text
make -> model -> generation
```

CompCars natively supplies make, model, year, bounding box, and one of five viewpoints for web-nature whole-car images. It does **not** supply an authoritative generation identifier. Generation labels must therefore be added through a versioned, independently reviewable mapping from `make + model + year` to generation.

The main scientific contribution is not conventional supervised fine-tuning. It is a comparison of frozen foundation-model representations and prototype construction strategies for fine-grained vehicle recognition, including viewpoint and foreground segmentation effects.

### Correct terminology

- The method is **prototype-based**, **frozen-feature**, or **training-free at the encoder level**.
- It is not pure zero-shot recognition because labeled support images are used to construct prototypes.
- A `make + model + year` class must be called a **model-year class**, not a generation.
- A generation is a curated interval or set of model years corresponding to a stable vehicle generation. Facelifts are stored separately.
- “Top-10” means Top-10 accuracy / Hit@10 / Recall@10 for a single relevant class. Do not call it `F10`.

---

## 3. Research questions and hypotheses

### RQ1 - Encoder quality

Which frozen encoder yields the best make, model, and generation recognition using identical support/query data and cosine prototype ranking?

**H1:** DINOv3 and SigLIP2 will outperform original CLIP on model and generation discrimination because their visual features are newer and better suited to dense/fine-grained structure.

### RQ2 - Viewpoint handling

Does viewpoint-aware prototype construction improve recognition over one unweighted global mean per class?

**H2:** view-balanced and view-conditioned prototypes will improve macro-level metrics, especially for classes with imbalanced viewpoint coverage.

### RQ3 - Segmentation

Does removing the background improve class separability and recognition?

**H3:** segmentation will help when backgrounds correlate spuriously with class or source, but may hurt when mask errors remove fine vehicle details. The effect may vary by encoder.

### RQ4 - Dimensionality

How much performance is retained at lower embedding dimensions?

**H4:** PCA-reduced 512-dimensional features will retain most native performance while reducing storage and search latency; 256 dimensions may show a larger loss on generation recognition than on make recognition.

### RQ5 - Few-shot support size

How does performance change with the number of support images per class?

**H5:** gains will saturate after a moderate support size, but viewpoint-balanced sampling will reach the plateau earlier than random sampling.

### RQ6 - Hierarchical error structure

When the exact generation is wrong, does the ranked list still preserve the correct model or make?

**H6:** most errors will remain within the correct make, followed by the correct model but adjacent generation.

---

## 4. Scope and non-goals

### In scope

- CompCars web-nature whole-car images.
- Official CompCars 431-model benchmark protocol.
- A new curated make-model-generation protocol on the larger eligible web-nature set.
- Frozen encoder extraction and exact cosine evaluation.
- Prototype, k-NN, and frozen-feature linear-probe baselines.
- Viewpoint, segmentation, dimension, and support-size ablations.
- Prototype ranking and image-gallery retrieval.
- FiftyOne qualitative and quantitative embedding analysis.
- Reproducible experiment tracking and publication-ready tables/figures.
- Multi-car end-to-end API service.

### Out of scope for the first article

- End-to-end encoder fine-tuning.
- Vehicle instance re-identification.
- License-plate recognition.
- Color recognition as a primary task.
- Open-world discovery of unseen generations as the main experiment.
- Surveillance-nature versus web-nature domain adaptation, except as an optional extension.
- Claiming overall state of the art against supervised systems that use a different split or task definition.

---

## 5. Dataset protocols

The article must contain two separate tracks. Results from one track must never be mixed with the other.

### Track A - CompCars benchmark-comparable model recognition

Purpose: comparison with established CompCars make/model recognition literature.

- Source: web-nature whole-car images from official Part-I.
- Classes: 431 make-model classes.
- Images: 30,955 whole-car images in the original paper's Part-I.
- Label: `make + model`; different years are merged into the same class.
- Split: use the official released split files if available. The original experiment divided the Part-I images half for training and half for testing.
- Split identity: record CompCars release/version, filenames, row counts, and SHA-256 hashes because published CompCars releases and secondary papers report different dataset counts.
- Prototype support: official training half only.
- Query set: official test half only.
- Validation: create a deterministic, class-stratified nested validation partition from the official training half for hyperparameters and calibration. Rebuild final prototypes from the complete official training half only after all choices are frozen.
- Primary comparability metrics: Top-1, Top-5, and Top-10 model accuracy, plus make accuracy.

Do not substitute a new random split and still call the result “official benchmark.” If the distributed split files cannot be recovered or their exact identity cannot be verified, label the experiment “reconstructed 50/50 Part-I split,” publish the manifest, and do not claim exact comparability.

### Track B - Curated generation recognition

Purpose: evaluate the novel make-model-generation task and viewpoint-aware prototypes.

- Source: eligible CompCars web-nature whole-car images.
- Native labels: make, model, year, viewpoint, bounding box.
- Derived label: generation from the curated mapping described below.
- Main class: `make + model + generation`.
- Minimum class eligibility, initial rule:
  - at least 20 valid images after deduplication;
  - at least two represented CompCars viewpoints;
  - at least five query-eligible images;
  - generation mapping status is `approved`;
  - no unresolved overlap between generations.
- Main split: group-aware 60% support / 20% validation / 20% test, stratified by class and approximately by viewpoint.
- Primary split: seed `73`, used for the main table and cluster-aware bootstrap confidence intervals.
- Sensitivity splits: seeds `13, 37, 101, 137`, combined with the primary split only for a five-split mean and standard deviation sensitivity analysis. Store split manifests; never regenerate them implicitly.
- Group constraints: exact duplicates, perceptual near-duplicate clusters, and known same-source image series must remain in one partition.
- Primary uncertainty: 95% hierarchical bootstrap interval on the fixed primary split.
- Split sensitivity: mean and standard deviation across the five fixed splits; do not present five-split intervals as precise confidence intervals.
- Full-support experiment: all support images contribute to the prototype.
- Few-shot experiment: `K in {1, 2, 5, 10, 20}` support images per class where eligible. Use both random and viewpoint-balanced selection.

If the eligibility rule leaves too few generation classes, report the class-frequency audit before relaxing the threshold. Any changed threshold requires a new protocol version.

### Optional Track C - Cross-domain stress test

Only after Tracks A and B are complete: construct prototypes from web-nature data and test compatible labels on surveillance-nature images. Report this as domain shift, not as part of the primary benchmark.

---

## 6. Generation mapping workstream

Generation curation is a research artifact, not an incidental CSV.

### Mapping unit

Each row maps a CompCars record `(make_id, model_id, year)` to a generation.

Required columns:

```text
make_id
make_name
model_id
model_name
compcars_year
canonical_model_name
generation_id
generation_name
generation_start_year
generation_end_year
facelift_phase
body_style_or_variant
market_region
source_url_1
source_url_2
source_title_1
source_title_2
accessed_at
mapping_status
reviewer_1
reviewer_2
review_notes
ambiguity_code
```

### Curation rules

1. Prefer manufacturer archives, official brochures, homologation records, or established automotive catalogues.
2. Require two sources for each approved mapping when possible.
3. Store source provenance and access date; do not store only a free-text conclusion.
4. Distinguish calendar year, production year, and marketed model year.
5. Record market-region differences. Do not assume a generation started in the same year worldwide.
6. Keep facelift/pre-facelift as a separate field. The primary generation label merges facelift phases unless the visual and metadata evidence justifies a registered secondary experiment.
7. Assign stable identifiers independent of display names, for example `dacia_logan_g2`.
8. Mark uncertain rows `ambiguous`, not `approved`.
9. Exclude unresolved overlaps, rebadged naming collisions, and mixed body-style folders from the primary generation evaluation.
10. Two human reviewers, including at least one person with automotive-domain competence, must approve each generation interval. Automated annotation tools do not count as independent scientific annotators. Agreement and adjudication counts must be reported.
11. Use a declared market convention. Because much of CompCars reflects the Chinese market, prefer China-market production/model-year intervals where reliable sources exist; otherwise record the applicable region and exclude unresolved cross-market conflicts.
12. After metadata mapping, visually audit the images because a correct year-to-generation table cannot detect mixed or mislabeled CompCars folders. Audit at least one image per retained generation plus a stratified sample of at least 500 images, oversampling generation boundaries and ambiguous records. Report an estimated label-noise rate with a Wilson 95% interval.

### Mapping quality report

Generate:

- number of mapped and unmapped model-year records;
- number of images retained and excluded;
- classes and images per generation;
- mapping status distribution;
- reviewer agreement before adjudication;
- visual-audit sample, failure categories, and estimated image-level label-noise rate;
- ambiguity reasons;
- checksum and semantic version of the mapping.

---

## 7. Data audit and leakage prevention

Before embedding extraction, create an immutable metadata table with one row per image.

Required fields:

```text
image_id
relative_path
sha256
phash
width
height
make_id
make_name
model_id
model_name
year
generation_id
viewpoint
bbox_xyxy
split_track_a
split_track_b_seed
duplicate_group_id
source_group_id
mapping_version
validity_status
exclusion_reason
```

Leakage controls:

- SHA-256 exact duplicate grouping.
- Perceptual duplicate grouping with a recorded pHash threshold and visual audit near the threshold.
- Optional DINO feature duplicate detection only as a diagnostic; do not use test embeddings to tune a threshold.
- Keep every duplicate/source group in one split.
- Fit PCA, whitening, temperature calibration, thresholds, clustering, and linear probes on support/training data only.
- Do not use test images to choose encoder layers, resolution, mask style, prototype type, or support count.
- Do not use query viewpoint labels in a deployable method unless the service also predicts viewpoint. Oracle-viewpoint results must be clearly labeled as an upper bound.
- Review whether publishing dataset-relative paths in split manifests is permitted by the CompCars license. If not, publish stable opaque IDs and hashes rather than paths.

---

## 8. Image preprocessing and segmentation variants

Every variant must preserve a traceable link to the original image and transform metadata.

### P0 - Original image

Use the full CompCars image with the encoder's canonical preprocessing.

### P1 - Ground-truth bounding-box crop

Use the CompCars bounding box, expanded by 5% on each side and clipped to image bounds. Preserve aspect ratio with encoder-specific padding rather than distortion.

### P2 - Predicted detector crop

Use the frozen detector/segmenter output that will also be used in the service. This measures deployment realism. The default adapter should support RF-DETR segmentation because it has already been validated on sample vehicle images, but the interface must allow another detector.

### P3 - Segmented vehicle

Use the vehicle instance mask and replace the background with a fixed neutral RGB value, initially `(127, 127, 127)`. Do not use transparency as encoder input. Store the binary mask separately.

For the scientific segmentation ablation, project the predicted mask into the exact P1 ground-truth bounding-box crop so P1 and P3 differ only by masking. A P3 variant based on the predicted P2 crop belongs to the deployment-realism experiment and must be reported separately.

### Required segmentation study

Compare at least:

- P1 bounding-box crop without masking;
- P3 bounding-box crop with neutral-background segmentation.

Keep crop, resolution, encoder, support split, and prototype logic identical. Report mask failure rate and manually inspect a stratified sample of at least 200 masks across class frequency, viewpoint, and image quality.

---

## 9. Encoder registry

All encoders are frozen in the primary study. The registry must capture model ID, source, revision/commit, license, parameter count, input resolution, native embedding dimension, pooling method, preprocessing, dtype, and checkpoint checksum.

The primary comparison uses each checkpoint's documented official pooled output. Record whether this is a CLS token, attention pool, or another pooling operation. Mean-patch pooling or alternative layers are separate registered ablations. Because checkpoints differ in objective, training data, size, resolution, and pooling, conclusions describe checkpoint/system performance and must not claim that the pretraining objective alone caused the difference.

### Primary candidates

| Family | Primary candidate | Expected native dimension | Role |
|---|---|---:|---|
| DINOv3 | ViT-L/16 pretrained on LVD-1689M | 1024 | modern self-supervised visual encoder |
| SigLIP2 | L/16 at 384 px | 1024 | modern vision-language encoder |
| OpenAI CLIP | ViT-L/14 at 336 px | 768 | canonical CLIP baseline |
| OpenCLIP | ViT-H/14 checkpoint with 1024-D output | 1024 | modern CLIP-family baseline |

Before implementation, verify the exact downloadable checkpoint identifiers and output shapes in code. Do not infer dimension from transformer width; use the actual pooled output.

### Optional compute-aware candidates

- DINOv3 ViT-B/16;
- SigLIP2 B/16;
- SigLIP2 NaFlex as a separate aspect-ratio experiment;
- smaller OpenCLIP model.

Do not mix base and large models in the primary “encoder family” table without reporting parameters, FLOPs, input resolution, throughput, and native dimension.

### Embedding extraction contract

For each image:

1. apply the checkpoint's official preprocessing;
2. run in evaluation and inference mode;
3. obtain the documented pooled image representation;
4. convert to float32 before persistent normalization;
5. L2-normalize each image embedding;
6. store the normalized embedding with encoder and preprocessing version metadata.

Store raw pooled embeddings only if storage permits; normalized embeddings are mandatory.

---

## 10. Dimension experiment

The dimension study must not create information by expanding a smaller vector.

### Conditions

- `native`: each encoder's native embedding.
- `pca_512`: PCA to 512 dimensions.
- `pca_256`: PCA to 256 dimensions.
- `pca_1024`: only for native dimensions greater than 1024.
- `native_1024_comparison`: a separate comparison restricted to encoders with genuine 1024-dimensional outputs.

Original OpenAI CLIP ViT-L/14 remains in native/512/256 comparisons but is excluded from the strict native-1024 table.

PCA rules:

- fit on support embeddings only;
- persist the fitted transformer and checksum;
- primary PCA uses no whitening;
- L2-normalize after projection;
- if whitening is studied, register it as a separate ablation;
- use the same support rows for every encoder in a paired comparison.

---

## 11. Prototype construction

Let `z_i` be an individually L2-normalized support embedding.

### M0 - Global class mean

```text
p_c = normalize(mean(z_i for i in class c))
```

This is the principal baseline.

### M1 - View-balanced class mean

Compute one centroid per available viewpoint, normalize it, average viewpoint centroids with equal weight, then normalize again. This prevents frequent viewpoints from dominating a class prototype.

### M2 - View-conditioned prototypes

Store one prototype for each `(class, viewpoint)`. Evaluate two modes:

- **oracle viewpoint:** use ground-truth query viewpoint; report only as an upper bound;
- **predicted viewpoint:** use a frozen viewpoint classifier and include its errors and latency.

If the query viewpoint is unknown or low-confidence, search all class-view prototypes. Maximum similarity can favor classes with more available viewpoints, so the primary M2 comparison must use an equal prototype count per eligible class or predicted-view routing. Record class viewpoint coverage. Any variable-count maximum or log-sum-exp aggregation is a separately calibrated secondary analysis.

### M3 - Multi-prototype clustering

Within each class, cluster support embeddings into `k` prototypes using support data only. Initial `k in {2, 3}`. Use deterministic k-means initialization and a fixed seed. Require at least `3k` support images and at least three samples per resulting cluster; otherwise fall back to the largest valid lower `k`, then to M0. Avoid class-specific hyperparameter tuning on the test set. Report this as multi-prototype retrieval, not a mean prototype.

### M4 - Support-image k-NN baseline

Store support embeddings rather than class means. Evaluate `k in {1, 3, 5}` with similarity-weighted voting. This measures information lost by prototype averaging.

### M5 - Linear probe baseline

Train multinomial logistic regression or a single linear layer on frozen support embeddings. Select regularization on validation data only. This is a supervised frozen-feature baseline and must be reported separately from training-free prototypes.

---

## 12. Search and vector storage

### Scientific evaluation

Use exact cosine search:

- normalized vectors;
- matrix multiplication or FAISS `IndexFlatIP`;
- deterministic ranking and explicit tie handling;
- no approximate index in reported article metrics.

### Online service

Use Qdrant by default. Keep collections versioned by:

```text
dataset_protocol + mapping_version + encoder + preprocessing + dimension + prototype_method
```

Payload fields include class hierarchy, viewpoint, support count, prototype version, model revision, preprocessing version, and experiment run ID.

The service may use approximate search only after exact-versus-ANN recall and latency are measured. Default to exact search when the prototype count is small.

---

## 13. Evaluation protocols and metrics

### Registered primary outcomes

- Track A primary outcome: Top-1 make-model accuracy on the official test split.
- Track B primary outcome: macro F1 for exact make-model-generation classification on the fixed primary split.
- Image-retrieval primary outcome: mAP for exact-generation relevance.

All other metrics are secondary unless a protocol amendment is registered before test evaluation. Pairwise significance testing is organized into three corresponding families, with Holm-Bonferroni correction within each family.

### 13.1 Closed-set classification metrics

Report at each available hierarchy level:

- Top-1 accuracy;
- Top-5 accuracy;
- Top-10 accuracy;
- macro precision, recall, and F1;
- weighted precision, recall, and F1;
- balanced accuracy;
- per-class recall and F1;
- confusion matrix, normalized by true class;
- hierarchical exact accuracy for make, model, and generation;
- conditional generation accuracy given correct model;
- error decomposition: wrong generation/same model, wrong model/same make, wrong make.

Micro F1 equals accuracy for standard single-label closed-set classification; report it only if needed for comparison and explain this equivalence.

Use `zero_division=0` for classwise precision/F1 when a class has no predicted positives, and report how many classes were affected. Compute macro values over all eligible test classes, not only classes predicted at least once.

### 13.2 Prototype ranking metrics

With one correct class prototype per query, report:

- CMC / Hit@k / Recall@k at `k = 1, 5, 10, 20`;
- mean reciprocal rank (MRR);
- mean rank;
- median rank;
- rank percentiles P75, P90, P95;
- normalized rank;
- hierarchical NDCG@k with pre-registered gains.

For one relevant prototype, AP equals reciprocal rank, and Recall@k equals Hit@k. Do not inflate the metric table with mathematically redundant values.

Define normalized rank for a list of `C` candidate classes as `(rank - 1) / (C - 1)`, with zero as best. Break exact score ties deterministically by stable class ID and report the number of tied queries.

### 13.3 Image-gallery retrieval metrics

Create a separate gallery/query protocol with multiple relevant images. Ensure duplicate/source groups do not cross query and gallery.

- Track A gallery: official training images; Track A queries: official test images.
- Track B gallery: support images from the chosen split; Track B queries: test images from that split.
- Validation images select retrieval, calibration, and aggregation settings but are never added to the final query set.
- Self-matches are impossible across the prescribed partitions; still enforce and test explicit query-ID exclusion.
- Every evaluated query must have at least one relevant gallery item at the evaluated relevance level. Report and exclude zero-relevant queries from that metric's denominator rather than assigning an arbitrary score.

Report:

- mAP;
- mAP@R, where R is the number of relevant gallery items;
- Precision@1, @5, @10, @20;
- Recall@1, @5, @10, @20;
- Hit Rate@k;
- MRR;
- NDCG@k;
- CMC curve;
- mean and median positive rank.

When gallery class size varies, report both global query-weighted metrics and class-macro retrieval metrics.

Evaluate relevance at three levels:

1. exact generation;
2. same model regardless of generation;
3. same make.

For hierarchical NDCG, register one gain scheme before looking at test results, for example:

```text
3 = same generation
2 = same model, different generation
1 = same make, different model
0 = different make
```

### 13.4 Verification metrics - optional but valuable

For same-class versus different-class pair verification:

- ROC-AUC;
- PR-AUC;
- equal error rate;
- TAR at fixed FAR values;
- threshold selected on validation only.

Use balanced and realistic hard-negative pair sets separately. Hard negatives should emphasize same-make/different-model and same-model/different-generation pairs.

### 13.5 Calibration and rejection

Cosine similarity is not a probability. If calibrated confidence is needed, fit temperature scaling or another simple calibration on validation data.

Report:

- expected calibration error;
- Brier score;
- negative log-likelihood after calibrated softmax, where appropriate;
- top-1/top-2 similarity margin;
- risk-coverage curve;
- area under the risk-coverage curve;
- accuracy at fixed coverage levels.

Use 15 equal-frequency bins for the primary ECE and provide a reliability diagram. For multiclass Brier score, use the sum of squared probability errors across classes averaged over queries. Fit temperature by minimizing validation negative log-likelihood and freeze it before test evaluation.

Open-set rejection is not a primary claim. If included, create disjoint unknown classes and report AUROC, AUPR, FPR@95TPR, and OSCR.

### 13.6 Efficiency metrics

Report per encoder and end-to-end configuration:

- model parameters;
- native and stored dimension;
- prototype/index size;
- embedding throughput;
- mean and p50/p95/p99 stage latency;
- exact search latency;
- end-to-end API latency;
- peak GPU memory;
- CPU memory;
- storage per 100,000 images;
- energy only if measured with a documented tool and interval.

### 13.7 Statistical analysis

- 95% confidence intervals by hierarchical bootstrap: sample classes, then source/duplicate groups within classes, then include their queries.
- Paired bootstrap for metric differences between methods using identical queries.
- McNemar's test for paired Top-1 correctness where applicable.
- Bootstrap/permutation confidence intervals for mAP and MRR differences.
- Effect size and absolute percentage-point difference, not only p-values.
- Holm-Bonferroni correction for a registered family of multiple pairwise tests.
- Report the fixed primary Track B split with hierarchical-bootstrap intervals; report mean and standard deviation across the five Track B splits as sensitivity analysis.

The resampling unit must respect duplicate/source groups when they exist. Document bootstrap replicate count, random seed, interval method, and failed/degenerate replicate handling.

---

## 14. Experiment matrix and execution order

Do not run the full Cartesian product initially. Use staged experiments.

### Stage E0 - Data and pipeline validation

- 10 classes, at least 5 images per class;
- one encoder;
- P1 preprocessing;
- M0 prototypes;
- verify manifests, no leakage, metrics, deterministic rerun, and artifact logging.

### Stage E1 - Primary encoder comparison

- Tracks A and B;
- encoders: DINOv3, SigLIP2, OpenAI CLIP, OpenCLIP;
- P1 bounding-box crops;
- native dimension;
- M0 global prototype;
- exact search.

### Stage E2 - Dimension ablation

- native, 512, and 256 dimensions for all encoders;
- strict 1024-D table for eligible encoders only;
- otherwise identical to E1.

### Stage E3 - Viewpoint ablation

- M0 global mean;
- M1 view-balanced mean;
- M2 oracle and predicted-viewpoint prototypes;
- M3 multi-prototype clustering;
- use the strongest two encoders from validation, then confirm the winning method on all encoders if budget permits.

### Stage E4 - Segmentation ablation

- P1 versus P3;
- same crop geometry and support/query rows;
- strongest two encoders first;
- include mask-quality audit.

### Stage E5 - Few-shot curves

- support K = 1, 2, 5, 10, 20, and full;
- random versus viewpoint-balanced selection;
- five Track B split seeds;
- five deterministic support-selection replicates inside each split/K condition to separate support-sampling variance from split variance;
- for `K=1`, viewpoint-balanced selection means deterministic selection from the class's most consistently represented viewpoint, not multi-view coverage. Report this limitation explicitly.

### Stage E6 - Image retrieval and verification

- exact generation, model, and make relevance;
- standard and hard-negative settings.

### Stage E7 - Service validation

- raw scene images with zero, one, and multiple cars;
- detection/segmentation/recognition integration;
- exact-versus-Qdrant parity and latency.
- detector precision, recall, AP, and bbox IoU where ground-truth boxes are available;
- mask IoU on a manually annotated subset, plus the larger qualitative mask audit;
- recognition conditional on successful detection;
- end-to-end recognition in which missed detections count as failures.

---

## 15. FiftyOne analysis

Create one FiftyOne dataset or versioned view per protocol with fields for labels, split, viewpoint, preprocessing variant, predictions, ranks, similarity scores, and embedding keys.

Required views:

- original image beside bbox crop and segmented crop;
- correct versus incorrect Top-1 queries;
- correct class at ranks 2-5 and 6-10;
- same-make errors;
- same-model/different-generation errors;
- low-confidence and small-margin predictions;
- segmentation failures;
- per-viewpoint performance slices;
- rare versus frequent classes.

Embedding visualization:

- use UMAP as the primary 2-D view with fixed seeds;
- optionally provide t-SNE as a qualitative secondary view;
- never compare absolute distances between independently fitted 2-D projections;
- color by make, model, generation, viewpoint, split, and preprocessing;
- compare before/after segmentation using paired samples.

Quantitative cluster diagnostics:

- silhouette score;
- Davies-Bouldin index;
- Calinski-Harabasz score;
- k-NN label purity;
- intra-class cosine distribution;
- nearest inter-class cosine distribution;
- centroid separation;
- viewpoint predictability from embeddings as a diagnostic of viewpoint leakage.

Cluster scores must be computed in the original/native or registered PCA space, not in UMAP coordinates.

---

## 16. End-to-end service

### Pipeline

```text
input image
  -> vehicle detection
  -> instance segmentation
  -> crop and mask preprocessing
  -> encoder embedding
  -> dimension transform
  -> prototype/vector search
  -> hierarchy-aware ranked results per vehicle
```

### API endpoints

#### `POST /v1/recognize`

Input:

- multipart image;
- optional `profile_id`;
- optional `top_k`, default 10;
- optional `return_masks`, default false;
- optional `return_embeddings`, default false and disabled in public deployment.

Output skeleton:

```json
{
  "request_id": "uuid",
  "profile_id": "track_b__dinov3_vitl16__p3__native__m1",
  "index_version": "string",
  "vehicles": [
    {
      "detection_id": 0,
      "bbox_xyxy": [0, 0, 100, 100],
      "detection_score": 0.99,
      "viewpoint": {"label": "front_side", "score": 0.88},
      "predictions": [
        {
          "rank": 1,
          "make": "Dacia",
          "model": "Logan",
          "generation": "II",
          "class_id": "dacia_logan_g2",
          "cosine_similarity": 0.81,
          "calibrated_confidence": 0.72
        }
      ],
      "decision": "accepted",
      "latency_ms": {
        "detection": 0.0,
        "segmentation": 0.0,
        "embedding": 0.0,
        "search": 0.0
      }
    }
  ]
}
```

#### Other endpoints

- `GET /health/live`
- `GET /health/ready`
- `GET /v1/profiles`
- `GET /v1/indexes/current`
- `POST /v1/indexes/reload` with authentication

### Service behavior

- return an empty `vehicles` array when no car is detected;
- process every detected car above the configured threshold;
- preserve input-to-detection coordinate mapping;
- reject invalid or oversized files safely;
- batch car crops from one image for encoder inference;
- expose model/index versions in every response;
- never describe cosine similarity as probability;
- make calibrated confidence nullable;
- log timing and version metadata without logging raw images by default.

---

## 17. Reproducibility and experiment tracking

Use:

- Python package with typed interfaces;
- configuration files rather than hard-coded experiment choices;
- MLflow for parameters, metrics, figures, tables, and run relationships;
- DVC or equivalent immutable manifests for dataset metadata and derived artifacts;
- Docker for reproducible runtime;
- pinned Python dependencies and model revisions;
- deterministic seeds where supported;
- pre-commit formatting/linting and automated tests.

Every run manifest must contain:

```text
run_id
git_commit
dirty_worktree_flag
dataset_protocol
dataset_manifest_hash
split_manifest_hash
generation_mapping_hash
encoder_id
encoder_revision
encoder_checksum
preprocessing_id
dimension_transform_id
prototype_method
support_k
seed
hardware
software_versions
start_time
end_time
```

Large image data must not be committed to Git. Respect the CompCars non-commercial research license and do not redistribute images.

---

## 18. Proposed repository structure

```text
advanced-vehicle-recognition/
├── CONTRIBUTING.md
├── pyproject.toml
├── README.md
├── configs/
│   ├── datasets/
│   ├── encoders/
│   ├── preprocessing/
│   ├── experiments/
│   └── service/
├── data/
│   ├── external/              # ignored; CompCars files
│   ├── manifests/
│   ├── mappings/
│   └── splits/
├── src/avr/
│   ├── data/
│   ├── generation_mapping/
│   ├── detection/
│   ├── segmentation/
│   ├── preprocessing/
│   ├── encoders/
│   ├── dimensions/
│   ├── prototypes/
│   ├── retrieval/
│   ├── evaluation/
│   ├── visualization/
│   ├── tracking/
│   └── service/
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── regression/
│   └── fixtures/
├── scripts/
├── notebooks/                 # analysis only; no canonical pipeline logic
├── reports/
│   ├── figures/
│   ├── tables/
│   └── paper_inputs/
├── docs/
│   ├── adr/
│   ├── protocols/
│   └── handoffs/
└── docker/
```

Canonical logic belongs in `src/avr`, never only in notebooks.

---

## 19. Workstream structure

The research lead is the integrator. They create shared schemas first, assign isolated tasks, review evidence, and prevent workstreams from editing the same file concurrently.

### A1 - Dataset and split workstream

Responsibilities:

- parse CompCars metadata and official splits;
- build immutable image manifest;
- deduplicate and group images;
- create Track A and Track B split manifests;
- produce distribution and leakage reports.

May not define generation labels.

### A2 - Generation mapping workstream

Responsibilities:

- build the mapping schema and curation queue;
- collect source provenance;
- implement validation rules;
- produce reviewer sheets and mapping coverage reports.

May not act as either reviewer.

### A3 - Encoder and preprocessing workstream

Responsibilities:

- implement encoder adapters and checkpoint registry;
- implement P0-P3 preprocessing;
- verify output dimensions, normalization, determinism, throughput, and storage.

### A4 - Prototype and retrieval workstream

Responsibilities:

- implement M0-M5;
- exact FAISS/matrix search;
- Qdrant export/import and parity tests;
- dimension transformations.

### A5 - Evaluation and statistics workstream

Responsibilities:

- implement metric definitions;
- paired confidence intervals and significance tests;
- hierarchical relevance and error taxonomy;
- publication tables with comparability guards.

### A6 - FiftyOne and visualization workstream

Responsibilities:

- populate FiftyOne fields and views;
- UMAP and cluster diagnostics;
- publication-ready plots and qualitative failure panels.

### A7 - Service workstream

Responsibilities:

- implement FastAPI schemas and pipeline orchestration;
- multi-car batching;
- health/version endpoints;
- latency instrumentation and Docker deployment.

### A8 - Reproducibility and QA workstream

Responsibilities:

- tests, CI, run manifests, config validation, environment capture;
- reproduce a selected experiment from a clean environment;
- audit artifact lineage and licensing constraints.

### A9 - Scientific reporting workstream

Responsibilities:

- maintain related-work comparison table;
- verify task/split/label comparability;
- draft methods, experiment protocol, limitations, and figure captions from registered artifacts only.

May not invent or manually transcribe result values without source artifacts.

### Required handoff format

Every workstream returns:

```text
task_id
scope_completed
files_changed
commands_run
tests_and_results
artifacts_created
assumptions
unresolved_issues
recommended_next_action
```

---

## 20. Repeatable workflows

Document project workflows only where they encode a repeatable procedure. Workflow documents must not contain result values or mutable dataset facts.

Recommended workflows:

- experiment design and hypothesis registration;
- CompCars manifest validation;
- generation-mapping review;
- encoder adapter conformance testing;
- metric validation against hand-calculated fixtures;
- experiment launch and MLflow artifact checks;
- FiftyOne dataset refresh;
- publication figure/table generation;
- verification before completion;
- code review before merging.

Use systematic debugging for failed tests or inconsistent metrics. Parallelise only tasks with fixed interfaces and no shared mutable files. Require an independent review for split logic, metric logic, and generation mapping because errors there can invalidate the article.

---

## 21. Scientific reporting and SOTA comparison rules

Maintain a literature comparison table with these columns:

```text
paper
year
dataset
dataset_subset
split
number_of_images
number_of_classes
label_level
training_regime
backbone
input_resolution
preprocessing
metric_definition
reported_result
directly_comparable
reason_if_not_comparable
```

Rules:

1. Direct comparisons require the same dataset subset, split, class definition, and metric.
2. Track A can be compared with official 431-model Part-I results only when the official split is reproduced.
3. Track B is a new protocol; compare methods internally and compare external papers only descriptively unless they use the published Track B manifest.
4. Separate frozen prototypes, frozen linear probes, fine-tuned models, and fully supervised bespoke architectures.
5. Report parameter count and compute alongside performance.
6. Never claim SOTA based on a metric computed on a different split.
7. Publish mapping and split metadata when licensing permits, but never redistribute CompCars images.

---

## 22. Required publication figures and tables

### Figures

1. End-to-end system architecture.
2. Dataset and two-track protocol diagram.
3. Prototype construction variants.
4. Encoder comparison with 95% confidence intervals.
5. Top-k/CMC curves.
6. Few-shot performance versus K.
7. Performance versus embedding dimension and index size.
8. Viewpoint prototype ablation.
9. Before/after segmentation comparison.
10. FiftyOne/UMAP qualitative cluster panels.
11. Hierarchical confusion/error flow.
12. Accuracy-latency-memory Pareto plot.

### Tables

1. Dataset/protocol statistics.
2. Generation mapping coverage and agreement.
3. Encoder/checkpoint characteristics.
4. Track A benchmark results.
5. Track B generation results.
6. Ranking and image-retrieval results.
7. Viewpoint and segmentation ablations.
8. Dimension and few-shot ablations.
9. Calibration and rejection results.
10. Efficiency and deployment results.
11. Literature comparison with comparability flags.

All figures and tables must be generated from machine-readable result artifacts, not manually edited values.

---

## 23. Tests and acceptance criteria

### Data

- all manifest paths resolve;
- labels match CompCars metadata;
- no exact/near-duplicate group crosses a split;
- every Track B test class has support examples;
- all Track B generation rows are approved;
- split regeneration from the same manifest and seed is byte-identical.

### Embeddings

- expected shape and finite values;
- norm approximately 1 after normalization;
- batch and single-image outputs agree within tolerance;
- saved and reloaded embeddings agree;
- checkpoint revision and checksum recorded.

### Prototypes and ranking

- hand-calculated fixture matches cosine ranking;
- prototype order is stable;
- ties follow a documented deterministic rule;
- no query embedding enters prototype construction;
- FAISS exact and matrix search return identical ranks.

### Metrics

- compare core metrics against scikit-learn or a trusted reference on fixtures;
- verify Top-k, MRR, AP, mAP, NDCG, CMC, and hierarchical gains by hand on small examples;
- verify the single-relevant-item identities noted above;
- bootstrap is deterministic for a fixed seed.

### API

- zero, one, and multiple vehicle cases;
- invalid input and oversized input;
- deterministic response schema;
- model/index version present;
- Qdrant and exact-search top-k parity on test fixtures;
- stage timings are non-negative and complete.

### Detector and segmenter

- predicted boxes use the documented coordinate convention and round-trip correctly through crop transforms;
- bbox metrics match a trusted implementation on fixtures;
- manually annotated mask subset is versioned and reviewer-checked;
- conditional recognition and true end-to-end recognition are both reported;
- missed vehicles are not silently removed from the end-to-end denominator.

### Project completion

The project is not scientifically complete until:

- both primary tracks run end to end;
- all primary metrics have confidence intervals;
- generation mapping has provenance and dual review;
- no leakage audit failure remains;
- the selected experiment is reproduced from a clean environment;
- every paper table is regenerable from stored artifacts;
- the service produces per-car ranked results on multi-car images.

---

## 24. Risks and mitigations

| Risk | Consequence | Mitigation |
|---|---|---|
| CompCars year is confused with generation | invalid scientific claim | curated, versioned mapping with provenance and dual review |
| Near duplicates cross splits | inflated performance | hash grouping and source-group split constraints |
| Official split is unavailable or altered | invalid SOTA comparison | verify split files; label reconstructed results clearly |
| Viewpoint imbalance dominates mean prototypes | unfair class prototypes | view-balanced and per-view ablations |
| Segmentation masks remove discriminative parts | reduced performance | mask audit, P1/P3 paired comparison, failure slices |
| Different encoder sizes/resolutions confound results | weak attribution | report compute/model characteristics and dimension-controlled results |
| Forced 1024-D expansion | meaningless fairness claim | never up-project; use native and PCA reductions |
| Full factorial experiment explosion | excessive compute | staged experiment matrix and validation-based pruning |
| Approximate vector search changes ranking | incorrect paper metrics | exact search for scientific evaluation |
| Prototype method compared directly with fine-tuned SOTA | misleading conclusion | separate training regimes and comparability flags |
| 2-D cluster plots are over-interpreted | unsupported conclusions | quantitative metrics in original space and fixed visualization protocol |

---

## 25. Milestones and gates

### M0 - Protocol freeze

Outputs: approved context, ADRs, dataset license note, exact Track A split identity, generation schema.

### M1 - Dataset foundation

Outputs: validated manifest, deduplication report, split manifests, initial mapping coverage report.

### M2 - Embedding baseline

Outputs: all four encoder adapters, P1 embeddings, M0 prototypes, deterministic E0/E1 run.

### M3 - Scientific core

Outputs: dimension, viewpoint, segmentation, and few-shot results with confidence intervals.

### M4 - Retrieval and visualization

Outputs: image-gallery results, FiftyOne dataset/views, cluster diagnostics, failure taxonomy.

### M5 - Service

Outputs: FastAPI pipeline, Qdrant index, Docker deployment, correctness and latency report.

### M6 - Article package

Outputs: reproducible figures/tables, literature matrix, methods/protocol text, limitations, clean-run reproduction report.

Each milestone requires a review of artifacts and acceptance tests before the next begins.

---

## 26. Open decisions that implementation must confirm

These are technical confirmations, not invitations to silently redesign the study:

1. Exact local path and release version of CompCars.
2. Availability and checksum of the official Part-I train/test files.
3. Exact model IDs/revisions for DINOv3, SigLIP2, OpenAI CLIP, and OpenCLIP.
4. Compute budget and whether the RTX 3080 10 GB or larger server GPUs will run extraction.
5. Final RF-DETR segmentation checkpoint and license.
6. Automotive sources and the two human reviewers for the generation mapping. Until named, the mapping remains provisional and Track B cannot be called final.
7. Minimum Track B class threshold after the first coverage audit.
8. Whether the optional cross-domain and open-set studies fit the article schedule.

Any decision that changes a primary hypothesis, split, metric, or task definition requires an ADR and protocol version increment.

Engineer-owned defaults that may be finalized in an ADR without new research approval include Parquet manifests, Pydantic schemas, Hydra/OmegaConf configuration, FAISS `IndexFlatIP`, lexicographic tie-breaking, neutral-background implementation details, UMAP parameters, MLflow directory conventions, and internal FastAPI module boundaries.

---

## 27. First operational task

The first task after approving this context:

> Read the complete project context and inspect the repository without modifying files. Produce a gap analysis covering current code, available CompCars metadata and split files, model-access constraints, generation-mapping inputs, compute limits, and reproducibility risks. Then propose a staged implementation plan beginning with M0 and M1. Do not implement until the dataset protocol, file schemas, workstream ownership, tests, and acceptance criteria are approved. The main scientific constraints are frozen encoders, two separate evaluation tracks, leakage-safe support/query splits, exact search for reported metrics, and a provenance-backed generation mapping.

---

## 28. Primary references

- Yang, L., Luo, P., Loy, C. C., and Tang, X. *A Large-Scale Car Dataset for Fine-Grained Categorization and Verification*. CVPR 2015 / arXiv extension. https://arxiv.org/abs/1506.08959
- Official CompCars dataset page. https://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/
- Buzzelli, M., and Segantin, L. *Revisiting the CompCars Dataset for Hierarchical Car Classification: New Annotations, Experiments, and Results*. Sensors 2021. https://doi.org/10.3390/s21020596
- Oquab et al. / Meta AI. DINOv3 official repository and model documentation. https://github.com/facebookresearch/dinov3
- Tschannen et al. *SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features*. https://arxiv.org/abs/2502.14786
- Radford et al. *Learning Transferable Visual Models From Natural Language Supervision*. https://arxiv.org/abs/2103.00020
- OpenAI CLIP official repository. https://github.com/openai/CLIP
- OpenCLIP official repository. https://github.com/mlfoundations/open_clip

