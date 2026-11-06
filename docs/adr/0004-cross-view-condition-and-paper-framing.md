---
status: accepted
date: 2026-11-06
---

# Register a cross-view evaluation condition; adopt the paper framing

Working title: **"Fine-Grained Vehicle Generation Recognition on CompCars with Frozen Visual Encoders and Viewpoint-Aware Prototypes"**. The gap the paper is built around: how reliably frozen visual encoders distinguish generations of the same vehicle model when few labeled reference images are available and the query viewpoint differs from the references. The decisive experiment is M0 versus M1 prototypes on identical support, reported as exact-generation macro F1 and the wrong-generation/same-model error rate (already the Track B primary outcome and §13.1 decomposition). Track A remains the link to prior work.

Protocol v1.1 had no condition in which the query viewpoint is absent from support. We add a **cross-view condition** as a registered secondary analysis on Track B: for each CompCars viewpoint `v`, support is restricted to images of the other viewpoints and queries are restricted to viewpoint `v`, derived from the primary seed-73 split without moving any image between partitions. A class is eligible for held-out view `v` only if it keeps at least five support images and at least one test image after the restriction; eligibility counts are reported per view. The condition is crossed with E3 prototype methods (M1 view-balanced over the remaining views; M2 cannot use `v` and routes to the nearest available view) and with E5 support sizes. Cross-view results form a fourth significance family with Holm-Bonferroni correction. Primary outcomes, splits and hypotheses are unchanged; H2 gains a sub-hypothesis H2b that view-balanced prototypes lose less than global-mean prototypes under cross-view. Protocol version is now 1.2.

Target journals, in order: Machine Vision and Applications then Sensors for a benchmark-and-system paper; Image and Vision Computing then MVA if a consistent methodological gain is established. Journal choice affects figure order and length only, not the protocol; eligibility under the doctoral school's index and quartile rules is still to be confirmed. The gap is a candidate until the literature review in `docs/research/` confirms it against Buzzelli & Segantin (Sensors 2021) and the NTOU-MMR generation-recognition work.
