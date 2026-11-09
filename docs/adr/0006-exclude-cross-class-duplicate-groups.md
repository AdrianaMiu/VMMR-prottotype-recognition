---
status: accepted
date: 2026-11-09
---

# Images whose duplicate group spans more than one class are excluded from Track B

The leakage audit (`reports/leakage/track_b_seed_*.md`, first run) found 91 to 103 duplicate groups per seed crossing Track B partitions. Every one of them spans two or three model-year classes: the same photograph, or a near-identical one, is filed under different CompCars folders. A group-aware split that assigns groups per class cannot keep such a group in one partition without also deciding which class owns it.

Decision: before Track B eligibility, every image whose `duplicate_group_id` contains images of more than one class is marked `excluded` with reason `cross_class_duplicate`, counted in the eligibility audit, and left out of all Track B partitions and cross-view variants. This follows PROTOCOL.md §6 rule 9 (exclude unresolved overlaps) and §7 (every duplicate group stays in one partition). The images remain in the public manifest and in Track A, where the official split is not altered and the 52 crossing groups are reported instead. The excluded set is roughly 360 images, about 0.3 % of the eligible pool.
