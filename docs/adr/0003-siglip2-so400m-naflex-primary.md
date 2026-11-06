---
status: accepted
date: 2026-11-06
supersedes: context §9 SigLIP2 row (L/16 at 384 px)
---

# SigLIP2 primary checkpoint is so400m NaFlex at native 1152-D

Protocol v1.0 listed SigLIP2 L/16-384 (1024-D) as the SigLIP2 primary. Hugging Face publishes NaFlex variants only for base and so400m, and the research lead chose NaFlex so SigLIP2 sees the P1 crop at its native aspect ratio rather than a resized square. The primary is therefore `google/siglip2-so400m-patch16-naflex`, pooled output 1152-D, used at its native dimension, with `max_num_patches` fixed in the encoder registry. Base NaFlex (768-D) was rejected because it would make SigLIP2 the smallest model in the comparison and weaken H1.

Consequences: under §10, SigLIP2 gains a `pca_1024` condition and is excluded from the strict native-1024 table; the §9 caveat that conclusions describe checkpoints, not objectives, applies with extra force since this model differs from the others in size, resolution handling and dimension. The L/16-384 square variant may be registered later as a separate aspect-ratio ablation. Protocol version is now 1.1.
