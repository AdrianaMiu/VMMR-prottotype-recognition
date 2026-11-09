# Embedding extraction report (M1-13, PROTOCOL.md §9)

DINOv3 ViT-B/16 embeddings for the first 2,000 valid manifest images (`image_id` order) in
the P1 and P3 variants, extracted with `scripts/extract_embeddings.py` on one RTX 3080 10 GB
shared with the running P3 mask build (about 2 GB, GPU busy). The full valid manifest is
the M1 gate's job (#16); the store is resumable per shard so the same command continues.

## Encoder and contract

| field | value |
|---|---|
| registry entry | `configs/encoders/dinov3_vitb16.yaml` (`facebook/dinov3-vitb16-pretrain-lvd1689m`) |
| revision | `5931719e67bbdb9737e363e781fb0c67687896bc` |
| checkpoint sha256 | `9a21ac3df0c63839d62612dda6f454d816c25611cc7a52966ed5a5a94921dc8b` (re-hashed at load, must equal the registry) |
| preprocessing | `dinov3_vit16_224_v1` (official `DINOv3ViTImageProcessor`: resize 256, centre crop 224) applied to the P1 or P3 crop |
| pooled output | `pooler_output` (CLS token after the final norm), 768-D |
| compute dtype | float16 on CUDA; cast to float32, then L2-normalised |
| store | `data/embeddings/dinov3_vitb16/<variant>/shard_<k>.parquet` (`image_id`, `embedding` fixed-size list<float32>[768]) + `shard_<k>.json` (identity above, row count, dim, Parquet sha256) |

## Runs

| quantity | P1 | P3 |
|---|---|---|
| images requested / embedded | 2,000 / 2,000 | 2,000 / 1,996 |
| excluded (`missing.json`) | 0 | 4: 2 `no_detection`, 2 `low_overlap` (never replaced by P1) |
| store shape | (2000, 768) | (1996, 768) |
| batch / shard size | 32 / 1,000 | 32 / 1,000 |
| throughput | 92.9 img/s | 37.8 img/s |
| peak GPU memory (torch allocated) | 302 MB | 302 MB |
| wall time | 21.5 s | 52.8 s |

P3 is slower because every image also decodes its mask PNG and applies the mask on the CPU
(single process) while the GPU was shared with the mask build; it is not an encoder cost.
Extrapolated to the 136,365 valid images: about 25 min for P1 and about 60 min for P3.

## Checks

- Every stored vector is finite with norm within 1.2e-7 of 1.
- Same-image P1 vs P3 cosine: mean 0.967 (min 0.323); unrelated pairs: mean 0.108, so the
  masks line up with the crops.
- Batch-of-8 vs single-image embeddings agree to 2.3e-4 max-abs in fp16 (test bound 1e-3,
  see `tests/integration/test_frozen_encoder_dinov3.py`).
- Interrupt-and-resume: a run stopped after one 50-image shard and rerun with a larger limit
  leaves shard 0 byte-identical (same mtime and sha256) and embeds only the new shard
  (`tests/integration/test_extract_embeddings_resume.py`).

Masks were read from the M1-19 build (`extraction.masks_dir=<that checkout>/data/masks`);
the metadata used is `data/manifests/p3_mask_metadata.parquet` (first 5,000 images).
