# P3 segmentation report (M1-19, PROTOCOL.md §8)

Masks for every valid manifest image (136,365 images, `image_id` order, `0000545a3c3a441b`
to `ffff6f6fad6ad0c8`) built with `scripts/build_p3_masks.py` on one RTX 3080 10 GB. The
build is resumable per shard: the first 5,000 images were segmented on 2026-11-06 (see
"History" below) and the remaining 131,365 on 2026-11-09, continuing from shard 5. The
numbers in this report come from `data/manifests/p3_mask_metadata.summary.json` and the
merged metadata table.

## Frozen segmenter

Pinned in `configs/segmentation/rfdetr_seg.yaml` (same discipline as the encoder registry).

| field | value |
|---|---|
| package | `rfdetr` 1.12.0 (`uv.lock`), variant `RFDETRSegMedium`, DINOv2-small backbone, 432 px |
| checkpoint | `rf-detr-seg-medium.pt`, downloaded from `https://repo.roboflow.com/rfdetr/rf-detr-seg-m-ft.pth` |
| revision | `rf-detr-seg-m-ft` (COCO fine-tune, Roboflow release; no further version tag) |
| licence | Apache-2.0 (Roboflow, 2025) |
| size | 143,024,058 bytes |
| md5 | `a49af1562c3719227ad43d0ca53b4c7a` |
| sha256 | `3ad325094735f431aee9962a8d204d68eb5bfc393d53e7e836e70998fef5ea58` |
| detection threshold | 0.3 (`threshold`); labels kept: car, truck, bus |
| instance selection | predicted box with the highest IoU against the CompCars GT box; `low_overlap` below `min_iou` 0.5 (`configs/segmentation/rfdetr_seg.yaml`) |
| crop geometry | P1: GT box expanded by `bbox_expand_fraction` 0.05 per side (`configs/preprocessing/p1_bbox_crop.yaml`), clipped, padded with RGB (127, 127, 127); P3 differs only by masking |

## Full run and throughput

| quantity | value |
|---|---|
| images processed | 136,365 (every valid manifest image) |
| batch size / shard size | 16 / 1,000 (137 shards, the last with 365 images) |
| wall time of the resumed run (131,365 images, shards 5-136) | 5,362 s |
| throughput | 25.4 images/s |

Timing comes from the file timestamps of the masks and shards written by the resumed run
(last shard closed 2026-11-09 10:28:49 local), excluding model load and checkpoint
verification; it matches the 25.4 images/s measured on the 5,000-image sample.

Outputs: one PNG mask per image under `data/masks/` (git-ignored), per-shard Parquet files
under `data/masks/_shards/`, and the merged metadata
`data/manifests/p3_mask_metadata.parquet` (DVC-tracked, 3.3 MB; sha256
`bb17a917201b43ca4db4aeb748b784b1d7271ca417bd2fd692137d9223d663b8` in the committed
`p3_mask_metadata.sha256`) with `p3_mask_metadata.summary.json`. The M1 gate records this
digest as `mask_metadata_sha256` for every P3 run (`reports/m1_gate/README.md`).

## Mask status and failure rate

| mask_status | images | share |
|---|---|---|
| ok | 136,097 | 99.80 % |
| no_detection | 223 | 0.16 % |
| low_overlap | 45 | 0.03 % |

Mask failure rate (`no_detection` + `low_overlap`): **268 / 136,365 = 0.20 %**.

Failed images keep the unmasked P1 crop and an all-true mask so they stay traceable; downstream
experiments filter on `mask_status` (the P3 embedding store therefore holds 136,097 rows). The 45
`low_overlap` images had best box IoU between 0.000 and 0.497 (median 0.316) at detection scores
between 0.306 and 0.972 (median 0.918), i.e. a vehicle was found but not the annotated one.

## Quality statistics over the 136,097 `ok` masks

`gt_iou` is the IoU between the selected predicted box and the CompCars GT box;
`mask_area_fraction` is the share of P1-crop pixels covered by the mask; `detection_score`
is the RF-DETR confidence of the selected instance.

| statistic | gt_iou | mask_area_fraction | detection_score |
|---|---|---|---|
| min | 0.5080 | 0.2231 | 0.3000 |
| p5 | 0.9587 | 0.5407 | 0.5870 |
| p25 | 0.9750 | 0.5763 | 0.9244 |
| median | 0.9818 | 0.6047 | 0.9601 |
| mean | 0.9789 | 0.6094 | 0.9063 |
| p75 | 0.9869 | 0.6404 | 0.9680 |
| p95 | 0.9925 | 0.6921 | 0.9735 |
| max | 0.9994 | 0.9895 | 0.9812 |
| std | 0.0160 | 0.0487 | 0.1326 |

## History: the 5,000-image sample (2026-11-06)

The first 5,000 valid images (`0000545a3c3a441b` to `093275e3296c6666`) were segmented on
2026-11-06 while the GPU was shared with another job: 196.6 s wall (first mask written to last
shard closed, 17:02:46 to 17:06:02 local), 25.4 images/s (steady-state shards 2-5 about 28
images/s). Status counts: ok 4,990 (99.80 %), no_detection 7 (0.14 %), low_overlap 3 (0.06 %);
failure rate 10 / 5,000 = 0.20 %. The merged metadata of that sample had sha256
`e952d42ce4179cbd37a466799120c024e1a47c1860147fa422f3d4381630ed70`; its quality statistics
(gt_iou mean 0.9791, mask_area_fraction mean 0.6094, detection_score mean 0.9066) are within
0.0003 of the full-run values above, so the sample extrapolation (about 5,370 s for the full
manifest) held.

## Open items

- Manual inspection of a stratified sample of at least 200 masks across class frequency,
  viewpoint and image quality (PROTOCOL.md §8) is not yet done.
- `gt_iou` is a box IoU, not a mask-vs-box IoU; a pixel-level audit belongs to the mask inspection.
