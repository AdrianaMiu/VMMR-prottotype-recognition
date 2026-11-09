# P3 segmentation report (M1-19, PROTOCOL.md §8)

Masks for the first 5,000 valid manifest images (``image_id`` order, ``0000545a3c3a441b`` to
``093275e3296c6666``) built with `scripts/build_p3_masks.py` on one RTX 3080 10 GB that was
shared with another running job. The full valid manifest (136,365 images) is still to be run;
the build is resumable per shard, so rerunning the script continues from shard 5.

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
| detection threshold | 0.3; labels kept: car, truck, bus |
| instance selection | predicted box with the highest IoU against the CompCars GT box; `low_overlap` below IoU 0.5 |
| crop geometry | P1: GT box expanded 5 % per side, clipped, padded with RGB (127, 127, 127); P3 differs only by masking |

## Sample and throughput

| quantity | value |
|---|---|
| images processed | 5,000 (first valid images in `image_id` order) |
| batch size / shard size | 16 / 1,000 |
| wall time (first mask written to last shard closed) | 196.6 s |
| throughput | 25.4 images/s (steady-state shards 2-5: about 28 images/s) |
| extrapolated full valid manifest (136,365 images) | about 5,370 s, i.e. about 90 min |

Timing comes from the file timestamps of the masks and shards written by the run
(2026-11-06 17:02:46 to 17:06:02 local), excluding model load and checkpoint verification.
The GPU was shared with another job during the run, so a free GPU should be at least as fast.

Outputs: one PNG mask per image under `data/masks/` (git-ignored), per-shard Parquet files
under `data/masks/_shards/`, and the merged metadata
`data/manifests/p3_mask_metadata.parquet` (sha256
`e952d42ce4179cbd37a466799120c024e1a47c1860147fa422f3d4381630ed70`, see
`p3_mask_metadata.sha256`) with `p3_mask_metadata.summary.json`.

## Mask status and failure rate

| mask_status | images | share |
|---|---|---|
| ok | 4,990 | 99.80 % |
| no_detection | 7 | 0.14 % |
| low_overlap | 3 | 0.06 % |

Mask failure rate (`no_detection` + `low_overlap`): **10 / 5,000 = 0.20 %**.

Failed images keep the unmasked P1 crop and an all-true mask so they stay traceable; downstream
experiments filter on `mask_status`. The three `low_overlap` images had best box IoU 0.316,
0.000 and 0.227 (detection scores 0.966, 0.959 and 0.539), i.e. a vehicle was found but not the
annotated one.

## Quality statistics over the 4,990 `ok` masks

`gt_iou` is the IoU between the selected predicted box and the CompCars GT box;
`mask_area_fraction` is the share of P1-crop pixels covered by the mask; `detection_score`
is the RF-DETR confidence of the selected instance.

| statistic | gt_iou | mask_area_fraction | detection_score |
|---|---|---|---|
| min | 0.6129 | 0.2805 | 0.3003 |
| p5 | 0.9597 | 0.5414 | 0.5841 |
| p25 | 0.9753 | 0.5775 | 0.9245 |
| median | 0.9818 | 0.6051 | 0.9606 |
| mean | 0.9791 | 0.6094 | 0.9066 |
| p75 | 0.9870 | 0.6401 | 0.9680 |
| p95 | 0.9925 | 0.6907 | 0.9735 |
| max | 0.9984 | 0.9708 | 0.9805 |
| std | 0.0164 | 0.0476 | 0.1335 |

## Open items

- Run the remaining 131,365 valid images (about 90 min on a free RTX 3080) and refresh this
  report from `p3_mask_metadata.summary.json`.
- Manual inspection of a stratified sample of at least 200 masks across class frequency,
  viewpoint and image quality (PROTOCOL.md §8) is not yet done.
- `gt_iou` is a box IoU, not a mask-vs-box IoU; a pixel-level audit belongs to the mask inspection.
