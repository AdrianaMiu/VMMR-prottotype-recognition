# Track A split identity

**Label to use in every Track A table:** official CVPR 2015 half/half classification split (30,955 images, 431 models); release version unverified; hashes recorded.

Recorded 2026-11-06 from the local CompCars web-nature release under `train_test_split/`.
The machine-readable record is `data/manifests/track_a_split_identity.json` (full hashes for all
22 distributed split files). `tests/regression/test_track_a_split_identity.py` re-derives it from
the files wherever the release is present and fails if any of them changed.

## Which official setting this is

CompCars was published with two official classification settings for the same 431 models
(see `docs/research/generation-recognition-gap.md`, §4(c) item 7):

| Setting | Images | Train / test | Where described |
|---|---:|---|---|
| CVPR 2015 half/half | 30,955 | 16,016 / 14,939 | Yang et al. 2015, conference version §4 |
| arXiv v2 70/30 | 52,083 | 36,456 / 15,627 | Yang et al. 2015, arXiv v2 §5 |

The distributed files sum to **30,955** lines, so they are the **CVPR 2015 half/half**
setting. No release notes or version string ship with the dataset, hence "release version
unverified" (PROTOCOL.md §5 asks for the release version to be recorded). Consequences for the
§21 literature matrix, to be applied when it is built (M6):

- Direct comparison is allowed only with papers that report on this 30,955-image split.
- Papers on the 70/30 setting (e.g. EP-CNN, IEEE TCSVT 2022) get `directly_comparable = no`,
  reason "different official split version".
- Buzzelli & Segantin (Sensors 2021) use their own feature-distance split and are never
  directly comparable; their leakage finding is why the duplicate audit (#8, #12) is reported
  alongside every Track A number.

## Classification split (the official split)

| | train.txt | test.txt |
|---|---|---|
| lines | 16,016 | 14,939 |
| makes referenced | 75 | 75 |
| models referenced | 431 | 431 |
| SHA-256 of file | `9966d3e91b7dc735e224e04f5c53f33fbbe48d7c98e15509ca08e897d120ae9b` | `474c4eac44f79ec378b7f755c79b25b14bcc1251450a2766bf03cdfcb7bb2dd0` |
| SHA-256 of first 3 lines | `e64b4e1d35f84555ee5fba88969e54cc7ab2f5633f27153932238a66de9a64f1` | `a1b61434c4f39f183169b077d0b36fe0a0305547cab93ba028b1ef99630928c9` |
| SHA-256 of last 3 lines | `82b978cb8db9893837fe00f2bbf7ef581caeb32c362e0c71e4683ecf5ecf3890` | `1dc64aea66fc29e4a40713aafb32a37461b625566ba7a8dc9759f2b8dc71bfee` |

No path appears in both files. Both sides reference all 431 models and 75 of the 163 makes.
Each line is `make_id/model_id/year/file.jpg`. Image paths themselves are not published
(ADR-0002): boundaries are pinned by hashing the first and last three lines.

## Other distributed split files

Hashed for completeness; not used by Track A. Full hashes and make/model counts (counted over
every path on a line, so pair files count both sides) are in the JSON record.

| File | Lines | SHA-256 (prefix) |
|---|---:|---|
| `part/train_part_1.txt` | 1,470 | `d7133d783c2a80c8…` |
| `part/train_part_2.txt` | 1,419 | `05c08dac8403ff31…` |
| `part/train_part_3.txt` | 1,308 | `ed9a30d5c9f6326b…` |
| `part/train_part_4.txt` | 1,374 | `93322481d98d8936…` |
| `part/train_part_5.txt` | 1,359 | `b3824f7786c6e2c9…` |
| `part/train_part_6.txt` | 1,382 | `a4b7bdc58e25cb3a…` |
| `part/train_part_7.txt` | 1,373 | `808ec4d963881e54…` |
| `part/train_part_8.txt` | 1,374 | `c17ebfe82564a2b8…` |
| `part/test_part_1.txt` | 1,246 | `f745695fd7c56b7e…` |
| `part/test_part_2.txt` | 1,191 | `69fdb35de1e0f098…` |
| `part/test_part_3.txt` | 1,100 | `48fbf136b2f14b74…` |
| `part/test_part_4.txt` | 1,146 | `31f7d28bc85cb8fe…` |
| `part/test_part_5.txt` | 1,130 | `24cd6fd33ac06ef9…` |
| `part/test_part_6.txt` | 1,166 | `7d563c90c1c39aec…` |
| `part/test_part_7.txt` | 1,152 | `dc8ea3a2f190c1b1…` |
| `part/test_part_8.txt` | 1,159 | `c8ced4ca306e967b…` |
| `verification/verification_train.txt` | 4,454 | `0641189066a6a086…` |
| `verification/verification_pairs_easy.txt` | 20,000 | `7bb1dafd3d7c1310…` |
| `verification/verification_pairs_medium.txt` | 20,000 | `96489265d965aa34…` |
| `verification/verification_pairs_hard.txt` | 20,000 | `9a153f3c415d0c1a…` |
