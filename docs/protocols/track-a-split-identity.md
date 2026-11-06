# Track A split identity

**Label to use in every Track A table:** official CVPR 2015 half/half classification split (30,955 images, 431 models); release version unverified; hashes recorded.

Recorded 2026-11-06 from the local CompCars web-nature release under `train_test_split/`.
Machine-readable copy: `data/manifests/track_a_split_identity.json`, re-verified by
`tests/regression/test_track_a_split_identity.py` wherever the release is present.

## Which official setting this is

CompCars was published with two official classification settings for the same 431 models
(see `docs/research/generation-recognition-gap.md`, §4(c) item 7):

| Setting | Images | Train / test | Where described |
|---|---:|---|---|
| CVPR 2015 half/half | 30,955 | 16,016 / 14,939 | Yang et al. 2015, conference version §4 |
| arXiv v2 70/30 | 52,083 | 36,456 / 15,627 | Yang et al. 2015, arXiv v2 §5 |

The distributed files sum to **30,955** lines, so they are the **CVPR 2015 half/half**
setting. No release notes or version string ship with the dataset, hence "release version
unverified". Consequences for PROTOCOL.md §21:

- Direct comparison is allowed only with papers that report on this 30,955-image split.
- Papers on the 70/30 setting (e.g. EP-CNN, IEEE TCSVT 2022) get `directly_comparable = no`,
  reason "different official split version".
- Buzzelli & Segantin (Sensors 2021) use their own feature-distance split and are never
  directly comparable; their leakage finding is why the duplicate audit (#8, #12) is reported
  alongside every Track A number.

## Classification split

| | train.txt | test.txt |
|---|---|---|
| lines | 16,016 | 14,939 |
| makes referenced | 75 | 75 |
| models referenced | 431 | 431 |
| first line | `78/1/2010/439374a1456969.jpg` | `78/1/2014/3ac218c0c6c378.jpg` |
| last line | `37/1993/2014/200a4671b58588.jpg` | `37/1993/2014/fcfbe76e8f9650.jpg` |
| SHA-256 | `9966d3e91b7dc735e224e04f5c53f33fbbe48d7c98e15509ca08e897d120ae9b` | `474c4eac44f79ec378b7f755c79b25b14bcc1251450a2766bf03cdfcb7bb2dd0` |

No path appears in both files. Both sides reference all 431 models and 75 of the 163 makes;
the Part-I 431-model subset covers 75 makes. Each line is `make_id/model_id/year/file.jpg`.

## All distributed split files

| File | Lines | Makes | Models | SHA-256 |
|---|---:|---:|---:|---|
| `classification/train.txt` | 16,016 | 75 | 431 | `9966d3e91b7dc735e224e04f5c53f33fbbe48d7c98e15509ca08e897d120ae9b` |
| `classification/test.txt` | 14,939 | 75 | 431 | `474c4eac44f79ec378b7f755c79b25b14bcc1251450a2766bf03cdfcb7bb2dd0` |
| `part/train_part_1.txt` | 1,470 | 75 | 431 | `d7133d783c2a80c87c8126e9b6c84008aba6f728f588e469e8dfdae9921fdb73` |
| `part/train_part_2.txt` | 1,419 | 75 | 431 | `05c08dac8403ff31886fa1328b62ab2e6cb32e3f018f122527fa7fe38eeaae2c` |
| `part/train_part_3.txt` | 1,308 | 75 | 431 | `ed9a30d5c9f6326bc0f6d2db5e65f0a971ee84892a09df53c46e53fc26e13acc` |
| `part/train_part_4.txt` | 1,374 | 75 | 431 | `93322481d98d8936956404d371fb7b2324cd4cab0d5a30139b682427cc09f90e` |
| `part/train_part_5.txt` | 1,359 | 75 | 431 | `b3824f7786c6e2c969b18c5141760cd641816bec11aa0c7a9c0fb04b1bc588f7` |
| `part/train_part_6.txt` | 1,382 | 75 | 431 | `a4b7bdc58e25cb3a1ab9fe8c795125a0fea8dc8b491bec3451b9603e4ff9b5b8` |
| `part/train_part_7.txt` | 1,373 | 75 | 431 | `808ec4d963881e5400450377d9fec71e2e6d507139a812127ea7348083068414` |
| `part/train_part_8.txt` | 1,374 | 75 | 431 | `c17ebfe82564a2b8dd0b41ec140c0a90575787fb4d902f294b6935875ffbf833` |
| `part/test_part_1.txt` | 1,246 | 75 | 431 | `f745695fd7c56b7eabf4470cbcb2b08d0733739c4bdf1080f13ff1edf1941b58` |
| `part/test_part_2.txt` | 1,191 | 75 | 431 | `69fdb35de1e0f0989a60c7f3018a1c9e26623b3ace0b5fdd5fcf4f49470be9db` |
| `part/test_part_3.txt` | 1,100 | 75 | 431 | `48fbf136b2f14b74d5a8c7d590fb6f609e640af6fd049f41a2722ae91cf1cd55` |
| `part/test_part_4.txt` | 1,146 | 75 | 431 | `31f7d28bc85cb8fe9a1b3dc1b002ae6f1d6f60bb885d01c0267618a201f737e7` |
| `part/test_part_5.txt` | 1,130 | 75 | 431 | `24cd6fd33ac06ef956a1b500798eceac645d4a6c98c6b88207146a4e2cc7eabb` |
| `part/test_part_6.txt` | 1,166 | 74 | 430 | `7d563c90c1c39aecc397a5bd8b99bb802e7bc288e54a0809710e8501b268f990` |
| `part/test_part_7.txt` | 1,152 | 74 | 430 | `dc8ea3a2f190c1b14fe3c5212be48a5b4d5176d242e10430bc89ef03d7a84f8c` |
| `part/test_part_8.txt` | 1,159 | 75 | 431 | `c8ced4ca306e967b8706ae11eeef279678450231c406b43fe9127b0f1f9e216e` |
| `verification/verification_train.txt` | 4,454 | 54 | 111 | `0641189066a6a086aee1ea171f371ab8cd2aefa3e2b05a21e8be561e7ffe3e63` |
| `verification/verification_pairs_easy.txt` | 20,000 | 156 | 1137 | `7bb1dafd3d7c131027e2e3ac62bd7e1a3e8010d1c6b28b26fcc2df41eb7ca997` |
| `verification/verification_pairs_medium.txt` | 20,000 | 156 | 1144 | `96489265d965aa34f1d2ae75b98eeba8649ec8d55910f07b668e52b0c1748b28` |
| `verification/verification_pairs_hard.txt` | 20,000 | 156 | 1140 | `9a153f3c415d0c1ab0979d7287a7afa9799119e259962cb866b2bcf8f9ff2f49` |

`part/*` files list car-part crops for the part-based classification experiment and
`verification/*` list image pairs; they are recorded for completeness and are not used
by Track A. Make and model counts for the verification pair files count the first path
of each line only and are informational.
