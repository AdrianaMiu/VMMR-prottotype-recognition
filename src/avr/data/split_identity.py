"""Identity of CompCars split files (PROTOCOL.md §5 Track A).

A split file lists image paths as ``make_id/model_id/year/file.jpg`` (one per line, or
two paths plus a label for verification pairs). Its identity is the content hash plus
the counts a reader needs to tell which official setting it is (CVPR 2015 half/half over
30,955 images vs arXiv v2 70/30 over 52,083); see
``docs/research/generation-recognition-gap.md`` §4(c) item 7.

Image paths are never published (ADR-0002); file boundaries are pinned by hashing the
first and last three lines instead.
"""

from __future__ import annotations

import hashlib
from datetime import date
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

BOUNDARY_LINES = 3


class SplitFileIdentity(BaseModel):
    """What makes a split file *this* split file, without exposing any image path."""

    sha256: str
    line_count: int
    make_count: int
    model_count: int
    head_sha256: str
    tail_sha256: str


class SplitIdentityRecord(BaseModel):
    """Published identity of every distributed split file for one dataset release."""

    dataset: Literal["compcars_web"]
    setting: Literal["cvpr2015_half_half", "arxiv_v2_70_30"]
    label: str
    recorded_at: date
    files: dict[str, SplitFileIdentity]


OFFICIAL_SPLIT_FILES = (
    "train_test_split/classification/train.txt",
    "train_test_split/classification/test.txt",
)
OTHER_SPLIT_FILES = (
    *(f"train_test_split/part/{s}_part_{i}.txt" for s in ("train", "test") for i in range(1, 9)),
    "train_test_split/verification/verification_train.txt",
    *(
        f"train_test_split/verification/verification_pairs_{d}.txt"
        for d in ("easy", "medium", "hard")
    ),
)

TRACK_A_LABEL = (
    "official CVPR 2015 half/half classification split (30,955 images, 431 models); "
    "release version unverified; hashes recorded"
)


def _sha256_of_lines(lines: list[str]) -> str:
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def describe_split_file(path: Path) -> SplitFileIdentity:
    """Hash ``path`` and count the makes and models referenced by every path token in it."""
    raw = path.read_bytes()
    lines = [line for line in raw.decode("ascii").splitlines() if line]
    if not lines:
        raise ValueError(f"{path} is empty; a split file must list at least one image")
    image_paths = [tok for line in lines for tok in line.split() if "/" in tok]
    makes = {p.split("/")[0] for p in image_paths}
    models = {"/".join(p.split("/")[:2]) for p in image_paths}
    return SplitFileIdentity(
        sha256=hashlib.sha256(raw).hexdigest(),
        line_count=len(lines),
        make_count=len(makes),
        model_count=len(models),
        head_sha256=_sha256_of_lines(lines[:BOUNDARY_LINES]),
        tail_sha256=_sha256_of_lines(lines[-BOUNDARY_LINES:]),
    )


def record_split_identity(root: Path, recorded_at: date) -> SplitIdentityRecord:
    """Describe the official classification split and every other distributed split file."""
    return SplitIdentityRecord(
        dataset="compcars_web",
        setting="cvpr2015_half_half",
        label=TRACK_A_LABEL,
        recorded_at=recorded_at,
        files={
            rel: describe_split_file(root / rel)
            for rel in (*OFFICIAL_SPLIT_FILES, *OTHER_SPLIT_FILES)
        },
    )
