"""Identity of CompCars split files (PROTOCOL.md §5 Track A).

A split file lists one image path per line as ``make_id/model_id/year/file.jpg``.
Its identity is the content hash plus the counts a reader needs to know which
official setting it is (CVPR 2015 half/half over 30,955 images vs arXiv v2 70/30
over 52,083), see ``docs/research/generation-recognition-gap.md`` §4(c) item 7.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from pydantic import BaseModel


class SplitFileIdentity(BaseModel):
    """What makes a split file *this* split file."""

    sha256: str
    line_count: int
    make_count: int
    model_count: int
    first_line: str
    last_line: str


def describe_split_file(path: Path) -> SplitFileIdentity:
    """Hash ``path`` and count the makes and models its lines reference."""
    raw = path.read_bytes()
    lines = [line for line in raw.decode("ascii").splitlines() if line]
    makes = {line.split("/")[0] for line in lines}
    models = {"/".join(line.split("/")[:2]) for line in lines}
    return SplitFileIdentity(
        sha256=hashlib.sha256(raw).hexdigest(),
        line_count=len(lines),
        make_count=len(makes),
        model_count=len(models),
        first_line=lines[0],
        last_line=lines[-1],
    )


class SplitIdentityRecord(BaseModel):
    """Published identity of every distributed split file for one dataset release."""

    dataset: str
    setting: str
    label: str
    recorded_at: str
    files: dict[str, SplitFileIdentity]


SPLIT_FILES = (
    "train_test_split/classification/train.txt",
    "train_test_split/classification/test.txt",
    *(f"train_test_split/part/{s}_part_{i}.txt" for s in ("train", "test") for i in range(1, 9)),
    "train_test_split/verification/verification_train.txt",
    *(f"train_test_split/verification/verification_pairs_{d}.txt" for d in ("easy", "medium", "hard")),
)


def record_split_identity(root: Path, recorded_at: str) -> SplitIdentityRecord:
    """Describe every distributed CompCars split file under ``root``."""
    return SplitIdentityRecord(
        dataset="compcars_web",
        setting="cvpr2015_half_half",
        label=(
            "official CVPR 2015 half/half classification split "
            "(30,955 images, 431 models); release version unverified; hashes recorded"
        ),
        recorded_at=recorded_at,
        files={rel: describe_split_file(root / rel) for rel in SPLIT_FILES},
    )
