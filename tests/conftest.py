"""Shared fixture builders: a synthetic CompCars tree small enough to hand-check."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import scipy.io
from PIL import Image


def write_name_mat(path: Path, makes: list[str | None], models: list[str | None]) -> Path:
    """Write a .mat in the layout of the CompCars file: (n,1) cell columns, None = empty cell."""

    def column(names: list[str | None]) -> np.ndarray:
        cells = np.empty((len(names), 1), dtype=object)
        for i, name in enumerate(names):
            cells[i, 0] = np.array([name]) if name is not None else np.array([], dtype="<U1")
        return cells

    path.parent.mkdir(parents=True, exist_ok=True)
    scipy.io.savemat(path, {"make_names": column(makes), "model_names": column(models)})
    return path


# (relative stem, size, fill colour, label text) for the synthetic tree; the label text is
# written byte for byte, so CRLF variants are preserved.
SYNTHETIC_IMAGES = {
    "1/1/2010/aaa": ((8, 8), (255, 0, 0), "4\n1\n1 1 6 6\n"),
    "1/1/unknown/bbb": ((8, 8), (0, 255, 0), "1\r\n1\n0 0 7 7\n"),
    "1/1/5008/ccc": ((10, 6), (0, 0, 255), "2\n1\n0 0 9 5\n"),
    "2/3/2014/ddd": ((8, 8), (10, 20, 30), "-1\n1\n2 2 2 2\n"),
    "2/2/2012/eee": ((6, 10), (40, 50, 60), "5\r\n1\n1 2 5 9\n"),
}


@pytest.fixture
def synthetic_compcars(tmp_path: Path) -> Path:
    """Two makes, three models, five 8-pixel-ish JPEGs covering every exclusion reason."""
    root = tmp_path / "compcar"
    write_name_mat(
        root / "misc" / "make_model_name.mat",
        makes=["ABT", "Lamorghini "],
        models=["Audi A3 hatchback", None, " Hector"],
    )
    for stem, (size, colour, label) in SYNTHETIC_IMAGES.items():
        image_path = root / "image" / f"{stem}.jpg"
        label_path = root / "label" / f"{stem}.txt"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", size, colour).save(image_path, format="JPEG")
        label_path.write_bytes(label.encode("ascii"))
    return root
