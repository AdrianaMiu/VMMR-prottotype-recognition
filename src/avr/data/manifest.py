"""Immutable one-row-per-image manifest of the CompCars web-nature release (PROTOCOL.md §7).

Layout: ``image/<make_id>/<model_id>/<year>/<stem>.jpg`` paired with
``label/<make_id>/<model_id>/<year>/<stem>.txt``. One row is built per image file; rows
that cannot enter an experiment keep their metadata but carry ``validity_status =
"excluded"`` and one ``exclusion_reason`` (the first that applies, in ``EXCLUSION_ORDER``).

Two tables are written (ADR-0002, docs/protocols/compcars-licence.md): the public manifest
with opaque ``image_id`` + ``sha256`` and every §7 field except ``relative_path``, and the
private ``image_id -> relative_path`` table under ``data/paths/``. Rows are sorted by
``image_id`` and written with fixed schemas so repeated builds are byte-identical.

``build_manifest`` leaves ``phash``, ``duplicate_group_id`` and ``source_group_id`` null;
``avr.data.dedup`` fills them (M1-8). Columns reserved for later tickets stay null here:
``split_track_a`` (M1-9), ``split_track_b_seed`` (M1-10), ``generation_id`` and
``mapping_version`` (generation mapping).
"""

from __future__ import annotations

import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Literal

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image
from pydantic import BaseModel

from avr.data.labels import parse_label
from avr.data.names import NameTables, load_name_tables
from avr.io import sha256_of_file, write_sha256_sidecar

NAME_MAT = Path("misc") / "make_model_name.mat"
IMAGE_DIR = "image"
LABEL_DIR = "label"
UNKNOWN_YEAR_FOLDER = "unknown"
# Folder names outside this window are typos (the release has one: ``5008``).
MIN_YEAR, MAX_YEAR = 1886, 2030
IMAGE_ID_HEX = 16

ExclusionReason = Literal[
    "unknown_year",
    "invalid_year_folder",
    "empty_make_name",
    "empty_model_name",
    "degenerate_bbox",
    "unreadable_image",
]
EXCLUSION_ORDER: tuple[ExclusionReason, ...] = (
    "unreadable_image",
    "unknown_year",
    "invalid_year_folder",
    "empty_make_name",
    "empty_model_name",
    "degenerate_bbox",
)


class ImageRow(BaseModel):
    """One image with every PROTOCOL.md §7 field; ``relative_path`` is private."""

    image_id: str
    relative_path: str
    sha256: str
    phash: str | None = None
    width: int | None
    height: int | None
    make_id: int
    make_name: str | None
    model_id: int
    model_name: str | None
    year: int | None
    generation_id: str | None = None
    viewpoint: int
    bbox_xyxy: tuple[int, int, int, int]
    split_track_a: str | None = None
    split_track_b_seed: str | None = None
    duplicate_group_id: str | None = None
    source_group_id: str | None = None
    mapping_version: str | None = None
    validity_status: Literal["valid", "excluded"]
    exclusion_reason: ExclusionReason | None


PUBLIC_SCHEMA = pa.schema(
    [
        ("image_id", pa.string()),
        ("sha256", pa.string()),
        ("phash", pa.string()),
        ("width", pa.int32()),
        ("height", pa.int32()),
        ("make_id", pa.int32()),
        ("make_name", pa.string()),
        ("model_id", pa.int32()),
        ("model_name", pa.string()),
        ("year", pa.int32()),
        ("generation_id", pa.string()),
        ("viewpoint", pa.int8()),
        ("bbox_xyxy", pa.list_(pa.int32(), 4)),
        ("split_track_a", pa.string()),
        ("split_track_b_seed", pa.string()),
        ("duplicate_group_id", pa.string()),
        ("source_group_id", pa.string()),
        ("mapping_version", pa.string()),
        ("validity_status", pa.string()),
        ("exclusion_reason", pa.string()),
    ]
)
PRIVATE_SCHEMA = pa.schema([("image_id", pa.string()), ("relative_path", pa.string())])
PUBLIC_COLUMNS = tuple(PUBLIC_SCHEMA.names)
PRIVATE_COLUMNS = tuple(PRIVATE_SCHEMA.names)

MANIFEST_NAME = "compcars_web_images"
PATHS_NAME = "compcars_web_image_paths"


class ManifestOutputs(BaseModel):
    manifest: Path
    sha256: Path
    summary: Path
    paths: Path


def list_images(root: Path) -> list[str]:
    """Relative paths of every image, sorted numerically by make, model, then year and file."""
    image_dir = root / IMAGE_DIR
    paths = [p.relative_to(root).as_posix() for p in image_dir.rglob("*.jpg")]

    def key(rel: str) -> tuple:
        _, make, model, year, name = rel.split("/")
        return (int(make), int(model), year, name)

    return sorted(paths, key=key)


def _hash_and_measure(args: tuple[str, str]) -> tuple[str, int | None, int | None]:
    """Worker: SHA-256 of the file bytes and the pixel size read from the JPEG header."""
    root, relative_path = args
    path = Path(root) / relative_path
    digest = sha256_of_file(path)
    try:
        with Image.open(path) as image:
            width, height = image.size
    except OSError:
        return digest, None, None
    return digest, width, height


def _parse_year(folder: str) -> tuple[int | None, ExclusionReason | None]:
    if folder == UNKNOWN_YEAR_FOLDER:
        return None, "unknown_year"
    if folder.isdigit() and MIN_YEAR <= int(folder) <= MAX_YEAR:
        return int(folder), None
    return None, "invalid_year_folder"


def _row(
    root: Path,
    relative_path: str,
    names: NameTables,
    measured: tuple[str, int | None, int | None],
    ordinal: int,
) -> ImageRow:
    _, make, model, year_folder, file_name = relative_path.split("/")
    make_id, model_id = int(make), int(model)
    label_path = root / LABEL_DIR / make / model / year_folder / f"{Path(file_name).stem}.txt"
    label = parse_label(label_path)
    sha256, width, height = measured
    year, year_reason = _parse_year(year_folder)
    make_name = names.makes[make_id].name if make_id in names.makes else None
    model_name = names.models[model_id].name if model_id in names.models else None

    applicable: set[ExclusionReason] = set()
    if width is None:
        applicable.add("unreadable_image")
    if year_reason is not None:
        applicable.add(year_reason)
    if make_name is None:
        applicable.add("empty_make_name")
    if model_name is None:
        applicable.add("empty_model_name")
    if label.bbox_degenerate:
        applicable.add("degenerate_bbox")
    reason = next((r for r in EXCLUSION_ORDER if r in applicable), None)

    return ImageRow(
        image_id=sha256[:IMAGE_ID_HEX] + (f"-{ordinal}" if ordinal else ""),
        relative_path=relative_path,
        sha256=sha256,
        width=width,
        height=height,
        make_id=make_id,
        make_name=make_name,
        model_id=model_id,
        model_name=model_name,
        year=year,
        viewpoint=label.viewpoint,
        bbox_xyxy=label.bbox_xyxy,
        validity_status="valid" if reason is None else "excluded",
        exclusion_reason=reason,
    )


def build_manifest(root: Path, workers: int | None = None) -> list[ImageRow]:
    """Build every row for the release at ``root``; hashing runs in a process pool."""
    names = load_name_tables(root / NAME_MAT)
    relative_paths = list_images(root)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        measured = list(
            pool.map(_hash_and_measure, ((str(root), p) for p in relative_paths), chunksize=64)
        )
    seen: Counter[str] = Counter()
    rows = []
    for relative_path, item in zip(relative_paths, measured, strict=True):
        prefix = item[0][:IMAGE_ID_HEX]
        rows.append(_row(root, relative_path, names, item, ordinal=seen[prefix]))
        seen[prefix] += 1
    rows.sort(key=lambda r: r.image_id)
    return rows


def summarise(rows: list[ImageRow]) -> dict:
    summary = {
        "row_count": len(rows),
        "valid_count": sum(r.validity_status == "valid" for r in rows),
        "exclusion_reason_counts": dict(
            sorted(Counter(r.exclusion_reason for r in rows if r.exclusion_reason).items())
        ),
        "viewpoint_counts": {
            str(k): v for k, v in sorted(Counter(r.viewpoint for r in rows).items())
        },
        "make_count": len({r.make_id for r in rows}),
        "model_count": len({r.model_id for r in rows}),
    }
    group_sizes = Counter(r.duplicate_group_id for r in rows if r.duplicate_group_id)
    if group_sizes:
        multi = [n for n in group_sizes.values() if n > 1]
        summary["duplicate_group_count"] = len(multi)
        summary["duplicate_image_count"] = sum(multi)
    return summary


def write_manifest(rows: list[ImageRow], manifests_dir: Path, paths_dir: Path) -> ManifestOutputs:
    """Write the public manifest (+ .sha256 and summary JSON) and the private path table."""
    rows = sorted(rows, key=lambda r: r.image_id)
    manifests_dir.mkdir(parents=True, exist_ok=True)
    paths_dir.mkdir(parents=True, exist_ok=True)
    out = ManifestOutputs(
        manifest=manifests_dir / f"{MANIFEST_NAME}.parquet",
        sha256=manifests_dir / f"{MANIFEST_NAME}.sha256",
        summary=manifests_dir / f"{MANIFEST_NAME}.summary.json",
        paths=paths_dir / f"{PATHS_NAME}.parquet",
    )
    public = pa.Table.from_pylist(
        [r.model_dump(exclude={"relative_path"}) for r in rows], schema=PUBLIC_SCHEMA
    )
    private = pa.Table.from_pylist(
        [{"image_id": r.image_id, "relative_path": r.relative_path} for r in rows],
        schema=PRIVATE_SCHEMA,
    )
    pq.write_table(public, out.manifest, compression="zstd")
    pq.write_table(private, out.paths, compression="zstd")
    write_sha256_sidecar(out.manifest)
    out.summary.write_text(json.dumps(summarise(rows), indent=2) + "\n")
    return out
