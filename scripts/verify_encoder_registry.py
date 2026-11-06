"""Verify every configs/encoders entry with one real CompCars image (PROTOCOL.md §9).

For each entry the checkpoint is downloaded (or taken from the Hugging Face cache), one
image is run through the official preprocessing, and the measured pooled output shape,
parameter count, resolved revision, weight-file SHA-256 and processor are written back
into the YAML. A measured last dimension that differs from ``native_dim`` is reported and
left unwritten rather than silently substituted.

Usage: uv run python scripts/verify_encoder_registry.py [--only NAME ...] [--device cpu]
       [--dtype float32] [--image PATH]
Needs the ``gpu`` extra. The dataset root comes from configs/datasets (COMPCARS_ROOT
overrides it); any image under it will do.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image
from pydantic import ValidationError

from avr.config import CONFIG_DIR, load_config
from avr.encoders.registry import ENCODERS_DIR, load_registry, write_entry
from avr.encoders.verify import apply_result, verify_entry

REPO_ROOT = CONFIG_DIR.parent


def first_image(root: Path) -> Path:
    """The first JPEG under ``root/image`` in sorted directory order; any image will do."""

    def search(directory: Path) -> Path | None:
        entries = sorted(directory.iterdir())
        for entry in entries:
            if entry.is_file() and entry.suffix == ".jpg":
                return entry
        for entry in entries:
            if entry.is_dir() and (found := search(entry)) is not None:
                return found
        return None

    found = search(root / "image")
    if found is None:
        raise FileNotFoundError(f"no .jpg under {root / 'image'}")
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", nargs="*", default=None, help="entry names to verify")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--image", type=Path, default=None)
    args = parser.parse_args(argv)

    image_path = args.image or first_image((REPO_ROOT / load_config().dataset.root).resolve())
    image = Image.open(image_path)
    print(f"image: {image_path.name} {image.size} device={args.device} dtype={args.dtype}")

    registry = load_registry(ENCODERS_DIR)
    names = args.only or list(registry)
    rows: list[tuple[str, ...]] = []
    failures: list[str] = []
    for name in names:
        spec = registry[name]
        try:
            result = verify_entry(spec, image, device=args.device, dtype=args.dtype)
            verified = apply_result(spec, result)
        except ValidationError as error:
            failures.append(f"{name}: measured shape disagrees with the registry: {error}")
            continue
        except Exception as error:  # noqa: BLE001 - every failure must be reported, not raised
            failures.append(f"{name}: {type(error).__name__}: {error}")
            continue
        write_entry(ENCODERS_DIR / f"{name}.yaml", verified)
        rows.append(
            (
                name,
                "x".join(map(str, result.output_shape)),
                spec.pooling,
                result.revision[:12],
                result.checkpoint_sha256[:12],
                f"{result.parameter_count:,}",
            )
        )

    header = ("entry", "shape", "pooling", "revision", "sha256", "params")
    widths = [max(len(r[i]) for r in [header, *rows]) for i in range(len(header))]
    for row in (header, *rows):
        print("  ".join(cell.ljust(w) for cell, w in zip(row, widths, strict=True)))
    for failure in failures:
        print(f"FAILED {failure}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
