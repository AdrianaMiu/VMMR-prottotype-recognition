"""CompCars label files ``label/<make_id>/<model_id>/<year>/<stem>.txt`` (PROTOCOL.md §7).

Every file has three lines: the viewpoint code, the bounding-box count (always 1 in the
web-nature release) and ``x1 y1 x2 y2`` in original-image pixels. 77,516 of the 136,726
files end the first line with CRLF, so every line is stripped before parsing.

Viewpoint codes follow the CompCars README: -1 uncertain, 1 front, 2 rear, 3 side,
4 front-side, 5 rear-side. Ten files carry a degenerate box (zero area or a negative
coordinate); those are flagged, not rejected, so the manifest can exclude the row with a
reason instead of dropping the image silently.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

VIEWPOINTS = frozenset({-1, 1, 2, 3, 4, 5})
LABEL_LINES = 3


class ImageLabel(BaseModel):
    viewpoint: int
    bbox_xyxy: tuple[int, int, int, int]
    bbox_degenerate: bool


def _parse_viewpoint(token: str) -> int:
    try:
        viewpoint = int(token)
    except ValueError:
        raise ValueError(f"viewpoint {token!r} is not an integer") from None
    if viewpoint not in VIEWPOINTS:
        raise ValueError(f"viewpoint {viewpoint} is outside {sorted(VIEWPOINTS)}")
    return viewpoint


def is_degenerate(bbox: tuple[int, int, int, int]) -> bool:
    x1, y1, x2, y2 = bbox
    return x2 <= x1 or y2 <= y1 or min(bbox) < 0


def parse_label_text(text: str) -> ImageLabel:
    """Parse the content of one label file; tolerant of CRLF and trailing blank lines."""
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    if len(lines) < LABEL_LINES:
        raise ValueError(f"expected {LABEL_LINES} non-empty lines, got {len(lines)}")
    viewpoint = _parse_viewpoint(lines[0])
    if lines[1] != "1" or len(lines) != LABEL_LINES:
        raise ValueError(f"bbox count {lines[1]!r} with {len(lines) - 2} bbox lines; expected 1")
    coords = lines[2].split()
    if len(coords) != 4:
        raise ValueError(f"bbox line {lines[2]!r} does not hold four integers")
    bbox = (int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3]))
    return ImageLabel(viewpoint=viewpoint, bbox_xyxy=bbox, bbox_degenerate=is_degenerate(bbox))


def parse_label(path: Path) -> ImageLabel:
    return parse_label_text(path.read_bytes().decode("ascii"))
