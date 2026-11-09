"""Gradio demo for manual testing of single images against the recognition service.

Run with ``uv run python -m avr.service.demo`` (needs the ``demo`` extra). The default mode
calls :class:`avr.service.pipeline.RecognitionPipeline` in-process through the same
:class:`avr.service.app.ServiceResources` the HTTP API uses, so the page and ``POST
/v1/recognize`` cannot drift; ``--api-url`` points the page at a running API instead.

One upload is run through every selected profile, so encoders and variants are compared
on the same vehicles. The page shows the annotated input (boxes numbered by ``detection_id``,
translucent masks), a table of the top-k labels per vehicle and profile, the per-stage latency
and the profile and index versions. Cosine similarity is a similarity, never a probability.
Example images are CompCars query images referenced by ``image_id`` and resolved through the
private path table at runtime; no image is copied into the repository.
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from avr.service.schemas import STAGES, ProfilesResponse, RecognizeResponse, Vehicle

log = logging.getLogger("avr.service.demo")

REPO_ROOT = Path(__file__).resolve().parents[3]
EXAMPLES_JSON = REPO_ROOT / "tests" / "fixtures" / "demo" / "examples.json"
PATHS_TABLE = Path("data") / "paths" / "compcars_web_image_paths.parquet"
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
NOT_A_PROBABILITY = "Cosine similarity is not a probability."
TABLE_HEADER = ("Rank", "Make", "Model", "Year / class", "Cosine similarity", "Decision")
MASK_ALPHA = 0.4
PALETTE: tuple[tuple[int, int, int], ...] = (
    (230, 25, 75),
    (60, 180, 75),
    (0, 130, 200),
    (245, 130, 48),
    (145, 30, 180),
    (70, 240, 240),
    (240, 50, 230),
    (210, 245, 60),
    (0, 128, 128),
    (170, 110, 40),
)


def colour_for(index: int) -> tuple[int, int, int]:
    return PALETTE[index % len(PALETTE)]


# --- annotation -------------------------------------------------------------------------


@dataclass(frozen=True)
class Detection:
    """What the annotator needs of one vehicle: its id, box (pixels) and optional mask."""

    detection_id: int
    bbox_xyxy: tuple[float, float, float, float]
    mask: np.ndarray | None = None


def decode_mask(mask_png_base64: str) -> np.ndarray:
    with Image.open(io.BytesIO(base64.b64decode(mask_png_base64))) as im:
        return np.asarray(im.convert("L")) > 127


def detections_from(vehicles: Sequence[Vehicle]) -> list[Detection]:
    return [
        Detection(
            v.detection_id,
            (v.bbox_xyxy[0], v.bbox_xyxy[1], v.bbox_xyxy[2], v.bbox_xyxy[3]),
            decode_mask(v.mask_png_base64) if v.mask_png_base64 else None,
        )
        for v in vehicles
    ]


def annotate(
    image: Image.Image,
    detections: Sequence[Detection],
    show_masks: bool = True,
    show_boxes: bool = True,
    alpha: float = MASK_ALPHA,
) -> Image.Image:
    """Return a copy of ``image`` with translucent masks and numbered boxes, one colour each.

    The mask pixels become ``round((1 - alpha) * pixel + alpha * colour)``; boxes are a
    3 px outline with a filled tag carrying the ``detection_id`` above the top-left corner.
    """
    base = image.convert("RGB")
    if show_masks:
        pixels = np.asarray(base, dtype=np.float64).copy()
        for det in detections:
            if det.mask is None or det.mask.shape != pixels.shape[:2]:
                continue
            colour = np.asarray(colour_for(det.detection_id), dtype=np.float64)
            region = det.mask.astype(bool)
            pixels[region] = (1.0 - alpha) * pixels[region] + alpha * colour
        base = Image.fromarray(np.rint(pixels).astype(np.uint8), "RGB")
    if not show_boxes:
        return base
    draw = ImageDraw.Draw(base)
    font = ImageFont.load_default(size=max(12, min(base.size) // 30))
    for det in detections:
        colour = colour_for(det.detection_id)
        x1, y1, x2, y2 = (int(round(v)) for v in det.bbox_xyxy)
        draw.rectangle((x1, y1, x2 - 1, y2 - 1), outline=colour, width=3)
        text = str(det.detection_id)
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        tag_w, tag_h = right - left + 8, bottom - top + 8
        ty = y1 - tag_h if y1 - tag_h >= 0 else y1
        draw.rectangle((x1, ty, x1 + tag_w - 1, ty + tag_h - 1), fill=colour)
        draw.text((x1 + 4 - left, ty + 4 - top), text, fill=(255, 255, 255), font=font)
    return base


# --- recognition seam -------------------------------------------------------------------


@dataclass
class Recognition:
    """One profile's answer for one image, whichever mode produced it."""

    profile_id: str
    index_version: str
    versions: dict[str, str | None]
    vehicles: list[Vehicle]
    latency_ms: dict[str, float] = field(default_factory=dict)


class Recognizer(Protocol):
    def profiles(self) -> ProfilesResponse: ...

    def recognize(
        self, image: Image.Image, profile_id: str, top_k: int, return_masks: bool
    ) -> Recognition: ...


class InProcessRecognizer:
    """Runs the service pipeline in this process through a loaded ``ServiceResources``."""

    def __init__(self, resources: Any) -> None:
        self.resources = resources

    def profiles(self) -> ProfilesResponse:
        r = self.resources
        return ProfilesResponse(
            default_profile_id=r.profiles.default_profile_id,
            profiles=[r.profile_info(p) for p in r.profiles],
        )

    def recognize(
        self, image: Image.Image, profile_id: str, top_k: int, return_masks: bool
    ) -> Recognition:
        r = self.resources
        profile = r.profiles[profile_id]
        result = r.pipeline_for(profile).recognize(image, top_k=top_k, return_masks=return_masks)
        return Recognition(
            profile_id=profile_id,
            index_version=profile.index_version,
            versions=r.versions(profile),
            vehicles=result.vehicles,
            latency_ms=dict(result.latency_ms),
        )


class HttpRecognizer:
    """Second mode: the same page against a running service (``GET /v1/profiles``)."""

    def __init__(self, base_url: str, timeout_s: float = 120.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    def profiles(self) -> ProfilesResponse:
        import httpx

        response = httpx.get(f"{self.base_url}/v1/profiles", timeout=self.timeout_s)
        response.raise_for_status()
        return ProfilesResponse.model_validate(response.json())

    def recognize(
        self, image: Image.Image, profile_id: str, top_k: int, return_masks: bool
    ) -> Recognition:
        import httpx

        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="PNG")
        response = httpx.post(
            f"{self.base_url}/v1/recognize",
            files={"image": ("upload.png", buffer.getvalue(), "image/png")},
            data={
                "profile_id": profile_id,
                "top_k": str(top_k),
                "return_masks": str(return_masks).lower(),
            },
            timeout=self.timeout_s,
        )
        response.raise_for_status()
        body = RecognizeResponse.model_validate(response.json())
        latency = body.vehicles[0].latency_ms.model_dump() if body.vehicles else {}
        return Recognition(
            profile_id=body.profile_id,
            index_version=body.index_version,
            versions=body.versions,
            vehicles=body.vehicles,
            latency_ms=latency,
        )


# --- formatting -------------------------------------------------------------------------


def prediction_rows(vehicle: Vehicle) -> list[list[str]]:
    """Top-k rows ``(rank, make, model, year/class, similarity, decision)`` for one vehicle.

    The decision belongs to the top-1 hypothesis, so it is shown on rank 1 only. The
    year/class cell is the provisional generation when there is one, else the class id.
    """
    rows = []
    for p in vehicle.predictions:
        rows.append(
            [
                str(p.rank),
                p.make or "-",
                p.model or "-",
                p.generation or p.class_id,
                f"{p.cosine_similarity:.4f}",
                vehicle.decision if p.rank == 1 else "",
            ]
        )
    return rows


def latency_rows(recognitions: Sequence[Recognition]) -> list[list[str | float]]:
    rows: list[list[str | float]] = []
    for rec in recognitions:
        values = [round(float(rec.latency_ms.get(stage, 0.0)), 1) for stage in STAGES]
        rows.append([rec.profile_id, *values, round(sum(values), 1)])
    return rows


LATENCY_HEADER = ("Profile", *(f"{s} (ms)" for s in STAGES), "total (ms)")


def versions_markdown(recognitions: Sequence[Recognition]) -> str:
    lines = []
    for rec in recognitions:
        v = rec.versions
        lines.append(
            f"- **{rec.profile_id}**: index `{rec.index_version[:16]}`, "
            f"encoder `{v.get('encoder')}` @ `{v.get('encoder_revision')}`, "
            f"preprocessing `{v.get('preprocessing_id')}`, method `{v.get('prototype_method')}`, "
            f"segmenter `{v.get('segmenter')}`, backend `{v.get('search_backend')}`, "
            f"avr `{v.get('avr')}`"
        )
    lines.append("")
    lines.append(f"*{NOT_A_PROBABILITY}* Decisions compare the top-1 similarity to a profile cut.")
    return "\n".join(lines)


CSS = """
.avr-vehicle { border: 1px solid var(--border-color-primary); border-radius: 8px;
  margin: 0 0 12px; padding: 8px 12px; }
.avr-vehicle summary { font-weight: 600; cursor: pointer; }
.avr-cards { display: flex; flex-wrap: wrap; gap: 12px; margin-top: 8px; }
.avr-card { flex: 1 1 360px; min-width: 0; overflow-x: auto; }
.avr-card h4 { margin: 4px 0; font-size: 0.9em; }
.avr-card table { border-collapse: collapse; width: 100%; font-size: 0.85em; }
.avr-card th, .avr-card td { border: 1px solid var(--border-color-primary);
  padding: 2px 6px; text-align: left; }
.avr-swatch { display: inline-block; width: 0.9em; height: 0.9em; margin-right: 6px;
  border-radius: 2px; vertical-align: middle; }
"""


def _table_html(vehicle: Vehicle) -> str:
    head = "".join(f"<th>{html.escape(h)}</th>" for h in TABLE_HEADER)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(c)}</td>" for c in row) + "</tr>"
        for row in prediction_rows(vehicle)
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def vehicles_html(recognitions: Sequence[Recognition]) -> str:
    """One collapsible block per vehicle, profiles side by side inside it."""
    if not recognitions or not any(r.vehicles for r in recognitions):
        return "<p>No vehicle detected.</p>"
    ids = sorted({v.detection_id for r in recognitions for v in r.vehicles})
    blocks = []
    for det_id in ids:
        first = next(v for r in recognitions for v in r.vehicles if v.detection_id == det_id)
        r_, g_, b_ = colour_for(det_id)
        x1, y1, x2, y2 = (round(c) for c in first.bbox_xyxy)
        title = (
            f'<span class="avr-swatch" style="background:rgb({r_},{g_},{b_})"></span>'
            f"Vehicle {det_id} &middot; detection score {first.detection_score:.2f} "
            f"&middot; box ({x1}, {y1}, {x2}, {y2})"
        )
        cards = []
        for rec in recognitions:
            vehicle = next((v for v in rec.vehicles if v.detection_id == det_id), None)
            inner = "<p>no result</p>" if vehicle is None else _table_html(vehicle)
            cards.append(
                f'<div class="avr-card"><h4>{html.escape(rec.profile_id)}</h4>{inner}</div>'
            )
        blocks.append(
            f'<details class="avr-vehicle" open><summary>{title}</summary>'
            f'<div class="avr-cards">{"".join(cards)}</div></details>'
        )
    return "".join(blocks)


# --- one request ------------------------------------------------------------------------


@dataclass
class DemoReport:
    annotated: Image.Image
    vehicles_html: str
    latency: list[list[str | float]]
    versions_md: str
    recognitions: list[Recognition]


class SessionLog:
    """Opt-in JSON-lines log: timestamp, profile, vehicle count and the top-1 per vehicle."""

    def __init__(self, path: Path | None) -> None:
        self.path = None if path is None else Path(path)

    def write(self, rec: Recognition) -> None:
        if self.path is None:
            return
        top1 = [
            {
                "detection_id": v.detection_id,
                "class_id": v.predictions[0].class_id,
                "make": v.predictions[0].make,
                "model": v.predictions[0].model,
                "cosine_similarity": v.predictions[0].cosine_similarity,
                "decision": v.decision,
            }
            for v in rec.vehicles
            if v.predictions
        ]
        line = {
            "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
            "profile": rec.profile_id,
            "vehicles": len(rec.vehicles),
            "top1": top1,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(line) + "\n")


def run_demo(
    recognizer: Recognizer,
    image: Image.Image,
    profile_ids: Sequence[str],
    top_k: int,
    show_masks: bool,
    show_boxes: bool,
    session_log: SessionLog | None = None,
) -> DemoReport:
    """Run ``image`` through each selected profile and assemble everything the page shows."""
    if not profile_ids:
        raise ValueError("select at least one profile")
    image = image.convert("RGB")
    recognitions = [
        recognizer.recognize(image, pid, int(top_k), return_masks=show_masks and i == 0)
        for i, pid in enumerate(profile_ids)
    ]
    # every profile shares the one detector, so the first profile's boxes and masks stand for all
    annotated = annotate(image, detections_from(recognitions[0].vehicles), show_masks, show_boxes)
    if session_log is not None:
        for rec in recognitions:
            session_log.write(rec)
    return DemoReport(
        annotated=annotated,
        vehicles_html=vehicles_html(recognitions),
        latency=latency_rows(recognitions),
        versions_md=versions_markdown(recognitions),
        recognitions=recognitions,
    )


# --- examples ---------------------------------------------------------------------------


def find_dataset_root() -> Path | None:
    """The CompCars release: ``COMPCARS_ROOT`` or ``../compcar`` of the main checkout."""
    candidates = []
    if os.environ.get("COMPCARS_ROOT"):
        candidates.append(Path(os.environ["COMPCARS_ROOT"]))
    for base in (REPO_ROOT, *REPO_ROOT.parents[:2]):
        candidates.append(base / ".." / "compcar")
        candidates.append(base / "compcar")
    for root in candidates:
        if (root / "image").is_dir():
            return root.resolve()
    return None


def find_paths_table(paths_table: Path | None = None) -> Path | None:
    candidates = [paths_table] if paths_table else []
    candidates += [REPO_ROOT / PATHS_TABLE, *(p / PATHS_TABLE for p in REPO_ROOT.parents[:2])]
    return next((c for c in candidates if c is not None and Path(c).is_file()), None)


def compcars_examples(
    examples_json: Path = EXAMPLES_JSON,
    paths_table: Path | None = None,
    dataset_root: Path | None = None,
) -> list[Path]:
    """Resolve the committed ``image_id`` list to local files; missing pieces give ``[]``."""
    if not Path(examples_json).is_file():
        return []
    table, root = find_paths_table(paths_table), dataset_root or find_dataset_root()
    if table is None or root is None:
        log.warning("CompCars path table or release not found; no CompCars examples")
        return []
    import pandas as pd

    ids = json.loads(Path(examples_json).read_text())["image_ids"]
    relative = pd.read_parquet(table).set_index("image_id")["relative_path"]
    found = [root / relative[i] for i in ids if i in relative.index]
    return [p for p in found if p.is_file()]


def directory_examples(examples_dir: Path | None) -> list[Path]:
    if examples_dir is None or not Path(examples_dir).is_dir():
        return []
    return sorted(p for p in Path(examples_dir).iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


# --- the page ---------------------------------------------------------------------------


def profile_label(info: Any) -> str:
    return f"{info.profile_id}  (index {info.index_version[:8]}, {info.prototype_count} classes)"


def build_demo(
    recognizer: Recognizer,
    examples: Sequence[Path] = (),
    session_log: SessionLog | None = None,
):
    import gradio as gr

    listing = recognizer.profiles()
    choices = [(profile_label(p), p.profile_id) for p in listing.profiles]

    def handle(image, profile_ids, top_k, show_masks, show_boxes):
        if image is None:
            raise gr.Error("Upload an image first.")
        if not profile_ids:
            raise gr.Error("Select at least one profile.")
        report = run_demo(
            recognizer,
            image,
            profile_ids,
            int(top_k),
            bool(show_masks),
            bool(show_boxes),
            session_log,
        )
        return report.annotated, report.vehicles_html, report.latency, report.versions_md

    with gr.Blocks(title="AVR recognition demo", css=CSS) as demo:
        gr.Markdown(
            "## AVR recognition demo\nUpload a scene with one or several cars, pick one or "
            f"more profiles and compare their top-k labels per vehicle. {NOT_A_PROBABILITY}"
        )
        with gr.Row():
            with gr.Column(scale=1):
                image = gr.Image(type="pil", label="Input image", sources=["upload"])
                profiles = gr.CheckboxGroup(
                    choices=choices, value=[listing.default_profile_id], label="Profiles"
                )
                top_k = gr.Slider(1, 20, value=5, step=1, label="top_k")
                show_masks = gr.Checkbox(value=True, label="Show masks")
                show_boxes = gr.Checkbox(value=True, label="Show boxes")
                run = gr.Button("Recognise", variant="primary")
            with gr.Column(scale=2):
                annotated = gr.Image(type="pil", label="Detections", interactive=False)
                vehicles = gr.HTML(label="Vehicles")
                latency = gr.Dataframe(
                    headers=list(LATENCY_HEADER),
                    label="Per-stage latency",
                    interactive=False,
                    wrap=True,
                )
                versions = gr.Markdown()
        inputs = [image, profiles, top_k, show_masks, show_boxes]
        outputs = [annotated, vehicles, latency, versions]
        run.click(handle, inputs, outputs, api_name="recognize")
        if examples:
            gr.Examples(
                examples=[[str(p)] for p in examples],
                inputs=[image],
                label="Examples",
                examples_per_page=12,
            )
    demo.queue(default_concurrency_limit=1)
    return demo


# --- entry point ------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> None:  # pragma: no cover - CLI wiring
    parser = argparse.ArgumentParser(description="AVR recognition demo (Gradio)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--examples-dir", type=Path, help="extra photos shown as examples")
    parser.add_argument("--log-session", type=Path, help="append JSON-lines session records")
    parser.add_argument("--api-url", help="use a running service instead of in-process models")
    args = parser.parse_args(argv)
    os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
    logging.basicConfig(level=logging.INFO)

    recognizer: Recognizer
    resources = None
    if args.api_url:
        recognizer = HttpRecognizer(args.api_url)
    else:
        from avr.service.app import ServiceResources
        from avr.service.settings import ServiceSettings

        resources = ServiceResources(ServiceSettings())
        resources.load()
        recognizer = InProcessRecognizer(resources)

    examples = [*compcars_examples(), *directory_examples(args.examples_dir)]
    allowed = {p.parent for p in examples}
    demo = build_demo(recognizer, examples, SessionLog(args.log_session))
    try:
        demo.launch(
            server_name=args.host, server_port=args.port, allowed_paths=[str(p) for p in allowed]
        )
    finally:
        if resources is not None:
            resources.close()


if __name__ == "__main__":  # pragma: no cover
    main()
