"""Run one image through a registry checkpoint and measure what PROTOCOL.md §9 asks for.

This module needs the ``gpu`` extra (torch, transformers, open_clip); the registry schema
in :mod:`avr.encoders.registry` does not, so unit tests never import it. Each loader
returns the checkpoint's documented pooled image representation (§9 extraction contract
step 3): DINOv3 ``pooler_output`` (CLS token after the final norm), SigLIP2
``pooler_output`` (attention-pool head of the vision tower), CLIP ``get_image_features``
(visual projection) and open_clip ``encode_image`` (image projection).
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from pydantic import BaseModel

from avr.encoders.registry import EncoderSpec, TorchDtype

TORCH_DTYPES: dict[TorchDtype, torch.dtype] = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


class VerificationResult(BaseModel):
    """What one forward pass established about a checkpoint."""

    name: str
    revision: str
    checkpoint_path: Path
    checkpoint_sha256: str
    parameter_count: int
    processor: str
    output_shape: list[int]
    device: str
    dtype: TorchDtype
    verified_at: datetime


def sha256_of_file(path: Path, chunk_size: int = 1 << 22) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _count_parameters(*modules: torch.nn.Module) -> int:
    return sum(p.numel() for module in modules for p in module.parameters())


# Each loader returns (image_encoder_fn, image_parameter_count, processor_name). The encoder
# function maps a sequence of PIL images to the pooled tensor ``[n, native_dim]``. The
# parameter count covers only the modules the image path executes: text towers are never
# run in this project, so they are not counted.
Encode = Callable[[Sequence[Image.Image]], torch.Tensor]
Loader = Callable[[EncoderSpec, str, str, torch.dtype], tuple[Encode, int, str]]


def _load_dinov3(spec, revision, device, dtype):
    from transformers import AutoImageProcessor, AutoModel

    processor = AutoImageProcessor.from_pretrained(spec.model_id, revision=revision)
    model = AutoModel.from_pretrained(spec.model_id, revision=revision, dtype=dtype)
    model = model.to(device).eval()

    def encode(images: Sequence[Image.Image]) -> torch.Tensor:
        inputs = processor(images=list(images), return_tensors="pt").to(device, dtype)
        return model(**inputs).pooler_output

    return encode, _count_parameters(model), type(processor).__name__


def _load_siglip2(spec, revision, device, dtype):
    from transformers import AutoImageProcessor, Siglip2VisionModel

    settings = spec.preprocessing.model_dump()
    processor = AutoImageProcessor.from_pretrained(
        spec.model_id, revision=revision, max_num_patches=settings["max_num_patches"]
    )
    model = Siglip2VisionModel.from_pretrained(spec.model_id, revision=revision, dtype=dtype)
    model = model.to(device).eval()

    def encode(images: Sequence[Image.Image]) -> torch.Tensor:
        inputs = processor(images=list(images), return_tensors="pt").to(device)
        inputs["pixel_values"] = inputs["pixel_values"].to(dtype)
        return model(**inputs).pooler_output

    return encode, _count_parameters(model), type(processor).__name__


def _load_clip(spec, revision, device, dtype):
    from transformers import AutoImageProcessor, CLIPModel

    processor = AutoImageProcessor.from_pretrained(spec.model_id, revision=revision)
    model = CLIPModel.from_pretrained(spec.model_id, revision=revision, dtype=dtype)
    model = model.to(device).eval()

    def encode(images: Sequence[Image.Image]) -> torch.Tensor:
        inputs = processor(images=list(images), return_tensors="pt").to(device, dtype)
        features = model.get_image_features(**inputs)
        # transformers >= 5 returns the vision output with the projection written into
        # pooler_output; earlier versions return the projected tensor directly.
        return getattr(features, "pooler_output", features)

    parameters = _count_parameters(model.vision_model, model.visual_projection)
    return encode, parameters, type(processor).__name__


def _load_open_clip(spec, revision, device, dtype):
    import open_clip

    model_name = spec.preprocessing.model_dump()["open_clip_model"]
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=revision, device=device, precision=str(dtype).removeprefix("torch.")
    )
    model.eval()

    def encode(images: Sequence[Image.Image]) -> torch.Tensor:
        pixels = torch.stack([preprocess(image) for image in images]).to(device, dtype)
        return model.encode_image(pixels)

    return encode, _count_parameters(model.visual), f"open_clip:{model_name}"


def select_loader(spec: EncoderSpec) -> Loader:
    if spec.source == "open_clip":
        return _load_open_clip
    repo = spec.model_id.split("/")[-1]
    if repo.startswith("dinov3"):
        return _load_dinov3
    if repo.startswith("siglip2"):
        return _load_siglip2
    if repo.startswith("clip"):
        return _load_clip
    raise ValueError(f"{spec.name}: no loader for {spec.model_id}")


def resolve_checkpoint(spec: EncoderSpec) -> tuple[Path, str]:
    """Download (or find in the cache) the main weight file; return its path and commit."""
    revision = None if spec.source == "open_clip" else spec.revision
    # The returned path is ``.../snapshots/<commit>/<file>``; it must not be resolved, as the
    # file is a symlink into the content-addressed ``blobs/`` directory.
    path = Path(hf_hub_download(spec.model_id, spec.checkpoint_file, revision=revision))
    return path, path.parent.name


def verify_entry(
    spec: EncoderSpec, image: Image.Image, device: str = "cpu", dtype: TorchDtype = "float32"
) -> VerificationResult:
    """Load ``spec``'s checkpoint, encode ``image`` once and report the measured facts."""
    checkpoint_path, commit = resolve_checkpoint(spec)
    revision = spec.revision if spec.source == "open_clip" else commit
    torch_dtype = TORCH_DTYPES[dtype]
    encode, parameter_count, processor = select_loader(spec)(spec, revision, device, torch_dtype)
    with torch.inference_mode():
        output = encode([image.convert("RGB")])
    return VerificationResult(
        name=spec.name,
        revision=revision,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=sha256_of_file(checkpoint_path),
        parameter_count=parameter_count,
        processor=processor,
        output_shape=list(output.shape),
        device=device,
        dtype=dtype,
        verified_at=datetime.now(UTC).replace(microsecond=0),
    )


def apply_result(spec: EncoderSpec, result: VerificationResult) -> EncoderSpec:
    """Return ``spec`` with the measured fields filled in; ``native_dim`` is re-validated."""
    entry = spec.model_dump()
    entry.update(
        revision=result.revision,
        parameter_count=result.parameter_count,
        preprocessing={**spec.preprocessing.model_dump(), "processor": result.processor},
        checkpoint_sha256=result.checkpoint_sha256,
        verified_at=result.verified_at,
        verified_dtype=result.dtype,
        verified_output_shape=result.output_shape,
    )
    return EncoderSpec.model_validate(entry)
