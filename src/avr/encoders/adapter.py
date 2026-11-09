"""Frozen encoder adapter: the PROTOCOL.md §9 embedding extraction contract, batched.

:class:`FrozenEncoder` loads a registry checkpoint through the same loader as the
verifier (:mod:`avr.encoders.verify`), keeps it in eval mode, and :meth:`FrozenEncoder.embed`
runs the six contract steps on a batch of PIL images: official preprocessing, inference
mode, documented pooled output, float32 cast, L2 normalisation; storing with the encoder and
preprocessing identity is the job of :class:`avr.encoders.store.EmbeddingStore`, fed by
:meth:`FrozenEncoder.identity`. Compute dtype is the registry ``dtype`` (fp16 for every
entry) on CUDA and float32 on CPU, where fp16 matmuls are not supported.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from avr.encoders.registry import EncoderSpec, TorchDtype
from avr.encoders.store import StoreIdentity
from avr.encoders.verify import (
    TORCH_DTYPES,
    Encode,
    resolve_checkpoint,
    select_loader,
    sha256_of_file,
)


class FrozenEncoder:
    def __init__(
        self,
        spec: EncoderSpec,
        encode: Encode,
        revision: str,
        checkpoint_path: Path,
        checkpoint_sha256: str,
        device: str,
        dtype: TorchDtype,
    ) -> None:
        self.spec = spec
        self._encode = encode
        self.revision = revision
        self.checkpoint_path = checkpoint_path
        self.checkpoint_sha256 = checkpoint_sha256
        self.device = device
        self.dtype: TorchDtype = dtype

    @classmethod
    def from_spec(
        cls, spec: EncoderSpec, device: str | None = None, dtype: TorchDtype | None = None
    ) -> FrozenEncoder:
        """Resolve, checksum and load ``spec``'s checkpoint on ``device`` (default: CUDA if any)."""
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if dtype is None:
            dtype = spec.dtype if device.startswith("cuda") else "float32"
        checkpoint_path, commit = resolve_checkpoint(spec)
        revision = spec.revision if spec.source == "open_clip" else commit
        checksum = sha256_of_file(checkpoint_path)
        if spec.checkpoint_sha256 is not None and checksum != spec.checkpoint_sha256:
            raise ValueError(
                f"{spec.name}: checkpoint sha256 {checksum} differs from the registry entry "
                f"{spec.checkpoint_sha256}"
            )
        encode, _, _ = select_loader(spec)(spec, revision, device, TORCH_DTYPES[dtype])
        return cls(spec, encode, revision, checkpoint_path, checksum, device, dtype)

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def native_dim(self) -> int:
        return self.spec.native_dim

    @property
    def preprocessing_id(self) -> str:
        return self.spec.preprocessing.id

    def identity(self, variant: str) -> StoreIdentity:
        return StoreIdentity(
            encoder=self.name,
            revision=self.revision,
            checkpoint_sha256=self.checkpoint_sha256,
            preprocessing_id=self.preprocessing_id,
            variant=variant,
            dtype=self.dtype,
        )

    def embed(self, images: Sequence[Image.Image]) -> np.ndarray:
        """L2-normalised float32 embeddings ``[len(images), native_dim]``."""
        if len(images) == 0:
            return np.zeros((0, self.native_dim), dtype=np.float32)
        with torch.inference_mode():
            pooled = self._encode([image.convert("RGB") for image in images])
            pooled = pooled.to(torch.float32)
            normalised = torch.nn.functional.normalize(pooled, p=2, dim=1)
        out = normalised.cpu().numpy().astype(np.float32, copy=False)
        if out.shape != (len(images), self.native_dim):
            raise RuntimeError(
                f"{self.name}: pooled output {out.shape} is not [{len(images)}, {self.native_dim}]"
            )
        return out
