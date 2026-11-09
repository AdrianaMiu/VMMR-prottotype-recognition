"""FastAPI application of the recognition service (PROTOCOL.md §16 "API endpoints").

``create_app`` wires :class:`ServiceResources` (profiles, segmenter, encoders and one search
backend per profile) into the endpoints. Models load in the lifespan so ``/health/ready``
reports 503 until the first request can be served. Requests are logged with their id,
profile, vehicle count and stage timings only: raw images are never logged.
"""

from __future__ import annotations

import io
import logging
import uuid
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError

import avr
from avr.encoders.registry import EncoderSpec
from avr.retrieval.qdrant_store import LocalTarget, QdrantTarget, ServerTarget
from avr.segmentation.base import Segmenter
from avr.service.pipeline import Embedder, RecognitionPipeline
from avr.service.profiles import Profile, ProfileRegistry, load_profiles
from avr.service.schemas import (
    HealthResponse,
    IndexesResponse,
    ProfileInfo,
    ProfilesResponse,
    RecognizeResponse,
)
from avr.service.search import ExactSearch, QdrantSearch, SearchBackend
from avr.service.settings import ServiceSettings

log = logging.getLogger("avr.service")

SegmenterFactory = Callable[[], Segmenter]
EncoderFactory = Callable[[EncoderSpec], Embedder]


def default_segmenter_factory() -> Segmenter:
    from avr.config import load_config
    from avr.segmentation.rfdetr_seg import RFDETRSegmenter

    return RFDETRSegmenter.from_config(load_config().segmentation)


def default_segmenter_version() -> str:
    from avr.config import load_config

    cfg = load_config().segmentation
    return f"{cfg.name}:{cfg.variant}:{cfg.checkpoint.sha256[:12]}"


def make_encoder_factory(device: str | None) -> EncoderFactory:
    def factory(spec: EncoderSpec) -> Embedder:
        from avr.encoders.adapter import FrozenEncoder

        return FrozenEncoder.from_spec(spec, device=device)

    return factory


class ServiceResources:
    """Everything the endpoints need, loaded once and rebuilt on ``reload``."""

    def __init__(
        self,
        settings: ServiceSettings,
        segmenter_factory: SegmenterFactory | None = None,
        encoder_factory: EncoderFactory | None = None,
        segmenter_version: str | None = None,
    ) -> None:
        self.settings = settings
        self._segmenter_factory = segmenter_factory or default_segmenter_factory
        self._encoder_factory = encoder_factory or make_encoder_factory(settings.device)
        self._segmenter_version = segmenter_version
        self.profiles: ProfileRegistry | None = None
        self.segmenter: Segmenter | None = None
        self.encoders: dict[str, Embedder] = {}
        self.backends: dict[str, SearchBackend] = {}
        self._qdrant_client = None

    @property
    def ready(self) -> bool:
        return self.profiles is not None and self.segmenter is not None

    def load(self) -> None:
        self.reload_indexes()
        if self.segmenter is None:
            self.segmenter = self._segmenter_factory()
            if self._segmenter_version is None:
                self._segmenter_version = default_segmenter_version()
        assert self.profiles is not None
        for profile in self.profiles:
            if profile.spec.encoder not in self.encoders:
                self.encoders[profile.spec.encoder] = self._encoder_factory(profile.encoder_spec)

    def reload_indexes(self) -> ProfileRegistry:
        """Re-read ``profiles.yaml`` and the prototype tables; rebuild the search backends."""
        s = self.settings
        profiles = load_profiles(s.profiles_path, s.prototypes_dir, s.encoders_dir)
        backends = {p.profile_id: self._backend(p) for p in profiles}
        self.profiles, self.backends = profiles, backends
        return profiles

    def _backend(self, profile: Profile) -> SearchBackend:
        s = self.settings
        if s.search_backend == "exact":
            return ExactSearch.from_parquet(profile.prototype_path)
        if s.search_backend == "qdrant_server":
            target: QdrantTarget = ServerTarget(
                url=s.qdrant_url, api_key=s.qdrant_api_key, timeout_s=s.qdrant_timeout_s
            )
        else:
            target = LocalTarget(path=Path(s.qdrant_dir))
        backend = QdrantSearch(
            target, profile.collection, client=self._qdrant_client, exact=s.qdrant_exact
        )
        self._qdrant_client = backend.client  # one client per process, shared by the profiles
        return backend

    def close(self) -> None:
        if self._qdrant_client is not None:
            self._qdrant_client.close()
            self._qdrant_client = None

    def pipeline_for(self, profile: Profile) -> RecognitionPipeline:
        if not self.ready:
            raise RuntimeError("service resources are not loaded")
        assert self.segmenter is not None
        return RecognitionPipeline(
            segmenter=self.segmenter,
            encoder=self.encoders[profile.spec.encoder],
            search=self.backends[profile.profile_id],
            variant=profile.spec.variant,
            accept_threshold=profile.spec.accept_threshold,
        )

    def versions(self, profile: Profile | None = None) -> dict[str, str | None]:
        versions: dict[str, str | None] = {"avr": avr.__version__}
        if profile is not None:
            spec = profile.encoder_spec
            encoder = self.encoders.get(profile.spec.encoder)
            versions.update(
                encoder=spec.name,
                encoder_revision=getattr(encoder, "revision", spec.revision),
                encoder_checksum=getattr(encoder, "checkpoint_sha256", spec.checkpoint_sha256),
                preprocessing_id=profile.preprocessing_id,
                prototype_method=profile.spec.prototype_method,
                collection=profile.collection,
                index_version=profile.index_version,
                search_backend=self.settings.search_backend,
            )
        versions["segmenter"] = self._segmenter_version
        return versions

    def profile_info(self, profile: Profile) -> ProfileInfo:
        assert self.profiles is not None
        return ProfileInfo(
            profile_id=profile.profile_id,
            track=profile.spec.track,
            encoder=profile.spec.encoder,
            variant=profile.spec.variant,
            dimension=profile.spec.dimension,
            prototype_method=profile.spec.prototype_method,
            collection=profile.collection,
            index_version=profile.index_version,
            prototype_count=profile.prototype_count,
            default=profile.profile_id == self.profiles.default_profile_id,
        )

    def indexes_response(self) -> IndexesResponse:
        assert self.profiles is not None
        return IndexesResponse(
            search_backend=self.settings.search_backend,
            indexes=[self.profile_info(p) for p in self.profiles],
        )


async def _read_limited(upload: UploadFile, limit: int) -> bytes:
    chunks, total = [], 0
    while True:
        chunk = await upload.read(1024 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > limit:
            raise HTTPException(413, f"image exceeds {limit} bytes")
        chunks.append(chunk)
    return b"".join(chunks)


def _decode_image(data: bytes, max_pixels: int) -> Image.Image:
    try:
        image = Image.open(io.BytesIO(data))
        width, height = image.size
        if width * height > max_pixels:
            raise HTTPException(413, f"image has more than {max_pixels} pixels")
        image.load()
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
        raise HTTPException(400, f"not a decodable image: {exc.__class__.__name__}") from exc
    return image.convert("RGB")


def create_app(
    settings: ServiceSettings | None = None, resources: ServiceResources | None = None
) -> FastAPI:
    settings = settings or ServiceSettings()
    resources = resources or ServiceResources(settings)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        resources.load()
        log.info("service ready: %s", resources.versions())
        yield
        resources.close()

    app = FastAPI(title="AVR recognition service", version=avr.__version__, lifespan=lifespan)
    app.state.resources = resources

    def require_ready() -> ServiceResources:
        if not resources.ready:
            raise HTTPException(503, "models are still loading")
        return resources

    @app.get("/health/live", response_model=HealthResponse)
    def live() -> HealthResponse:
        return HealthResponse(status="ok", versions=resources.versions())

    @app.get("/health/ready", response_model=HealthResponse)
    def ready():
        if not resources.ready:
            return JSONResponse(
                status_code=503,
                content=HealthResponse(
                    status="loading", versions=resources.versions()
                ).model_dump(),
            )
        return HealthResponse(status="ok", versions=resources.versions())

    @app.get("/v1/profiles", response_model=ProfilesResponse)
    def profiles() -> ProfilesResponse:
        r = require_ready()
        assert r.profiles is not None
        return ProfilesResponse(
            default_profile_id=r.profiles.default_profile_id,
            profiles=[r.profile_info(p) for p in r.profiles],
        )

    @app.get("/v1/indexes/current", response_model=IndexesResponse)
    def indexes_current() -> IndexesResponse:
        return require_ready().indexes_response()

    @app.post("/v1/indexes/reload", response_model=IndexesResponse)
    def indexes_reload(
        x_reload_token: Annotated[str | None, Header()] = None,
    ) -> IndexesResponse:
        if settings.reload_token is None:
            raise HTTPException(403, "index reload is disabled (no AVR_RELOAD_TOKEN)")
        if x_reload_token != settings.reload_token:
            raise HTTPException(401, "invalid reload token")
        r = require_ready()
        r.reload_indexes()
        log.info("indexes reloaded: %s", [p.index_version for p in r.profiles or []])
        return r.indexes_response()

    @app.post("/v1/recognize", response_model=RecognizeResponse)
    async def recognize(
        image: Annotated[UploadFile, File()],
        profile_id: Annotated[str | None, Form()] = None,
        top_k: Annotated[int | None, Form(ge=1)] = None,
        return_masks: Annotated[bool, Form()] = False,
        return_embeddings: Annotated[bool, Form()] = False,
    ) -> RecognizeResponse:
        r = require_ready()
        assert r.profiles is not None
        if return_embeddings and not settings.allow_embeddings:
            raise HTTPException(403, "return_embeddings is disabled in this deployment")
        k = settings.default_top_k if top_k is None else top_k
        if k > settings.max_top_k:
            raise HTTPException(422, f"top_k must be at most {settings.max_top_k}")
        pid = profile_id or r.profiles.default_profile_id
        if pid not in r.profiles:
            raise HTTPException(404, f"unknown profile_id {pid!r}")
        profile = r.profiles[pid]

        data = await _read_limited(image, settings.max_upload_bytes)
        scene = _decode_image(data, settings.max_image_pixels)

        request_id = str(uuid.uuid4())
        result = r.pipeline_for(profile).recognize(
            scene, top_k=k, return_masks=return_masks, return_embeddings=return_embeddings
        )
        log.info(
            "request %s profile=%s size=%sx%s vehicles=%d latency_ms=%s",
            request_id,
            pid,
            *result.image_size,
            len(result.vehicles),
            {s: round(v, 2) for s, v in result.latency_ms.items()},
        )
        return RecognizeResponse(
            request_id=request_id,
            profile_id=pid,
            index_version=profile.index_version,
            versions=r.versions(profile),
            vehicles=result.vehicles,
        )

    return app


def main() -> None:  # pragma: no cover - uvicorn entry point
    import uvicorn

    uvicorn.run(create_app(), host="0.0.0.0", port=8000)
