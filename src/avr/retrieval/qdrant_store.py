"""Prototype collections in Qdrant, embedded or server mode (PROTOCOL.md §12 "Online service").

The collection name is the §12 version key
``<dataset_protocol>__<mapping_version>__<encoder>__<preprocessing>__<dimension>__<prototype_method>``.
Every row of the prototype table (:mod:`avr.prototypes.persist`) becomes one point whose id
is the row position and whose payload holds ``class_id``, ``support_count`` and the keyword
fields (``make_id``, ``make_name``, ``model_id``, ``model_name``, ``year``, ``generation_id``,
``mapping_status``) as strings, plus the provenance columns, so a service can filter by make,
model or year. A keyword payload index is declared for every keyword field; embedded local
mode accepts the declaration but searches by brute force, so exact search is what both
backends do, and the parity check against the matrix ranking lives in the M1 gate.

A :data:`QdrantTarget` names the backend: :class:`LocalTarget` is the embedded store, a
directory (``data/qdrant/`` by default, git-ignored) that one client may hold at a time;
:class:`ServerTarget` is a Qdrant server reached over REST (``docker/docker-compose.yml``,
``configs/service/qdrant.yaml``). Every function here opens a client and closes it before
returning, and a bare :class:`~pathlib.Path` is still accepted as a local target.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
from pydantic import BaseModel, ConfigDict

from avr.prototypes.persist import PAYLOAD_COLUMNS

KEYWORD_FIELDS: tuple[str, ...] = PAYLOAD_COLUMNS
PROVENANCE_FIELDS = (
    "prototype_method",
    "encoder",
    "encoder_revision",
    "encoder_checksum",
    "preprocessing_id",
    "run_id",
)
UPSERT_BATCH = 512


def collection_name(
    dataset_protocol: str,
    mapping_version: str,
    encoder: str,
    preprocessing: str,
    dimension: int,
    prototype_method: str,
) -> str:
    parts = (dataset_protocol, mapping_version, encoder, preprocessing, str(dimension))
    for part in (*parts, prototype_method):
        if not part or "__" in part:
            raise ValueError(f"collection name part {part!r} must be non-empty without '__'")
    return "__".join((*parts, prototype_method))


class LocalTarget(BaseModel):
    """Embedded local mode: the store is a directory."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    path: Path


class ServerTarget(BaseModel):
    """A Qdrant server reached over REST; ``api_key`` is sent when set."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    url: str
    api_key: str | None = None
    timeout_s: int = 30


QdrantTarget = LocalTarget | ServerTarget


def resolve_target(target: QdrantTarget | Path | str | Mapping) -> QdrantTarget:
    """A target from a model, a local path or a ``{"path": ...}`` / ``{"url": ...}`` mapping."""
    if isinstance(target, LocalTarget | ServerTarget):
        return target
    if isinstance(target, Path | str):
        return LocalTarget(path=Path(target))
    mapping = dict(target)
    if ("path" in mapping) == ("url" in mapping):
        raise ValueError("a Qdrant target has exactly one of 'path' or 'url'")
    if "path" in mapping:
        return LocalTarget(**mapping)
    return ServerTarget(**mapping)


class ExportSummary(BaseModel):
    collection: str
    point_count: int
    dimension: int
    keyword_indexes: list[str]
    payload_sample: dict[str, str | int | None]


@dataclass(frozen=True)
class Hit:
    class_id: str
    score: float
    payload: dict


def _client(target: QdrantTarget | Path):
    from qdrant_client import QdrantClient

    target = resolve_target(target)
    if isinstance(target, LocalTarget):
        target.path.mkdir(parents=True, exist_ok=True)
        return QdrantClient(path=str(target.path))
    # The pinned server (docker/docker-compose.yml) may trail the client by more than one
    # minor version; the compatibility check only warns, and the REST calls used here
    # (collections, payload indexes, query_points) exist on both sides.
    return QdrantClient(
        url=target.url,
        api_key=target.api_key,
        timeout=target.timeout_s,
        check_compatibility=False,
    )


def _keyword(value) -> str | None:
    if value is None:
        return None
    return str(value)


def _payloads(table: pa.Table) -> list[dict]:
    rows = table.to_pylist()
    payloads = []
    for row in rows:
        payload: dict = {"class_id": row["class_id"], "support_count": int(row["support_count"])}
        for field in KEYWORD_FIELDS:
            payload[field] = _keyword(row[field])
        for field in PROVENANCE_FIELDS:
            payload[field] = row[field]
        payload["dimension"] = int(row["dimension"])
        payloads.append(payload)
    return payloads


def export_prototypes(
    target: QdrantTarget | Path, collection: str, table: pa.Table
) -> ExportSummary:
    """(Re)create ``collection`` on ``target`` from a prototype table."""
    from qdrant_client import models

    column = table.column("embedding").combine_chunks()
    dim = int(column.type.list_size)
    vectors = np.asarray(column.values.to_numpy(zero_copy_only=False), dtype=np.float32)
    vectors = vectors.reshape(table.num_rows, dim)
    payloads = _payloads(table)

    client = _client(target)
    try:
        if client.collection_exists(collection):
            client.delete_collection(collection)
        client.create_collection(
            collection,
            vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
        )
        with warnings.catch_warnings():
            # embedded mode warns that payload indexes have no effect; the declaration is
            # kept so the same export works against a server
            warnings.simplefilter("ignore", UserWarning)
            for field in KEYWORD_FIELDS:
                client.create_payload_index(
                    collection, field, field_schema=models.PayloadSchemaType.KEYWORD
                )
        for start in range(0, table.num_rows, UPSERT_BATCH):
            stop = min(start + UPSERT_BATCH, table.num_rows)
            client.upsert(
                collection,
                points=models.Batch(
                    ids=list(range(start, stop)),
                    vectors=vectors[start:stop].tolist(),
                    payloads=payloads[start:stop],
                ),
            )
        count = client.count(collection, exact=True).count
    finally:
        client.close()
    return ExportSummary(
        collection=collection,
        point_count=int(count),
        dimension=dim,
        keyword_indexes=list(KEYWORD_FIELDS),
        payload_sample=payloads[0] if payloads else {},
    )


def query_top_k(
    target: QdrantTarget | Path,
    collection: str,
    queries: np.ndarray,
    k: int,
    exact: bool = True,
) -> list[list[Hit]]:
    """Top-``k`` hits per query, best first; ``exact=True`` asks the server for exact search."""
    from qdrant_client import models

    queries = np.ascontiguousarray(queries, dtype=np.float32)
    client = _client(target)
    try:
        requests = [
            models.QueryRequest(
                query=q.tolist(),
                limit=k,
                with_payload=True,
                params=models.SearchParams(exact=exact),
            )
            for q in queries
        ]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            responses = client.query_batch_points(collection, requests=requests)
    finally:
        client.close()
    return [
        [
            Hit(class_id=p.payload["class_id"], score=float(p.score), payload=dict(p.payload))
            for p in response.points
        ]
        for response in responses
    ]


def list_collections(target: QdrantTarget | Path) -> list[str]:
    """Collection names on ``target``, sorted."""
    client = _client(target)
    try:
        names = [c.name for c in client.get_collections().collections]
    finally:
        client.close()
    return sorted(names)


def top_k_class_ids(hits: Sequence[Sequence[Hit]]) -> list[list[str]]:
    return [[h.class_id for h in row] for row in hits]
