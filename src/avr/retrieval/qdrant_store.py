"""Prototype collections in Qdrant, embedded local mode (PROTOCOL.md §12 "Online service").

The collection name is the §12 version key
``<dataset_protocol>__<mapping_version>__<encoder>__<preprocessing>__<dimension>__<prototype_method>``.
Every row of the prototype table (:mod:`avr.prototypes.persist`) becomes one point whose id
is the row position and whose payload holds ``class_id``, ``support_count`` and the keyword
fields (``make_id``, ``make_name``, ``model_id``, ``model_name``, ``year``, ``generation_id``,
``mapping_status``) as strings, plus the provenance columns, so a service can filter by make,
model or year. A keyword payload index is declared for every keyword field; embedded local
mode accepts the declaration but searches by brute force, so exact search is what both
backends do, and the parity check against the matrix ranking lives in the M1 gate.

The local store is a directory (``data/qdrant/`` by default, git-ignored) that one client
may hold at a time; every function here opens a client and closes it before returning.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
from pydantic import BaseModel

from avr.io import read_fixed_list_matrix
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


def _client(path: Path):
    from qdrant_client import QdrantClient

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(path))


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


def export_prototypes(path: Path, collection: str, table: pa.Table) -> ExportSummary:
    """(Re)create ``collection`` under ``path`` from a prototype table."""
    from qdrant_client import models

    vectors = read_fixed_list_matrix(table, "embedding")
    dim = int(vectors.shape[1])
    payloads = _payloads(table)

    client = _client(path)
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


def query_top_k(path: Path, collection: str, queries: np.ndarray, k: int) -> list[list[Hit]]:
    """Exact top-``k`` hits per query, best first, from the local collection."""
    from qdrant_client import models

    queries = np.ascontiguousarray(queries, dtype=np.float32)
    client = _client(path)
    try:
        requests = [
            models.QueryRequest(query=q.tolist(), limit=k, with_payload=True) for q in queries
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


def top_k_class_ids(hits: Sequence[Sequence[Hit]]) -> list[list[str]]:
    return [[h.class_id for h in row] for row in hits]
