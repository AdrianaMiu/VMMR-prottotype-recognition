"""Prototype search backends behind the service (PROTOCOL.md §12 "Online service").

``SearchBackend`` is the one seam the pipeline searches through: L2-normalised query
embeddings in, top-``k`` :class:`Hit` rows per query out, best first, each hit carrying the
hierarchy payload with native types (``year`` as ``int``). Two implementations exist today:

* :class:`ExactSearch`: the §12 exact cosine ranking (``avr.retrieval.exact``) over the
  prototype table held in memory; the reference every other backend must match;
* :class:`EmbeddedQdrantSearch`: the embedded local Qdrant store written by the M1 gate
  (``avr.retrieval.qdrant_store``), exact by construction in embedded mode.

A Qdrant server backend (``url`` mode) plugs in through the same protocol.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
import pyarrow as pa

from avr.prototypes.persist import PAYLOAD_COLUMNS, read_prototype_table
from avr.retrieval.exact import rank_by_cosine

INT_PAYLOAD_FIELDS = frozenset({"make_id", "model_id", "year", "support_count", "dimension"})


@dataclass(frozen=True)
class Hit:
    class_id: str
    score: float
    payload: dict


@runtime_checkable
class SearchBackend(Protocol):
    def search(self, queries: np.ndarray, k: int) -> list[list[Hit]]: ...


def _native(payload: dict) -> dict:
    """Qdrant stores keyword fields as strings; give the pipeline the table's native types."""
    out = dict(payload)
    for field in INT_PAYLOAD_FIELDS:
        value = out.get(field)
        if value is not None and not isinstance(value, int):
            out[field] = int(value)
    return out


class ExactSearch:
    """Exact cosine ranking over an in-memory prototype table (the §12 reference)."""

    def __init__(self, table: pa.Table, prototypes: np.ndarray) -> None:
        self.class_ids: list[str] = table.column("class_id").to_pylist()
        self.prototypes = np.ascontiguousarray(prototypes, dtype=np.float32)
        rows = table.select(["class_id", "support_count", *PAYLOAD_COLUMNS]).to_pylist()
        self._payloads = [_native(row) for row in rows]

    @classmethod
    def from_parquet(cls, path: Path) -> ExactSearch:
        table, matrix = read_prototype_table(Path(path))
        return cls(table, matrix)

    @property
    def size(self) -> int:
        return len(self.class_ids)

    @property
    def dimension(self) -> int:
        return int(self.prototypes.shape[1])

    def search(self, queries: np.ndarray, k: int) -> list[list[Hit]]:
        queries = np.ascontiguousarray(queries, dtype=np.float32)
        if queries.shape[0] == 0:
            return []
        ranking = rank_by_cosine(queries, self.prototypes, self.class_ids)
        k = min(int(k), self.size)
        return [
            [
                Hit(self.class_ids[idx], float(ranking.scores[q, r]), self._payloads[idx])
                for r, idx in enumerate(ranking.ranks[q, :k])
            ]
            for q in range(queries.shape[0])
        ]


class EmbeddedQdrantSearch:
    """Top-k from an embedded local Qdrant collection, exact in embedded mode.

    The embedded store admits one client per process, and opening one loads the collection
    from disk (hundreds of milliseconds), so the client is opened once and kept; several
    backends on the same store share it through ``client``. ``close`` releases the directory.
    """

    def __init__(self, path: Path, collection: str, client=None) -> None:
        self.path = Path(path)
        self.collection = collection
        self._client = client

    @property
    def client(self):
        if self._client is None:
            from qdrant_client import QdrantClient

            self._client = QdrantClient(path=str(self.path))
        return self._client

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def search(self, queries: np.ndarray, k: int) -> list[list[Hit]]:
        from qdrant_client import models

        queries = np.ascontiguousarray(queries, dtype=np.float32)
        if queries.shape[0] == 0:
            return []
        requests = [
            models.QueryRequest(query=q.tolist(), limit=int(k), with_payload=True) for q in queries
        ]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            responses = self.client.query_batch_points(self.collection, requests=requests)
        return [
            [Hit(p.payload["class_id"], float(p.score), _native(p.payload)) for p in r.points]
            for r in responses
        ]
