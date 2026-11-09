"""Exact search on the Qdrant server versus the matrix ranking (issue M5-1, PROTOCOL.md §12).

Gated on ``AVR_QDRANT_URL``: the server from docker/docker-compose.yml must be running and
``scripts/load_qdrant_server.py`` must have loaded the Track B / P1 collection. Queries are the
first 1,000 Track B test images with an embedding; the top-10 from the server (exact search)
must equal the ``rank_by_cosine`` top-10, up to swaps whose cosine gap is within the M1 gate
tolerance (1e-5). Per-query latency of the exact top-10 is written to
reports/qdrant_server/parity_track_b_p1.json. Paths default to this checkout's data directory
and can be pointed elsewhere with ``AVR_PROTOTYPES_DIR``, ``AVR_EMBEDDINGS_DIR`` and
``AVR_SPLIT_PATH``.
"""

from __future__ import annotations

import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from avr.encoders.store import EmbeddingStore
from avr.pipeline.gate import PARITY_K, PARITY_SCORE_TOLERANCE
from avr.prototypes.persist import read_prototype_table
from avr.retrieval.exact import rank_by_cosine
from avr.retrieval.qdrant_load import collection_for_table
from avr.retrieval.qdrant_store import ServerTarget, list_collections, query_top_k

REPO = Path(__file__).resolve().parents[2]
QUERIES = 1000
ENCODER, VARIANT = "dinov3_vitb16", "p1"
REPORT = REPO / "reports" / "qdrant_server" / f"parity_track_b_{VARIANT}.json"

pytestmark = pytest.mark.skipif(
    "AVR_QDRANT_URL" not in os.environ, reason="AVR_QDRANT_URL not set: no Qdrant server"
)


def _env_path(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, default)).expanduser()


@pytest.fixture(scope="module")
def target() -> ServerTarget:
    return ServerTarget(
        url=os.environ["AVR_QDRANT_URL"], api_key=os.environ.get("AVR_QDRANT_API_KEY")
    )


@pytest.fixture(scope="module")
def queries() -> np.ndarray:
    split_path = _env_path("AVR_SPLIT_PATH", REPO / "data/splits/track_b/seed_73.parquet")
    embeddings_dir = _env_path("AVR_EMBEDDINGS_DIR", REPO / "data/embeddings")
    if not split_path.exists() or not embeddings_dir.exists():
        pytest.skip("Track B split or embedding store not available")
    split = pd.read_parquet(split_path)
    test_ids = sorted(split.loc[split.partition == "test", "image_id"])
    ids, matrix = EmbeddingStore.from_directory(embeddings_dir, ENCODER, VARIANT).load_all()
    row_of = {image_id: row for row, image_id in enumerate(ids)}
    rows = [row_of[i] for i in test_ids if i in row_of][:QUERIES]
    if len(rows) < QUERIES:
        pytest.skip(f"only {len(rows)} test queries with embeddings")
    return np.ascontiguousarray(matrix[rows], dtype=np.float32)


def test_server_exact_top_10_matches_the_matrix_ranking_on_1000_queries(target, queries):
    prototypes_dir = _env_path("AVR_PROTOTYPES_DIR", REPO / "data/prototypes")
    table_path = prototypes_dir / ENCODER / VARIANT / "m0_track_b_seed_73.parquet"
    if not table_path.exists():
        pytest.skip(f"{table_path} not available")
    table, prototypes = read_prototype_table(table_path)
    class_ids = table.column("class_id").to_pylist()
    collection = collection_for_table(table_path, table)
    if collection not in list_collections(target):
        pytest.skip(f"{collection} not loaded on {target.url}")

    k = min(PARITY_K, len(class_ids))
    ranking = rank_by_cosine(queries, prototypes, class_ids)
    matrix_top = [[class_ids[r] for r in ranking.ranks[q, :k]] for q in range(len(queries))]

    latencies_ms = []
    server_top = []
    for q in queries:
        started = time.perf_counter()
        hits = query_top_k(target, collection, q[None, :], k, exact=True)
        latencies_ms.append((time.perf_counter() - started) * 1000.0)
        server_top.append([h.class_id for h in hits[0]])

    raw = queries @ prototypes.T
    index = {c: i for i, c in enumerate(class_ids)}
    ordered = within = mismatched = 0
    for q in range(len(queries)):
        if server_top[q] == matrix_top[q]:
            ordered += 1
            continue
        gaps = [
            abs(float(raw[q, index[a]]) - float(raw[q, index[b]]))
            for a, b in zip(server_top[q], matrix_top[q], strict=True)
            if a != b
        ]
        if len(server_top[q]) == k and max(gaps) <= PARITY_SCORE_TOLERANCE:
            within += 1
        else:
            mismatched += 1

    latency = np.asarray(latencies_ms)
    report = {
        "collection": collection,
        "server": target.url,
        "exact": True,
        "queries": len(queries),
        "k": k,
        "ordered_equal": ordered,
        "equal_within_tolerance": within,
        "mismatched": mismatched,
        "score_tolerance": PARITY_SCORE_TOLERANCE,
        "latency_ms": {
            "p50": round(float(np.percentile(latency, 50)), 3),
            "p95": round(float(np.percentile(latency, 95)), 3),
            "p99": round(float(np.percentile(latency, 99)), 3),
            "mean": round(float(latency.mean()), 3),
            "max": round(float(latency.max()), 3),
        },
        "note": "one REST request per query, client round trip included; internal validation only",
        "measured_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2) + "\n")

    assert mismatched == 0, f"{mismatched} of {len(queries)} queries differ beyond tolerance"
    assert ordered + within == len(queries)
