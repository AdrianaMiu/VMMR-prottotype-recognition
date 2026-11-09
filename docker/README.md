# Qdrant server with monitoring (M5-1)

`docker-compose.yml` runs the online-service search backend from PROTOCOL.md §12 and its
monitoring: a pinned Qdrant server, Prometheus scraping Qdrant's `/metrics`, and Grafana with
a provisioned Prometheus datasource and one dashboard. Embedded local mode
(`data/qdrant/`) stays the default for unit tests and the M1 gate; the server is the service
path. Nothing under `data/qdrant_server/` is committed.

| Service    | Image                      | URL                                                |
|------------|----------------------------|----------------------------------------------------|
| qdrant     | `qdrant/qdrant:v1.13.6`    | REST `http://localhost:6333`, dashboard `http://localhost:6333/dashboard`, gRPC `localhost:6334` |
| prometheus | `prom/prometheus:v3.2.1`   | `http://localhost:9090` (targets: `/targets`)      |
| grafana    | `grafana/grafana:11.5.2`   | `http://localhost:3000` (admin / admin; anonymous viewer enabled), dashboard "AVR Qdrant" |

## Start and stop

```bash
mkdir -p data/qdrant_server                              # once, so the mount is owned by you
docker compose -f docker/docker-compose.yml up -d        # start (pulls the pinned images)
docker compose -f docker/docker-compose.yml ps           # health
docker compose -f docker/docker-compose.yml logs -f qdrant
docker compose -f docker/docker-compose.yml down         # stop; data/qdrant_server/ is kept
docker compose -f docker/docker-compose.yml down -v      # also drop Prometheus/Grafana volumes
```

Qdrant storage is bind-mounted at `data/qdrant_server/` (git-ignored), so collections survive
`down`/`up`. The container runs as `AVR_UID:AVR_GID` (default 1000:1000; export your own
`id -u`/`id -g` if they differ) so the files stay yours. To start clean, stop the stack and
delete that directory.

## Loading the prototype collections

`scripts/load_qdrant_server.py` walks `data/prototypes/<encoder>/<variant>/m0_<protocol_id>.parquet`
(written by `scripts/run_m1_gate.py`), checks every table against its `.sha256` sidecar, and
(re)creates one collection per table under its §12 name
`<dataset_protocol>__<mapping_version>__<encoder>__<preprocessing>__<dimension>__<prototype_method>`
with keyword payload indexes on `make_id`, `make_name`, `model_id`, `model_name`, `year`,
`generation_id` and `mapping_status`. It prints the collections on the server when done.

```bash
uv run python scripts/load_qdrant_server.py                       # target from configs/service/qdrant.yaml
uv run python scripts/load_qdrant_server.py --prototypes-dir /abs/other/checkout/data/prototypes
uv run python scripts/load_qdrant_server.py --url http://host:6333 --api-key ...
uv run python scripts/load_qdrant_server.py --path data/qdrant    # embedded local store instead
```

The target is `configs/service/qdrant.yaml` (`url`, `api_key`, `timeout_s`, `exact: true`);
`AVR_QDRANT_URL` and `AVR_QDRANT_API_KEY` override it. The four M1 gate tables (Track A and
Track B, P1 and P3, `dinov3_vitb16`, 768-d) load in about 15 s.

## Parity and latency

`tests/integration/test_qdrant_server_parity.py` runs only when `AVR_QDRANT_URL` is set. It
ranks the first 1,000 Track B test queries against the Track B / P1 prototypes with
`rank_by_cosine` and asks the server for the exact top-10
(`SearchParams(exact=True)`), one request per query; the lists must agree up to swaps within
the M1 gate tolerance (1e-5 cosine). It writes `reports/qdrant_server/parity_track_b_p1.json`
with the agreement counts and the client-side latency percentiles.

```bash
AVR_QDRANT_URL=http://localhost:6333 uv run pytest tests/integration/test_qdrant_server_parity.py
```

Measured on 2026-11-09 (Track B / P1, 2,645 prototypes, 768-d, 1,000 queries, exact top-10,
server on the same machine): 1,000 of 1,000 top-10 lists identical in order. Client round
trip (one `QdrantClient` opened per call, as `query_top_k` does) p50 48 ms, p95 112 ms;
server-side `rest_responses_duration_seconds` for the query endpoint: mean 3.9 ms,
875/1000 requests under 5 ms, 961/1000 under 10 ms, max 17.8 ms. The gap is client
connection set-up, not search; a long-lived client is the obvious service-side change.
Internal validation only, not an article result.

`tests/integration/test_docker_compose.py` renders the compose file with
`docker compose config` (nothing is started) and checks images are pinned, ports, the storage
mount, the Prometheus scrape target and the Grafana provisioning.

## Monitoring

Prometheus scrapes `qdrant:6333/metrics` every 15 s. The Grafana dashboard
(`grafana/dashboards/qdrant.json`, provisioned read-only from this folder) shows:

- **Collection point counts**: `collections_vector_total` and `collections_total`. Qdrant's
  `/metrics` exposes totals only; per-collection counts are printed by the loading script or
  returned by `GET /collections/<name>`.
- **Request latency**: p50/p95 from `rest_responses_duration_seconds_bucket` per endpoint, plus
  the lifetime average of the query endpoint. Qdrant's histogram buckets are coarse
  (1, 5, 10, 50, 100, 500 ms, ...), so the quantiles are interpolated upper estimates; the
  average series and the parity report give the finer numbers.
- **Memory**: `memory_resident_bytes`, `memory_allocated_bytes`, `memory_active_bytes`,
  `memory_retained_bytes`.
- REST request rate and failures.

The future `GET /v1/indexes/current` endpoint (PROTOCOL.md §16) is expected to report the
collection names from `avr.retrieval.qdrant_store.list_collections` and the per-collection
point counts.
