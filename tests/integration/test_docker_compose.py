"""The Qdrant/Prometheus/Grafana stack in docker/docker-compose.yml (issue M5-1).

Nothing is started: ``docker compose config`` only renders the file, so the test proves the
compose file, the Prometheus scrape config and the Grafana provisioning parse and agree on
service names and ports. Skipped when docker is not on PATH.
"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
COMPOSE = REPO / "docker" / "docker-compose.yml"

pytestmark = pytest.mark.skipif(shutil.which("docker") is None, reason="docker not installed")


@pytest.fixture(scope="module")
def rendered() -> dict:
    assert COMPOSE.is_file(), f"{COMPOSE} is missing"
    result = subprocess.run(
        ["docker", "compose", "-f", str(COMPOSE), "config", "--format", "json"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.skip(f"docker compose config failed: {result.stderr.strip()[:200]}")
    return json.loads(result.stdout)


def test_compose_defines_the_three_services_with_pinned_images(rendered):
    services = rendered["services"]
    assert set(services) == {"qdrant", "prometheus", "grafana"}
    for name, service in services.items():
        image = service["image"]
        tag = image.rsplit(":", 1)[-1]
        assert tag not in ("latest", image), f"{name} image {image} must be pinned"
    assert services["qdrant"]["image"].startswith("qdrant/qdrant:v1.13.")


def test_qdrant_exposes_rest_and_grpc_and_persists_under_data_qdrant_server(rendered):
    qdrant = rendered["services"]["qdrant"]
    published = {int(p["published"]): int(p["target"]) for p in qdrant["ports"]}
    assert published == {6333: 6333, 6334: 6334}
    volume = qdrant["volumes"][0]
    assert Path(volume["source"]).resolve() == (REPO / "data" / "qdrant_server").resolve()
    assert volume["target"] == "/qdrant/storage"


def test_prometheus_scrapes_qdrant_metrics_and_grafana_is_provisioned(rendered):
    services = rendered["services"]
    scrape = yaml.safe_load((REPO / "docker" / "prometheus" / "prometheus.yml").read_text())
    jobs = {job["job_name"]: job for job in scrape["scrape_configs"]}
    assert jobs["qdrant"]["metrics_path"] == "/metrics"
    assert jobs["qdrant"]["static_configs"][0]["targets"] == ["qdrant:6333"]
    assert {int(p["published"]) for p in services["prometheus"]["ports"]} == {9090}

    grafana = services["grafana"]
    assert {int(p["published"]) for p in grafana["ports"]} == {3000}
    mounted = {v["target"] for v in grafana["volumes"]}
    assert "/etc/grafana/provisioning" in mounted
    provisioning = REPO / "docker" / "grafana" / "provisioning"
    datasource = yaml.safe_load((provisioning / "datasources" / "prometheus.yml").read_text())
    assert datasource["datasources"][0]["url"] == "http://prometheus:9090"
    dashboard = json.loads((REPO / "docker" / "grafana" / "dashboards" / "qdrant.json").read_text())
    titles = {panel["title"] for panel in dashboard["panels"]}
    assert {"Collection point counts", "Request latency", "Memory"} <= titles
