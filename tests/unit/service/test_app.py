"""FastAPI endpoints of the recognition service (PROTOCOL.md §16, §23 "API")."""

import hashlib
import io
import json

import numpy as np
import pytest
import yaml
from fastapi.testclient import TestClient
from PIL import Image

from avr.prototypes.persist import prototype_table, write_prototype_table
from avr.service.app import ServiceResources, create_app
from avr.service.schemas import STAGES
from avr.service.settings import ServiceSettings

from .conftest import FIVE_CLASS_IDS, FIVE_COLLECTION, PROVENANCE
from .fakes import ColourEncoder, ScriptedSegmenter, box_instance, scene
from .test_profiles import TOY_ENCODER

RED, BLUE = (220, 20, 20), (20, 20, 220)
SIZE = (120, 80)
BOX_A = (10, 10, 50, 40)
BOX_B = (60, 30, 110, 75)
PROFILE = "toy__toy__p1__native__m0"
TOKEN = "s3cret"


def png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def service(tmp_path, five_prototype_files):
    """A client factory: ``make(instances, **settings)`` -> (TestClient, resources)."""
    encoders = tmp_path / "encoders"
    encoders.mkdir()
    (encoders / "toy.yaml").write_text(yaml.safe_dump(TOY_ENCODER))
    profiles_path = tmp_path / "profiles.yaml"
    profiles_path.write_text(
        yaml.safe_dump(
            {
                "default_profile_id": PROFILE,
                "profiles": [
                    {
                        "profile_id": PROFILE,
                        "track": "track_a",
                        "dataset_protocol": "toy",
                        "mapping_version": "none",
                        "encoder": "toy",
                        "variant": "p1",
                        "prototype_method": "m0_global_mean",
                        "prototype_table": "toy/p1/m0_toy.parquet",
                    }
                ],
            }
        )
    )

    def make(instances, **overrides):
        settings = ServiceSettings(
            **{
                "profiles_path": profiles_path,
                "prototypes_dir": five_prototype_files["parquet"].parents[2],
                "qdrant_dir": five_prototype_files["qdrant"],
                "encoders_dir": encoders,
                "reload_token": TOKEN,
                **overrides,
            }
        )
        resources = ServiceResources(
            settings,
            segmenter_factory=lambda: ScriptedSegmenter(instances),
            encoder_factory=lambda spec: ColourEncoder(),
            segmenter_version="scripted",
        )
        app = create_app(settings, resources)
        return TestClient(app), resources

    return make


def recognize(client, image, **form):
    data = {k: str(v).lower() if isinstance(v, bool) else str(v) for k, v in form.items()}
    return client.post(
        "/v1/recognize", files={"image": ("scene.png", png_bytes(image), "image/png")}, data=data
    )


def test_health_endpoints(service):
    client, _ = service([])
    with client:
        assert client.get("/health/live").json()["status"] == "ok"
        ready = client.get("/health/ready")
        assert ready.status_code == 200
        assert ready.json()["versions"]["avr"]


def test_ready_is_503_before_models_are_loaded(service):
    client, _ = service([])
    # no ``with client``: the lifespan did not run, so nothing is loaded yet
    assert client.get("/health/ready").status_code == 503
    assert client.get("/health/live").status_code == 200


def test_profiles_and_current_index_list_the_collection_and_version(service, five_prototype_files):
    client, _ = service([])
    expected_version = hashlib.sha256(five_prototype_files["parquet"].read_bytes()).hexdigest()
    with client:
        profiles = client.get("/v1/profiles").json()
        indexes = client.get("/v1/indexes/current").json()

    assert profiles["default_profile_id"] == PROFILE
    (profile,) = profiles["profiles"]
    assert profile["collection"] == FIVE_COLLECTION
    assert profile["index_version"] == expected_version
    assert profile["prototype_count"] == 5
    assert profile["default"] is True
    assert indexes["search_backend"] == "qdrant_embedded"
    assert indexes["indexes"] == profiles["profiles"]


def test_zero_vehicles_returns_an_empty_list_with_versions(service, five_prototype_files):
    client, _ = service([])
    with client:
        response = recognize(client, scene(SIZE, {}))

    assert response.status_code == 200
    body = response.json()
    assert body["vehicles"] == []
    assert body["profile_id"] == PROFILE
    assert (
        body["index_version"]
        == hashlib.sha256(five_prototype_files["parquet"].read_bytes()).hexdigest()
    )
    assert body["versions"]["encoder"] == "toy"
    assert body["versions"]["encoder_revision"] == "r1"
    assert body["versions"]["collection"] == FIVE_COLLECTION
    assert body["versions"]["segmenter"] == "scripted"
    assert body["versions"]["avr"]


def test_one_vehicle_top_k_and_timings(service):
    client, _ = service([box_instance(BOX_A, SIZE, 0.9)])
    with client:
        body = recognize(client, scene(SIZE, {BOX_A: RED}), top_k=3).json()

    (vehicle,) = body["vehicles"]
    assert vehicle["bbox_xyxy"] == list(map(float, BOX_A))
    assert [p["class_id"] for p in vehicle["predictions"]] == ["c0", "c1", "c2"]
    assert vehicle["predictions"][0]["calibrated_confidence"] is None
    assert vehicle["viewpoint"] is None
    assert list(vehicle["latency_ms"]) == list(STAGES)
    assert all(v >= 0 for v in vehicle["latency_ms"].values())
    assert vehicle["mask_png_base64"] is None and vehicle["embedding"] is None


def test_multiple_vehicles_each_get_their_own_ranking(service):
    client, _ = service([box_instance(BOX_A, SIZE, 0.9), box_instance(BOX_B, SIZE, 0.8)])
    with client:
        body = recognize(client, scene(SIZE, {BOX_A: RED, BOX_B: BLUE}), top_k=1).json()

    assert [v["detection_id"] for v in body["vehicles"]] == [0, 1]
    assert [v["predictions"][0]["class_id"] for v in body["vehicles"]] == ["c0", "c2"]


def test_default_top_k_is_ten_capped_by_the_collection(service):
    client, _ = service([box_instance(BOX_A, SIZE)])
    with client:
        body = recognize(client, scene(SIZE, {BOX_A: RED})).json()
    assert len(body["vehicles"][0]["predictions"]) == 5


def test_response_schema_is_deterministic_apart_from_request_id_and_timings(service):
    client, _ = service([box_instance(BOX_A, SIZE)])
    image = scene(SIZE, {BOX_A: RED})

    def normalised(body):
        body = json.loads(json.dumps(body))
        body["request_id"] = "x"
        for v in body["vehicles"]:
            v["latency_ms"] = {k: 0 for k in v["latency_ms"]}
        return body

    with client:
        a = recognize(client, image, top_k=2).json()
        b = recognize(client, image, top_k=2).json()
    assert a["request_id"] != b["request_id"]
    assert normalised(a) == normalised(b)
    assert list(a) == ["request_id", "profile_id", "index_version", "versions", "vehicles"]


def test_invalid_file_is_rejected_with_400(service):
    client, _ = service([])
    with client:
        response = client.post(
            "/v1/recognize", files={"image": ("x.jpg", b"not an image", "image/jpeg")}
        )
    assert response.status_code == 400


def test_oversized_upload_is_rejected_with_413(service):
    client, _ = service([], max_upload_bytes=1024)
    with client:
        response = recognize(client, scene((400, 400), {(0, 0, 400, 400): RED}))
    assert response.status_code == 413


def test_too_many_pixels_is_rejected_with_413(service):
    client, _ = service([], max_image_pixels=1000)
    with client:
        response = recognize(client, scene(SIZE, {}))
    assert response.status_code == 413


def test_unknown_profile_is_404_and_bad_top_k_is_422(service):
    client, _ = service([])
    with client:
        assert (
            recognize(client, scene(SIZE, {}), profile_id="nope__x__p1__native__m0").status_code
            == 404
        )
        assert recognize(client, scene(SIZE, {}), top_k=0).status_code == 422
        assert recognize(client, scene(SIZE, {}), top_k=1000).status_code == 422
        assert client.post("/v1/recognize", data={"top_k": "3"}).status_code == 422


def test_return_embeddings_is_forbidden_unless_the_dev_flag_is_set(service):
    client, _ = service([box_instance(BOX_A, SIZE)])
    with client:
        assert recognize(client, scene(SIZE, {}), return_embeddings=True).status_code == 403
    client, _ = service([box_instance(BOX_A, SIZE)], allow_embeddings=True)
    with client:
        body = recognize(client, scene(SIZE, {BOX_A: RED}), return_embeddings=True).json()
    assert len(body["vehicles"][0]["embedding"]) == 5


def test_return_masks_gives_a_png_in_input_geometry(service):
    client, _ = service([box_instance(BOX_A, SIZE)])
    with client:
        body = recognize(client, scene(SIZE, {BOX_A: RED}), return_masks=True).json()
    import base64

    mask = Image.open(io.BytesIO(base64.b64decode(body["vehicles"][0]["mask_png_base64"])))
    assert mask.size == SIZE


def test_reload_needs_the_token_and_picks_up_a_rewritten_table(service, five_prototype_files):
    client, _ = service([])
    parquet = five_prototype_files["parquet"]
    with client:
        before = client.get("/v1/indexes/current").json()["indexes"][0]["index_version"]
        assert client.post("/v1/indexes/reload").status_code == 401
        assert (
            client.post("/v1/indexes/reload", headers={"X-Reload-Token": "no"}).status_code == 401
        )

        import pandas as pd

        payload = pd.DataFrame(
            {
                "class_id": FIVE_CLASS_IDS,
                "make_id": [1] * 5,
                "make_name": ["Z"] * 5,
                "model_id": [1] * 5,
                "model_name": ["z"] * 5,
                "year": [2000] * 5,
                "generation_id": ["g"] * 5,
                "mapping_status": ["provisional"] * 5,
            }
        )
        table = prototype_table(
            np.eye(5, dtype=np.float32), FIVE_CLASS_IDS, [1] * 5, payload, PROVENANCE
        )
        write_prototype_table(table, parquet)
        response = client.post("/v1/indexes/reload", headers={"X-Reload-Token": TOKEN})
        after = client.get("/v1/indexes/current").json()["indexes"][0]["index_version"]

    assert response.status_code == 200
    assert response.json()["indexes"][0]["index_version"] == after
    assert after == hashlib.sha256(parquet.read_bytes()).hexdigest() != before


def test_reload_is_disabled_without_a_configured_token(service):
    client, _ = service([], reload_token=None)
    with client:
        response = client.post("/v1/indexes/reload", headers={"X-Reload-Token": TOKEN})
    assert response.status_code == 403


def test_exact_backend_is_selectable_and_agrees_with_qdrant(service):
    instances = [box_instance(BOX_A, SIZE), box_instance(BOX_B, SIZE)]
    image = scene(SIZE, {BOX_A: RED, BOX_B: BLUE})
    client_q, _ = service(instances, search_backend="qdrant_embedded")
    client_e, _ = service(instances, search_backend="exact")
    with client_q, client_e:
        q = recognize(client_q, image, top_k=5).json()["vehicles"]
        e = recognize(client_e, image, top_k=5).json()["vehicles"]
        assert client_e.get("/v1/indexes/current").json()["search_backend"] == "exact"

    for vq, ve in zip(q, e, strict=True):
        assert [p["class_id"] for p in vq["predictions"]] == [
            p["class_id"] for p in ve["predictions"]
        ]
        assert np.allclose(
            [p["cosine_similarity"] for p in vq["predictions"]],
            [p["cosine_similarity"] for p in ve["predictions"]],
            atol=1e-5,
        )
