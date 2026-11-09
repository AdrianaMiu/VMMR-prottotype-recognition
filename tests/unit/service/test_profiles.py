"""Profile registry: profile id -> track, encoder, variant, method, collection, index version."""

import hashlib
from pathlib import Path

import pytest
import yaml

from avr.service.profiles import ProfileRegistry, load_profiles

TOY_ENCODER = {
    "name": "toy",
    "model_id": "toy/toy",
    "source": "huggingface",
    "revision": "r1",
    "license": "MIT",
    "input_resolution": 8,
    "native_dim": 5,
    "pooling": "cls",
    "preprocessing": {"id": "toy_pre"},
    "dtype": "float32",
    "checkpoint_file": "model.safetensors",
}


@pytest.fixture
def profiles_yaml(tmp_path: Path, five_prototype_files) -> Path:
    encoders = tmp_path / "encoders"
    encoders.mkdir()
    (encoders / "toy.yaml").write_text(yaml.safe_dump(TOY_ENCODER))
    doc = {
        "default_profile_id": "toy__toy__p1__native__m0",
        "profiles": [
            {
                "profile_id": "toy__toy__p1__native__m0",
                "track": "track_a",
                "dataset_protocol": "toy",
                "mapping_version": "none",
                "encoder": "toy",
                "variant": "p1",
                "dimension": "native",
                "prototype_method": "m0_global_mean",
                "prototype_table": "toy/p1/m0_toy.parquet",
                "accept_threshold": 0.6,
            }
        ],
    }
    path = tmp_path / "profiles.yaml"
    path.write_text(yaml.safe_dump(doc))
    return path


def test_registry_resolves_collection_name_and_index_version(profiles_yaml, five_prototype_files):
    registry = load_profiles(
        profiles_yaml,
        prototypes_dir=five_prototype_files["parquet"].parents[2],
        encoders_dir=profiles_yaml.parent / "encoders",
    )

    assert isinstance(registry, ProfileRegistry)
    profile = registry["toy__toy__p1__native__m0"]
    assert registry.default_profile_id == profile.profile_id
    assert profile.collection == "toy__none__toy__toy_pre+p1__5__m0_global_mean"
    assert profile.preprocessing_id == "toy_pre+p1"
    assert profile.dimension_value == 5
    expected = hashlib.sha256(five_prototype_files["parquet"].read_bytes()).hexdigest()
    assert profile.index_version == expected
    assert profile.prototype_path == five_prototype_files["parquet"]


def test_profile_id_must_follow_the_section_16_pattern(profiles_yaml, five_prototype_files):
    doc = yaml.safe_load(profiles_yaml.read_text())
    doc["profiles"][0]["profile_id"] = "toy-p1"
    doc["default_profile_id"] = "toy-p1"
    profiles_yaml.write_text(yaml.safe_dump(doc))

    with pytest.raises(ValueError, match="profile_id"):
        load_profiles(
            profiles_yaml,
            prototypes_dir=five_prototype_files["parquet"].parents[2],
            encoders_dir=profiles_yaml.parent / "encoders",
        )


def test_missing_prototype_table_fails_closed(profiles_yaml, tmp_path):
    with pytest.raises(FileNotFoundError):
        load_profiles(
            profiles_yaml,
            prototypes_dir=tmp_path / "nowhere",
            encoders_dir=profiles_yaml.parent / "encoders",
        )


def test_index_version_differs_from_sidecar_when_table_was_rewritten(
    profiles_yaml, five_prototype_files
):
    parquet = five_prototype_files["parquet"]
    parquet.with_suffix(".sha256").write_text(f"{'0' * 64}  {parquet.name}\n")

    with pytest.raises(ValueError, match="sidecar"):
        load_profiles(
            profiles_yaml,
            prototypes_dir=parquet.parents[2],
            encoders_dir=profiles_yaml.parent / "encoders",
        )


def test_shipped_profiles_cover_the_four_m1_gate_collections():
    from avr.config import CONFIG_DIR

    doc = yaml.safe_load((CONFIG_DIR / "service" / "profiles.yaml").read_text())
    ids = {p["profile_id"] for p in doc["profiles"]}

    assert ids == {
        "track_a__dinov3_vitb16__p1__native__m0",
        "track_a__dinov3_vitb16__p3__native__m0",
        "track_b__dinov3_vitb16__p1__native__m0",
        "track_b__dinov3_vitb16__p3__native__m0",
    }
    assert doc["default_profile_id"] in ids
