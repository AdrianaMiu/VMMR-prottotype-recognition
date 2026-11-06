"""Encoder registry schema and loader (PROTOCOL.md §9, ADR-0003, issue M0-5).

Expected values are literals taken from the model cards and the protocol; nothing here
touches the network or loads a checkpoint.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from avr.encoders.registry import EncoderSpec

VALID_ENTRY = {
    "name": "clip_vitl14_336",
    "model_id": "openai/clip-vit-large-patch14-336",
    "source": "huggingface",
    "revision": "ce19dc912ca5cd21c8a653c79e251e808ccabcd1",
    "license": "MIT",
    "parameter_count": 427944193,
    "input_resolution": 336,
    "native_dim": 768,
    "pooling": "projection",
    "preprocessing": {"id": "clip_vitl14_336_v1"},
    "dtype": "float16",
    "checkpoint_file": "model.safetensors",
    "checkpoint_sha256": "a" * 64,
    "verified_at": "2026-11-06T10:00:00+00:00",
    "verified_dtype": "float32",
    "verified_output_shape": [1, 768],
}


def test_complete_entry_validates_and_keeps_protocol_fields():
    spec = EncoderSpec.model_validate(VALID_ENTRY)

    assert spec.model_id == "openai/clip-vit-large-patch14-336"
    assert spec.source == "huggingface"
    assert spec.pooling == "projection"
    assert spec.native_dim == 768
    assert spec.preprocessing.id == "clip_vitl14_336_v1"
    assert spec.verified_output_shape == [1, 768]
    assert isinstance(spec.verified_at, datetime)


def test_unverified_entry_is_allowed_with_null_verification_fields():
    entry = {
        **VALID_ENTRY,
        "parameter_count": None,
        "checkpoint_sha256": None,
        "verified_at": None,
        "verified_dtype": None,
        "verified_output_shape": None,
    }

    spec = EncoderSpec.model_validate(entry)

    assert spec.checkpoint_sha256 is None and spec.verified_output_shape is None


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("source", "torch_hub"),
        ("pooling", "max_patch"),
        ("dtype", "int8"),
        ("checkpoint_sha256", "abc"),
        ("native_dim", 0),
        ("name", "Clip ViT"),
    ],
)
def test_invalid_field_values_are_rejected(field, bad_value):
    with pytest.raises(ValidationError):
        EncoderSpec.model_validate({**VALID_ENTRY, field: bad_value})


def test_native_dim_must_equal_last_element_of_verified_output_shape():
    with pytest.raises(ValidationError, match="native_dim"):
        EncoderSpec.model_validate({**VALID_ENTRY, "verified_output_shape": [1, 1024]})


def test_unknown_top_level_keys_are_rejected():
    with pytest.raises(ValidationError):
        EncoderSpec.model_validate({**VALID_ENTRY, "embedding_dim": 768})


def test_preprocessing_block_keeps_checkpoint_specific_keys():
    entry = {**VALID_ENTRY, "preprocessing": {"id": "naflex_v1", "max_num_patches": 256}}

    spec = EncoderSpec.model_validate(entry)

    assert spec.preprocessing.model_dump()["max_num_patches"] == 256


def write_yaml_entry(directory: Path, entry: dict) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{entry['name']}.yaml"
    path.write_text(yaml.safe_dump(entry))
    return path


def test_load_registry_returns_validated_entries_keyed_by_name(tmp_path: Path):
    from avr.encoders.registry import load_registry

    write_yaml_entry(tmp_path, VALID_ENTRY)
    write_yaml_entry(tmp_path, {**VALID_ENTRY, "name": "other_encoder"})

    registry = load_registry(tmp_path)

    assert sorted(registry) == ["clip_vitl14_336", "other_encoder"]
    assert registry["other_encoder"].model_id == VALID_ENTRY["model_id"]


def test_load_registry_rejects_a_file_whose_stem_differs_from_its_name(tmp_path: Path):
    from avr.encoders.registry import load_registry

    (tmp_path / "misnamed.yaml").write_text(yaml.safe_dump(VALID_ENTRY))

    with pytest.raises(ValueError, match="misnamed.yaml"):
        load_registry(tmp_path)


def test_write_entry_uses_the_schema_key_order_and_round_trips(tmp_path: Path):
    from avr.encoders.registry import EncoderSpec, write_entry

    spec = EncoderSpec.model_validate(VALID_ENTRY)
    path = tmp_path / "clip_vitl14_336.yaml"

    write_entry(path, spec)

    keys = list(yaml.safe_load(path.read_text()))
    assert keys == list(EncoderSpec.model_fields)
    assert EncoderSpec.model_validate(yaml.safe_load(path.read_text())) == spec


def test_write_entry_keeps_the_leading_comment_block(tmp_path: Path):
    from avr.encoders.registry import EncoderSpec, write_entry

    path = tmp_path / "clip_vitl14_336.yaml"
    path.write_text("# why this checkpoint\n# second line\nname: clip_vitl14_336\n")

    write_entry(path, EncoderSpec.model_validate(VALID_ENTRY))

    assert path.read_text().startswith("# why this checkpoint\n# second line\nname: ")
