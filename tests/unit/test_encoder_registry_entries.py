"""The checked-in registry under configs/encoders validates (PROTOCOL.md §9, ADR-0003).

Expected identifiers and dimensions are the protocol's: §9 primary table, the ADR-0003
SigLIP2 override and the M1 gate candidate. ``native_dim`` must come from a verified
run, never from transformer width.
"""

from __future__ import annotations

import pytest

from avr.encoders.registry import ENCODERS_DIR, load_registry

EXPECTED = {
    "dinov3_vitl16": ("facebook/dinov3-vitl16-pretrain-lvd1689m", 1024),
    "dinov3_vitb16": ("facebook/dinov3-vitb16-pretrain-lvd1689m", 768),
    "siglip2_so400m_naflex": ("google/siglip2-so400m-patch16-naflex", 1152),
    "clip_vitl14_336": ("openai/clip-vit-large-patch14-336", 768),
    "openclip_vith14_laion2b": ("laion/CLIP-ViT-H-14-laion2B-s32B-b79K", 1024),
}


@pytest.fixture(scope="module")
def registry():
    return load_registry(ENCODERS_DIR)


def test_registry_has_exactly_the_protocol_entries(registry):
    assert set(registry) == set(EXPECTED)


@pytest.mark.parametrize(("name", "expected"), EXPECTED.items())
def test_entry_matches_the_protocol_identifier_and_dimension(registry, name, expected):
    model_id, native_dim = expected

    assert registry[name].model_id == model_id
    assert registry[name].native_dim == native_dim


def test_every_entry_is_verified_and_its_shape_ends_in_native_dim(registry):
    for spec in registry.values():
        assert spec.verified_output_shape is not None, spec.name
        assert spec.verified_output_shape[-1] == spec.native_dim, spec.name
        assert spec.checkpoint_sha256 is not None, spec.name
        assert spec.parameter_count is not None, spec.name


def test_siglip2_naflex_pins_max_num_patches(registry):
    assert registry["siglip2_so400m_naflex"].preprocessing.model_dump()["max_num_patches"] == 256
    assert registry["siglip2_so400m_naflex"].pooling == "attention_pool"


def test_open_clip_entry_uses_the_pretrained_tag_as_revision(registry):
    spec = registry["openclip_vith14_laion2b"]

    assert spec.source == "open_clip"
    assert spec.revision == "laion2b_s32b_b79k"
