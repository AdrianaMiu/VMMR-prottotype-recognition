"""Real forward pass per registry checkpoint (PROTOCOL.md §9). Needs the ``gpu`` extra,
the checkpoints (network or Hugging Face cache) and several GB of RAM, so it only runs
with AVR_RUN_ENCODER_VERIFY=1. Expected dimensions are the protocol's, not the model's.
"""

from __future__ import annotations

import os

import pytest
from PIL import Image

from avr.encoders.registry import ENCODERS_DIR, load_registry

pytestmark = pytest.mark.skipif(
    os.environ.get("AVR_RUN_ENCODER_VERIFY") != "1",
    reason="set AVR_RUN_ENCODER_VERIFY=1 to run the checkpoint verification",
)

EXPECTED_DIM = {
    "dinov3_vitl16": 1024,
    "dinov3_vitb16": 768,
    "siglip2_so400m_naflex": 1152,
    "clip_vitl14_336": 768,
    "openclip_vith14_laion2b": 1024,
}


@pytest.mark.parametrize("name", sorted(EXPECTED_DIM))
def test_one_image_yields_the_protocol_dimension(name: str):
    from avr.encoders.verify import verify_entry

    spec = load_registry(ENCODERS_DIR)[name]
    image = Image.new("RGB", (640, 480), (90, 120, 150))

    result = verify_entry(spec, image, device=os.environ.get("AVR_DEVICE", "cpu"))

    assert result.output_shape == [1, EXPECTED_DIM[name]]
    assert result.checkpoint_sha256 == spec.checkpoint_sha256
