"""open_clip names its precisions ``fp32`` / ``fp16`` / ``bf16`` (open_clip 2.32 ``create_model``).

An unrecognised string silently loads fp32 weights, which then meet half-precision inputs in
the forward pass; the loader must translate the torch dtype name. Expected strings are the
documented open_clip option names, not read back from the code under test.
"""

import pytest

torch = pytest.importorskip("torch")

from avr.encoders.verify import open_clip_precision  # noqa: E402


@pytest.mark.parametrize(
    ("dtype", "expected"),
    [(torch.float16, "fp16"), (torch.bfloat16, "bf16"), (torch.float32, "fp32")],
)
def test_torch_dtype_maps_to_the_open_clip_precision_name(dtype, expected: str):
    assert open_clip_precision(dtype) == expected
