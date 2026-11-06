"""M0 global class mean prototypes (PROTOCOL.md §11 "M0", §23 "Prototypes and ranking")."""

import numpy as np
import pytest

from avr.prototypes.m0 import build_global_mean_prototypes

INV_SQRT2 = 1 / np.sqrt(2)


def test_prototype_is_renormalised_mean_of_support_embeddings():
    # Class "x" has two support embeddings (1,0) and (0,1): mean (0.5,0.5), norm 0.5*sqrt(2),
    # re-normalised (1/sqrt2, 1/sqrt2). Class "w" has one support embedding (0,1).
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]], dtype=np.float32)
    prototypes, class_ids = build_global_mean_prototypes(
        embeddings, class_ids=["x", "x", "w"], image_ids=["i1", "i2", "i3"]
    )
    assert class_ids == ["w", "x"]
    np.testing.assert_allclose(prototypes, [[0.0, 1.0], [INV_SQRT2, INV_SQRT2]], atol=1e-7)
    assert prototypes.dtype == np.float32


def test_prototype_order_is_stable_under_input_permutation():
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]], dtype=np.float32)
    forward, ids_forward = build_global_mean_prototypes(
        embeddings, class_ids=["x", "x", "w"], image_ids=["i1", "i2", "i3"]
    )
    backward, ids_backward = build_global_mean_prototypes(
        embeddings[::-1], class_ids=["w", "x", "x"], image_ids=["i3", "i2", "i1"]
    )
    assert ids_forward == ids_backward
    np.testing.assert_array_equal(forward, backward)


def test_query_image_in_support_rows_raises():
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    with pytest.raises(ValueError, match="query"):
        build_global_mean_prototypes(
            embeddings, class_ids=["x", "x"], image_ids=["i1", "q7"], query_image_ids={"q7"}
        )


def test_mismatched_lengths_raise():
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    with pytest.raises(ValueError):
        build_global_mean_prototypes(embeddings, class_ids=["x"], image_ids=["i1", "i2"])
