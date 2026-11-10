"""Search backends behind the service (PROTOCOL.md §12, §23 "Qdrant and exact-search parity")."""

import numpy as np
import pytest

from avr.service.search import EmbeddedQdrantSearch, ExactSearch, SearchBackend

from .conftest import FIVE_COLLECTION

# query -> expected top-3 class ids, read off the axis-aligned prototypes by hand
CASES = [
    ([0.9, 0.3, 0.1, 0.0, 0.0], ["c0", "c1", "c2"]),
    ([0.0, 0.0, 0.2, 0.7, 0.4], ["c3", "c4", "c2"]),
    ([0.1, 0.1, 0.1, 0.1, 0.8], ["c4", "c0", "c1"]),  # tie among c0..c3 -> class-id order
]


def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


@pytest.fixture
def backends(five_prototype_files):
    exact = ExactSearch.from_parquet(five_prototype_files["parquet"])
    qdrant = EmbeddedQdrantSearch(five_prototype_files["qdrant"], FIVE_COLLECTION)
    return exact, qdrant


def test_both_backends_satisfy_the_protocol(backends):
    for backend in backends:
        assert isinstance(backend, SearchBackend)


@pytest.mark.parametrize("query,expected", CASES)
def test_exact_top_k_matches_hand_ranking(backends, query, expected):
    exact, _ = backends
    hits = exact.search(_unit(query)[None, :], k=3)

    assert [h.class_id for h in hits[0]] == expected


@pytest.mark.parametrize("query,expected", CASES[:2])
def test_qdrant_and_exact_agree_on_top_k_and_scores(backends, query, expected):
    exact, qdrant = backends
    e = exact.search(_unit(query)[None, :], k=3)[0]
    q = qdrant.search(_unit(query)[None, :], k=3)[0]

    assert [h.class_id for h in q] == [h.class_id for h in e] == expected
    assert np.allclose([h.score for h in q], [h.score for h in e], atol=1e-5)


def test_hits_carry_the_hierarchy_payload_with_native_types(backends):
    exact, qdrant = backends
    for backend in backends:
        hit = backend.search(_unit([0, 0, 0, 1, 0])[None, :], k=1)[0][0]
        assert hit.class_id == "c3"
        assert hit.payload["make_name"] == "B"
        assert hit.payload["model_name"] == "d"
        assert hit.payload["generation_id"] == "g3"
        assert hit.payload["year"] == 2013
    hit = qdrant.search(_unit([0, 0, 1, 0, 0])[None, :], k=1)[0][0]
    assert hit.payload["year"] is None and hit.payload["generation_id"] is None


def test_batched_queries_return_one_row_per_query(backends):
    exact, qdrant = backends
    queries = np.stack([_unit(c[0]) for c in CASES])
    assert len(exact.search(queries, k=2)) == 3
    assert len(qdrant.search(queries, k=2)) == 3
    assert exact.search(np.zeros((0, 5), np.float32), k=2) == []


def test_exact_search_reports_size_and_dimension(backends):
    exact, _ = backends
    assert exact.size == 5 and exact.dimension == 5


def test_embedded_backends_can_share_one_client_and_close_it(five_prototype_files):
    first = EmbeddedQdrantSearch(five_prototype_files["qdrant"], FIVE_COLLECTION)
    second = EmbeddedQdrantSearch(
        five_prototype_files["qdrant"], FIVE_COLLECTION, client=first.client
    )
    assert second.client is first.client
    assert [h.class_id for h in second.search(_unit([0, 1, 0, 0, 0])[None, :], 1)[0]] == ["c1"]
    first.close()
    # a fresh client can open the store again once the shared one is closed
    third = EmbeddedQdrantSearch(five_prototype_files["qdrant"], FIVE_COLLECTION)
    assert [h.class_id for h in third.search(_unit([0, 0, 1, 0, 0])[None, :], 1)[0]] == ["c2"]
    third.close()
