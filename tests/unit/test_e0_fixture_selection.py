"""Selection rule of the E0 ten-class fixture (PROTOCOL.md §14 "Stage E0").

The rows below are synthetic; expected values are worked out by hand from the rule in
``avr.pipeline.e0.select_fixture_split``: the classes with the most valid images, then
per class a seeded shuffle of the image ids, at most one image per duplicate group, the
first ``n_support`` as support and the next ``n_query`` as query.
"""

from __future__ import annotations

import pytest

from avr.pipeline.e0 import FixtureSplit, class_id_of, select_fixture_split


def row(image_id, make, model, group=None, valid=True):
    return {
        "image_id": image_id,
        "make_id": make,
        "model_id": model,
        "duplicate_group_id": group or image_id,
        "validity_status": "valid" if valid else "excluded",
    }


def rows_for(make, model, count, prefix):
    return [row(f"{prefix}{i:02d}", make, model) for i in range(count)]


def test_class_id_is_zero_padded_make_and_model():
    assert class_id_of(1, 7) == "001_0007"
    assert class_id_of(123, 2004) == "123_2004"


def test_picks_the_classes_with_most_valid_images_and_fills_support_then_query():
    rows = (
        rows_for(1, 1, 4, "a")  # 4 valid: too few for 2 + 1
        + rows_for(2, 2, 6, "b")
        + rows_for(3, 3, 5, "c")
        + rows_for(4, 4, 3, "d")
        + [row("x1", 5, 5, valid=False)] * 10  # excluded rows never count
    )

    split = select_fixture_split(rows, n_classes=2, n_support=2, n_query=1, seed=73)

    assert isinstance(split, FixtureSplit)
    assert split.class_ids == ["002_0002", "003_0003"]
    assert all(len(split.support[c]) == 2 for c in split.class_ids)
    assert all(len(split.query[c]) == 1 for c in split.class_ids)
    chosen = [i for c in split.class_ids for i in split.support[c] + split.query[c]]
    assert len(set(chosen)) == 6
    assert all(i[0] in "bc" for i in chosen)


def test_at_most_one_image_per_duplicate_group_enters_the_fixture():
    rows = [row(f"g{i}", 1, 1, group="g0") for i in range(6)] + rows_for(1, 1, 3, "s")

    split = select_fixture_split(rows, n_classes=1, n_support=2, n_query=1, seed=73)

    chosen = split.support["001_0001"] + split.query["001_0001"]
    assert sum(i.startswith("g") for i in chosen) == 1
    assert sum(i.startswith("s") for i in chosen) == 2


def test_selection_is_deterministic_for_a_seed_and_changes_with_it():
    rows = rows_for(1, 1, 30, "a") + rows_for(2, 2, 30, "b")

    first = select_fixture_split(rows, n_classes=2, n_support=5, n_query=3, seed=73)
    again = select_fixture_split(rows, n_classes=2, n_support=5, n_query=3, seed=73)
    other = select_fixture_split(rows, n_classes=2, n_support=5, n_query=3, seed=74)

    assert first == again
    assert first != other
    assert first.support["001_0001"] == sorted(first.support["001_0001"])


def test_class_with_too_few_images_raises():
    rows = rows_for(1, 1, 7, "a")
    with pytest.raises(ValueError, match="fewer than"):
        select_fixture_split(rows, n_classes=1, n_support=5, n_query=3, seed=73)
