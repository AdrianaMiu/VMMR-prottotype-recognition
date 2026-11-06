import hashlib
from pathlib import Path

import pytest

from avr.data.split_identity import describe_split_file

FIXTURES = Path(__file__).parents[1] / "fixtures" / "split_files"


def test_describe_split_file_records_hash_counts_and_boundaries():
    identity = describe_split_file(FIXTURES / "tiny_train.txt")

    # sha256sum tests/fixtures/split_files/tiny_train.txt, computed outside this code
    assert identity.sha256 == "b9b9ac6a70c3a50ccb28e46512dfb0d6880a32538487e95b6ddebc8dd50a2616"
    assert identity.line_count == 3
    assert identity.make_count == 2
    assert identity.model_count == 2
    # three lines: head and tail both cover the whole file, no path is stored
    whole = hashlib.sha256(
        b"78/1/2010/aaa.jpg\n78/1/2011/bbb.jpg\n37/1993/2014/ccc.jpg"
    ).hexdigest()
    assert identity.head_sha256 == whole
    assert identity.tail_sha256 == whole
    assert "aaa.jpg" not in identity.model_dump_json()


def test_verification_pairs_count_models_on_both_sides_of_a_pair():
    identity = describe_split_file(FIXTURES / "tiny_pairs.txt")

    assert identity.line_count == 2
    assert identity.make_count == 2
    assert identity.model_count == 3


def test_empty_split_file_is_rejected(tmp_path: Path):
    empty = tmp_path / "empty.txt"
    empty.write_text("")

    with pytest.raises(ValueError, match="empty"):
        describe_split_file(empty)
