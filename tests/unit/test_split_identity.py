from pathlib import Path

from avr.data.split_identity import describe_split_file

FIXTURE = Path(__file__).parents[1] / "fixtures" / "split_files" / "tiny_train.txt"


def test_describe_split_file_records_hash_counts_and_boundaries():
    identity = describe_split_file(FIXTURE)

    # sha256sum tests/fixtures/split_files/tiny_train.txt, computed outside this code
    assert identity.sha256 == "b9b9ac6a70c3a50ccb28e46512dfb0d6880a32538487e95b6ddebc8dd50a2616"
    assert identity.line_count == 3
    assert identity.make_count == 2
    assert identity.model_count == 2
    assert identity.first_line == "78/1/2010/aaa.jpg"
    assert identity.last_line == "37/1993/2014/ccc.jpg"
