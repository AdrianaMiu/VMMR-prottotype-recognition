"""The official Track A split files must never change under us (PROTOCOL.md §5 Track A).

Runs only where the CompCars release is present (COMPCARS_ROOT or ../compcar);
CI skips it. The stored record is the published split identity.
"""

import os
from pathlib import Path

import pytest

from avr.data.split_identity import SplitIdentityRecord, describe_split_file

RECORD = Path(__file__).parents[2] / "data" / "manifests" / "track_a_split_identity.json"
ROOT = Path(os.environ.get("COMPCARS_ROOT", Path(__file__).parents[2] / ".." / "compcar"))


@pytest.mark.skipif(not ROOT.is_dir(), reason="CompCars release not present on this machine")
def test_local_split_files_match_the_published_record():
    record = SplitIdentityRecord.model_validate_json(RECORD.read_text())

    for relative_path, recorded in record.files.items():
        assert describe_split_file(ROOT / relative_path) == recorded, relative_path


def test_record_names_the_cvpr_half_half_setting():
    record = SplitIdentityRecord.model_validate_json(RECORD.read_text())
    classification = {k: v for k, v in record.files.items() if "classification" in k}

    assert sum(v.line_count for v in classification.values()) == 30955
    assert all(v.model_count == 431 for v in classification.values())
    assert record.setting == "cvpr2015_half_half"
