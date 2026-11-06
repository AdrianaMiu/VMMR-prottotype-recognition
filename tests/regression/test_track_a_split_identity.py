"""The official Track A split files must never change under us (PROTOCOL.md §5 Track A).

The file comparison runs wherever the CompCars release is present (dataset root from
configs/datasets, COMPCARS_ROOT overrides); CI has no release and skips it. The second
test checks the published record itself and runs everywhere.
"""

from pathlib import Path

import pytest

from avr.config import CONFIG_DIR, load_config
from avr.data.split_identity import (
    OFFICIAL_SPLIT_FILES,
    TRACK_A_LABEL,
    SplitIdentityRecord,
    describe_split_file,
)

REPO_ROOT = CONFIG_DIR.parent
RECORD = REPO_ROOT / "data" / "manifests" / "track_a_split_identity.json"
ROOT = (REPO_ROOT / load_config().dataset.root).resolve()


def published_record() -> SplitIdentityRecord:
    return SplitIdentityRecord.model_validate_json(RECORD.read_text())


@pytest.mark.skipif(not ROOT.is_dir(), reason="CompCars release not present on this machine")
def test_local_split_files_match_the_published_record():
    for relative_path, recorded in published_record().files.items():
        assert describe_split_file(ROOT / Path(relative_path)) == recorded, relative_path


def test_record_names_the_cvpr_half_half_setting():
    record = published_record()
    official = [record.files[f] for f in OFFICIAL_SPLIT_FILES]

    assert sum(f.line_count for f in official) == 30955
    assert all(f.model_count == 431 for f in official)
    assert record.setting == "cvpr2015_half_half"
    assert record.label == TRACK_A_LABEL
