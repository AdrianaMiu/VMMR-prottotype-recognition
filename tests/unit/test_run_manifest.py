import pytest
from pydantic import ValidationError

from avr.tracking.manifest import REQUIRED_FIELDS, RunManifest

# PROTOCOL.md §17, copied by hand from the fenced list
PROTOCOL_17_FIELDS = [
    "run_id",
    "git_commit",
    "dirty_worktree_flag",
    "dataset_protocol",
    "dataset_manifest_hash",
    "split_manifest_hash",
    "generation_mapping_hash",
    "encoder_id",
    "encoder_revision",
    "encoder_checksum",
    "preprocessing_id",
    "dimension_transform_id",
    "prototype_method",
    "support_k",
    "seed",
    "hardware",
    "software_versions",
    "start_time",
    "end_time",
]


def test_manifest_fields_are_exactly_the_protocol_17_list_in_order():
    assert list(RunManifest.model_fields) == PROTOCOL_17_FIELDS


def test_every_field_except_end_time_is_required_for_a_metric():
    assert REQUIRED_FIELDS == tuple(f for f in PROTOCOL_17_FIELDS if f != "end_time")


def test_empty_manifest_is_constructible_and_reports_missing_fields():
    manifest = RunManifest(run_id="r1", seed=73)

    missing = manifest.missing_fields()

    assert "run_id" not in missing
    assert "seed" not in missing
    assert "git_commit" in missing
    assert "end_time" not in missing
    assert not manifest.is_complete()


def test_unknown_field_is_rejected():
    with pytest.raises(ValidationError):
        RunManifest(run_id="r1", extra_field=1)


def test_capture_git_commit_matches_rev_parse_head_of_a_fresh_repo(tmp_path):
    import subprocess

    from avr.tracking.manifest import capture_dirty_worktree_flag, capture_git_commit

    subprocess.run(["git", "init", "-q", tmp_path], check=True)
    (tmp_path / "a.txt").write_text("a\n")
    env = {
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@t",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@t",
        "HOME": str(tmp_path),
    }
    subprocess.run(["git", "-C", tmp_path, "add", "a.txt"], check=True, env=env)
    subprocess.run(["git", "-C", tmp_path, "commit", "-q", "-m", "x"], check=True, env=env)
    head = subprocess.run(
        ["git", "-C", tmp_path, "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()

    assert capture_git_commit(tmp_path) == head
    assert len(head) == 40
    assert capture_dirty_worktree_flag(tmp_path) is False

    (tmp_path / "a.txt").write_text("changed\n")
    assert capture_dirty_worktree_flag(tmp_path) is True


def test_software_versions_name_python_numpy_pandas_and_avr():
    import platform

    from avr.tracking.manifest import capture_hardware, capture_software_versions

    versions = capture_software_versions()
    assert versions["python"] == platform.python_version()
    assert versions["avr"] == "0.1.0"
    assert set(versions) >= {"python", "numpy", "pandas", "avr"}

    hardware = capture_hardware()
    assert set(hardware) >= {"cpu", "gpu"}


def test_software_versions_record_torch_without_importing_it():
    import subprocess
    import sys

    program = (
        "import sys\n"
        "from avr.tracking.manifest import capture_software_versions\n"
        "versions = capture_software_versions()\n"
        "assert 'torch' not in sys.modules, 'torch was imported'\n"
        "print('torch' in versions)\n"
    )
    result = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
