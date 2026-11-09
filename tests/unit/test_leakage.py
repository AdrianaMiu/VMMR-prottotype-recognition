"""Leakage audit checks (PROTOCOL.md §7 leakage controls, §23 Data; issue M1-12).

Every expected count below is read off the hand-written frames; nothing is recomputed
with the code under test.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from avr.data.leakage import (
    OFFICIAL_SPLIT_STATUS,
    audit_split,
    check_classes_have_support,
    check_labels_match,
    check_no_group_crosses,
    check_paths_resolve,
    check_regeneration,
    check_split_ids_in_manifest,
    track_a_findings,
    track_b_findings,
)
from avr.data.names import NameTable, NameTables


def rows(*records: tuple[str, str, str, str]) -> pd.DataFrame:
    """(image_id, class_id, duplicate_group_id, partition) rows; source group = duplicate group."""
    frame = pd.DataFrame.from_records(
        list(records), columns=["image_id", "class_id", "duplicate_group_id", "partition"]
    )
    frame["source_group_id"] = frame["duplicate_group_id"]
    frame["validity_status"] = "valid"
    return frame


CLEAN = rows(
    ("a1", "c1", "g1", "support"),
    ("a2", "c1", "g1", "support"),
    ("a3", "c1", "g3", "query"),
    ("b1", "c2", "g4", "support"),
    ("b2", "c2", "g5", "query"),
)


def test_group_check_passes_when_every_group_stays_in_one_partition():
    finding = check_no_group_crosses(CLEAN, "duplicate_group_id", "partition")

    assert finding.passed
    assert finding.must_pass
    assert finding.status == "passed"
    assert finding.counts == {"groups": 4, "crossing_groups": 0, "crossing_images": 0}
    assert finding.offending_ids == []


def test_planted_cross_split_duplicate_fails_the_check():
    planted = pd.concat([CLEAN, rows(("a4", "c1", "g1", "query"))], ignore_index=True)

    finding = check_no_group_crosses(planted, "duplicate_group_id", "partition")

    assert not finding.passed
    assert finding.status == "failed"
    # g1 now holds a1, a2 (support) and a4 (query)
    assert finding.counts == {"groups": 4, "crossing_groups": 1, "crossing_images": 3}
    assert finding.offending_ids == ["g1"]


def test_official_split_crossing_is_reported_but_does_not_fail():
    planted = pd.concat([CLEAN, rows(("a4", "c1", "g1", "query"))], ignore_index=True)

    finding = check_no_group_crosses(
        planted, "duplicate_group_id", "partition", must_pass=False, status=OFFICIAL_SPLIT_STATUS
    )

    assert not finding.passed
    assert not finding.must_pass
    assert finding.status == "reported, official split not altered"
    assert finding.counts["crossing_groups"] == 1


def test_group_check_counts_crossing_groups_that_span_more_than_one_class():
    # g1 crosses within class c1; g4 crosses between classes c2 and c3
    planted = pd.concat(
        [CLEAN, rows(("a4", "c1", "g1", "query"), ("d1", "c3", "g4", "query"))],
        ignore_index=True,
    )

    finding = check_no_group_crosses(
        planted, "duplicate_group_id", "partition", class_column="class_id"
    )

    assert finding.counts == {
        "groups": 4,
        "crossing_groups": 2,
        "crossing_images": 5,
        "crossing_groups_multi_class": 1,
    }
    assert finding.offending_ids == ["g1", "g4"]


def test_offending_ids_are_capped_at_fifty():
    many = rows(*[(f"s{i}", "c", f"g{i:03d}", "support") for i in range(60)])
    many = pd.concat(
        [many, rows(*[(f"q{i}", "c", f"g{i:03d}", "query") for i in range(60)])],
        ignore_index=True,
    )

    finding = check_no_group_crosses(many, "duplicate_group_id", "partition")

    assert finding.counts["crossing_groups"] == 60
    assert len(finding.offending_ids) == 50
    assert finding.offending_ids[0] == "g000"


# --- every target class has support ----------------------------------------------------


def test_every_query_class_with_support_passes():
    finding = check_classes_have_support(CLEAN, "partition", "support", "query")

    assert finding.passed
    assert finding.counts == {"target_classes": 2, "classes_without_support": 0}


def test_query_class_without_support_fails_and_names_the_class():
    orphan = pd.concat([CLEAN, rows(("z1", "c9", "g9", "query"))], ignore_index=True)

    finding = check_classes_have_support(orphan, "partition", "support", "query")

    assert not finding.passed
    assert finding.counts == {"target_classes": 3, "classes_without_support": 1}
    assert finding.offending_ids == ["c9"]


def test_excluded_support_images_do_not_count_as_support():
    frame = CLEAN.copy()
    frame.loc[frame.image_id == "b1", "validity_status"] = "excluded"

    finding = check_classes_have_support(frame, "partition", "support", "query")

    assert finding.offending_ids == ["c2"]


# --- split ids resolve in the manifest -------------------------------------------------


def manifest(*records: tuple[str, str]) -> pd.DataFrame:
    return pd.DataFrame.from_records(list(records), columns=["image_id", "sha256"])


def test_split_ids_resolve_when_every_id_and_hash_is_in_the_manifest():
    split = manifest(("a1", "h1"), ("a2", "h2"))

    finding = check_split_ids_in_manifest(split, manifest(("a1", "h1"), ("a2", "h2"), ("a3", "h3")))

    assert finding.passed
    assert finding.counts == {"split_rows": 2, "unresolved": 0, "sha256_mismatch": 0}


def test_split_ids_fail_on_unknown_id_or_changed_hash():
    split = manifest(("a1", "h1"), ("a2", "XX"), ("a9", "h9"))

    finding = check_split_ids_in_manifest(split, manifest(("a1", "h1"), ("a2", "h2")))

    assert not finding.passed
    assert finding.counts == {"split_rows": 3, "unresolved": 1, "sha256_mismatch": 1}
    assert finding.offending_ids == ["a2", "a9"]


# --- manifest paths resolve (dataset-gated) -------------------------------------------


def test_paths_resolve_when_every_id_has_an_existing_file(tmp_path: Path):
    (tmp_path / "image").mkdir()
    (tmp_path / "image" / "x.jpg").write_bytes(b"x")
    paths = pd.DataFrame({"image_id": ["a1"], "relative_path": ["image/x.jpg"]})

    finding = check_paths_resolve(pd.DataFrame({"image_id": ["a1"]}), paths, tmp_path)

    assert finding.passed
    assert finding.counts == {"manifest_rows": 1, "unresolved": 0, "missing_files": 0}


def test_paths_fail_on_id_without_path_and_on_missing_file(tmp_path: Path):
    paths = pd.DataFrame({"image_id": ["a1"], "relative_path": ["image/gone.jpg"]})

    finding = check_paths_resolve(pd.DataFrame({"image_id": ["a1", "a2"]}), paths, tmp_path)

    assert not finding.passed
    assert finding.counts == {"manifest_rows": 2, "unresolved": 1, "missing_files": 1}
    assert finding.offending_ids == ["a1", "a2"]


# --- labels match CompCars metadata (dataset-gated) -----------------------------------


def names() -> NameTables:
    return NameTables(
        makes={1: NameTable(name_raw="ABT", name="ABT")},
        models={1: NameTable(name_raw=" Hector", name="Hector")},
        make_slots=1,
        model_slots=2,
        missing_make_ids=(),
        missing_model_ids=(2,),
    )


def labelled(**overrides) -> pd.DataFrame:
    row = dict(
        image_id="a1",
        make_id=1,
        make_name="ABT",
        model_id=1,
        model_name="Hector",
        year=2010,
        relative_path="image/1/1/2010/aaa.jpg",
    )
    row.update(overrides)
    return pd.DataFrame([row])


def test_labels_match_when_ids_names_and_path_components_agree():
    frame = labelled()

    finding = check_labels_match(frame.drop(columns="relative_path"), frame, names())

    assert finding.passed
    assert finding.counts == {"manifest_rows": 1, "mismatched": 0}


def test_labels_fail_when_a_name_or_a_path_component_disagrees():
    frame = pd.concat(
        [
            labelled(),
            labelled(image_id="a2", model_name="Hector GT"),
            labelled(image_id="a3", relative_path="image/1/2/2010/ccc.jpg"),
            labelled(image_id="a4", year=None, relative_path="image/1/1/unknown/ddd.jpg"),
            labelled(image_id="a5", year=2011),
        ],
        ignore_index=True,
    )

    finding = check_labels_match(frame.drop(columns="relative_path"), frame, names())

    assert finding.counts == {"manifest_rows": 5, "mismatched": 3}
    assert finding.offending_ids == ["a2", "a3", "a5"]


# --- regeneration byte-identical --------------------------------------------------------


def test_regeneration_passes_when_the_rebuilt_digest_equals_the_sidecar(tmp_path: Path):
    rebuilt = tmp_path / "x.parquet"
    rebuilt.write_bytes(b"abc")
    sidecar = tmp_path / "x.sha256"
    # sha256sum of the bytes "abc"
    sidecar.write_text(
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad  x.parquet\n"
    )

    finding = check_regeneration("x", rebuilt, sidecar)

    assert finding.passed
    assert finding.counts == {"compared": 1, "mismatched": 0}


def test_regeneration_fails_when_the_rebuilt_bytes_differ(tmp_path: Path):
    rebuilt = tmp_path / "x.parquet"
    rebuilt.write_bytes(b"abd")
    sidecar = tmp_path / "x.sha256"
    sidecar.write_text(
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad  x.parquet\n"
    )

    finding = check_regeneration("x", rebuilt, sidecar)

    assert not finding.passed
    assert finding.offending_ids == ["x"]


# --- the runner ---------------------------------------------------------------------------


def test_audit_split_fails_only_on_must_pass_findings_and_writes_the_report(tmp_path: Path):
    reported = check_no_group_crosses(
        pd.concat([CLEAN, rows(("a4", "c1", "g1", "query"))], ignore_index=True),
        "duplicate_group_id",
        "partition",
        must_pass=False,
        status=OFFICIAL_SPLIT_STATUS,
    )
    ok = check_classes_have_support(CLEAN, "partition", "support", "query")

    result = audit_split("demo", [reported, ok], tmp_path, inputs={"split sha256": "deadbeef"})

    assert result.passed
    assert result.report_path == tmp_path / "demo.md"
    text = result.report_path.read_text()
    assert "reported, official split not altered" in text
    assert "deadbeef" in text
    assert "g1" in text

    failed = audit_split(
        "demo2",
        [
            ok,
            check_no_group_crosses(
                pd.concat([CLEAN, rows(("a4", "c1", "g1", "query"))], ignore_index=True),
                "duplicate_group_id",
                "partition",
            ),
        ],
        tmp_path,
    )
    assert not failed.passed
    assert failed.failed_checks == ["no_duplicate_group_id_crosses_partition"]


# --- Track A assembly: the official crossing is reported, the nested partition must hold --


def public_manifest(*records: tuple[str, str, int, int, str, str]) -> pd.DataFrame:
    """(image_id, sha256, make_id, model_id, duplicate_group_id, validity_status)."""
    frame = pd.DataFrame.from_records(
        list(records),
        columns=[
            "image_id",
            "sha256",
            "make_id",
            "model_id",
            "duplicate_group_id",
            "validity_status",
        ],
    )
    frame["source_group_id"] = frame["duplicate_group_id"]
    return frame


MANIFEST = public_manifest(
    ("a1", "h1", 1, 1, "a1", "valid"),
    ("a2", "h2", 1, 1, "a1", "valid"),  # exact duplicate of a1, lands on the query side
    ("a3", "h3", 1, 1, "a3", "valid"),
    ("a4", "h4", 1, 1, "a4", "valid"),
    ("b1", "h5", 2, 2, "b1", "valid"),
    ("b2", "h6", 2, 2, "b2", "valid"),
    ("b3", "h7", 2, 2, "b3", "excluded"),
)


def track_a_split(*records: tuple[str, str, str | None]) -> pd.DataFrame:
    frame = pd.DataFrame.from_records(
        list(records), columns=["image_id", "split_track_a", "track_a_validation"]
    )
    return frame.merge(MANIFEST[["image_id", "sha256", "make_id", "model_id"]], on="image_id")


def test_track_a_official_crossing_is_reported_while_the_audit_passes():
    split = track_a_split(
        ("a1", "support", "fit"),
        ("a2", "query", None),
        ("a3", "support", "validation"),
        ("b1", "support", "fit"),
        ("b2", "query", None),
        ("b3", "query", None),
    )

    findings = track_a_findings(MANIFEST, split)
    by_check = {f.check: f for f in findings}

    official = by_check["no_duplicate_group_id_crosses_support_query"]
    assert not official.passed and not official.must_pass
    assert official.status == OFFICIAL_SPLIT_STATUS
    assert official.counts == {
        "groups": 5,
        "crossing_groups": 1,
        "crossing_images": 2,
        "crossing_groups_multi_class": 0,
    }
    assert by_check["no_duplicate_group_id_crosses_fit_validation"].passed
    assert by_check["no_source_group_id_crosses_fit_validation"].passed
    assert by_check["every_query_class_has_support"].counts == {
        "target_classes": 2,
        "classes_without_support": 0,
    }
    assert by_check["split_ids_resolve_in_manifest"].counts["split_rows"] == 6
    assert all(f.passed for f in findings if f.must_pass)


def test_track_a_fit_validation_crossing_fails():
    split = track_a_split(
        ("a1", "support", "fit"),
        ("a2", "support", "validation"),  # same duplicate group as a1
        ("a3", "query", None),
    )

    findings = track_a_findings(MANIFEST, split)
    by_check = {f.check: f for f in findings}

    nested = by_check["no_duplicate_group_id_crosses_fit_validation"]
    assert not nested.passed and nested.must_pass
    assert nested.offending_ids == ["a1"]


# --- Track B assembly ---------------------------------------------------------------------


def track_b_split(*records: tuple[str, str, str]) -> pd.DataFrame:
    frame = pd.DataFrame.from_records(list(records), columns=["image_id", "class_id", "partition"])
    return frame.merge(MANIFEST[["image_id", "sha256", "make_id", "model_id"]], on="image_id")


def test_track_b_findings_fail_on_a_test_class_without_support_and_on_a_crossing_group():
    split = track_b_split(
        ("a1", "1/1/2010", "support"),
        ("a2", "1/1/2010", "test"),  # a1's duplicate group crosses into test
        ("a3", "1/1/2010", "validation"),
        ("b1", "2/2/2012", "test"),  # no support image for this class
    )

    findings = track_b_findings(MANIFEST, split)
    by_check = {f.check: f for f in findings}

    assert by_check["no_duplicate_group_id_crosses_partition"].offending_ids == ["a1"]
    assert by_check["no_source_group_id_crosses_partition"].offending_ids == ["a1"]
    assert by_check["every_test_class_has_support"].offending_ids == ["2/2/2012"]
    assert all(f.must_pass for f in findings)
