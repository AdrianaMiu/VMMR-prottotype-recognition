"""Leakage audit over the published splits (PROTOCOL.md §7 leakage controls, §23 Data).

Every check is a pure function over in-memory tables that returns a ``Finding``: the
check name, whether it passed, the counts a reader needs and at most ``ID_CAP``
offending identifiers. Checks that must hold carry ``must_pass = True``; a finding with
``must_pass = False`` is reported in the audit without failing it. The one such finding
is the official Track A split, where duplicate groups span the distributed support and
query halves: it is a property of the official files, so it is reported and the split is
left as distributed (``OFFICIAL_SPLIT_STATUS``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd
from pydantic import BaseModel

from avr.data.manifest import IMAGE_DIR, MAX_YEAR, MIN_YEAR, UNKNOWN_YEAR_FOLDER
from avr.data.names import NameTables
from avr.io import read_sidecar_sha256, sha256_of_file

ID_CAP = 50
OFFICIAL_SPLIT_STATUS = "reported, official split not altered"


class Finding(BaseModel):
    """Outcome of one check; ``offending_ids`` holds at most ``ID_CAP`` sorted identifiers."""

    check: str
    passed: bool
    must_pass: bool = True
    status: str
    counts: dict[str, int]
    offending_ids: list[str]
    note: str = ""


def _finding(
    check: str,
    counts: dict[str, int],
    offending: list[str],
    must_pass: bool = True,
    status: str | None = None,
    note: str = "",
) -> Finding:
    passed = not offending
    return Finding(
        check=check,
        passed=passed,
        must_pass=must_pass,
        status=status or ("passed" if passed else "failed"),
        counts=counts,
        offending_ids=sorted(offending)[:ID_CAP],
        note=note,
    )


def check_no_group_crosses(
    rows: pd.DataFrame,
    group_column: str,
    partition_column: str,
    must_pass: bool = True,
    status: str | None = None,
    check: str | None = None,
    class_column: str | None = None,
) -> Finding:
    """No value of ``group_column`` may appear in more than one ``partition_column`` value.

    With ``class_column`` the finding also counts the crossing groups whose members belong
    to more than one class: a split that assigns groups class by class can only leak
    through those.
    """
    present = rows[rows[partition_column].notna() & rows[group_column].notna()]
    partitions_per_group = present.groupby(group_column)[partition_column].nunique()
    crossing = partitions_per_group.index[partitions_per_group > 1]
    members = present[present[group_column].isin(crossing)]
    counts = {
        "groups": int(len(partitions_per_group)),
        "crossing_groups": int(len(crossing)),
        "crossing_images": int(len(members)),
    }
    if class_column is not None:
        classes_per_group = members.groupby(group_column)[class_column].nunique()
        counts["crossing_groups_multi_class"] = int((classes_per_group > 1).sum())
    return _finding(
        check or f"no_{group_column}_crosses_{partition_column}",
        counts,
        [str(g) for g in crossing],
        must_pass=must_pass,
        status=status if crossing.size else None,
    )


def check_classes_have_support(
    rows: pd.DataFrame,
    partition_column: str,
    support_label: str,
    target_label: str,
    class_column: str = "class_id",
) -> Finding:
    """Every class with a valid ``target_label`` image has at least one valid support image."""
    valid = rows[rows["validity_status"] == "valid"]
    target_classes = set(valid.loc[valid[partition_column] == target_label, class_column])
    support_classes = set(valid.loc[valid[partition_column] == support_label, class_column])
    without = target_classes - support_classes
    counts = {"target_classes": len(target_classes), "classes_without_support": len(without)}
    return _finding(
        f"every_{target_label}_class_has_{support_label}", counts, [str(c) for c in without]
    )


def check_split_ids_in_manifest(split: pd.DataFrame, manifest: pd.DataFrame) -> Finding:
    """Every split row names a manifest ``image_id`` and carries the manifest ``sha256``."""
    joined = split[["image_id", "sha256"]].merge(
        manifest[["image_id", "sha256"]], on="image_id", how="left", suffixes=("", "_manifest")
    )
    unresolved = joined["sha256_manifest"].isna()
    mismatch = ~unresolved & (joined["sha256"] != joined["sha256_manifest"])
    counts = {
        "split_rows": int(len(split)),
        "unresolved": int(unresolved.sum()),
        "sha256_mismatch": int(mismatch.sum()),
    }
    return _finding(
        "split_ids_resolve_in_manifest", counts, list(joined.image_id[unresolved | mismatch])
    )


def check_paths_resolve(manifest: pd.DataFrame, paths: pd.DataFrame, root: Path) -> Finding:
    """Every manifest ``image_id`` has a private path and the file exists under ``root``."""
    joined = manifest[["image_id"]].merge(paths, on="image_id", how="left")
    unresolved = joined["relative_path"].isna()
    exists = [
        (root / rel).is_file() if isinstance(rel, str) else False for rel in joined.relative_path
    ]
    missing = ~unresolved & ~pd.Series(exists, index=joined.index)
    counts = {
        "manifest_rows": int(len(manifest)),
        "unresolved": int(unresolved.sum()),
        "missing_files": int(missing.sum()),
    }
    return _finding("manifest_paths_resolve", counts, list(joined.image_id[unresolved | missing]))


def _expected_year(folder: str) -> int | None:
    """The year the manifest records for a folder: null for ``unknown`` and typo folders."""
    if folder == UNKNOWN_YEAR_FOLDER or not folder.isdigit():
        return None
    return int(folder) if MIN_YEAR <= int(folder) <= MAX_YEAR else None


def check_labels_match(manifest: pd.DataFrame, paths: pd.DataFrame, names: NameTables) -> Finding:
    """``make_id``/``model_id``/``year`` agree with the path and the names with the .mat file.

    A year folder outside the valid window (or ``UNKNOWN_YEAR_FOLDER``) is recorded as a
    null year by the manifest; those rows agree when ``year`` is null.
    """
    columns = ["image_id", "make_id", "make_name", "model_id", "model_name", "year"]
    joined = manifest[columns].merge(paths[["image_id", "relative_path"]], on="image_id")
    offending: list[str] = []
    for row in joined.itertuples(index=False):
        prefix, make, model, year_folder, _ = row.relative_path.split("/")
        year = None if pd.isna(row.year) else int(row.year)
        make_name = names.makes[row.make_id].name if row.make_id in names.makes else None
        model_name = names.models[row.model_id].name if row.model_id in names.models else None
        agrees = (
            prefix == IMAGE_DIR
            and int(make) == row.make_id
            and int(model) == row.model_id
            and year == _expected_year(year_folder)
            and (row.make_name if isinstance(row.make_name, str) else None) == make_name
            and (row.model_name if isinstance(row.model_name, str) else None) == model_name
        )
        if not agrees:
            offending.append(row.image_id)
    counts = {"manifest_rows": int(len(manifest)), "mismatched": len(offending)}
    return _finding("labels_match_compcars_metadata", counts, offending)


def check_regeneration(name: str, rebuilt: Path, sidecar: Path) -> Finding:
    """The rebuilt table hashes to the digest pinned in the published sidecar."""
    digest = sha256_of_file(rebuilt)
    published = read_sidecar_sha256(sidecar)
    mismatched = [name] if digest != published else []
    return _finding(
        "regeneration_byte_identical",
        {"compared": 1, "mismatched": len(mismatched)},
        mismatched,
        note=f"rebuilt {digest}, published {published}",
    )


# --- runner ---------------------------------------------------------------------------------


class AuditResult(BaseModel):
    split: str
    findings: list[Finding]
    report_path: Path

    @property
    def failed_checks(self) -> list[str]:
        return [f.check for f in self.findings if f.must_pass and not f.passed]

    @property
    def passed(self) -> bool:
        return not self.failed_checks


def render_report(split: str, findings: Sequence[Finding], inputs: Mapping[str, str]) -> str:
    """Markdown report: inputs, one row per check and the capped offending ids."""
    verdict = "PASS" if all(f.passed or not f.must_pass for f in findings) else "FAIL"
    lines = [
        f"# Leakage audit: {split}",
        "",
        "Generated by `scripts/run_leakage_audit.py` (PROTOCOL.md §7 leakage controls, §23",
        "Data); do not edit by hand. A check marked must-pass fails the audit; a finding",
        "that is not must-pass is reported only. Offending identifiers are capped at",
        f"{ID_CAP}.",
        "",
        f"Verdict: **{verdict}**",
        "",
        "## Inputs",
        "",
        *[f"- {k}: `{v}`" for k, v in inputs.items()],
        "",
        "## Checks",
        "",
        "| check | must pass | status | counts |",
        "|-------|-----------|--------|--------|",
    ]
    for f in findings:
        counts = ", ".join(f"{k}={v}" for k, v in f.counts.items())
        lines.append(f"| {f.check} | {'yes' if f.must_pass else 'no'} | {f.status} | {counts} |")
    lines += ["", "## Details", ""]
    for f in findings:
        if not f.offending_ids and not f.note:
            continue
        lines.append(f"### {f.check}")
        lines.append("")
        if f.note:
            lines.append(f"{f.note}")
            lines.append("")
        if f.offending_ids:
            shown = len(f.offending_ids)
            lines.append(f"Offending ids ({shown} shown):")
            lines.append("")
            lines.append(", ".join(f"`{i}`" for i in f.offending_ids))
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def audit_split(
    split: str,
    findings: Sequence[Finding],
    reports_dir: Path,
    inputs: Mapping[str, str] | None = None,
) -> AuditResult:
    """Write ``reports_dir/<split>.md`` and return the findings with the pass/fail verdict."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / f"{split}.md"
    path.write_text(render_report(split, findings, inputs or {}))
    return AuditResult(split=split, findings=list(findings), report_path=path)


# --- per-split assembly -----------------------------------------------------------------------

GROUP_COLUMNS = ("duplicate_group_id", "source_group_id")
MANIFEST_JOIN = ("image_id", "validity_status", *GROUP_COLUMNS)


def _with_groups(split: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    """Split rows with the manifest's validity and group columns; a missing group is the id."""
    rows = split.merge(manifest[list(MANIFEST_JOIN)], on="image_id", how="left")
    for column in GROUP_COLUMNS:
        rows[column] = rows[column].where(rows[column].notna(), rows["image_id"])
    return rows


def track_a_findings(manifest: pd.DataFrame, split: pd.DataFrame) -> list[Finding]:
    """Checks on the Track A table (``split_track_a``, ``track_a_validation``).

    Support vs query is the official split, so a crossing group is reported with
    ``OFFICIAL_SPLIT_STATUS`` and does not fail; fit vs validation is ours and must hold.
    """
    rows = _with_groups(split, manifest)
    rows["class_id"] = [f"{m}/{mo}" for m, mo in zip(rows.make_id, rows.model_id, strict=True)]
    findings = [check_split_ids_in_manifest(split, manifest)]
    for column in GROUP_COLUMNS:
        findings.append(
            check_no_group_crosses(
                rows,
                column,
                "split_track_a",
                must_pass=False,
                status=OFFICIAL_SPLIT_STATUS,
                check=f"no_{column}_crosses_support_query",
                class_column="class_id",
            )
        )
    for column in GROUP_COLUMNS:
        findings.append(
            check_no_group_crosses(
                rows,
                column,
                "track_a_validation",
                check=f"no_{column}_crosses_fit_validation",
                class_column="class_id",
            )
        )
    findings.append(check_classes_have_support(rows, "split_track_a", "support", "query"))
    return findings


def track_b_findings(manifest: pd.DataFrame, split: pd.DataFrame) -> list[Finding]:
    """Checks on one Track B seed table (``class_id``, ``partition``); all must pass."""
    rows = _with_groups(split, manifest)
    findings = [check_split_ids_in_manifest(split, manifest)]
    findings += [
        check_no_group_crosses(rows, column, "partition", class_column="class_id")
        for column in GROUP_COLUMNS
    ]
    findings.append(check_classes_have_support(rows, "partition", "support", "test"))
    return findings


def e0_fixture_findings(fixture_dir: Path) -> tuple[list[Finding], dict[str, str]]:
    """Checks on ``tests/fixtures/e0_10class``: ``split.json`` against ``manifest_slice``.

    The slice cannot regenerate the split (the selection rule reads the whole manifest), so
    the regeneration check is replaced by the slice checksum pinned in ``split.json``.
    """
    import json

    slice_path = fixture_dir / "manifest_slice.parquet"
    manifest = pd.read_parquet(slice_path)
    split_file = json.loads((fixture_dir / "split.json").read_text())
    records = [
        (image_id, class_id, partition)
        for partition in ("support", "query")
        for class_id, ids in split_file[partition].items()
        for image_id in ids
    ]
    listed = pd.DataFrame.from_records(records, columns=["image_id", "class_id", "partition"])
    split = listed.merge(manifest[["image_id", "sha256"]], on="image_id", how="left")
    split["sha256"] = split["sha256"].fillna("")
    rows = _with_groups(split, manifest)

    slice_digest = sha256_of_file(slice_path)
    pinned = split_file["manifest_slice_sha256"]
    findings = [check_split_ids_in_manifest(split, manifest)]
    findings += [check_no_group_crosses(rows, column, "partition") for column in GROUP_COLUMNS]
    findings.append(check_classes_have_support(rows, "partition", "support", "query"))
    findings.append(
        _finding(
            "fixture_slice_checksum",
            {"compared": 1, "mismatched": int(slice_digest != pinned)},
            [] if slice_digest == pinned else ["manifest_slice.parquet"],
            note=f"slice {slice_digest}, split.json {pinned}",
        )
    )
    inputs = {"fixture": str(fixture_dir), "manifest_slice sha256": slice_digest}
    return findings, inputs
