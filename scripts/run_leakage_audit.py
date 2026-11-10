"""Run the leakage audit and write reports/leakage/<split>.md (PROTOCOL.md §7, §23 Data).

Default mode audits the published splits against the image manifest: Track A
(data/splits/track_a/official_cvpr2015.parquet) and every Track B seed
(data/splits/track_b/seed_<s>.parquet, built by scripts/build_track_b_splits.py). The
dataset-gated checks (paths resolve, labels match misc/make_model_name.mat, regeneration
byte-identical) run when the private path table and the CompCars root are present and are
recorded as skipped otherwise. ``--cross-view`` audits the Track B cross-view variants
(data/splits/track_b/cross_view/seed_<s>/view_<v>.parquet, ADR-0004) instead and writes
reports/leakage/track_b_cross_view_seed_<s>.md per seed. ``--fixture <dir>`` audits an E0
fixture directory instead, with no dataset. Exit status 1 when any must-pass check fails.
Usage: uv run python scripts/run_leakage_audit.py
       [--fixture tests/fixtures/e0_10class | --cross-view] [--reports-dir reports/leakage]
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from avr.config import CONFIG_DIR, load_config
from avr.data.cross_view import CrossViewThresholds, build_cross_view_splits
from avr.data.leakage import (
    AuditResult,
    Finding,
    audit_split,
    check_labels_match,
    check_paths_resolve,
    check_regeneration,
    cross_view_findings,
    e0_fixture_findings,
    track_a_findings,
    track_b_findings,
)
from avr.data.manifest import MANIFEST_NAME, NAME_MAT, PATHS_NAME
from avr.data.names import load_name_tables
from avr.data.split_track_a import SPLIT_NAME, build_track_a, read_official_split
from avr.data.split_track_b import EligibilityThresholds, build_track_b_splits
from avr.data.splits_io import write_split_manifest
from avr.io import read_sidecar_sha256, sha256_of_file

REPO_ROOT = CONFIG_DIR.parent
MANIFEST = REPO_ROOT / "data" / "manifests" / f"{MANIFEST_NAME}.parquet"
PATHS = REPO_ROOT / "data" / "paths" / f"{PATHS_NAME}.parquet"
TRACK_A = REPO_ROOT / "data" / "splits" / "track_a" / f"{SPLIT_NAME}.parquet"
TRACK_B_DIR = REPO_ROOT / "data" / "splits" / "track_b"
CROSS_VIEW_DIR = TRACK_B_DIR / "cross_view"
_B_CFG = load_config().split_track_b
CROSS_VIEW_SEEDS = [_B_CFG.primary_seed, *_B_CFG.sensitivity_seeds]
CROSS_VIEW_VIEWS = list(_B_CFG.cross_view.views)
CROSS_VIEW_THRESHOLDS = CrossViewThresholds(
    min_support=_B_CFG.cross_view.min_support, min_test=_B_CFG.cross_view.min_test
)
REPORTS_DIR = REPO_ROOT / "reports" / "leakage"


def _skipped(check: str, note: str) -> Finding:
    return Finding(
        check=check, passed=True, status="skipped", counts={}, offending_ids=[], note=note
    )


def _print(result: AuditResult) -> None:
    verdict = "PASS" if result.passed else "FAIL"
    path = result.report_path
    shown = path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path
    print(f"{result.split}: {verdict} -> {shown}")
    for f in result.findings:
        counts = ", ".join(f"{k}={v}" for k, v in f.counts.items())
        print(f"  {f.check}: {f.status} ({counts})")


def audit_fixture(fixture_dir: Path, reports_dir: Path) -> AuditResult:
    findings, inputs = e0_fixture_findings(fixture_dir)
    return audit_split("e0_fixture", findings, reports_dir, inputs)


def audit_published(reports_dir: Path, tmp: Path) -> list[AuditResult]:
    cfg = load_config()
    manifest = pd.read_parquet(MANIFEST)
    manifest_digest = sha256_of_file(MANIFEST)
    root = (REPO_ROOT / cfg.dataset.root).resolve()
    dataset_ready = PATHS.is_file() and (root / NAME_MAT).is_file()
    paths = pd.read_parquet(PATHS) if dataset_ready else None
    common_inputs = {
        "manifest": str(MANIFEST.relative_to(REPO_ROOT)),
        "manifest sha256": manifest_digest,
    }

    if dataset_ready:
        names = load_name_tables(root / NAME_MAT)
        manifest_findings = [
            check_paths_resolve(manifest, paths, root),
            check_labels_match(manifest, paths, names),
        ]
    else:
        note = "private path table or CompCars root not present"
        manifest_findings = [
            _skipped("manifest_paths_resolve", note),
            _skipped("labels_match_compcars_metadata", note),
        ]

    results = []

    # Track A
    split_a = pd.read_parquet(TRACK_A)
    findings = [*manifest_findings, *track_a_findings(manifest, split_a)]
    if dataset_ready:
        split_cfg = load_config("splits/track_a").splits
        build = build_track_a(
            manifest,
            paths,
            read_official_split(root),
            fraction=split_cfg.validation_fraction,
            seed=split_cfg.validation_seed,
        )
        rebuilt = write_split_manifest(build.table, tmp / "track_a" / TRACK_A.name)
        findings.append(
            check_regeneration(TRACK_A.name, rebuilt.manifest, TRACK_A.with_suffix(".sha256"))
        )
    else:
        findings.append(_skipped("regeneration_byte_identical", "dataset not present"))
    inputs = {
        **common_inputs,
        "split": str(TRACK_A.relative_to(REPO_ROOT)),
        "split sha256": sha256_of_file(TRACK_A),
    }
    results.append(audit_split("track_a", findings, reports_dir, inputs))

    # Track B: regenerate every seed once, then audit seed by seed
    b_cfg = cfg.split_track_b
    seeds = [b_cfg.primary_seed, *b_cfg.sensitivity_seeds]
    rebuilt_b = build_track_b_splits(
        manifest,
        EligibilityThresholds(**{k: b_cfg[k] for k in EligibilityThresholds.model_fields}),
        dict(b_cfg.fractions),
        seeds=seeds,
        audit_path=tmp / "track_b" / "eligibility_audit.md",
        out_dir=tmp / "track_b",
    )
    for seed in seeds:
        split_path = TRACK_B_DIR / f"seed_{seed}.parquet"
        split_b = pd.read_parquet(split_path)
        findings = [
            *manifest_findings,
            *track_b_findings(manifest, split_b),
            check_regeneration(
                split_path.name, rebuilt_b.outputs[seed].manifest, split_path.with_suffix(".sha256")
            ),
        ]
        inputs = {
            **common_inputs,
            "split": str(split_path.relative_to(REPO_ROOT)),
            "split sha256": sha256_of_file(split_path),
        }
        results.append(audit_split(f"track_b_seed_{seed}", findings, reports_dir, inputs))
    return results


def _prefixed(view: int, findings: Sequence[Finding]) -> list[Finding]:
    return [f.model_copy(update={"check": f"view_{view}/{f.check}"}) for f in findings]


def audit_cross_view(reports_dir: Path, tmp: Path) -> list[AuditResult]:
    """One report per seed over data/splits/track_b/cross_view/seed_<s>/view_<v>.parquet.

    Every variant gets the Track B checks (ids resolve, no group crosses partitions, every
    test class has support) plus the ADR-0004 invariant that a row keeps the partition of
    its parent seed row; the variants are regenerated from the stored parent and must hash
    to the published sidecars. The parent itself must match its sidecar first.
    """
    manifest = pd.read_parquet(MANIFEST)
    views = CROSS_VIEW_VIEWS
    results = []
    for seed in CROSS_VIEW_SEEDS:
        parent_path = TRACK_B_DIR / f"seed_{seed}.parquet"
        parent_digest = sha256_of_file(parent_path)
        if parent_digest != read_sidecar_sha256(parent_path.with_suffix(".sha256")):
            raise SystemExit(f"{parent_path} does not match its .sha256 sidecar")
        parent = pd.read_parquet(parent_path)
        rebuilt = build_cross_view_splits(
            parent,
            seed=seed,
            views=views,
            thresholds=CROSS_VIEW_THRESHOLDS,
            report_path=tmp / "cross_view" / f"seed_{seed}.md",
            out_dir=tmp / "cross_view" / f"seed_{seed}",
        )
        inputs = {
            "manifest": str(MANIFEST.relative_to(REPO_ROOT)),
            "manifest sha256": sha256_of_file(MANIFEST),
            "parent split": str(parent_path.relative_to(REPO_ROOT)),
            "parent split sha256": parent_digest,
        }
        findings: list[Finding] = []
        for view in views:
            variant_path = CROSS_VIEW_DIR / f"seed_{seed}" / f"view_{view}.parquet"
            variant = pd.read_parquet(variant_path)
            inputs[f"view_{view}"] = str(variant_path.relative_to(REPO_ROOT))
            inputs[f"view_{view} sha256"] = sha256_of_file(variant_path)
            findings += _prefixed(
                view,
                [
                    *cross_view_findings(manifest, parent, variant),
                    check_regeneration(
                        variant_path.name,
                        rebuilt.outputs[view].manifest,
                        variant_path.with_suffix(".sha256"),
                    ),
                ],
            )
        results.append(
            audit_split(f"track_b_cross_view_seed_{seed}", findings, reports_dir, inputs)
        )
    return results


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fixture", type=Path, help="audit an E0 fixture directory instead")
    parser.add_argument(
        "--cross-view", action="store_true", help="audit the Track B cross-view variants instead"
    )
    parser.add_argument("--reports-dir", type=Path, default=REPORTS_DIR)
    args = parser.parse_args(argv)

    if args.fixture is not None:
        results = [audit_fixture(args.fixture.resolve(), args.reports_dir)]
    elif args.cross_view:
        with tempfile.TemporaryDirectory() as tmp:
            results = audit_cross_view(args.reports_dir, Path(tmp))
    else:
        with tempfile.TemporaryDirectory() as tmp:
            results = audit_published(args.reports_dir, Path(tmp))

    for result in results:
        _print(result)
    failed = {r.split: r.failed_checks for r in results if not r.passed}
    if failed:
        print(f"leakage audit FAILED: {failed}", file=sys.stderr)
        return 1
    print("leakage audit passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
