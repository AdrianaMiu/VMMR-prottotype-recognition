"""Markdown report rendered from the M1 gate result files (issue M1-16)."""

import json
from pathlib import Path

from avr.pipeline.gate_report import load_checklist_context, render_readme


def toy_results(track: str, variant: str) -> dict:
    level = {
        "ranking": {
            "top_k": {"1": 0.5, "5": 0.75, "10": 1.0, "20": 1.0},
            "mrr": 0.6,
            "median_rank": 1.0,
            "rank_p90": 7.0,
            "class_count": 4,
        },
        "classification": {
            "accuracy": 0.5,
            "macro_precision": 0.4,
            "macro_recall": 0.45,
            "macro_f1": 0.42,
            "weighted_f1": 0.5,
            "balanced_accuracy": 0.45,
            "classes_without_predicted_positives": 1,
            "class_count": 4,
        },
    }
    return {
        "track": track,
        "variant": variant,
        "protocol_id": f"{track}_x",
        "split_label": "LABEL-A" if track == "track_a" else f"{track}_x",
        "mapping_status": "not_applicable" if track == "track_a" else "provisional",
        "mask_metadata_sha256": "not_applicable" if variant == "p1" else "cd" * 32,
        "embedding_store": {"rows": 10, "shards": 1, "dimension": 2},
        "evaluations": {
            "official_full": {
                "support_images": 4,
                "query_images": 4,
                "class_count": 4,
                "images_without_embedding": {"support": 0, "query": 1},
                "queries_not_eligible_unknown_viewpoint": 0,
                "tied_queries_top1": 0,
                "levels": {"make_model": level, "make": level},
                "error_decomposition": {
                    "wrong_generation_same_model_fraction": 0.0,
                    "wrong_model_same_make_fraction": 0.25,
                    "wrong_make_fraction": 0.25,
                    "generation_accuracy_given_model": 1.0,
                },
            }
        },
        "prototypes": {
            "parquet": "data/prototypes/toy/p1/m0_x.parquet",
            "parquet_sha256": "ab" * 32,
            "class_count": 4,
            "support_counts": {"min": 1, "max": 1, "total": 4},
        },
        "qdrant": {
            "collection": "c__n__e__p__2__m0",
            "point_count": 4,
            "parity": {
                "queries": 4,
                "k": 4,
                "ordered_equal": 4,
                "equal_within_tolerance": 0,
                "score_tolerance": 1e-5,
                "passed": True,
            },
        },
        "run_manifest": {"run_id": "r", "git_commit": "abc", "dirty_worktree_flag": False},
        "runtime_s": 1.5,
        "peak_rss_mb": 100.0,
        "rerun": {
            "repeats": 2,
            "comparisons": [
                {
                    "prototypes_byte_identical": True,
                    "ranks_byte_identical": {"official_full": True},
                    "embeddings_max_abs_difference": 0.0,
                    "embeddings_within_tolerance": True,
                    "prototype_parquet_sha256_equal": True,
                    "metrics_equal": True,
                    "determinism_hashes_equal": True,
                    "runtime_s": [1.5, 1.4],
                    "peak_rss_mb": [100.0, 101.0],
                }
            ],
        },
    }


def toy_repo(root: Path) -> Path:
    """A repository tree whose counts differ from every real one (7, 5, 3, ...)."""
    (root / "data/manifests").mkdir(parents=True)
    (root / "data/splits/track_b").mkdir(parents=True)
    (root / "reports/leakage").mkdir(parents=True)
    (root / "reports/splits/track_b/cross_view").mkdir(parents=True)
    (root / "data/manifests/compcars_web_images.summary.json").write_text(
        json.dumps({"row_count": 7001, "valid_count": 6999, "duplicate_group_count": 11})
    )
    (root / "data/manifests/p3_mask_metadata.summary.json").write_text(
        json.dumps({"row_count": 6999, "mask_failure_rate": 0.0123})
    )
    (root / "data/splits/track_b/summary.json").write_text(
        json.dumps({"eligible_classes": 33, "eligible_images": 4444})
    )
    (root / "reports/leakage/track_a.md").write_text(
        "Verdict: **PASS**\n| no_duplicate_group_id_crosses_support_query | no | reported | "
        "groups=9, crossing_groups=4, crossing_images=8 |\n"
    )
    (root / "reports/leakage/track_b_seed_73.md").write_text("Verdict: **FAIL**\n")
    (root / "reports/splits/track_b/cross_view/seed_73.md").write_text("x")
    return root


def test_readme_carries_labels_tables_and_verdicts(tmp_path: Path):
    context = load_checklist_context(toy_repo(tmp_path))
    text = render_readme(
        [toy_results(t, v) for t in ("track_a", "track_b") for v in ("p1", "p3")], context
    )

    assert "not article results" in text
    assert "LABEL-A" in text
    assert "mapping_status = provisional" in text
    assert "| make_model | 4 | 0.5000 | 0.7500 | 1.0000 | 1.0000 | 0.6000 |" in text
    assert "byte-identical" in text and "yes" in text
    assert "P3 mask metadata sha256: `not_applicable`" in text
    assert f"P3 mask metadata sha256: `{'cd' * 32}`" in text
    assert "## PROTOCOL.md §25 M1 checklist" in text
    assert "## Not yet run" not in text


def test_checklist_numbers_come_from_the_committed_records_not_from_literals(tmp_path: Path):
    context = load_checklist_context(toy_repo(tmp_path))
    text = render_readme([toy_results("track_a", "p1")], context)

    assert "7,001 rows, 6,999 valid" in text
    assert "11 duplicate groups" in text
    assert "33 eligible classes, 4,444 images" in text
    assert "6,999 images, mask failure rate 1.23%" in text
    assert "`reports/leakage/track_a.md` (PASS; 4 official-split crossing groups" in text
    assert "`reports/leakage/track_b_seed_73.md` (FAIL)" in text
    assert "`reports/splits/track_b/cross_view/seed_73.md`" in text
    # the real repository's figures must not survive in the source of the renderer
    for literal in ("136,726", "136,365", "2,645", "52 official"):
        assert literal not in text


def test_checklist_links_the_segmentation_and_embedding_reports_and_waives_the_mapping(
    tmp_path: Path,
):
    context = load_checklist_context(toy_repo(tmp_path))
    text = render_readme([toy_results("track_a", "p1")], context)

    assert "reports/segmentation/README.md" in text
    assert "reports/embeddings/README.md" in text
    assert (
        "§23 'all Track B generation rows approved': WAIVED, protocol provisional (issue #17)"
        in text
    )


def test_missing_variant_is_listed_as_not_yet_run(tmp_path: Path):
    context = load_checklist_context(toy_repo(tmp_path))
    text = render_readme([toy_results("track_a", "p1")], context)
    assert "## Not yet run" in text
    assert "track_b p1" in text
