"""Markdown report rendered from the M1 gate result files (issue M1-16)."""

from avr.pipeline.gate_report import render_readme


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


def test_readme_carries_labels_tables_and_verdicts():
    text = render_readme([toy_results(t, v) for t in ("track_a", "track_b") for v in ("p1", "p3")])

    assert "not article results" in text
    assert "LABEL-A" in text
    assert "mapping_status = provisional" in text
    assert "| make_model | 4 | 0.5000 | 0.7500 | 1.0000 | 1.0000 | 0.6000 |" in text
    assert "byte-identical" in text and "yes" in text
    assert "## PROTOCOL.md §25 M1 checklist" in text
    assert "reports/leakage/track_a.md" in text
    assert "## Not yet run" not in text


def test_missing_variant_is_listed_as_not_yet_run():
    text = render_readme([toy_results("track_a", "p1")])
    assert "## Not yet run" in text
    assert "track_b p1" in text
