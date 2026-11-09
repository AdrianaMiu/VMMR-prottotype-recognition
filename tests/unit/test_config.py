from avr.config import load_config


def test_default_config_composes_the_three_experiment_groups():
    cfg = load_config()

    assert set(cfg.keys()) >= {"dataset", "encoder", "experiment"}
    assert cfg.experiment.name == "e0_pipeline_validation"


def test_overrides_select_a_named_group_entry():
    cfg = load_config(overrides=["encoders@encoder=siglip2_so400m_naflex"])

    assert cfg.encoder.model_id == "google/siglip2-so400m-patch16-naflex"


def test_tracking_group_points_at_the_local_file_store():
    cfg = load_config()

    assert cfg.tracking.backend == "local_file"
    assert cfg.tracking.tracking_dir == "mlruns"
    assert cfg.tracking.experiment_name == "avr"


def test_dedup_group_records_the_phash_threshold():
    cfg = load_config()

    assert cfg.dedup.phash_hamming_threshold == 2  # ADR-0005
    assert cfg.dedup.phash_hash_size == 8


def test_track_b_split_group_records_the_protocol_eligibility_thresholds():
    cfg = load_config()

    # PROTOCOL.md §5 Track B: >= 20 valid images, >= 2 viewpoints, >= 5 query-eligible,
    # approved mapping; no approved rows exist yet, so the provisional fallback is on.
    assert cfg.split_track_b.min_images == 20
    assert cfg.split_track_b.min_viewpoints == 2
    assert cfg.split_track_b.min_query_eligible == 5
    assert cfg.split_track_b.require_approved_mapping is True
    assert cfg.split_track_b.provisional_fallback is True
    assert cfg.split_track_b.primary_seed == 73
    assert list(cfg.split_track_b.sensitivity_seeds) == [13, 37, 101, 137]
    assert dict(cfg.split_track_b.fractions) == {"support": 0.6, "validation": 0.2, "test": 0.2}
