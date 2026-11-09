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

    assert cfg.dedup.phash_hamming_threshold == 6
    assert cfg.dedup.phash_hash_size == 8
