from avr.config import load_config


def test_default_config_composes_the_three_experiment_groups():
    cfg = load_config()

    assert set(cfg.keys()) >= {"dataset", "encoder", "experiment"}
    assert cfg.experiment.name == "e0_pipeline_validation"


def test_overrides_select_a_named_group_entry():
    cfg = load_config(overrides=["encoders@encoder=siglip2_so400m_naflex"])

    assert cfg.encoder.model_id == "google/siglip2-so400m-patch16-naflex"
