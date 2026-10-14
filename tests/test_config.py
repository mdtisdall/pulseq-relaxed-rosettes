from pathlib import Path

import pytest

from relaxed_rosettes.config import Config, ConfigError, RfPhase


def test_defaults_are_valid() -> None:
    config = Config()
    assert config.k_max_per_m == pytest.approx(500.0)
    assert config.raster_points_per_tr == 5000
    assert config.excitation.rf_phase is RfPhase.RANDOM


def test_toml_round_trip() -> None:
    config = Config().with_overrides(["seed=7", "excitation.rf_phase=quadratic117"])
    again = Config.from_toml(config.to_toml())
    assert again == config
    assert again.digest() == config.digest()


def test_partial_file_takes_defaults(tmp_path: Path) -> None:
    path = tmp_path / "c.toml"
    path.write_text("seed = 3\n[readout]\ntr_s = 0.025\nadc_num_samples = 7000\n")
    config = Config.load(path)
    assert config.seed == 3
    assert config.readout.tr_s == 0.025
    assert config.readout.adc_dwell_s == Config().readout.adc_dwell_s
    assert config.raster_points_per_tr == 2500


def test_digest_changes_with_values() -> None:
    assert Config().digest() != Config().with_overrides(["seed=1"]).digest()


def test_int_accepted_for_float() -> None:
    config = Config().with_overrides(["excitation.flip_angle_deg=12"])
    assert config.excitation.flip_angle_deg == 12.0
    assert isinstance(config.excitation.flip_angle_deg, float)


@pytest.mark.parametrize(
    ("toml", "message"),
    [
        ("bogus = 1\n", "unknown key"),
        ("[readout]\nbogus = 1\n", "unknown key"),
        ("[readout]\nadc_num_samples = 1.5\n", "wrong type"),
        ("[excitation]\nrf_phase = 'linear'\n", "must be one of"),
        ("readout = 3\n", "must be a table"),
        ("[readout\n", "invalid TOML"),
    ],
)
def test_bad_files_rejected(toml: str, message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        Config.from_toml(toml)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ("gradients.max_grad_mT_per_m=40", "exceeds system"),
        ("gradients.boundary_slew_T_per_m_per_s=110", "exceeds max_slew"),
        ("readout.adc_dwell_s=4.5e-6", "undersamples"),
        ("readout.adc_num_samples=20000", "exceeds tr_s"),
        ("readout.tr_s=0.050005", "whole number"),
        ("shapes.tip_radius_min=0", "tip_radius_min"),
        ("readout.tr_s=-1", "must be > 0"),
        ("nope.key=1", "unknown key"),
        ("readout.tr_s", "not key=value"),
    ],
)
def test_invalid_values_rejected(override: str, message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        Config().with_overrides([override])
