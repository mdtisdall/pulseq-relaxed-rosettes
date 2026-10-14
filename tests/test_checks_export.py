import json
from pathlib import Path

import numpy as np
import pypulseq as pp
import pytest
from pypulseq.utils.safe_pns_prediction import safe_example_hw

from relaxed_rosettes.checks import (
    acquired_time_range,
    gradient_report,
    pns_report,
    spectrum_report,
)
from relaxed_rosettes.cli import main
from relaxed_rosettes.config import Config
from relaxed_rosettes.optimize import (
    Shape,
    k_weights,
    kmax_per_mT_per_m_raster,
    optimize_shapes,
    shape_metrics,
)
from relaxed_rosettes.sequence import Schedule, build_sequence, make_schedule
from relaxed_rosettes.trajectory import (
    Trajectory,
    kspace_at_times,
    load_trajectory,
    make_trajectory,
    seq_kspace,
)

SMALL = Config().with_overrides(
    [
        "readout.tr_s=0.01",
        "readout.adc_num_samples=2000",
        "shapes.echoes_per_tr=4",
        "shapes.num_shapes=2",
        "scan.num_trs=6",
        "scan.num_dummy_trs=2",
    ]
)


@pytest.fixture(scope="module")
def shapes() -> list[Shape]:
    return optimize_shapes(SMALL, jobs=1)


@pytest.fixture(scope="module")
def schedule() -> Schedule:
    return make_schedule(SMALL)


@pytest.fixture(scope="module")
def seq(shapes: list[Shape], schedule: Schedule) -> pp.Sequence:
    return build_sequence(SMALL, shapes, schedule)


@pytest.fixture(scope="module")
def trajectory(shapes: list[Shape], schedule: Schedule) -> Trajectory:
    return make_trajectory(SMALL, schedule, shapes)


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A .seq file and its design file, made through the CLI."""
    directory = tmp_path_factory.mktemp("built")
    config_path = directory / "small.toml"
    config_path.write_text(SMALL.to_toml())
    shapes_path = directory / "shapes.npz"
    seq_path = directory / "small.seq"
    assert main(["shapes", str(config_path), "-o", str(shapes_path), "--jobs", "1"]) == 0
    assert main(["sequence", str(shapes_path), "-o", str(seq_path)]) == 0
    return seq_path


def test_kspace_at_times_matches_optimizer_weights(shapes: list[Shape]) -> None:
    g = shapes[0].g_mT_per_m
    raster = SMALL.system.grad_raster_time_s
    edges = np.array([1, 7, 250, len(g) - 1, len(g)])
    k = kspace_at_times(g, raster, edges * raster) / SMALL.k_max_per_m
    scale = kmax_per_mT_per_m_raster(SMALL)
    expected = np.stack([scale * (k_weights(len(g), e) @ g) for e in edges])
    np.testing.assert_allclose(k, expected, atol=1e-12)


def test_trajectory_matches_pypulseq(seq: pp.Sequence, trajectory: Trajectory) -> None:
    n = len(trajectory.adc_time_in_tr_s)
    expected = seq_kspace(seq, trajectory.tr_index, n)
    np.testing.assert_allclose(trajectory.k_per_m(), expected, atol=1e-6 * SMALL.k_max_per_m)
    # Also against pypulseq's k-space of the whole sequence.
    k_all, _, _, _, t_all = seq.calculate_kspace()
    np.testing.assert_allclose(
        trajectory.k_per_m().reshape(-1, 3), k_all.T, atol=1e-6 * SMALL.k_max_per_m
    )
    t_expected = trajectory.tr_start_s[:, None] + trajectory.adc_time_in_tr_s[None, :]
    np.testing.assert_allclose(t_expected.ravel(), t_all, atol=1e-9)


def test_trajectory_fields(trajectory: Trajectory, schedule: Schedule) -> None:
    acquired = np.flatnonzero(schedule.acquired)
    np.testing.assert_array_equal(trajectory.tr_index, acquired)
    np.testing.assert_allclose(trajectory.tr_start_s, acquired * SMALL.readout.tr_s)
    first = SMALL.adc_start_s + SMALL.readout.adc_dwell_s / 2 - SMALL.rf_center_s
    assert trajectory.time_since_excitation_s[0] == pytest.approx(first)
    np.testing.assert_allclose(
        np.diff(trajectory.time_since_excitation_s), SMALL.readout.adc_dwell_s
    )
    assert trajectory.k_per_m(slice(1, 3)).shape == (2, SMALL.readout.adc_num_samples, 3)


def test_gradient_report(shapes: list[Shape], schedule: Schedule) -> None:
    report = gradient_report(SMALL, schedule, shapes)
    vector_max = max(shape_metrics(SMALL, s).max_grad_mT_per_m for s in shapes)
    assert report["max_grad_mT_per_m"] == pytest.approx(vector_max)
    assert max(report["max_grad_per_axis_mT_per_m"]) <= report["max_grad_mT_per_m"] + 1e-9
    assert max(report["max_slew_per_axis_T_per_m_per_s"]) <= report["max_slew_T_per_m_per_s"] + 1e-9
    assert report["nyquist_ok"]


def test_spectrum_report(seq: pp.Sequence) -> None:
    time_range = acquired_time_range(SMALL, 6)
    report = spectrum_report(seq, [(100.0, 300.0), (5000.0, 6000.0)], time_range)
    assert 0 < report["bands"][0]["relative_peak"] <= 1
    assert report["bands"][1]["relative_peak"] is not None
    with pytest.raises(ValueError, match="at least"):
        spectrum_report(seq, [], acquired_time_range(SMALL, 4))


def test_pns_report(seq: pp.Sequence) -> None:
    report = pns_report(seq, safe_example_hw(), acquired_time_range(SMALL, 4))
    assert report["ok"]
    assert 0 < report["max_pns"] < 1


def test_cli_check(built: Path, capsys: pytest.CaptureFixture[str]) -> None:
    capsys.readouterr()
    assert main(["check", str(built), "--band", "500:700", "--trs", "6", "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["config_digest"] == SMALL.digest()
    assert report["timing"]["ok"]
    assert report["spectrum"]["bands"][0]["band_hz"] == [500.0, 700.0]
    assert main(["check", str(built), "--trs", "6"]) == 0
    assert "Nyquist: ok" in capsys.readouterr().out


def test_cli_export_traj(built: Path, tmp_path: Path, trajectory: Trajectory) -> None:
    path = tmp_path / "traj.npz"
    assert main(["export-traj", str(built), "-o", str(path)]) == 0
    config, loaded = load_trajectory(path)
    assert config == SMALL
    np.testing.assert_allclose(loaded.k_per_m(), trajectory.k_per_m())
    np.testing.assert_array_equal(loaded.shape_index, trajectory.shape_index)


def test_cli_rejects_mismatched_design(
    built: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    other = SMALL.with_overrides(["seed=1"])
    config_path = tmp_path / "other.toml"
    config_path.write_text(other.to_toml())
    shapes_path = tmp_path / "shapes.npz"
    seq_path = tmp_path / "other.seq"
    assert main(["shapes", str(config_path), "-o", str(shapes_path), "--jobs", "1"]) == 0
    assert main(["sequence", str(shapes_path), "-o", str(seq_path)]) == 0
    # Pair the other design file with the first .seq file.
    mismatched = tmp_path / "mismatched.seq"
    mismatched.write_bytes(built.read_bytes())
    seq_path.with_suffix(".npz").rename(mismatched.with_suffix(".npz"))
    capsys.readouterr()
    assert main(["check", str(mismatched), "--trs", "6"]) == 1
    assert "config digest" in capsys.readouterr().err


def test_cli_rejects_bad_band() -> None:
    with pytest.raises(SystemExit):
        main(["check", "x.seq", "--band", "700:500"])
