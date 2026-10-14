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
from relaxed_rosettes.config import GAMMA_HZ_PER_T, Config
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
    concomitant_phase_rad,
    gradient_outer_integral_at_times,
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


def test_gradient_outer_integral_matches_dense_quadrature(shapes: list[Shape]) -> None:
    g = shapes[0].g_mT_per_m
    raster = SMALL.system.grad_raster_time_s
    t = np.array([0.0, 3.3, 7.5, 251.2, len(g) - 0.7, len(g)]) * raster
    gg = gradient_outer_integral_at_times(g, raster, t)
    # Trapezoid on a fine grid of the same piecewise-linear waveform.
    fine = np.linspace(0.0, len(g) * raster, 200 * len(g) + 1)
    t_points = np.concatenate([[0.0], (np.arange(len(g)) + 0.5) * raster, [len(g) * raster]])
    g_points = np.concatenate([np.zeros((1, 3)), g, np.zeros((1, 3))]) * 1e-3
    g_fine = np.stack([np.interp(fine, t_points, g_points[:, i]) for i in range(3)], axis=1)
    outer = np.einsum("ti,tj->tij", g_fine, g_fine)
    steps = np.diff(fine)[:, None, None] * (outer[:-1] + outer[1:]) / 2
    cumulative = np.concatenate([np.zeros((1, 3, 3)), np.cumsum(steps, axis=0)])
    # Every time in t is on the fine grid.
    expected = cumulative[np.rint(t / (raster / 200)).astype(int)]
    np.testing.assert_allclose(gg, expected, rtol=1e-6, atol=1e-18)
    np.testing.assert_allclose(gg, gg.transpose(0, 2, 1), atol=0)
    assert np.all(np.linalg.eigvalsh(gg) > -1e-18)
    assert np.all(np.diff(np.trace(gg, axis1=1, axis2=2)) >= 0)


def test_gradient_outer_integral_rotates_with_waveform(shapes: list[Shape]) -> None:
    g = shapes[1].g_mT_per_m
    raster = SMALL.system.grad_raster_time_s
    t = np.array([0.4, 120.0, 900.6]) * raster
    r = make_schedule(SMALL).rotation_matrix[3]
    rotated = gradient_outer_integral_at_times(g @ r.T, raster, t)
    expected = r @ gradient_outer_integral_at_times(g, raster, t) @ r.T
    np.testing.assert_allclose(rotated, expected, rtol=1e-12, atol=1e-20)


def test_concomitant_phase_of_constant_gz() -> None:
    # Gz alone, held for time T: B_c = Gz² (x² + y²) / (8 B0).
    gz, duration = 20e-3, 5e-3
    gg = np.zeros((3, 3))
    gg[2, 2] = gz**2 * duration
    r = np.array([[0.1, 0.0, 0.0], [0.0, 0.05, 0.3], [0.0, 0.0, 0.1]])
    expected = (
        2 * np.pi * GAMMA_HZ_PER_T * gz**2 * (r[:, 0] ** 2 + r[:, 1] ** 2) / (8 * SMALL.system.B0_T)
    ) * duration
    np.testing.assert_allclose(concomitant_phase_rad(SMALL, gg, r), expected)
    # Gx alone: B_c = Gx² z² / (2 B0), and the x z cross term with Gx and Gz.
    g = np.array([10e-3, 0.0, 10e-3])
    gg = np.outer(g, g) * duration
    xz = np.array([[0.1, 0.0, 0.1]])
    bc = (g[2] ** 2 * 0.1**2 / 4 + g[0] ** 2 * 0.1**2 - g[0] * g[2] * 0.1 * 0.1) / (
        2 * SMALL.system.B0_T
    )
    np.testing.assert_allclose(
        concomitant_phase_rad(SMALL, gg[None], xz), [[2 * np.pi * GAMMA_HZ_PER_T * bc * duration]]
    )


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
    gg = trajectory.gg_T2s_per_m2(slice(1, 3))
    assert gg.shape == (2, SMALL.readout.adc_num_samples, 3, 3)
    # A rotation keeps the trace (the sum of ∫ g_i² dt).
    logical = trajectory.gg_shape_T2s_per_m2[trajectory.shape_index[1:3]]
    np.testing.assert_allclose(
        np.trace(gg, axis1=2, axis2=3), np.trace(logical, axis1=2, axis2=3), rtol=1e-12
    )


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
    np.testing.assert_allclose(loaded.gg_T2s_per_m2(), trajectory.gg_T2s_per_m2())
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
