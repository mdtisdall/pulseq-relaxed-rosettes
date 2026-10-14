import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from relaxed_rosettes.cli import main
from relaxed_rosettes.config import Config
from relaxed_rosettes.coverage import CoverageGrid, ktau_metrics, window_metrics, window_samples
from relaxed_rosettes.optimize import optimize_shapes
from relaxed_rosettes.sequence import make_schedule
from relaxed_rosettes.trajectory import Trajectory, make_trajectory

SMALL = Config().with_overrides(
    [
        "readout.tr_s=0.01",
        "readout.adc_num_samples=2000",
        "shapes.echoes_per_tr=4",
        "shapes.num_shapes=2",
        "scan.num_trs=12",
        "scan.num_dummy_trs=2",
    ]
)
RESOLUTIONS = np.array([8e-3, 4e-3, 2e-3])


@pytest.fixture(scope="module")
def trajectory() -> Trajectory:
    return make_trajectory(SMALL, make_schedule(SMALL), optimize_shapes(SMALL, jobs=1))


def test_grid_counts_cells_in_each_ball() -> None:
    config = Config().with_overrides(["geometry.fov_m=0.016", "geometry.resolution_m=0.002"])
    resolutions = np.array([8e-3, 4e-3, 2e-3])
    grid = CoverageGrid.make(config, resolutions)
    cell = 1 / 0.016
    half = grid.size // 2
    axis = (np.arange(grid.size) - half) * cell
    radius = np.sqrt(axis[:, None, None] ** 2 + axis[None, :, None] ** 2 + axis[None, None, :] ** 2)
    for r, expected in zip(resolutions, grid.cells_in_ball, strict=True):
        assert np.count_nonzero(radius <= 1 / (2 * r) + 1e-9) == expected
    # One sample at k = 0 covers one cell of each ball.
    np.testing.assert_allclose(grid.coverage(np.zeros((5, 3))), 1 / grid.cells_in_ball)
    # Samples in every cell cover everything.
    k = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
    np.testing.assert_allclose(grid.coverage(k), 1.0)


def test_grid_rejects_bad_resolutions() -> None:
    with pytest.raises(ValueError, match="decreasing"):
        CoverageGrid.make(SMALL, np.array([2e-3, 4e-3]))
    with pytest.raises(ValueError, match="finer"):
        CoverageGrid.make(SMALL, np.array([0.5e-3]))


def test_window_samples_are_inside_the_window(trajectory: Trajectory) -> None:
    start = float(trajectory.tr_start_s[3]) + 0.004
    k, tau = window_samples(trajectory, SMALL.readout.tr_s, start, 0.015)
    dwell = SMALL.readout.adc_dwell_s
    # 15 ms from 4 ms into a TR: the end of that TR's ADC and the start of the next.
    assert len(k) == len(tau)
    assert 0 < len(k) <= 0.015 / dwell + 1
    assert tau.min() >= trajectory.time_since_excitation_s[0]


def test_metrics_grow_with_window_length(trajectory: Trajectory) -> None:
    m = window_metrics(SMALL, trajectory, [0.01, 0.05, 0.12], RESOLUTIONS, num_starts=1)
    assert m.coverage.shape == (3, 1, 3) and m.gap_cells.shape == (3, 1, 3)
    assert np.all((m.coverage > 0) & (m.coverage <= 1))
    # With one start, every window starts at the first acquired TR, so they nest.
    assert np.all(np.diff(m.coverage[:, 0], axis=0) >= 0)
    assert np.all(np.diff(m.gap_cells[:, 0], axis=0) <= 1e-9)


def test_tau_bins_report_the_worst_bin(trajectory: Trajectory) -> None:
    one = window_metrics(SMALL, trajectory, [0.05], RESOLUTIONS, num_starts=2)
    four = window_metrics(SMALL, trajectory, [0.05], RESOLUTIONS, num_starts=2, tau_bins=4)
    assert np.all(four.coverage <= one.coverage + 1e-12)
    assert np.all(four.gap_cells >= one.gap_cells - 1e-9)


def test_gap_detects_clumped_sampling(trajectory: Trajectory) -> None:
    """Every TR playing the same shape unrotated leaves bigger gaps."""
    n = len(trajectory.tr_index)
    clumped = dataclasses.replace(
        trajectory,
        rotation_matrix=np.broadcast_to(np.eye(3), (n, 3, 3)).copy(),
        shape_index=np.zeros(n, dtype=np.int64),
    )
    design = window_metrics(SMALL, trajectory, [0.1], RESOLUTIONS, num_starts=1)
    bad = window_metrics(SMALL, clumped, [0.1], RESOLUTIONS, num_starts=1)
    assert np.all(bad.gap_cells > design.gap_cells)
    assert np.all(bad.coverage < design.coverage)


def test_cli_coverage(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config_path = tmp_path / "small.toml"
    config_path.write_text(SMALL.to_toml())
    shapes_path = tmp_path / "shapes.npz"
    seq_path = tmp_path / "small.seq"
    assert main(["shapes", str(config_path), "-o", str(shapes_path), "--jobs", "1"]) == 0
    assert main(["sequence", str(shapes_path), "-o", str(seq_path)]) == 0
    args = ["coverage", str(seq_path), "--windows", "0.02,0.05", "--resolutions-mm", "4,8"]
    capsys.readouterr()
    assert main([*args, "--starts", "2", "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["windows_s"] == [0.02, 0.05]
    assert report["resolutions_m"] == [0.008, 0.004]
    assert np.array(report["coverage"]).shape == (2, 2, 2)
    assert main([*args, "--starts", "2"]) == 0
    assert "gap:" in capsys.readouterr().out


def test_ktau_metrics_on_a_known_trajectory() -> None:
    """One shape whose |k| rises linearly with tau, from 0 to k_max."""
    config = SMALL
    n = 1000
    tau = np.linspace(0.0, 10e-3, n, endpoint=False)
    radius = np.linspace(0.0, config.k_max_per_m, n, endpoint=False)
    k_shape = np.zeros((1, n, 3))
    k_shape[0, :, 0] = radius
    trajectory = Trajectory(
        k_shape_per_m=k_shape,
        adc_time_in_tr_s=tau + config.rf_center_s,
        time_since_excitation_s=tau,
        tr_index=np.arange(3),
        tr_start_s=np.arange(3, dtype=np.float64) * config.readout.tr_s,
        shape_index=np.zeros(3, dtype=np.int64),
        rotation_matrix=np.broadcast_to(np.eye(3), (3, 3, 3)).copy(),
        rf_phase_rad=np.zeros(3),
    )
    m = ktau_metrics(config, trajectory, shells=10, tau_bin_s=1e-3, early_s=2e-3)
    # Shell i is reached only during tau bin i: the diagonal is full.
    np.testing.assert_array_equal(m.counts, np.diag(np.full(10, 3 * 100)))
    assert m.empty_fraction == pytest.approx(0.9)
    assert m.empty_fraction_early == pytest.approx(0.9)
    assert m.empty_fraction_outer_half == pytest.approx(0.9)
    # Shell 0 is empty from 1 ms to 10 ms; shell 9 from 0 to 9 ms.
    np.testing.assert_allclose(m.max_tau_gap_s[[0, 9]], [9e-3, 9e-3])
    np.testing.assert_allclose(m.max_tau_gap_s[5], 5e-3)


def test_ktau_more_shapes_fill_the_plane() -> None:
    few = SMALL.with_overrides(["shapes.num_shapes=1"])
    many = SMALL.with_overrides(["shapes.num_shapes=4"])
    metrics = [
        ktau_metrics(c, make_trajectory(c, make_schedule(c), optimize_shapes(c, jobs=1)))
        for c in (few, many)
    ]
    assert metrics[1].empty_fraction < metrics[0].empty_fraction
