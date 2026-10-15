from pathlib import Path

import numpy as np
import pypulseq as pp
import pytest

from relaxed_rosettes.cli import main
from relaxed_rosettes.config import GAMMA_HZ_PER_T, Config
from relaxed_rosettes.optimize import (
    Shape,
    ShapeOptimizationError,
    _optimize_one,
    _Work,
    adc_samples,
    calibrated_petal_time_factor,
    cells_to_kmax,
    k_weights,
    load_shapes,
    num_zero_samples,
    optimize_shape,
    optimize_shapes,
    sample_kspace,
    shape_metrics,
    shortest_petal_time_factor,
    solve_shape,
    target_kspace,
)
from relaxed_rosettes.system import make_opts
from relaxed_rosettes.targets import make_shape_targets, make_targets, shape_streams
from relaxed_rosettes.trajectory import kspace_at_times

# A short TR with few echoes keeps most tests fast.
SMALL = Config().with_overrides(
    [
        "readout.tr_s=0.01",
        "readout.adc_num_samples=2000",
        "shapes.echoes_per_tr=4",
        "shapes.num_shapes=2",
    ]
)
TOL = 1e-6


@pytest.fixture(scope="module")
def small_shapes() -> list[Shape]:
    return optimize_shapes(SMALL, jobs=1)


@pytest.mark.parametrize("edge", [1, 2, 3, 57, 99, 100])
def test_k_weights_integrate_played_waveform(edge: int) -> None:
    n = 100
    g = np.random.default_rng(edge).standard_normal(n)
    # The waveform pypulseq plays: first = 0, samples at raster centers, last = 0.
    t_points = np.concatenate([[0.0], np.arange(n) + 0.5, [float(n)]])
    g_points = np.concatenate([[0.0], g, [0.0]])
    # The trapezoid rule is exact on the corners of a piecewise-linear waveform.
    t = np.union1d(t_points[t_points < edge], [edge])
    integral = np.trapezoid(np.interp(t, t_points, g_points), t)
    assert k_weights(n, edge) @ g == pytest.approx(integral, abs=1e-8)


def test_small_shapes_meet_constraints(small_shapes: list[Shape]) -> None:
    g = SMALL.gradients
    for shape in small_shapes:
        assert shape.status == "optimal"
        m = shape_metrics(SMALL, shape)
        assert m.max_grad_mT_per_m <= g.max_grad_mT_per_m * (1 + TOL)
        assert m.max_slew_T_per_m_per_s <= g.max_slew_T_per_m_per_s * (1 + TOL)
        assert m.max_boundary_slew_T_per_m_per_s <= g.boundary_slew_T_per_m_per_s * (1 + TOL)
        assert m.max_target_error_kmax < TOL
        assert m.max_grad_in_rf_window_mT_per_m < TOL
        assert m.end_grad_mT_per_m < TOL


def test_rf_window_is_gradient_free() -> None:
    raster = SMALL.system.grad_raster_time_s
    n_zero = num_zero_samples(SMALL)
    # The first free sample ramps from 0 starting half a raster earlier.
    assert (n_zero - 0.5) * raster >= SMALL.rf_end_s - 1e-12
    assert (n_zero - 1.5) * raster < SMALL.rf_end_s


def test_parallel_matches_serial(small_shapes: list[Shape]) -> None:
    parallel = optimize_shapes(SMALL, jobs=2)
    for a, b in zip(parallel, small_shapes, strict=True):
        np.testing.assert_allclose(a.g_mT_per_m, b.g_mT_per_m, atol=1e-9)


def test_unreachable_targets_raise() -> None:
    config = SMALL.with_overrides(
        [
            "gradients.max_slew_T_per_m_per_s=0.5",
            "gradients.boundary_slew_T_per_m_per_s=0.5",
        ]
    )
    with pytest.raises(ShapeOptimizationError):
        optimize_shape(config, make_targets(config)[0])


def test_pypulseq_kspace_hits_targets() -> None:
    """The full-size default shape, played by pypulseq, passes every target."""
    config = Config().with_overrides(["shapes.num_shapes=1"])
    # The provisional petal time factor, so as not to calibrate it here.
    shape = optimize_shape(config, make_targets(config, petal_time_factor=1.1)[0])
    system = make_opts(config)
    raster = config.system.grad_raster_time_s

    g_hz = shape.g_mT_per_m * 1e-3 * GAMMA_HZ_PER_T
    grads = [
        pp.make_arbitrary_grad(
            channel=ch, waveform=g_hz[:, i], first=0, last=0, delay=0, system=system
        )
        for i, ch in enumerate("xyz")
    ]
    rf = pp.make_block_pulse(
        flip_angle=np.deg2rad(config.excitation.flip_angle_deg),
        duration=config.excitation.rf_duration_s,
        delay=config.system.rf_dead_time_s,
        system=system,
        use="excitation",
    )
    # ADC samples (at delay + (i + 0.5) * dwell) on every raster edge from
    # the end of the RF pulse to the end of the TR, to read k at the targets.
    first_edge = int(np.ceil(config.adc_start_s / raster))
    num_samples = config.raster_points_per_tr - first_edge + 1
    adc = pp.make_adc(
        num_samples=num_samples,
        dwell=raster,
        delay=(first_edge - 0.5) * raster,
        system=system,
    )
    seq = pp.Sequence(system=system)
    seq.add_block(rf, *grads, adc)
    k_adc, _, _, _, t_adc = seq.calculate_kspace()

    edges = np.round(shape.targets.t_s / raster).astype(int)
    k_at_targets = k_adc[:, edges - first_edge].T / config.k_max_per_m
    np.testing.assert_allclose(t_adc[edges - first_edge], edges * raster, atol=1e-9)
    np.testing.assert_allclose(k_at_targets, shape.targets.k, atol=1e-6)


def test_cli_writes_loadable_file(tmp_path: Path, small_shapes: list[Shape]) -> None:
    path = tmp_path / "shapes.npz"
    config_path = tmp_path / "small.toml"
    config_path.write_text(SMALL.to_toml())
    assert main(["shapes", str(config_path), "-o", str(path), "--jobs", "1"]) == 0
    config, loaded = load_shapes(path)
    assert config == SMALL
    for a, b in zip(loaded, small_shapes, strict=True):
        np.testing.assert_allclose(a.g_mT_per_m, b.g_mT_per_m, atol=1e-9)
        np.testing.assert_array_equal(a.targets.t_s, b.targets.t_s)


FITTED = SMALL.with_overrides(["shapes.echoes_per_tr=0", "shapes.timing_calibration_shapes=2"])


def test_fitted_shapes_calibrate_and_solve() -> None:
    factor = calibrated_petal_time_factor(FITTED, jobs=1)
    # The straight-path petal time is a lower bound; the turns at k = 0 add a little.
    assert 1.0 < factor < 1.5
    g = FITTED.gradients
    for shape in optimize_shapes(FITTED, jobs=1):
        assert shape.targets.num_echoes >= 2
        m = shape_metrics(FITTED, shape)
        assert m.max_slew_T_per_m_per_s <= g.max_slew_T_per_m_per_s * (1 + TOL)
        assert m.max_target_error_kmax < TOL


def test_infeasible_fitted_shape_drops_tips() -> None:
    # Too small a factor: the petals get less than their shortest time.
    stream = shape_streams(FITTED)[0]
    targets = make_shape_targets(FITTED, stream, None, 0.8)
    with pytest.raises(ShapeOptimizationError):
        optimize_shape(FITTED, targets)
    shape = _optimize_one(_Work(FITTED, targets, 0, None, 0.8))
    assert shape.targets.num_echoes < targets.num_echoes
    assert shape_metrics(FITTED, shape).max_target_error_kmax < TOL


RELAXED = SMALL.with_overrides(["shapes.crossing_tolerance_cells=1", "shapes.tip_width=0.25"])


def slew_energy(shape: Shape) -> float:
    return float(np.sum(np.diff(shape.g_mT_per_m, axis=0) ** 2))


@pytest.fixture(scope="module")
def relaxed_pair() -> tuple[Shape, Shape]:
    """The hairpin and the relaxed waveform of the same targets."""
    targets = make_targets(RELAXED)[0]
    return optimize_shape(RELAXED, targets, relax=False), solve_shape(RELAXED, targets)


def test_sample_kspace_matches_played_waveform(small_shapes: list[Shape]) -> None:
    g = small_shapes[0].g_mT_per_m
    raster = SMALL.system.grad_raster_time_s
    t = (np.arange(len(g), dtype=np.float64) + 0.5) * raster
    expected = kspace_at_times(g, raster, t) / SMALL.k_max_per_m
    np.testing.assert_allclose(sample_kspace(SMALL, g), expected, atol=1e-12)


def test_relaxed_shape_meets_its_regions(relaxed_pair: tuple[Shape, Shape]) -> None:
    hairpin, shape = relaxed_pair
    s = RELAXED.shapes
    assert shape_metrics(RELAXED, shape).max_target_error_kmax < TOL
    k = target_kspace(RELAXED, shape.g_mT_per_m, shape.targets)
    tolerance = cells_to_kmax(RELAXED, s.crossing_tolerance_cells)
    assert np.linalg.norm(k[1:-1:2], axis=1).max() <= tolerance + TOL
    radii = np.linalg.norm(shape.targets.tips, axis=1)
    assert np.all(np.linalg.norm(k[0:-1:2], axis=1) >= radii - TOL)
    assert np.all(
        np.linalg.norm(k[0:-1:2] - shape.targets.tips, axis=1) <= s.tip_width * radii + TOL
    )
    np.testing.assert_allclose(k[-1], shape.targets.spoil, atol=TOL)
    adc = adc_samples(RELAXED)
    assert np.linalg.norm(sample_kspace(RELAXED, shape.g_mT_per_m)[adc], axis=1).max() <= 1 + TOL
    # The point targets are inside the regions, so relaxing never costs slew energy.
    assert slew_energy(shape) < slew_energy(hairpin)


def test_relinearization_never_increases_slew_energy(relaxed_pair: tuple[Shape, Shape]) -> None:
    _, shape = relaxed_pair
    config = RELAXED.with_overrides(["shapes.tip_relinearizations=2"])
    again = solve_shape(config, shape.targets)
    assert slew_energy(again) <= slew_energy(shape) * (1 + 1e-6)
    assert shape_metrics(config, again).max_target_error_kmax < TOL


def test_tube_keeps_the_path_near_the_hairpin(relaxed_pair: tuple[Shape, Shape]) -> None:
    hairpin, shape = relaxed_pair
    config = RELAXED.with_overrides(["shapes.tube_cells=0.5"])
    tubed = solve_shape(config, shape.targets)
    distance = np.linalg.norm(
        sample_kspace(config, tubed.g_mT_per_m) - sample_kspace(config, hairpin.g_mT_per_m), axis=1
    )
    assert distance.max() <= cells_to_kmax(config, 0.5) + TOL
    # Without a tube, the relaxed path goes farther.
    free = np.linalg.norm(
        sample_kspace(config, shape.g_mT_per_m) - sample_kspace(config, hairpin.g_mT_per_m), axis=1
    )
    assert free.max() > distance.max()


def test_relaxed_targets_never_calibrate_longer_petals() -> None:
    stream = shape_streams(FITTED)[0]
    hairpin = shortest_petal_time_factor(FITTED, stream)
    config = FITTED.with_overrides(["shapes.crossing_tolerance_cells=1", "shapes.tip_width=0.25"])
    relaxed = shortest_petal_time_factor(config, stream)
    # The relaxed problem is feasible wherever the hairpin is (the bisection
    # uses the hairpin as proof when the relaxed solve fails).
    assert 1.0 <= relaxed <= hairpin
