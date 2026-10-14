from pathlib import Path

import numpy as np
import pypulseq as pp
import pytest

from relaxed_rosettes.cli import main
from relaxed_rosettes.config import GAMMA_HZ_PER_T, Config
from relaxed_rosettes.optimize import (
    Shape,
    ShapeOptimizationError,
    k_weights,
    load_shapes,
    num_zero_samples,
    optimize_shape,
    optimize_shapes,
    shape_metrics,
)
from relaxed_rosettes.system import make_opts
from relaxed_rosettes.targets import make_targets

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
    shape = optimize_shape(config, make_targets(config)[0])
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
