from pathlib import Path

import numpy as np
import pytest

from relaxed_rosettes.cli import main
from relaxed_rosettes.config import Config
from relaxed_rosettes.targets import (
    ShapeTargets,
    blue_noise_1d,
    blue_noise_directions,
    fitted_num_tips,
    load_targets,
    make_targets,
    petal_times_s,
    petal_window_s,
    save_targets,
    shape_streams,
    stack_targets,
    unstack_targets,
)

# Fewer shapes keep the tests fast; each shape is independent of the count.
# A fixed tip count needs no calibration against the shape optimizer.
CONFIG = Config().with_overrides(["shapes.num_shapes=4", "shapes.echoes_per_tr=19"])
FITTED = Config().with_overrides(["shapes.num_shapes=8"])
FACTOR = 1.1


@pytest.fixture(scope="module")
def targets() -> list[ShapeTargets]:
    return make_targets(CONFIG)


def test_directions_are_unit_and_alternate() -> None:
    rng = np.random.default_rng(0)
    d = blue_noise_directions(rng, 20, -0.5)
    np.testing.assert_allclose(np.linalg.norm(d, axis=1), 1.0)
    assert np.all(np.sum(d[1:] * d[:-1], axis=1) < -0.5)


def test_blue_noise_1d_is_well_spread() -> None:
    rng = np.random.default_rng(0)
    v = np.sort(blue_noise_1d(rng, 19, 0.2, 1.0))
    assert v.min() >= 0.2 and v.max() < 1.0
    # Greedy farthest-point sampling keeps the gaps near the even spacing.
    assert np.diff(v).min() > 0.4 * (1.0 - 0.2) / 18


def test_positions(targets: list[ShapeTargets]) -> None:
    s = CONFIG.shapes
    for shape in targets:
        assert shape.k.shape == (2 * s.echoes_per_tr + 1, 3)
        np.testing.assert_array_equal(shape.k[1::2], 0.0)
        radii = np.linalg.norm(shape.tips, axis=1)
        assert np.all(radii >= s.tip_radius_min) and np.all(radii <= s.tip_radius_max)
        assert np.linalg.norm(shape.spoil) == pytest.approx(s.spoil_moment_kmax)
        directions = shape.k[0::2] / np.linalg.norm(shape.k[0::2], axis=1, keepdims=True)
        assert np.all(np.sum(directions[1:] * directions[:-1], axis=1) < s.max_consecutive_tip_dot)


def test_times(targets: list[ShapeTargets]) -> None:
    raster = CONFIG.system.grad_raster_time_s
    for shape in targets:
        t = shape.t_s
        assert np.all(np.diff(t) > 0)
        np.testing.assert_allclose(t / raster, np.round(t / raster), atol=1e-9)
        assert t[0] > CONFIG.rf_end_s + CONFIG.gradients.boundary_duration_s
        assert shape.crossing_times_s[-1] < CONFIG.adc_end_s
        assert t[-1] == pytest.approx(CONFIG.readout.tr_s)


def test_shapes_differ_in_crossing_times(targets: list[ShapeTargets]) -> None:
    crossings = np.stack([shape.crossing_times_s for shape in targets])
    assert len({tuple(c) for c in crossings}) == len(targets)
    # The petals fill the window: the last crossing is at its end.
    _, end = petal_window_s(CONFIG)
    np.testing.assert_allclose(crossings[:, -1], end, atol=CONFIG.system.grad_raster_time_s)


def test_petal_time_follows_petal_model(targets: list[ShapeTargets]) -> None:
    raster = CONFIG.system.grad_raster_time_s
    for shape in targets:
        petals = np.diff(np.concatenate([[shape.t_s[0]], shape.crossing_times_s]))[1:]
        model = petal_times_s(CONFIG, np.linalg.norm(shape.tips, axis=1))[1:]
        ratio = petals / model
        np.testing.assert_allclose(ratio, ratio.mean(), atol=2 * raster / model.min())


def test_petal_times_regimes() -> None:
    g = CONFIG.gradients
    speed = 42.576e6 * g.max_grad_mT_per_m * 1e-3 / CONFIG.k_max_per_m
    accel = 42.576e6 * g.max_slew_T_per_m_per_s / CONFIG.k_max_per_m
    full = speed**2 / (2 * accel)
    assert 0.3 < full < 0.4  # at the default limits
    # Slew-limited below full speed: time proportional to sqrt(radius).
    small = petal_times_s(CONFIG, np.array([full / 4, full]))
    assert small[1] / small[0] == pytest.approx(2.0)
    # Continuous at full speed, then linear at 2 / speed per unit radius.
    t = petal_times_s(CONFIG, np.array([full - 1e-9, full + 1e-9, 0.8, 1.0]))
    assert t[0] == pytest.approx(t[1])
    assert (t[3] - t[2]) / 0.2 == pytest.approx(2 / speed)


def test_fitted_tips_fill_the_window() -> None:
    targets = make_targets(FITTED, petal_time_factor=FACTOR)
    start, end = petal_window_s(FITTED)
    factor = FITTED.shapes.segment_time_margin * FACTOR
    counts = [t.num_echoes for t in targets]
    assert len(set(counts)) > 1  # the counts differ between shapes
    assert 30 < np.mean(counts) < 60
    for t, stream in zip(targets, shape_streams(FITTED), strict=True):
        shortest = petal_times_s(FITTED, np.linalg.norm(t.tips, axis=1))
        assert factor * shortest.sum() <= end - start
        assert fitted_num_tips(FITTED, stream, factor) == t.num_echoes
        # Each petal gets at least margin * factor times its shortest time.
        petals = np.diff(np.concatenate([[start], t.crossing_times_s]))
        assert np.all(petals >= factor * shortest - 2 * FITTED.system.grad_raster_time_s)


def test_larger_margin_fits_fewer_tips() -> None:
    quiet = FITTED.with_overrides(["shapes.segment_time_margin=1.5"])

    def mean_count(config: Config) -> float:
        return float(np.mean([t.num_echoes for t in make_targets(config, None, FACTOR)]))

    assert mean_count(quiet) < 0.8 * mean_count(FITTED)


def test_stack_round_trip_with_different_counts(tmp_path: Path) -> None:
    targets = make_targets(FITTED, petal_time_factor=FACTOR)
    k, t_s = stack_targets(targets)
    assert k.shape == (len(targets), max(len(t.t_s) for t in targets), 3)
    path = tmp_path / "targets.npz"
    save_targets(path, FITTED, targets)
    _, loaded = load_targets(path)
    for a, b in zip([*unstack_targets(k, t_s), *loaded], [*targets, *targets], strict=True):
        np.testing.assert_array_equal(a.k, b.k)
        np.testing.assert_array_equal(a.t_s, b.t_s)


def test_reproducible_and_independent_of_shape_count(targets: list[ShapeTargets]) -> None:
    again = make_targets(CONFIG.with_overrides(["shapes.num_shapes=2"]))
    for a, b in zip(again, targets[:2], strict=True):
        np.testing.assert_array_equal(a.k, b.k)
        np.testing.assert_array_equal(a.t_s, b.t_s)
    other_seed = make_targets(CONFIG.with_overrides(["seed=1"]))
    assert not np.array_equal(other_seed[0].k, targets[0].k)


def test_cli_writes_loadable_file(tmp_path: Path) -> None:
    path = tmp_path / "targets.npz"
    overrides = ["shapes.num_shapes=2", "shapes.echoes_per_tr=19"]
    assert main(["targets", *[a for o in overrides for a in ("--set", o)], "-o", str(path)]) == 0
    config, loaded = load_targets(path)
    assert config == Config().with_overrides(overrides)
    for a, b in zip(loaded, make_targets(config), strict=True):
        np.testing.assert_array_equal(a.k, b.k)
        np.testing.assert_array_equal(a.t_s, b.t_s)


@pytest.mark.parametrize("power", [0.5, 1.0, 2.0])
def test_tip_radius_power(power: float) -> None:
    config = CONFIG.with_overrides([f"shapes.tip_radius_power={power}"])
    s = config.shapes
    radii = np.concatenate([np.linalg.norm(t.tips, axis=1) for t in make_targets(config)])
    assert radii.min() >= s.tip_radius_min - 1e-12
    assert radii.max() <= s.tip_radius_max + 1e-12
    # Blue noise in radius ** (1 / power): evenly spread in that variable.
    for shape in make_targets(config):
        u = np.sort(np.linalg.norm(shape.tips, axis=1) ** (1 / power))
        span = s.tip_radius_max ** (1 / power) - s.tip_radius_min ** (1 / power)
        assert np.diff(u).min() > 0.4 * span / (s.echoes_per_tr - 1)


def test_lower_power_moves_tips_outward() -> None:
    def mean_radius(power: float) -> float:
        config = CONFIG.with_overrides([f"shapes.tip_radius_power={power}"])
        return float(np.mean([np.linalg.norm(t.tips, axis=1) for t in make_targets(config)]))

    assert mean_radius(0.5) > mean_radius(1.0) > mean_radius(2.0)
