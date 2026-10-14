from pathlib import Path

import numpy as np
import pytest

from relaxed_rosettes.cli import main
from relaxed_rosettes.config import Config
from relaxed_rosettes.targets import (
    ShapeTargets,
    blue_noise_1d,
    blue_noise_directions,
    load_targets,
    make_targets,
    target_times,
)

# Fewer shapes keep the tests fast; each shape is independent of the count.
CONFIG = Config().with_overrides(["shapes.num_shapes=4"])


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
    # Including the last crossing, which sets the spoil segment.
    assert len(set(crossings[:, -1])) == len(targets)


def test_without_jitter_time_follows_sqrt_radius() -> None:
    config = CONFIG.with_overrides(["shapes.timing_jitter=0"])
    shape = make_targets(config)[0]
    t = target_times(config, shape.k, np.random.default_rng(0))
    durations = np.diff(t[:-1])  # segments 1 .. 2E-1, each from a target to the next
    radii = np.repeat(np.linalg.norm(shape.tips, axis=1), 2)[1:]
    ratio = durations / np.sqrt(radii)
    raster = config.system.grad_raster_time_s
    np.testing.assert_allclose(ratio, ratio.mean(), atol=2 * raster / np.sqrt(radii).min())


def test_reproducible_and_independent_of_shape_count(targets: list[ShapeTargets]) -> None:
    again = make_targets(CONFIG.with_overrides(["shapes.num_shapes=2"]))
    for a, b in zip(again, targets[:2], strict=True):
        np.testing.assert_array_equal(a.k, b.k)
        np.testing.assert_array_equal(a.t_s, b.t_s)
    other_seed = make_targets(CONFIG.with_overrides(["seed=1"]))
    assert not np.array_equal(other_seed[0].k, targets[0].k)


def test_cli_writes_loadable_file(tmp_path: Path) -> None:
    path = tmp_path / "targets.npz"
    assert main(["targets", "--set", "shapes.num_shapes=2", "-o", str(path)]) == 0
    config, loaded = load_targets(path)
    assert config == Config().with_overrides(["shapes.num_shapes=2"])
    for a, b in zip(loaded, make_targets(config), strict=True):
        np.testing.assert_array_equal(a.k, b.k)
        np.testing.assert_array_equal(a.t_s, b.t_s)
