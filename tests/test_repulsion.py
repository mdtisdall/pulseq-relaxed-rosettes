import math

import numpy as np
import pytest
from scipy.spatial.distance import cdist, pdist

from relaxed_rosettes import repulsion
from relaxed_rosettes.config import Config, ConfigError
from relaxed_rosettes.optimize import (
    Shape,
    adc_samples,
    cells_to_kmax,
    optimize_shapes,
    sample_kspace,
    shape_metrics,
    solve_shape,
)
from relaxed_rosettes.repulsion import (
    ACCEPT_RATIO,
    GROW_RATIO,
    MAX_TRUST_CELLS,
    SHRINK_RATIO,
    ShellTarget,
    energy,
    energy_distance,
    initial_target,
    model_target,
    repel_shape,
    sample_operator,
    shell_distance,
    slew_energy,
    target_self_energy,
)
from relaxed_rosettes.targets import make_targets
from relaxed_rosettes.trajectory import kspace_at_times

SMALL = Config().with_overrides(
    [
        "readout.tr_s=0.01",
        "readout.adc_num_samples=2000",
        "shapes.echoes_per_tr=4",
        "shapes.num_shapes=1",
    ]
)
RELAXED = SMALL.with_overrides(["shapes.crossing_tolerance_cells=1", "shapes.tip_width=0.25"])
REPEL = RELAXED.with_overrides(["shapes.repulsion_iterations=4"])
TOL = 1e-6


def uniform_on_shells(target: ShellTarget, count: int, rng: np.random.Generator) -> np.ndarray:
    radii = rng.choice(target.radii, size=count, p=target.weights)
    u = rng.standard_normal((count, 3))
    return radii[:, None] * u / np.linalg.norm(u, axis=1, keepdims=True)


@pytest.mark.parametrize(("r", "s"), [(0.2, 0.7), (0.7, 0.2), (0.5, 0.5), (0.0, 0.3)])
def test_shell_distance_matches_monte_carlo(r: float, s: float) -> None:
    rng = np.random.default_rng(0)
    y = uniform_on_shells(ShellTarget(np.array([s]), np.array([1.0])), 200_000, rng)
    expected = np.linalg.norm(y - np.array([r, 0.0, 0.0]), axis=1).mean()
    d, _ = shell_distance(np.array([r]), np.array([s]))
    assert d[0] == pytest.approx(expected, rel=3e-3)


def test_shell_distance_derivative() -> None:
    r = np.array([0.1, 0.35, 0.8])
    s = np.array([0.4])
    h = 1e-6
    _, derivative = shell_distance(r, s)
    numeric = (shell_distance(r + h, s)[0] - shell_distance(r - h, s)[0]) / (2 * h)
    np.testing.assert_allclose(derivative, numeric, rtol=1e-6)


def test_energy_gradient_matches_finite_differences() -> None:
    rng = np.random.default_rng(1)
    x = rng.uniform(-0.6, 0.6, (40, 3))
    target = initial_target(np.linalg.norm(rng.uniform(-0.6, 0.6, (500, 3)), axis=1))
    _, gradient = energy(x, target)
    h = 1e-6
    numeric = np.zeros_like(x)
    for i in range(len(x)):
        for j in range(3):
            step = np.zeros_like(x)
            step[i, j] = h
            numeric[i, j] = (energy(x + step, target)[0] - energy(x - step, target)[0]) / (2 * h)
    np.testing.assert_allclose(gradient, numeric, rtol=1e-5, atol=1e-10)


def test_blocked_energy_matches_direct_sum(monkeypatch: pytest.MonkeyPatch) -> None:
    rng = np.random.default_rng(2)
    x = rng.uniform(-0.5, 0.5, (300, 3))
    target = ShellTarget(np.array([0.2, 0.6]), np.array([0.3, 0.7]))
    monkeypatch.setattr(repulsion, "BLOCK", 64)  # 300 = 4 full blocks + 44
    value, gradient = energy(x, target)
    m = len(x)
    phi = shell_distance(np.linalg.norm(x, axis=1)[:, None], target.radii[None, :])[0]
    direct = (phi @ target.weights).mean() - pdist(x).sum() / m**2
    assert value == pytest.approx(direct, rel=1e-10)
    d = cdist(x, x)
    np.fill_diagonal(d, np.inf)
    push = (x[:, None, :] - x[None, :, :]) / d[..., None]
    radial = (shell_distance(np.linalg.norm(x, axis=1)[:, None], target.radii[None, :])[1]) @ (
        target.weights
    )
    expected = radial[:, None] * x / np.linalg.norm(x, axis=1)[:, None] / m - push.sum(1) / m**2
    np.testing.assert_allclose(gradient, expected, rtol=1e-8, atol=1e-12)


def test_energy_distance_is_small_for_samples_of_the_target() -> None:
    rng = np.random.default_rng(3)
    target = ShellTarget(np.array([0.1, 0.5, 0.9]), np.array([0.2, 0.3, 0.5]))
    assert target_self_energy(target) > 0
    matched = energy_distance(uniform_on_shells(target, 3000, rng), target)
    other = ShellTarget(np.array([0.1]), np.array([1.0]))
    mismatched = energy_distance(uniform_on_shells(other, 3000, rng), target)
    assert abs(matched) < 0.01 < mismatched


def test_model_target_follows_the_tip_density_profile() -> None:
    config = SMALL.with_overrides(["shapes.tip_density=exponent"])
    target = model_target(config)
    assert target.weights.sum() == pytest.approx(1.0)
    # Samples per unit radius fall as radius ** (2 - alpha) beyond the
    # coarsest resolution (1/8 of k_max).
    outer = target.weights[(target.radii > 0.13) & (target.radii < 1)]
    assert np.all(np.diff(outer) < 0)


@pytest.fixture(scope="module")
def relaxed_shape() -> Shape:
    return solve_shape(RELAXED, make_targets(RELAXED)[0])


def test_sample_operator_matches_played_waveform(relaxed_shape: Shape) -> None:
    g = relaxed_shape.g_mT_per_m
    r = SMALL.readout
    t = SMALL.adc_start_s + (np.arange(r.adc_num_samples, dtype=np.float64) + 0.5) * r.adc_dwell_s
    expected = kspace_at_times(g, SMALL.system.grad_raster_time_s, t) / SMALL.k_max_per_m
    np.testing.assert_allclose(sample_operator(SMALL)(SMALL, g), expected, atol=1e-12)


@pytest.fixture(scope="module")
def repelled(relaxed_shape: Shape) -> tuple[Shape, list[repulsion.RepulsionStep]]:
    return repel_shape(REPEL, relaxed_shape)


def test_repulsion_keeps_every_constraint(
    repelled: tuple[Shape, list[repulsion.RepulsionStep]],
) -> None:
    shape, history = repelled
    assert len(history) == REPEL.shapes.repulsion_iterations
    g = REPEL.gradients
    m = shape_metrics(REPEL, shape)
    assert m.max_grad_mT_per_m <= g.max_grad_mT_per_m * (1 + TOL)
    assert m.max_slew_T_per_m_per_s <= g.max_slew_T_per_m_per_s * (1 + TOL)
    assert m.max_boundary_slew_T_per_m_per_s <= g.boundary_slew_T_per_m_per_s * (1 + TOL)
    assert m.max_target_error_kmax < TOL
    assert m.max_grad_in_rf_window_mT_per_m < TOL
    assert m.end_grad_mT_per_m < TOL
    k = sample_kspace(REPEL, shape.g_mT_per_m)[adc_samples(REPEL)]
    # The ball is a cone constraint, met to the solver tolerance (~1e-3 cells).
    assert np.linalg.norm(k, axis=1).max() <= 1 + 1e-5


def test_repulsion_lowers_the_energy(
    relaxed_shape: Shape, repelled: tuple[Shape, list[repulsion.RepulsionStep]]
) -> None:
    shape, history = repelled
    assert any(step.accepted for step in history)
    samples = sample_operator(REPEL)
    before = samples(REPEL, relaxed_shape.g_mT_per_m)
    after = samples(REPEL, shape.g_mT_per_m)
    target = initial_target(np.linalg.norm(before, axis=1))
    assert energy_distance(after, target) < energy_distance(before, target)
    energies = [step.energy for step in history]
    assert all(b <= a + 1e-15 for a, b in zip(energies, energies[1:], strict=False))
    # Spreading costs slew energy.
    assert slew_energy(shape.g_mT_per_m) > slew_energy(relaxed_shape.g_mT_per_m)
    assert shape.solve_time_s > relaxed_shape.solve_time_s


def test_repulsion_steps_stay_in_the_trust_region(
    relaxed_shape: Shape, repelled: tuple[Shape, list[repulsion.RepulsionStep]]
) -> None:
    shape, history = repelled
    adc = adc_samples(REPEL)
    moved = np.linalg.norm(
        sample_kspace(REPEL, shape.g_mT_per_m)[adc]
        - sample_kspace(REPEL, relaxed_shape.g_mT_per_m)[adc],
        axis=1,
    )
    budget = sum(step.trust_cells for step in history if step.accepted)
    assert moved.max() <= cells_to_kmax(REPEL, budget) + TOL


def test_trust_region_follows_the_ratio(
    repelled: tuple[Shape, list[repulsion.RepulsionStep]],
) -> None:
    _, history = repelled
    for step, following in zip(history, history[1:], strict=False):
        failed = math.isnan(step.ratio)
        assert step.accepted == (not failed and step.ratio >= ACCEPT_RATIO)
        if failed or step.ratio < SHRINK_RATIO:
            expected = step.trust_cells / 2
        elif step.ratio > GROW_RATIO:
            expected = min(2 * step.trust_cells, MAX_TRUST_CELLS)
        else:
            expected = step.trust_cells
        assert following.trust_cells == expected


def test_repulsion_with_point_targets_keeps_them() -> None:
    config = SMALL.with_overrides(["shapes.repulsion_iterations=2"])
    (shape,) = optimize_shapes(config, jobs=1)
    assert shape_metrics(config, shape).max_target_error_kmax < TOL
    (hairpin,) = optimize_shapes(SMALL, jobs=1)
    adc = adc_samples(config)
    ball = max(1.0, np.linalg.norm(sample_kspace(config, hairpin.g_mT_per_m)[adc], axis=1).max())
    k = sample_kspace(config, shape.g_mT_per_m)[adc]
    assert np.linalg.norm(k, axis=1).max() <= ball + TOL
    assert slew_energy(shape.g_mT_per_m) > slew_energy(hairpin.g_mT_per_m)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ("shapes.repulsion_iterations=-1", "repulsion_iterations must be >= 0"),
        ("shapes.repulsion_weight=0", "repulsion_weight must be > 0"),
        ("shapes.repulsion_trust_cells=0", "repulsion_trust_cells must be > 0"),
        ("shapes.tube_cells=1", "does not support shapes.tube_cells"),
        ("shapes.repulsion_target=model", "needs tip_density exponent or filter"),
    ],
)
def test_invalid_repulsion_config(override: str, message: str) -> None:
    base = ["shapes.repulsion_iterations=2"] if "iterations" not in override else []
    with pytest.raises(ConfigError, match=message):
        SMALL.with_overrides([*base, override])
