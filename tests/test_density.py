import numpy as np
import pytest

from relaxed_rosettes.config import Config
from relaxed_rosettes.density import (
    estimate_speed,
    exponent_profile,
    filter_density,
    hann,
    power_survival,
    radial_grid,
    survival_from_profile,
)
from relaxed_rosettes.targets import make_targets, tip_radii

GRID = radial_grid()
Q = (np.arange(4000) + 0.5) / 4000


def test_exponent_profile_regions() -> None:
    p = exponent_profile(GRID, 2.5, 0.125, 0.8)
    # Inside the coarsest ball: flat, with the power law's total.
    assert np.all(p[GRID < 0.125] == pytest.approx(0.125**-0.5 / 0.5))
    assert np.all(p[GRID > 0.8] == 0)
    inside = (GRID >= 0.125) & (GRID <= 0.8)
    np.testing.assert_allclose(p[inside], GRID[inside] ** -0.5)


def test_survival_is_an_achievable_tip_distribution() -> None:
    increasing = GRID**2  # more samples outside than inside: not achievable as is
    t = survival_from_profile(GRID, increasing, np.ones_like(GRID), 0.04, 1.0)
    assert t.survival[0] == 1.0
    assert np.all(np.diff(t.survival) <= 1e-12)
    # The majorant of an increasing profile is flat: every tip at the maximum.
    np.testing.assert_allclose(t.radii(Q), 1.0)


def test_exponent_profile_gives_the_resolution_time_law() -> None:
    """Samples inside each ball grow as rho ** (3 - alpha) over the range."""
    fine = np.linspace(0, 1, 200_001)
    p = exponent_profile(fine, 2.5, 0.125, 1.0)
    inside = np.cumsum(p) * (fine[1] - fine[0])
    for rho in (0.125, 0.25, 0.5, 1.0):
        assert inside[int(rho * 200_000)] / rho**0.5 == pytest.approx(2.0, rel=1e-3)


def test_constant_profile_puts_tips_at_max_and_alpha_2_matches() -> None:
    flat = survival_from_profile(GRID, np.ones_like(GRID), np.ones_like(GRID), 0.04, 0.9)
    np.testing.assert_allclose(flat.radii(Q), 0.9)
    alpha2 = exponent_profile(GRID, 2.0, 0.125, 1.0)
    t = survival_from_profile(GRID, alpha2, np.ones_like(GRID), 0.04, 1.0)
    np.testing.assert_allclose(t.radii(Q), 1.0)


def test_radii_invert_the_survival() -> None:
    uniform = power_survival(GRID, 1.0, 0.0, 1.0)
    # Within one grid step (the top quantiles land on the maximum).
    np.testing.assert_allclose(uniform.radii(Q), Q, atol=3e-3)
    # n ~ 1 - rho on [0, 1] (speed 1): survival 1 - rho, tips uniform.
    t = survival_from_profile(GRID, 1 - GRID, np.ones_like(GRID), 0.0, 1.0)
    np.testing.assert_allclose(t.radii(Q), Q, atol=3e-3)


def test_single_filter_density_is_the_filter() -> None:
    # One resolution: the SNR-optimal density is the filter itself.
    d = filter_density(GRID, 1e-3, 2e-3, 2e-3 * (1 + 1e-12), 0.0)
    np.testing.assert_allclose(d, hann(GRID, 0.5), atol=1e-9)


def test_estimate_speed_of_constant_speed_paths() -> None:
    # Radial petals at constant speed: samples evenly spaced along 0 -> R -> 0.
    rng = np.random.default_rng(0)
    tips = [rng.uniform(0.3, 1.0, 19) for _ in range(4)]
    samples = [
        np.concatenate([np.concatenate([np.arange(0, r, 1e-4), np.arange(r, 0, -1e-4)]) for r in t])
        for t in tips
    ]
    speed = estimate_speed(GRID, samples)
    inside = (GRID > 0.02) & (GRID < 0.95)
    np.testing.assert_allclose(speed[inside], np.median(speed[inside]), rtol=0.1)


def test_power_tip_radii_match_the_power_law() -> None:
    config = Config().with_overrides(["shapes.tip_radius_power=0.5"])
    r = tip_radii(config, Q)
    u = 0.04**2 + Q * (1 - 0.04**2)
    np.testing.assert_allclose(r, u**0.5)


SMALL = Config().with_overrides(
    [
        "readout.tr_s=0.01",
        "readout.adc_num_samples=2000",
        "shapes.echoes_per_tr=4",
        "shapes.num_shapes=3",
        "shapes.calibration_shapes=2",
        "shapes.calibration_iterations=1",
    ]
)


@pytest.mark.parametrize("model", ["exponent", "filter"])
def test_derived_models_make_targets(model: str) -> None:
    config = SMALL.with_overrides([f"shapes.tip_density={model}"])
    targets = make_targets(config)
    assert len(targets) == 3
    for t in targets:
        radii = np.linalg.norm(t.tips, axis=1)
        assert np.all(radii >= config.shapes.tip_radius_min - 1e-9)
        assert np.all(radii <= config.shapes.tip_radius_max + 1e-9)
