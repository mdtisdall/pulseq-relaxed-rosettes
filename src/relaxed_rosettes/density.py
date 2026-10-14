"""Tip-radius distributions derived from a target k-space sampling density.

Every petal goes from k = 0 out to its tip radius R and back, so it crosses
every radius below R twice. With ``T(rho)`` the fraction of tips beyond
radius ``rho`` (the tip *survival function*) and ``v(rho)`` the speed of the
trajectory at that radius, the number of samples per unit radius is

    n(rho)  ~  T(rho) / v(rho).

So a target radial sample profile ``n`` fixes the tips: ``T ~ n * v``. Since
every petal passes through the center, ``T`` (and so the achievable ``n * v``)
can only decrease with radius; a target that increases is replaced by its
smallest non-increasing majorant (the trajectory samples at least the target
everywhere, and more near the center, which it cannot avoid).

Two targets:

- ``exponent``: the resolution-time exponent ``alpha``. For a window of scan
  time to cover the k-space ball of a resolution ``res`` in a time
  ``W(res) ~ res ** -alpha``, the samples inside radius ``rho`` must grow as
  ``rho ** (3 - alpha)`` (the Nyquist count grows as ``rho ** 3``), so
  ``n(rho) ~ rho ** (2 - alpha)``, ``2 <= alpha < 3``. The law applies over
  the radii of the resolutions of interest; inside the coarsest, the target
  is flat at the level that gives that ball its share of samples, and
  beyond the finest it is 0.
- ``filter``: SNR-optimal for a set of retrospective reconstructions.
  Reconstruction at resolution ``r`` applies a k-space filter ``w_r`` (Hann,
  cut off at ``1 / (2 r)``); with sampling density ``D(k)`` its noise
  variance is proportional to ``integral(w_r(k)**2 / D(k))``. Minimizing the
  ``pi(r)``-weighted sum over resolutions for a fixed number of samples gives
  ``D(k) ~ sqrt(sum_r pi(r) w_r(k)**2)`` (for one filter, the classic
  acquisition-weighting result ``D ~ w``), and ``n(rho) = rho**2 D(rho)``.

``v(rho)`` depends on the optimized waveforms, so it is estimated from a few
calibration shapes and the tips are recomputed (a fixed-point iteration).
All radii here are in units of k_max.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from relaxed_rosettes.config import Config, TipDensity

FloatArray = NDArray[np.float64]

GRID_POINTS = 401
"""Points of the radial grid on [0, 1] (units of k_max)."""

FILTER_RESOLUTIONS = 64
"""Resolutions on the log grid of the filter model."""


@dataclass(frozen=True)
class TipSurvival:
    """Fraction of tips beyond each radius, on a grid over [0, 1] k_max.

    ``survival[0] == 1``; it is non-increasing, 1 below the minimum tip
    radius and 0 beyond the maximum. ``survival`` just below the maximum tip
    radius is the fraction of tips placed exactly at the maximum.
    """

    grid: FloatArray
    survival: FloatArray
    tip_radius_max: float

    def radii(self, u: FloatArray) -> FloatArray:
        """Tip radii for quantiles ``u`` in [0, 1) (inverse CDF)."""
        cdf = 1.0 - self.survival
        # Positions where the CDF increases; the rest of the mass is at the max.
        result = np.interp(u, cdf, self.grid)
        top = cdf[self.grid < self.tip_radius_max].max()
        out: FloatArray = np.where(u >= top, self.tip_radius_max, result)
        return out


def radial_grid() -> FloatArray:
    return np.linspace(0.0, 1.0, GRID_POINTS)


def exponent_profile(
    grid: FloatArray, alpha: float, radius_low: float, radius_high: float
) -> FloatArray:
    """Target samples per unit radius for ``W(res) ~ res ** -alpha``.

    The law holds on ``[radius_low, radius_high]`` (units of k_max). For it to
    hold at the coarsest resolution, the samples inside ``radius_low`` must
    total ``integral_0^low rho ** (2 - alpha)`` = ``low ** (3 - alpha) /
    (3 - alpha)``; the target is flat there at that total (achievable: it is
    not below the law at ``radius_low``). Beyond ``radius_high`` it is 0.
    """
    inner = radius_low ** (2.0 - alpha) / (3.0 - alpha)
    law = np.maximum(grid, radius_low) ** (2.0 - alpha)
    profile: FloatArray = np.where(
        grid < radius_low, inner, np.where(grid <= radius_high, law, 0.0)
    )
    return profile


def hann(radius: FloatArray, cutoff: float) -> FloatArray:
    """Hann filter in |k| that reaches 0 at ``cutoff``."""
    w: FloatArray = np.where(radius < cutoff, 0.5 * (1.0 + np.cos(np.pi * radius / cutoff)), 0.0)
    return w


def filter_density(
    grid: FloatArray,
    design_resolution_m: float,
    resolution_min_m: float,
    resolution_max_m: float,
    prior_power: float,
) -> FloatArray:
    """SNR-optimal volumetric density ``sqrt(sum_r pi(r) w_r**2)``.

    Resolutions are log-spaced from ``resolution_min_m`` (finest) to
    ``resolution_max_m``, with weight ``pi(r) ~ r ** prior_power`` per unit
    of log resolution (0: every octave of resolution equally important).
    """
    resolutions = np.geomspace(resolution_min_m, resolution_max_m, FILTER_RESOLUTIONS)
    weights = resolutions**prior_power
    weights /= weights.sum()
    # A resolution r reaches k = 1 / (2 r) = (design resolution / r) k_max.
    power = sum(
        wt * hann(grid, design_resolution_m / r) ** 2
        for wt, r in zip(weights, resolutions, strict=True)
    )
    density: FloatArray = np.sqrt(power)
    return density


def target_profile(config: Config, grid: FloatArray) -> FloatArray:
    """Target samples per unit radius (arbitrary scale) of the config's model."""
    s = config.shapes
    design = config.geometry.resolution_m
    if s.tip_density is TipDensity.EXPONENT:
        return exponent_profile(
            grid,
            s.resolution_time_exponent,
            design / s.density_resolution_max_m,
            design / s.density_resolution_min_m,
        )
    if s.tip_density is TipDensity.FILTER:
        density = filter_density(
            grid,
            design,
            s.density_resolution_min_m,
            s.density_resolution_max_m,
            s.filter_prior_power,
        )
        profile: FloatArray = grid**2 * density
        return profile
    raise ValueError(f"tip density {s.tip_density} has no target profile")


def survival_from_profile(
    grid: FloatArray,
    profile: FloatArray,
    speed: FloatArray,
    tip_radius_min: float,
    tip_radius_max: float,
) -> TipSurvival:
    """Tip survival ``T ~ profile * speed``, made achievable.

    ``T`` is the smallest non-increasing majorant of ``profile * speed`` over
    the tip range, constant below the minimum tip radius and 0 beyond the
    maximum, and normalized to 1 at the center.
    """
    raw = profile * speed
    in_range = (grid >= tip_radius_min) & (grid < tip_radius_max)
    t = np.where(in_range, raw, 0.0)
    # Running maximum from the outside in: the smallest non-increasing majorant.
    t = np.maximum.accumulate(t[::-1])[::-1]
    below = grid < tip_radius_min
    if below.any() and in_range.any():
        t[below] = t[in_range][0]
    if not t[0] > 0:
        raise ValueError("the target sampling profile is zero over the tip range")
    survival = t / t[0]
    return TipSurvival(grid=grid, survival=survival, tip_radius_max=tip_radius_max)


def power_survival(grid: FloatArray, power: float, r_min: float, r_max: float) -> TipSurvival:
    """Survival of the ``power`` model: radii ``u ** power``, u uniform."""
    u_min, u_max = r_min ** (1 / power), r_max ** (1 / power)
    u = np.clip(grid, r_min, r_max) ** (1 / power)
    survival = (u_max - u) / (u_max - u_min)
    survival = np.where(grid < r_min, 1.0, np.where(grid >= r_max, 0.0, survival))
    return TipSurvival(grid=grid, survival=survival, tip_radius_max=r_max)


def estimate_speed(
    grid: FloatArray,
    sample_radii: list[FloatArray],
    smoothing_points: int = 9,
) -> FloatArray:
    """Relative trajectory speed at each radius, from optimized shapes.

    ``sample_radii`` holds, per shape, the radius (units of k_max) of every
    ADC sample in time order. The samples per unit radius at ``rho`` are
    (number of times the path crosses ``rho``) / (speed at ``rho``); both are
    counted from the samples.
    """
    edges = np.concatenate([[0.0], (grid[1:] + grid[:-1]) / 2, [1.0]])
    counts = np.zeros(len(grid))
    crossings = np.zeros(len(grid))
    for samples in sample_radii:
        counts += np.histogram(samples[samples < 1.0], bins=edges)[0]
        above = samples[:, None] > grid[None, :]
        crossings += np.count_nonzero(np.diff(above, axis=0), axis=0)
    density = counts / np.diff(edges)
    valid = (density > 0) & (crossings > 0)
    speed = np.ones(len(grid))
    speed[valid] = crossings[valid] / density[valid]
    # Fill radii that no sample crossed, and smooth.
    if valid.any():
        speed = np.interp(grid, grid[valid], speed[valid])
    kernel = np.ones(smoothing_points) / smoothing_points
    padded = np.pad(speed, smoothing_points // 2, mode="edge")
    smooth: FloatArray = np.convolve(padded, kernel, mode="valid")
    return smooth / smooth.mean()
