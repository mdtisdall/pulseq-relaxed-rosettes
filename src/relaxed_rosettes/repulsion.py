"""Attraction-repulsion spreading of the samples within each shape.

Step 2 of ``docs/sample-spreading-plan.md``. The ADC samples ``x_1 .. x_M`` of
one shape (units of k_max) are spread to follow a target density ``rho`` with
blue-noise spacing by lowering the energy (SPARKLING; Boyer et al. 2016)

    F(x) = (1 / M) sum_m phi(|x_m|) - (1 / (2 M^2)) sum_{m, n} |x_m - x_n|,

where ``phi(r)`` is the mean distance from a point at radius r to ``rho``.
``2 F - E|Y - Y'|`` (Y, Y' independent draws from ``rho``) is the energy
distance between the samples and ``rho``, which is >= 0 and 0 only when they
have the same distribution. The repulsion is the exact full-range sum,
O(M^2), in blocks, with each pair computed once.

``rho`` is isotropic, a mixture of thin shells (``ShellTarget``), for which
``phi`` is analytic: the mean distance from a point at radius r to a uniform
point on the sphere of radius s is ``max(r, s) + min(r, s)^2 / (3 max(r, s))``.

The samples are linear in the waveform, ``x = x(g)`` (``SampleOperator``,
exact). From a feasible waveform ``g``, each step of sequential convex
programming solves

    min  lam * grad F(x(g))^T x(g') + cost(g')

subject to every constraint of the solved shape (``waveform_constraints``,
and ``|k| <= 1`` in the ADC, or the hairpin's largest |k| if that is more)
and a trust region ``|k(g') - k(g)| <= delta``
at every gradient sample in the ADC. ``cost`` is the optimizer's objective
(``gradient_cost``: slew energy, plus the band penalty if the config has
penalty bands), so repulsion does not undo the band penalty. Every iterate
is a feasible waveform. The step is accepted when the merit
``lam * F + cost`` falls by at
least ``ACCEPT_RATIO`` of the decrease that the linear model predicts. The
trust region follows the ratio of actual to predicted decrease: it halves
below ``SHRINK_RATIO`` (or when the solve fails), doubles above
``GROW_RATIO`` (up to ``MAX_TRUST_CELLS``), and is kept between them.
Doubling on every accepted step instead made it alternate between a size
that is accepted with a low ratio and twice that size, which is rejected. With
relaxed tips, each step re-linearizes the reach constraint at the current
tips, as ``tip_relinearizations`` does.
"""

from __future__ import annotations

import dataclasses
import math
import time
from dataclasses import dataclass

import cvxpy as cp
import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from relaxed_rosettes.config import Config, RepulsionTarget
from relaxed_rosettes.density import radial_grid, target_profile
from relaxed_rosettes.optimize import (
    Shape,
    adc_samples,
    cells_to_kmax,
    gradient_cost,
    gradient_cost_value,
    kmax_per_mT_per_m_raster,
    relaxed,
    sample_kspace,
    target_kspace,
    waveform_constraints,
)

FloatArray = NDArray[np.float64]

INITIAL_TARGET_SHELLS = 40
"""Shells of the ``initial`` target (the radial histogram of the samples)."""

ACCEPT_RATIO = 0.1
"""Fraction of the predicted merit decrease that a step must achieve."""

SHRINK_RATIO = 0.25
"""Below this ratio of actual to predicted decrease, the trust region halves."""

GROW_RATIO = 0.75
"""Above this ratio, the trust region doubles (up to ``MAX_TRUST_CELLS``)."""

MAX_TRUST_CELLS = 8.0
"""Largest trust region, in k-space cells."""

MIN_TRUST_CELLS = 1 / 64
"""The steps stop when the trust region falls below this, in k-space cells."""

BLOCK = 512
"""Side of the blocks of the pairwise distance matrix computed at once."""


@dataclass(frozen=True)
class ShellTarget:
    """An isotropic density: weight ``weights[i]`` uniform on the sphere of
    radius ``radii[i]`` (units of k_max). The weights sum to 1."""

    radii: FloatArray
    weights: FloatArray


@dataclass(frozen=True)
class RepulsionStep:
    """One SCP step. ``energy`` and ``slew_energy`` are of the iterate after
    the step (the previous one if the step was rejected)."""

    energy: float
    slew_energy: float
    trust_cells: float
    """Trust region of this step."""
    accepted: bool
    ratio: float
    """Actual over predicted decrease of the merit; NaN if the solve failed."""
    solve_time_s: float


def initial_target(sample_radii: FloatArray) -> ShellTarget:
    """The radial histogram of the samples, in ``INITIAL_TARGET_SHELLS`` shells."""
    edges = np.linspace(0.0, 1.0, INITIAL_TARGET_SHELLS + 1)
    counts = np.histogram(np.minimum(sample_radii, 1.0), bins=edges)[0].astype(np.float64)
    return ShellTarget(radii=(edges[1:] + edges[:-1]) / 2, weights=counts / counts.sum())


def model_target(config: Config) -> ShellTarget:
    """The radial sample profile of the config's tip density
    (``relaxed_rosettes.density.target_profile``), on its radial grid."""
    grid = radial_grid()
    weights = target_profile(config, grid) * np.gradient(grid)
    return ShellTarget(radii=grid, weights=weights / weights.sum())


def shell_distance(r: FloatArray, s: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Mean distance from a point at radius ``r`` to a uniform point on the
    sphere of radius ``s`` (broadcast), and its derivative in ``r``."""
    inside = r < s
    outer = np.maximum(np.maximum(r, s), 1e-300)
    inner = np.minimum(r, s)
    distance = outer + inner**2 / (3 * outer)
    derivative = np.where(inside, 2 * r / (3 * outer), 1 - s**2 / (3 * outer**2))
    return distance, derivative


def target_self_energy(target: ShellTarget) -> float:
    """``E|Y - Y'|`` for independent draws from the target."""
    d, _ = shell_distance(target.radii[:, None], target.radii[None, :])
    return float(target.weights @ d @ target.weights)


def energy(x: FloatArray, target: ShellTarget) -> tuple[float, FloatArray]:
    """``F`` of the samples ``x`` (M, 3) and its gradient in ``x``."""
    m = len(x)
    r = np.linalg.norm(x, axis=1)
    phi, dphi = shell_distance(r[:, None], target.radii[None, :])
    attraction = phi @ target.weights
    radial = (dphi @ target.weights) / np.maximum(r, 1e-300)
    squares = np.sum(x**2, axis=1)
    repulsion = 0.0
    push = np.zeros_like(x)  # sum_n (x_m - x_n) / |x_m - x_n|
    # Each pair once: blocks (a, b) with b >= a, and (b, a) from the transpose.
    for a in range(0, m, BLOCK):
        xa, sa = x[a : a + BLOCK], squares[a : a + BLOCK]
        for b in range(a, m, BLOCK):
            xb, sb = x[b : b + BLOCK], squares[b : b + BLOCK]
            # |x_m - x_n|^2 from the Gram matrix.
            d2 = sa[:, None] + sb[None, :] - 2 * (xa @ xb.T)
            if a == b:
                np.fill_diagonal(d2, 0.0)  # exactly, not rounding noise
            d = np.sqrt(np.maximum(d2, 0.0))
            inverse = np.divide(1.0, d, out=np.zeros_like(d), where=d > 1e-12)
            push[a : a + BLOCK] += xa * inverse.sum(axis=1)[:, None] - inverse @ xb
            if a == b:
                repulsion += float(d.sum())
            else:
                repulsion += 2 * float(d.sum())
                push[b : b + BLOCK] += xb * inverse.sum(axis=0)[:, None] - inverse.T @ xa
    value = float(attraction.mean()) - repulsion / (2 * m**2)
    gradient: FloatArray = radial[:, None] * x / m - push / m**2
    return value, gradient


def energy_distance(x: FloatArray, target: ShellTarget) -> float:
    """Energy distance ``2 F - E|Y - Y'|`` between the samples and the target."""
    return 2 * energy(x, target)[0] - target_self_energy(target)


@dataclass(frozen=True)
class SampleOperator:
    """The ADC samples of a waveform, ``x = from_k @ k + from_g @ g``, with
    ``k = sample_kspace(config, g)``. Exact for the piecewise-linear waveform:
    a sample a fraction ``f`` of a raster after gradient sample ``i`` is
    ``k_i`` plus the integral of g from sample ``i``, which is
    ``scale * ((f - f^2 / 2) g_i + f^2 / 2 g_{i+1})``."""

    from_k: sp.csr_matrix
    from_g: sp.csr_matrix

    def __call__(self, config: Config, g: FloatArray) -> FloatArray:
        x: FloatArray = self.from_k @ sample_kspace(config, g) + self.from_g @ g
        return x


def sample_operator(config: Config) -> SampleOperator:
    """The ``SampleOperator`` of the config's ADC."""
    r = config.readout
    raster = config.system.grad_raster_time_s
    n = config.raster_points_per_tr
    t = config.adc_start_s + (np.arange(r.adc_num_samples) + 0.5) * r.adc_dwell_s
    u = t / raster - 0.5
    i0 = np.clip(np.floor(u).astype(np.int64), 0, n - 2)
    f = u - i0
    rows = np.arange(len(t))
    from_k = sp.csr_matrix((np.ones(len(t)), (rows, i0)), shape=(len(t), n))
    scale = kmax_per_mT_per_m_raster(config)
    values = scale * np.stack([f - f**2 / 2, f**2 / 2], axis=1).ravel()
    cols = np.stack([i0, i0 + 1], axis=1).ravel()
    from_g = sp.csr_matrix((values, (np.repeat(rows, 2), cols)), shape=(len(t), n))
    return SampleOperator(from_k, from_g)


def repulsion_target(config: Config, x: FloatArray) -> ShellTarget:
    """The config's target for a shape with samples ``x``."""
    if config.shapes.repulsion_target is RepulsionTarget.MODEL:
        return model_target(config)
    return initial_target(np.linalg.norm(x, axis=1))


def slew_energy(g: FloatArray) -> float:
    """The objective of ``optimize_shape``: sum of squared sample differences."""
    return float(np.sum(np.diff(g, axis=0) ** 2))


def _scp_step(
    config: Config,
    shape: Shape,
    g: FloatArray,
    linear_k: FloatArray,
    linear_g: FloatArray,
    trust_cells: float,
    ball: float,
) -> FloatArray | None:
    """One SCP subproblem from the waveform ``g``: minimize
    ``linear_k . k + linear_g . g' + cost`` subject to the solved
    shape's constraints, ``|k| <= ball`` in the ADC, and the trust region.
    None if the solver fails or ends ``optimal_inaccurate`` (which happens
    at large trust regions), since that waveform may miss a constraint.

    The problem is built from constants at every step. cvxpy parameters
    would let it compile once, but their parameter-to-data maps used more
    than 12 GB for one full-size shape; with the sparse k variable, building
    the problem takes a small part of the solve time.
    """
    targets = shape.targets
    adc = adc_samples(config)
    k_now = sample_kspace(config, g)
    directions = None
    if config.shapes.tip_width > 0 and relaxed(config):
        # Re-linearize the reach constraint at the current tips (always feasible).
        k_tips = target_kspace(config, g, targets)[0:-1:2]
        directions = k_tips / np.linalg.norm(k_tips, axis=1, keepdims=True)
    variable = cp.Variable(g.shape)
    k, constraints = waveform_constraints(config, targets, variable, relaxed(config), directions)
    if not relaxed(config):
        constraints.append(cp.norm(k[adc], 2, axis=1) <= ball)
    trust = cells_to_kmax(config, trust_cells)
    constraints.append(cp.norm(k[adc] - k_now[adc], 2, axis=1) <= trust)
    objective = (
        cp.sum(cp.multiply(linear_k, k))
        + cp.sum(cp.multiply(linear_g, variable))
        + gradient_cost(config, variable)
    )
    problem = cp.Problem(cp.Minimize(objective), constraints)
    try:
        problem.solve(solver=cp.CLARABEL)
    except cp.error.SolverError:
        return None
    if problem.status != cp.OPTIMAL or variable.value is None:
        return None
    return np.asarray(variable.value, dtype=np.float64)


def repel_shape(config: Config, shape: Shape) -> tuple[Shape, list[RepulsionStep]]:
    """Spread the samples of a solved shape by ``repulsion_iterations`` SCP steps.

    ``shape`` must meet the config's constraints (it is the first iterate).
    ``solve_time_s`` of the result adds the time of the steps.
    """
    s = config.shapes
    adc = adc_samples(config)
    samples = sample_operator(config)

    g = shape.g_mT_per_m
    x = samples(config, g)
    target = repulsion_target(config, x)
    f, grad = energy(x, target)
    delta_cells = s.repulsion_trust_cells
    # A full first step along the energy gradient is worth about the cost.
    lam = (
        s.repulsion_weight
        * gradient_cost_value(config, g)
        / (np.abs(grad).sum() * cells_to_kmax(config, delta_cells))
    )
    # The hairpin can pass slightly beyond k_max between its targets.
    ball = max(1.0, float(np.linalg.norm(sample_kspace(config, g)[adc], axis=1).max()))

    merit = lam * f + gradient_cost_value(config, g)
    history: list[RepulsionStep] = []
    start = time.perf_counter()
    for _ in range(s.repulsion_iterations):
        if delta_cells < MIN_TRUST_CELLS:
            break
        step_start = time.perf_counter()
        candidate = _scp_step(
            config,
            shape,
            g,
            lam * (samples.from_k.T @ grad),
            lam * (samples.from_g.T @ grad),
            delta_cells,
            ball,
        )
        solve_time = time.perf_counter() - step_start
        accepted = False
        ratio = math.nan
        trust = delta_cells
        if candidate is not None:
            x_new = samples(config, candidate)
            predicted = merit - (
                lam * (f + float(np.sum(grad * (x_new - x))))
                + gradient_cost_value(config, candidate)
            )
            f_new, grad_new = energy(x_new, target)
            merit_new = lam * f_new + gradient_cost_value(config, candidate)
            ratio = (merit - merit_new) / predicted if predicted > 0 else math.nan
            accepted = bool(predicted > 0 and ratio >= ACCEPT_RATIO)
            if accepted:
                g, x, f, grad, merit = candidate, x_new, f_new, grad_new, merit_new
        if math.isnan(ratio) or ratio < SHRINK_RATIO:
            delta_cells /= 2
        elif ratio > GROW_RATIO:
            delta_cells = min(2 * delta_cells, MAX_TRUST_CELLS)
        history.append(RepulsionStep(f, slew_energy(g), trust, accepted, ratio, solve_time))
    elapsed = time.perf_counter() - start
    repelled = dataclasses.replace(shape, g_mT_per_m=g, solve_time_s=shape.solve_time_s + elapsed)
    return repelled, history
