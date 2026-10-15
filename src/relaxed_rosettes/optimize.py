"""Gradient waveform optimization for the rosette shapes.

Each shape is one TR's gradient waveform, ``g`` with shape ``(N, 3)`` in mT/m
on the gradient raster. It is played as a pypulseq arbitrary gradient: sample
``i`` at time ``(i + 0.5) * raster`` from the start of the TR block, linearly
interpolated, with ``first = last = 0`` at times 0 and ``N * raster``.

The waveform minimizes slew energy subject to:

- g = 0 from the start of the block to the end of the RF pulse (the
  excitation is played with the gradients off), and at the end of the TR;
- |g| <= max gradient, and |slew| <= max slew, tightened to the boundary
  slew in the first and last ``boundary_duration_s`` of the gradient;
- k(t) = target at each target time, with k = 0 at the RF center.

All constraints are linear or second-order cone constraints, and the objective
is convex quadratic, so each shape is a convex problem with a unique optimum.
It is solved with cvxpy and Clarabel.
"""

from __future__ import annotations

import dataclasses
import math
import os
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cvxpy as cp
import numpy as np
from numpy.typing import NDArray

from relaxed_rosettes.config import GAMMA_HZ_PER_T, Config, TipDensity
from relaxed_rosettes.density import (
    TipSurvival,
    estimate_speed,
    radial_grid,
    survival_from_profile,
    target_profile,
)
from relaxed_rosettes.targets import (
    PROVISIONAL_PETAL_TIME_FACTOR,
    ShapeTargets,
    fitted_num_tips,
    make_shape_targets,
    make_targets,
    petal_times_s,
    petal_window_s,
    shape_streams,
    stack_targets,
    target_positions,
    target_times,
    unstack_targets,
)

FloatArray = NDArray[np.float64]

LIMIT_MARGIN = 1e-4
"""The optimizer uses the gradient and slew limits reduced by this fraction,
so that solver tolerance never puts a waveform over a limit."""

TIMING_CALIBRATION_STEPS = 6
"""Bisection steps for the shortest feasible petal time factor of a shape."""

MAX_TIP_RETRIES = 3
"""Tips a shape may drop, one at a time, when its fitted targets are infeasible."""


class ShapeOptimizationError(RuntimeError):
    """The solver did not find an optimal waveform for a shape."""


@dataclass(frozen=True)
class Shape:
    """One optimized gradient waveform and its targets."""

    g_mT_per_m: FloatArray
    targets: ShapeTargets
    status: str
    solve_time_s: float


@dataclass(frozen=True)
class ShapeMetrics:
    """How well a waveform meets the constraints, in physical units."""

    max_grad_mT_per_m: float
    max_slew_T_per_m_per_s: float
    max_boundary_slew_T_per_m_per_s: float
    max_target_error_kmax: float
    max_grad_in_rf_window_mT_per_m: float
    end_grad_mT_per_m: float


def num_zero_samples(config: Config) -> int:
    """Samples held at 0 from the start of the block, so g = 0 until RF end.

    The interpolated waveform leaves 0 half a raster before the first nonzero
    sample, so sample ``i`` may be nonzero only if ``(i - 0.5) * raster`` is
    at or after the end of the RF pulse.
    """
    raster = config.system.grad_raster_time_s
    return math.ceil(config.rf_end_s / raster + 0.5 - 1e-9)


def k_weights(num_samples: int, edge: int) -> FloatArray:
    """Weights ``w`` such that ``raster * (w @ g)`` = integral of g up to an edge.

    ``edge`` is a raster edge index (time ``edge * raster``, 1 <= edge <= N).
    The integral is exact for the piecewise-linear waveform that pypulseq
    plays, with ``first = last = 0``.
    """
    if not 1 <= edge <= num_samples:
        raise ValueError(f"edge {edge} is outside 1..{num_samples}")
    w = np.zeros(num_samples)
    w[0] += 0.25  # from first = 0 at t = 0 to sample 0 at t = raster / 2
    w[: edge - 1] += 0.5  # full intervals between consecutive samples
    w[1:edge] += 0.5
    if edge < num_samples:
        # Half interval from sample edge - 1 to the midpoint with sample edge.
        w[edge - 1] += 3 / 8
        w[edge] += 1 / 8
    else:
        w[edge - 1] += 0.25  # from the last sample to last = 0
    return w


def kmax_per_mT_per_m_raster(config: Config) -> float:
    """k (in units of k_max) per (1 mT/m for one raster time)."""
    return GAMMA_HZ_PER_T * 1e-3 * config.system.grad_raster_time_s / config.k_max_per_m


def target_edges(config: Config, targets: ShapeTargets) -> NDArray[np.int64]:
    """Raster edge index of each target time."""
    edges = np.round(targets.t_s / config.system.grad_raster_time_s).astype(np.int64)
    if edges.min() < 1 or edges.max() > config.raster_points_per_tr:
        raise ValueError("a target time is outside the TR")
    return edges


def slew_limits(config: Config) -> FloatArray:
    """Slew limit (mT/m per raster) of each difference ``g[i + 1] - g[i]``.

    Difference ``i`` is at time ``(i + 1) * raster``. It gets the boundary
    limit within ``boundary_duration_s`` after the RF end or before the TR end.
    """
    raster = config.system.grad_raster_time_s
    g = config.gradients
    t = (np.arange(config.raster_points_per_tr - 1) + 1) * raster
    boundary = (t <= config.rf_end_s + g.boundary_duration_s + 1e-12) | (
        t >= config.readout.tr_s - g.boundary_duration_s - 1e-12
    )
    limits = np.where(boundary, g.boundary_slew_T_per_m_per_s, g.max_slew_T_per_m_per_s)
    # T/m/s -> mT/m per raster
    result: FloatArray = limits * 1e3 * raster
    return result


def optimize_shape(config: Config, targets: ShapeTargets) -> Shape:
    """Minimum slew-energy waveform that hits the targets of one shape."""
    n = config.raster_points_per_tr
    n_zero = num_zero_samples(config)
    scale = kmax_per_mT_per_m_raster(config)
    weights = np.stack([k_weights(n, e) for e in target_edges(config, targets)])

    g = cp.Variable((n, 3))
    slew = g[1:] - g[:-1]
    constraints = [
        g[:n_zero] == 0,
        g[n - 1] == 0,
        cp.norm(g, 2, axis=1) <= config.gradients.max_grad_mT_per_m * (1 - LIMIT_MARGIN),
        cp.norm(slew, 2, axis=1) <= slew_limits(config) * (1 - LIMIT_MARGIN),
        scale * (weights @ g) == targets.k,
    ]
    problem = cp.Problem(cp.Minimize(cp.sum_squares(slew)), constraints)
    start = time.perf_counter()
    try:
        problem.solve(solver=cp.CLARABEL)
    except cp.error.SolverError as e:
        # Clarabel can fail outright on problems at the edge of feasibility.
        raise ShapeOptimizationError(f"solver failed: {e}") from None
    elapsed = time.perf_counter() - start
    if problem.status != cp.OPTIMAL or g.value is None:
        raise ShapeOptimizationError(f"solver status: {problem.status}")
    return Shape(
        g_mT_per_m=np.asarray(g.value, dtype=np.float64),
        targets=targets,
        status=str(problem.status),
        solve_time_s=elapsed,
    )


def shape_metrics(config: Config, shape: Shape) -> ShapeMetrics:
    """Measure a waveform against the constraints."""
    g = shape.g_mT_per_m
    raster = config.system.grad_raster_time_s
    # Include the half-raster steps from first = 0 and to last = 0.
    padded = np.vstack([np.zeros((1, 3)), g, np.zeros((1, 3))])
    steps = np.diff(padded, axis=0)
    durations = np.full(len(steps), raster)
    durations[[0, -1]] = raster / 2
    slew = np.linalg.norm(steps, axis=1) / durations * 1e-3  # T/m/s
    t = np.arange(len(steps)) * raster  # time of each step: 0, r, 2r, ..., N r
    t[-1] = config.readout.tr_s
    gc = config.gradients
    boundary = (t <= config.rf_end_s + gc.boundary_duration_s + 1e-12) | (
        t >= config.readout.tr_s - gc.boundary_duration_s - 1e-12
    )

    scale = kmax_per_mT_per_m_raster(config)
    edges = target_edges(config, shape.targets)
    k = np.stack([scale * (k_weights(len(g), e) @ g) for e in edges])
    return ShapeMetrics(
        max_grad_mT_per_m=float(np.linalg.norm(g, axis=1).max()),
        max_slew_T_per_m_per_s=float(slew.max()),
        max_boundary_slew_T_per_m_per_s=float(slew[boundary].max()),
        max_target_error_kmax=float(np.abs(k - shape.targets.k).max()),
        max_grad_in_rf_window_mT_per_m=float(np.abs(g[: num_zero_samples(config)]).max()),
        end_grad_mT_per_m=float(np.abs(g[-1]).max()),
    )


@dataclass(frozen=True)
class _Work:
    """One shape to optimize, and how to make it again with fewer tips."""

    config: Config
    targets: ShapeTargets
    index: int
    survival: TipSurvival | None = None
    petal_time_factor: float | None = None
    """The factor that fitted the tips; None when the tip count is fixed."""


def _optimize_one(work: _Work) -> Shape:
    """Optimize one shape. Fitted targets (``echoes_per_tr`` = 0) that are
    infeasible are made again with one tip fewer, up to ``MAX_TIP_RETRIES``
    times: the calibrated petal time factor holds for most shapes, not all.
    """
    targets = work.targets
    retries = MAX_TIP_RETRIES if work.petal_time_factor is not None else 0
    stream = shape_streams(work.config)[work.index] if retries else None
    for attempt in range(retries + 1):
        try:
            return optimize_shape(work.config, targets)
        except ShapeOptimizationError as e:
            if attempt == retries or targets.num_echoes == 1:
                raise ShapeOptimizationError(f"shape {work.index}: {e}") from None
        assert stream is not None and work.petal_time_factor is not None
        targets = make_shape_targets(
            work.config,
            stream,
            work.survival,
            work.petal_time_factor,
            num_tips=targets.num_echoes - 1,
        )
    raise AssertionError("unreachable")


def optimize_shapes(config: Config, jobs: int | None = None) -> list[Shape]:
    """Targets and optimized waveforms of every shape, solved in parallel.

    ``jobs`` is the number of worker processes (default: the CPU count, at
    most the number of shapes); 1 solves in this process. A derived tip
    density is calibrated first, then (with ``echoes_per_tr`` = 0) the petal
    time factor.
    """
    survival = None
    if config.shapes.tip_density is not TipDensity.POWER:
        survival = calibrated_survival(config, jobs)
    factor = None
    if config.shapes.echoes_per_tr == 0:
        factor = calibrated_petal_time_factor(config, survival, jobs)
    targets = make_targets(config, survival, factor or PROVISIONAL_PETAL_TIME_FACTOR)
    return _optimize_targets(config, targets, jobs, survival, factor)


def _optimize_targets(
    config: Config,
    targets: list[ShapeTargets],
    jobs: int | None,
    survival: TipSurvival | None = None,
    petal_time_factor: float | None = None,
) -> list[Shape]:
    work = [_Work(config, t, i, survival, petal_time_factor) for i, t in enumerate(targets)]
    return _parallel_map(_optimize_one, work, jobs)


def _parallel_map[T, R](fn: Callable[[T], R], items: list[T], jobs: int | None) -> list[R]:
    jobs = min(jobs or os.cpu_count() or 1, len(items))
    if jobs == 1:
        return [fn(item) for item in items]
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        return list(pool.map(fn, items))


def shortest_petal_time_factor(
    config: Config, stream: np.random.SeedSequence, survival: TipSurvival | None = None
) -> float:
    """Smallest factor on ``petal_times_s`` at which one shape is feasible.

    The shape gets the tips that fit at ``segment_time_margin`` times
    ``PROVISIONAL_PETAL_TIME_FACTOR``, packed from the start of the petal
    window at the factor, with the rest of the window going to the path out
    to the spoil moment. The factor is found by bisection between 1 (the
    straight-path bound, never beaten) and the factor that fills the window,
    to ``TIMING_CALIBRATION_STEPS`` halvings.
    """
    s = config.shapes
    start, end = petal_window_s(config)
    n = fitted_num_tips(
        config, stream, s.segment_time_margin * PROVISIONAL_PETAL_TIME_FACTOR, survival
    )
    while True:
        k = target_positions(config, np.random.default_rng(stream), n, survival)
        high = (end - start) / petal_times_s(config, np.linalg.norm(k[0:-1:2], axis=1)).sum()
        if _feasible(config, k, high):
            break
        if n == 1:
            raise ShapeOptimizationError("one petal is infeasible even when it fills the window")
        n -= 1
    low = 1.0
    for _ in range(TIMING_CALIBRATION_STEPS):
        mid = (low + high) / 2
        if _feasible(config, k, mid):
            high = mid
        else:
            low = mid
    return high


def _feasible(config: Config, k: FloatArray, factor: float) -> bool:
    try:
        optimize_shape(config, ShapeTargets(k=k, t_s=target_times(config, k, factor)))
    except (ShapeOptimizationError, ValueError):
        return False
    return True


def _shortest_factor_one(args: tuple[Config, np.random.SeedSequence, TipSurvival | None]) -> float:
    return shortest_petal_time_factor(*args)


def calibrated_petal_time_factor(
    config: Config, survival: TipSurvival | None = None, jobs: int | None = None
) -> float:
    """The petal time factor: the largest ``shortest_petal_time_factor`` of the
    first ``timing_calibration_shapes`` shapes.

    The straight out-and-back petal of ``petal_times_s`` is a lower bound;
    the optimizer needs a little more, because the path turns at k = 0
    between petals (about 1.1 at the default limits, the same within ±2.5%
    for every shape). With the largest of the calibration shapes, nearly all
    shapes are feasible at ``segment_time_margin`` = 1.
    """
    streams = shape_streams(config)[: config.shapes.timing_calibration_shapes]
    factors = _parallel_map(_shortest_factor_one, [(config, st, survival) for st in streams], jobs)
    return max(factors)


def calibrated_survival(config: Config, jobs: int | None = None) -> TipSurvival:
    """Tip survival of a derived tip density, with the trajectory speed calibrated.

    Starts from a constant speed; then, ``calibration_iterations`` times,
    optimizes the first ``calibration_shapes`` shapes, estimates the speed at
    each radius from their samples, and recomputes the tips
    (``relaxed_rosettes.density``).
    """
    from relaxed_rosettes.trajectory import kspace_at_times

    s = config.shapes
    grid = radial_grid()
    profile = target_profile(config, grid)
    speed = np.ones_like(grid)
    survival = survival_from_profile(grid, profile, speed, s.tip_radius_min, s.tip_radius_max)
    calibration = dataclasses.replace(
        config, shapes=dataclasses.replace(s, num_shapes=s.calibration_shapes)
    )
    r = config.readout
    adc_times = np.asarray(
        config.adc_start_s + (np.arange(r.adc_num_samples) + 0.5) * r.adc_dwell_s,
        dtype=np.float64,
    )
    # The speed estimate needs only roughly right timing, so the petal time
    # factor is not calibrated yet.
    factor = PROVISIONAL_PETAL_TIME_FACTOR if s.echoes_per_tr == 0 else None
    for _ in range(s.calibration_iterations):
        targets = make_targets(calibration, survival, PROVISIONAL_PETAL_TIME_FACTOR)
        shapes = _optimize_targets(calibration, targets, jobs, survival, factor)
        samples = [
            np.linalg.norm(
                kspace_at_times(sh.g_mT_per_m, config.system.grad_raster_time_s, adc_times),
                axis=1,
            )
            / config.k_max_per_m
            for sh in shapes
        ]
        speed = estimate_speed(grid, samples)
        survival = survival_from_profile(grid, profile, speed, s.tip_radius_min, s.tip_radius_max)
    return survival


def save_shapes(path: Path, config: Config, shapes: list[Shape]) -> None:
    """Write shapes and the config that made them to an ``.npz`` file.

    Arrays: ``g_mT_per_m`` (shapes, raster points, 3), the targets ``k``
    (shapes, targets, 3; units of k_max) and ``t_s`` (shapes, targets), both
    NaN-padded (``relaxed_rosettes.targets.stack_targets``), ``status`` and
    ``solve_time_s`` (shapes), ``config_toml``, and ``config_digest``.
    """
    k, t_s = stack_targets([s.targets for s in shapes])
    np.savez(
        path,
        g_mT_per_m=np.stack([s.g_mT_per_m for s in shapes]),
        k=k,
        t_s=t_s,
        status=np.array([s.status for s in shapes]),
        solve_time_s=np.array([s.solve_time_s for s in shapes]),
        config_toml=config.to_toml(),
        config_digest=config.digest(),
    )


def load_shapes(path: Path) -> tuple[Config, list[Shape]]:
    """Read a file written by ``save_shapes``."""
    with np.load(path) as data:
        config = Config.from_toml(str(data["config_toml"]))
        if str(data["config_digest"]) != config.digest():
            raise ValueError(f"{path}: config digest does not match its config")
        shapes = [
            Shape(
                g_mT_per_m=g,
                targets=targets,
                status=str(status),
                solve_time_s=float(solve_time),
            )
            for g, targets, status, solve_time in zip(
                data["g_mT_per_m"],
                unstack_targets(data["k"], data["t_s"]),
                data["status"],
                data["solve_time_s"],
                strict=True,
            )
        ]
    return config, shapes
