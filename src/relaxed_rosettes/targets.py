"""k-space targets and their times for each rosette shape.

A shape is one TR's gradient waveform. In each TR the trajectory starts at
k = 0 (the RF pulse is played with the gradients off), visits a number of
petal tips with a k = 0 crossing after each one (a gradient echo), and ends at
the spoil moment. The shape optimizer finds gradients that hit these targets
at these times.

Each petal (k = 0, tip, k = 0) gets time in proportion to ``petal_times_s``,
the shortest time of a straight out-and-back path to its tip. The shape
optimizer needs somewhat more than that (the path must also turn at k = 0
between petals); the factor is calibrated by
``relaxed_rosettes.optimize.calibrated_petal_time_factor``.

k-space positions are in units of k_max. Times are from the start of the TR
block.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from relaxed_rosettes.config import GAMMA_HZ_PER_T, Config, TipDensity
from relaxed_rosettes.density import TipSurvival

FloatArray = NDArray[np.float64]

NUM_DIRECTION_CANDIDATES = 10_000
NUM_RADIUS_CANDIDATES = 1_000

PROVISIONAL_PETAL_TIME_FACTOR = 1.1
"""Petal time factor used before it is calibrated (while the tip density is
calibrated). About what the default config calibrates to."""


@dataclass(frozen=True)
class ShapeTargets:
    """Targets of one shape: tip, 0, tip, 0, ..., tip, 0, spoil.

    ``k`` has shape ``(2 * echoes + 1, 3)`` in units of k_max, and ``t_s`` has
    shape ``(2 * echoes + 1,)``: the time at which the trajectory is at each
    target, on the gradient raster, from the start of the TR block.
    """

    k: FloatArray
    t_s: FloatArray

    @property
    def num_echoes(self) -> int:
        return (len(self.t_s) - 1) // 2

    @property
    def tips(self) -> FloatArray:
        return self.k[0:-1:2]

    @property
    def tip_times_s(self) -> FloatArray:
        return self.t_s[0:-1:2]

    @property
    def crossing_times_s(self) -> FloatArray:
        return self.t_s[1::2]

    @property
    def spoil(self) -> FloatArray:
        return self.k[-1, :]


def blue_noise_directions(
    rng: np.random.Generator,
    n: int,
    max_consecutive_dot: float,
    num_candidates: int = NUM_DIRECTION_CANDIDATES,
) -> FloatArray:
    """``n`` unit vectors, each well separated from all the earlier ones.

    The first is uniformly random. Each later one is, of ``num_candidates``
    uniformly random unit vectors whose dot product with the previous vector
    is below ``max_consecutive_dot``, the one farthest from all earlier vectors
    (greedy farthest-point sampling).
    """
    points = np.empty((n, 3))
    for i in range(n):
        candidates = rng.standard_normal((num_candidates, 3))
        candidates /= np.linalg.norm(candidates, axis=1, keepdims=True)
        if i == 0:
            points[0] = candidates[0]
            continue
        candidates = candidates[candidates @ points[i - 1] < max_consecutive_dot]
        if len(candidates) == 0:
            raise ValueError(f"no candidate has dot < {max_consecutive_dot} with the previous")
        distances = np.linalg.norm(candidates[:, None, :] - points[None, :i, :], axis=2)
        points[i] = candidates[np.argmax(distances.min(axis=1))]
    return points


def blue_noise_1d(
    rng: np.random.Generator,
    n: int,
    low: float,
    high: float,
    num_candidates: int = NUM_RADIUS_CANDIDATES,
) -> FloatArray:
    """``n`` values in ``[low, high)``, each well separated from the earlier ones.

    Greedy farthest-point sampling from ``num_candidates`` uniform candidates
    for each value; the first value is uniformly random.
    """
    values = np.empty(n)
    for i in range(n):
        candidates = rng.uniform(low, high, num_candidates)
        if i == 0:
            values[0] = candidates[0]
            continue
        distances = np.abs(candidates[:, None] - values[None, :i])
        values[i] = candidates[np.argmax(distances.min(axis=1))]
    return values


def tip_radii(
    config: Config, quantiles: FloatArray, survival: TipSurvival | None = None
) -> FloatArray:
    """Tip radii (units of k_max) for quantiles in [0, 1) of the radius distribution.

    The ``power`` model maps quantile ``q`` to ``u ** power`` with ``u``
    uniform between ``tip_radius_min ** (1 / power)`` and
    ``tip_radius_max ** (1 / power)``. The derived models need the tip
    ``survival`` (``relaxed_rosettes.density``).
    """
    s = config.shapes
    if s.tip_density is TipDensity.POWER:
        p = s.tip_radius_power
        u_min, u_max = s.tip_radius_min ** (1 / p), s.tip_radius_max ** (1 / p)
        radii: FloatArray = (u_min + quantiles * (u_max - u_min)) ** p
        return radii
    if survival is None:
        raise ValueError(f"tip density {s.tip_density} needs a tip survival function")
    return survival.radii(quantiles)


def target_positions(
    config: Config,
    rng: np.random.Generator,
    num_tips: int,
    survival: TipSurvival | None = None,
) -> FloatArray:
    """Target k-space positions of one shape: tip, 0, tip, ..., 0, spoil.

    The tip radii are at quantiles of the radius distribution that are blue
    noise in [0, 1), shuffled.
    """
    s = config.shapes
    directions = blue_noise_directions(rng, num_tips + 1, s.max_consecutive_tip_dot)
    quantiles = blue_noise_1d(rng, num_tips, 0.0, 1.0)
    rng.shuffle(quantiles)
    radii = np.append(tip_radii(config, quantiles, survival), s.spoil_moment_kmax)

    k = np.zeros((2 * num_tips + 1, 3))
    k[0::2] = directions * radii[:, None]
    return k


def petal_times_s(config: Config, radii: FloatArray) -> FloatArray:
    """Shortest time of a straight path from k = 0 out to each radius and back.

    The path may pass k = 0 at any speed up to the gradient limit and must
    stop at the tip, with acceleration at most the slew limit: bang-bang in
    one dimension. Below the radius at which it reaches full speed, the time
    is proportional to sqrt(radius); above it, the time grows linearly. Any
    trajectory through k = 0 and the tip takes at least this long, since its
    component along the tip direction obeys the same limits.

    Earlier designs gave each petal time proportional to sqrt(radius). That
    is relatively close: at the default limits (28 mT/m, 100 T/m/s, 1 mm)
    full speed is reached at 0.33 k_max, and from there to k_max the two
    differ by about 15% in relative time. Calibrated against the shape
    optimizer, both predict the shortest feasible timing of a shape to about
    ±2.5%. This model also holds when the limits or the resolution change.
    """
    g = config.gradients
    speed = GAMMA_HZ_PER_T * g.max_grad_mT_per_m * 1e-3 / config.k_max_per_m  # k_max/s
    accel = GAMMA_HZ_PER_T * g.max_slew_T_per_m_per_s / config.k_max_per_m  # k_max/s^2
    full_speed_radius = speed**2 / (2 * accel)
    r = np.asarray(radii, dtype=np.float64)
    times: FloatArray = np.where(
        r >= full_speed_radius,
        2 * ((r - full_speed_radius) / speed + speed / accel),
        2 * np.sqrt(2 * np.maximum(r, 0.0) / accel),
    )
    return times


def petal_window_s(config: Config) -> tuple[float, float]:
    """Start and end of the time that the petals fill.

    From the end of the RF pulse plus the gentle boundary window to one
    raster time before the end of the ADC, so that the last crossing is
    sampled. The path then goes out to the spoil moment, reached at the end of
    the TR.
    """
    start = config.rf_end_s + config.gradients.boundary_duration_s
    return start, config.adc_end_s - config.system.grad_raster_time_s


def target_times(config: Config, k: FloatArray, factor: float | None = None) -> FloatArray:
    """Times at which the trajectory is at each target.

    Petal ``i`` (k = 0, tip, k = 0) takes ``factor * petal_times_s``, half
    out and half back, starting at the start of the petal window. With
    ``factor`` None, the factor is the one that makes the petals fill the
    window. The spoil moment is reached at the end of the TR. The first
    segment also gets the boundary window, in which the gradient ramps up
    gently.
    """
    raster = config.system.grad_raster_time_s
    start, end = petal_window_s(config)
    halves = np.repeat(petal_times_s(config, np.linalg.norm(k[0:-1:2], axis=1)) / 2, 2)
    if factor is None:
        factor = (end - start) / halves.sum()
    t = start + factor * np.cumsum(halves)
    if t[-1] > end + raster / 2:
        raise ValueError("the petals do not fit in the petal window")
    t = np.append(t, config.readout.tr_s)
    t = np.round(t / raster) * raster
    if np.any(np.diff(t) <= 0) or t[0] <= start:
        raise ValueError("target times are not strictly increasing on the gradient raster")
    return t


def fitted_num_tips(
    config: Config,
    stream: np.random.SeedSequence,
    factor: float,
    survival: TipSurvival | None = None,
) -> int:
    """The most tips of a shape whose petals, at ``factor``, fit in the window.

    The tips for ``n`` are ``target_positions`` from the shape's stream, so
    the result is reproducible.
    """
    start, end = petal_window_s(config)

    def fits(n: int) -> bool:
        k = target_positions(config, np.random.default_rng(stream), n, survival)
        total = factor * petal_times_s(config, np.linalg.norm(k[0:-1:2], axis=1)).sum()
        return bool(total <= end - start)

    if not fits(1):
        raise ValueError("not even one petal fits in the petal window")
    # Start from the count that the mean petal time of a trial set predicts.
    trial = target_positions(config, np.random.default_rng(stream), 16, survival)
    mean_time = factor * petal_times_s(config, np.linalg.norm(trial[0:-1:2], axis=1)).mean()
    n = max(1, int((end - start) / mean_time))
    while n > 1 and not fits(n):
        n -= 1
    while fits(n + 1):
        n += 1
    return n


def shape_streams(config: Config) -> list[np.random.SeedSequence]:
    """The random stream of each shape, from the config seed and the shape index."""
    return np.random.SeedSequence(config.seed).spawn(config.shapes.num_shapes)


def make_shape_targets(
    config: Config,
    stream: np.random.SeedSequence,
    survival: TipSurvival | None = None,
    petal_time_factor: float = PROVISIONAL_PETAL_TIME_FACTOR,
    num_tips: int | None = None,
) -> ShapeTargets:
    """Positions and times of the targets of one shape.

    The number of tips is ``num_tips`` if given, else ``echoes_per_tr`` if
    positive, else as many as fit at ``segment_time_margin`` times
    ``petal_time_factor`` (``fitted_num_tips``). The petals then fill the
    window, so each gets at least that factor times its shortest time.
    """
    s = config.shapes
    if num_tips is None:
        factor = s.segment_time_margin * petal_time_factor
        num_tips = s.echoes_per_tr or fitted_num_tips(config, stream, factor, survival)
    k = target_positions(config, np.random.default_rng(stream), num_tips, survival)
    return ShapeTargets(k=k, t_s=target_times(config, k))


def make_targets(
    config: Config,
    survival: TipSurvival | None = None,
    petal_time_factor: float | None = None,
) -> list[ShapeTargets]:
    """Targets of every shape.

    Each shape has its own random stream, derived from the config seed and
    the shape index, so shape ``i`` does not depend on ``num_shapes``. A
    derived tip density (``exponent``, ``filter``) without a given
    ``survival`` is calibrated first, which optimizes ``calibration_shapes``
    shapes ``calibration_iterations`` times. With ``echoes_per_tr`` = 0 and
    no ``petal_time_factor``, the factor is calibrated, which optimizes
    ``timing_calibration_shapes`` shapes several times each.
    """
    s = config.shapes
    if s.tip_density is not TipDensity.POWER and survival is None:
        from relaxed_rosettes.optimize import calibrated_survival

        survival = calibrated_survival(config)
    if petal_time_factor is None:
        petal_time_factor = PROVISIONAL_PETAL_TIME_FACTOR
        if s.echoes_per_tr == 0:
            from relaxed_rosettes.optimize import calibrated_petal_time_factor

            petal_time_factor = calibrated_petal_time_factor(config, survival)
    return [
        make_shape_targets(config, stream, survival, petal_time_factor)
        for stream in shape_streams(config)
    ]


def stack_targets(targets: list[ShapeTargets]) -> tuple[FloatArray, FloatArray]:
    """``k`` (shapes, targets, 3) and ``t_s`` (shapes, targets) of all shapes.

    Shapes with fewer targets than the most are padded with NaN at the end.
    """
    size = max(len(t.t_s) for t in targets)
    k = np.full((len(targets), size, 3), np.nan)
    t_s = np.full((len(targets), size), np.nan)
    for i, t in enumerate(targets):
        k[i, : len(t.t_s)] = t.k
        t_s[i, : len(t.t_s)] = t.t_s
    return k, t_s


def unstack_targets(k: FloatArray, t_s: FloatArray) -> list[ShapeTargets]:
    """The targets of each shape, from ``stack_targets`` arrays."""
    valid = ~np.isnan(t_s)
    return [ShapeTargets(k=ki[v], t_s=ti[v]) for ki, ti, v in zip(k, t_s, valid, strict=True)]


def save_targets(path: Path, config: Config, targets: list[ShapeTargets]) -> None:
    """Write targets and the config that made them to an ``.npz`` file.

    Arrays: ``k`` (shapes, targets, 3) in units of k_max, ``t_s`` (shapes,
    targets), both NaN-padded (``stack_targets``), ``k_max_per_m``,
    ``config_toml``, and ``config_digest``.
    """
    k, t_s = stack_targets(targets)
    np.savez(
        path,
        k=k,
        t_s=t_s,
        k_max_per_m=config.k_max_per_m,
        config_toml=config.to_toml(),
        config_digest=config.digest(),
    )


def load_targets(path: Path) -> tuple[Config, list[ShapeTargets]]:
    """Read a file written by ``save_targets``."""
    with np.load(path) as data:
        config = Config.from_toml(str(data["config_toml"]))
        if str(data["config_digest"]) != config.digest():
            raise ValueError(f"{path}: config digest does not match its config")
        targets = unstack_targets(data["k"], data["t_s"])
    return config, targets
