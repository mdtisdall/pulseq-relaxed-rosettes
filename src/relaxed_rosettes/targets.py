"""k-space targets and their times for each rosette shape.

A shape is one TR's gradient waveform. In each TR the trajectory starts at
k = 0 (the RF pulse is played with the gradients off), visits
``echoes_per_tr`` petal tips with a k = 0 crossing after each one (a gradient
echo), and ends at the spoil moment. The shape optimizer finds gradients that
hit these targets at these times.

k-space positions are in units of k_max. Times are from the start of the TR
block.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from relaxed_rosettes.config import Config

FloatArray = NDArray[np.float64]

NUM_DIRECTION_CANDIDATES = 10_000
NUM_RADIUS_CANDIDATES = 1_000


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


def target_positions(config: Config, rng: np.random.Generator) -> FloatArray:
    """Target k-space positions of one shape: tip, 0, tip, ..., 0, spoil."""
    s = config.shapes
    directions = blue_noise_directions(rng, s.echoes_per_tr + 1, s.max_consecutive_tip_dot)
    power = s.tip_radius_power
    u = blue_noise_1d(
        rng, s.echoes_per_tr, s.tip_radius_min ** (1 / power), s.tip_radius_max ** (1 / power)
    )
    rng.shuffle(u)
    radii = np.append(u**power, s.spoil_moment_kmax)

    k = np.zeros((2 * s.echoes_per_tr + 1, 3))
    k[0::2] = directions * radii[:, None]
    return k


def target_times(config: Config, k: FloatArray, rng: np.random.Generator) -> FloatArray:
    """Times at which the trajectory is at each target.

    Each segment between consecutive targets gets time proportional to the
    sqrt of the distance it travels from or to k = 0 (the tip radius), times a
    random jitter factor. The tips and crossings fill the ADC window: from the
    end of the RF pulse plus the gentle boundary window, to the end of the
    ADC less one more (average, jittered) segment, so the ADC also samples the start of
    the path out to the spoil moment. The spoil moment is reached at the end
    of the TR. The first segment also gets the boundary window, in which the
    gradient ramps up gently.
    """
    s = config.shapes
    raster = config.system.grad_raster_time_s
    tip_radii = np.linalg.norm(k[0:-1:2], axis=1)
    # Segment i ends at target i; it goes 0 -> tip (even i) or tip -> 0 (odd i).
    distances = np.repeat(tip_radii, 2)
    # One more segment of average sqrt distance, after the last crossing, in
    # the ADC. It is jittered too, so the last crossing moves between shapes.
    sqrt_distances = np.append(np.sqrt(distances), np.sqrt(distances).mean())
    jitter = rng.uniform(1 - s.timing_jitter, 1 + s.timing_jitter, len(sqrt_distances))
    weights = sqrt_distances * jitter

    start = config.rf_end_s + config.gradients.boundary_duration_s
    durations = weights / weights.sum() * (config.adc_end_s - start)
    t = start + np.cumsum(durations)[:-1]
    t = np.append(t, config.readout.tr_s)
    t = np.round(t / raster) * raster
    if np.any(np.diff(t) <= 0) or t[0] <= start:
        raise ValueError("target times are not strictly increasing on the gradient raster")
    return t


def make_shape_targets(config: Config, rng: np.random.Generator) -> ShapeTargets:
    """Positions and times of the targets of one shape."""
    k = target_positions(config, rng)
    return ShapeTargets(k=k, t_s=target_times(config, k, rng))


def make_targets(config: Config) -> list[ShapeTargets]:
    """Targets of every shape.

    Each shape has its own random stream, derived from the config seed and
    the shape index, so shape ``i`` does not depend on ``num_shapes``.
    """
    streams = np.random.SeedSequence(config.seed).spawn(config.shapes.num_shapes)
    return [make_shape_targets(config, np.random.default_rng(stream)) for stream in streams]


def save_targets(path: Path, config: Config, targets: list[ShapeTargets]) -> None:
    """Write targets and the config that made them to an ``.npz`` file.

    Arrays: ``k`` (shapes, targets, 3) in units of k_max, ``t_s`` (shapes,
    targets), ``k_max_per_m``, ``config_toml``, and ``config_digest``.
    """
    np.savez(
        path,
        k=np.stack([t.k for t in targets]),
        t_s=np.stack([t.t_s for t in targets]),
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
        targets = [ShapeTargets(k=k, t_s=t) for k, t in zip(data["k"], data["t_s"], strict=True)]
    return config, targets
