"""Per-TR rotations of the rosette shapes.

Every TR plays one of the shapes, rotated. The rotations of the acquired TRs
are a super-Fibonacci spiral on SO(3) (Alexa, "Super-Fibonacci Spirals: Fast,
Low-Discrepancy Sampling of SO(3)", CVPR 2022), turned by one random rotation
from the seed. Shape ``s`` of ``S`` gets the spiral points ``i`` with
``i % S == s``, so the rotations of each shape are spread over SO(3), and
together they are the full spiral. The order of the TRs is then shuffled.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from relaxed_rosettes.config import Config

# psi is the real root of psi**4 = psi + 4.
_PHI = np.sqrt(2.0)
_PSI = 1.533751168755204288118041


def super_fibonacci_quaternions(n: int) -> NDArray[np.float64]:
    """``n`` unit quaternions ``(x, y, z, w)`` spread uniformly over SO(3)."""
    s = np.arange(n) + 0.5
    r = np.sqrt(s / n)
    big_r = np.sqrt(1.0 - s / n)
    alpha = 2.0 * np.pi * s / _PHI
    beta = 2.0 * np.pi * s / _PSI
    return np.stack(
        [r * np.sin(alpha), r * np.cos(alpha), big_r * np.sin(beta), big_r * np.cos(beta)],
        axis=1,
    )


def acquired_schedule(
    config: Config, rng: np.random.Generator
) -> tuple[NDArray[np.int64], Rotation]:
    """Shape index and rotation of each acquired TR, in play order."""
    n = config.scan.num_trs
    spiral = Rotation.from_quat(super_fibonacci_quaternions(n))
    rotations = Rotation.random(random_state=rng) * spiral
    shape_index = np.arange(n) % config.shapes.num_shapes
    order = rng.permutation(n)
    return shape_index[order], rotations[order]


def dummy_schedule(
    config: Config, rng: np.random.Generator
) -> tuple[NDArray[np.int64], Rotation | None]:
    """Shape index and rotation of each dummy TR: random shapes and rotations."""
    n = config.scan.num_dummy_trs
    if n == 0:
        return np.zeros(0, dtype=np.int64), None
    shape_index = rng.integers(0, config.shapes.num_shapes, n)
    return shape_index, Rotation.random(n, random_state=rng)
