"""Subspace basis for the signal over time since excitation (τ).

Within one readout, the signal of a voxel decays as ``exp(-R2* τ)``. Over a
range of R2*, these curves are close to a low-dimensional subspace: the
leading left singular vectors of a matrix of sampled curves. A
reconstruction that resolves τ represents each voxel by its coefficients in
that subspace, ``x(r, τ) = sum_l c_l(r) phi_l(τ)``.

The basis is of decay only. Off-resonance stays in the field map and the
time segmentation of the reconstruction (``relaxed_rosettes.recon.b0``).

Each curve is normalized before the SVD, so that slowly and quickly decaying
curves count equally. The rank is fixed, or the smallest whose worst
relative projection error over the curves is at most a tolerance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class TauBasis:
    """An orthonormal basis of decay curves on the τ of the ADC samples."""

    tau_s: FloatArray
    """(samples,): time since excitation of each ADC sample."""
    functions: FloatArray
    """(rank, samples): orthonormal rows, ``phi_l(τ)``."""
    singular_values: FloatArray
    """(min(curves, samples),): all singular values of the normalized curves."""
    r2star_per_s: FloatArray
    """(curves,): R2* of the curves the basis was built from."""
    max_relative_error: float
    """Worst relative projection error of those curves onto the basis."""

    @property
    def rank(self) -> int:
        return int(self.functions.shape[0])

    def project(self, signal: NDArray[Any]) -> NDArray[Any]:
        """(..., rank) coefficients of (..., samples) signals."""
        coefficients: NDArray[Any] = signal @ self.functions.T
        return coefficients

    def expand(self, coefficients: NDArray[Any]) -> NDArray[Any]:
        """(..., samples) signals of (..., rank) coefficients."""
        signal: NDArray[Any] = coefficients @ self.functions
        return signal


def decay_basis(
    tau_s: FloatArray,
    r2star_min_per_s: float,
    r2star_max_per_s: float,
    *,
    curves: int = 256,
    rank: int = 0,
    tolerance: float = 1e-3,
) -> TauBasis:
    """Basis of ``exp(-R2* τ)`` for R2* evenly spaced in ``[r2star_min, r2star_max]``.

    ``rank = 0`` takes the smallest rank whose worst relative projection
    error over the curves is at most ``tolerance``.
    """
    tau = np.asarray(tau_s, dtype=np.float64)
    if tau.ndim != 1 or len(tau) < 2:
        raise ValueError("tau_s must have at least two samples")
    if not 0 <= r2star_min_per_s < r2star_max_per_s:
        raise ValueError("need 0 <= r2star_min_per_s < r2star_max_per_s")
    if curves < 2:
        raise ValueError("curves must be >= 2")
    if rank < 0 or rank > min(curves, len(tau)):
        raise ValueError(f"rank must be in [0, {min(curves, len(tau))}]")
    if rank == 0 and not 0 < tolerance < 1:
        raise ValueError("tolerance must be in (0, 1)")

    r2star = np.linspace(r2star_min_per_s, r2star_max_per_s, curves)
    matrix = np.exp(-np.outer(tau, r2star))  # (samples, curves)
    matrix /= np.linalg.norm(matrix, axis=0, keepdims=True)
    u, singular, _ = np.linalg.svd(matrix, full_matrices=False)
    # Each curve has unit norm, so its relative error at rank L is the norm of
    # its part outside the first L singular vectors.
    coefficients = u.T @ matrix  # (min, curves)
    tail = np.sqrt(np.maximum(np.cumsum((coefficients**2)[::-1], axis=0)[::-1], 0.0))
    errors = np.append(tail.max(axis=1), 0.0)  # errors[L]: worst error at rank L
    if rank == 0:
        rank = int(np.argmax(errors <= tolerance))
        rank = max(rank, 1)
    return TauBasis(
        tau_s=tau,
        functions=np.ascontiguousarray(u[:, :rank].T),
        singular_values=singular,
        r2star_per_s=r2star,
        max_relative_error=float(errors[rank]),
    )
