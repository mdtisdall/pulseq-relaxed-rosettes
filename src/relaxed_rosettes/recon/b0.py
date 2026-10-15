"""Off-resonance: the self-calibrated field map and the time segmentation.

The signal model with off-resonance ``f(r)`` (Hz) is

    y(k, tau) = sum_r image(r) exp(2πi (k·r + f(r) tau)),

with ``tau`` the time since excitation (k = 0 at the RF center) and the same
sign as the encoding phase. ``f`` absorbs B0 inhomogeneity and, to first
order, anything else that adds phase linearly in time.

The field map comes from coil-combined images of consecutive windows of
``tau``, each at the mean ``tau`` of its samples: a weighted least-squares
fit of the phase steps between consecutive windows, smoothed in space
(which weights each voxel by its magnitude). It needs no phase unwrapping
while each step is below half a cycle.

The time segmentation approximates ``exp(2πi f tau)`` by
``sum_l b_l(tau) exp(2πi f tau_l)`` at segment times ``tau_l``, with
coefficients ``b_l(tau)`` that minimize the error over the histogram of the
field map (least squares; Sutton et al., IEEE TMI 22:178, 2003), so the
forward model costs one NUFFT per segment.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from relaxed_rosettes.recon.coilmaps import smooth

ComplexArray = NDArray[np.complex64]
FloatArray = NDArray[np.float64]


def field_map_hz(
    combined: list[ComplexArray], tau_s: FloatArray, smoothing_voxels: int
) -> NDArray[np.float32]:
    """(n, n, n) frequency from coil-combined images of windows at times ``tau_s`` (increasing).

    For each pair of consecutive windows, ``p = smooth(c_{w+1} conj(c_w))``
    gives a phase step ``angle(p)`` over ``dt = tau_{w+1} - tau_w``. The
    frequency is their least-squares fit, weighted by ``|p|``:
    ``sum |p| dt angle(p) / (2π sum |p| dt²)``. Each step must be below half
    a cycle, i.e. ``|f| < 1 / (2 dt)``.
    """
    if len(combined) < 2 or len(combined) != len(tau_s):
        raise ValueError("the field map needs at least two windows, each with a time")
    numerator = np.zeros(combined[0].shape, np.float64)
    denominator = np.zeros(combined[0].shape, np.float64)
    for w in range(len(combined) - 1):
        step = float(tau_s[w + 1] - tau_s[w])
        product = (combined[w + 1] * combined[w].conj())[None].astype(np.complex64)
        p = smooth(product, smoothing_voxels)[0]
        weight = np.abs(p)
        numerator += weight * step * np.angle(p)
        denominator += weight * step**2
    frequency: NDArray[np.float32] = (
        numerator / (2 * np.pi * np.maximum(denominator, np.finfo(np.float64).tiny))
    ).astype(np.float32)
    return frequency


def clip_to_signal(
    frequency_hz: NDArray[np.float32], weight: NDArray[np.float32], tail: float = 0.005
) -> NDArray[np.float32]:
    """The field map clipped to the weighted ``[tail, 1 - tail]`` quantiles.

    Where there is little signal (e.g. the background) the phase is noise,
    and its frequency can be anything up to the unwrapping limit. Weighted by
    the image magnitude, those voxels barely count in the quantiles, so the
    clipped map keeps the object's range, and the segmentation does not have
    to cover the noise.
    """
    order = np.argsort(frequency_hz, axis=None)
    cumulative = np.cumsum(weight.ravel()[order], dtype=np.float64)
    if cumulative[-1] <= 0:
        return frequency_hz
    cumulative /= cumulative[-1]
    sorted_frequency = frequency_hz.ravel()[order]
    low = sorted_frequency[int(np.searchsorted(cumulative, tail))]
    high = sorted_frequency[min(int(np.searchsorted(cumulative, 1 - tail)), len(order) - 1)]
    clipped: NDArray[np.float32] = np.clip(frequency_hz, low, high)
    return clipped


@dataclass(frozen=True)
class Segmentation:
    """Time segmentation of ``exp(2πi f tau)``."""

    segment_tau_s: FloatArray
    """(segments,) segment times."""
    coefficients: ComplexArray
    """(segments, taus) ``b_l(tau)`` at the taus it was made for."""
    rms_error: float
    """Largest RMS error over the taus, weighted by the frequency histogram."""


def segmentation(
    tau_s: FloatArray,
    segments: int,
    frequency_hz: NDArray[np.float32],
    weight: NDArray[np.float32],
    bins: int = 101,
) -> Segmentation:
    """Least-squares segmentation for the taus, over a weighted frequency histogram.

    The segments are evenly spaced over ``[min tau, max tau]``. The histogram
    has ``bins`` bins over the range of ``frequency_hz``, weighted by ``weight``
    (e.g. the image magnitude, so the background does not count).
    """
    segment_tau = np.linspace(tau_s.min(), tau_s.max(), segments)
    low, high = float(frequency_hz.min()), float(frequency_hz.max())
    if high - low < 1e-6:
        low, high = low - 0.5, high + 0.5
    counts, edges = np.histogram(frequency_hz, bins=bins, range=(low, high), weights=weight)
    if counts.sum() <= 0:
        counts = np.ones(bins, dtype=counts.dtype)
    f = (edges[:-1] + edges[1:]) / 2
    root = np.sqrt(counts / counts.sum())[:, None]
    basis = root * np.exp(2j * np.pi * f[:, None] * segment_tau[None])  # (bins, segments)
    target = root * np.exp(2j * np.pi * f[:, None] * tau_s[None])  # (bins, taus)
    coefficients, *_ = np.linalg.lstsq(basis, target, rcond=None)
    error = np.sqrt(np.sum(np.abs(basis @ coefficients - target) ** 2, axis=0))
    return Segmentation(
        segment_tau_s=segment_tau,
        coefficients=coefficients.astype(np.complex64),
        rms_error=float(error.max()),
    )
