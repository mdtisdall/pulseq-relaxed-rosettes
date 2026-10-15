"""Coil sensitivity maps by Inati's iterative method.

Inati, Hansen, and Kellman (ISMRM 2014, p. 4407) find, at every voxel, the
dominant eigenvector of the coil covariance of a neighbourhood (as Walsh et
al., MRM 43:682, 2000) by power iteration on all voxels at once: with maps
``s`` that vary slowly in space, ``sum_y d_y d_yᴴ s ≈ smooth(d (dᴴ s))``. The
maps have unit norm over the coils at every voxel, so the coil combination
``sᴴ d`` is the optimal-SNR combination for white noise (the data are
prewhitened). The phase of the maps is referenced to the coil combination
of the whole image, so it is smooth in space and the same in every map.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import uniform_filter

ComplexArray = NDArray[np.complex64]


def smooth(images: ComplexArray, size: int) -> ComplexArray:
    """Box filter of ``size`` voxels over the spatial axes of (channels, n, n, n) images."""
    if size <= 1:
        return images
    window = (1, size, size, size)
    real = uniform_filter(images.real, size=window, mode="nearest")
    imag = uniform_filter(images.imag, size=window, mode="nearest")
    smoothed: ComplexArray = (real + 1j * imag).astype(np.complex64)
    return smoothed


def inati_maps(images: ComplexArray, smoothing_voxels: int, iterations: int) -> ComplexArray:
    """(coils, n, n, n) sensitivity maps from (coils, n, n, n) coil images."""
    coils = images.shape[0]
    reference = images.reshape(coils, -1).sum(axis=1)
    reference /= max(float(np.linalg.norm(reference)), float(np.finfo(np.float32).tiny))
    reference_4d = reference[:, None, None, None].astype(np.complex64)
    maps = np.broadcast_to(reference_4d, images.shape).astype(np.complex64)
    for _ in range(iterations):
        combined = np.sum(maps.conj() * images, axis=0)
        maps = smooth(images * combined.conj()[None], smoothing_voxels)
        norm = np.sqrt(np.sum(np.abs(maps) ** 2, axis=0))
        maps /= np.maximum(norm, np.finfo(np.float32).tiny)[None]
        phase = np.sum(reference_4d.conj() * maps, axis=0)
        maps *= np.exp(-1j * np.angle(phase))[None].astype(np.complex64)
    return maps


def combine(images: ComplexArray, maps: ComplexArray) -> ComplexArray:
    """(n, n, n) coil combination ``sᴴ d`` of (coils, n, n, n) images."""
    combined: ComplexArray = np.sum(maps.conj() * images, axis=0)
    return combined


def smoothing_voxels(width_m: float, resolution_m: float) -> int:
    """Odd box filter width, in voxels, for a width in meters."""
    voxels = max(1, round(width_m / resolution_m))
    return voxels if voxels % 2 else voxels + 1
