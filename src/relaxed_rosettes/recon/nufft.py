"""3D non-uniform FFT between a voxel grid and k-space samples (FINUFFT, CPU).

The image is ``n`` voxels of size ``resolution`` along each physical
gradient axis (pypulseq x, y, z after the rotation event). Array index ``i``
is at position ``(i - n // 2) * resolution``, so the center voxel is at the
isocenter of the gradient frame. The signal model is

    y(k) = sum_r image(r) exp(+2πi k·r),

the sign of the encoding phase in ``relaxed_rosettes.trajectory``.
``forward`` evaluates it (FINUFFT type 2) and ``adjoint`` is its adjoint
(type 1). Neither is normalized.

This is the CPU backend. A CUDA backend (CuPy and cuFINUFFT) can implement
the same methods.

FINUFFT cannot share a process with PyTorch on macOS: both wheels bundle
their own OpenMP runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import finufft
import numpy as np
from numpy.typing import NDArray

ENCODING_SIGN = 1
"""Sign of the encoding phase: the signal is sum image(r) exp(ENCODING_SIGN 2πi k·r)."""

ComplexArray = NDArray[np.complex64]
KArray = NDArray[np.floating[Any]]
"""k-space points (points, 3), in 1/m; float32 or float64."""


@dataclass(frozen=True)
class Nufft:
    """NUFFT for an ``n``³ image of isotropic voxels ``resolution_m``."""

    n: int
    resolution_m: float
    eps: float = 1e-4
    upsampling: float = 2.0
    threads: int = 0

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.n, self.n, self.n)

    def coordinates(self, k_per_m: KArray) -> NDArray[np.float32]:
        """(3, points) FINUFFT coordinates (radians) of k-space points (points, 3).

        They are in ``[-π, π]`` for ``|k| <= 1 / (2 * resolution_m)``.
        """
        x: NDArray[np.float32] = np.ascontiguousarray(
            (2 * np.pi * self.resolution_m * k_per_m).T, dtype=np.float32
        )
        return x

    def forward(self, image: ComplexArray, k_per_m: KArray) -> ComplexArray:
        """(channels, points) signal of (channels, n, n, n) images at k (points, 3)."""
        x = self.coordinates(k_per_m)
        image = np.ascontiguousarray(image, dtype=np.complex64)
        y: ComplexArray = finufft.nufft3d2(
            x[0], x[1], x[2], image, isign=ENCODING_SIGN, **self._options()
        )
        return y.reshape(image.shape[0], -1)

    def adjoint(self, values: ComplexArray, k_per_m: KArray) -> ComplexArray:
        """(channels, n, n, n) adjoint of (channels, points) values at k (points, 3)."""
        x = self.coordinates(k_per_m)
        values = np.ascontiguousarray(values, dtype=np.complex64)
        if values.shape[1] == 0:
            return np.zeros((values.shape[0], *self.shape), dtype=np.complex64)
        image: ComplexArray = finufft.nufft3d1(
            x[0], x[1], x[2], values, self.shape, isign=-ENCODING_SIGN, **self._options()
        )
        return image.reshape(values.shape[0], *self.shape)

    def _options(self) -> dict[str, Any]:
        options: dict[str, Any] = {"eps": self.eps, "upsampfac": self.upsampling}
        if self.threads > 0:
            options["nthreads"] = self.threads
        return options
