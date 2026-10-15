"""Density compensation and time-averaged gridding reconstruction.

``pipe_menon_dcf`` finds weights ``w`` with ``sum_j w_j K(k_i - k_j) ≈ 1``
at every point (Pipe and Menon, MRM 41:179, 1999). The kernel ``K`` must be
non-negative for the iteration to converge, so it is not the point spread
function of the ``n``³ reconstruction grid (a Dirichlet kernel, with negative
lobes): it is the Fourier transform of a separable triangle window that is 1
at the center of a ``2n``³ grid and 0 at its edges. That is a product of
sinc² kernels, about 1/FOV wide. With full Cartesian sampling it gives
``w = 1/n³``, so the weighted adjoint is the inverse DFT, with the right
scale.

``grid`` is the weighted adjoint of all the selected data: one image per
virtual coil, averaged over the whole scan and the whole window of time
since excitation.

Both stream the selected samples a batch of TRs at a time; the weights of
each batch are kept in memory (one float32 per selected sample).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from relaxed_rosettes.config import Config
from relaxed_rosettes.recon.config import ReconConfig
from relaxed_rosettes.recon.nufft import KArray, Nufft
from relaxed_rosettes.recon.preprocess import CoilTransform, Selection, coil_transform
from relaxed_rosettes.recon.rawdata import RawData
from relaxed_rosettes.trajectory import Trajectory

ComplexArray = NDArray[np.complex64]
FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class Dcf:
    """Density compensation weights of each batch, and their convergence."""

    weights: list[NDArray[np.float32]]
    change: FloatArray
    """(iterations,): relative change of the weights in each iteration."""


def pipe_menon_dcf(nufft: Nufft, k_batches: Callable[[], Iterator[KArray]], iterations: int) -> Dcf:
    """Pipe-Menon weights for the points of ``k_batches()`` (a fresh iterator of (points, 3) k).

    Each iteration is ``w <- w / (K * w)``. Without iterations, the weights
    are uniform, scaled so that ``K * w`` averages 1.
    """
    wide = replace(nufft, n=2 * nufft.n)
    window = 1.0 - np.abs(np.arange(wide.n) - nufft.n) / nufft.n
    window_3d = (window[:, None, None] * window[None, :, None] * window[None, None, :]).astype(
        np.float32
    )
    weights = [np.ones(len(k), dtype=np.float32) for k in k_batches()]
    change = np.zeros(iterations)
    for it in range(-1, iterations):
        grid = np.zeros((1, *wide.shape), dtype=np.complex64)
        for k, w in zip(k_batches(), weights, strict=True):
            grid += wide.adjoint(w[None].astype(np.complex64), k)
        grid *= window_3d
        num = 0.0
        den = 0.0
        for i, k in enumerate(k_batches()):
            psf = np.abs(wide.forward(grid, k)[0])
            if it < 0:
                # Uniform start, scaled so that K * w averages 1.
                num += float(psf.sum())
                den += len(psf)
                continue
            new = weights[i] / np.maximum(psf, np.finfo(np.float32).tiny)
            num += float(np.sum((new - weights[i]) ** 2))
            den += float(np.sum(weights[i].astype(np.float64) ** 2))
            weights[i] = new.astype(np.float32)
        if it < 0:
            scale = den / num if num > 0 else 1.0
            weights = [(w * scale).astype(np.float32) for w in weights]
        else:
            change[it] = np.sqrt(num / den) if den > 0 else 0.0
    return Dcf(weights=weights, change=change)


def grid(
    nufft: Nufft,
    batches: Iterable[tuple[KArray, ComplexArray]],
    dcf: Dcf,
    channels: int,
) -> ComplexArray:
    """(channels, n, n, n) weighted adjoint of the (k, values) batches."""
    image = np.zeros((channels, *nufft.shape), dtype=np.complex64)
    for (k, values), w in zip(batches, dcf.weights, strict=True):
        image += nufft.adjoint(values * w[None], k)
    return image


def root_sum_of_squares(images: ComplexArray) -> NDArray[np.float32]:
    """(n, n, n) root sum of squares over the channel axis."""
    rss: NDArray[np.float32] = np.sqrt(np.sum(np.abs(images) ** 2, axis=0)).astype(np.float32)
    return rss


@dataclass(frozen=True)
class GriddingResult:
    """A time-averaged gridding reconstruction and what it was made from."""

    coil_images: ComplexArray
    """(virtual coils, n, n, n): one image per virtual coil."""
    rss: NDArray[np.float32]
    """(n, n, n): root sum of squares of the coil images."""
    coils: CoilTransform
    dcf_change: FloatArray
    num_trs: int
    num_samples: int
    recon: ReconConfig
    design: Config
    seq_signature: str


def reconstruct(
    raw: RawData, design: Config, trajectory: Trajectory, recon: ReconConfig
) -> GriddingResult:
    """Time-averaged gridding reconstruction of raw data with its trajectory."""
    raw.check_trajectory(design, trajectory)
    selection = Selection.make(recon, design, trajectory, raw.num_trs)
    coils = coil_transform(raw, trajectory, selection, recon, design.readout.adc_dwell_s)
    n = recon.nufft
    nufft = Nufft(
        n=recon.matrix_size(design),
        resolution_m=recon.data.resolution_m,
        eps=n.eps,
        upsampling=n.upsampling,
        threads=n.threads,
    )
    dcf = pipe_menon_dcf(nufft, lambda: selection.k_batches(trajectory), recon.dcf.iterations)
    images = grid(nufft, selection.batches(raw, trajectory, coils), dcf, coils.num_virtual_coils)
    return GriddingResult(
        coil_images=images,
        rss=root_sum_of_squares(images),
        coils=coils,
        dcf_change=dcf.change,
        num_trs=len(selection.trs),
        num_samples=sum(len(w) for w in dcf.weights),
        recon=recon,
        design=design,
        seq_signature=raw.seq_signature,
    )


def save_gridding(path: Path, result: GriddingResult, *, coil_images: bool = False) -> None:
    """Write a gridding reconstruction to an ``.npz`` file.

    Arrays: ``rss`` (n, n, n), ``coil_images`` (virtual coils, n, n, n; only
    with ``coil_images``), ``resolution_m``, ``coil_matrix`` (virtual coils ×
    coils; prewhitening then compression), ``noise_covariance`` (coils ×
    coils; empty if the raw data had no noise), ``singular_values``,
    ``dcf_change``, ``num_trs``, ``num_samples``, ``recon_config_toml``,
    ``recon_config_digest``, ``design_config_digest``, ``seq_signature``.
    Axis ``i`` of an image is physical gradient axis ``i`` (pypulseq x, y, z),
    and index ``j`` is at ``(j - n // 2) * resolution_m`` from the isocenter.
    """
    c = result.coils
    np.savez(
        path,
        rss=result.rss,
        coil_images=result.coil_images if coil_images else np.zeros((0,), np.complex64),
        resolution_m=result.recon.data.resolution_m,
        coil_matrix=c.matrix,
        noise_covariance=(
            c.noise_covariance if c.noise_covariance is not None else np.zeros((0, 0))
        ),
        singular_values=c.singular_values,
        dcf_change=result.dcf_change,
        num_trs=result.num_trs,
        num_samples=result.num_samples,
        recon_config_toml=result.recon.to_toml(),
        recon_config_digest=result.recon.digest(),
        design_config_digest=result.design.digest(),
        seq_signature=result.seq_signature,
    )
