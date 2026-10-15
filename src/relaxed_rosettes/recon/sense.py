"""Static CG-SENSE reconstruction, with an optional off-resonance term.

The encoding operator ``E`` maps one complex image ``x`` (n³) to the data of
every virtual coil at every selected sample:

    (E x)_v(k, tau) = sum_l b_l(tau) NUFFT[s_v exp(2πi f tau_l) x](k)

with coil maps ``s_v`` (``coilmaps``) and the time segmentation ``b_l`` of
the field map ``f`` (``b0``); without the off-resonance term it is
``NUFFT[s_v x]``. CG solves the normal equations
``(Eᴴ W E + λ I) x = Eᴴ W y``, with ``W`` the density compensation (or the
identity), a batch of TRs at a time over data cached in memory.

``reconstruct`` estimates everything it needs from the scan: the coil maps
and the field map from low-resolution gridding of the cached data, then the
image. It is averaged over the whole scan (one image; no scan-time axis).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from relaxed_rosettes.config import Config
from relaxed_rosettes.recon.b0 import Segmentation, clip_to_signal, field_map_hz, segmentation
from relaxed_rosettes.recon.coilmaps import combine, inati_maps, smoothing_voxels
from relaxed_rosettes.recon.config import ReconConfig, Weighting
from relaxed_rosettes.recon.gridding import Dcf, grid, pipe_menon_dcf
from relaxed_rosettes.recon.nufft import Nufft
from relaxed_rosettes.recon.preprocess import (
    CachedBatch,
    CoilTransform,
    Selection,
    cache_data,
    coil_transform,
)
from relaxed_rosettes.recon.rawdata import RawData
from relaxed_rosettes.trajectory import Trajectory

ComplexArray = NDArray[np.complex64]
FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class OffResonance:
    """The off-resonance term: segment phase images and per-sample coefficients."""

    phases: ComplexArray
    """(segments, n, n, n) ``exp(2πi f tau_l)``."""
    coefficients: ComplexArray
    """(segments, samples) ``b_l(tau)`` of each ADC sample in ``Selection.samples``."""


@dataclass(frozen=True)
class SenseOperator:
    """The encoding operator ``E`` for coil maps and an optional off-resonance term."""

    nufft: Nufft
    maps: ComplexArray
    """(virtual coils, n, n, n), unit norm over the coils."""
    off_resonance: OffResonance | None = None

    def forward(self, image: ComplexArray, batch: CachedBatch) -> ComplexArray:
        """(virtual coils, points) ``E x`` at the points of a batch."""
        k = batch.k_per_m
        if self.off_resonance is None:
            return self.nufft.forward(self.maps * image[None], k)
        o = self.off_resonance
        out = np.zeros((self.maps.shape[0], len(k)), np.complex64)
        for phase, coefficients in zip(o.phases, o.coefficients, strict=True):
            out += coefficients[batch.sample][None] * self.nufft.forward(
                self.maps * (phase * image)[None], k
            )
        return out

    def adjoint(self, values: ComplexArray, batch: CachedBatch) -> ComplexArray:
        """(n, n, n) ``Eᴴ y`` of the (virtual coils, points) values of a batch."""
        k = batch.k_per_m
        if self.off_resonance is None:
            return combine(self.nufft.adjoint(values, k), self.maps)
        o = self.off_resonance
        out = np.zeros(self.nufft.shape, np.complex64)
        for phase, coefficients in zip(o.phases, o.coefficients, strict=True):
            images = self.nufft.adjoint(coefficients[batch.sample].conj()[None] * values, k)
            out += phase.conj() * combine(images, self.maps)
        return out


@dataclass(frozen=True)
class CgResult:
    image: ComplexArray
    residual: FloatArray
    """(iterations + 1,) residual norm of the normal equations, relative to the first."""


def cg_sense(
    operator: SenseOperator,
    batches: Sequence[CachedBatch],
    weights: Sequence[NDArray[np.float32]] | None,
    tikhonov: float,
    iterations: int,
    tolerance: float,
) -> CgResult:
    """Conjugate gradients on ``(Eᴴ W E + λ I) x = Eᴴ W y`` from ``x = 0``."""

    def weighted(i: int, values: ComplexArray) -> ComplexArray:
        return values if weights is None else values * weights[i][None]

    def normal(x: ComplexArray) -> ComplexArray:
        out = tikhonov * x
        for i, batch in enumerate(batches):
            out += operator.adjoint(weighted(i, operator.forward(x, batch)), batch)
        return out

    rhs = np.zeros(operator.nufft.shape, np.complex64)
    for i, batch in enumerate(batches):
        rhs += operator.adjoint(weighted(i, batch.values), batch)
    x = np.zeros_like(rhs)
    r = rhs.copy()
    p = r.copy()
    rr = float(np.vdot(r, r).real)
    initial = np.sqrt(rr)
    residual = [1.0]
    for _ in range(iterations):
        if initial == 0 or np.sqrt(rr) <= tolerance * initial:
            break
        q = normal(p)
        alpha = rr / float(np.vdot(p, q).real)
        x += alpha * p
        r -= alpha * q
        rr_new = float(np.vdot(r, r).real)
        p = r + (rr_new / rr) * p
        rr = rr_new
        residual.append(np.sqrt(rr) / initial)
    return CgResult(image=x, residual=np.array(residual))


@dataclass(frozen=True)
class SenseResult:
    """A static CG-SENSE reconstruction and what it was made from."""

    image: ComplexArray
    maps: ComplexArray
    field_map_hz: NDArray[np.float32] | None
    segmentation: Segmentation | None
    residual: FloatArray
    coils: CoilTransform
    num_trs: int
    num_samples: int
    recon: ReconConfig
    design: Config
    seq_signature: str


def _grid_subset(
    nufft: Nufft, batches: Sequence[CachedBatch], iterations: int, channels: int
) -> ComplexArray:
    """Density-compensated gridding of the batches."""
    dcf = pipe_menon_dcf(nufft, lambda: (b.k_per_m for b in batches), iterations)
    return grid(nufft, ((b.k_per_m, b.values) for b in batches), dcf, channels)


def estimate_maps(nufft: Nufft, cached: Sequence[CachedBatch], recon: ReconConfig) -> ComplexArray:
    """Coil maps from the low-resolution part of the cached data."""
    m = recon.coil_maps
    resolution = max(m.resolution_m, recon.data.resolution_m)
    low = [b.subset(1.0 / (2.0 * resolution)) for b in cached]
    images = _grid_subset(nufft, low, recon.dcf.iterations, cached[0].values.shape[0])
    return inati_maps(images, smoothing_voxels(m.smoothing_m, nufft.resolution_m), m.iterations)


def estimate_off_resonance(
    nufft: Nufft,
    cached: Sequence[CachedBatch],
    maps: ComplexArray,
    window_tau_s: FloatArray,
    recon: ReconConfig,
) -> tuple[NDArray[np.float32], Segmentation, OffResonance]:
    """Field map from windows of tau, and its segmentation for ``window_tau_s``.

    ``window_tau_s`` is the tau of each ADC sample in ``Selection.samples``.
    """
    b = recon.b0
    k_max = 1.0 / (2.0 * max(b.resolution_m, recon.data.resolution_m))
    edges = np.linspace(window_tau_s.min(), window_tau_s.max(), b.windows + 1)
    edges[-1] = np.nextafter(edges[-1], np.inf)
    combined = []
    times = []
    for t0, t1 in zip(edges[:-1], edges[1:], strict=True):
        sub = [c.subset(k_max, float(t0), float(t1)) for c in cached]
        count = sum(len(c.tau_s) for c in sub)
        if count == 0:
            continue
        images = _grid_subset(nufft, sub, recon.dcf.iterations, maps.shape[0])
        combined.append(combine(images, maps))
        times.append(sum(float(c.tau_s.sum(dtype=np.float64)) for c in sub) / count)
    if len(combined) < 2:
        raise ValueError("the field map needs samples in at least two b0.windows")
    frequency = field_map_hz(
        combined, np.array(times), smoothing_voxels(b.smoothing_m, nufft.resolution_m)
    )
    magnitude = np.abs(sum(combined, start=np.zeros_like(combined[0]))).astype(np.float32)
    frequency = clip_to_signal(frequency, magnitude)
    seg = segmentation(window_tau_s, b.segments, frequency.ravel(), magnitude.ravel())
    phases = np.exp(2j * np.pi * frequency[None] * seg.segment_tau_s[:, None, None, None]).astype(
        np.complex64
    )
    return frequency, seg, OffResonance(phases=phases, coefficients=seg.coefficients)


def reconstruct(
    raw: RawData, design: Config, trajectory: Trajectory, recon: ReconConfig
) -> SenseResult:
    """Static CG-SENSE reconstruction of raw data with its trajectory."""
    raw.check_trajectory(design, trajectory)
    selection = Selection.make(recon, design, trajectory, raw.num_trs)
    coils = coil_transform(raw, trajectory, selection, recon, design.readout.adc_dwell_s)
    cached = cache_data(selection, raw, trajectory, coils)
    n = recon.nufft
    nufft = Nufft(
        n=recon.matrix_size(design),
        resolution_m=recon.data.resolution_m,
        eps=n.eps,
        upsampling=n.upsampling,
        threads=n.threads,
    )
    maps = estimate_maps(nufft, cached, recon)
    frequency = None
    seg = None
    off_resonance = None
    if recon.b0.segments:
        window_tau = trajectory.time_since_excitation_s[selection.samples]
        frequency, seg, off_resonance = estimate_off_resonance(
            nufft, cached, maps, window_tau, recon
        )
    operator = SenseOperator(nufft=nufft, maps=maps, off_resonance=off_resonance)
    weights: list[NDArray[np.float32]] | None = None
    if recon.cg.weighting is Weighting.DCF:
        dcf: Dcf = pipe_menon_dcf(nufft, lambda: (b.k_per_m for b in cached), recon.dcf.iterations)
        weights = dcf.weights
    g = recon.cg
    result = cg_sense(operator, cached, weights, g.tikhonov, g.iterations, g.tolerance)
    return SenseResult(
        image=result.image,
        maps=maps,
        field_map_hz=frequency,
        segmentation=seg,
        residual=result.residual,
        coils=coils,
        num_trs=len(selection.trs),
        num_samples=sum(len(b.tau_s) for b in cached),
        recon=recon,
        design=design,
        seq_signature=raw.seq_signature,
    )


def save_sense(path: Path, result: SenseResult, *, coil_maps: bool = False) -> None:
    """Write a CG-SENSE reconstruction to an ``.npz`` file.

    Arrays: ``image`` (n, n, n complex), ``magnitude``, ``resolution_m``,
    ``field_map_hz`` (n, n, n; empty without the off-resonance term),
    ``segment_tau_s`` and ``segmentation_rms_error``, ``coil_maps`` (virtual
    coils, n, n, n; only with ``coil_maps``), ``cg_residual``, ``coil_matrix``,
    ``num_trs``, ``num_samples``, ``recon_config_toml``,
    ``recon_config_digest``, ``design_config_digest``, ``seq_signature``.
    The image axes are as in ``gridding.save_gridding``.
    """
    seg = result.segmentation
    np.savez(
        path,
        image=result.image,
        magnitude=np.abs(result.image).astype(np.float32),
        resolution_m=result.recon.data.resolution_m,
        field_map_hz=(
            result.field_map_hz if result.field_map_hz is not None else np.zeros((0,), np.float32)
        ),
        segment_tau_s=seg.segment_tau_s if seg is not None else np.zeros((0,)),
        segmentation_rms_error=seg.rms_error if seg is not None else 0.0,
        coil_maps=result.maps if coil_maps else np.zeros((0,), np.complex64),
        cg_residual=result.residual,
        coil_matrix=result.coils.matrix,
        num_trs=result.num_trs,
        num_samples=result.num_samples,
        recon_config_toml=result.recon.to_toml(),
        recon_config_digest=result.recon.digest(),
        design_config_digest=result.design.digest(),
        seq_signature=result.seq_signature,
    )
