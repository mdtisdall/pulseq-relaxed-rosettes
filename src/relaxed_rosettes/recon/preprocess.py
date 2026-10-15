"""Preprocessing: noise prewhitening, coil compression, and sample selection.

The data of each TR go through one complex matrix, ``CoilTransform.matrix``
(virtual coils × coils): the noise prewhitening, then the coil compression.
The whitening makes the noise of every channel white with unit variance at
the ADC dwell; the compression keeps the strongest singular vectors of the
whitened data, so the virtual coils keep unit-variance white noise.

A ``Selection`` picks the TRs and samples to use: a window of time since
excitation (the same samples in every TR) and the k-space ball of the
reconstruction's resolution (different samples in each TR, which has its own
rotation). ``Selection.batches`` streams the selected k and data, a batch of
TRs at a time, so the full data never need to be in memory.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from relaxed_rosettes.config import Config
from relaxed_rosettes.recon.config import ReconConfig
from relaxed_rosettes.recon.rawdata import RawData
from relaxed_rosettes.trajectory import Trajectory

ComplexArray = NDArray[np.complex64]
FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class CoilTransform:
    """Prewhitening and compression, as one matrix applied to the coil axis."""

    whitening: NDArray[np.complex128]
    """(coils, coils): inverse Cholesky factor of the noise covariance (identity if no noise)."""
    noise_covariance: NDArray[np.complex128] | None
    """(coils, coils): noise covariance at the ADC dwell, or None if the raw data have no noise."""
    compression: NDArray[np.complex128]
    """(virtual coils, coils): rows are orthonormal singular vectors of the whitened data."""
    singular_values: FloatArray
    """(coils,): singular values of the whitened calibration data, decreasing."""

    @property
    def matrix(self) -> ComplexArray:
        """(virtual coils, coils): compression @ whitening."""
        m: ComplexArray = (self.compression @ self.whitening).astype(np.complex64)
        return m

    @property
    def num_virtual_coils(self) -> int:
        return int(self.compression.shape[0])


def noise_whitening(
    raw: RawData, adc_dwell_s: float
) -> tuple[NDArray[np.complex128], NDArray[np.complex128] | None]:
    """(whitening matrix, noise covariance at the ADC dwell) from the raw data's noise.

    The noise variance per sample is proportional to the receiver bandwidth,
    1 / dwell, so a covariance measured at the noise dwell is scaled by
    ``noise_dwell / adc_dwell``. If the noise dwell is unknown, the noise is
    taken to be at the ADC dwell. Without noise samples, the whitening is the
    identity.
    """
    coils = raw.num_coils
    if raw.noise.shape[1] == 0:
        return np.eye(coils, dtype=np.complex128), None
    noise = raw.noise.astype(np.complex128)
    covariance = noise @ noise.conj().T / noise.shape[1]
    if raw.noise_dwell_s is not None:
        covariance *= raw.noise_dwell_s / adc_dwell_s
    factor = np.linalg.cholesky(covariance)
    whitening: NDArray[np.complex128] = np.linalg.inv(factor)
    return whitening, covariance


@dataclass(frozen=True)
class Selection:
    """The TRs and samples a reconstruction uses."""

    trs: NDArray[np.int64]
    """Acquired TR numbers (rows of the raw data and the trajectory)."""
    samples: NDArray[np.int64]
    """ADC samples in the window of time since excitation (the same in every TR)."""
    k_max_per_m: float
    """Radius of the k-space ball."""
    batch_trs: int

    @classmethod
    def make(
        cls, recon: ReconConfig, design: Config, trajectory: Trajectory, num_trs: int
    ) -> Selection:
        """Selection of the first ``num_trs`` TRs (or ``data.max_trs``, if fewer)."""
        d = recon.data
        recon.check_design(design)
        if d.max_trs:
            num_trs = min(num_trs, d.max_trs)
        tau = trajectory.time_since_excitation_s
        in_window = tau >= d.tau_min_s
        if d.tau_max_s:
            in_window &= tau <= d.tau_max_s
        samples = np.flatnonzero(in_window).astype(np.int64)
        if num_trs == 0 or len(samples) == 0:
            raise ValueError("the selection has no samples (check data.max_trs and the tau window)")
        return cls(
            trs=np.arange(num_trs, dtype=np.int64),
            samples=samples,
            k_max_per_m=1.0 / (2.0 * d.resolution_m),
            batch_trs=d.batch_trs,
        )

    def tr_batches(self) -> list[NDArray[np.int64]]:
        """The TRs in batches of ``batch_trs``, in order."""
        return [self.trs[i : i + self.batch_trs] for i in range(0, len(self.trs), self.batch_trs)]

    def k(
        self, trajectory: Trajectory, trs: NDArray[np.int64]
    ) -> tuple[FloatArray, NDArray[np.bool_]]:
        """(points, 3) selected k of some TRs, and the mask that picked them.

        The mask is (TRs, samples in the window).
        """
        k = trajectory.k_per_m(trs)[:, self.samples]
        inside = np.einsum("tsi,tsi->ts", k, k) <= self.k_max_per_m**2
        selected: FloatArray = k[inside]
        return selected, inside

    def k_batches(self, trajectory: Trajectory) -> Iterator[FloatArray]:
        """Selected k of each batch of TRs."""
        for trs in self.tr_batches():
            yield self.k(trajectory, trs)[0]

    def batches(
        self, raw: RawData, trajectory: Trajectory, coils: CoilTransform
    ) -> Iterator[tuple[FloatArray, ComplexArray]]:
        """(k (points, 3), data (virtual coils, points)) of each batch of TRs."""
        matrix = coils.matrix
        for trs in self.tr_batches():
            k, inside = self.k(trajectory, trs)
            data = raw.read(trs)[:, :, self.samples]
            values = np.einsum("vc,tcs->vts", matrix, data)[:, inside]
            yield k, values.astype(np.complex64)


def coil_transform(
    raw: RawData,
    trajectory: Trajectory,
    selection: Selection,
    recon: ReconConfig,
    adc_dwell_s: float,
) -> CoilTransform:
    """Prewhitening from the noise, and compression from calibration TRs.

    The calibration TRs are ``coils.calibration_trs`` of the selected TRs,
    spread evenly; only their selected samples are used.
    """
    whitening, covariance = noise_whitening(raw, adc_dwell_s)
    count = min(recon.coils.calibration_trs, len(selection.trs))
    picks = selection.trs[
        np.unique(np.linspace(0, len(selection.trs) - 1, count).round().astype(np.int64))
    ]
    gram = np.zeros((raw.num_coils, raw.num_coils), dtype=np.complex128)
    for i in range(0, len(picks), selection.batch_trs):
        trs = picks[i : i + selection.batch_trs]
        _, inside = selection.k(trajectory, trs)
        data = raw.read(trs)[:, :, selection.samples]
        values = np.einsum("dc,tcs->dts", whitening, data)[:, inside]
        gram += values @ values.conj().T
    eigenvalues, vectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    virtual = recon.coils.virtual_coils or raw.num_coils
    virtual = min(virtual, raw.num_coils)
    compression = vectors[:, order[:virtual]].conj().T
    return CoilTransform(
        whitening=whitening,
        noise_covariance=covariance,
        compression=compression,
        singular_values=np.sqrt(np.clip(eigenvalues[order], 0.0, None)),
    )
