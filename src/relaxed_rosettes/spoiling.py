"""Gradient spoiling of a design: residual coherences and where the readouts sample them.

Each TR ends at its shape's spoil moment, turned by the TR's rotation, so the
schedule decides where the spoiled magnetization of every earlier TR lies.
This module follows the magnetization of one voxel through the schedule with
an extended phase graph whose dephasing states are continuous 3D vectors
(k, in cycles/m) instead of integers:

- RF pulse of flip α and phase φ (Weigel, JMRI 41:266, 2015)::

      F+'_k = cos²(α/2) F+_k + e^{2iφ} sin²(α/2) F-_k - i e^{iφ} sin α Z_k
      F-'_k = e^{-2iφ} sin²(α/2) F+_k + cos²(α/2) F-_k + i e^{-iφ} sin α Z_k
      Z'_k  = -(i/2) e^{-iφ} sin α F+_k + (i/2) e^{iφ} sin α F-_k + cos α Z_k

  with ``F-_k = conj(F+_{-k})`` and ``Z_{-k} = conj(Z_k)``. Only ``F+``
  states are stored, at any vector; a ``Z`` state at ``k != 0`` stands for
  the pair ``(Z_k, Z_{-k})``, and ``Z_0`` is real.
- Readout: a state ``F+_k`` at the start of TR ``q`` is dephased by
  ``k + k_q(t)`` at sample ``t``, so it is sampled where the readout passes
  ``-k``. The *closest approach* is the smallest ``|k + k_q(t)|`` over the
  TR's ADC samples, in cells (1/FOV). The fresh FID (``F+_0`` from ``Z_0``)
  is the signal the reconstruction expects; every other state is spurious.
- Relaxation over the TR: ``F`` by ``exp(-TR/T2)``, ``Z`` by ``exp(-TR/T1)``
  with recovery of ``Z_0``, and diffusion: ``exp(-b D)`` with
  ``b = (2π)² TR (|k|² + k·s + |s|²/3)`` for a transverse state that moves
  from ``k`` to ``k + s`` (a linear ramp, ignoring the readout's
  excursions), and ``(2π|k|)² TR`` for a stored one.
- Spoiler: ``F+_k -> F+_{k+s}`` with ``s`` the TR's rotated spoil moment
  (the k-space position at the end of its shape).

States below ``threshold`` (of the equilibrium magnetization) are dropped,
and at most ``max_states`` are kept, the largest. Intravoxel off-resonance
(T2') is ignored, so transverse pathways are overestimated.

Each state carries flags: conjugated (it has passed through ``F-``) and
stored (it has passed through ``Z``). A spurious state that is neither is a
transverse chain.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from relaxed_rosettes.config import Config
from relaxed_rosettes.optimize import Shape
from relaxed_rosettes.sequence import Schedule
from relaxed_rosettes.trajectory import kspace_at_times

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]

CONJUGATED = 1
STORED = 2

NEAR_CELLS = (1.0, 3.0, 10.0)
"""Closest approaches (cells) at which spurious amplitude is summed."""


@dataclass(frozen=True)
class Tissue:
    """Relaxation and diffusion of the simulated voxel."""

    t1_s: float
    t2_s: float
    diffusion_m2_per_s: float = 0.0


@dataclass(frozen=True)
class SpoilingResult:
    """Residual coherences at each acquired TR."""

    tissue: Tissue
    fid: FloatArray
    """(TRs,): |F+_0| after the TR's pulse (the expected signal)."""
    ideal_fid: FloatArray
    """(TRs,): the FID with ideal spoiling (every transverse state destroyed),
    including the approach to the steady state."""
    near_amplitude: FloatArray
    """(TRs, len(NEAR_CELLS)): sum of the spurious amplitudes (relative to the
    TR's FID) whose closest approach is within each distance."""
    near_by_kind: FloatArray
    """(TRs, 3): the sum within NEAR_CELLS[1], for transverse chains,
    conjugated, and stored pathways."""
    closest_cells: FloatArray
    """(TRs,): closest approach of the strongest spurious state within
    NEAR_CELLS[-1] (inf if none)."""
    energy_ratio: FloatArray
    """(TRs,): spurious over expected signal energy, summed over the TR's
    samples, for the given object (NaN without one)."""
    states: NDArray[np.int64]
    """(TRs,): states tracked after the TR's pulse."""
    capped: bool
    """Whether ``max_states`` was reached."""


class _States:
    """Transverse (F+) and stored (Z, k != 0) states, and Z_0."""

    def __init__(self) -> None:
        self.f_k = np.zeros((0, 3))
        self.f_a = np.zeros(0, dtype=np.complex128)
        self.f_flag = np.zeros(0, dtype=np.int64)
        self.z_k = np.zeros((0, 3))
        self.z_a = np.zeros(0, dtype=np.complex128)
        self.z_flag = np.zeros(0, dtype=np.int64)
        self.z0 = 1.0


ENERGY_STATES = 32
"""The strongest spurious states near a readout that the energy ratio samples."""


def _merge(
    k: FloatArray, a: ComplexArray, flag: NDArray[np.int64], fov_m: float, full: bool
) -> tuple[FloatArray, ComplexArray, NDArray[np.int64]]:
    """Sum states at the same vector (to 1e-6 cells).

    Without ``full``, only the states at k = 0: elsewhere, two pathways
    reach the same vector only if the schedule repeats a spoil vector.
    """
    if not full:
        zero = _is_zero(k, fov_m)
        if zero.sum() < 2:
            return k, a, flag
        merged_a = np.append(a[~zero], np.sum(a[zero]))
        merged_flag = np.append(flag[~zero], np.bitwise_or.reduce(flag[zero]))
        return np.vstack([k[~zero], np.zeros((1, 3))]), merged_a, merged_flag
    if len(k) < 2:
        return k, a, flag
    keys = np.round(k * fov_m * 1e6).astype(np.int64)
    unique, first, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    if len(unique) == len(k):
        return k, a, flag
    inverse = inverse.ravel()
    merged = np.zeros(len(unique), dtype=np.complex128)
    np.add.at(merged, inverse, a)
    flags = np.zeros(len(unique), dtype=np.int64)
    np.bitwise_or.at(flags, inverse, flag)
    return k[first], merged, flags


def _is_zero(k: FloatArray, fov_m: float) -> NDArray[np.bool_]:
    zero: NDArray[np.bool_] = np.all(np.abs(k * fov_m) < 1e-6, axis=1)
    return zero


def simulate_spoiling(
    config: Config,
    schedule: Schedule,
    shapes: list[Shape],
    tissue: Tissue,
    *,
    rf_phase_rad: FloatArray | None = None,
    rotation_matrix: FloatArray | None = None,
    threshold: float = 1e-5,
    max_states: int = 200_000,
    obj: object | None = None,
) -> SpoilingResult:
    """Residual coherences of a design's schedule, for one tissue.

    ``rf_phase_rad`` and ``rotation_matrix`` (one per TR, dummies first)
    replace the schedule's. ``obj`` is a ``relaxed_rosettes.recon.phantom
    .PhantomConfig``; with it, each spurious state near the readout is
    sampled from the phantom's spectrum to give the energy ratio (the
    ``ENERGY_STATES`` strongest within ``NEAR_CELLS[-1]``).
    """
    fov = config.geometry.fov_m
    tr = config.readout.tr_s
    alpha = np.deg2rad(config.excitation.flip_angle_deg)
    c2, s2, sa, ca = np.cos(alpha / 2) ** 2, np.sin(alpha / 2) ** 2, np.sin(alpha), np.cos(alpha)
    e1, e2 = np.exp(-tr / tissue.t1_s), np.exp(-tr / tissue.t2_s)
    phases = schedule.rf_phase_rad if rf_phase_rad is None else rf_phase_rad
    rotations = schedule.rotation_matrix if rotation_matrix is None else rotation_matrix

    raster = config.system.grad_raster_time_s
    adc_time = np.asarray(
        config.adc_start_s
        + (np.arange(config.readout.adc_num_samples) + 0.5) * config.readout.adc_dwell_s,
        dtype=np.float64,
    )
    k_adc = [kspace_at_times(s.g_mT_per_m, raster, adc_time) for s in shapes]
    spoil = np.stack(
        [
            kspace_at_times(s.g_mT_per_m, raster, np.array([len(s.g_mT_per_m) * raster]))[0]
            for s in shapes
        ]
    )
    reach = max(float(np.linalg.norm(k, axis=1).max()) for k in k_adc) * fov + NEAR_CELLS[-1]
    trees = [cKDTree(-k * fov) for k in k_adc]
    two_pi2_tr_d = (2 * np.pi) ** 2 * tr * tissue.diffusion_m2_per_s
    spoils = np.einsum("tij,tj->ti", rotations, spoil[schedule.shape_index])
    keys = np.round(spoils * fov * 1e6).astype(np.int64)
    full_merge = len(np.unique(np.concatenate([keys, -keys]), axis=0)) < 2 * len(keys)

    states = _States()
    acquired = np.flatnonzero(schedule.acquired)
    n_acq = len(acquired)
    fid = np.zeros(n_acq)
    ideal = np.zeros(n_acq)
    z_ideal = 1.0
    near = np.zeros((n_acq, len(NEAR_CELLS)))
    by_kind = np.zeros((n_acq, 3))
    closest = np.full(n_acq, np.inf)
    energy = np.full(n_acq, np.nan)
    counts = np.zeros(n_acq, dtype=np.int64)
    capped = False
    acquired_slot = {int(t): i for i, t in enumerate(acquired)}

    for t in range(len(schedule.shape_index)):
        phi = float(phases[t])
        e_p, e_2p = np.exp(1j * phi), np.exp(2j * phi)
        s = states
        # RF pulse.
        fid_amp = -1j * e_p * sa * s.z0
        new_f_k = [s.f_k, -s.f_k, s.z_k, -s.z_k, np.zeros((1, 3))]
        new_f_a = [
            c2 * s.f_a,
            e_2p * s2 * np.conj(s.f_a),
            -1j * e_p * sa * s.z_a,
            -1j * e_p * sa * np.conj(s.z_a),
            np.array([fid_amp]),
        ]
        new_f_flag = [
            s.f_flag,
            s.f_flag | CONJUGATED,
            s.z_flag,
            s.z_flag | CONJUGATED,
            np.zeros(1, dtype=np.int64),
        ]
        z_from_f = -0.5j * np.exp(-1j * phi) * sa * s.f_a
        z_k = np.concatenate([s.z_k, s.f_k])
        z_a = np.concatenate([ca * s.z_a, z_from_f])
        z_flag = np.concatenate([s.z_flag, s.f_flag | STORED])
        zero = _is_zero(z_k, fov)
        z0 = ca * s.z0 + float(np.sum(2 * np.real(z_a[zero])))
        f_k = np.concatenate(new_f_k)
        f_a = np.concatenate(new_f_a)
        f_flag = np.concatenate(new_f_flag)
        f_k, f_a, f_flag = _merge(f_k, f_a, f_flag, fov, full_merge)
        z_k, z_a, z_flag = _merge(z_k[~zero], z_a[~zero], z_flag[~zero], fov, full_merge)

        # Readout.
        if t in acquired_slot:
            i = acquired_slot[t]
            ideal[i] = sa * z_ideal
            at_zero = _is_zero(f_k, fov)
            signal = abs(complex(np.sum(f_a[at_zero])))
            fid[i] = signal
            counts[i] = len(f_k) + len(z_k)
            spurious = ~at_zero & (np.abs(f_a) > 0)
            candidates = np.flatnonzero(spurious & (np.linalg.norm(f_k, axis=1) * fov <= reach))
            if len(candidates) and signal > 0:
                shape = int(schedule.shape_index[t])
                logical = f_k[candidates] @ rotations[t]  # Rᵀ k, row-wise
                distance, _ = trees[shape].query(logical * fov, k=1)
                relative = np.abs(f_a[candidates]) / signal
                for j, cells in enumerate(NEAR_CELLS):
                    near[i, j] = float(np.sum(relative[distance <= cells]))
                inner = distance <= NEAR_CELLS[1]
                flags = f_flag[candidates][inner]
                rel_inner = relative[inner]
                by_kind[i, 0] = float(np.sum(rel_inner[flags == 0]))
                conjugated = ((flags & CONJUGATED) > 0) & ((flags & STORED) == 0)
                by_kind[i, 1] = float(np.sum(rel_inner[conjugated]))
                by_kind[i, 2] = float(np.sum(rel_inner[(flags & STORED) > 0]))
                within = distance <= NEAR_CELLS[-1]
                if within.any():
                    strongest = np.argmax(np.where(within, relative, -1.0))
                    closest[i] = float(distance[strongest])
                if obj is not None:
                    order = np.argsort(np.where(within, relative, -1.0))[::-1]
                    strongest_near = candidates[order[: min(ENERGY_STATES, int(within.sum()))]]
                    energy[i] = _energy_ratio(
                        obj, rotations[t] @ k_adc[shape].T, f_k, f_a, strongest_near, at_zero
                    )
            elif obj is not None:
                energy[i] = 0.0

        # Relaxation, diffusion, and the spoiler.
        spoil_t = spoils[t]
        if two_pi2_tr_d:
            b_f = np.sum(f_k**2, axis=1) + f_k @ spoil_t + spoil_t @ spoil_t / 3
            f_a = f_a * e2 * np.exp(-two_pi2_tr_d * b_f)
            z_a = z_a * e1 * np.exp(-two_pi2_tr_d * np.sum(z_k**2, axis=1))
        else:
            f_a = f_a * e2
            z_a = z_a * e1
        z0 = z0 * e1 + (1 - e1)
        z_ideal = ca * z_ideal * e1 + (1 - e1)
        f_k = f_k + spoil_t

        keep_f = np.abs(f_a) >= threshold
        keep_z = np.abs(z_a) >= threshold
        f_k, f_a, f_flag = f_k[keep_f], f_a[keep_f], f_flag[keep_f]
        z_k, z_a, z_flag = z_k[keep_z], z_a[keep_z], z_flag[keep_z]
        if len(f_a) + len(z_a) > max_states:
            capped = True
            floor = np.sort(np.abs(np.concatenate([f_a, z_a])))[-max_states]
            keep_f, keep_z = np.abs(f_a) >= floor, np.abs(z_a) >= floor
            f_k, f_a, f_flag = f_k[keep_f], f_a[keep_f], f_flag[keep_f]
            z_k, z_a, z_flag = z_k[keep_z], z_a[keep_z], z_flag[keep_z]
        s.f_k, s.f_a, s.f_flag = f_k, f_a, f_flag
        s.z_k, s.z_a, s.z_flag = z_k, z_a, z_flag
        s.z0 = z0

    return SpoilingResult(
        tissue=tissue,
        fid=fid,
        ideal_fid=ideal,
        near_amplitude=near,
        near_by_kind=by_kind,
        closest_cells=closest,
        energy_ratio=energy,
        states=counts,
        capped=capped,
    )


def _energy_ratio(
    obj: object,
    k_tr: FloatArray,
    f_k: FloatArray,
    f_a: ComplexArray,
    spurious: NDArray[np.int64],
    at_zero: NDArray[np.bool_],
) -> float:
    """Spurious over expected signal energy over one TR's samples.

    ``k_tr`` (3, samples) is the TR's physical k; each state ``F+_k`` adds
    its amplitude times the object's spectrum at ``k + k_tr``.
    """
    signal = getattr(obj, "signal")  # noqa: B009 - duck-typed PhantomConfig
    k_samples = k_tr.T
    zero = np.zeros(len(k_samples))
    expected = complex(np.sum(f_a[at_zero])) * signal(k_samples, zero, zero)
    residual = np.zeros(len(k_samples), dtype=np.complex128)
    for j in spurious:
        residual += f_a[j] * signal(k_samples + f_k[j], zero, zero)
    total = float(np.sum(np.abs(expected) ** 2))
    return float(np.sum(np.abs(residual) ** 2)) / total if total > 0 else float("nan")


def format_spoiling(results: list[SpoilingResult]) -> str:
    """A summary table, one row per tissue."""
    lines = [
        f"{'T1 (s)':>6} {'T2 (s)':>6} {'D (um2/ms)':>10} {'TRs >1%/>10% (<=3)':>18} "
        + " ".join(f"{f'<={c:g} cells mean/max':>22}" for c in NEAR_CELLS)
        + f" {'chain/conj/stored (<=3)':>24} {'energy mean/max':>16} {'states':>8}"
    ]
    for r in results:
        t = r.tissue
        inner = r.near_amplitude[:, 1]
        fractions = f"{np.mean(inner > 0.01):.1%}/{np.mean(inner > 0.1):.1%}"
        near = " ".join(
            f"{np.mean(r.near_amplitude[:, j]):>10.2e}/{np.max(r.near_amplitude[:, j]):<11.2e}"
            for j in range(len(NEAR_CELLS))
        )
        kinds = "/".join(f"{np.mean(r.near_by_kind[:, j]):.1e}" for j in range(3))
        energy = (
            f"{np.nanmean(r.energy_ratio):.2e}/{np.nanmax(r.energy_ratio):.2e}"
            if np.any(np.isfinite(r.energy_ratio))
            else "-"
        )
        lines.append(
            f"{t.t1_s:>6g} {t.t2_s:>6g} {t.diffusion_m2_per_s * 1e9:>10g} {fractions:>18} "
            f"{near} {kinds:>24} {energy:>16} "
            f"{int(r.states.max()):>7d}{'*' if r.capped else ' '}"
        )
    lines.append(
        "Amplitudes are relative to each TR's FID; TRs >1%/>10%: the fraction of TRs whose "
        "spurious amplitude within 3 cells exceeds 1% and 10%; '*': the state cap was reached "
        "(states: after the pulse, before pruning). Energy: spurious/expected over each TR's "
        "samples, for the phantom."
    )
    return "\n".join(lines)
