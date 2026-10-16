"""k-space trajectory at the ADC samples, for reconstruction.

Every acquired TR samples the same ADC times within its block, and plays one
shape rotated by its rotation matrix. The trajectory is therefore stored
compactly: k of each shape at the ADC sample times (logical axes), plus the
shape index and rotation of each acquired TR. The k-space position of sample
``s`` of acquired TR ``a`` is ``rotation[a] @ k_shape[shape_index[a], s]``.

k is computed as the exact integral of the piecewise-linear gradient that
pypulseq plays, with k = 0 at the RF center (the gradients are off during the
RF pulse).

For the concomitant (Maxwell) field, the export also has the time integral of
the gradient outer product, ``gg(t) = ∫ g gᵀ dt`` (3 × 3, symmetric), of each
shape at the ADC samples. The lowest-order concomitant field is quadratic in
the gradient, so in a TR with rotation ``R`` its phase at every position
follows from ``R @ gg @ Rᵀ`` (``concomitant_phase_rad``).

Gradient delays. A delay ``δ_i`` on physical axis ``i`` plays that axis's
gradient late: ``g_i(t) = g_nominal_i(t - δ_i)``, so ``k_i(t) =
k_nominal_i(t - δ_i)``. With delays, the trajectory also stores k of each
shape at the ADC times shifted by each axis's delay (``k_shape_delayed_per_m``),
and ``k_per_m`` takes physical component ``i`` from the shape's k at
``t - δ_i``. This is exact for the piecewise-linear waveform. ``gg`` stays
nominal: a delay changes the concomitant phase only to second order.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pypulseq as pp
from numpy.typing import NDArray

from relaxed_rosettes.config import GAMMA_HZ_PER_T, Config
from relaxed_rosettes.optimize import Shape
from relaxed_rosettes.sequence import Schedule

FloatArray = NDArray[np.float64]


def kspace_at_times(g_mT_per_m: FloatArray, raster_s: float, t_s: FloatArray) -> FloatArray:
    """k (1/m) of a shape at times ``t_s`` from the start of its block.

    The waveform is the one pypulseq plays: ``first = 0`` at time 0, sample
    ``i`` at ``(i + 0.5) * raster``, ``last = 0`` at ``N * raster``, linear in
    between. Its integral is exact (quadratic within each interval).
    """
    n = len(g_mT_per_m)
    t_points = np.concatenate([[0.0], (np.arange(n) + 0.5) * raster_s, [n * raster_s]])
    g_points = np.concatenate([np.zeros((1, 3)), g_mT_per_m, np.zeros((1, 3))])
    dt = np.diff(t_points)
    slope = np.diff(g_points, axis=0) / dt[:, None]
    integral = np.concatenate(
        [np.zeros((1, 3)), np.cumsum((g_points[:-1] + g_points[1:]) / 2 * dt[:, None], axis=0)]
    )
    if np.any(t_s < 0) or np.any(t_s > t_points[-1]):
        raise ValueError("a time is outside the block")
    j = np.clip(np.searchsorted(t_points, t_s, side="right") - 1, 0, len(dt) - 1)
    tau = (t_s - t_points[j])[:, None]
    g_integral = integral[j] + g_points[j] * tau + 0.5 * slope[j] * tau**2
    k: FloatArray = g_integral * 1e-3 * GAMMA_HZ_PER_T
    return k


def gradient_outer_integral_at_times(
    g_mT_per_m: FloatArray, raster_s: float, t_s: FloatArray
) -> FloatArray:
    """(times, 3, 3) ``∫ g gᵀ dt`` (T² s / m²) of a shape from the start of its block.

    The waveform is the piecewise-linear one of ``kspace_at_times``. Its
    integral is exact (cubic within each interval).
    """
    n = len(g_mT_per_m)
    t_points = np.concatenate([[0.0], (np.arange(n) + 0.5) * raster_s, [n * raster_s]])
    g_points = np.concatenate([np.zeros((1, 3)), g_mT_per_m, np.zeros((1, 3))]) * 1e-3
    dt = np.diff(t_points)
    a, b = g_points[:-1], g_points[1:]
    # ∫ over an interval of a line from a to b: dt (2 a aᵀ + a bᵀ + b aᵀ + 2 b bᵀ) / 6.
    ab = np.einsum("ti,tj->tij", a, b)
    pieces = (
        2 * np.einsum("ti,tj->tij", a, a)
        + ab
        + ab.transpose(0, 2, 1)
        + 2 * np.einsum("ti,tj->tij", b, b)
    ) * (dt / 6)[:, None, None]
    integral = np.concatenate([np.zeros((1, 3, 3)), np.cumsum(pieces, axis=0)])
    if np.any(t_s < 0) or np.any(t_s > t_points[-1]):
        raise ValueError("a time is outside the block")
    j = np.clip(np.searchsorted(t_points, t_s, side="right") - 1, 0, len(dt) - 1)
    tau = (t_s - t_points[j])[:, None, None]
    g0 = g_points[j]
    slope = (b - a)[j] / dt[j][:, None]
    g0_slope = np.einsum("ti,tj->tij", g0, slope)
    gg: FloatArray = (
        integral[j]
        + np.einsum("ti,tj->tij", g0, g0) * tau
        + (g0_slope + g0_slope.transpose(0, 2, 1)) * tau**2 / 2
        + np.einsum("ti,tj->tij", slope, slope) * tau**3 / 3
    )
    return gg


def concomitant_phase_rad(config: Config, gg_T2s_per_m2: FloatArray, r_m: FloatArray) -> FloatArray:
    """(..., positions) phase (rad) of the lowest-order concomitant field.

    ``gg_T2s_per_m2`` (..., 3, 3) is ``∫ g gᵀ dt`` on the physical axes (z along
    B0), and ``r_m`` (positions, 3) are physical positions from the isocenter.
    The field is (Bernstein et al., MRM 39:300, 1998)
    ``B_c = [Gz² (x² + y²) / 4 + (Gx² + Gy²) z² - Gx Gz x z - Gy Gz y z] / (2 B0)``.
    The phase is ``2π γ ∫ B_c dt``, with the same sign as the encoding phase
    ``2π k·r``. It does not depend on the gradient polarity.
    """
    q = gg_T2s_per_m2
    x, y, z = r_m[:, 0], r_m[:, 1], r_m[:, 2]
    field_integral = (
        q[..., 2, 2, None] * (x**2 + y**2) / 4
        + (q[..., 0, 0, None] + q[..., 1, 1, None]) * z**2
        - q[..., 0, 2, None] * x * z
        - q[..., 1, 2, None] * y * z
    ) / (2 * config.system.B0_T)
    phase: FloatArray = 2 * np.pi * GAMMA_HZ_PER_T * field_integral
    return phase


@dataclass(frozen=True)
class Trajectory:
    """Compact k-space trajectory of the acquired TRs."""

    k_shape_per_m: FloatArray
    """(shapes, samples, 3): k of each shape at the ADC samples, logical axes."""
    gg_shape_T2s_per_m2: FloatArray
    """(shapes, samples, 3, 3): ∫ g gᵀ dt of each shape at the ADC samples, logical axes."""
    adc_time_in_tr_s: FloatArray
    """(samples,): ADC sample times from the start of the TR block."""
    time_since_excitation_s: FloatArray
    """(samples,): ADC sample times from the RF center of the same TR."""
    tr_index: NDArray[np.int64]
    """(acquired TRs,): index of each acquired TR in the full schedule."""
    tr_start_s: FloatArray
    """(acquired TRs,): start time of each acquired TR in the sequence."""
    shape_index: NDArray[np.int64]
    """(acquired TRs,): shape played in each acquired TR."""
    rotation_matrix: FloatArray
    """(acquired TRs, 3, 3): physical k = R @ logical k."""
    rf_phase_rad: FloatArray
    """(acquired TRs,): RF and ADC phase of each acquired TR."""
    gradient_delay_s: FloatArray = field(default_factory=lambda: np.zeros(3))
    """(3,): delay of each physical gradient axis (x, y, z); zeros for the nominal trajectory."""
    k_shape_delayed_per_m: FloatArray | None = None
    """(3, shapes, samples, 3): k of each shape at the ADC times minus the delay
    of physical axis 0, 1, 2; None without delays."""

    @property
    def has_delays(self) -> bool:
        return self.k_shape_delayed_per_m is not None

    def k_per_m(self, trs: slice | NDArray[np.int64] | None = None) -> FloatArray:
        """(TRs, samples, 3) physical k of the acquired TRs (default: all).

        All TRs of the default design need about 1.7 GB in float64; pass a
        slice or index array for fewer.
        """
        selection = slice(None) if trs is None else trs
        rotation = self.rotation_matrix[selection]
        if self.k_shape_delayed_per_m is not None:
            # Physical component i is R[i] @ k_shape(t - δ_i).
            k_axes = self.k_shape_delayed_per_m[:, self.shape_index[selection]]
            delayed: FloatArray = np.einsum("tij,itsj->tsi", rotation, k_axes)
            return delayed
        k_logical = self.k_shape_per_m[self.shape_index[selection]]
        k: FloatArray = np.einsum("tij,tsj->tsi", rotation, k_logical)
        return k

    def gg_T2s_per_m2(self, trs: slice | NDArray[np.int64] | None = None) -> FloatArray:
        """(TRs, samples, 3, 3) physical ``R @ gg @ Rᵀ`` of the acquired TRs (default: all).

        The file stores one tensor per shape. This returns one per TR and
        sample, because each TR has its own rotation. All 5000 TRs of the
        default design are about 5 GB in float64; pass ``trs`` (a slice or
        index array) to compute a subset, e.g. a batch of a reconstruction.
        """
        selection = slice(None) if trs is None else trs
        gg_logical = self.gg_shape_T2s_per_m2[self.shape_index[selection]]
        r = self.rotation_matrix[selection]
        gg: FloatArray = np.einsum("tik,tskl,tjl->tsij", r, gg_logical, r)
        return gg


def make_trajectory(
    config: Config,
    schedule: Schedule,
    shapes: list[Shape],
    gradient_delay_s: Sequence[float] | None = None,
) -> Trajectory:
    """Trajectory of a design, optionally with a delay on each physical gradient axis."""
    r = config.readout
    raster = config.system.grad_raster_time_s
    adc_time = np.asarray(
        config.adc_start_s + (np.arange(r.adc_num_samples) + 0.5) * r.adc_dwell_s,
        dtype=np.float64,
    )
    k_shape = np.stack([kspace_at_times(s.g_mT_per_m, raster, adc_time) for s in shapes])
    delays = np.zeros(3) if gradient_delay_s is None else np.asarray(gradient_delay_s, np.float64)
    if delays.shape != (3,):
        raise ValueError("gradient_delay_s needs one delay for each of the 3 physical axes")
    k_delayed = None
    if np.any(delays != 0):
        k_delayed = np.stack(
            [
                np.stack([kspace_at_times(s.g_mT_per_m, raster, adc_time - d) for s in shapes])
                for d in delays
            ]
        )
    gg_shape = np.stack(
        [gradient_outer_integral_at_times(s.g_mT_per_m, raster, adc_time) for s in shapes]
    )
    acquired = np.flatnonzero(schedule.acquired)
    return Trajectory(
        k_shape_per_m=k_shape,
        gg_shape_T2s_per_m2=gg_shape,
        adc_time_in_tr_s=adc_time,
        time_since_excitation_s=adc_time - config.rf_center_s,
        tr_index=acquired.astype(np.int64),
        tr_start_s=np.asarray(acquired * r.tr_s, dtype=np.float64),
        shape_index=schedule.shape_index[acquired],
        rotation_matrix=schedule.rotation_matrix[acquired],
        rf_phase_rad=schedule.rf_phase_rad[acquired],
        gradient_delay_s=delays,
        k_shape_delayed_per_m=k_delayed,
    )


def save_trajectory(path: Path, config: Config, trajectory: Trajectory) -> None:
    """Write a trajectory and its config to an ``.npz`` file.

    The arrays are the ``Trajectory`` fields, plus ``config_toml`` and
    ``config_digest``. Without gradient delays, the physical k of sample ``s``
    of acquired TR ``a`` is ``rotation_matrix[a] @ k_shape_per_m[shape_index[a], s]``;
    with them (``gradient_delay_s`` not zero), its component ``i`` is
    ``rotation_matrix[a][i] @ k_shape_delayed_per_m[i, shape_index[a], s]``.
    Its physical ``∫ g gᵀ dt`` is ``R @ gg_shape_T2s_per_m2[shape_index[a], s] @ Rᵀ``
    with ``R = rotation_matrix[a]``.
    """
    delayed: dict[str, Any] = {}
    if trajectory.k_shape_delayed_per_m is not None:
        delayed["k_shape_delayed_per_m"] = trajectory.k_shape_delayed_per_m
    np.savez(
        path,
        **delayed,
        k_shape_per_m=trajectory.k_shape_per_m,
        gg_shape_T2s_per_m2=trajectory.gg_shape_T2s_per_m2,
        adc_time_in_tr_s=trajectory.adc_time_in_tr_s,
        time_since_excitation_s=trajectory.time_since_excitation_s,
        tr_index=trajectory.tr_index,
        tr_start_s=trajectory.tr_start_s,
        shape_index=trajectory.shape_index,
        rotation_matrix=trajectory.rotation_matrix,
        rf_phase_rad=trajectory.rf_phase_rad,
        gradient_delay_s=trajectory.gradient_delay_s,
        config_toml=config.to_toml(),
        config_digest=config.digest(),
    )


def load_trajectory(path: Path) -> tuple[Config, Trajectory]:
    """Read a file written by ``save_trajectory`` (one without delays if it predates them)."""
    with np.load(path) as data:
        config = Config.from_toml(str(data["config_toml"]))
        if str(data["config_digest"]) != config.digest():
            raise ValueError(f"{path}: config digest does not match its config")
        trajectory = Trajectory(
            k_shape_per_m=data["k_shape_per_m"],
            gg_shape_T2s_per_m2=data["gg_shape_T2s_per_m2"],
            adc_time_in_tr_s=data["adc_time_in_tr_s"],
            time_since_excitation_s=data["time_since_excitation_s"],
            tr_index=data["tr_index"],
            tr_start_s=data["tr_start_s"],
            shape_index=data["shape_index"],
            rotation_matrix=data["rotation_matrix"],
            rf_phase_rad=data["rf_phase_rad"],
            gradient_delay_s=(
                data["gradient_delay_s"] if "gradient_delay_s" in data else np.zeros(3)
            ),
            k_shape_delayed_per_m=(
                data["k_shape_delayed_per_m"] if "k_shape_delayed_per_m" in data else None
            ),
        )
    return config, trajectory


def seq_kspace(seq: pp.Sequence, tr_indices: NDArray[np.int64], num_samples: int) -> FloatArray:
    """(TRs, samples, 3) k (1/m) that pypulseq computes for the given TRs.

    The TRs (indices in the full schedule; one block each) are copied into a
    new sequence, so pypulseq evaluates only those blocks.
    """
    sub = pp.Sequence(system=seq.system)
    for tr in tr_indices:
        sub.add_block(seq.get_block(int(tr) + 1))
    k_adc = sub.calculate_kspace()[0]
    k: FloatArray = k_adc.T.reshape(len(tr_indices), num_samples, 3)
    return k


def verify_trajectory(seq: pp.Sequence, trajectory: Trajectory, num_trs: int) -> float:
    """Largest difference (1/m) between the trajectory and pypulseq's k-space,
    over the first ``num_trs`` acquired TRs of the ``.seq`` file."""
    n = min(num_trs, len(trajectory.tr_index))
    if n == 0:
        return 0.0
    expected = seq_kspace(seq, trajectory.tr_index[:n], len(trajectory.adc_time_in_tr_s))
    return float(np.abs(trajectory.k_per_m(slice(0, n)) - expected).max())
