"""k-space trajectory at the ADC samples, for reconstruction.

Every acquired TR samples the same ADC times within its block, and plays one
shape rotated by its rotation matrix. The trajectory is therefore stored
compactly: k of each shape at the ADC sample times (logical axes), plus the
shape index and rotation of each acquired TR. The k-space position of sample
``s`` of acquired TR ``a`` is ``rotation[a] @ k_shape[shape_index[a], s]``.

k is computed as the exact integral of the piecewise-linear gradient that
pypulseq plays, with k = 0 at the RF center (the gradients are off during the
RF pulse).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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


@dataclass(frozen=True)
class Trajectory:
    """Compact k-space trajectory of the acquired TRs."""

    k_shape_per_m: FloatArray
    """(shapes, samples, 3): k of each shape at the ADC samples, logical axes."""
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

    def k_per_m(self, trs: slice | NDArray[np.int64] | None = None) -> FloatArray:
        """(TRs, samples, 3) physical k of the acquired TRs (default: all).

        All TRs of the default design need about 1.7 GB in float64; pass a
        slice or index array for fewer.
        """
        selection = slice(None) if trs is None else trs
        k_logical = self.k_shape_per_m[self.shape_index[selection]]
        k: FloatArray = np.einsum("tij,tsj->tsi", self.rotation_matrix[selection], k_logical)
        return k


def make_trajectory(config: Config, schedule: Schedule, shapes: list[Shape]) -> Trajectory:
    """Trajectory of a design."""
    r = config.readout
    adc_time = np.asarray(
        config.adc_start_s + (np.arange(r.adc_num_samples) + 0.5) * r.adc_dwell_s,
        dtype=np.float64,
    )
    k_shape = np.stack(
        [kspace_at_times(s.g_mT_per_m, config.system.grad_raster_time_s, adc_time) for s in shapes]
    )
    acquired = np.flatnonzero(schedule.acquired)
    return Trajectory(
        k_shape_per_m=k_shape,
        adc_time_in_tr_s=adc_time,
        time_since_excitation_s=adc_time - config.rf_center_s,
        tr_index=acquired.astype(np.int64),
        tr_start_s=np.asarray(acquired * r.tr_s, dtype=np.float64),
        shape_index=schedule.shape_index[acquired],
        rotation_matrix=schedule.rotation_matrix[acquired],
        rf_phase_rad=schedule.rf_phase_rad[acquired],
    )


def save_trajectory(path: Path, config: Config, trajectory: Trajectory) -> None:
    """Write a trajectory and its config to an ``.npz`` file.

    The arrays are the ``Trajectory`` fields, plus ``config_toml`` and
    ``config_digest``. The physical k of sample ``s`` of acquired TR ``a`` is
    ``rotation_matrix[a] @ k_shape_per_m[shape_index[a], s]``.
    """
    np.savez(
        path,
        k_shape_per_m=trajectory.k_shape_per_m,
        adc_time_in_tr_s=trajectory.adc_time_in_tr_s,
        time_since_excitation_s=trajectory.time_since_excitation_s,
        tr_index=trajectory.tr_index,
        tr_start_s=trajectory.tr_start_s,
        shape_index=trajectory.shape_index,
        rotation_matrix=trajectory.rotation_matrix,
        rf_phase_rad=trajectory.rf_phase_rad,
        config_toml=config.to_toml(),
        config_digest=config.digest(),
    )


def load_trajectory(path: Path) -> tuple[Config, Trajectory]:
    """Read a file written by ``save_trajectory``."""
    with np.load(path) as data:
        config = Config.from_toml(str(data["config_toml"]))
        if str(data["config_digest"]) != config.digest():
            raise ValueError(f"{path}: config digest does not match its config")
        trajectory = Trajectory(
            k_shape_per_m=data["k_shape_per_m"],
            adc_time_in_tr_s=data["adc_time_in_tr_s"],
            time_since_excitation_s=data["time_since_excitation_s"],
            tr_index=data["tr_index"],
            tr_start_s=data["tr_start_s"],
            shape_index=data["shape_index"],
            rotation_matrix=data["rotation_matrix"],
            rf_phase_rad=data["rf_phase_rad"],
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
