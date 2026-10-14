"""PNS prediction (SAFE model) over every TR of a design.

pypulseq's ``Sequence.calculate_pns`` evaluates the SAFE model on waveforms
rebuilt from the ``.seq`` file, and implements each RC low-pass filter as a
direct convolution with a truncated exponential (thousands of taps at the
gradient raster), which is too slow for a whole scan. Here the same model is
evaluated on the physical waveforms of the design (each TR's shape rotated by
its rotation matrix, which is what the scanner plays) with each low-pass as a
one-pole recursion, which is the same filter in linear time. The scan is
processed in chunks of TRs; each chunk starts with enough of the TRs before
it (25 of the longest time constants) that the filters carry the history
across chunk boundaries.

The model (Hebrank & Gebhardt, ISMRM 2000, #2007; the pypulseq / SAFE
implementation by Witzel and Szczepankiewicz), per axis with slew ``s(t)``:

    stim = (a1 |lp(s, tau1)| + a2 lp(|s|, tau2) + a3 |lp(s, tau3)|)
           / stim_limit * g_scale

and the PNS level is the vector norm over the axes (1 = 100 % of the
stimulation threshold).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
from numpy.typing import NDArray
from scipy.signal import lfilter

from relaxed_rosettes.config import Config
from relaxed_rosettes.optimize import Shape
from relaxed_rosettes.sequence import Schedule

FloatArray = NDArray[np.float64]

WARMUP_TIME_CONSTANTS = 25


def rc_lowpass(x: FloatArray, tau_ms: float, dt_s: float) -> FloatArray:
    """The SAFE RC low-pass: ``y[n] = alpha x[n] + (1 - alpha) y[n - 1]``."""
    dt_ms = dt_s * 1000
    alpha = dt_ms / (tau_ms + dt_ms)
    y: FloatArray = lfilter([alpha], [1.0, -(1.0 - alpha)], x, axis=0)
    return y


def axis_stimulation(
    slew_T_per_m_per_s: FloatArray, dt_s: float, axis: SimpleNamespace
) -> FloatArray:
    """SAFE stimulation of one axis, as a fraction of its threshold."""
    s = slew_T_per_m_per_s
    stim = (
        axis.a1 * np.abs(rc_lowpass(s, axis.tau1, dt_s))
        + axis.a2 * rc_lowpass(np.abs(s), axis.tau2, dt_s)
        + axis.a3 * np.abs(rc_lowpass(s, axis.tau3, dt_s))
    )
    result: FloatArray = stim / axis.stim_limit * axis.g_scale
    return result


@dataclass(frozen=True)
class PnsResult:
    """Peak PNS of every TR (fraction of the stimulation threshold)."""

    peak_per_tr: FloatArray
    """(TRs,) peak of the PNS norm in each TR (dummies included)."""
    peak_per_axis: FloatArray
    """(3,) peak of each axis over the scan."""

    @property
    def peak(self) -> float:
        return float(self.peak_per_tr.max())

    @property
    def worst_tr(self) -> int:
        return int(np.argmax(self.peak_per_tr))


def physical_waveform(shapes: list[Shape], schedule: Schedule, trs: range) -> FloatArray:
    """(samples, 3) physical gradient (T/m) of consecutive TRs, on the raster."""
    parts = [
        shapes[schedule.shape_index[tr]].g_mT_per_m @ schedule.rotation_matrix[tr].T for tr in trs
    ]
    g: FloatArray = np.concatenate(parts) * 1e-3
    return g


def design_pns(
    config: Config,
    schedule: Schedule,
    shapes: list[Shape],
    hardware: SimpleNamespace,
    chunk_trs: int = 100,
) -> PnsResult:
    """SAFE PNS over every TR of a design."""
    dt = config.system.grad_raster_time_s
    n = config.raster_points_per_tr
    # Enough preceding TRs for the filters to forget the start (e^-25).
    longest_tau_s = 1e-3 * max(
        getattr(getattr(hardware, a), t) for a in "xyz" for t in ("tau1", "tau2", "tau3")
    )
    warm_trs = max(1, math.ceil(WARMUP_TIME_CONSTANTS * longest_tau_s / config.readout.tr_s))
    peak_per_tr = np.zeros(schedule.num_trs)
    peak_per_axis = np.zeros(3)
    for start in range(0, schedule.num_trs, chunk_trs):
        stop = min(start + chunk_trs, schedule.num_trs)
        warm = min(start, warm_trs)
        g = physical_waveform(shapes, schedule, range(start - warm, stop))
        # Start and end at 0 (every TR does), so the first difference is the
        # ramp from 0.
        g = np.vstack([np.zeros((1, 3)), g])
        slew = np.diff(g, axis=0) / dt
        stim = np.stack(
            [axis_stimulation(slew[:, i], dt, getattr(hardware, a)) for i, a in enumerate("xyz")],
            axis=1,
        )[warm * n :]
        norm = np.linalg.norm(stim, axis=1)
        peak_per_tr[start:stop] = norm.reshape(stop - start, n).max(axis=1)
        peak_per_axis = np.maximum(peak_per_axis, stim.max(axis=0))
    return PnsResult(peak_per_tr=peak_per_tr, peak_per_axis=peak_per_axis)
