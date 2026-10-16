"""Gradient delays estimated from the echo times of the k = 0 crossings.

Every petal of a shape passes through (or near) k = 0, where the signal has a
gradient echo. A delay ``δ_i`` on physical gradient axis ``i`` plays that axis
late, ``k_i(t) = k_nominal_i(t - δ_i)`` (``relaxed_rosettes.trajectory``).

Echo model. Near its peak ``k_0``, the magnitude of the object's spectrum is
``|S(k)| ≈ S_0 (1 - (k - k_0)ᵀ Q (k - k_0) / 2)``, with ``Q`` proportional to
the spatial covariance of the object (isotropic for a sphere). Near a crossing
at time ``t_c`` with physical gradient ``g``, a straight delayed path is
``k_i(t) = γ g_i (t - t_c - δ_i)``, and the echo peaks at

    Δt = t - t_c = gᵀ Q D g / gᵀ Q g  +  gᵀ Q k_0 / (γ gᵀ Q g),   D = diag(δ).

With different delays on the axes the path misses k = 0 sideways, and where
along the path the echo peaks then depends on ``Q``: an isotropic model biases
the delays of an elongated object by about 10% of their differences.

``k_0`` is 0 for a real, non-negative object. The fit lets it be a static
offset (object phase) plus ``-γ G τ`` for a background gradient ``G`` (a
linear off-resonance ``γ G·r``, which adds the phase ``2π γ (G·r) τ`` at time
``τ`` since excitation). The delay terms are even in the direction of ``g``
and the offset terms odd, so the rotations of the TRs separate them. Equal
delays on all axes are a shift of the ADC relative to the gradients.

Measurements. For each crossing of each TR, the coil-combined magnitude (root
sum of squares of the prewhitened coils) is fitted with a parabola through its
largest sample and the two neighbours: the vertex is the echo time, to a
fraction of a dwell, and the curvature over the peak value is
``γ² dwell² gᵀ Q g``. That is linear in ``Q``, so ``Q`` (up to scale) is a
least-squares fit over all crossings, with no image.

Fit. Gauss-Newton: each pass takes the trajectory delayed by the current
estimate, finds each crossing as the minimum of ``kᵀ Q k`` along it (the
vertex of a parabola through the samples around the minimum; exact for a
straight path), fits the residual delays and the offsets by linear least
squares, and adds the delays, until the update is below 1% of its standard
error. At the true delays the delayed path's crossing is the echo, so the
fixed point does not depend on how good the linear model is. The delayed k
is interpolated linearly between ADC samples of the nominal trajectory (an
error of about ``γ slew dwell² / 8``, far below a cell). Each least-squares
fit is done once, then again without the crossings whose residual is more
than ``outlier_mads`` scaled median absolute deviations from zero.

The crossings are the local minima of ``|k|`` of each shape that are within
``max_crossing_distance_cells`` of k = 0.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from relaxed_rosettes.config import GAMMA_HZ_PER_T, Config
from relaxed_rosettes.recon.preprocess import noise_whitening
from relaxed_rosettes.recon.rawdata import RawData
from relaxed_rosettes.trajectory import Trajectory

FloatArray = NDArray[np.float64]

NUM_PARAMETERS = 9
"""Delays (3), static k offset (3), background gradient (3)."""


@dataclass(frozen=True)
class Crossings:
    """The k = 0 crossings of each shape, from the nominal trajectory."""

    shape: NDArray[np.int64]
    """(crossings,): shape of each crossing."""
    sample: NDArray[np.int64]
    """(crossings,): ADC sample closest to k = 0."""
    distance_cells: FloatArray
    """(crossings,): closest approach to k = 0, in cells (1/FOV)."""


def find_crossings(
    config: Config, trajectory: Trajectory, max_crossing_distance_cells: float, margin: int
) -> Crossings:
    """Crossings of every shape at least ``margin`` samples from the ends of the ADC."""
    cell = 1.0 / config.geometry.fov_m
    shapes, samples, distances = [], [], []
    for s, k in enumerate(trajectory.k_shape_per_m):
        r2 = np.sum(k**2, axis=1)
        inner = np.arange(max(margin, 1), len(r2) - max(margin, 1))
        minima = inner[(r2[inner] <= r2[inner - 1]) & (r2[inner] < r2[inner + 1])]
        _, vertex, _ = _parabola(r2[minima - 1], r2[minima], r2[minima + 1])
        distance = np.sqrt(np.maximum(vertex, 0.0)) / cell
        keep = distance <= max_crossing_distance_cells
        shapes.append(np.full(int(keep.sum()), s))
        samples.append(minima[keep])
        distances.append(distance[keep])
    return Crossings(
        shape=np.concatenate(shapes).astype(np.int64),
        sample=np.concatenate(samples).astype(np.int64),
        distance_cells=np.concatenate(distances),
    )


def _parabola(
    a: FloatArray, b: FloatArray, c: FloatArray
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """(vertex offset from the middle sample, vertex value, curvature) of the
    parabolas through (-1, a), (0, b), (1, c). The offset is clipped to [-1, 1]
    and is 0 where the points are collinear."""
    curvature = a - 2 * b + c
    safe = np.where(curvature != 0, curvature, 1.0)
    offset = np.clip(np.where(curvature != 0, 0.5 * (a - c) / safe, 0.0), -1.0, 1.0)
    value = b - 0.25 * (a - c) * offset
    return offset, value, curvature


@dataclass(frozen=True)
class Peaks:
    """Echo peaks of the coil-combined magnitude around each crossing."""

    position: FloatArray
    """(crossings,): sub-sample position of the peak (NaN: no peak in the window)."""
    width: FloatArray
    """(crossings,): -curvature / value of the peak parabola (1/samples²; NaN: no peak)."""


def find_peaks(magnitude: FloatArray, centre: NDArray[np.int64], half_width: int) -> Peaks:
    """The peak of ``magnitude`` (samples,) within ``half_width`` samples of each centre.

    A window whose largest sample is at its edge has no peak.
    """
    offsets = np.arange(-half_width, half_width + 1)
    window = magnitude[centre[:, None] + offsets[None]]
    best = np.argmax(window, axis=1)
    position = np.full(len(centre), np.nan)
    width = np.full(len(centre), np.nan)
    rows = np.flatnonzero((best > 0) & (best < len(offsets) - 1))
    at = best[rows]
    shift, value, curvature = _parabola(
        window[rows, at - 1], window[rows, at], window[rows, at + 1]
    )
    position[rows] = centre[rows] + offsets[at] + shift
    width[rows] = -curvature / value
    return Peaks(position, width)


def crossing_geometry(
    config: Config,
    trajectory: Trajectory,
    shape: NDArray[np.int64],
    sample: NDArray[np.int64],
    tr: NDArray[np.int64],
    delays_s: FloatArray,
    metric: FloatArray,
    half_width: int,
) -> tuple[FloatArray, FloatArray]:
    """(sample position of each crossing, (crossings, 3) physical gradient) with delays.

    Physical component ``i`` of k is ``R[i] @ k_shape(t - δ_i)``, with
    ``k_shape`` interpolated linearly between ADC samples. The crossing is
    the minimum of ``kᵀ metric k`` within ``half_width`` samples of ``sample``.
    """
    dwell = config.readout.adc_dwell_s
    offsets = np.arange(-half_width, half_width + 1)
    where = sample[:, None] + offsets[None]
    rotation = trajectory.rotation_matrix[tr]
    k_shape = trajectory.k_shape_per_m
    k = np.zeros((len(sample), len(offsets), 3))
    for i, delay in enumerate(delays_s):
        position = where - delay / dwell
        low = np.floor(position).astype(np.int64)
        frac = (position - low)[..., None]
        k_logical = (1 - frac) * k_shape[shape[:, None], low] + frac * k_shape[
            shape[:, None], low + 1
        ]
        k[..., i] = np.einsum("nj,nwj->nw", rotation[:, i, :], k_logical)
    r2 = np.einsum("nwi,ij,nwj->nw", k, metric, k)
    best = 1 + np.argmin(r2[:, 1:-1], axis=1)
    rows = np.arange(len(sample))
    offset, _, _ = _parabola(r2[rows, best - 1], r2[rows, best], r2[rows, best + 1])
    position = sample + offsets[best] + offset
    g = (k[rows, best + 1] - k[rows, best - 1]) / (2 * dwell * GAMMA_HZ_PER_T)
    return position, g


def fit_metric(g_T_per_m: FloatArray, width: FloatArray, used: NDArray[np.bool_]) -> FloatArray:
    """(3, 3) ``Q`` from the peak widths ``∝ gᵀ Q g``, scaled to trace 3.

    The identity if the fit is not positive definite.
    """
    g = g_T_per_m[used]
    x, y, z = g[:, 0], g[:, 1], g[:, 2]
    rows = np.stack([x * x, y * y, z * z, 2 * x * y, 2 * x * z, 2 * y * z], axis=1)
    q = np.linalg.lstsq(rows, width[used], rcond=None)[0]
    metric = np.array([[q[0], q[3], q[4]], [q[3], q[1], q[5]], [q[4], q[5], q[2]]])
    if np.any(np.linalg.eigvalsh(metric) <= 0):
        return np.eye(3)
    scaled: FloatArray = 3 * metric / np.trace(metric)
    return scaled


def design_matrix(g_T_per_m: FloatArray, tau_s: FloatArray, metric: FloatArray) -> FloatArray:
    """(crossings, 9) model of the echo time shift: delays, k offset, background gradient."""
    qg = g_T_per_m @ metric
    gqg = np.sum(g_T_per_m * qg, axis=1, keepdims=True)
    return np.hstack(
        [g_T_per_m * qg / gqg, qg / (GAMMA_HZ_PER_T * gqg), -tau_s[:, None] * qg / gqg]
    )


@dataclass(frozen=True)
class _Fit:
    params: FloatArray
    se: FloatArray
    residual: FloatArray
    used: NDArray[np.bool_]
    rms: float


def _fit(a_matrix: FloatArray, shift: FloatArray, outlier_mads: float) -> _Fit:
    used = np.isfinite(shift)
    if used.sum() <= NUM_PARAMETERS:
        raise ValueError(f"only {used.sum()} crossings have an echo peak")
    params = np.linalg.lstsq(a_matrix[used], shift[used], rcond=None)[0]
    residual = shift - a_matrix @ params
    mad = 1.4826 * np.median(np.abs(residual[used]))
    used &= np.abs(np.nan_to_num(residual, nan=np.inf)) <= outlier_mads * mad
    params = np.linalg.lstsq(a_matrix[used], shift[used], rcond=None)[0]
    residual = shift - a_matrix @ params
    rms = float(np.sqrt(np.sum(residual[used] ** 2) / (used.sum() - NUM_PARAMETERS)))
    covariance = rms**2 * np.linalg.inv(a_matrix[used].T @ a_matrix[used])
    return _Fit(params, np.sqrt(np.diag(covariance)), residual, used, rms)


@dataclass(frozen=True)
class DelayEstimate:
    """Fitted gradient delays, echo offsets, and the measurements they come from."""

    gradient_delay_s: FloatArray
    """(3,): delay of physical axes x, y, z."""
    gradient_delay_se_s: FloatArray
    k_offset_per_m: FloatArray
    """(3,): static k-space position of the echo peak."""
    k_offset_se_per_m: FloatArray
    background_gradient_T_per_m: FloatArray
    """(3,): linear off-resonance, as a gradient."""
    background_gradient_se_T_per_m: FloatArray
    metric: FloatArray
    """(3, 3): ``Q`` of the echo model, trace 3 (the identity for a sphere)."""
    residual_rms_s: float
    iterations: int
    max_shift_cells: float
    """Bound on the k error the delays cause at any sample of any TR,
    ``γ max|δ_i| max|g|``, in cells (1/FOV)."""
    perpendicular_miss_cells: FloatArray
    """(crossings,): miss distance of the nominal trajectory perpendicular to g,
    which a time shift cannot absorb."""
    tr: NDArray[np.int64]
    """(crossings,): acquired TR of each measured crossing."""
    crossing_shape: NDArray[np.int64]
    crossing_time_s: FloatArray
    """(crossings,): crossing time of the delayed trajectory, from the start of the TR block."""
    g_T_per_m: FloatArray
    """(crossings, 3): physical gradient at each crossing."""
    echo_time_s: FloatArray
    """(crossings,): measured echo peak, from the start of the TR block (NaN: no peak)."""
    residual_s: FloatArray
    used: NDArray[np.bool_]


def estimate_delays(
    raw: RawData,
    design: Config,
    trajectory: Trajectory,
    *,
    max_shift_s: float = 20e-6,
    max_crossing_distance_cells: float = 1.0,
    outlier_mads: float = 5.0,
    max_iterations: int = 20,
    batch_trs: int = 64,
) -> DelayEstimate:
    """Fit gradient delays to the echo times of every crossing of every TR of ``raw``."""
    if trajectory.has_delays:
        raise ValueError("the trajectory already has gradient delays; use the nominal one")
    raw.check_trajectory(design, trajectory)
    dwell = design.readout.adc_dwell_s
    half_width = int(np.ceil(max_shift_s / dwell)) + 1
    # Room for the peak search, and for the crossing search on the delayed k.
    margin = 2 * half_width + 2
    crossings = find_crossings(design, trajectory, max_crossing_distance_cells, margin)
    if len(crossings.shape) == 0:
        raise ValueError("the trajectory has no k = 0 crossings")
    whitening, _ = noise_whitening(raw, dwell)

    trs, rows, positions, widths = [], [], [], []
    for start in range(0, raw.num_trs, batch_trs):
        batch = np.arange(start, min(start + batch_trs, raw.num_trs))
        data = np.einsum("dc,tcs->tds", whitening, raw.read(batch))
        magnitude = np.sqrt(np.sum(np.abs(data) ** 2, axis=1))
        for j, a in enumerate(batch):
            mine = np.flatnonzero(crossings.shape == trajectory.shape_index[a])
            peaks = find_peaks(magnitude[j], crossings.sample[mine], half_width)
            trs.append(np.full(len(mine), a))
            rows.append(mine)
            positions.append(peaks.position)
            widths.append(peaks.width)
    tr = np.concatenate(trs).astype(np.int64)
    row = np.concatenate(rows)
    peak = np.concatenate(positions)
    width = np.concatenate(widths) / (GAMMA_HZ_PER_T * dwell) ** 2
    shape, sample = crossings.shape[row], crossings.sample[row]

    delays = np.zeros(3)
    metric = np.eye(3)
    iterations = 0
    while True:
        iterations += 1
        position, g = crossing_geometry(
            design, trajectory, shape, sample, tr, delays, metric, half_width
        )
        tau = trajectory.time_since_excitation_s[0] + position * dwell
        fit = _fit(design_matrix(g, tau, metric), (peak - position) * dwell, outlier_mads)
        delays = delays + fit.params[:3]
        new_metric = fit_metric(g, width, fit.used & np.isfinite(width))
        converged = np.all(np.abs(fit.params[:3]) <= 0.01 * fit.se[:3]) and np.allclose(
            new_metric, metric, atol=1e-3
        )
        metric = new_metric
        if converged or iterations == max_iterations:
            break

    cell = 1.0 / design.geometry.fov_m
    g_all = np.gradient(trajectory.k_shape_per_m, dwell, axis=1) / GAMMA_HZ_PER_T
    max_g = float(np.max(np.linalg.norm(g_all, axis=2)))
    error = GAMMA_HZ_PER_T * delays[None] * g
    unit = g / np.linalg.norm(g, axis=1, keepdims=True)
    along = np.sum(error * unit, axis=1, keepdims=True) * unit
    t0 = trajectory.adc_time_in_tr_s[0]
    return DelayEstimate(
        gradient_delay_s=delays,
        gradient_delay_se_s=fit.se[:3],
        k_offset_per_m=fit.params[3:6],
        k_offset_se_per_m=fit.se[3:6],
        background_gradient_T_per_m=fit.params[6:],
        background_gradient_se_T_per_m=fit.se[6:],
        metric=metric,
        residual_rms_s=fit.rms,
        iterations=iterations,
        max_shift_cells=GAMMA_HZ_PER_T * float(np.max(np.abs(delays))) * max_g / cell,
        perpendicular_miss_cells=np.linalg.norm(error - along, axis=1) / cell,
        tr=tr,
        crossing_shape=shape,
        crossing_time_s=t0 + position * dwell,
        g_T_per_m=g,
        echo_time_s=t0 + peak * dwell,
        residual_s=fit.residual,
        used=fit.used,
    )


def save_delays(path: Path, estimate: DelayEstimate, raw: RawData) -> None:
    """Write a delay estimate to an ``.npz`` file.

    Arrays: the ``DelayEstimate`` fields, plus ``design_config_digest`` and
    ``seq_signature`` of the raw data.
    """
    np.savez(
        path,
        **{name: getattr(estimate, name) for name in estimate.__dataclass_fields__},
        design_config_digest=raw.config.digest(),
        seq_signature=raw.seq_signature,
    )


def format_delays(estimate: DelayEstimate) -> str:
    """A summary of the estimate, with the delays ready for ``--gradient-delays-s``."""
    e = estimate

    def axes(values: FloatArray, se: FloatArray, scale: float, unit: str) -> str:
        parts = (
            f"{name} {v * scale:+.3f} ± {s * scale:.3f}"
            for name, v, s in zip("xyz", values, se, strict=True)
        )
        return ", ".join(parts) + f" {unit}"

    delays = ",".join(f"{d:.4g}" for d in e.gradient_delay_s)
    eigenvalues = ", ".join(f"{v:.2f}" for v in np.linalg.eigvalsh(e.metric))
    return "\n".join(
        [
            f"gradient delays: {axes(e.gradient_delay_s, e.gradient_delay_se_s, 1e6, 'µs')}",
            f"echo k offset: {axes(e.k_offset_per_m, e.k_offset_se_per_m, 1.0, '1/m')}",
            "background gradient: "
            + axes(e.background_gradient_T_per_m, e.background_gradient_se_T_per_m, 1e3, "mT/m"),
            f"echo metric eigenvalues (trace 3): {eigenvalues}",
            f"crossings: {int(e.used.sum())} used of {len(e.used)}; "
            f"residual RMS {e.residual_rms_s * 1e6:.3f} µs; {e.iterations} iterations",
            f"k error from the delays: at most {e.max_shift_cells:.3f} cells; "
            f"perpendicular miss at the crossings: 95th percentile "
            f"{np.percentile(e.perpendicular_miss_cells, 95):.3f} cells",
            f"--gradient-delays-s {delays}",
        ]
    )
