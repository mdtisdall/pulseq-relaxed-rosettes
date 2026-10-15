"""k-space coverage of windows of scan time.

For a window of scan time ``[t0, t0 + W)``, every ADC sample in the window is
assigned to its nearest Cartesian k-space cell (spacing 1 / FOV). The
coverage at a resolution is the fraction of the cells inside the ball
``|k| <= 1 / (2 * resolution)`` that hold at least one sample.

The gap at a resolution is a percentile (default 95th) of the distance, in
cells, from points spread uniformly in that ball to the nearest sampled cell:
how far k-space must be filled in (e.g. by coil sensitivities or a low-rank
model) between samples. Unlike the coverage, it shows clumping when the
sampling is sparse.

These measure the resolution-time trade-off of the acquisition without
choosing a reconstruction: a long window covers fine resolutions; a short
window covers only coarse ones. Comparing windows that start at different
times shows how even the acquisition order is; with ``tau_bins > 1``, the
metrics are computed separately for samples in each bin of time since
excitation, and the worst bin is reported.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from relaxed_rosettes.config import Config
from relaxed_rosettes.trajectory import Trajectory

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class CoverageGrid:
    """The Cartesian cells of k-space up to k_max, and their radii."""

    size: int
    """Cells along each axis (odd; k = 0 is the center cell)."""
    cell_per_m: float
    """Cell spacing, 1 / FOV."""
    radius_bin: NDArray[np.int16]
    """(size**3,) index of the smallest resolution ball containing each cell,
    or ``len(resolutions)`` if none does."""
    cells_in_ball: NDArray[np.int64]
    """(resolutions,) number of cells in each resolution ball."""

    @classmethod
    def make(cls, config: Config, resolutions_m: FloatArray) -> CoverageGrid:
        """Grid for a config; ``resolutions_m`` must be in decreasing order."""
        if np.any(np.diff(resolutions_m) >= 0):
            raise ValueError("resolutions must be strictly decreasing")
        if resolutions_m[-1] < config.geometry.resolution_m - 1e-12:
            raise ValueError("a resolution is finer than the design resolution")
        cell = 1.0 / config.geometry.fov_m
        half = int(np.ceil(config.k_max_per_m / cell))
        size = 2 * half + 1
        axis = (np.arange(size) - half) * cell
        radius = np.sqrt(
            axis[:, None, None] ** 2 + axis[None, :, None] ** 2 + axis[None, None, :] ** 2
        ).ravel()
        k_ball = 1.0 / (2.0 * resolutions_m)  # increasing
        radius_bin = np.searchsorted(k_ball, radius * (1 - 1e-12)).astype(np.int16)
        counts = np.bincount(radius_bin, minlength=len(k_ball) + 1)[: len(k_ball)]
        return cls(
            size=size,
            cell_per_m=cell,
            radius_bin=radius_bin,
            cells_in_ball=np.cumsum(counts),
        )

    def coverage(self, k_per_m: FloatArray) -> FloatArray:
        """(resolutions,) fraction of each ball's cells hit by the samples."""
        half = self.size // 2
        index = np.rint(k_per_m / self.cell_per_m).astype(np.int64) + half
        inside = np.all((index >= 0) & (index < self.size), axis=1)
        index = index[inside]
        linear = (index[:, 0] * self.size + index[:, 1]) * self.size + index[:, 2]
        hit = np.unique(linear)
        counts = np.bincount(self.radius_bin[hit], minlength=len(self.cells_in_ball) + 1)
        result: FloatArray = np.cumsum(counts[: len(self.cells_in_ball)]) / self.cells_in_ball
        return result


def window_starts(trajectory: Trajectory, tr_s: float, window_s: float, count: int) -> FloatArray:
    """``count`` evenly spaced window starts within the acquired TRs."""
    first = float(trajectory.tr_start_s[0])
    last = float(trajectory.tr_start_s[-1]) + tr_s - window_s
    if last < first:
        raise ValueError(f"a {window_s} s window is longer than the acquisition")
    return np.linspace(first, last, count)


def window_samples(
    trajectory: Trajectory, tr_s: float, start_s: float, window_s: float
) -> tuple[FloatArray, FloatArray]:
    """k (1/m) and time since excitation of every sample in a window."""
    end = start_s + window_s
    trs = np.flatnonzero((trajectory.tr_start_s < end) & (trajectory.tr_start_s + tr_s > start_s))
    k = trajectory.k_per_m(trs)  # (TRs, samples, 3)
    t = trajectory.tr_start_s[trs, None] + trajectory.adc_time_in_tr_s[None, :]
    inside = (t >= start_s) & (t < end)
    tau = np.broadcast_to(trajectory.time_since_excitation_s, t.shape)
    return k[inside], tau[inside]


@dataclass(frozen=True)
class WindowMetrics:
    """Metrics of windows: axes (windows, starts, resolutions)."""

    windows_s: list[float]
    resolutions_m: FloatArray
    coverage: FloatArray
    """Fraction of the cells in each resolution ball that hold a sample."""
    gap_cells: FloatArray
    """Percentile distance (cells) from the ball to the nearest sample."""
    gap_percentile: float
    tau_bins: int


def ball_points(k_ball_per_m: float, count: int, rng: np.random.Generator) -> FloatArray:
    """``count`` points uniformly distributed in a ball."""
    points = rng.standard_normal((count, 3))
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    result: FloatArray = points * (k_ball_per_m * rng.random(count) ** (1 / 3))[:, None]
    return result


def window_metrics(
    config: Config,
    trajectory: Trajectory,
    windows_s: list[float],
    resolutions_m: FloatArray,
    num_starts: int = 8,
    tau_bins: int = 1,
    num_gap_points: int = 20_000,
    gap_percentile: float = 95.0,
) -> WindowMetrics:
    """Coverage and gap of windows of every length, at evenly spaced starts.

    With ``tau_bins > 1``, each metric is the worst over bins that split the
    ADC readout (time since excitation) into equal parts.
    """
    grid = CoverageGrid.make(config, resolutions_m)
    rng = np.random.default_rng(0)
    queries = [ball_points(1.0 / (2.0 * r), num_gap_points, rng) for r in resolutions_m]
    tau = trajectory.time_since_excitation_s
    edges = np.linspace(tau[0], tau[-1], tau_bins + 1)
    edges[-1] = np.inf
    tr_s = config.readout.tr_s
    shape = (len(windows_s), num_starts, len(resolutions_m))
    coverage = np.empty(shape)
    gap = np.empty(shape)
    for w, window in enumerate(windows_s):
        for s, start in enumerate(window_starts(trajectory, tr_s, window, num_starts)):
            k, sample_tau = window_samples(trajectory, tr_s, float(start), window)
            bins = np.searchsorted(edges, sample_tau, side="right") - 1
            per_bin_coverage = []
            per_bin_gap = []
            for b in range(tau_bins):
                k_bin = k[bins == b]
                if len(k_bin) == 0:
                    per_bin_coverage.append(np.zeros(len(resolutions_m)))
                    per_bin_gap.append(np.full(len(resolutions_m), np.inf))
                    continue
                per_bin_coverage.append(grid.coverage(k_bin))
                cells = np.unique(np.rint(k_bin / grid.cell_per_m).astype(np.int64), axis=0)
                tree = cKDTree(cells * grid.cell_per_m)
                per_bin_gap.append(
                    np.array(
                        [
                            np.percentile(tree.query(q)[0], gap_percentile) / grid.cell_per_m
                            for q in queries
                        ]
                    )
                )
            coverage[w, s] = np.min(per_bin_coverage, axis=0)
            gap[w, s] = np.max(per_bin_gap, axis=0)
    return WindowMetrics(
        windows_s=windows_s,
        resolutions_m=resolutions_m,
        coverage=coverage,
        gap_cells=gap,
        gap_percentile=gap_percentile,
        tau_bins=tau_bins,
    )


def format_metrics(metrics: WindowMetrics) -> str:
    """Text tables: coverage and gap, median (worst) over window starts."""
    m = metrics
    header = "window s" + "".join(f"{r * 1e3:>11.3g} mm" for r in m.resolutions_m)
    bins = f"; worst of {m.tau_bins} relaxation-time bins" if m.tau_bins > 1 else ""
    lines = [
        f"coverage: fraction of k-space cells sampled, median (minimum) over starts{bins}",
        header,
    ]
    for w, window in enumerate(m.windows_s):
        lines.append(
            f"{window:>8g}"
            + "".join(
                f"{np.median(m.coverage[w, :, r]):>8.0%} ({np.min(m.coverage[w, :, r]):>3.0%})"
                for r in range(len(m.resolutions_m))
            )
        )
    lines += [
        "",
        f"gap: {m.gap_percentile:g}th percentile distance to a sample, in cells (1/FOV), "
        f"median (maximum) over starts{bins}",
        header,
    ]
    for w, window in enumerate(m.windows_s):
        lines.append(
            f"{window:>8g}"
            + "".join(
                f"{np.median(m.gap_cells[w, :, r]):>8.1f} ({np.max(m.gap_cells[w, :, r]):>4.1f})"
                for r in range(len(m.resolutions_m))
            )
        )
    return "\n".join(lines)


@dataclass(frozen=True)
class KTauMetrics:
    """Sampling of the (|k|, time since excitation) plane over the whole scan.

    Rotations do not change |k|(t), so this plane is sampled only by the
    shapes: more shapes, not more TRs, fill it in. ``counts`` has shape
    (shells, tau bins): equal-width |k| shells over [0, k_max) and equal tau
    bins over the ADC readout, counting the samples of every acquired TR.
    """

    counts: NDArray[np.int64]
    tau_bin_s: float
    early_s: float

    @property
    def empty(self) -> NDArray[np.bool_]:
        empty: NDArray[np.bool_] = self.counts == 0
        return empty

    @property
    def empty_fraction(self) -> float:
        return float(self.empty.mean())

    @property
    def empty_fraction_early(self) -> float:
        """Empty fraction of the tau bins before ``early_s``."""
        return float(self.empty[:, : max(1, round(self.early_s / self.tau_bin_s))].mean())

    @property
    def empty_fraction_outer_half(self) -> float:
        """Empty fraction of the shells from k_max / 2 to k_max."""
        return float(self.empty[len(self.counts) // 2 :].mean())

    @property
    def max_tau_gap_s(self) -> FloatArray:
        """(shells,) longest run of empty tau bins in each shell, in seconds."""
        gaps = np.zeros(len(self.counts))
        for shell, empty in enumerate(self.empty):
            run = best = 0
            for e in empty:
                run = run + 1 if e else 0
                best = max(best, run)
            gaps[shell] = best * self.tau_bin_s
        return gaps


def ktau_metrics(
    config: Config,
    trajectory: Trajectory,
    shells: int = 20,
    tau_bin_s: float = 0.5e-3,
    early_s: float = 5e-3,
) -> KTauMetrics:
    """(|k|, tau) sampling of the acquired TRs."""
    tau = trajectory.time_since_excitation_s
    n_tau = int(np.ceil(tau[-1] / tau_bin_s + 1e-9))
    tau_bin = np.minimum((tau / tau_bin_s).astype(np.int64), n_tau - 1)
    uses = np.bincount(trajectory.shape_index, minlength=len(trajectory.k_shape_per_m))
    counts = np.zeros((shells, n_tau), dtype=np.int64)
    for k_shape, n in zip(trajectory.k_shape_per_m, uses, strict=True):
        if n == 0:
            continue
        radius = np.linalg.norm(k_shape, axis=1) / config.k_max_per_m
        inside = radius < 1.0
        shell = (radius[inside] * shells).astype(np.int64)
        np.add.at(counts, (shell, tau_bin[inside]), n)
    return KTauMetrics(counts=counts, tau_bin_s=tau_bin_s, early_s=early_s)


def within_tr_redundancy(
    k_shapes_per_m: list[FloatArray],
    cell_per_m: float,
    radius_cells: float = 1.0,
    min_path_cells: float = 3.0,
) -> FloatArray:
    """Fraction of each shape's ADC samples that retrace an earlier part of it.

    A sample is redundant when an earlier sample of the same shape is within
    ``radius_cells`` of it and at least ``min_path_cells`` of path before it,
    so that neighbors along the path do not count. A hairpin petal's return
    leg and the repeated visits to k = 0 are redundant; a loop is not.
    """
    fractions = np.zeros(len(k_shapes_per_m))
    for i, k in enumerate(k_shapes_per_m):
        cells = k / cell_per_m
        path = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(cells, axis=0), axis=1))])
        pairs = cKDTree(cells).query_pairs(radius_cells, output_type="ndarray")
        far = path[pairs[:, 1]] - path[pairs[:, 0]] >= min_path_cells
        redundant = np.zeros(len(k), dtype=bool)
        redundant[pairs[far, 1]] = True
        fractions[i] = redundant.mean()
    return fractions


def format_redundancy(fractions: FloatArray) -> str:
    """Text summary of ``within_tr_redundancy``."""
    return (
        f"within-TR redundancy: {np.median(fractions):.1%} of a shape's samples (median; "
        f"{fractions.min():.1%} to {fractions.max():.1%}) are within 1 cell of an earlier "
        "pass of the same shape"
    )


def format_ktau(m: KTauMetrics) -> str:
    """Text summary of the (|k|, tau) sampling."""
    shells = len(m.counts)
    gaps = m.max_tau_gap_s * 1e3
    return "\n".join(
        [
            f"(|k|, tau) sampling over the scan: {shells} |k| shells x "
            f"{m.counts.shape[1]} tau bins of {m.tau_bin_s * 1e3:g} ms",
            f"  empty cells: {m.empty_fraction:.1%} overall, "
            f"{m.empty_fraction_early:.1%} before {m.early_s * 1e3:g} ms, "
            f"{m.empty_fraction_outer_half:.1%} in the outer half of k-space",
            f"  longest tau gap in a shell: median {np.median(gaps):.1f} ms, "
            f"worst {gaps.max():.1f} ms (shell {int(np.argmax(gaps))} of {shells}, "
            "0 = center)",
        ]
    )
