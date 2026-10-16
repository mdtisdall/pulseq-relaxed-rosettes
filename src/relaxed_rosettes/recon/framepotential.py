"""Frame potential of the linear reconstruction model of each coverage requirement.

A requirement (``docs/example-requirements-design.md``, section 3) defines a
linear model: voxels of ``resolution_m`` over the design's field of view, a
basis over time since excitation τ (a rank-``tau_rank`` decay basis, or a
constant if τ is averaged), and a basis over scan time T (hat functions
``scan_time_width_s`` apart, or a constant if T is averaged). It uses the
samples inside the k-space ball of its resolution, as ``grid`` and ``sense``
select them, optionally weighted by a Hann taper of ``|k|``. No coils, no
off-resonance. Sample ``m`` is the row

    a_m(v, l, j) = h(k_m) exp(2πi k_m·r_v) phi_l(τ_m) b_j(T_m)

of the forward operator ``A``. Its frame potential is

    FP = ||A Aᴴ||_F² = sum_{m,n} h_m² h_n² |D(k_m - k_n)|² K_τ(m, n)² K_T(m, n)²,

with ``D`` the Dirichlet kernel of the voxel grid, ``K_τ = sum_l phi_l phi_l``
and ``K_T = sum_j b_j b_j``. The score is

    ratio = N FP / trace(A Aᴴ)²  >= 1,

with ``N`` the number of coefficients: the mean of the squared eigenvalues of
``Aᴴ A`` over the square of their mean, ``1 + CV²``. It is 1 for a tight frame
(``Aᴴ A`` a multiple of the identity) and grows as the reconstruction gets
worse conditioned. It does not depend on the scale of the bases or the data.
The ball does not fill the band of the cubic voxel grid, so the ratio does
not reach 1: an offset that is the same for every design.

Evaluation. ``|D(Δk)|² = sum_d w(d) exp(2πi Δk·d)`` over the differences
``d`` of voxel positions, with ``w(d)`` the number of voxel pairs at
difference ``d``. So each term is ``sum_d w(d) |P(d)|²``, with ``P`` a type-1
NUFFT of per-sample weights onto a grid of twice the size: one channel for
each pair of τ basis functions, and, for terms local in T, one NUFFT for each
hat function and its right neighbour.

τ averaging. A model that averages τ fits the decaying signal of every sample
with one value per voxel. The frame potential cannot see the error that this
leaves (``docs/example-requirements-design.md``, experiment 1a): it depends on
how τ is spread over k-space, and on where the object's signal is. In each
Nyquist cell (spacing ``1 / fov``) inside the ball, the samples average the
decay basis to ``mean_cell(phi_l)``, where the model assumes the readout mean
``mean(phi_l)``. The score is the mismatch, weighted by the energy ``s_l²``
of each basis function over the R2* range and by a prior object spectrum
``S(k) = 1 / (1 + (|k| size)^power)``:

    mismatch = sum_cells S sum_l s_l² (mean_cell(phi_l) - mean(phi_l))²
               / (sum_cells S · sum_l s_l² mean(phi_l)²).

It is 0 when every occupied cell sees the readout's mix of τ. On designs A–M
it predicts the τ-averaging error of the static reconstruction (Spearman 0.95,
for size 0.2 m and power 4); a flat prior does not.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from relaxed_rosettes.config import ConfigError, digest_of
from relaxed_rosettes.recon.basis import decay_basis
from relaxed_rosettes.recon.nufft import Nufft
from relaxed_rosettes.recon.tomltables import (
    check_value,
    plain_table,
    table_item,
    toml_value,
)
from relaxed_rosettes.trajectory import Trajectory

FloatArray = NDArray[np.float64]


class Taper(StrEnum):
    """Weighting of the samples by ``|k|``."""

    NONE = "none"
    HANN = "hann"
    """``(1 + cos(π |k| / k_res)) / 2``, with ``k_res = 1 / (2 resolution_m)``."""


@dataclass(frozen=True)
class Requirement:
    """One reconstruction that the data must support."""

    name: str
    resolution_m: float
    tau_rank: int = 0
    """0: averaged over τ; L >= 1: resolved with a rank-L decay basis."""
    r2star_min_per_s: float = 5.0
    r2star_max_per_s: float = 100.0
    scan_time_width_s: float = 0.0
    """0: averaged over scan time; > 0: hat functions this far apart."""
    taper: Taper = Taper.NONE
    priority: float = 1.0


def _default_requirements() -> tuple[Requirement, ...]:
    # The examples of docs/example-requirements-design.md, section 1. The
    # navigator's width, BOLD's resolution and the τ ranks are placeholders.
    return (
        Requirement("navigator", 8e-3, scan_time_width_s=0.15),
        Requirement("relaxation_4d", 1.5e-3, tau_rank=4),
        Requirement("static", 1e-3),
        Requirement("bold", 5e-3, tau_rank=2, scan_time_width_s=0.5),
    )


@dataclass(frozen=True)
class RequirementsConfig:
    """Requirements, and the accuracy of their evaluation. TOML: scalars, then
    ``[[requirement]]`` tables."""

    eps: float = 1e-4
    """FINUFFT tolerance."""
    threads: int = 0
    """FINUFFT threads (0: its default)."""
    batch_trs: int = 500
    """TRs per NUFFT, for the terms averaged over scan time."""
    object_size_m: float = 0.2
    """Size of the object in the prior spectrum of the τ-averaging mismatch."""
    object_spectrum_power: float = 4.0
    """Power of ``|k|`` in that prior (4: a piecewise-smooth object)."""
    requirements: tuple[Requirement, ...] = field(default_factory=_default_requirements)

    def __post_init__(self) -> None:
        errors = _validate(self)
        if errors:
            raise ConfigError("invalid requirements config:\n  " + "\n  ".join(errors))

    def to_dict(self) -> dict[str, Any]:
        """Plain dict in TOML layout."""
        return {
            "eps": self.eps,
            "threads": self.threads,
            "batch_trs": self.batch_trs,
            "object_size_m": self.object_size_m,
            "object_spectrum_power": self.object_spectrum_power,
            "requirement": [plain_table(r) for r in self.requirements],
        }

    def digest(self) -> str:
        """Short SHA-256 of the canonical config, for tagging outputs."""
        return digest_of(self.to_dict())

    def to_toml(self) -> str:
        """TOML text that ``from_toml`` reads back into an equal config."""
        data = self.to_dict()
        lines = [f"{k} = {toml_value(v)}" for k, v in data.items() if k != "requirement"]
        for table in data["requirement"]:
            lines += ["", "[[requirement]]"]
            lines += [f"{k} = {toml_value(v)}" for k, v in table.items()]
        return "\n".join(lines) + "\n"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RequirementsConfig:
        """Build a config from TOML-layout data; without ``[[requirement]]``
        tables, the default requirements."""
        data = dict(data)
        kwargs: dict[str, Any] = {}
        for name in ("eps", "object_size_m", "object_spectrum_power"):
            if name in data:
                kwargs[name] = float(check_value(name, data.pop(name), float))
        for name in ("threads", "batch_trs"):
            if name in data:
                kwargs[name] = check_value(name, data.pop(name), int)
        if "requirement" in data:
            tables = data.pop("requirement")
            if not isinstance(tables, list):
                raise ConfigError("[[requirement]] must be an array of tables")
            enums: dict[str, type[StrEnum]] = {"Taper": Taper}
            kwargs["requirements"] = tuple(
                table_item("requirement", Requirement, t, enums) for t in tables
            )
        if data:
            raise ConfigError(f"unknown key(s): {', '.join(sorted(data))}")
        return cls(**kwargs)

    @classmethod
    def from_toml(cls, text: str) -> RequirementsConfig:
        """Parse TOML text."""
        try:
            return cls.from_dict(tomllib.loads(text))
        except tomllib.TOMLDecodeError as e:
            raise ConfigError(f"invalid TOML: {e}") from e

    @classmethod
    def load(cls, path: Path) -> RequirementsConfig:
        """Read a TOML requirements file."""
        return cls.from_toml(path.read_text())


@dataclass(frozen=True)
class FramePotential:
    """The frame potential of one requirement's model."""

    requirement: Requirement
    frame_potential: float
    trace: float
    coefficients: int
    samples: int

    @property
    def ratio(self) -> float:
        """``N FP / trace²``: 1 for a tight frame."""
        return self.coefficients * self.frame_potential / self.trace**2

    @property
    def cv(self) -> float:
        """Coefficient of variation of the eigenvalues of ``Aᴴ A``."""
        return float(np.sqrt(max(self.ratio - 1.0, 0.0)))


def frame_potential(
    requirement: Requirement,
    trajectory: Trajectory,
    fov_m: float,
    *,
    eps: float = 1e-4,
    threads: int = 0,
    batch_trs: int = 500,
) -> FramePotential:
    """Frame potential of a requirement's model on a trajectory."""
    res = requirement.resolution_m
    n = max(1, round(fov_m / res))
    k_max = 1 / (2 * res)
    nufft = Nufft(n=2 * n, resolution_m=res, eps=eps, upsampling=1.25, threads=threads)
    w1 = np.maximum(n - np.abs(np.arange(2 * n) - n), 0).astype(np.float64)

    def energy(grids: NDArray[np.complex64], multiplicity: FloatArray) -> float:
        """sum_c multiplicity_c sum_d w(d) |P_c(d)|²."""
        power = grids.real**2 + grids.imag**2
        along_k = (power @ w1.astype(np.float32)).astype(np.float64)
        weighted = along_k @ w1 @ w1
        return float(np.dot(multiplicity, weighted))

    # τ: the products of pairs of basis functions at the ADC samples.
    tau = trajectory.time_since_excitation_s
    if requirement.tau_rank == 0:
        phi = np.ones((len(tau), 1))
    else:
        basis = decay_basis(
            tau,
            requirement.r2star_min_per_s,
            requirement.r2star_max_per_s,
            rank=requirement.tau_rank,
        )
        phi = basis.functions.T
    rank = phi.shape[1]
    pairs = [(a, b) for a in range(rank) for b in range(a, rank)]
    tau_products = np.stack([phi[:, a] * phi[:, b] for a, b in pairs])  # (pairs, samples)
    tau_multiplicity = np.array([1.0 if a == b else 2.0 for a, b in pairs])
    tau_diagonal = np.sum(phi**2, axis=1)  # K_τ(m, m)

    num_trs = len(trajectory.tr_index)
    sample_time = trajectory.adc_time_in_tr_s

    def selected(trs: NDArray[np.int64]) -> tuple[FloatArray, FloatArray, FloatArray]:
        """k, squared taper, and (TR, sample) index of the samples in the ball."""
        k = trajectory.k_per_m(trs)
        radius = np.linalg.norm(k, axis=-1)
        keep = radius <= k_max
        weight = np.ones(int(keep.sum()))
        if requirement.taper is Taper.HANN:
            weight = (0.5 * (1 + np.cos(np.pi * radius[keep] / k_max))) ** 2
        tr_of, sample_of = np.nonzero(keep)
        return k[keep], weight, np.stack([trs[tr_of], sample_of])

    fp = 0.0
    trace = 0.0
    samples = 0
    width = requirement.scan_time_width_s
    if width == 0:
        grids = np.zeros((len(pairs), 2 * n, 2 * n, 2 * n), dtype=np.complex64)
        for start in range(0, num_trs, batch_trs):
            trs = np.arange(start, min(start + batch_trs, num_trs))
            k, weight, index = selected(trs)
            if len(k) == 0:
                continue
            values = (tau_products[:, index[1]] * weight[None]).astype(np.complex64)
            grids += nufft.adjoint(values, k)
            trace += float(np.sum(weight * tau_diagonal[index[1]]))
            samples += len(k)
        fp = energy(grids, tau_multiplicity)
        coefficients = n**3 * rank
    else:
        first = float(trajectory.tr_start_s[0] + sample_time[0])
        last = float(trajectory.tr_start_s[-1] + sample_time[-1])
        num_knots = int(np.floor((last - first) / width)) + 2
        knots = first + width * np.arange(num_knots)
        multiplicity = np.concatenate([tau_multiplicity, 2 * tau_multiplicity])
        for j in range(num_knots):
            # Samples in the support of hat j: [t_(j-1), t_(j+1)].
            low, high = knots[j] - width, knots[j] + width
            first_tr = np.searchsorted(trajectory.tr_start_s + sample_time[-1], low)
            last_tr = np.searchsorted(trajectory.tr_start_s + sample_time[0], high, "right")
            trs = np.arange(first_tr, last_tr)
            if len(trs) == 0:
                continue
            k, weight, index = selected(trs)
            time = trajectory.tr_start_s[index[0]] + sample_time[index[1]]
            hat = np.maximum(1 - np.abs(time - knots[j]) / width, 0.0)
            right = np.maximum(1 - np.abs(time - knots[j] - width) / width, 0.0)  # hat j + 1
            inside = hat > 0
            if not inside.any():
                continue
            k, weight, index = k[inside], weight[inside], index[:, inside]
            hat, right = hat[inside], right[inside]
            products = tau_products[:, index[1]] * weight[None]
            values = np.concatenate([products * hat**2, products * hat * right]).astype(
                np.complex64
            )
            fp += energy(nufft.adjoint(values, k), multiplicity)
            trace += float(np.sum(weight * tau_diagonal[index[1]] * hat**2))
            samples += int(np.sum(time >= knots[j]))  # each sample counted once
        coefficients = n**3 * rank * num_knots
    return FramePotential(
        requirement=requirement,
        frame_potential=fp,
        trace=trace * n**3,
        coefficients=coefficients,
        samples=samples,
    )


@dataclass(frozen=True)
class TauAveraging:
    """The τ-averaging mismatch of a requirement that averages τ and scan time."""

    requirement: Requirement
    mismatch: float
    cells: int
    """Occupied Nyquist cells in the ball."""
    samples: int


def averages_tau(requirement: Requirement) -> bool:
    """Whether the τ-averaging mismatch applies: τ and scan time both averaged."""
    return requirement.tau_rank == 0 and requirement.scan_time_width_s == 0


def tau_averaging(
    requirement: Requirement,
    trajectory: Trajectory,
    fov_m: float,
    *,
    object_size_m: float = 0.2,
    spectrum_power: float = 4.0,
    rank: int = 2,
    batch_trs: int = 500,
) -> TauAveraging:
    """τ-averaging mismatch of a requirement whose model averages τ and scan time."""
    if not averages_tau(requirement):
        raise ValueError(f"requirement {requirement.name!r} does not average tau and scan time")
    n = max(1, round(fov_m / requirement.resolution_m))
    k_max = 1 / (2 * requirement.resolution_m)
    basis = decay_basis(
        trajectory.time_since_excitation_s,
        requirement.r2star_min_per_s,
        requirement.r2star_max_per_s,
        rank=rank,
    )
    phi = basis.functions  # (rank, samples)
    energy = basis.singular_values[:rank] ** 2
    readout_mean = phi.mean(axis=1)
    count = np.zeros(n**3)
    sums = np.zeros((rank, n**3))
    samples = 0
    num_trs = len(trajectory.tr_index)
    for start in range(0, num_trs, batch_trs):
        k = trajectory.k_per_m(np.arange(start, min(start + batch_trs, num_trs)))
        keep = np.linalg.norm(k, axis=-1) <= k_max
        _, sample_of = np.nonzero(keep)
        cell = np.clip(np.rint(k[keep] * fov_m).astype(np.int64) + n // 2, 0, n - 1)
        flat = np.ravel_multi_index(tuple(cell.T), (n, n, n))
        count += np.bincount(flat, minlength=n**3)
        for row, values in zip(sums, phi[:, sample_of], strict=True):
            row += np.bincount(flat, values, minlength=n**3)
        samples += len(flat)
    occupied = np.flatnonzero(count)
    difference = sums[:, occupied] / count[occupied] - readout_mean[:, None]
    error = energy @ difference**2
    position = np.stack(np.unravel_index(occupied, (n, n, n)), axis=1) - n // 2
    radius = np.linalg.norm(position, axis=1) / fov_m
    prior = 1 / (1 + (radius * object_size_m) ** spectrum_power)
    mismatch = float(prior @ error / (prior.sum() * (energy @ readout_mean**2)))
    return TauAveraging(requirement, mismatch, len(occupied), samples)


def format_tau_averaging(results: list[TauAveraging]) -> str:
    """A table of the τ-averaging mismatch of each requirement."""
    lines = [f"{'requirement':<16} {'res (mm)':>8} {'cells':>10} {'samples':>12} {'mismatch':>10}"]
    for r in results:
        lines.append(
            f"{r.requirement.name:<16} {r.requirement.resolution_m * 1e3:>8g} "
            f"{r.cells:>10} {r.samples:>12} {r.mismatch:>10.4g}"
        )
    return "\n".join(lines)


def format_frame_potentials(results: list[FramePotential]) -> str:
    """A table of the frame potential of each requirement."""
    header = (
        f"{'requirement':<16} {'res (mm)':>8} {'tau rank':>8} {'T width (s)':>11} "
        f"{'taper':>5} {'samples':>12} {'coefficients':>13} {'ratio':>10} {'CV':>8}"
    )
    lines = [header]
    for r in results:
        q = r.requirement
        width = f"{q.scan_time_width_s:g}" if q.scan_time_width_s else "averaged"
        rank = str(q.tau_rank) if q.tau_rank else "averaged"
        lines.append(
            f"{q.name:<16} {q.resolution_m * 1e3:>8g} {rank:>8} {width:>11} "
            f"{q.taper.value:>5} {r.samples:>12} {r.coefficients:>13} "
            f"{r.ratio:>10.4g} {r.cv:>8.4g}"
        )
    return "\n".join(lines)


def _validate(c: RequirementsConfig) -> list[str]:
    errors: list[str] = []
    if not 0 < c.eps < 1:
        errors.append("eps must be in (0, 1)")
    if c.threads < 0:
        errors.append("threads must be >= 0")
    if c.batch_trs <= 0:
        errors.append("batch_trs must be > 0")
    if not c.object_size_m > 0:
        errors.append("object_size_m must be > 0")
    if not c.object_spectrum_power > 0:
        errors.append("object_spectrum_power must be > 0")
    names = [r.name for r in c.requirements]
    if not names:
        errors.append("need at least one requirement")
    if len(set(names)) != len(names):
        errors.append("requirement names must be unique")
    for r in c.requirements:
        if not r.name:
            errors.append("every requirement needs a name")
        if not r.resolution_m > 0:
            errors.append(f"requirement {r.name!r}: resolution_m must be > 0")
        if r.tau_rank < 0:
            errors.append(f"requirement {r.name!r}: tau_rank must be >= 0")
        if not 0 <= r.r2star_min_per_s < r.r2star_max_per_s:
            errors.append(f"requirement {r.name!r}: need 0 <= r2star_min_per_s < r2star_max_per_s")
        if r.scan_time_width_s < 0:
            errors.append(f"requirement {r.name!r}: scan_time_width_s must be >= 0")
        if r.priority < 0:
            errors.append(f"requirement {r.name!r}: priority must be >= 0")
    return errors
