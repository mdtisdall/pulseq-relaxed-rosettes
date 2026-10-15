"""Analytic numerical phantom with relaxation and scan-time dependence, for simulated scans.

The object is a set of compartments: uniform ellipsoids and Gaussian blobs.
Each has a proton density ρ, an R2*, and an off-resonance Δf, uniform inside
it, and each of the three can vary over scan time T (sums of sines and of
smoothed blocks). The magnetization of a compartment at time since
excitation τ and scan time T is

    m_c(τ, T) = ρ_c(T) exp(-R2*_c(T) τ + 2πi Δf_c(T) τ),

and the signal at k-space position k (1/m) is

    S(k, τ, T) = sum_c m_c(τ, T) F_c(k),   F_c(k) = ∫ shape_c(r) exp(+2πi k·r) dr  (m³),

with the sign of the encoding phase in ``relaxed_rosettes.trajectory``.
``F_c`` is closed-form (the Fourier transform of a ball or of a Gaussian,
under the affine map of the compartment), so the signal is exact at every
sample: no NUFFT and no time segmentation, so a reconstruction that uses
them is not tested on data made with its own model.

Compartments add where they overlap, unless a compartment is ``inside``
another: then it replaces that compartment's tissue within its own shape (it
adds ``(m_c - m_parent) F_c``). It must lie within its parent. In the
discrete model of ``relaxed_rosettes.recon.nufft.Nufft`` with voxels of size
``v``, the signal corresponds to the image ``m v³`` (``image`` returns ``m``).

Coils have constant complex sensitivities. Noise is white and independent
across coils, with standard deviation ``noise_relative × |S(0, 0, 0)|`` in
each of the real and imaginary parts. The data are as the scanner's after the
ADC phase is applied (no RF phase), without concomitant fields.
"""

from __future__ import annotations

import tomllib
from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from relaxed_rosettes.config import ConfigError, digest_of
from relaxed_rosettes.recon.tomltables import (
    check_value,
    float_list,
    plain_table,
    table_item,
    toml_value,
)
from relaxed_rosettes.trajectory import Trajectory

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


class CompartmentShape(StrEnum):
    """Profile of a compartment in its own coordinates ``u``."""

    ELLIPSOID = "ellipsoid"
    """1 for ``|u| <= 1``, else 0; ``semi_axes_m`` are the semi-axes."""
    GAUSSIAN = "gaussian"
    """``exp(-|u|² / 2)``; ``semi_axes_m`` are the standard deviations."""


class Property(StrEnum):
    """A property that a modulation changes over scan time."""

    DENSITY = "density"
    """Relative: ``ρ(T) = ρ (1 + sum of modulations)``."""
    R2STAR = "r2star"
    """Added, in 1/s."""
    OFF_RESONANCE = "off_resonance"
    """Added, in Hz."""


@dataclass(frozen=True)
class Compartment:
    """A uniform ellipsoid or a Gaussian blob.

    Its points are ``center_m + R diag(semi_axes_m) u``, with ``R`` the
    rotation by ``rotation_deg`` about the fixed x, y, then z axes.
    """

    name: str
    shape: CompartmentShape = CompartmentShape.ELLIPSOID
    center_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    semi_axes_m: tuple[float, float, float] = (0.05, 0.05, 0.05)
    rotation_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)
    density: float = 1.0
    r2star_per_s: float = 20.0
    off_resonance_hz: float = 0.0
    inside: str = ""
    """Name of the compartment whose tissue this one replaces (empty: none)."""

    def matrix(self) -> FloatArray:
        """(3, 3) ``R diag(semi_axes_m)``: physical offset of unit coordinates."""
        rotation = Rotation.from_euler("xyz", self.rotation_deg, degrees=True).as_matrix()
        matrix: FloatArray = rotation * np.asarray(self.semi_axes_m)[None, :]
        return matrix


@dataclass(frozen=True)
class Sine:
    """``amplitude sin(2π frequency_hz T + phase_rad)``, added to a property."""

    compartment: str
    property: Property
    amplitude: float
    frequency_hz: float
    phase_rad: float = 0.0

    def value(self, time_s: FloatArray) -> FloatArray:
        value: FloatArray = self.amplitude * np.sin(
            2 * np.pi * self.frequency_hz * time_s + self.phase_rad
        )
        return value


@dataclass(frozen=True)
class Blocks:
    """Periodic blocks of ``amplitude``, with raised-cosine edges, added to a property.

    Each period of ``period_s`` starts at ``start_s + n period_s`` with a rise
    of ``ramp_s``; the fall starts ``on_s`` after the rise starts and lasts
    ``ramp_s``.
    """

    compartment: str
    property: Property
    amplitude: float
    period_s: float
    on_s: float
    ramp_s: float = 0.0
    start_s: float = 0.0

    def value(self, time_s: FloatArray) -> FloatArray:
        u = np.mod(np.asarray(time_s, dtype=np.float64) - self.start_s, self.period_s)
        ramp = max(self.ramp_s, 1e-30)
        rise = 0.5 * (1 - np.cos(np.pi * np.clip(u / ramp, 0.0, 1.0)))
        fall = 0.5 * (1 + np.cos(np.pi * np.clip((u - self.on_s) / ramp, 0.0, 1.0)))
        level = np.where(u < self.on_s, rise, fall)
        value: FloatArray = self.amplitude * level
        return value


def _default_compartments() -> tuple[Compartment, ...]:
    e, g = CompartmentShape.ELLIPSOID, CompartmentShape.GAUSSIAN
    return (
        Compartment("head", e, (0.0, 0.0, 0.0), (0.075, 0.095, 0.085), density=0.9),
        Compartment(
            "ventricle_left", e, (-0.015, 0.008, 0.012), (0.010, 0.032, 0.014),
            (0.0, 0.0, 12.0), density=1.0, r2star_per_s=6.0, inside="head",
        ),
        Compartment(
            "ventricle_right", e, (0.015, 0.008, 0.012), (0.010, 0.032, 0.014),
            (0.0, 0.0, -12.0), density=1.0, r2star_per_s=6.0, inside="head",
        ),
        Compartment(
            "bold", e, (0.035, -0.040, 0.030), (0.010, 0.010, 0.010),
            density=0.9, r2star_per_s=20.0, inside="head",
        ),
        Compartment(
            "vessel", e, (-0.020, 0.050, 0.0), (0.003, 0.003, 0.030),
            density=1.0, r2star_per_s=15.0, inside="head",
        ),
        Compartment(
            "detail_4mm", e, (-0.030, -0.040, -0.030), (0.004, 0.004, 0.004),
            density=0.5, r2star_per_s=30.0, inside="head",
        ),
        Compartment(
            "detail_2.5mm", e, (-0.010, -0.040, -0.030), (0.0025, 0.0025, 0.0025),
            density=0.5, r2star_per_s=30.0, inside="head",
        ),
        Compartment(
            "detail_1.5mm", e, (0.005, -0.040, -0.030), (0.0015, 0.0015, 0.0015),
            density=0.5, r2star_per_s=30.0, inside="head",
        ),
        Compartment(
            "blob", g, (0.030, 0.030, -0.020), (0.006, 0.004, 0.008),
            density=0.4, r2star_per_s=40.0, inside="head",
        ),
    )  # fmt: skip


def _default_sines() -> tuple[Sine, ...]:
    # Cardiac pulsation of the vessel: a fundamental and its second harmonic.
    return (
        Sine("vessel", Property.DENSITY, 0.2, 1.1),
        Sine("vessel", Property.DENSITY, 0.08, 2.2, 0.5),
    )


def _default_blocks() -> tuple[Blocks, ...]:
    # A BOLD block design: R2* falls by 0.6/s during the blocks, about +1.8%
    # signal at τ = 30 ms.
    return (Blocks("bold", Property.R2STAR, -0.6, period_s=40.0, on_s=20.0, ramp_s=5.0),)


@dataclass(frozen=True)
class PhantomConfig:
    """A phantom, its coils, and its noise. TOML: scalars, then
    ``[[compartment]]``, ``[[sine]]`` and ``[[blocks]]`` tables."""

    seed: int = 0
    noise_relative: float = 1e-3
    noise_readouts: int = 4
    coil_sensitivity_real: tuple[float, ...] = (1.0,)
    coil_sensitivity_imag: tuple[float, ...] = (0.0,)
    compartments: tuple[Compartment, ...] = field(default_factory=_default_compartments)
    sines: tuple[Sine, ...] = field(default_factory=_default_sines)
    blocks: tuple[Blocks, ...] = field(default_factory=_default_blocks)

    def __post_init__(self) -> None:
        errors = _validate(self)
        if errors:
            raise ConfigError("invalid phantom config:\n  " + "\n  ".join(errors))

    @property
    def sensitivities(self) -> NDArray[np.complex128]:
        """(coils,) complex coil sensitivities."""
        real = np.asarray(self.coil_sensitivity_real, dtype=np.float64)
        return real + 1j * np.asarray(self.coil_sensitivity_imag, dtype=np.float64)

    @property
    def modulations(self) -> list[Sine | Blocks]:
        """Every modulation: the sines, then the blocks."""
        return [*self.sines, *self.blocks]

    # Serialization.

    def to_dict(self) -> dict[str, Any]:
        """Plain dict in TOML layout."""
        data: dict[str, Any] = {
            "seed": self.seed,
            "noise_relative": self.noise_relative,
            "noise_readouts": self.noise_readouts,
            "coil_sensitivity_real": list(self.coil_sensitivity_real),
            "coil_sensitivity_imag": list(self.coil_sensitivity_imag),
        }
        for key, items in _TABLES.items():
            data[key] = [plain_table(item) for item in getattr(self, items[0])]
        return data

    def digest(self) -> str:
        """Short SHA-256 of the canonical config, for tagging outputs."""
        return digest_of(self.to_dict())

    def to_toml(self) -> str:
        """TOML text that ``from_toml`` reads back into an equal config."""
        data = self.to_dict()
        lines = [f"{k} = {toml_value(v)}" for k, v in data.items() if k not in _TABLES]
        for key in _TABLES:
            for table in data[key]:
                lines += ["", f"[[{key}]]"]
                lines += [f"{k} = {toml_value(v)}" for k, v in table.items()]
        return "\n".join(lines) + "\n"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PhantomConfig:
        """Build a config from TOML-layout data. Missing scalars take defaults;
        a missing table list is empty unless every table list is missing."""
        data = dict(data)
        kwargs: dict[str, Any] = {}
        for name in ("seed", "noise_readouts"):
            if name in data:
                kwargs[name] = check_value(name, data.pop(name), int)
        if "noise_relative" in data:
            kwargs["noise_relative"] = float(
                check_value("noise_relative", data.pop("noise_relative"), float)
            )
        for name in ("coil_sensitivity_real", "coil_sensitivity_imag"):
            if name in data:
                kwargs[name] = tuple(float_list(name, data.pop(name), None))
        if any(key in data for key in _TABLES):
            for key, (attr, item_cls) in _TABLES.items():
                tables = data.pop(key, [])
                if not isinstance(tables, list):
                    raise ConfigError(f"[[{key}]] must be an array of tables")
                kwargs[attr] = tuple(table_item(key, item_cls, t, _ENUMS) for t in tables)
        if data:
            raise ConfigError(f"unknown key(s): {', '.join(sorted(data))}")
        return cls(**kwargs)

    @classmethod
    def from_toml(cls, text: str) -> PhantomConfig:
        """Parse TOML text."""
        try:
            return cls.from_dict(tomllib.loads(text))
        except tomllib.TOMLDecodeError as e:
            raise ConfigError(f"invalid TOML: {e}") from e

    @classmethod
    def load(cls, path: Path) -> PhantomConfig:
        """Read a TOML phantom file."""
        return cls.from_toml(path.read_text())

    # The object.

    def properties(
        self, compartment: Compartment, time_s: FloatArray
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        """ρ, R2* (1/s), and Δf (Hz) of a compartment at scan times ``time_s``."""
        time = np.asarray(time_s, dtype=np.float64)
        change = {p: np.zeros_like(time) for p in Property}
        for modulation in self.modulations:
            if modulation.compartment == compartment.name:
                change[modulation.property] = change[modulation.property] + modulation.value(time)
        density = compartment.density * (1 + change[Property.DENSITY])
        r2star = compartment.r2star_per_s + change[Property.R2STAR]
        off_resonance = compartment.off_resonance_hz + change[Property.OFF_RESONANCE]
        return density, r2star, off_resonance

    def magnetization(
        self, compartment: Compartment, tau_s: FloatArray, time_s: FloatArray
    ) -> ComplexArray:
        """``m_c(τ, T)`` of a compartment, minus its parent's if it is ``inside`` one."""
        m = self._own_magnetization(compartment, tau_s, time_s)
        if compartment.inside:
            m = m - self._own_magnetization(self._by_name(compartment.inside), tau_s, time_s)
        return m

    def signal(self, k_per_m: FloatArray, tau_s: FloatArray, time_s: FloatArray) -> ComplexArray:
        """(points,) noiseless signal at k (points, 3), τ (points,), and T (points,)."""
        k = np.asarray(k_per_m, dtype=np.float64).reshape(-1, 3)
        tau = np.broadcast_to(np.asarray(tau_s, dtype=np.float64), (len(k),))
        time = np.broadcast_to(np.asarray(time_s, dtype=np.float64), (len(k),))
        total = np.zeros(len(k), dtype=np.complex128)
        for c in self.compartments:
            total += self.magnetization(c, tau, time) * fourier_transform(c, k)
        return total

    def reference_signal(self) -> float:
        """``|S(0, 0, 0)|``, the scale of the noise."""
        zero = np.zeros(1)
        return float(np.abs(self.signal(np.zeros((1, 3)), zero, zero))[0])

    def image(
        self, n: int, resolution_m: float, tau_s: float = 0.0, time_s: float = 0.0
    ) -> ComplexArray:
        """(n, n, n) magnetization at voxel centers, on the reconstruction's axes.

        Index ``i`` is at ``(i - n // 2) * resolution_m`` along each physical
        gradient axis. Shapes are sampled at the voxel centers (no partial
        volume).
        """
        axis = (np.arange(n) - n // 2) * resolution_m
        r = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
        tau, time = np.array([tau_s]), np.array([time_s])
        out = np.zeros(len(r), dtype=np.complex128)
        for c in self.compartments:
            out += self.magnetization(c, tau, time)[0] * profile(c, r)
        return out.reshape(n, n, n)

    def _own_magnetization(
        self, compartment: Compartment, tau_s: FloatArray, time_s: FloatArray
    ) -> ComplexArray:
        density, r2star, off_resonance = self.properties(compartment, time_s)
        exponent = (-r2star + 2j * np.pi * off_resonance) * np.asarray(tau_s, dtype=np.float64)
        m: ComplexArray = np.asarray(density * np.exp(exponent), dtype=np.complex128)
        return m

    def _by_name(self, name: str) -> Compartment:
        return next(c for c in self.compartments if c.name == name)


def fourier_transform(compartment: Compartment, k_per_m: FloatArray) -> ComplexArray:
    """(points,) ``F_c(k) = ∫ shape_c(r) exp(+2πi k·r) dr`` (m³) at k (points, 3)."""
    matrix = compartment.matrix()
    q = np.linalg.norm(k_per_m @ matrix, axis=1)  # |Aᵀ k|: frequency in unit coordinates
    volume = abs(np.linalg.det(matrix))
    if compartment.shape is CompartmentShape.ELLIPSOID:
        x = 2 * np.pi * q
        small = x < 1e-2
        safe = np.where(small, 1.0, x)
        ball = np.where(
            small,
            4 * np.pi / 3 * (1 - x**2 / 10 + x**4 / 280),
            4 * np.pi * (np.sin(safe) - safe * np.cos(safe)) / safe**3,
        )
    else:
        ball = (2 * np.pi) ** 1.5 * np.exp(-2 * np.pi**2 * q**2)
    phase = np.exp(2j * np.pi * (k_per_m @ np.asarray(compartment.center_m)))
    transform: ComplexArray = volume * ball * phase
    return transform


def profile(compartment: Compartment, r_m: FloatArray) -> FloatArray:
    """(points,) the shape of a compartment at positions r (points, 3)."""
    u = (r_m - np.asarray(compartment.center_m)) @ np.linalg.inv(compartment.matrix()).T
    radius2 = np.sum(u**2, axis=1)
    if compartment.shape is CompartmentShape.ELLIPSOID:
        return (radius2 <= 1.0).astype(np.float64)
    value: FloatArray = np.exp(-radius2 / 2)
    return value


def simulate_trs(
    phantom: PhantomConfig, trajectory: Trajectory, *, batch_trs: int = 64
) -> Iterator[NDArray[np.complex64]]:
    """(coils, samples) data of each acquired TR, in order, with noise.

    The scan time of a sample is its TR's start plus its time in the TR.
    """
    rng = np.random.default_rng([phantom.seed, 0])
    sensitivities = phantom.sensitivities
    sigma = phantom.noise_relative * phantom.reference_signal()
    tau = trajectory.time_since_excitation_s
    num_trs = len(trajectory.tr_index)
    for start in range(0, num_trs, batch_trs):
        trs = np.arange(start, min(start + batch_trs, num_trs))
        k = trajectory.k_per_m(trs)
        time = trajectory.tr_start_s[trs, None] + trajectory.adc_time_in_tr_s[None, :]
        signal = phantom.signal(
            k.reshape(-1, 3), np.broadcast_to(tau, time.shape).ravel(), time.ravel()
        ).reshape(len(trs), -1)
        shape = (len(trs), len(sensitivities), len(tau))
        noise = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
        data = sensitivities[None, :, None] * signal[:, None, :] + sigma * noise
        yield from data.astype(np.complex64)


def simulate_noise(phantom: PhantomConfig, samples: int) -> list[NDArray[np.complex64]]:
    """``noise_readouts`` (coils, samples) noise-only readouts, at the ADC dwell."""
    rng = np.random.default_rng([phantom.seed, 1])
    sigma = phantom.noise_relative * phantom.reference_signal()
    shape = (phantom.noise_readouts, len(phantom.sensitivities), samples)
    noise = sigma * (rng.standard_normal(shape) + 1j * rng.standard_normal(shape))
    return list(noise.astype(np.complex64))


# Serialization helpers.

_TABLES: dict[str, tuple[str, type]] = {
    "compartment": ("compartments", Compartment),
    "sine": ("sines", Sine),
    "blocks": ("blocks", Blocks),
}
_ENUMS: dict[str, type[StrEnum]] = {"CompartmentShape": CompartmentShape, "Property": Property}


def _validate(c: PhantomConfig) -> list[str]:
    errors: list[str] = []
    if c.noise_relative < 0:
        errors.append("noise_relative must be >= 0")
    if c.noise_readouts < 0:
        errors.append("noise_readouts must be >= 0")
    coils = len(c.coil_sensitivity_real)
    if coils == 0 or len(c.coil_sensitivity_imag) != coils:
        errors.append("coil_sensitivity_real and coil_sensitivity_imag need the same length >= 1")
    names = [comp.name for comp in c.compartments]
    if not names:
        errors.append("need at least one compartment")
    if len(set(names)) != len(names):
        errors.append("compartment names must be unique")
    for comp in c.compartments:
        if not comp.name:
            errors.append("every compartment needs a name")
        if min(comp.semi_axes_m) <= 0:
            errors.append(f"compartment {comp.name!r}: semi_axes_m must be > 0")
        if comp.r2star_per_s < 0:
            errors.append(f"compartment {comp.name!r}: r2star_per_s must be >= 0")
        if comp.inside and (comp.inside == comp.name or comp.inside not in names):
            errors.append(f"compartment {comp.name!r}: inside {comp.inside!r} is not another one")
    for m in c.modulations:
        if m.compartment not in names:
            errors.append(f"modulation of unknown compartment {m.compartment!r}")
    for b in c.blocks:
        if not (0 <= b.ramp_s <= b.on_s and b.on_s + b.ramp_s <= b.period_s):
            errors.append(
                f"blocks of {b.compartment!r}: need 0 <= ramp_s <= on_s "
                "and on_s + ramp_s <= period_s"
            )
    return errors
