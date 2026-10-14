"""Sequence design parameters.

Every parameter of a sequence design is in one frozen ``Config``. A design is
reproducible from its config, which includes the random seed. Configs are read
from and written to TOML files with one table per section. All quantities are
SI unless the field name gives another unit.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import tomllib
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

GAMMA_HZ_PER_T = 42.576e6
"""Proton gyromagnetic ratio / 2π, the same value that pypulseq uses."""


class RfPhase(StrEnum):
    """RF (and ADC) phase scheme across TRs."""

    RANDOM = "random"
    QUADRATIC_117 = "quadratic117"


@dataclass(frozen=True)
class SystemConfig:
    """Scanner limits and timing, passed to ``pypulseq.Opts``."""

    max_grad_mT_per_m: float = 28.0
    max_slew_T_per_m_per_s: float = 120.0
    rf_dead_time_s: float = 100e-6
    rf_ringdown_time_s: float = 20e-6
    adc_dead_time_s: float = 0.0
    grad_raster_time_s: float = 10e-6
    B0_T: float = 2.89
    adc_samples_limit: int = 8192
    adc_samples_divisor: int = 4


@dataclass(frozen=True)
class GeometryConfig:
    """Imaging field of view and isotropic resolution."""

    fov_m: float = 0.256
    resolution_m: float = 1e-3


@dataclass(frozen=True)
class ExcitationConfig:
    """Hard-pulse excitation, played with all gradients off."""

    flip_angle_deg: float = 15.0
    rf_duration_s: float = 100e-6
    rf_phase: RfPhase = RfPhase.RANDOM


@dataclass(frozen=True)
class ReadoutConfig:
    """TR and ADC sampling."""

    tr_s: float = 50e-3
    adc_dwell_s: float = 3.2e-6
    adc_num_samples: int = 14000


@dataclass(frozen=True)
class GradientConfig:
    """Limits for the trajectory optimizer (at most the system limits).

    The boundary windows are the first and last ``boundary_duration_s`` of the
    gradient waveform in each TR. There the slew is limited to
    ``boundary_slew_T_per_m_per_s`` for quieter transitions around the RF pulse.
    """

    max_grad_mT_per_m: float = 28.0
    max_slew_T_per_m_per_s: float = 100.0
    boundary_slew_T_per_m_per_s: float = 25.0
    boundary_duration_s: float = 1e-3


@dataclass(frozen=True)
class ShapeConfig:
    """Rosette shape design. k-space radii are in units of k_max.

    Each TR visits ``echoes_per_tr`` petal tips, crossing k = 0 between
    consecutive tips (one gradient echo each), then ends at the spoil moment.
    Tip radii are blue noise in sqrt(radius) between the square roots of
    ``tip_radius_min`` and ``tip_radius_max``. Each segment between targets
    gets time proportional to the sqrt of its target radius, scaled by a
    random factor in ``[1 - timing_jitter, 1 + timing_jitter]``.
    """

    num_shapes: int = 32
    echoes_per_tr: int = 19
    tip_radius_min: float = 0.04
    tip_radius_max: float = 1.0
    max_consecutive_tip_dot: float = -0.5
    spoil_moment_kmax: float = 5.0
    timing_jitter: float = 0.2


@dataclass(frozen=True)
class ScanConfig:
    """Number of TRs played."""

    num_trs: int = 5000
    num_dummy_trs: int = 100


@dataclass(frozen=True)
class Config:
    """All parameters of one sequence design."""

    seed: int = 0
    system: SystemConfig = field(default_factory=SystemConfig)
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    excitation: ExcitationConfig = field(default_factory=ExcitationConfig)
    readout: ReadoutConfig = field(default_factory=ReadoutConfig)
    gradients: GradientConfig = field(default_factory=GradientConfig)
    shapes: ShapeConfig = field(default_factory=ShapeConfig)
    scan: ScanConfig = field(default_factory=ScanConfig)

    def __post_init__(self) -> None:
        errors = _validate(self)
        if errors:
            raise ConfigError("invalid config:\n  " + "\n  ".join(errors))

    @property
    def k_max_per_m(self) -> float:
        """Largest k-space radius, 1 / (2 resolution)."""
        return 1.0 / (2.0 * self.geometry.resolution_m)

    @property
    def raster_points_per_tr(self) -> int:
        """Number of gradient raster points in one TR."""
        return round(self.readout.tr_s / self.system.grad_raster_time_s)

    # Times within a TR block, from the start of the block.

    @property
    def rf_center_s(self) -> float:
        """Center of the hard pulse (k = 0 reference)."""
        return self.system.rf_dead_time_s + self.excitation.rf_duration_s / 2

    @property
    def rf_end_s(self) -> float:
        """End of the hard pulse; the gradients are off until here."""
        return self.system.rf_dead_time_s + self.excitation.rf_duration_s

    @property
    def adc_start_s(self) -> float:
        """Start of the ADC, after the RF ringdown."""
        return self.rf_end_s + self.system.rf_ringdown_time_s

    @property
    def adc_end_s(self) -> float:
        """End of the ADC."""
        return self.adc_start_s + self.readout.adc_num_samples * self.readout.adc_dwell_s

    def to_dict(self) -> dict[str, Any]:
        """Plain dict of all values (enums as strings), in TOML layout."""
        plain: dict[str, Any] = _plain(dataclasses.asdict(self))
        return plain

    def digest(self) -> str:
        """Short SHA-256 of the canonical config, for tagging outputs."""
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()[:12]

    def to_toml(self) -> str:
        """TOML text that ``from_toml`` reads back into an equal config."""
        data = self.to_dict()
        lines = [f"seed = {_toml_value(data.pop('seed'))}"]
        for section, values in data.items():
            lines += ["", f"[{section}]"]
            lines += [f"{key} = {_toml_value(value)}" for key, value in values.items()]
        return "\n".join(lines) + "\n"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Config:
        """Build a config from TOML-layout data. Missing values take defaults."""
        data = dict(data)
        kwargs: dict[str, Any] = {}
        for f in dataclasses.fields(cls):
            if f.name not in data:
                continue
            value = data.pop(f.name)
            if f.name == "seed":
                kwargs["seed"] = _check_type("seed", value, int)
                continue
            if not isinstance(value, dict):
                raise ConfigError(f"[{f.name}] must be a table")
            kwargs[f.name] = _section_from_dict(f.name, _SECTIONS[f.name], value)
        if data:
            raise ConfigError(f"unknown key(s): {', '.join(sorted(data))}")
        return cls(**kwargs)

    @classmethod
    def from_toml(cls, text: str) -> Config:
        """Parse TOML text."""
        try:
            return cls.from_dict(tomllib.loads(text))
        except tomllib.TOMLDecodeError as e:
            raise ConfigError(f"invalid TOML: {e}") from e

    @classmethod
    def load(cls, path: Path) -> Config:
        """Read a TOML config file."""
        return cls.from_toml(path.read_text())

    def with_overrides(self, overrides: list[str]) -> Config:
        """Apply ``section.key=value`` (or ``seed=value``) overrides.

        Each value is parsed as a TOML value, falling back to a bare string.
        """
        data = self.to_dict()
        for item in overrides:
            path, sep, raw = item.partition("=")
            if not sep:
                raise ConfigError(f"override {item!r} is not key=value")
            keys = path.strip().split(".")
            try:
                value = tomllib.loads(f"v = {raw.strip()}")["v"]
            except tomllib.TOMLDecodeError:
                value = raw.strip()
            if keys == ["seed"]:
                data["seed"] = value
            elif len(keys) == 2 and keys[0] in _SECTIONS:
                if keys[1] not in data[keys[0]]:
                    raise ConfigError(f"unknown key: {path}")
                data[keys[0]][keys[1]] = value
            else:
                raise ConfigError(f"unknown key: {path}")
        return Config.from_dict(data)


class ConfigError(ValueError):
    """A config file or override is invalid."""


_SECTIONS: dict[str, type] = {
    f.name: f.default_factory  # type: ignore[misc]
    for f in dataclasses.fields(Config)
    if f.name != "seed"
}


def _section_from_dict(name: str, cls: type, values: dict[str, Any]) -> Any:
    known = {f.name: f for f in dataclasses.fields(cls)}
    unknown = sorted(set(values) - set(known))
    if unknown:
        raise ConfigError(f"unknown key(s) in [{name}]: {', '.join(unknown)}")
    kwargs: dict[str, Any] = {}
    for key, value in values.items():
        default = getattr(cls(), key)
        label = f"{name}.{key}"
        if isinstance(default, StrEnum):
            try:
                kwargs[key] = type(default)(value)
            except ValueError:
                allowed = ", ".join(m.value for m in type(default))
                raise ConfigError(f"{label} must be one of: {allowed}") from None
        elif isinstance(default, int):
            kwargs[key] = _check_type(label, value, int)
        else:
            kwargs[key] = float(_check_type(label, value, (int, float)))
    return cls(**kwargs)


def _check_type(label: str, value: Any, types: type | tuple[type, ...]) -> Any:
    if isinstance(value, bool) or not isinstance(value, types):
        raise ConfigError(f"{label} has the wrong type: {value!r}")
    return value


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, StrEnum):
        return value.value
    return value


def _toml_value(value: Any) -> str:
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, float):
        return repr(value)
    return str(value)


def _validate(c: Config) -> list[str]:
    errors: list[str] = []

    def positive(label: str, value: float) -> None:
        if not value > 0:
            errors.append(f"{label} must be > 0 (got {value})")

    s, g, r, sh = c.system, c.gradients, c.readout, c.shapes
    positive("system.grad_raster_time_s", s.grad_raster_time_s)
    positive("geometry.fov_m", c.geometry.fov_m)
    positive("geometry.resolution_m", c.geometry.resolution_m)
    positive("excitation.flip_angle_deg", c.excitation.flip_angle_deg)
    positive("excitation.rf_duration_s", c.excitation.rf_duration_s)
    positive("readout.tr_s", r.tr_s)
    positive("readout.adc_dwell_s", r.adc_dwell_s)
    positive("readout.adc_num_samples", r.adc_num_samples)
    positive("gradients.max_grad_mT_per_m", g.max_grad_mT_per_m)
    positive("gradients.max_slew_T_per_m_per_s", g.max_slew_T_per_m_per_s)
    positive("gradients.boundary_slew_T_per_m_per_s", g.boundary_slew_T_per_m_per_s)
    positive("shapes.num_shapes", sh.num_shapes)
    positive("shapes.echoes_per_tr", sh.echoes_per_tr)
    positive("shapes.spoil_moment_kmax", sh.spoil_moment_kmax)
    positive("scan.num_trs", c.scan.num_trs)
    if errors:
        return errors

    if g.max_grad_mT_per_m > s.max_grad_mT_per_m:
        errors.append("gradients.max_grad_mT_per_m exceeds system.max_grad_mT_per_m")
    if g.max_slew_T_per_m_per_s > s.max_slew_T_per_m_per_s:
        errors.append("gradients.max_slew_T_per_m_per_s exceeds system.max_slew_T_per_m_per_s")
    if g.boundary_slew_T_per_m_per_s > g.max_slew_T_per_m_per_s:
        errors.append("gradients.boundary_slew_T_per_m_per_s exceeds max_slew_T_per_m_per_s")
    if g.boundary_duration_s < 0 or 2 * g.boundary_duration_s >= r.tr_s:
        errors.append("gradients.boundary_duration_s must be in [0, tr_s / 2)")
    if not 0 < sh.tip_radius_min < sh.tip_radius_max <= 1:
        errors.append("need 0 < shapes.tip_radius_min < shapes.tip_radius_max <= 1")
    if not -1 < sh.max_consecutive_tip_dot <= 1:
        errors.append("shapes.max_consecutive_tip_dot must be in (-1, 1]")
    if not 0 <= sh.timing_jitter < 1:
        errors.append("shapes.timing_jitter must be in [0, 1)")
    if c.scan.num_dummy_trs < 0:
        errors.append("scan.num_dummy_trs must be >= 0")

    raster_steps = r.tr_s / s.grad_raster_time_s
    if abs(raster_steps - round(raster_steps)) > 1e-6:
        errors.append("readout.tr_s must be a whole number of gradient raster times")

    # The ADC must fit in the TR after the RF pulse.
    adc_start = s.rf_dead_time_s + c.excitation.rf_duration_s + s.rf_ringdown_time_s
    if adc_start + r.adc_num_samples * r.adc_dwell_s > r.tr_s:
        errors.append("the ADC (adc_num_samples * adc_dwell_s after the RF pulse) exceeds tr_s")

    # Nyquist along the trajectory: at peak gradient, consecutive samples
    # must be at most 1 / FOV apart in k-space.
    dk_per_sample = GAMMA_HZ_PER_T * g.max_grad_mT_per_m * 1e-3 * r.adc_dwell_s
    if dk_per_sample * c.geometry.fov_m > 1.0:
        max_dwell = 1.0 / (GAMMA_HZ_PER_T * g.max_grad_mT_per_m * 1e-3 * c.geometry.fov_m)
        errors.append(
            f"readout.adc_dwell_s undersamples along the path at "
            f"{g.max_grad_mT_per_m} mT/m; use at most {max_dwell:.3g} s"
        )
    return errors
