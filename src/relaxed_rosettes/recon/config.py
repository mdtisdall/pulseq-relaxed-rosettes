"""Reconstruction parameters.

Every reconstruction parameter is in one frozen ``ReconConfig``, separate from
the design ``Config``: one scan can be reconstructed many ways. It is read
from and written to TOML in the same layout as the design config, and its
digest is stored in every output, so a result is reproducible from the raw
data, the trajectory, and the recon config. All quantities are SI unless the
field name gives another unit.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from relaxed_rosettes.config import (
    Config,
    ConfigError,
    apply_overrides,
    digest_of,
    plain_dict,
    sections_from_dict,
    toml_text,
)


@dataclass(frozen=True)
class DataConfig:
    """Which samples the reconstruction uses.

    Samples are used if ``|k| <= 1 / (2 * resolution_m)`` and their time since
    excitation is in ``[tau_min_s, tau_max_s]`` (``tau_max_s = 0``: to the end
    of the readout). ``max_trs = 0`` uses every TR in the raw data; a positive
    value uses the first ``max_trs``. TRs are processed ``batch_trs`` at a time.
    """

    resolution_m: float = 2e-3
    tau_min_s: float = 0.0
    tau_max_s: float = 0.0
    max_trs: int = 0
    batch_trs: int = 128


@dataclass(frozen=True)
class CoilConfig:
    """Coil compression after noise prewhitening.

    The compression keeps the ``virtual_coils`` strongest singular vectors of
    the whitened data (0: no compression), estimated from
    ``calibration_trs`` TRs spread evenly over the scan.
    """

    virtual_coils: int = 12
    calibration_trs: int = 256


@dataclass(frozen=True)
class NufftConfig:
    """FINUFFT accuracy and threads (``threads = 0``: FINUFFT's default)."""

    eps: float = 1e-4
    upsampling: float = 2.0
    threads: int = 0


@dataclass(frozen=True)
class DcfConfig:
    """Pipe-Menon density compensation iterations."""

    iterations: int = 10


@dataclass(frozen=True)
class ReconConfig:
    """All parameters of one reconstruction."""

    data: DataConfig = field(default_factory=DataConfig)
    coils: CoilConfig = field(default_factory=CoilConfig)
    nufft: NufftConfig = field(default_factory=NufftConfig)
    dcf: DcfConfig = field(default_factory=DcfConfig)

    def __post_init__(self) -> None:
        errors = _validate(self)
        if errors:
            raise ConfigError("invalid recon config:\n  " + "\n  ".join(errors))

    def to_dict(self) -> dict[str, Any]:
        """Plain dict of all values, in TOML layout."""
        return plain_dict(self)

    def digest(self) -> str:
        """Short SHA-256 of the canonical config, for tagging outputs."""
        return digest_of(self.to_dict())

    def to_toml(self) -> str:
        """TOML text that ``from_toml`` reads back into an equal config."""
        return toml_text(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReconConfig:
        """Build a config from TOML-layout data. Missing values take defaults."""
        return cls(**sections_from_dict(cls, data))

    @classmethod
    def from_toml(cls, text: str) -> ReconConfig:
        """Parse TOML text."""
        try:
            return cls.from_dict(tomllib.loads(text))
        except tomllib.TOMLDecodeError as e:
            raise ConfigError(f"invalid TOML: {e}") from e

    @classmethod
    def load(cls, path: Path) -> ReconConfig:
        """Read a TOML config file."""
        return cls.from_toml(path.read_text())

    def with_overrides(self, overrides: list[str]) -> ReconConfig:
        """Apply ``section.key=value`` overrides (values parsed as TOML)."""
        return ReconConfig.from_dict(apply_overrides(self.to_dict(), overrides))

    def check_design(self, design: Config) -> None:
        """Fail if the config asks for more than the design acquires."""
        if self.data.resolution_m < design.geometry.resolution_m * (1 - 1e-9):
            raise ConfigError(
                f"data.resolution_m = {self.data.resolution_m} is finer than the design "
                f"resolution {design.geometry.resolution_m}"
            )

    def matrix_size(self, design: Config) -> int:
        """Voxels along each axis: the design FOV at ``data.resolution_m``."""
        return max(1, round(design.geometry.fov_m / self.data.resolution_m))


def _validate(c: ReconConfig) -> list[str]:
    errors: list[str] = []
    d = c.data
    if not d.resolution_m > 0:
        errors.append(f"data.resolution_m must be > 0 (got {d.resolution_m})")
    if d.tau_min_s < 0:
        errors.append("data.tau_min_s must be >= 0")
    if d.tau_max_s != 0 and d.tau_max_s <= d.tau_min_s:
        errors.append("data.tau_max_s must be 0 (to the end) or > data.tau_min_s")
    if d.max_trs < 0:
        errors.append("data.max_trs must be >= 0")
    if d.batch_trs <= 0:
        errors.append("data.batch_trs must be > 0")
    if c.coils.virtual_coils < 0:
        errors.append("coils.virtual_coils must be >= 0")
    if c.coils.calibration_trs <= 0:
        errors.append("coils.calibration_trs must be > 0")
    if not 0 < c.nufft.eps < 1:
        errors.append("nufft.eps must be in (0, 1)")
    if c.nufft.upsampling not in (1.25, 2.0):
        errors.append("nufft.upsampling must be 1.25 or 2.0")
    if c.nufft.threads < 0:
        errors.append("nufft.threads must be >= 0")
    if c.dcf.iterations < 0:
        errors.append("dcf.iterations must be >= 0")
    return errors
