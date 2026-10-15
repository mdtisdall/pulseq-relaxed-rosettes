"""Shared fixtures: a small design and a synthetic phantom scan of it."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray
from twix_helpers import NOISE_DWELL_NS, build_design, noise_mixing, write_meas

from relaxed_rosettes.cli import main
from relaxed_rosettes.config import Config
from relaxed_rosettes.sequence import check_adc_segments, read_seq_signature
from relaxed_rosettes.system import make_opts
from relaxed_rosettes.trajectory import load_trajectory

PHANTOM_DESIGN = Config().with_overrides(
    [
        "readout.tr_s=0.01",
        "readout.adc_num_samples=2000",
        "shapes.echoes_per_tr=4",
        "shapes.num_shapes=2",
        "scan.num_trs=40",
        "scan.num_dummy_trs=2",
    ]
)


@dataclass(frozen=True)
class PhantomScan:
    """A Gaussian blob, seen by coils with constant complex sensitivities.

    Its off-resonance is uniform, and the coil noise is correlated.
    """

    design: Config
    seq_path: Path
    traj_path: Path
    raw_path: Path
    center_m: NDArray[np.float64]
    sigma_m: float
    off_resonance_hz: float
    sensitivity: NDArray[np.complex128]
    mixing: NDArray[np.complex128]

    def truth(self, n: int, resolution_m: float) -> NDArray[np.float64]:
        """The blob on an n³ grid of the reconstruction's axes, peak 1."""
        axis = (np.arange(n) - n // 2) * resolution_m
        x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
        c = self.center_m
        r2 = (x - c[0]) ** 2 + (y - c[1]) ** 2 + (z - c[2]) ** 2
        blob: NDArray[np.float64] = np.exp(-r2 / (2 * self.sigma_m**2))
        return blob


@pytest.fixture(scope="session")
def phantom_scan(tmp_path_factory: pytest.TempPathFactory) -> PhantomScan:
    """PHANTOM_DESIGN built through the CLI, and a phantom scan converted to ISMRMRD."""
    directory = tmp_path_factory.mktemp("phantom")
    seq_path, traj_path = build_design(directory, PHANTOM_DESIGN)
    design, trajectory = load_trajectory(traj_path)
    rng = np.random.default_rng(1)
    center = np.array([0.02, -0.01, 0.0])
    sigma = 0.03
    off_resonance = 20.0
    sensitivity = np.array([1.0, 0.8j, -0.5, 0.3 - 0.3j])
    coils = len(sensitivity)
    mixing = noise_mixing(rng, coils)

    k = trajectory.k_per_m()
    tau = trajectory.time_since_excitation_s
    signal = (2 * np.pi * sigma**2) ** 1.5 * np.exp(
        -2 * np.pi**2 * sigma**2 * np.sum(k**2, axis=-1)
        + 2j * np.pi * (k @ center + off_resonance * tau[None])
    )
    scale = 1e-6
    shape = (coils, *k.shape[:2])
    white = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    data = sensitivity[:, None, None] * signal[None]
    data = data + scale * np.einsum("dc,cts->dts", mixing, white)
    data = data.transpose(1, 0, 2).astype(np.complex64)
    noise_white = rng.standard_normal((2, coils, 512)) + 1j * rng.standard_normal((2, coils, 512))
    # The noise scans are at NOISE_DWELL_NS, so their variance is scaled to match.
    noise_scale = scale * np.sqrt(design.readout.adc_dwell_s / (NOISE_DWELL_NS * 1e-9))
    noise = (noise_scale * np.einsum("dc,ncs->nds", mixing, noise_white)).astype(np.complex64)

    meas = directory / "meas.dat"
    segments, _ = check_adc_segments(design, make_opts(design))
    write_meas(
        meas, read_seq_signature(seq_path), data, noise, segments=segments, tr_s=design.readout.tr_s
    )
    raw_path = directory / "raw.h5"
    assert main(["to-ismrmrd", str(meas), str(seq_path), "-o", str(raw_path)]) == 0
    return PhantomScan(
        design=design,
        seq_path=seq_path,
        traj_path=traj_path,
        raw_path=raw_path,
        center_m=center,
        sigma_m=sigma,
        off_resonance_hz=off_resonance,
        sensitivity=sensitivity,
        mixing=mixing,
    )
