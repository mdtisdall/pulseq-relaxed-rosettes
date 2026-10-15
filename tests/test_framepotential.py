import json
from pathlib import Path

import numpy as np
import pytest
from conftest import PhantomScan
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from relaxed_rosettes.cli import main
from relaxed_rosettes.config import ConfigError
from relaxed_rosettes.recon.basis import decay_basis
from relaxed_rosettes.recon.framepotential import (
    Requirement,
    RequirementsConfig,
    Taper,
    frame_potential,
)
from relaxed_rosettes.trajectory import Trajectory

FloatArray = NDArray[np.float64]
FOV = 0.2
RES = 0.05


def _trajectory(rng: np.random.Generator, *, rotate: bool = True, trs: int = 6) -> Trajectory:
    """A tiny trajectory: 2 shapes of 8 samples, some outside the ball of RES."""
    samples = 8
    k_max = 1 / (2 * RES)
    direction = rng.standard_normal((2, samples, 3))
    direction /= np.linalg.norm(direction, axis=-1, keepdims=True)
    k_shape = direction * rng.uniform(0, 1.2 * k_max, (2, samples, 1))
    adc_time = np.linspace(0.001, 0.008, samples)
    rotations = (
        Rotation.random(trs, random_state=1).as_matrix()
        if rotate
        else np.tile(np.eye(3), (trs, 1, 1))
    )
    return Trajectory(
        k_shape_per_m=k_shape,
        gg_shape_T2s_per_m2=np.zeros((2, samples, 3, 3)),
        adc_time_in_tr_s=adc_time,
        time_since_excitation_s=adc_time - 0.0005,
        tr_index=np.arange(trs),
        tr_start_s=0.05 * np.arange(trs, dtype=np.float64),
        shape_index=np.arange(trs) % 2,
        rotation_matrix=rotations,
        rf_phase_rad=np.zeros(trs),
    )


def _explicit(requirement: Requirement, trajectory: Trajectory) -> tuple[float, float, int]:
    """FP, trace and N from the explicit forward matrix."""
    n = round(FOV / RES)
    k_max = 1 / (2 * RES)
    axis = (np.arange(n) - n // 2) * RES
    r = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
    tau = trajectory.time_since_excitation_s
    if requirement.tau_rank:
        phi = decay_basis(
            tau,
            requirement.r2star_min_per_s,
            requirement.r2star_max_per_s,
            rank=requirement.tau_rank,
        ).functions.T
    else:
        phi = np.ones((len(tau), 1))
    k = trajectory.k_per_m().reshape(-1, 3)
    time = (trajectory.tr_start_s[:, None] + trajectory.adc_time_in_tr_s[None]).ravel()
    sample = np.tile(np.arange(len(tau)), len(trajectory.tr_index))
    width = requirement.scan_time_width_s
    if width:
        num = int(np.floor((time.max() - time.min()) / width)) + 2
        knots = time.min() + width * np.arange(num)
        hats = np.maximum(1 - np.abs(time[:, None] - knots[None]) / width, 0.0)
    else:
        hats = np.ones((len(time), 1))
    radius = np.linalg.norm(k, axis=1)
    keep = radius <= k_max
    taper = np.ones(len(k))
    if requirement.taper is Taper.HANN:
        taper = 0.5 * (1 + np.cos(np.pi * radius / k_max))
    rows = (
        taper[keep, None, None, None]
        * np.exp(2j * np.pi * k[keep] @ r.T)[:, :, None, None]
        * phi[sample[keep]][:, None, :, None]
        * hats[keep][:, None, None, :]
    ).reshape(int(keep.sum()), -1)
    gram = rows @ rows.conj().T
    return float(np.sum(np.abs(gram) ** 2)), float(np.real(np.trace(gram))), rows.shape[1]


@pytest.mark.parametrize(
    "requirement",
    [
        Requirement("static", RES),
        Requirement("tau", RES, tau_rank=2, taper=Taper.HANN),
        Requirement("time", RES, scan_time_width_s=0.12),
        Requirement("both", RES, tau_rank=2, scan_time_width_s=0.07, taper=Taper.HANN),
    ],
)
def test_matches_the_explicit_frame_potential(requirement: Requirement) -> None:
    trajectory = _trajectory(np.random.default_rng(0))
    result = frame_potential(requirement, trajectory, FOV, eps=1e-6, batch_trs=4)
    fp, trace, coefficients = _explicit(requirement, trajectory)
    assert result.coefficients == coefficients
    assert result.frame_potential == pytest.approx(fp, rel=1e-4)
    assert result.trace == pytest.approx(trace, rel=1e-5)
    assert result.ratio == pytest.approx(coefficients * fp / trace**2, rel=1e-4)
    k = trajectory.k_per_m().reshape(-1, 3)
    assert result.samples == int(np.sum(np.linalg.norm(k, axis=1) <= 1 / (2 * RES)))
    assert result.ratio >= 1


def test_repeating_the_same_samples_is_worse_conditioned() -> None:
    requirement = Requirement("static", RES)
    spread = frame_potential(requirement, _trajectory(np.random.default_rng(2), trs=40), FOV)
    repeated = frame_potential(
        requirement, _trajectory(np.random.default_rng(2), rotate=False, trs=40), FOV
    )
    assert repeated.ratio > 1.2 * spread.ratio


def test_requirements_round_trip_and_validation(tmp_path: Path) -> None:
    default = RequirementsConfig()
    assert [r.name for r in default.requirements] == [
        "navigator",
        "relaxation_4d",
        "static",
        "bold",
    ]
    path = tmp_path / "req.toml"
    path.write_text(default.to_toml())
    assert RequirementsConfig.load(path) == default
    assert RequirementsConfig.from_toml("") == default
    custom = RequirementsConfig.from_toml(
        'eps = 1e-5\n[[requirement]]\nname = "a"\nresolution_m = 0.004\ntaper = "hann"\n'
    )
    assert custom.eps == 1e-5
    assert custom.requirements == (Requirement("a", 0.004, taper=Taper.HANN),)
    assert custom.digest() != default.digest()
    bad = {
        "eps = 2.0\n": "eps",
        "nope = 1\n": "unknown key",
        '[[requirement]]\nname = "a"\n': "missing",
        '[[requirement]]\nname = "a"\nresolution_m = 0.004\ntaper = "box"\n': "taper must be",
        '[[requirement]]\nname = "a"\nresolution_m = 0.004\ntau_rank = 1.5\n': "tau_rank",
        '[[requirement]]\nname = "a"\nresolution_m = 0.004\n'
        '[[requirement]]\nname = "a"\nresolution_m = 0.008\n': "unique",
        '[[requirement]]\nname = "a"\nresolution_m = 0.004\nscan_time_width_s = -1.0\n': "width",
    }
    for text, message in bad.items():
        with pytest.raises(ConfigError, match=message):
            RequirementsConfig.from_toml(text)


REQUIREMENTS = """\
[[requirement]]
name = "coarse"
resolution_m = 0.016

[[requirement]]
name = "coarse_tau_time"
resolution_m = 0.016
tau_rank = 2
scan_time_width_s = 0.1
taper = "hann"
"""


def test_frame_potential_cli(
    phantom_scan: PhantomScan, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "req.toml"
    assert main(["requirements", "init", str(path)]) == 0
    assert main(["requirements", "init", str(path)]) == 1
    capsys.readouterr()
    assert main(["requirements", "show", str(path)]) == 0
    assert f"# requirements digest: {RequirementsConfig().digest()}" in capsys.readouterr().out

    path.write_text(REQUIREMENTS)
    seq = str(phantom_scan.seq_path)
    assert main(["frame-potential", seq, "-r", str(path), "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["config_digest"] == phantom_scan.design.digest()
    names = [r["name"] for r in report["requirements"]]
    assert names == ["coarse", "coarse_tau_time"]
    for r in report["requirements"]:
        assert r["ratio"] >= 1 and r["samples"] > 0
    assert main(["frame-potential", seq, "-r", str(path), "--only", "coarse"]) == 0
    out = capsys.readouterr().out
    assert "coarse" in out and "coarse_tau_time" not in out
    assert main(["frame-potential", seq, "-r", str(path), "--only", "nope"]) == 1
