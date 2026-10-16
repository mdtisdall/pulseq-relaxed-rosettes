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
    tau_averaging,
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


def _line_trajectory(tau_order: str, trs: int = 12) -> Trajectory:
    """One straight shape through k = 0, along x, played with rotations about z.

    ``tau_order`` sets how τ runs along the line: "out" (center early, edge
    late), or "fixed" (every sample at one k, so each cell sees every τ).
    """
    samples = 16
    k_max = 1 / (2 * RES)
    if tau_order == "fixed":
        k_line = np.full(samples, 0.5 * k_max)
    else:
        k_line = np.linspace(0, 0.95 * k_max, samples)
    k_shape = np.zeros((1, samples, 3))
    k_shape[0, :, 0] = k_line
    adc_time = np.linspace(0.001, 0.04, samples)
    angles = 2 * np.pi * np.arange(trs) / trs
    return Trajectory(
        k_shape_per_m=k_shape,
        gg_shape_T2s_per_m2=np.zeros((1, samples, 3, 3)),
        adc_time_in_tr_s=adc_time,
        time_since_excitation_s=adc_time - 0.0005,
        tr_index=np.arange(trs),
        tr_start_s=0.05 * np.arange(trs, dtype=np.float64),
        shape_index=np.zeros(trs, dtype=np.int64),
        rotation_matrix=Rotation.from_euler("z", angles[:, None]).as_matrix(),
        rf_phase_rad=np.zeros(trs),
    )


def _explicit_mismatch(requirement: Requirement, trajectory: Trajectory, size: float) -> float:
    """The τ-averaging mismatch by direct loops over cells."""
    n = round(FOV / RES)
    basis = decay_basis(
        trajectory.time_since_excitation_s,
        requirement.r2star_min_per_s,
        requirement.r2star_max_per_s,
        rank=2,
    )
    energy = basis.singular_values[:2] ** 2
    readout_mean = basis.functions.mean(axis=1)
    k = trajectory.k_per_m()
    cells: dict[tuple[int, ...], list[FloatArray]] = {}
    for tr in range(k.shape[0]):
        for s in range(k.shape[1]):
            if np.linalg.norm(k[tr, s]) <= 1 / (2 * RES):
                cell = tuple(np.clip(np.rint(k[tr, s] * FOV).astype(int) + n // 2, 0, n - 1))
                cells.setdefault(cell, []).append(basis.functions[:, s])
    numerator = denominator = 0.0
    for cell, values in cells.items():
        radius = np.linalg.norm(np.array(cell) - n // 2) / FOV
        prior = 1 / (1 + (radius * size) ** 4)
        difference = np.mean(values, axis=0) - readout_mean
        numerator += prior * float(energy @ difference**2)
        denominator += prior
    return numerator / (denominator * float(energy @ readout_mean**2))


def test_tau_averaging_matches_the_explicit_mismatch() -> None:
    requirement = Requirement("static", RES)
    trajectory = _trajectory(np.random.default_rng(3), trs=10)
    result = tau_averaging(requirement, trajectory, FOV, object_size_m=0.1, batch_trs=3)
    assert result.mismatch == pytest.approx(
        _explicit_mismatch(requirement, trajectory, 0.1), rel=1e-10
    )
    k = trajectory.k_per_m().reshape(-1, 3)
    assert result.samples == int(np.sum(np.linalg.norm(k, axis=1) <= 1 / (2 * RES)))


def test_tau_averaging_is_zero_when_every_cell_sees_every_tau() -> None:
    requirement = Requirement("static", RES)
    fixed = tau_averaging(requirement, _line_trajectory("fixed"), FOV)
    assert fixed.mismatch == pytest.approx(0, abs=1e-12)
    outward = tau_averaging(requirement, _line_trajectory("out"), FOV)
    assert outward.mismatch > 1e-3
    # A steeper prior weights the early-τ center more, so the mismatch grows.
    steep = tau_averaging(requirement, _line_trajectory("out"), FOV, object_size_m=1.0)
    assert steep.mismatch > outward.mismatch


def test_tau_averaging_needs_an_averaged_model() -> None:
    trajectory = _line_trajectory("out")
    for requirement in (
        Requirement("tau", RES, tau_rank=2),
        Requirement("time", RES, scan_time_width_s=0.1),
    ):
        with pytest.raises(ValueError, match="does not average"):
            tau_averaging(requirement, trajectory, FOV)


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
    prior = RequirementsConfig.from_toml("object_size_m = 0.3\nobject_spectrum_power = 3.0\n")
    assert (prior.object_size_m, prior.object_spectrum_power) == (0.3, 3.0)
    assert prior.digest() != default.digest()
    assert custom.requirements == (Requirement("a", 0.004, taper=Taper.HANN),)
    assert custom.digest() != default.digest()
    bad = {
        "eps = 2.0\n": "eps",
        "object_size_m = 0.0\n": "object_size_m",
        "object_spectrum_power = -1.0\n": "object_spectrum_power",
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
    mismatch = [r["tau_averaging_mismatch"] for r in report["requirements"]]
    assert mismatch[0] is not None and mismatch[0] >= 0 and mismatch[1] is None
    assert main(["frame-potential", seq, "-r", str(path), "--only", "coarse"]) == 0
    out = capsys.readouterr().out
    assert "coarse" in out and "coarse_tau_time" not in out
    assert "# tau averaging" in out
    assert main(["frame-potential", seq, "-r", str(path), "--only", "nope"]) == 1
