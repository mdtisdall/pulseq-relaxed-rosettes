from pathlib import Path

import h5py
import numpy as np
import pytest
from conftest import PhantomScan
from numpy.typing import NDArray

from relaxed_rosettes.cli import main
from relaxed_rosettes.config import ConfigError
from relaxed_rosettes.recon.nufft import Nufft
from relaxed_rosettes.recon.phantom import (
    Blocks,
    Compartment,
    CompartmentShape,
    PhantomConfig,
    Property,
    Sine,
    fourier_transform,
    profile,
)
from relaxed_rosettes.recon.rawdata import PHANTOM_DIGEST, load_rawdata
from relaxed_rosettes.trajectory import load_trajectory

FloatArray = NDArray[np.float64]


def _ball(rng: np.random.Generator, n_points: int, radius: float) -> FloatArray:
    direction = rng.standard_normal((n_points, 3))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    points: FloatArray = direction * (radius * rng.uniform(0, 1, n_points) ** (1 / 3))[:, None]
    return points


def _grid(n: int, res: float) -> FloatArray:
    axis = (np.arange(n) - n // 2) * res
    r: FloatArray = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
    return r


def test_fourier_transform_matches_the_nufft_of_the_profile() -> None:
    """Center, rotation, and the sign of the phase agree with the recon's model."""
    rng = np.random.default_rng(0)
    n, res = 48, 0.005
    blob = Compartment(
        "blob",
        CompartmentShape.GAUSSIAN,
        center_m=(0.02, -0.015, 0.01),
        semi_axes_m=(0.02, 0.012, 0.016),
        rotation_deg=(20.0, -35.0, 50.0),
    )
    image = profile(blob, _grid(n, res)).reshape(1, n, n, n).astype(np.complex64)
    k = _ball(rng, 200, 1 / (2 * res))
    expected = Nufft(n=n, resolution_m=res, eps=1e-6).forward(image, k)[0] * res**3
    zero = abs(fourier_transform(blob, np.zeros((1, 3)))[0])
    np.testing.assert_allclose(fourier_transform(blob, k), expected, atol=1e-4 * zero)


def test_ellipsoid_transform() -> None:
    rng = np.random.default_rng(1)
    axes = (0.06, 0.04, 0.05)
    ellipsoid = Compartment("e", center_m=(0.01, 0.0, -0.02), semi_axes_m=axes)
    zero = fourier_transform(ellipsoid, np.zeros((1, 3)))[0]
    assert zero == pytest.approx(4 / 3 * np.pi * np.prod(axes))
    # Rotating by 90 degrees about z swaps the x and y semi-axes.
    k = _ball(rng, 100, 50.0)
    swapped = Compartment("s", center_m=(0.01, 0.0, -0.02), semi_axes_m=(0.04, 0.06, 0.05))
    rotated = Compartment(
        "r", center_m=(0.01, 0.0, -0.02), semi_axes_m=axes, rotation_deg=(0.0, 0.0, 90.0)
    )
    np.testing.assert_allclose(fourier_transform(rotated, k), fourier_transform(swapped, k))
    # Against a finely sampled indicator, at low k (the sampling error is O(voxel)).
    n, res = 96, 0.0025
    image = profile(ellipsoid, _grid(n, res)).reshape(1, n, n, n).astype(np.complex64)
    k = _ball(rng, 50, 20.0)
    expected = Nufft(n=n, resolution_m=res, eps=1e-6).forward(image, k)[0] * res**3
    np.testing.assert_allclose(fourier_transform(ellipsoid, k), expected, atol=0.02 * abs(zero))


def test_magnetization_decay_and_replacement() -> None:
    head = Compartment("head", semi_axes_m=(0.08, 0.08, 0.08), r2star_per_s=20.0)
    inner = Compartment(
        "inner",
        semi_axes_m=(0.02, 0.02, 0.02),
        density=2.0,
        r2star_per_s=5.0,
        off_resonance_hz=10.0,
        inside="head",
    )
    phantom = PhantomConfig(compartments=(head, inner), sines=(), blocks=())
    tau = 0.03
    image = phantom.image(20, 0.01, tau_s=tau)
    center, outside = image[10, 10, 10], image[10, 10, 16]
    assert center == pytest.approx(2.0 * np.exp(-5.0 * tau + 2j * np.pi * 10.0 * tau))
    assert outside == pytest.approx(np.exp(-20.0 * tau))
    # The signal at k = 0 is the integral of the image.
    volume = 4 / 3 * np.pi
    expected = np.exp(-20 * tau) * volume * (0.08**3 - 0.02**3) + center * volume * 0.02**3
    signal = phantom.signal(np.zeros((1, 3)), np.array([tau]), np.array([0.0]))[0]
    assert signal == pytest.approx(expected)


def test_modulations() -> None:
    blocks = Blocks("c", Property.R2STAR, 2.0, period_s=10.0, on_s=4.0, ramp_s=1.0, start_s=1.0)
    times = np.array([1.0, 1.5, 3.0, 5.5, 8.0, 11.5])
    np.testing.assert_allclose(blocks.value(times), [0.0, 1.0, 2.0, 1.0, 0.0, 1.0], atol=1e-12)
    sine = Sine("c", Property.DENSITY, 0.5, 2.0, np.pi / 2)
    np.testing.assert_allclose(sine.value(np.array([0.0, 0.25])), [0.5, -0.5], atol=1e-12)

    phantom = PhantomConfig(
        compartments=(Compartment("c", density=2.0),), sines=(sine,), blocks=(blocks,)
    )
    density, r2star, off_resonance = phantom.properties(phantom.compartments[0], times[:3])
    np.testing.assert_allclose(density, 2.0 * (1 + sine.value(times[:3])))
    np.testing.assert_allclose(r2star, 20.0 + blocks.value(times[:3]))
    np.testing.assert_allclose(off_resonance, 0.0)


def test_config_round_trip_and_validation(tmp_path: Path) -> None:
    default = PhantomConfig()
    path = tmp_path / "phantom.toml"
    path.write_text(default.to_toml())
    assert PhantomConfig.load(path) == default
    assert PhantomConfig.from_toml("") == default
    changed = PhantomConfig.from_toml(default.to_toml().replace("seed = 0", "seed = 3"))
    assert changed.seed == 3 and changed.digest() != default.digest()
    # Only compartments: no modulations.
    only = PhantomConfig.from_toml('[[compartment]]\nname = "a"\n')
    assert only.compartments == (Compartment("a"),) and not only.sines and not only.blocks

    bad = {
        "seed = 1.5\n": "seed",
        "nope = 1\n": "unknown key",
        '[[compartment]]\nname = "a"\nshape = "cube"\n': "shape must be one of",
        '[[compartment]]\nname = "a"\nsize = 1\n': "unknown key",
        '[[compartment]]\nname = "a"\ninside = "b"\n': "inside",
        '[[compartment]]\nname = "a"\n[[compartment]]\nname = "a"\n': "unique",
        '[[compartment]]\nname = "a"\ncenter_m = [0.0, 1.0]\n': "list of 3",
        '[[compartment]]\nname = "a"\n[[sine]]\ncompartment = "b"\nproperty = "density"\n'
        "amplitude = 1.0\nfrequency_hz = 1.0\n": "unknown compartment",
        '[[compartment]]\nname = "a"\n[[sine]]\ncompartment = "a"\n'
        'property = "density"\n': "missing",
        '[[compartment]]\nname = "a"\n[[blocks]]\ncompartment = "a"\nproperty = "r2star"\n'
        "amplitude = 1.0\nperiod_s = 10.0\non_s = 2.0\nramp_s = 3.0\n": "ramp_s",
        "coil_sensitivity_real = [1.0, 1.0]\n": "same length",
    }
    for text, message in bad.items():
        with pytest.raises(ConfigError, match=message):
            PhantomConfig.from_toml(text)


def test_phantom_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "phantom.toml"
    assert main(["phantom", "init", str(path)]) == 0
    assert main(["phantom", "init", str(path)]) == 1
    capsys.readouterr()
    assert main(["phantom", "show", str(path)]) == 0
    out = capsys.readouterr().out
    assert f"# phantom digest: {PhantomConfig().digest()}" in out
    assert PhantomConfig.from_toml(out) == PhantomConfig()


SIMULATED = """\
seed = 5
noise_relative = 1e-3
noise_readouts = 3
coil_sensitivity_real = [1.0, 0.0]
coil_sensitivity_imag = [0.0, 0.8]

[[compartment]]
name = "blob"
shape = "gaussian"
center_m = [0.02, -0.01, 0.0]
semi_axes_m = [0.03, 0.03, 0.03]
r2star_per_s = 30.0

[[blocks]]
compartment = "blob"
property = "density"
amplitude = 0.5
period_s = 0.2
on_s = 0.1
ramp_s = 0.02
"""


def _simulate(phantom_scan: PhantomScan, directory: Path) -> Path:
    phantom_path = directory / "phantom.toml"
    phantom_path.write_text(SIMULATED)
    raw = directory / "sim.h5"
    assert main(["simulate", str(phantom_path), str(phantom_scan.seq_path), "-o", str(raw)]) == 0
    return raw


def test_simulate_writes_the_phantom_signal(phantom_scan: PhantomScan, tmp_path: Path) -> None:
    raw_path = _simulate(phantom_scan, tmp_path)
    phantom = PhantomConfig.from_toml(SIMULATED)
    design, trajectory = load_trajectory(phantom_scan.traj_path)
    raw = load_rawdata(raw_path)
    raw.check_trajectory(design, trajectory)
    assert raw.num_coils == 2
    assert raw.noise.shape == (2, 3 * design.readout.adc_num_samples)
    assert raw.noise_dwell_s == pytest.approx(design.readout.adc_dwell_s)
    np.testing.assert_allclose(raw.time_stamp_s, trajectory.tr_start_s, atol=2.5e-3)

    data = raw.read()
    trs = np.arange(raw.num_trs)
    time = trajectory.tr_start_s[:, None] + trajectory.adc_time_in_tr_s[None, :]
    tau = np.broadcast_to(trajectory.time_since_excitation_s, time.shape)
    signal = phantom.signal(trajectory.k_per_m(trs).reshape(-1, 3), tau.ravel(), time.ravel())
    expected = phantom.sensitivities[None, :, None] * signal.reshape(len(trs), 1, -1)
    sigma = 1e-3 * phantom.reference_signal()
    residual = (data - expected).ravel()
    assert np.std(residual.real) == pytest.approx(sigma, rel=0.05)
    assert np.std(raw.noise.real) == pytest.approx(sigma, rel=0.1)
    with h5py.File(raw_path, "r") as f:
        xml = f["dataset/xml"][0].decode()
    assert PHANTOM_DIGEST in xml and phantom.digest() in xml

    # Reproducible from the phantom file.
    again = tmp_path / "again"
    again.mkdir()
    np.testing.assert_array_equal(load_rawdata(_simulate(phantom_scan, again)).read(), data)


def test_grid_reconstructs_the_simulated_phantom(phantom_scan: PhantomScan, tmp_path: Path) -> None:
    raw_path = _simulate(phantom_scan, tmp_path)
    out = tmp_path / "grid.npz"
    args = ["grid", str(raw_path), str(phantom_scan.traj_path), "-o", str(out)]
    args += ["--set", "data.resolution_m=0.016", "--set", "coils.virtual_coils=2"]
    assert main(args) == 0
    n = round(phantom_scan.design.geometry.fov_m / 0.016)
    with np.load(out) as result:
        rss = result["rss"]
    truth = np.abs(PhantomConfig.from_toml(SIMULATED).image(n, 0.016))
    correlation = np.sum(rss * truth) / (np.linalg.norm(rss) * np.linalg.norm(truth))
    assert correlation > 0.95
