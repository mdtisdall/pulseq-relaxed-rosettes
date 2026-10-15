from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from relaxed_rosettes.config import ConfigError
from relaxed_rosettes.recon.config import ReconConfig
from relaxed_rosettes.recon.gridding import grid, pipe_menon_dcf, root_sum_of_squares
from relaxed_rosettes.recon.nufft import Nufft

FloatArray = NDArray[np.float64]


def _ball(rng: np.random.Generator, n_points: int, radius: float) -> FloatArray:
    """Points uniform in a ball."""
    direction = rng.standard_normal((n_points, 3))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    r = radius * rng.uniform(0, 1, n_points) ** (1 / 3)
    points: FloatArray = direction * r[:, None]
    return points


def _blob_signal(k: FloatArray, center_m: FloatArray, sigma_m: float, voxel_m: float) -> FloatArray:
    """Signal of a Gaussian blob of peak 1, in the discrete model of ``Nufft``."""
    phase = 2 * np.pi * k @ center_m
    amplitude = (2 * np.pi * sigma_m**2) ** 1.5 / voxel_m**3
    signal: FloatArray = amplitude * np.exp(-2 * np.pi**2 * sigma_m**2 * np.sum(k**2, axis=1))
    return signal * np.exp(1j * phase)


def _blob_image(n: int, voxel_m: float, center_m: FloatArray, sigma_m: float) -> FloatArray:
    axis = (np.arange(n) - n // 2) * voxel_m
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    r2 = (x - center_m[0]) ** 2 + (y - center_m[1]) ** 2 + (z - center_m[2]) ** 2
    image: FloatArray = np.exp(-r2 / (2 * sigma_m**2))
    return image


def test_recon_config_round_trip(tmp_path: Path) -> None:
    config = ReconConfig().with_overrides(["data.resolution_m=0.004", "coils.virtual_coils=8"])
    path = tmp_path / "recon.toml"
    path.write_text(config.to_toml())
    assert ReconConfig.load(path) == config
    assert config.digest() != ReconConfig().digest()
    with pytest.raises(ConfigError, match="upsampling"):
        ReconConfig().with_overrides(["nufft.upsampling=1.5"])
    with pytest.raises(ConfigError, match="unknown key"):
        ReconConfig().with_overrides(["data.nope=1"])


def test_nufft_matches_direct_sum_and_is_adjoint() -> None:
    rng = np.random.default_rng(0)
    n, res = 6, 0.01
    nufft = Nufft(n=n, resolution_m=res, eps=1e-6)
    k = _ball(rng, 50, 1 / (2 * res))
    image = (rng.standard_normal((2, n, n, n)) + 1j * rng.standard_normal((2, n, n, n))).astype(
        np.complex64
    )
    axis = (np.arange(n) - n // 2) * res
    r = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
    encoding = np.exp(2j * np.pi * k @ r.T)  # (points, voxels)
    expected = image.reshape(2, -1) @ encoding.T
    np.testing.assert_allclose(nufft.forward(image, k), expected, rtol=1e-4, atol=1e-4)

    values = (rng.standard_normal((2, 50)) + 1j * rng.standard_normal((2, 50))).astype(np.complex64)
    lhs = np.vdot(nufft.forward(image, k), values)
    rhs = np.vdot(image, nufft.adjoint(values, k))
    assert lhs == pytest.approx(rhs, rel=1e-4)


def test_dcf_and_gridding_recover_a_blob() -> None:
    rng = np.random.default_rng(1)
    n, res = 24, 0.01
    nufft = Nufft(n=n, resolution_m=res, eps=1e-5)
    # Denser near the center, like the rosettes: half uniform in the ball,
    # half in the inner third.
    k_max = 1 / (2 * res)
    k = np.concatenate([_ball(rng, 60_000, k_max), _ball(rng, 60_000, k_max / 3)])
    center = np.array([0.03, -0.02, 0.01])
    sigma = 0.025
    signal = _blob_signal(k, center, sigma, res).astype(np.complex64)
    batches = [k[:50_000], k[50_000:]]
    values = [signal[None, :50_000], signal[None, 50_000:]]

    def k_batches() -> Iterator[FloatArray]:
        yield from batches

    dcf = pipe_menon_dcf(nufft, k_batches, iterations=15)
    assert dcf.change[-1] < dcf.change[0]
    image = grid(nufft, zip(batches, values, strict=True), dcf, channels=1)[0]
    truth = _blob_image(n, res, center, sigma)
    error = np.linalg.norm(image - truth) / np.linalg.norm(truth)
    assert error < 0.1
    assert np.unravel_index(np.argmax(np.abs(image)), image.shape) == np.unravel_index(
        np.argmax(truth), truth.shape
    )
    np.testing.assert_allclose(root_sum_of_squares(image[None]), np.abs(image), rtol=1e-6)
