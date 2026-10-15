from pathlib import Path

import numpy as np
import pytest
from conftest import PHANTOM_DESIGN, PhantomScan
from numpy.typing import NDArray
from twix_helpers import noise_mixing

from relaxed_rosettes.cli import main
from relaxed_rosettes.config import ConfigError
from relaxed_rosettes.recon.config import ReconConfig
from relaxed_rosettes.recon.preprocess import Selection, noise_whitening
from relaxed_rosettes.recon.rawdata import RawData
from relaxed_rosettes.trajectory import load_trajectory

COILS = 4


def _raw_with_noise(noise: NDArray[np.complex64], dwell_s: float | None) -> RawData:
    return RawData(
        path=Path("unused.h5"),
        config=PHANTOM_DESIGN,
        seq_signature="",
        num_coils=noise.shape[0],
        tr_index=np.zeros(0, np.int64),
        shape_index=np.zeros(0, np.int64),
        time_stamp_s=np.zeros(0),
        noise=noise,
        noise_dwell_s=dwell_s,
        _rows=np.zeros(0, np.int64),
    )


def test_noise_whitening_makes_white_noise() -> None:
    rng = np.random.default_rng(0)
    mixing = noise_mixing(rng, COILS)
    white = rng.standard_normal((COILS, 200_000)) + 1j * rng.standard_normal((COILS, 200_000))
    noise = (mixing @ white).astype(np.complex64)
    whitening, covariance = noise_whitening(_raw_with_noise(noise, None), 3.2e-6)
    assert covariance is not None
    np.testing.assert_allclose(covariance, 2 * mixing @ mixing.conj().T, rtol=0.05, atol=0.05)
    whitened = whitening @ noise
    np.testing.assert_allclose(
        whitened @ whitened.conj().T / noise.shape[1], np.eye(COILS), atol=1e-6
    )
    # Noise at twice the ADC dwell has half the variance per sample at the ADC dwell.
    _, scaled = noise_whitening(_raw_with_noise(noise, 6.4e-6), 3.2e-6)
    assert scaled is not None
    np.testing.assert_allclose(scaled, 2 * covariance, rtol=1e-6)
    identity, none = noise_whitening(_raw_with_noise(noise[:, :0], None), 3.2e-6)
    assert none is None
    np.testing.assert_array_equal(identity, np.eye(COILS))


def test_selection_window_and_ball(phantom_scan: PhantomScan) -> None:
    design, trajectory = load_trajectory(phantom_scan.traj_path)
    recon = ReconConfig().with_overrides(
        ["data.resolution_m=0.008", "data.tau_min_s=0.002", "data.tau_max_s=0.006"]
    )
    selection = Selection.make(recon, design, trajectory, num_trs=3)
    tau = trajectory.time_since_excitation_s[selection.samples]
    assert tau.min() >= 0.002 and tau.max() <= 0.006
    k, inside = selection.k(trajectory, selection.trs)
    assert np.all(np.linalg.norm(k, axis=1) <= 1 / (2 * 0.008))
    assert inside.shape == (3, len(selection.samples)) and len(k) == inside.sum()
    with pytest.raises(ConfigError, match="finer"):
        Selection.make(
            ReconConfig().with_overrides(["data.resolution_m=0.0005"]), design, trajectory, 3
        )


def test_grid_reconstructs_a_phantom(phantom_scan: PhantomScan, tmp_path: Path) -> None:
    scan = phantom_scan
    out = tmp_path / "grid.npz"
    args = ["grid", str(scan.raw_path), str(scan.traj_path), "-o", str(out), "--coil-images"]
    args += ["--set", "data.resolution_m=0.016", "--set", "coils.virtual_coils=2"]
    assert main(args) == 0

    n = round(scan.design.geometry.fov_m / 0.016)
    with np.load(out) as result:
        rss = result["rss"]
        assert rss.shape == (n, n, n)
        assert result["coil_images"].shape == (2, n, n, n)
        assert result["coil_matrix"].shape == (2, COILS)
        singular = result["singular_values"]
        assert singular[1] < 1e-2 * singular[0]
        assert int(result["num_trs"]) == scan.design.scan.num_trs
        assert str(result["design_config_digest"]) == scan.design.digest()

    truth = scan.truth(n, 0.016)
    correlation = np.sum(rss * truth) / (np.linalg.norm(rss) * np.linalg.norm(truth))
    assert correlation > 0.95
    axis = (np.arange(n) - n // 2) * 0.016
    peak = np.unravel_index(np.argmax(rss), rss.shape)
    assert np.allclose(axis[list(peak)], scan.center_m, atol=0.016)


def test_recon_config_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "recon.toml"
    assert main(["recon-config", "init", str(path)]) == 0
    assert main(["recon-config", "init", str(path)]) == 1
    capsys.readouterr()
    assert main(["recon-config", "show", str(path), "--set", "dcf.iterations=3"]) == 0
    out = capsys.readouterr().out
    assert "iterations = 3" in out
    assert ReconConfig.from_toml(out).dcf.iterations == 3
