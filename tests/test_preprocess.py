from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray
from twix_helpers import NOISE_DWELL_NS, build_design, write_meas

from relaxed_rosettes.cli import main
from relaxed_rosettes.config import Config, ConfigError
from relaxed_rosettes.recon.config import ReconConfig
from relaxed_rosettes.recon.preprocess import Selection, noise_whitening
from relaxed_rosettes.recon.rawdata import RawData
from relaxed_rosettes.sequence import check_adc_segments, read_seq_signature
from relaxed_rosettes.system import make_opts
from relaxed_rosettes.trajectory import load_trajectory

DESIGN = Config().with_overrides(
    [
        "readout.tr_s=0.01",
        "readout.adc_num_samples=2000",
        "shapes.echoes_per_tr=4",
        "shapes.num_shapes=2",
        "scan.num_trs=40",
        "scan.num_dummy_trs=2",
    ]
)
COILS = 4
CENTER_M = np.array([0.02, -0.01, 0.0])
SIGMA_M = 0.03


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """The .seq file (design .npz next to it) and trajectory of DESIGN."""
    return build_design(tmp_path_factory.mktemp("built"), DESIGN)


def _raw_with_noise(noise: NDArray[np.complex64], dwell_s: float | None) -> RawData:
    return RawData(
        path=Path("unused.h5"),
        config=DESIGN,
        seq_signature="",
        num_coils=noise.shape[0],
        tr_index=np.zeros(0, np.int64),
        shape_index=np.zeros(0, np.int64),
        time_stamp_s=np.zeros(0),
        noise=noise,
        noise_dwell_s=dwell_s,
        _rows=np.zeros(0, np.int64),
    )


def _mixing(rng: np.random.Generator) -> NDArray[np.complex128]:
    random = rng.standard_normal((COILS, COILS)) + 1j * rng.standard_normal((COILS, COILS))
    mixing: NDArray[np.complex128] = np.eye(COILS, dtype=np.complex128) + 0.4 * random
    return mixing


def test_noise_whitening_makes_white_noise() -> None:
    rng = np.random.default_rng(0)
    mixing = _mixing(rng)
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


def test_selection_window_and_ball(built: tuple[Path, Path]) -> None:
    design, trajectory = load_trajectory(built[1])
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


def test_grid_reconstructs_a_phantom(built: tuple[Path, Path], tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    seq_path, traj_path = built
    design, trajectory = load_trajectory(traj_path)

    # A Gaussian blob seen by coils with constant complex sensitivities, and
    # correlated noise.
    k = trajectory.k_per_m()
    signal = (2 * np.pi * SIGMA_M**2) ** 1.5 * np.exp(
        -2 * np.pi**2 * SIGMA_M**2 * np.sum(k**2, axis=-1) + 2j * np.pi * k @ CENTER_M
    )
    sensitivity = np.array([1.0, 0.8j, -0.5, 0.3 - 0.3j])
    mixing = _mixing(rng)
    scale = 1e-6
    shape = (COILS, *k.shape[:2])
    white = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    data = sensitivity[:, None, None] * signal[None]
    data = data + scale * np.einsum("dc,cts->dts", mixing, white)
    data = data.transpose(1, 0, 2).astype(np.complex64)
    noise_white = rng.standard_normal((2, COILS, 512)) + 1j * rng.standard_normal((2, COILS, 512))
    # The noise scans are at NOISE_DWELL_NS, so their variance is scaled to match.
    noise_scale = scale * np.sqrt(design.readout.adc_dwell_s / (NOISE_DWELL_NS * 1e-9))
    noise = (noise_scale * np.einsum("dc,ncs->nds", mixing, noise_white)).astype(np.complex64)

    meas = tmp_path / "meas.dat"
    signature = read_seq_signature(seq_path)
    segments, _ = check_adc_segments(design, make_opts(design))
    write_meas(meas, signature, data, noise, segments=segments, tr_s=design.readout.tr_s)
    raw_path = tmp_path / "raw.h5"
    assert main(["to-ismrmrd", str(meas), str(seq_path), "-o", str(raw_path)]) == 0

    out = tmp_path / "grid.npz"
    args = ["grid", str(raw_path), str(traj_path), "-o", str(out), "--coil-images"]
    args += ["--set", "data.resolution_m=0.016", "--set", "coils.virtual_coils=2"]
    assert main(args) == 0

    with np.load(out) as result:
        rss = result["rss"]
        n = round(design.geometry.fov_m / 0.016)
        assert rss.shape == (n, n, n)
        assert result["coil_images"].shape == (2, n, n, n)
        assert result["coil_matrix"].shape == (2, COILS)
        singular = result["singular_values"]
        assert singular[1] < 1e-2 * singular[0]
        assert int(result["num_trs"]) == DESIGN.scan.num_trs
        assert str(result["design_config_digest"]) == design.digest()

    axis = (np.arange(n) - n // 2) * 0.016
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    r2 = (x - CENTER_M[0]) ** 2 + (y - CENTER_M[1]) ** 2 + (z - CENTER_M[2]) ** 2
    truth = np.exp(-r2 / (2 * SIGMA_M**2))
    correlation = np.sum(rss * truth) / (np.linalg.norm(rss) * np.linalg.norm(truth))
    assert correlation > 0.95
    peak = np.unravel_index(np.argmax(rss), rss.shape)
    assert np.allclose(axis[list(peak)], CENTER_M, atol=0.016)


def test_recon_config_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "recon.toml"
    assert main(["recon-config", "init", str(path)]) == 0
    assert main(["recon-config", "init", str(path)]) == 1
    capsys.readouterr()
    assert main(["recon-config", "show", str(path), "--set", "dcf.iterations=3"]) == 0
    out = capsys.readouterr().out
    assert "iterations = 3" in out
    assert ReconConfig.from_toml(out).dcf.iterations == 3
