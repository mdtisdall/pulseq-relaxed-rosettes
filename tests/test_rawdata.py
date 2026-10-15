from pathlib import Path

import h5py
import ismrmrd
import ismrmrd.xsd
import numpy as np
import pytest
from numpy.typing import NDArray
from twix_helpers import NOISE_DWELL_NS, build_design, write_meas

from relaxed_rosettes.cli import main
from relaxed_rosettes.config import Config
from relaxed_rosettes.recon.rawdata import load_rawdata
from relaxed_rosettes.recon.twix import read_twix
from relaxed_rosettes.sequence import read_seq_signature
from relaxed_rosettes.trajectory import Trajectory, load_trajectory

# Two ADC segments of 1000 samples in each TR.
SMALL = Config().with_overrides(
    [
        "readout.tr_s=0.01",
        "readout.adc_num_samples=2000",
        "system.adc_samples_limit=1000",
        "shapes.echoes_per_tr=4",
        "shapes.num_shapes=2",
        "scan.num_trs=5",
        "scan.num_dummy_trs=2",
    ]
)
COILS = 3
SEGMENT = 1000


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """A .seq file with its design file, and the exported trajectory."""
    return build_design(tmp_path_factory.mktemp("built"), SMALL)


def _write_meas(
    path: Path, signature: str, num_trs: int, rng: np.random.Generator, segment: int = SEGMENT
) -> tuple[NDArray[np.complex64], NDArray[np.complex64]]:
    """A meas.dat file of random data, with two noise readouts; returns (data, noise)."""
    noise = rng.standard_normal((2, COILS, 256)) + 1j * rng.standard_normal((2, COILS, 256))
    noise = noise.astype(np.complex64)
    shape = (num_trs, COILS, 2 * segment)
    data = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(np.complex64)
    write_meas(path, signature, data, noise, segments=2, tr_s=SMALL.readout.tr_s)
    return data, noise


def _convert(meas: Path, seq: Path, out: Path, *extra: str) -> int:
    return main(["to-ismrmrd", str(meas), str(seq), "-o", str(out), *extra])


def test_read_seq_signature_matches_the_file(built: tuple[Path, Path], tmp_path: Path) -> None:
    seq_path, _ = built
    signature = read_seq_signature(seq_path)
    assert len(signature) == 32
    changed = tmp_path / "changed.seq"
    changed.write_bytes(seq_path.read_bytes().replace(b"[VERSION]", b"[VERSION] ", 1))
    with pytest.raises(ValueError, match="does not match"):
        read_seq_signature(changed)


def test_to_ismrmrd_round_trip(built: tuple[Path, Path], tmp_path: Path) -> None:
    seq_path, traj_path = built
    config, trajectory = load_trajectory(traj_path)
    signature = read_seq_signature(seq_path)
    rng = np.random.default_rng(0)
    meas = tmp_path / "meas.dat"
    data, noise = _write_meas(meas, signature, SMALL.scan.num_trs, rng)
    out = tmp_path / "raw.h5"
    assert _convert(meas, seq_path, out) == 0

    raw = load_rawdata(out)
    raw.check_trajectory(config, trajectory)
    assert raw.config == config
    assert raw.seq_signature == signature
    assert raw.num_coils == COILS
    np.testing.assert_array_equal(raw.tr_index, trajectory.tr_index)
    np.testing.assert_array_equal(raw.shape_index, trajectory.shape_index)
    np.testing.assert_array_equal(raw.read(), data)
    np.testing.assert_array_equal(raw.read(np.array([3, 1])), data[[3, 1]])
    np.testing.assert_array_equal(raw.noise, np.concatenate(list(noise), axis=1))
    assert raw.noise_dwell_s == pytest.approx(NOISE_DWELL_NS * 1e-9)
    np.testing.assert_allclose(np.diff(raw.time_stamp_s), SMALL.readout.tr_s)

    with ismrmrd.Dataset(str(out), "dataset", mode="r") as dataset:
        header = ismrmrd.xsd.CreateFromDocument(dataset.read_xml_header())
        assert header.experimentalConditions.H1resonanceFrequency_Hz == 123200000
        assert header.acquisitionSystemInformation.systemFieldStrength_T == pytest.approx(2.89)
        assert header.encoding[0].encodedSpace.matrixSize.x == 256
        acq = dataset.read_acquisition(dataset.number_of_acquisitions() - 1)
        assert acq.is_flag_set(ismrmrd.ACQ_LAST_IN_MEASUREMENT)
        assert acq.sample_time_us == pytest.approx(SMALL.readout.adc_dwell_s * 1e6)
        assert acq.physiology_time_stamp[0] == acq.acquisition_time_stamp + 7
        np.testing.assert_allclose(acq.position, [0.0, 0.0, 12.5])
        k = trajectory.k_per_m(np.array([SMALL.scan.num_trs - 1]))[0] * SMALL.geometry.fov_m
        np.testing.assert_allclose(acq.traj, k, rtol=1e-6, atol=1e-4)
        # As in siemens_to_ismrmrd, the identity quaternion has phase along x.
        np.testing.assert_allclose(acq.read_dir, [0.0, 1.0, 0.0], atol=1e-7)
        np.testing.assert_allclose(acq.phase_dir, [1.0, 0.0, 0.0], atol=1e-7)
        np.testing.assert_allclose(acq.slice_dir, [0.0, 0.0, 1.0], atol=1e-7)


def test_to_ismrmrd_without_trajectory(built: tuple[Path, Path], tmp_path: Path) -> None:
    seq_path, _ = built
    meas = tmp_path / "meas.dat"
    _write_meas(meas, read_seq_signature(seq_path), SMALL.scan.num_trs, np.random.default_rng(1))
    out = tmp_path / "raw.h5"
    assert _convert(meas, seq_path, out, "--no-trajectory") == 0
    with h5py.File(out, "r") as f:
        heads = f["dataset/data"].fields("head")[:]
    assert np.all(heads["trajectory_dimensions"] == 0)


def test_to_ismrmrd_signature_check(built: tuple[Path, Path], tmp_path: Path) -> None:
    seq_path, _ = built
    meas = tmp_path / "meas.dat"
    _write_meas(meas, "0" * 32, SMALL.scan.num_trs, np.random.default_rng(2))
    assert _convert(meas, seq_path, tmp_path / "a.h5") == 1
    assert not (tmp_path / "a.h5").exists()
    assert _convert(meas, seq_path, tmp_path / "b.h5", "--no-signature-check") == 0


def test_to_ismrmrd_partial_scan(built: tuple[Path, Path], tmp_path: Path) -> None:
    seq_path, traj_path = built
    config, trajectory = load_trajectory(traj_path)
    meas = tmp_path / "meas.dat"
    num_trs = SMALL.scan.num_trs - 2
    data, _ = _write_meas(meas, read_seq_signature(seq_path), num_trs, np.random.default_rng(3))
    assert _convert(meas, seq_path, tmp_path / "a.h5") == 1
    out = tmp_path / "b.h5"
    assert _convert(meas, seq_path, out, "--allow-partial") == 0
    raw = load_rawdata(out)
    raw.check_trajectory(config, trajectory)
    np.testing.assert_array_equal(raw.read(), data)


def test_to_ismrmrd_refuses_an_existing_output(built: tuple[Path, Path], tmp_path: Path) -> None:
    seq_path, _ = built
    meas = tmp_path / "meas.dat"
    _write_meas(meas, read_seq_signature(seq_path), SMALL.scan.num_trs, np.random.default_rng(4))
    out = tmp_path / "raw.h5"
    out.write_bytes(b"keep")
    assert _convert(meas, seq_path, out) == 1
    assert out.read_bytes() == b"keep"


def test_read_twix_rejects_wrong_segments(built: tuple[Path, Path], tmp_path: Path) -> None:
    seq_path, _ = built
    meas = tmp_path / "meas.dat"
    _write_meas(meas, "", 2, np.random.default_rng(5), segment=SEGMENT // 2)
    with pytest.raises(ValueError, match="segment"):
        read_twix(meas, SMALL)


def test_check_trajectory_rejects_another_design(built: tuple[Path, Path], tmp_path: Path) -> None:
    seq_path, traj_path = built
    _, trajectory = load_trajectory(traj_path)
    meas = tmp_path / "meas.dat"
    _write_meas(meas, read_seq_signature(seq_path), SMALL.scan.num_trs, np.random.default_rng(6))
    out = tmp_path / "raw.h5"
    assert _convert(meas, seq_path, out) == 0
    other = SMALL.with_overrides(["seed=1"])
    with pytest.raises(ValueError, match="digest"):
        load_rawdata(out).check_trajectory(other, trajectory)
    shorter = Trajectory(**{**trajectory.__dict__, "tr_index": trajectory.tr_index[:2]})
    with pytest.raises(ValueError, match="do not match"):
        load_rawdata(out).check_trajectory(SMALL, shorter)
