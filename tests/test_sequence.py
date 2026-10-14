from pathlib import Path

import numpy as np
import pypulseq as pp
import pytest
from scipy.spatial.transform import Rotation

from relaxed_rosettes.cli import main
from relaxed_rosettes.config import Config
from relaxed_rosettes.optimize import Shape, optimize_shapes
from relaxed_rosettes.rotations import super_fibonacci_quaternions
from relaxed_rosettes.sequence import (
    RF_SPOIL_INCREMENT_DEG,
    Schedule,
    build_sequence,
    check_adc_segments,
    load_design,
    make_schedule,
)
from relaxed_rosettes.system import make_opts

SMALL = Config().with_overrides(
    [
        "readout.tr_s=0.01",
        "readout.adc_num_samples=2000",
        "shapes.echoes_per_tr=4",
        "shapes.num_shapes=2",
        "scan.num_trs=6",
        "scan.num_dummy_trs=2",
    ]
)


@pytest.fixture(scope="module")
def shapes() -> list[Shape]:
    return optimize_shapes(SMALL, jobs=1)


@pytest.fixture(scope="module")
def schedule() -> Schedule:
    return make_schedule(SMALL)


@pytest.fixture(scope="module")
def seq(shapes: list[Shape], schedule: Schedule) -> pp.Sequence:
    return build_sequence(SMALL, shapes, schedule)


def test_super_fibonacci_is_spread_out() -> None:
    n = 500
    q = super_fibonacci_quaternions(n)
    np.testing.assert_allclose(np.linalg.norm(q, axis=1), 1.0)
    random = Rotation.random(n, random_state=0).as_quat()

    def min_distance(quats: np.ndarray) -> float:
        dots = np.abs(quats @ quats.T)
        np.fill_diagonal(dots, 0.0)
        return float(np.arccos(np.clip(dots.max(), -1, 1)))

    assert min_distance(q) > 3 * min_distance(random)


def test_schedule(schedule: Schedule) -> None:
    n_dummy, n_acq = SMALL.scan.num_dummy_trs, SMALL.scan.num_trs
    assert schedule.num_trs == n_dummy + n_acq
    assert not schedule.acquired[:n_dummy].any() and schedule.acquired[n_dummy:].all()
    counts = np.bincount(schedule.shape_index[n_dummy:], minlength=SMALL.shapes.num_shapes)
    assert counts.max() - counts.min() <= 1
    r = schedule.rotation_matrix
    np.testing.assert_allclose(
        r @ r.transpose(0, 2, 1), np.broadcast_to(np.eye(3), r.shape), atol=1e-12
    )
    np.testing.assert_allclose(np.linalg.det(r), 1.0)
    phases = schedule.rf_phase_rad
    assert np.all((phases >= 0) & (phases < 2 * np.pi))
    again = make_schedule(SMALL)
    np.testing.assert_array_equal(again.rotation_matrix, schedule.rotation_matrix)


def test_quadratic_rf_phase() -> None:
    config = SMALL.with_overrides(["excitation.rf_phase=quadratic117"])
    phases = np.rad2deg(make_schedule(config).rf_phase_rad)
    increments = np.mod(np.diff(phases), 360.0)
    np.testing.assert_allclose(
        increments, np.mod(RF_SPOIL_INCREMENT_DEG * np.arange(1, len(phases)), 360.0)
    )


def test_adc_segments() -> None:
    system = make_opts(Config())
    assert check_adc_segments(Config(), system) == (2, 7000)
    bad = Config().with_overrides(["readout.adc_num_samples=14001"])
    with pytest.raises(ValueError, match="ADC segments"):
        check_adc_segments(bad, make_opts(bad))


def test_blocks(seq: pp.Sequence, schedule: Schedule) -> None:
    assert len(seq.block_events) == schedule.num_trs
    assert seq.duration()[0] == pytest.approx(schedule.num_trs * SMALL.readout.tr_s)
    for i in range(schedule.num_trs):
        block = seq.get_block(i + 1)
        assert block.rf.phase_offset == pytest.approx(schedule.rf_phase_rad[i])
        if schedule.acquired[i]:
            assert block.adc.phase_offset == pytest.approx(schedule.rf_phase_rad[i])
        else:
            assert block.adc is None


def test_kspace_hits_rotated_targets(
    seq: pp.Sequence, shapes: list[Shape], schedule: Schedule
) -> None:
    """pypulseq's k-space passes through R @ target in every acquired TR."""
    k_adc, _, _, _, t_adc = seq.calculate_kspace()
    n = SMALL.readout.adc_num_samples
    tr = SMALL.readout.tr_s
    k_max = SMALL.k_max_per_m
    acquired = np.flatnonzero(schedule.acquired)
    for j, tr_index in enumerate(acquired):
        samples = slice(j * n, (j + 1) * n)
        t = t_adc[samples] - tr_index * tr
        k = k_adc[:, samples]
        targets = shapes[schedule.shape_index[tr_index]].targets
        inside = (targets.t_s >= t[0]) & (targets.t_s <= t[-1])
        assert inside.sum() >= 2 * targets.num_echoes
        expected = targets.k[inside] @ schedule.rotation_matrix[tr_index].T
        measured = np.stack([np.interp(targets.t_s[inside], t, k[a]) for a in range(3)], axis=1)
        np.testing.assert_allclose(measured / k_max, expected, atol=1e-4)


def test_write_read_round_trip(tmp_path: Path, seq: pp.Sequence) -> None:
    path = tmp_path / "small.seq"
    seq.write(str(path))
    read = pp.Sequence(system=make_opts(SMALL))
    read.read(str(path))
    assert read.get_definition("ConfigDigest") == SMALL.digest()
    k_original = seq.calculate_kspace()[0]
    k_read = read.calculate_kspace()[0]
    np.testing.assert_allclose(k_read, k_original, atol=1e-3)


def test_cli_writes_seq_and_design(tmp_path: Path) -> None:
    config_path = tmp_path / "small.toml"
    config_path.write_text(SMALL.to_toml())
    shapes_path = tmp_path / "shapes.npz"
    seq_path = tmp_path / "out.seq"
    assert main(["shapes", str(config_path), "-o", str(shapes_path), "--jobs", "1"]) == 0
    assert main(["sequence", str(shapes_path), "-o", str(seq_path)]) == 0
    assert seq_path.exists()
    config, schedule, shapes = load_design(seq_path.with_suffix(".npz"))
    assert config == SMALL
    np.testing.assert_array_equal(schedule.shape_index, make_schedule(SMALL).shape_index)
    assert len(shapes) == SMALL.shapes.num_shapes
