import json
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from conftest import PhantomScan
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from relaxed_rosettes.cli import main
from relaxed_rosettes.config import GAMMA_HZ_PER_T, Config
from relaxed_rosettes.optimize import Shape
from relaxed_rosettes.sequence import Schedule
from relaxed_rosettes.spoiling import Tissue, simulate_spoiling
from relaxed_rosettes.targets import ShapeTargets

CONFIG = Config().with_overrides(["readout.tr_s=0.01", "readout.adc_num_samples=2000"])
TISSUE = Tissue(1.0, 0.1)


def _line_shape(spoil_per_m: float) -> Shape:
    """No gradient for the first 2 ms (the first ADC samples are at k = 0),
    then a constant z gradient to ``spoil_per_m`` along z at the end of the TR."""
    raster = CONFIG.system.grad_raster_time_s
    n = round(CONFIG.readout.tr_s / raster)
    start = round(0.002 / raster)
    g = np.zeros((n, 3))
    # Samples start..n-1 at g, plus the half raster of the ramp down at the end.
    g[start:, 2] = spoil_per_m / (GAMMA_HZ_PER_T * 1e-3 * (n - start) * raster)
    return Shape(g, cast(ShapeTargets, None), "optimal", 0.0)


def _schedule(rotations: NDArray[np.float64], phases: NDArray[np.float64]) -> Schedule:
    n = len(rotations)
    return Schedule(
        shape_index=np.zeros(n, dtype=np.int64),
        rotation_matrix=rotations,
        rf_phase_rad=phases,
        acquired=np.ones(n, dtype=np.bool_),
    )


def _quadratic(n: int) -> NDArray[np.float64]:
    k = np.arange(n)
    phases: NDArray[np.float64] = np.deg2rad(np.mod(117.0 * k * (k + 1) / 2, 360.0))
    return phases


def _reference_epg(n_trs: int, phases: NDArray[np.float64], tissue: Tissue) -> NDArray[np.float64]:
    """|F+_0| after each pulse: a 1D phase graph with one state shift per TR."""
    alpha = np.deg2rad(CONFIG.excitation.flip_angle_deg)
    e1 = np.exp(-CONFIG.readout.tr_s / tissue.t1_s)
    e2 = np.exp(-CONFIG.readout.tr_s / tissue.t2_s)
    size = n_trs + 2
    fp = np.zeros(size, dtype=np.complex128)
    fm = np.zeros(size, dtype=np.complex128)
    z = np.zeros(size, dtype=np.complex128)
    z[0] = 1.0
    out = np.zeros(n_trs)
    c2, s2, sa, ca = np.cos(alpha / 2) ** 2, np.sin(alpha / 2) ** 2, np.sin(alpha), np.cos(alpha)
    for t in range(n_trs):
        e = np.exp(1j * phases[t])
        fp, fm, z = (
            c2 * fp + e**2 * s2 * fm - 1j * e * sa * z,
            np.conj(e) ** 2 * s2 * fp + c2 * fm + 1j * np.conj(e) * sa * z,
            -0.5j * np.conj(e) * sa * fp + 0.5j * e * sa * fm + ca * z,
        )
        out[t] = abs(fp[0])
        fp, fm, z = fp * e2, fm * e2, z * e1
        z[0] += 1 - e1
        fp = np.concatenate([[0.0], fp[:-1]])
        fm = np.concatenate([fm[1:], [0.0]])
        fp[0] = np.conj(fm[0])
    return out


def test_constant_spoiler_matches_the_1d_phase_graph() -> None:
    n = 40
    phases = _quadratic(n)
    schedule = _schedule(np.tile(np.eye(3), (n, 1, 1)), phases)
    result = simulate_spoiling(
        CONFIG, schedule, [_line_shape(2500.0)], TISSUE, threshold=1e-12, obj=None
    )
    np.testing.assert_allclose(result.fid, _reference_epg(n, phases, TISSUE), rtol=1e-9, atol=1e-12)


def test_random_rotations_approach_ideal_spoiling() -> None:
    n = 300
    rng = np.random.default_rng(0)
    rotations = Rotation.random(n, random_state=rng).as_matrix()
    schedule = _schedule(rotations, rng.uniform(0, 2 * np.pi, n))
    result = simulate_spoiling(CONFIG, schedule, [_line_shape(2500.0)], Tissue(1.0, 0.02))
    assert np.mean(result.fid / result.ideal_fid) == pytest.approx(1.0, abs=0.03)


def test_a_nearly_antiparallel_pair_is_found() -> None:
    """Spoilers +s then -s (to 0.1 degree) return TR 1's coherence within about
    one cell of k = 0 at TR 3, where the readout starts."""
    flip = Rotation.from_euler("x", 179.9, degrees=True).as_matrix()
    rotations = np.stack([np.eye(3), flip, np.eye(3)])
    schedule = _schedule(rotations, np.zeros(3))
    result = simulate_spoiling(CONFIG, schedule, [_line_shape(2500.0)], Tissue(1.0, 1.0))
    assert result.closest_cells[2] < 2.0
    assert result.near_amplitude[2, 1] > 0.1  # within 3 cells
    assert result.near_by_kind[2, 0] > 0.1  # a transverse chain
    assert result.near_amplitude[1, 2] == 0.0  # nothing near at TR 2

    rotations[1] = Rotation.from_euler("x", 90, degrees=True).as_matrix()
    perpendicular = simulate_spoiling(CONFIG, schedule, [_line_shape(2500.0)], Tissue(1.0, 1.0))
    assert perpendicular.near_amplitude[2, 2] == 0.0


def test_diffusion_damps_the_spoiled_states() -> None:
    n = 60
    rng = np.random.default_rng(1)
    schedule = _schedule(Rotation.random(n, random_state=rng).as_matrix(), _quadratic(n))
    shapes = [_line_shape(2500.0)]
    still = simulate_spoiling(CONFIG, schedule, shapes, Tissue(4.0, 1.5))
    moving = simulate_spoiling(CONFIG, schedule, shapes, Tissue(4.0, 1.5, 3e-9))
    assert moving.states[-1] < still.states[-1]


def test_spoiling_cli(
    phantom_scan: PhantomScan, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    seq = str(phantom_scan.seq_path)
    out = tmp_path / "spoil.npz"
    assert main(["spoiling", seq, "--json", "-o", str(out)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert len(report["tissues"]) == 2
    for tissue in report["tissues"]:
        assert tissue["fid_over_ideal"] == pytest.approx(1.0, abs=0.2)
        assert tissue["energy_mean"] is not None
    with np.load(out) as arrays:
        assert arrays["fid0"].shape == (phantom_scan.design.scan.num_trs,)
    args = ["spoiling", seq, "--tissue", "1,0.05", "--rotations", "random", "--no-phantom"]
    assert main([*args, "--rf-phase", "quadratic117"]) == 0
    assert "TRs >1%/>10%" in capsys.readouterr().out
