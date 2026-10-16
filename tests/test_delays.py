from pathlib import Path

import h5py
import numpy as np
import pytest
from conftest import PhantomScan
from numpy.typing import NDArray

from relaxed_rosettes.cli import main
from relaxed_rosettes.config import GAMMA_HZ_PER_T
from relaxed_rosettes.recon.delays import design_matrix, find_peaks, fit_metric
from relaxed_rosettes.recon.rawdata import SIMULATED_GRADIENT_DELAYS, load_rawdata
from relaxed_rosettes.sequence import load_design
from relaxed_rosettes.trajectory import (
    kspace_at_times,
    load_trajectory,
    make_trajectory,
    save_trajectory,
)

FloatArray = NDArray[np.float64]

DELAYS_S = (3e-6, -2e-6, 5e-6)


def test_delayed_trajectory_shifts_each_physical_axis(
    phantom_scan: PhantomScan, tmp_path: Path
) -> None:
    config, schedule, shapes = load_design(phantom_scan.seq_path.with_suffix(".npz"))
    nominal = make_trajectory(config, schedule, shapes)
    assert not nominal.has_delays
    zero = make_trajectory(config, schedule, shapes, [0.0, 0.0, 0.0])
    np.testing.assert_array_equal(zero.k_per_m(), nominal.k_per_m())

    delayed = make_trajectory(config, schedule, shapes, DELAYS_S)
    assert delayed.has_delays
    trs = np.array([0, 5])
    raster = config.system.grad_raster_time_s
    for a, k in zip(trs, delayed.k_per_m(trs), strict=True):
        shape = shapes[nominal.shape_index[a]]
        rotation = nominal.rotation_matrix[a]
        for i, delay in enumerate(DELAYS_S):
            late = kspace_at_times(shape.g_mT_per_m, raster, nominal.adc_time_in_tr_s - delay)
            np.testing.assert_allclose(k[:, i], late @ rotation[i], atol=1e-9)

    path = tmp_path / "traj.npz"
    save_trajectory(path, config, delayed)
    _, loaded = load_trajectory(path)
    np.testing.assert_array_equal(loaded.gradient_delay_s, DELAYS_S)
    np.testing.assert_array_equal(loaded.k_per_m(trs), delayed.k_per_m(trs))
    # A file from before the delays loads as nominal.
    with np.load(path) as data:
        old = {
            key: data[key]
            for key in data.files
            if key not in ("gradient_delay_s", "k_shape_delayed_per_m")
        }
    np.savez(tmp_path / "old.npz", **old)
    _, loaded = load_trajectory(tmp_path / "old.npz")
    assert not loaded.has_delays
    np.testing.assert_array_equal(loaded.gradient_delay_s, np.zeros(3))


def test_find_peaks_is_exact_for_a_parabola() -> None:
    t = np.arange(40, dtype=np.float64)
    centres = np.array([10, 25])
    magnitude = np.maximum(5.0 - 0.3 * (t - 10.4) ** 2, 0) + np.maximum(
        5.0 - 0.3 * (t - 26.7) ** 2, 0
    )
    peaks = find_peaks(magnitude, centres, 3)
    np.testing.assert_allclose(peaks.position, [10.4, 26.7])
    # A window whose largest sample is at its edge has no peak.
    assert np.isnan(find_peaks(magnitude, np.array([30]), 2).position[0])


def test_model_recovers_delays_offsets_and_metric() -> None:
    rng = np.random.default_rng(0)
    g = rng.standard_normal((500, 3)) * 0.02
    tau = rng.uniform(1e-3, 30e-3, 500)
    metric = np.diag([0.8, 1.2, 1.0])
    params = np.concatenate([DELAYS_S, [0.5, -0.2, 0.1], [1e-5, -2e-5, 3e-5]])
    shift = design_matrix(g, tau, metric) @ params
    np.testing.assert_allclose(
        np.linalg.lstsq(design_matrix(g, tau, metric), shift, rcond=None)[0], params
    )
    # The straight-path echo of a delayed path, closest to k_0 in the metric.
    d = np.array(DELAYS_S)
    k0 = params[3:6] - GAMMA_HZ_PER_T * tau[:, None] * params[6:]
    qg = g @ metric
    expected = (np.sum(qg * g * d, axis=1) + np.sum(qg * k0, axis=1) / GAMMA_HZ_PER_T) / np.sum(
        qg * g, axis=1
    )
    np.testing.assert_allclose(shift, expected)
    widths = np.einsum("ni,ij,nj->n", g, 7.0 * metric, g)
    np.testing.assert_allclose(fit_metric(g, widths, np.ones(500, bool)), metric, atol=1e-12)


def _simulate(phantom_scan: PhantomScan, directory: Path, delays: str | None) -> Path:
    phantom = directory / "phantom.toml"
    if not phantom.exists():
        assert main(["phantom", "init", str(phantom)]) == 0
    raw = directory / f"sim-{delays}.h5"
    args = ["simulate", str(phantom), str(phantom_scan.seq_path), "-o", str(raw)]
    if delays is not None:
        args += ["--gradient-delays-s", delays]
    assert main(args) == 0
    return raw


def _estimate(phantom_scan: PhantomScan, raw: Path, out: Path) -> FloatArray:
    assert main(["delays", str(raw), str(phantom_scan.traj_path), "-o", str(out)]) == 0
    with np.load(out) as result:
        delays: FloatArray = result["gradient_delay_s"]
    return delays


def test_delays_are_recovered_from_a_simulated_scan(
    phantom_scan: PhantomScan, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    raw = _simulate(phantom_scan, tmp_path, None)
    np.testing.assert_allclose(_estimate(phantom_scan, raw, tmp_path / "none.npz"), 0, atol=0.1e-6)

    raw = _simulate(phantom_scan, tmp_path, ",".join(map(str, DELAYS_S)))
    with h5py.File(raw, "r") as f:
        assert SIMULATED_GRADIENT_DELAYS in f["dataset/xml"][0].decode()
    # The stored trajectory stays nominal; the signal follows the delays.
    design, nominal = load_trajectory(phantom_scan.traj_path)
    data = load_rawdata(raw)
    with h5py.File(raw, "r") as f:
        record = f["dataset/data"][data.noise.shape[1] // design.readout.adc_num_samples]
    stored = record["traj"].reshape(-1, 3) / design.geometry.fov_m
    np.testing.assert_allclose(stored, nominal.k_per_m(np.array([0]))[0], atol=1e-3)

    estimate = _estimate(phantom_scan, raw, tmp_path / "delays.npz")
    np.testing.assert_allclose(estimate, DELAYS_S, atol=0.1e-6)
    assert "--gradient-delays-s" in capsys.readouterr().out

    # The estimate corrects the exported trajectory.
    corrected = tmp_path / "corrected.npz"
    args = ["export-traj", str(phantom_scan.seq_path), "-o", str(corrected)]
    assert main([*args, "--gradient-delays-s", ",".join(map(str, estimate))]) == 0
    _, trajectory = load_trajectory(corrected)
    np.testing.assert_allclose(trajectory.gradient_delay_s, estimate)
    capsys.readouterr()
    assert main(["delays", str(raw), str(corrected), "-o", str(tmp_path / "again.npz")]) == 1
    assert "already has gradient delays" in capsys.readouterr().err
