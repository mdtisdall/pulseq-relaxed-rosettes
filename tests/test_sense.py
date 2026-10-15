from pathlib import Path

import numpy as np
import pytest
from conftest import PhantomScan
from numpy.typing import NDArray

from relaxed_rosettes.cli import main
from relaxed_rosettes.recon.b0 import field_map_hz, segmentation
from relaxed_rosettes.recon.coilmaps import inati_maps
from relaxed_rosettes.recon.gridding import pipe_menon_dcf
from relaxed_rosettes.recon.nufft import Nufft
from relaxed_rosettes.recon.preprocess import CachedBatch
from relaxed_rosettes.recon.sense import OffResonance, SenseOperator, cg_sense

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex64]


def _grid(n: int, res: float) -> tuple[FloatArray, FloatArray, FloatArray]:
    axis = (np.arange(n) - n // 2) * res
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    return x, y, z


def _true_maps(n: int, res: float, coils: int) -> ComplexArray:
    """Smooth coil maps with unit norm over the coils."""
    x, y, z = _grid(n, res)
    maps = []
    for c in range(coils):
        angle = 2 * np.pi * c / coils
        center = 0.15 * np.array([np.cos(angle), np.sin(angle), 0.0])
        r2 = (x - center[0]) ** 2 + (y - center[1]) ** 2 + (z - center[2]) ** 2
        phase = 2 * np.pi * (0.3 * c + 2.0 * x)
        maps.append(np.exp(-r2 / (2 * 0.12**2)) * np.exp(1j * phase))
    stacked = np.array(maps)
    stacked /= np.sqrt(np.sum(np.abs(stacked) ** 2, axis=0))[None]
    return stacked.astype(np.complex64)


def _ball(rng: np.random.Generator, n_points: int, radius: float) -> FloatArray:
    direction = rng.standard_normal((n_points, 3))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    points: FloatArray = direction * (radius * rng.uniform(0, 1, n_points) ** (1 / 3))[:, None]
    return points


def test_inati_recovers_smooth_maps() -> None:
    n, res, coils = 24, 0.01, 4
    truth = _true_maps(n, res, coils)
    x, y, z = _grid(n, res)
    obj = np.exp(-(x**2 + y**2 + z**2) / (2 * 0.06**2))
    maps = inati_maps((truth * obj[None]).astype(np.complex64), smoothing_voxels=5, iterations=8)
    np.testing.assert_allclose(np.sum(np.abs(maps) ** 2, axis=0), 1.0, rtol=1e-5)
    # Equal up to a phase that is common to the coils.
    match = np.abs(np.sum(maps.conj() * truth, axis=0))
    assert np.all(match[obj > 0.2] > 0.99)


def test_segmentation_is_accurate_and_exact_at_segment_times() -> None:
    tau = np.linspace(0, 0.045, 400)
    frequency = np.linspace(-80, 80, 1000).astype(np.float32)
    seg = segmentation(tau, 12, frequency, np.ones_like(frequency))
    assert seg.rms_error < 1e-2
    f = np.array([-50.0, 13.0, 77.0])
    approx = np.exp(2j * np.pi * f[:, None] * seg.segment_tau_s[None]) @ seg.coefficients
    np.testing.assert_allclose(approx, np.exp(2j * np.pi * f[:, None] * tau[None]), atol=2e-2)


def test_field_map_from_windows() -> None:
    n, res = 16, 0.01
    x, y, z = _grid(n, res)
    obj = np.exp(-(x**2 + y**2 + z**2) / (2 * 0.04**2)).astype(np.complex64)
    truth = 40 * np.tanh(x / 0.03) + 15
    spacing = 0.004
    windows = [
        (obj * np.exp(2j * np.pi * truth * (0.001 + w * spacing))).astype(np.complex64)
        for w in range(6)
    ]
    times: FloatArray = 0.001 + spacing * np.arange(6, dtype=np.float64)
    estimate = field_map_hz(windows, times, smoothing_voxels=1)
    inside = np.abs(obj) > 0.2
    np.testing.assert_allclose(estimate[inside], truth[inside], atol=0.5)


def _operator_problem(
    rng: np.random.Generator, segments: int
) -> tuple[SenseOperator, list[CachedBatch], FloatArray, ComplexArray]:
    """A small problem: maps, a field map, and random k and tau in two batches."""
    n, res, coils = 12, 0.01, 4
    nufft = Nufft(n=n, resolution_m=res, eps=1e-6)
    maps = _true_maps(n, res, coils)
    x, y, z = _grid(n, res)
    frequency = (30 * np.tanh((x + 0.3 * y) / 0.03)).astype(np.float32)
    samples = np.linspace(0.0, 0.02, 200)
    seg = segmentation(samples, segments, frequency.ravel(), np.ones(frequency.size, np.float32))
    phases = np.exp(2j * np.pi * frequency[None] * seg.segment_tau_s[:, None, None, None])
    off = OffResonance(phases=phases.astype(np.complex64), coefficients=seg.coefficients)
    k_max = 1 / (2 * res)
    k = np.concatenate([_ball(rng, 15_000, k_max), _ball(rng, 15_000, k_max / 3)])
    sample = rng.integers(0, len(samples), len(k)).astype(np.int32)
    batches = [
        CachedBatch(
            k_per_m=k[part].astype(np.float32),
            tau_s=samples[sample[part]].astype(np.float32),
            sample=sample[part],
            values=np.zeros((coils, len(k[part])), np.complex64),
        )
        for part in (slice(0, 12_000), slice(12_000, None))
    ]
    obj = np.exp(-((x - 0.01) ** 2 + y**2 + (z + 0.01) ** 2) / (2 * 0.02**2))
    image = (obj * np.exp(1j * 2 * np.pi * x / 0.1)).astype(np.complex64)
    return SenseOperator(nufft=nufft, maps=maps, off_resonance=off), batches, samples, image


def test_sense_operator_is_adjoint() -> None:
    rng = np.random.default_rng(0)
    operator, batches, _, image = _operator_problem(rng, segments=6)
    batch = batches[0]
    values = (
        rng.standard_normal(batch.values.shape) + 1j * rng.standard_normal(batch.values.shape)
    ).astype(np.complex64)
    lhs = np.vdot(operator.forward(image, batch), values)
    rhs = np.vdot(image, operator.adjoint(values, batch))
    assert lhs == pytest.approx(rhs, rel=1e-4)


def test_cg_sense_corrects_off_resonance() -> None:
    rng = np.random.default_rng(1)
    # Data from a fine segmentation (nearly exact), reconstructed with a coarse one.
    exact, batches, _, image = _operator_problem(rng, segments=40)
    data = [CachedBatch(b.k_per_m, b.tau_s, b.sample, exact.forward(image, b)) for b in batches]
    coarse, *_ = _operator_problem(np.random.default_rng(1), segments=8)
    dcf = pipe_menon_dcf(exact.nufft, lambda: (b.k_per_m for b in data), iterations=10)

    def error(operator: SenseOperator) -> float:
        result = cg_sense(operator, data, dcf.weights, 1e-4, iterations=30, tolerance=1e-6)
        assert result.residual[-1] < result.residual[0]
        return float(np.linalg.norm(result.image - image) / np.linalg.norm(image))

    with_b0 = error(coarse)
    without_b0 = error(SenseOperator(nufft=coarse.nufft, maps=coarse.maps))
    assert with_b0 < 0.05
    assert without_b0 > 3 * with_b0


def test_sense_cli_reconstructs_the_phantom(phantom_scan: PhantomScan, tmp_path: Path) -> None:
    scan = phantom_scan
    out = tmp_path / "sense.npz"
    res = 0.016
    args = ["sense", str(scan.raw_path), str(scan.traj_path), "-o", str(out), "--coil-maps"]
    for setting in (
        f"data.resolution_m={res}",
        "coils.virtual_coils=2",
        "coil_maps.resolution_m=0.032",
        "b0.resolution_m=0.032",
        "b0.windows=4",
        "b0.segments=4",
        "cg.iterations=10",
    ):
        args += ["--set", setting]
    assert main(args) == 0

    n = round(scan.design.geometry.fov_m / res)
    truth = scan.truth(n, res)
    with np.load(out) as result:
        magnitude = result["magnitude"]
        assert magnitude.shape == (n, n, n)
        assert result["coil_maps"].shape == (2, n, n, n)
        residual = result["cg_residual"]
        assert residual[-1] < 0.1 * residual[0]
        field = result["field_map_hz"]
        assert str(result["design_config_digest"]) == scan.design.digest()
    correlation = np.sum(magnitude * truth) / (np.linalg.norm(magnitude) * np.linalg.norm(truth))
    assert correlation > 0.95
    inside = truth > 0.5
    np.testing.assert_allclose(np.median(field[inside]), scan.off_resonance_hz, atol=3.0)
