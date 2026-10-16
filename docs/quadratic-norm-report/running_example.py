"""The running example of docs/quadratic-norm-report.md: one 1D x-T phantom, three applications.

A 1D object of 128 cells, scanned for 2000 TRs of 50 ms (100 s). Each TR reads 128 samples
over a 45 ms readout at absolute times T_p + tau, along one of 32 shapes of 1D petals (random
order, random sign of k as the 1D rotation). The object decays with tau (R2* 20-50/s, the field
corrected) and changes over T:

- a phase drift of half a cycle over the scan (receiver or field drift), the same everywhere;
- region A (cells 47-53): a BOLD change of R2*, 1.5/s RMS, in 0.02-0.1 Hz;
- region B (cells 77-83): a non-BOLD change of S0 (inflow), 3% RMS, in the same band;
- region C (cells 62-66): region A's change of R2*, 1 s later;
- everywhere: a respiratory change of S0, 2% at 0.3 Hz (outside the band).

Noise of 0.02 (complex) is added to every sample. The script prints the numbers quoted in the
report and, with ``--figures DIR``, writes its four figures to DIR. Seeded; about a minute.
"""

from __future__ import annotations

import sys

import numpy as np
from numpy.typing import NDArray

M, SAMPLES, RANK, N_TR, TR_S, READOUT_S = 128, 128, 3, 2000, 0.05, 0.045
SHAPES, SPEED, NOISE, RESPIRATION, RCOND = 32, 4, 0.02, 0.02, 0.05
DURATION_S = N_TR * TR_S
TAUS = (np.arange(SAMPLES) + 0.5) * READOUT_S / SAMPLES
D_TAU = READOUT_S / SAMPLES
R = np.arange(M)
K_CELLS = np.fft.fftfreq(M, 1 / M).astype(int)
DENSITY = (np.abs(R - 64) < 40) * (1 + 0.25 * np.cos(2 * np.pi * R / 20))
OBJECT = DENSITY > 0
R2STAR = np.where(OBJECT, 20 + 30 * R / M, 0.0)
REGION_A = np.abs(R - 50) < 4
REGION_B = np.abs(R - 80) < 4
REGION_C = np.abs(R - 64) < 3
LAG_S = 1.0
BAND = np.concatenate([-np.arange(10, 1, -1), np.arange(2, 11)])  # 0.02-0.1 Hz, both signs
TOTAL = np.arange(-12, 13)  # the band of the total power: |f| <= 0.12 Hz

Complex = NDArray[np.complex128]
Real = NDArray[np.float64]


def band_course(rms: float, seed: int) -> tuple[Complex, float]:
    """Fourier coefficients over the band (positive bins 2-10) and the scale for the RMS."""
    rng = np.random.default_rng(seed)
    coefficients = rng.standard_normal(9) + 1j * rng.standard_normal(9)
    t = np.arange(N_TR) * TR_S
    course = np.real(np.exp(2j * np.pi * np.outer(t, np.arange(2, 11) / DURATION_S)) @ coefficients)
    return coefficients, rms / float(np.sqrt(np.mean(course**2)))


BOLD, BOLD_SCALE = band_course(1.5, 1)
INFLOW, INFLOW_SCALE = band_course(0.03, 2)


def course(coefficients: Complex, scale: float, t: Real) -> Real:
    phase = np.exp(2j * np.pi * np.multiply.outer(t, np.arange(2, 11) / DURATION_S))
    out: Real = scale * np.real(phase @ coefficients)
    return out


def drift(t: Real) -> Complex:
    out: Complex = np.exp(1j * np.pi * t / DURATION_S)
    return out


def image(t: Real) -> Complex:
    """x(r, tau, t) at the samples of one TR, (M, SAMPLES), for their times t (SAMPLES,)."""
    d_r2star = REGION_A[:, None] * course(BOLD, BOLD_SCALE, t)[None]
    d_r2star = d_r2star + REGION_C[:, None] * course(BOLD, BOLD_SCALE, t - LAG_S)[None]
    d_s0 = REGION_B[:, None] * course(INFLOW, INFLOW_SCALE, t)[None]
    d_s0 = d_s0 + RESPIRATION * np.sin(2 * np.pi * 0.3 * t)[None]
    decay = np.exp(-(R2STAR[:, None] + d_r2star) * TAUS[None])
    out: Complex = DENSITY[:, None] * (1 + d_s0) * decay * drift(t)[None]
    return out


def petal_shape(rng: np.random.Generator) -> NDArray[np.int64]:
    """The signed k cell at each tau sample: petals 0 -> tip -> 0, with centre-heavy tips."""
    k: list[int] = []
    while len(k) < SAMPLES:
        tip = max(4, round((M // 2 - 1) * rng.random() ** 2))
        out = np.arange(0, tip, SPEED)
        k.extend((np.concatenate([out, tip - out]) * rng.choice([-1, 1])).tolist())
    return np.array(k[:SAMPLES])


U_BASIS, _, _ = np.linalg.svd(np.exp(-np.outer(TAUS, np.linspace(10, 60, 11))))
PHI: Real = U_BASIS[:, :RANK]  # orthonormal tau basis of the decays


class Scan:
    """The acquisition, the truth from the full noiseless signal, and the estimators."""

    def __init__(self) -> None:
        rng = np.random.default_rng(21)
        self.library = [petal_shape(rng) for _ in range(SHAPES)]
        index = rng.integers(SHAPES, size=N_TR)
        signs = rng.choice([-1, 1], size=(N_TR, 1))
        self.paths = np.array([self.library[i] for i in index]) * signs
        self.times = np.arange(N_TR)[:, None] * TR_S + TAUS[None]
        self.data = np.empty((N_TR, SAMPLES), complex)
        self.energy = np.zeros(M)  # sum over T and tau of |x|^2, the total power
        mean = np.zeros((M, SAMPLES), complex)
        self.true_band = np.zeros((len(BAND), RANK, M), complex)  # drift corrected
        true_total = np.zeros((len(TOTAL), RANK, M), complex)
        for p in range(N_TR):
            x = image(self.times[p])
            self.energy += D_TAU * np.sum(np.abs(x) ** 2, 1)
            mean += x / N_TR
            k_space = np.fft.fft(x, axis=0, norm="ortho")
            self.data[p] = k_space[np.mod(self.paths[p], M), np.arange(SAMPLES)]
            weights = self.basis_t(BAND, self.times[p])  # (J, SAMPLES)
            corrected = x * np.conj(drift(self.times[p]))[None]
            self.true_band += np.einsum("js,ls,ms->jlm", weights, PHI.T, corrected)
            true_total += np.einsum("js,ls,ms->jlm", self.basis_t(TOTAL, self.times[p]), PHI.T, x)
        self.dc_energy = N_TR * D_TAU * np.sum(np.abs(mean) ** 2, 1)
        self.band_energy = D_TAU * np.sum(np.abs(true_total) ** 2, axis=(0, 1))  # |f| <= 0.12 Hz
        white = rng.standard_normal(self.data.shape) + 1j * rng.standard_normal(self.data.shape)
        self.data += NOISE * white / np.sqrt(2)
        self.halves = [np.arange(N_TR) % 2 == h for h in (0, 1)]

    @staticmethod
    def basis_t(bins: NDArray[np.int64], t: Real) -> Complex:
        """The conjugate Fourier basis over T, psi_j(t)^*, at the given times."""
        out: Complex = np.exp(-2j * np.pi * np.multiply.outer(bins, t) / DURATION_S) / np.sqrt(N_TR)
        return out

    def components(self, data: Complex, trs: NDArray[np.bool_], bins: NDArray[np.int64]) -> Complex:
        """Components c[j, l, r] from the TRs in ``trs``: for each k, N A_k^-1 sum phi psi^* y."""
        out = np.zeros((len(bins), RANK, M), complex)
        weights = np.moveaxis(self.basis_t(bins, self.times), 0, -1)  # (N, SAMPLES, J)
        for i, cell in enumerate(K_CELLS):
            rows, cols = np.nonzero((self.paths == cell) & trs[:, None])
            if rows.size == 0:
                continue
            phi = PHI[cols]
            rhs = phi.T @ (weights[rows, cols] * data[rows, cols][:, None])
            out[:, :, i] = N_TR * np.linalg.lstsq(phi.T @ phi, rhs, rcond=RCOND)[0].T
        result: Complex = np.fft.ifft(out, axis=2, norm="ortho")
        return result

    def subtract_static(self, data: Complex, resolved: bool) -> Complex:
        """The data minus the static model of each k: its mean, or its fit in the tau basis."""
        out = data.copy()
        for cell in K_CELLS:
            rows, cols = np.nonzero(self.paths == cell)
            if rows.size == 0:
                continue
            if resolved:
                fit = np.linalg.lstsq(PHI[cols], data[rows, cols], rcond=RCOND)[0]
                out[rows, cols] -= PHI[cols] @ fit
            else:
                out[rows, cols] -= data[rows, cols].mean()
        return out


def application_1(scan: Scan) -> dict[str, Real]:
    """Total power and DC term over tau and T, from products between interleaved halves."""
    halves = [scan.components(scan.data, h, TOTAL) for h in scan.halves]
    cross = np.real(halves[0].conj() * halves[1])
    dc = int(np.nonzero(TOTAL == 0)[0][0])
    full = scan.components(scan.data, np.ones(N_TR, bool), np.array([0]))[0]
    tau_mean = [PHI.mean(0) @ c[dc] for c in halves]  # mean over tau of the static image
    return {
        "total power": D_TAU * np.sum(cross, axis=(0, 1)),
        "DC term": D_TAU * np.sum(cross[dc], axis=0),
        "DC term, plain": D_TAU * np.sum(np.abs(full) ** 2, axis=0),
        "squared tau-average": READOUT_S * np.real(tau_mean[0].conj() * tau_mean[1]),
    }


def static_directions(scan: Scan) -> Complex:
    """For each voxel, the dual of [S0 direction, R2* direction] in the tau basis: (2, RANK, M)."""
    corrected = scan.data * np.conj(drift(scan.times))
    static = scan.components(corrected, np.ones(N_TR, bool), np.array([0]))[0] / np.sqrt(N_TR)
    curve = PHI @ static  # (SAMPLES, M): the static decay of each voxel
    s0 = PHI.T @ curve  # d x / d (relative S0)
    r2 = PHI.T @ (-TAUS[:, None] * curve)  # d x / d R2*
    dual = np.zeros((2, RANK, M), complex)
    for i in range(M):
        g = np.stack([s0[:, i], r2[:, i]], 1)
        if np.linalg.norm(g) > 1e-6:
            dual[:, :, i] = np.linalg.pinv(g)
    return dual


def application_2(scan: Scan, dual: Complex, resolved: bool) -> tuple[Real, Complex, Complex]:
    """Mean-square changes of S0 (relative) and R2* (1/s) over T in the band, and the halves."""
    corrected = scan.data * np.conj(drift(scan.times))
    residual = scan.subtract_static(corrected, resolved)
    halves = [scan.components(residual, h, BAND) for h in scan.halves]
    coefficients = [np.einsum("klm,jlm->kjm", dual, c) for c in halves]  # (2, J, M)
    power = np.sum(np.real(coefficients[0].conj() * coefficients[1]), axis=1) / N_TR
    return power, coefficients[0], coefficients[1]


def lagged(halves: tuple[Complex, Complex], a: int, b: int, lags: Real) -> Real:
    """Lagged correlation of voxels a and b from products between the halves' band components.

    ``halves`` holds each half's components (J, M); the numerator and both band powers are
    products between halves, so none of them has a noise bias.
    """
    first, second = halves
    shift = np.exp(2j * np.pi * np.outer(BAND / DURATION_S, lags))
    cross = 0.5 * (first[:, a].conj() * second[:, b] + second[:, a].conj() * first[:, b])
    power_a = np.sum(np.real(first[:, a].conj() * second[:, a]))
    power_b = np.sum(np.real(first[:, b].conj() * second[:, b]))
    out: Real = np.real(cross @ shift) / np.sqrt(abs(power_a * power_b))
    return out


def main(figure_dir: str | None) -> None:
    scan = Scan()
    energy = application_1(scan)
    print(
        "Application 1: energy over tau and T, sum over the object relative to the true total power"
    )
    print(f"   true DC term {scan.dc_energy[OBJECT].sum() / scan.energy[OBJECT].sum():.3f}")
    band = scan.band_energy[OBJECT].sum() / scan.energy[OBJECT].sum()
    print(f"   true total power within |f| <= 0.12 Hz {band:.3f}")
    for name, values in energy.items():
        ratio = values[OBJECT].sum() / scan.energy[OBJECT].sum()
        floor = values[~OBJECT].mean() / scan.energy[OBJECT].mean()
        print(f"   {name:22s} {ratio:.3f}   background {floor:+.4f}")

    dual = static_directions(scan)
    true_coefficients = np.einsum("klm,jlm->kjm", dual, scan.true_band)
    true_power = np.sum(np.abs(true_coefficients) ** 2, axis=1) / N_TR
    estimates = {resolved: application_2(scan, dual, resolved) for resolved in (False, True)}
    print("Application 2: RMS change in the band (S0 in %, R2* in 1/s), mean over each region")
    rest = OBJECT & ~REGION_A & ~REGION_B & ~REGION_C
    regions = (("A", REGION_A), ("B", REGION_B), ("C", REGION_C), ("rest", rest))
    for label, power in (
        ("truth", true_power),
        ("tau-averaged static", estimates[False][0]),
        ("tau-resolved static", estimates[True][0]),
    ):
        cells = []
        for name, mask in regions:
            s0 = 100 * np.sqrt(max(power[0, mask].mean(), 0))
            r2 = np.sqrt(max(power[1, mask].mean(), 0))
            cells.append(f"{name} {s0:4.1f}% {r2:4.2f}/s")
        print(f"   {label:20s} " + "  ".join(cells))

    lags = np.arange(-4, 4.001, 0.05)
    a, c = 50, 64
    truth = true_coefficients[1]
    truth_lag = lagged((truth, truth), a, c, lags)
    estimate_lag = lagged((estimates[True][1][1], estimates[True][2][1]), a, c, lags)
    print(
        f"Application 3: lag of C after A, truth {lags[np.argmax(truth_lag)]:+.2f} s,"
        f" estimate {lags[np.argmax(estimate_lag)]:+.2f} s (peak {estimate_lag.max():.2f})"
    )
    if figure_dir is not None:
        figures(scan, energy, true_power, estimates, lags, truth_lag, estimate_lag, figure_dir)


def figures(
    scan: Scan,
    energy: dict[str, Real],
    true_power: Real,
    estimates: dict[bool, tuple[Real, Complex, Complex]],
    lags: Real,
    truth_lag: Real,
    estimate_lag: Real,
    figure_dir: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    save = {"dpi": 120, "metadata": {"Software": None}}
    t = np.arange(N_TR) * TR_S

    fig, axes = plt.subplots(2, 2, figsize=(10, 6.4), constrained_layout=True)
    ax = axes[0, 0]
    ax.plot(R, DENSITY, "k-", label="S0 (density)")
    ax2 = ax.twinx()
    ax2.plot(R, R2STAR, "-", color="0.6")
    ax2.set_ylabel("R2* (1/s), grey")
    for mask, name in ((REGION_A, "A"), (REGION_B, "B"), (REGION_C, "C")):
        ax.axvspan(R[mask].min(), R[mask].max(), color="C1", alpha=0.25)
        ax.text(R[mask].mean(), 1.33, name, ha="center", va="bottom")
    ax.set_xlabel("x (cells)")
    ax.set_ylabel("S0")
    ax.set_title("the object", pad=16)
    ax = axes[0, 1]
    ax.plot(t, course(BOLD, BOLD_SCALE, t), label="A: ΔR2* (1/s), BOLD")
    ax.plot(t, course(BOLD, BOLD_SCALE, t - LAG_S), "--", label="C: A's change, 1 s later")
    ax.plot(t, 50 * course(INFLOW, INFLOW_SCALE, t), label="B: ΔS0 × 50, inflow")
    ax.plot(t, 2 * np.angle(drift(t)) / np.pi, "k:", label="phase drift (× 2/π rad)")
    ax.set_xlabel("T (s)")
    ax.set_title("changes over the scan (respiration not shown)")
    ax.legend(fontsize=7, loc="lower left")
    ax = axes[1, 0]
    for p in range(3):
        ax.plot(TAUS * 1e3, scan.paths[p], ".", ms=2, label=f"TR {p + 1}")
    ax.set_xlabel("τ (ms)")
    ax.set_ylabel("k (cells)")
    ax.set_title("three TRs: each its own shape and sign")
    ax.legend(fontsize=7)
    ax = axes[1, 1]
    for p in range(6):
        ax.plot(scan.times[p], scan.paths[p], ".", ms=1.5, color="C0")
    ax.set_xlabel("absolute time t (s)")
    ax.set_ylabel("k (cells)")
    ax.set_title("six TRs in absolute time: each a 45 ms band")
    fig.savefig(f"{figure_dir}/running-example.png", **save)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
    ax = axes[0]
    ax.plot(R, scan.energy, "k-", lw=2, label="total power, truth")
    ax.plot(R, scan.dc_energy, "-", color="0.6", lw=2, label="DC term, truth")
    ax.plot(R, energy["total power"], "--", label="total power, halves")
    ax.plot(R, energy["DC term"], ":", label="DC term, halves")
    ax.plot(R, energy["squared tau-average"], "-.", label="squared τ-average, halves")
    ax.set_xlabel("x (cells)")
    ax.set_ylabel("energy over τ and T")
    ax.set_title("a magnitude-like image")
    ax.legend(fontsize=7)
    ax = axes[1]
    background = ~OBJECT
    ax.plot(
        R[background], energy["DC term, plain"][background], "o", ms=3, label="plain sum of squares"
    )
    ax.plot(
        R[background], energy["DC term"][background], "o", ms=3, label="products between halves"
    )
    ax.axhline(0, color="k", lw=1)
    ax.set_xlabel("x (cells), outside the object")
    ax.set_ylabel("DC term")
    ax.set_title("the noise floor")
    ax.legend(fontsize=7)
    fig.savefig(f"{figure_dir}/running-total-power.png", **save)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
    for ax, k, title, unit, scale in (
        (axes[0], 1, "R2* (BOLD)", "RMS ΔR2* in the band (1/s)", 1.0),
        (axes[1], 0, "S0 (non-BOLD)", "RMS ΔS0 in the band (%)", 100.0),
    ):
        for label, power, style in (
            ("truth", true_power, "k-"),
            ("τ-resolved static subtracted", estimates[True][0], "--"),
            ("τ-averaged static subtracted", estimates[False][0], ":"),
        ):
            values = scale * np.sign(power[k]) * np.sqrt(np.abs(power[k]))
            ax.plot(R[OBJECT], values[OBJECT], style, label=label)
        ax.set_xlabel("x (cells)")
        ax.set_ylabel(unit)
        ax.set_title(f"band power, the {title} direction")
        ax.legend(fontsize=7)
    fig.savefig(f"{figure_dir}/running-band-power.png", **save)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3.6), constrained_layout=True)
    ax.plot(lags, truth_lag, "k-", lw=2, label="truth")
    ax.plot(lags, estimate_lag, "--", label="products between halves")
    ax.axvline(LAG_S, color="0.6", lw=1)
    ax.set_xlabel("lag of C after A (s)")
    ax.set_ylabel("correlation of ΔR2* in the band")
    ax.set_title("lagged correlation, A and C")
    ax.legend(fontsize=7)
    fig.savefig(f"{figure_dir}/running-lag.png", **save)
    plt.close(fig)


if __name__ == "__main__":
    main(sys.argv[sys.argv.index("--figures") + 1] if "--figures" in sys.argv else None)
