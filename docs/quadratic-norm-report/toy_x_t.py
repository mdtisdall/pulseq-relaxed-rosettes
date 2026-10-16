"""1D x-T toy for docs/quadratic-norm-report.md (section 7).

Each TR samples S of the M DFT coefficients of a 1D image, uniformly at random
(or from a periodic set of patterns). The per-TR estimate is the
density-compensated adjoint, z_t = (M / S) F^H P_t^H y_t, which is unbiased
over the random pattern. The target is the band power over T,
q(r) = sum_j |c_j(r)|^2 with c_j = sum_t psi_j(t)^* x_t(r) for J Fourier
components psi_j in the band. Two estimators:

- plain: q = sum_j |sum_t psi_j(t)^* z_t|^2;
- U-statistic: the same, without the t = t' terms, q_U = q - (J / N) sum_t |z_t|^2.

``readout_timing`` drops the assumption that a TR is an instant: sample s of TR p
is at time p TR + tau_s, and the image decays with tau. ``energy_tau_t`` is the
energy over tau and T (application 1); ``lagged_cross`` the lagged correlation
between regions (application 3).

Run: python docs/quadratic-norm-report/toy_x_t.py (about a minute). Seeded.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

M = 128  # cells
N = 2000  # TRs
BAND = np.arange(5, 40)  # DFT bins of the band over T (J = 35)
TRIALS = 30

Complex = NDArray[np.complex128]


def temporal_basis() -> Complex:
    t = np.arange(N)
    basis: Complex = np.exp(2j * np.pi * np.outer(t, BAND) / N) / np.sqrt(N)
    return basis


def band_signal(rng: np.random.Generator, rms: float) -> Complex:
    """A random time course with all its power in the band, with the given RMS."""
    coefficients = rng.standard_normal(len(BAND)) + 1j * rng.standard_normal(len(BAND))
    signal: Complex = temporal_basis() @ coefficients
    return signal * (rms / np.sqrt(np.mean(np.abs(signal) ** 2)))


def masks(rng: np.random.Generator, samples: int, period: int) -> NDArray[np.bool_]:
    """(N, M) sampling masks: independent per TR, or cycling through ``period`` patterns."""
    patterns = [rng.choice(M, samples, replace=False) for _ in range(period or N)]
    out = np.zeros((N, M), bool)
    for t in range(N):
        out[t, patterns[t % len(patterns)]] = True
    return out


def estimates(
    x: Complex,
    rng: np.random.Generator,
    samples: int,
    period: int = 0,
    noise: float = 0.0,
    subtract_static: bool = True,
) -> tuple[Complex, Complex, Complex]:
    """One trial: band components C (J, M), per-TR estimates Z (N, M), and the masks' energy."""
    k = np.fft.fft(x, axis=1, norm="ortho")
    mask = masks(rng, samples, period)
    noisy = k + noise * (
        rng.standard_normal(k.shape) + 1j * rng.standard_normal(k.shape)
    ) / np.sqrt(2)
    y = np.where(mask, noisy, 0)
    if subtract_static:
        # The static image from all TRs (the mean of each coefficient over the TRs
        # that sampled it), subtracted from every TR's data.
        static = y.sum(0) / np.maximum(mask.sum(0), 1)
        y = np.where(mask, y - static[None], 0)
    z: Complex = (M / samples) * np.fft.ifft(y, axis=1, norm="ortho")
    c: Complex = temporal_basis().conj().T @ z
    return c, z, mask


def scaling() -> None:
    """Leakage against the undersampling of one TR; the residual has all its power in the band."""
    rng = np.random.default_rng(1)
    r = np.arange(M)
    region = np.abs(r - M / 2) < 20
    x = np.zeros((N, M), complex)
    for i in np.flatnonzero(region):
        x[:, i] = band_signal(rng, 0.01)
    psi = temporal_basis()
    q_true = np.sum(np.abs(psi.conj().T @ x) ** 2, 0)
    energy_true = np.sum(np.abs(x) ** 2, 0)
    j = len(BAND)
    print("Leakage against rho = M / S - 1 (residual only, no static image)")
    print(" S   rho   q bias  predicted   q_U bias  q_U std   E / (E_true + rho total / M)")
    for samples in (8, 16, 32, 64, 96):
        rho = M / samples - 1
        q, qu, e = [], [], []
        for _ in range(TRIALS):
            c, z, _ = estimates(x, rng, samples, subtract_static=False)
            qq = np.sum(np.abs(c) ** 2, 0)
            ee = np.sum(np.abs(z) ** 2, 0)
            q.append(qq)
            qu.append(qq - (j / N) * ee)
            e.append(ee)
        qa, qua, ea = np.array(q), np.array(qu), np.array(e)
        ref = q_true[region].mean()
        predicted = j * (rho / M) * energy_true.sum() / N
        energy_model = energy_true + rho * energy_true.sum() / M
        print(
            f"{samples:3d} {rho:5.2f}  {(qa.mean(0) - q_true)[region].mean() / ref:+.3f}"
            f"  {predicted / ref:+.3f}     {(qua.mean(0) - q_true)[region].mean() / ref:+.3f}"
            f"    {qua.std(0)[region].mean() / ref:.3f}"
            f"    {(ea.mean(0)[region] / energy_model[region]).mean():.3f}"
        )


def object_cases() -> None:
    """A static image, band fluctuations in two regions, and an out-of-band global oscillation."""
    rng = np.random.default_rng(0)
    r = np.arange(M)
    t = np.arange(N)
    static = (np.abs(r - M / 2) < M / 4) * (1 + 0.3 * np.cos(2 * np.pi * r / 16))
    region_a = np.abs(r - M / 2 - 10) < 6
    region_b = np.abs(r - M / 2 + 20) < 4
    outside = ~(region_a | region_b)
    respiration = 0.02 * np.exp(2j * np.pi * 300 * t / N)  # out of the band, 2% of the image
    x = (
        static[None] * (1 + respiration[:, None])
        + region_a[None] * band_signal(rng, 0.01)[:, None]
        + region_b[None] * band_signal(rng, 0.01)[:, None]
    )
    psi = temporal_basis()
    c_true = psi.conj().T @ x
    q_true = np.sum(np.abs(c_true) ** 2, 0)
    seed_a, seed_b = M // 2 + 10, M // 2 - 20
    cross_true = np.sum(c_true[:, seed_a].conj() * c_true[:, seed_b])
    j = len(BAND)
    cases = [
        ("random, static not subtracted", {"subtract_static": False}),
        ("random", {}),
        ("random, with noise", {"noise": 0.05}),
    ]
    print("\nBand power (bins 5-39) relative to the true band power in region A; S = 32 (rho = 3)")
    for label, options in cases:
        q, qu, cr, cru = [], [], [], []
        for _ in range(TRIALS):
            c, z, _ = estimates(x, rng, 32, **options)  # type: ignore[arg-type]
            qq = np.sum(np.abs(c) ** 2, 0)
            q.append(qq)
            qu.append(qq - (j / N) * np.sum(np.abs(z) ** 2, 0))
            cross = np.sum(c[:, seed_a].conj() * c[:, seed_b])
            cr.append(cross)
            cru.append(cross - (j / N) * np.sum(z[:, seed_a].conj() * z[:, seed_b]))
        ref = q_true[region_a].mean()
        print(f"== {label}")
        for name, values in (("q", np.array(q)), ("q_U", np.array(qu))):
            print(
                f"   {name:4s} region A mean {values.mean(0)[region_a].mean() / ref:8.3f}"
                f"  std {values.std(0)[region_a].mean() / ref:6.3f}"
                f"  | outside mean {values.mean(0)[outside].mean() / ref:8.3f}"
            )
        for name, values in (("cross", np.array(cr)), ("cross_U", np.array(cru))):
            print(
                f"   {name:7s} A-B cross-power: true {abs(cross_true):.4f}"
                f"  mean {abs(values.mean()):.4f}  std {values.std():.4f}"
            )


def leakage_spectrum() -> None:
    """Where the aliasing of a static image goes over temporal frequency, for each schedule."""
    rng = np.random.default_rng(2)
    r = np.arange(M)
    static = (np.abs(r - M / 2) < M / 4) * (1 + 0.3 * np.cos(2 * np.pi * r / 16))
    x = np.broadcast_to(static, (N, M)).astype(complex)
    k = np.fft.fft(x, axis=1, norm="ortho")
    samples = 32
    print("\nLeakage of a static image over temporal frequency (all N bins, no subtraction),")
    print("relative to the mean over bins 1..N-1, at the voxels outside the object")
    print("schedule        band (5-39)  multiples of N/period  other bins")
    outside = np.abs(r - M / 2) >= M / 4
    for period in (0, 10, 100):
        power = np.zeros(N)
        for _ in range(TRIALS):
            mask = masks(rng, samples, period)
            z = (M / samples) * np.fft.ifft(np.where(mask, k, 0), axis=1, norm="ortho")
            spectrum = np.fft.fft(z[:, outside], axis=0, norm="ortho")
            power += np.mean(np.abs(spectrum) ** 2, axis=1)
        mean = power[1:].mean()
        bins = np.arange(N)
        harmonic = (bins % (N // period) == 0) if period else np.zeros(N, bool)
        harmonic[0] = False
        band = np.isin(bins, BAND)
        other = ~harmonic & ~band & (bins > 0)
        label = f"periodic {period:3d}" if period else "random      "
        harmonic_text = f"{power[harmonic].mean() / mean:8.2f}" if harmonic.any() else "       -"
        print(
            f"{label}    {power[band].mean() / mean:8.3f}     {harmonic_text}"
            f"              {power[other].mean() / mean:8.3f}"
        )


def readout_timing() -> None:
    """Each TR is a 45 ms readout in time, and the static image decays over tau."""
    rng = np.random.default_rng(3)
    samples, tr_s, readout_s, r2star = 32, 0.05, 0.045, 20.0
    taus = (np.arange(samples) + 0.5) * readout_s / samples
    r = np.arange(M)
    static = (np.abs(r - M / 2) < M / 4) * (1 + 0.3 * np.cos(2 * np.pi * r / 16))
    region = np.abs(r - M / 2 - 10) < 6
    rest = (np.abs(r - M / 2) < M / 4) & ~region
    band_hz = BAND / (N * tr_s)  # 0.05-0.4 Hz, in absolute time
    coefficients = rng.standard_normal(len(band_hz)) + 1j * rng.standard_normal(len(band_hz))

    def fluctuation(t: NDArray[np.float64]) -> Complex:
        out: Complex = np.exp(2j * np.pi * np.multiply.outer(t, band_hz)) @ coefficients
        return out

    scale = 0.01 / np.sqrt(np.mean(np.abs(fluctuation(np.arange(N) * tr_s)) ** 2))
    # The truth on a fine grid in absolute time, normalized to N points.
    fine = np.arange(20 * N) * tr_s / 20
    basis_fine = np.exp(2j * np.pi * np.outer(fine, band_hz)) / np.sqrt(len(fine))
    q_true = np.sum(np.abs(basis_fine.conj().T @ (scale * fluctuation(fine))) ** 2) * N / len(fine)
    dft = np.exp(-2j * np.pi * np.outer(np.arange(M), r) / M) / np.sqrt(M)
    print("\nReadout timing: TR 50 ms, readout 45 ms, R2* 20/s; band power relative to the truth")
    print("static model     basis time   q_U region (std)   q_U rest of object   q region")
    for model in ("mean", "tau"):
        for basis_time in ("TR", "sample"):
            q, qu = [], []
            for _ in range(6):
                k = np.array([rng.choice(M, samples, replace=False) for _ in range(N)])
                t = np.arange(N)[:, None] * tr_s + taus[None]
                y = np.empty((N, samples), complex)
                for p in range(N):
                    image = (
                        static[None] * np.exp(-r2star * taus)[:, None]
                        + region[None] * (scale * fluctuation(t[p]))[:, None]
                    )
                    y[p] = np.sum(dft[k[p]] * image, axis=1)
                # The static model for each coefficient over all the TRs that sampled it:
                # its mean, or a quadratic in tau (a static image resolved in tau).
                tau = np.broadcast_to(taus, (N, samples))
                residual = np.empty_like(y)
                for i in range(M):
                    sel = k == i
                    design = np.stack([np.ones(sel.sum())] + [tau[sel] ** n for n in (1, 2)], 1)
                    if model == "mean":
                        design = design[:, :1]
                    fit, *_ = np.linalg.lstsq(design, y[sel], rcond=None)
                    residual[sel] = y[sel] - design @ fit
                times = t if basis_time == "sample" else np.broadcast_to(t[:, :1], t.shape)
                weights = np.exp(-2j * np.pi * times[..., None] * band_hz) / np.sqrt(N)
                inverse = np.exp(2j * np.pi * k[..., None] * r / M) / np.sqrt(M)
                per_tr = (M / samples) * np.einsum("psj,ps,psm->pjm", weights, residual, inverse)
                qq = np.sum(np.abs(per_tr.sum(0)) ** 2, 0)
                q.append(qq)
                qu.append(qq - np.sum(np.abs(per_tr) ** 2, axis=(0, 1)))
            qa, qua = np.array(q), np.array(qu)
            region_mean = qua.mean(0)[region].mean() / q_true
            region_std = qua.std(0)[region].mean() / q_true
            rest_mean = qua.mean(0)[rest].mean() / q_true
            plain = qa.mean(0)[region].mean() / q_true
            print(
                f"{model:4s}             {basis_time:6s}"
                f"       {region_mean:6.3f} ({region_std:.3f})"
                f"      {rest_mean:7.3f}              {plain:7.3f}"
            )


def energy_tau_t() -> None:
    """The energy over tau and T of a decaying image: the DC term and the total power."""
    rng = np.random.default_rng(5)
    samples, readout_s, n_tr, half_width = 32, 0.045, 1000, 5
    taus = (np.arange(samples) + 0.5) * readout_s / samples
    d_tau = readout_s / samples
    r = np.arange(M)
    density = (np.abs(r - M / 2) < M / 4) * (1 + 0.3 * np.cos(2 * np.pi * r / 16))
    obj = density > 0
    r2star = 20 + 30 * r / M
    field_hz = 60 * np.sin(2 * np.pi * r / M)  # known, and corrected
    x = density[:, None] * np.exp((-r2star[:, None] + 2j * np.pi * field_hz[:, None]) * taus)
    k_space = np.fft.fft(x, axis=0, norm="ortho")  # (M, samples)
    # The total power is invariant to a phase that changes over T, so it is the same truth
    # for every case.
    energy = n_tr * d_tau * np.sum(np.abs(x) ** 2, 1)
    # A real tau basis for the decays (the field is demodulated first).
    decays = np.exp(-np.outer(taus, np.linspace(10, 60, 11)))
    u_basis, singular, _ = np.linalg.svd(decays, full_matrices=False)
    rank = int(np.searchsorted(np.cumsum(singular**2) / np.sum(singular**2), 1 - 1e-5)) + 1
    basis = u_basis[:, :rank]
    # Fourier components over T for the total power: |j| <= half_width bins.
    bins = np.arange(-half_width, half_width + 1)
    psi = np.exp(2j * np.pi * np.outer(np.arange(n_tr), bins) / n_tr) / np.sqrt(n_tr)
    j_over_n = len(bins) / n_tr
    print(
        f"\nEnergy over tau and T: {n_tr} TRs, {samples} samples per TR, tau basis of rank {rank}"
    )
    print("noise  drift   estimator                  object mean (std)   background")
    for noise, cycles in ((0.0, 0.0), (0.3, 0.0), (0.0, 0.5)):
        drift = np.exp(2j * np.pi * cycles * np.arange(n_tr) / n_tr)  # a phase over T
        results: dict[str, list[NDArray[np.float64]]] = {
            "squared tau-average": [],
            "DC term, plain": [],
            "DC term": [],
            "total power": [],
            "DC term, halves": [],
            "total power, halves": [],
        }
        for _ in range(32):
            k = np.array([rng.choice(M, samples, replace=False) for _ in range(n_tr)])
            shape = (n_tr, samples)
            white = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
            y = k_space[k, np.arange(samples)[None]] * drift[:, None] + noise * white / np.sqrt(2)
            # Each sample alone gives an unbiased image at its tau (k uniform over the TRs).
            phase = np.exp(2j * np.pi * k[..., None] * r / M) / np.sqrt(M)
            u = M * y[..., None] * phase
            u = u * np.exp(-2j * np.pi * field_hz * taus[:, None])[None]  # demodulate the field
            mean_tau = u.mean(1)
            results["squared tau-average"].append(
                n_tr
                * readout_s
                * (np.abs(mean_tau.sum(0)) ** 2 - np.sum(np.abs(mean_tau) ** 2, 0))
                / (n_tr * (n_tr - 1))
            )
            z = np.einsum("sl,psm->plm", basis.conj(), u)  # (TRs, rank, M)
            dc = np.abs(z.sum(0)) ** 2
            within = np.sum(np.abs(z) ** 2, 0)
            results["DC term, plain"].append(n_tr * d_tau * np.sum(dc, 0) / n_tr**2)
            results["DC term"].append(n_tr * d_tau * np.sum(dc - within, 0) / (n_tr * (n_tr - 1)))
            components = np.einsum("pj,plm->jlm", psi.conj(), z)
            band = np.sum(np.abs(components) ** 2, axis=(0, 1)) - j_over_n * np.sum(within, 0)
            results["total power"].append(d_tau * band / (1 - j_over_n))
            # Interleaved halves (odd and even TRs): both span the scan; keep the products
            # between them.
            halves = [2 * np.einsum("pj,plm->jlm", psi[h::2].conj(), z[h::2]) for h in (0, 1)]
            cross = np.real(halves[0].conj() * halves[1])
            results["DC term, halves"].append(d_tau * np.sum(cross[half_width], 0))
            results["total power, halves"].append(d_tau * np.sum(cross, axis=(0, 1)))
        for name, values in results.items():
            v = np.array(values)
            ratio = v.mean(0)[obj] / energy[obj]
            spread = v.std(0)[obj] / energy[obj]
            floor = v.mean(0)[~obj].mean() / energy[obj].mean()
            print(
                f"{noise:4.1f}   {cycles:4.1f}    {name:22s}     {ratio.mean():6.3f}"
                f" ({spread.mean():.3f})      {floor:+.4f}"
            )


def lagged_cross() -> None:
    """Lagged correlation between regions from the U-statistic cross-spectrum."""
    rng = np.random.default_rng(7)
    tr_s, samples, lag_s = 0.05, 32, 0.6
    r = np.arange(M)
    t = np.arange(N)
    band_hz = BAND / (N * tr_s)
    static = (np.abs(r - M / 2) < M / 4) * (1 + 0.3 * np.cos(2 * np.pi * r / 16))
    regions = {name: np.abs(r - M / 2 - offset) < 4 for name, offset in (("A", 10), ("B", -20))}
    regions["C"] = np.abs(r - M / 2 + 2) < 4
    coefficients = rng.standard_normal(len(BAND)) + 1j * rng.standard_normal(len(BAND))
    independent = rng.standard_normal(len(BAND)) + 1j * rng.standard_normal(len(BAND))

    def course(c: Complex, delay: float) -> Complex:
        out: Complex = np.exp(2j * np.pi * np.outer(t * tr_s - delay, band_hz)) @ c
        return out * (0.01 / np.sqrt(np.mean(np.abs(out) ** 2)))

    respiration = 0.02 * np.exp(2j * np.pi * 300 * t / N)
    x = (
        static[None] * (1 + respiration[:, None])
        + regions["A"][None] * course(coefficients, 0.0)[:, None]
        + regions["B"][None] * course(coefficients, lag_s)[:, None]
        + regions["C"][None] * course(independent, 0.0)[:, None]
    )
    psi = temporal_basis()
    lags = np.arange(-3, 3.001, 0.05)
    shift = np.exp(2j * np.pi * np.outer(band_hz, lags))
    voxel = {"A": M // 2 + 10, "B": M // 2 - 20, "C": M // 2 - 2}
    truth_c = psi.conj().T @ (x - x.mean(0))

    def correlation(c: Complex, z: Complex | None, u: int, v: int) -> NDArray[np.float64]:
        cross = c[:, u].conj() * c[:, v]
        power_u = np.sum(np.abs(c[:, u]) ** 2)
        power_v = np.sum(np.abs(c[:, v]) ** 2)
        if z is not None:  # leave out each TR's product with itself
            w = np.abs(psi) ** 2
            cross = cross - w.T @ (z[:, u].conj() * z[:, v])
            power_u -= np.sum(w.T @ np.abs(z[:, u]) ** 2)
            power_v -= np.sum(w.T @ np.abs(z[:, v]) ** 2)
        out: NDArray[np.float64] = np.real(cross @ shift) / np.sqrt(abs(power_u * power_v))
        return out

    print(f"\nLagged correlation, band 0.05-0.4 Hz, S = {samples}; B is A delayed by {lag_s} s")
    print("pair   true peak (lag)      estimated peak (lag)   max |error| over lags, mean (std)")
    for pair in (("A", "B"), ("A", "C")):
        u, v = voxel[pair[0]], voxel[pair[1]]
        truth = correlation(truth_c, None, u, v)
        estimates_r, errors = [], []
        for _ in range(12):
            c, z, _ = estimates(x, rng, samples)
            est = correlation(c, z, u, v)
            estimates_r.append(est)
            errors.append(np.abs(est - truth).max())
        mean_r = np.mean(estimates_r, 0)
        i_true, i_est = np.argmax(np.abs(truth)), np.argmax(np.abs(mean_r))
        print(
            f"{pair[0]}-{pair[1]}    {truth[i_true]:+.3f} ({lags[i_true]:+.2f} s)"
            f"       {mean_r[i_est]:+.3f} ({lags[i_est]:+.2f} s)"
            f"        {np.mean(errors):.3f} ({np.std(errors):.3f})"
        )


if __name__ == "__main__":
    scaling()
    object_cases()
    leakage_spectrum()
    readout_timing()
    energy_tau_t()
    lagged_cross()
