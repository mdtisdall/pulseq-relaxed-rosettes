# Tip-radius density: derivation and use

The relaxed rosette visits `echoes_per_tr` petal tips per TR, crossing $k = 0$ between them. Where the tips are, in radius, decides how the trajectory's samples are spread over k-space, and so which reconstructions the data supports. This note derives the tip-radius distribution from a stated goal, instead of choosing it by hand, in two ways:

- **Resolution–time exponent** (`tip_density = "exponent"`): make the time a window of scan time needs to cover each resolution follow a chosen power law.
- **SNR-optimal for filtered reconstructions** (`tip_density = "filter"`): minimize the noise of a weighted set of retrospective reconstructions at different resolutions.

It then compares the designs, including an idea that did not work (section 5). All radii are in units of $k_{max} = 1/(2\,\Delta x)$.

## 1. From tips to samples

Each petal goes from $k = 0$ to its tip radius $R$ and back, so it crosses every radius $\rho < R$ twice. Let $T(\rho)$ be the fraction of tips with $R > \rho$ (the tip *survival function*) and $v(\rho)$ the speed of the trajectory in $|k|$ at radius $\rho$. The number of ADC samples per unit radius is then

$$
n(\rho) \;\propto\; \frac{T(\rho)}{v(\rho)} .
\qquad (1)
$$

Inverting, a target sample profile $n(\rho)$ fixes the tips:

$$
T(\rho) \;\propto\; n(\rho)\, v(\rho), \qquad f(R) = -\frac{dT}{dR},
\qquad (2)
$$

where $f$ is the tip-radius density. Tips are drawn from it by inverting the CDF $1 - T$: a quantile $q \in [0, 1)$ maps to the radius where $1 - T = q$.

**What is achievable.** $T$ is a survival function, so it can only decrease with $\rho$: every petal passes through the center, and the center is crossed at least as often as any other radius. The achievable profiles are those with $n v$ non-increasing. A target that increases somewhere (for example a uniform volumetric density, $n \propto \rho^2$) is replaced by its smallest non-increasing majorant: the trajectory samples at least the target everywhere, and more near the center, which it cannot avoid. The flattest achievable profile is $n v$ = constant, which puts every tip at the maximum radius (a 3D radial-like density, $\propto 1/\rho^2$ volumetrically).

**The power model as a special case.** The original design (`tip_density = "power"`) draws blue noise $u$ uniformly and sets $R = u^p$. That is blue noise in the quantile of $f(R) \propto R^{1/p - 1}$; choosing $p$ chooses a density, without saying what the density is for. Blue noise in the quantile of any $f$ is the same as blue noise under the metric $d(R_1, R_2) = |F(R_1) - F(R_2)|$, with $F$ the CDF of $f$. The two models below choose $f$ from the goal.

## 2. Model A: the resolution–time exponent

For retrospective reconstruction at any resolution and time scale, a useful summary of the acquisition is $W(\text{res})$: the scan time a window needs before it holds as many samples inside the k-space ball of a resolution as the ball has Nyquist points. The ball of resolution $\text{res}$ has radius $\rho = \Delta x/\text{res}$ and about $\rho^3$ Nyquist points (times a constant). The samples per unit time inside it are $\int_0^\rho n$. So

$$
W(\text{res}) \;\propto\; \frac{\rho^3}{\int_0^\rho n(\rho')\,d\rho'} .
$$

Asking for $W(\text{res}) \propto \text{res}^{-\alpha}$, that is $W \propto \rho^{\alpha}$, requires $\int_0^\rho n \propto \rho^{3 - \alpha}$, and so

$$
n(\rho) \;\propto\; \rho^{\,2 - \alpha}, \qquad 2 \le \alpha < 3 .
\qquad (3)
$$

The bounds come from achievability: $n v$ must not increase, so $\alpha \ge 2$ (with $v$ constant); and $\int_0^\rho n$ must be finite, so $\alpha < 3$.

- $\alpha = 2$: $n$ constant, all tips at $k_{max}$. Fine scales fill fastest relative to their Nyquist count.
- $\alpha \to 3$: $n \propto 1/\rho$; every resolution fills its own ball in the same time. The strongest center weighting.
- Between: halving the voxel size needs $2^\alpha$ times longer.

The law is applied only over the resolutions of interest, from `density_resolution_min_m` (finest) to `density_resolution_max_m` (coarsest), that is $\rho \in [\rho_{lo}, \rho_{hi}] = [\Delta x/\text{res}_{max},\ \Delta x/\text{res}_{min}]$. Beyond $\rho_{hi}$ the target is 0 (no tips). Inside $\rho_{lo}$ the shape of $n$ does not matter, but for the law to hold at the coarsest resolution the ball must still hold its share of samples, $\int_0^{\rho_{lo}} \rho^{2-\alpha}\, d\rho = \rho_{lo}^{3-\alpha}/(3-\alpha)$. The target there is flat at that total, $n = \rho_{lo}^{2-\alpha}/(3-\alpha)$, which is above the law at $\rho_{lo}$ and so achievable; it puts a share of the tips exactly at $\rho_{lo}$. (A first version made the inner target flat at the law's value at $\rho_{lo}$; that gives the ball only a $(3 - \alpha)$ share of the samples it needs, and the achieved α fell well short of the target.)

**Use.** One number, $\alpha$, sets how fast fine detail fills in compared with coarse detail, which is the trade-off retrospective 5D reconstruction chooses along. It does not say anything about noise.

## 3. Model B: SNR-optimal for a set of filtered reconstructions

Consider a reconstruction at resolution $r$ that applies a k-space filter $w_r(\mathbf{k})$ (here a Hann window in $|k|$ that reaches 0 at $1/(2r)$). With $D(\mathbf{k})$ samples per unit k-space volume, each with noise variance $\sigma^2$, and density compensation, the noise variance of the reconstruction is

$$
\sigma_r^2 \;\propto\; \int \frac{w_r(\mathbf{k})^2}{D(\mathbf{k})}\, d^3k .
$$

The resolution is chosen after the scan, so weight each resolution by $\pi(r)$ (how much that reconstruction matters) and minimize the weighted total for a fixed number of samples $\int D = N$:

$$
\min_D \;\sum_r \pi(r) \int \frac{w_r^2}{D}\, d^3k
\quad \text{subject to} \quad \int D\, d^3k = N .
$$

The Lagrangian condition $-\sum_r \pi(r)\, w_r^2 / D^2 + \lambda = 0$ gives

$$
D(\mathbf{k}) \;\propto\; \sqrt{\sum_r \pi(r)\, w_r(\mathbf{k})^2} .
\qquad (4)
$$

For a single reconstruction this is $D \propto |w|$, the classic acquisition-weighting result (sample where the filter weights the data). For many it is the root-mean-square filter. The radial profile is $n(\rho) = 4\pi\rho^2 D(\rho)$, made achievable with the majorant of section 1.

The resolutions are log-spaced between `density_resolution_min_m` and `density_resolution_max_m`, with $\pi(r) \propto r^\beta$ per octave (`filter_prior_power` $= \beta$; 0 makes every octave equally important). In 5D reconstruction, coarse resolutions are reconstructed from short windows, so their noise per window is larger by the ratio of window lengths; to weight each scale's *per-window* noise equally, fold the window length into $\pi(r)$, i.e. $\pi(r) \propto 1/W(r)$.

**A consequence of the filter.** A Hann window reaches 0 at its cutoff, so even the finest reconstruction gives no weight to samples at $k_{max}$, and the optimal density there is 0: this model puts no tips at $k_{max}$. A reconstruction that keeps more weight near its cutoff (a flatter window) would move tips outward.

**Use.** It turns a statement about reconstructions ("these resolutions, this much each, with this filter") into the sampling that minimizes their noise. It does not say anything about the time scale each resolution is reconstructed at, except through $\pi(r)$.

## 4. Calibrating the speed

Equation (2) needs $v(\rho)$, which depends on the optimized waveforms: the path slows at the petal tips (it turns around) and is fastest through the center. It is estimated from the shapes themselves (`relaxed_rosettes.density.estimate_speed`): for each radius, the number of samples per unit radius and the number of times the sampled path crosses that radius, both counted from the ADC samples; their ratio is proportional to $v$. Starting from $v$ = constant, the design optimizes `calibration_shapes` shapes, estimates $v$, recomputes the tips, and repeats `calibration_iterations` times (a fixed-point iteration). Calibration is deterministic, so a design is still reproducible from its config.

## 5. Tested and rejected: stratifying radii across shapes

Rotations do not change $|k|(t)$, so the plane of $(|k|,\ \tau)$, with $\tau$ the time since excitation, is sampled only by the shapes, however many TRs are played. If each shape draws its radii independently, nothing stops many shapes from reaching similar radii in the same time slot, which could leave bands in that plane. The idea was to assign the radius quantile of tip slot $j$ of shape $s$ from a two-dimensional Kronecker lattice, $q_{s,j} = \mathrm{frac}(q_0 + s/g + j/g^2)$ with $g$ the plastic number, so every time slot sees the whole radius distribution across shapes.

It made the $(|k|, \tau)$ sampling slightly *worse* in every case tried (section 6), and was removed. With 128 independently drawn shapes, cells later than 5 ms are already about 99% filled; the remaining empty cells are early (below), where the lattice does not help.

## 6. Results

All designs: 19 echoes per TR, 128 shapes, TR 50 ms, 1 mm resolution, 256 mm FOV, the gradient limits of the default config, and `density_resolution_min_m` / `max_m` = 1 / 8 mm for the derived models. Columns:

- **α measured**: the resolution–time exponent achieved, from $W(\text{res}) \propto \rho^3 / (\text{share of samples inside } \rho)$ fitted over 1–8 mm.
- **Slew**: peak slew over the shapes (T/m/s), a rough proxy for acoustic noise and PNS.
- **In ball**: share of ADC samples inside the k-space ball of 8 / 4 / 2 mm.
- **Noise**: noise variance of a Hann-filtered reconstruction at 1 / 2 / 4 / 8 mm, per sample, relative to the baseline (lower is better; equation (4)'s objective, before undersampling).
- **Gap**: the 95th-percentile distance (in $1/\text{FOV}$ cells) from the k-space ball to the nearest sample, for a window of scan time: 4 mm / 2 s, 2 mm / 10 s, 1 mm / 60 s (`relaxed-rosettes coverage`).
- **Empty $(|k|, \tau)$**: share of empty cells (20 $|k|$ shells × 0.5 ms $\tau$ bins) among those a gradient ramping as fast as allowed could reach (99% of all cells), before / after $\tau$ = 5 ms.

| Design | α measured | Slew | In ball 8 / 4 / 2 mm | Noise 1 / 2 / 4 / 8 mm | Gap 4 mm / 2 mm / 1 mm | Empty $(\|k\|,\tau)$ early / late |
|---|---|---|---|---|---|---|
| power 0.5 (baseline, default) | 2.11 | 37 | 16 / 30 / 59% | 1.00 / 1.00 / 1.00 / 1.00 | 2.2 / 2.1 / 3.6 | 9.2% / 1.1% |
| exponent α = 2.25 | 2.41 | 27 | 29 / 45 / 67% | 1.41 / 1.13 / 0.68 / 0.57 | 2.6 / 2.6 / 3.4 | 10.3% / 1.2% |
| exponent α = 2.5 | 2.56 | 20 | 40 / 56 / 75% | 1.76 / 1.25 / 0.61 / 0.43 | 3.2 / 3.1 / 4.3 | 14.1% / 1.4% |
| exponent α = 2.75 | 2.66 | 15 | 49 / 64 / 82% | 2.24 / 1.33 / 0.68 / 0.35 | 3.5 / 3.7 / 5.4 | 16.2% / 1.7% |
| filter (Hann, 1–8 mm, equal per octave) | 2.05 | 33 | 15 / 30 / 66% | 0.98 / 0.92 / 0.98 / 1.04 | 2.1 / 2.0 / 5.3 | 15.7% / 1.9% |
| power 0.5, stratified (rejected) | 2.09 | 36 | 15 / 30 / 59% | 0.99 / 0.98 / 0.94 / 1.01 | 2.1 / 2.0 / 3.9 | 11.9% / 1.4% |
| exponent α = 2.5, stratified (rejected) | 2.59 | 22 | 42 / 58 / 76% | 1.83 / 1.24 / 0.62 / 0.41 | 3.0 / 3.3 / 4.8 | 15.7% / 1.8% |

**The exponent model does what it is derived to do.** The achieved α is within about 0.15 of the target (2.5 → 2.56), and the designs trade continuously along it: from α = 2.25 to 2.75, the share of samples in the 8 mm ball goes from 29% to 49%, coarse-scale noise falls by up to 3× against the baseline, and 1 mm noise rises by up to 2.2×. The peak slew also falls (27 → 15 T/m/s), so these designs are quieter. One property of the tips: about a quarter of them sit exactly at the radius of the coarsest resolution of interest (0.125 $k_{max}$), which supplies the samples the 8 mm ball needs; those short petals shorten the path, which is why the mid-scale gaps grow.

**Calibration converges quickly.** For α = 2.5 measured on 32 shapes, the achieved α is 2.64 with no calibration and 2.46–2.56 after 1 to 5 iterations; the residual error of the profile (about 18% RMS) is local, from where the path slows at petal tips. Two iterations (the default) are enough.

**The hand-chosen power 0.5 is close to SNR-optimal for equal-octave Hann reconstructions.** The filter model, which minimizes exactly that noise, improves on it by at most 8% (at 2 mm). Because a Hann window reaches 0 at its cutoff, it places no tips near $k_{max}$, and its 1 mm coverage is worse (gap 5.3 against 3.6 cells). The filter model is most useful with the filters and weights of the actual reconstructions; with a flatter window or with $\pi(r) \propto 1/W(r)$ (section 3), it moves away from the baseline.

**Early relaxation times are the remaining gap in the $(|k|, \tau)$ plane.** After 5 ms, about 1–2% of reachable cells are empty in every design. Before 5 ms, 9–16% are, although a gradient ramping as fast as allowed reaches almost every cell within about 1 ms: every shape starts with the same gentle ramp (25 T/m/s for 1 ms) and a slow, smooth first segment. Designs that put more tips at small radii are worse there.

## 7. Using the models

`tip_density = "power"` remains the default, with power 0.5 (α ≈ 2.1): the best mid-scale coverage and near-optimal noise for equal-octave Hann reconstructions. Choose a derived model when the goal is stated more precisely:

- **Retrospective 5D, coarse scales in short windows:** `tip_density = "exponent"` with `resolution_time_exponent` 2.25–2.5, and `density_resolution_min_m` / `max_m` set to the finest and coarsest resolutions you will reconstruct. Each 0.25 of α buys about a third less coarse-scale noise for about a third more fine-scale noise, and lowers the peak slew.
- **A known set of reconstructions:** `tip_density = "filter"`, with the resolution range and `filter_prior_power` describing how much each resolution matters (to weight per-window noise in 5D, $\pi(r) \propto 1/W(r)$, i.e. a positive power).

For example, in a design TOML:

```toml
[shapes]
tip_density = "exponent"
resolution_time_exponent = 2.25
density_resolution_min_m = 0.001
density_resolution_max_m = 0.008
```

Then `relaxed-rosettes shapes design.toml -o shapes.npz` calibrates and optimizes as usual (the calibration adds about 15 s), and `relaxed-rosettes coverage` measures the result.

**Next steps suggested by the results:** (1) the first segment after the RF pulse sets the early-$\tau$ sampling; varying it more across shapes (timing, or the start of the gentle ramp) is the lever for the remaining banding; (2) the point mass of tips at the coarsest radius could be spread over smaller radii, trading some of the coarse-ball samples for mid-scale coverage.
