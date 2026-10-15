# Spreading the samples: relaxed targets, then repulsion

Status: step 1 is implemented (relaxed targets, the within-TR redundancy metric); steps 2 and 3 are plans. Measurements that corrected the plan are marked **Measured**. It follows the petal timing of PR #16 and uses the density models of [sampling-density.md](sampling-density.md).

The run times in this plan are estimates. As each step is implemented and run, record the measured times and results in section 6, in the same PR as the step, and correct the estimates in the text where they were wrong.

## Question

After PR #16, each shape fits as many petals as the gradient and slew limits allow (40–43 per TR at the default limits). The trajectory is still slow and wasteful in two ways:

- **Hairpins.** Each petal goes from $k = 0$ to a single tip point and back along almost the same line. The path must nearly stop at the tip, so it is slew-limited there: mean $|g|$ is 14.8 mT/m, about half of $g_{max}$.
- **Retracing.** The return leg covers nearly the same radial line as the outward leg, and every $k = 0$ crossing visits the same point. Many samples add little new coverage.

  **Measured (step 1):** this is much smaller than assumed. Only about 5% of a shape's samples come within one cell of an earlier pass (within-TR redundancy, section 1), for both the 19-tip and the PR #16 designs. At speed, the return leg is already more than a cell from the outward leg except near the tip.

How should the trajectory be changed so that its samples spread over k-space, and over time, as evenly as the target density allows?

## Short answer

In three steps. Each step is measured before the next one starts:

1. **Relaxed targets** (section 2). Replace the point targets with convex regions: a small ball around $k = 0$ for each crossing, and a region around each tip. Start from the hairpin solution. This is cheap, and it lets each petal become a loop.
2. **Repulsion within each shape** (section 3). Add an attraction–repulsion energy, as in SPARKLING, and minimize it by sequential convex programming that starts from the step 1 solution. Each step remains feasible.
3. **Repulsion between shapes, weighted by scan time** (section 4). Do this only if step 2 pays off. With the current schedule, most of what it could gain is already captured, so first try the cheaper alternative of choosing the rotations.

## 1. Baseline

All comparisons use `main` at PR #16 (merge commit `8e0280c`) as the baseline: 128 shapes, seed 0, the default config, and `relaxed-rosettes coverage` with its default windows. The gap is the 95th-percentile distance to the nearest sample, in Nyquist cells ($1/\text{FOV}$).

| design | tips/TR | path/TR | mean $\|g\|$ | peak $\|g\|$ | gap 4 mm / 1 s | gap 2 mm / 5 s | gap 1 mm / 30 s | empty $(\|k\|, \tau)$ cells |
|---|---|---|---|---|---|---|---|---|
| before #16 (19 tips, $\sqrt r$ + jitter) | 19 | 27 $k_{max}$ | 7.1 | 14.5 | 3.0 | 3.0 | 5.3 | 2.7% |
| #16 (fill per shape, margin 1.1) | 40–43 | 57 $k_{max}$ | 14.8 | 26.1 | 2.1 | 2.1 | 4.2 | 1.4% |

Gaps scale as (path length)$^{-1/3}$, as for any 3D sampling: twice the path gave 0.77× the gap. A change that only speeds up the path will follow the same law. A change that spreads the samples should do better than it.

**A metric to add.** Add **within-TR redundancy** to `coverage`: the fraction of a TR's ADC samples within one cell of an earlier sample of the same TR. It measures retracing directly. The hairpin design should score high, and steps 1 and 2 should lower it.

**Measured:** implemented as `within_tr_redundancy` in `relaxed_rosettes.coverage`, and printed by `coverage`. Neighbors along the path (closer than 3 cells of path) do not count. The hairpin designs score only about 5%, and relaxed targets do not lower it (the crossings still land within a cell or two of each other).

## 2. Step 1: relaxed targets

### 2.1 Constraints

All positions are in units of $k_{max}$. $k(t)$ is linear in the gradient samples $g$: the current optimizer already writes $k$ at a target time as $w^\top g$ (`k_weights`), and $k$ at every raster point is a scaled cumulative sum of $g$. Each constraint below is linear or a second-order cone in $g$, so with the slew-energy objective and fixed target times, each shape is still a convex SOCP solved with Clarabel, with a unique optimum.

- **Crossings.** $\lVert k(t_c) \rVert \le \varepsilon_c$, with $\varepsilon_c$ = `crossing_tolerance_cells` / (FOV $\cdot\, k_{max}$). Eddy currents already move the real crossings by a fraction of a cell. With multi-coil data, a sample within one or two cells of the center still gives the center signal for the navigator, B0 drift, physiological signals and the $\tau$ basis. A ball also spreads the 40-odd crossings of a TR over different cells, which should help low-resolution coverage in short windows. It does not speed up the path: at a crossing the path turns by at most 60°, and at full speed that turn is spread over about 0.7 $k_{max}$ of path, far more than a few cells.
- **Tips.** For tip $i$, with radius $r_i$ and direction $u_i$:
  - *reach*: $u_i \cdot k(t_i) \ge r_i$, a half-space;
  - *anchor*: $\lVert k(t_i) - r_i u_i^{(0)} \rVert \le w_i$, with $w_i$ = `tip_width` $\cdot\, r_i$ and $u_i^{(0)}$ the original blue-noise direction.

  The reach constraint is a convex inner approximation of $\lVert k \rVert \ge r_i$: since $u \cdot k \le \lVert k \rVert$, any point that meets it is at least as far out as $r_i$. The anchor keeps the tips' blue-noise spread in direction and radius.
- **The k-space ball.** $\lVert k(t) \rVert \le 1$ at every raster point in the ADC, so that wide turns near the edge do not leave the sphere. This adds one cone per raster point (about 4500 per shape).
- **Spoil moment, RF window, end of TR:** unchanged.

With `tip_width` = 0 and `crossing_tolerance_cells` = 0, these constraints are the current point targets, so the defaults keep today's behavior until the sweep has chosen new ones.

**What loops can and cannot do.** A petal that is a loop through (or near) $k = 0$ can turn with a radius up to about $r/2$, instead of stopping. The largest speed on a turn of radius $\rho$ is $g = \sqrt{S \rho / \gamma}$: at $r = 1$ ($\rho = 0.5\,k_{max}$, $S$ = 100 T/m/s) that is about 24 mT/m.

**Measured, and a correction:** loops do **not** make petals shorter. The component of the path along the tip direction must still go from near 0 out to $r$ and back, under the same gradient and slew limits, so the straight-path time of `petal_times_s` remains a lower bound (up to the small crossing tolerance). A loop puts more path into a petal of the same duration. With the slew-energy objective, the optimizer has no reason to use that freedom for more path, so path per TR rises only about 3%. Rewarding path or spreading is the job of step 2. The measured gain of step 1 is elsewhere: tips that can move within their anchor spread out, which improves fine-resolution coverage (section 6).

### 2.2 Timing

The petal time model of PR #16 (`petal_times_s`) assumes a straight out-and-back path, which is a lower bound only for point targets. For the relaxed problem:

- **Calibrate as now.** Find the largest, over the calibration shapes, of the smallest feasible factor on `petal_times_s`, by bisection from 1 (still a lower bound, see above) to the factor that fills the window.
  - **The hairpin as proof of feasibility.** Every hairpin waveform meets the relaxed constraints, because a point target lies inside its region. Near the edge of feasibility the solver sometimes fails on a feasible relaxed problem. **Measured:** without a guard this made one calibration shape need 1.166 at `tip_width` = 0.25, against 1.061 at 0.5 for the same shape, which is impossible for nested regions. When a relaxed solve fails, the bisection now tries the hairpin at the same timing, and counts the factor feasible if the hairpin is.
  - **Measured:** the per-shape factors fall only from about 1.09 (hairpin) to about 1.07 (`tip_width` = 0.5).
- **Fill per shape** at `segment_time_margin` × the relaxed factor, as now.
- **Later, optionally:** a petal time model for loops, so that the relative petal times stay accurate when `tip_width` changes.

### 2.3 Re-linearizing the tips

After a solve, a tip may sit off to the side of $u_i$, where the half-space is stricter than $\lVert k \rVert \ge r_i$. Set $u_i \leftarrow k(t_i)/\lVert k(t_i) \rVert$ and solve again (the anchor stays at $r_i u_i^{(0)}$). This is the convex–concave procedure:

- each round is a convex problem;
- the previous solution is feasible for it, so the objective never increases.

One or two rounds (`tip_relinearizations`) should be enough. The hairpin solution is the starting point, which is where it matters that the problem is not convex.

### 2.4 Optional tube

To keep the relaxed path close to the designed one (and so keep the radial density and the $(|k|, \tau)$ sampling as designed), add $\lVert k(t) - k_{hairpin}(t) \rVert \le \delta$ at every raster point, with $\delta$ = `tube_cells`. A first prototype of this constraint (with a lower bound on $|g|$, 2026-10-14) solved in about the same time as the base problem.

### 2.5 Config

| field | default | meaning |
|---|---|---|
| `crossing_tolerance_cells` | 0 | radius of the crossing ball, in cells |
| `tip_width` | 0 | radius of the tip anchor, as a fraction of the tip radius |
| `tip_relinearizations` | 0 | extra solves that re-linearize the reach constraint |
| `tube_cells` | 0 (off) | maximum distance from the hairpin path, in cells |

The hairpin solve is only needed when `tip_width`, `crossing_tolerance_cells` or `tube_cells` is non-zero.

### 2.6 Experiments

Sweep a subset of the combinations rather than all 12:

- `tip_width` ∈ {0, 0.1, 0.25, 0.5} at `crossing_tolerance_cells` = 1;
- then `crossing_tolerance_cells` ∈ {0, 2, 4} at the best `tip_width`;
- `tip_relinearizations` ∈ {0, 2} at the best pair.

That is about 8 designs. For each, report tips/TR, path/TR, mean and peak $|g|$, within-TR redundancy, the coverage gaps (including 8 mm / 0.1 s and 4 mm / 0.2 s, the short windows that the crossing ball should help), the $(|k|, \tau)$ empty cells, and `check` (slew, gradient spectrum). A design takes about 10 min (hairpin solve, calibration, relaxed solve); the sweep takes about 1.5 h.

**Measured:** only the first part was run: `tip_width` ∈ {0, 0.1, 0.25, 0.5} at `crossing_tolerance_cells` = 1 (section 6). A design took 6–7.5 min (no hairpin solve is needed without a tube). A crossing tolerance of 1 cell alone changed nothing measurable, and re-linearization only lowers slew energy, so the second and third parts (other crossing tolerances, re-linearization) were not run. The defaults stay at 0; choosing new ones is left to a follow-up.

### 2.7 Risks

- **Solve time.** The k-space ball adds about 4500 cones per shape. If solves slow down too much, apply it only where the hairpin solution comes within one tip width of $k_{max}$.
- **Radial density.** Loops spend more time at large radius, which shifts the radial sample density. The derived density models (`exponent`, `filter`) recalibrate the speed from the samples, so they absorb this. The `power` model does not.
- **The navigator tolerance is not measured here.** How close to $k = 0$ the crossings must stay depends on the navigator and B0 analysis. That choice belongs to whoever designs those analyses.

## 3. Step 2: repulsion within each shape

### 3.1 Energy

SPARKLING (Lazarus et al. 2019) chooses sample positions $x_1, \dots, x_M$ to minimize the energy distance between their empirical distribution and a target density $\rho$ (Boyer et al. 2016):

$$
F(x) \;=\; \underbrace{\frac{1}{M} \sum_{m} \int \lVert x_m - y \rVert\, \rho(y)\, dy}_{\text{attraction}}
\;-\; \underbrace{\frac{1}{2M^2} \sum_{m, n} \lVert x_m - x_n \rVert}_{\text{repulsion}} .
\qquad (1)
$$

At a minimum, the samples follow $\rho$ with blue-noise spacing.

- **The samples** are the ADC samples of one shape, $M \approx 14\,000$. $x = B g$ for a fixed sparse matrix $B$: linear interpolation of the cumulative sum at the sample times.
- **The target** $\rho$ comes from `density.py`: the sample profile $n(\rho)$ of the `exponent` or `filter` model, turned into a volumetric density. It is isotropic, so the attraction depends only on $\lVert x_m \rVert$. Precompute $\varphi(r) = \int \lVert x - y \rVert \rho(y)\, dy$ for $\lVert x \rVert = r$ on a radial grid, once per config. The attraction gradient is then $\varphi'(r)\, x/r$, which costs almost nothing.
- **The repulsion gradient** is $\sum_n (x_m - x_n)/\lVert x_m - x_n \rVert$. For one shape this is about $2 \times 10^8$ pair terms, about 1–2 s in chunked NumPy. Fast summation is not needed until step 3.
- **Optional τ weighting.** Multiply each pair term by $w(|\tau_m - \tau_n|)$, so that samples close in time since excitation repel more strongly. This spreads the $(|k|, \tau)$ sampling for the relaxation dimension. Start without it ($w \equiv 1$) and add it as a variant.

### 3.2 Sequential convex programming

From the current waveform $g^j$, with samples $x^j = B g^j$, linearize $F$ and solve:

$$
\begin{aligned}
\min_g \quad & \lambda\, \nabla F(x^j)^\top B\, g \;+\; \sum_i \lVert g_{i+1} - g_i \rVert^2 \\
\text{s.t.} \quad & \text{gradient, slew, RF window, spoil, and the step 1 targets}, \\
& \lVert k(t) - k^j(t) \rVert \le \delta_j \quad \text{at every raster point (trust region)}.
\end{aligned}
\qquad (2)
$$

- **Accept or reject.** Accept the step if the true $F$ decreases by at least a fixed fraction of the predicted decrease. Then double $\delta_j$ (up to a cap). Otherwise reject the step and halve $\delta_j$.
- **Stop** when the relative decrease falls below a tolerance, or after `repulsion_iterations` steps.
- **Every iterate is a feasible waveform.** In SPARKLING the hardware constraints are met only through a projection, whereas here they hold exactly at every step.
- **The trust region** has the same form as the tube of section 2.4, centered on the current iterate instead of on the hairpin.
- **The slew-energy term** keeps the waveforms smooth. $\lambda$ sets the balance between spreading and smoothness.
- **Timing and tip count** are those of step 1 and are not changed in step 2.

The step 1 solution is the starting point (the "core shape"). Because $F$ is not convex, the starting point decides which local minimum is reached.

### 3.3 What stays a constraint

- **Crossings stay** as the ball $\lVert k(t_c) \rVert \le \varepsilon_c$. Repulsion pushes samples out of the crowded center, so without this constraint the navigator echoes would drift away.
- **Tips are a choice.** One variant keeps the tip anchors, so the blue-noise tip design still sets the coarse structure. The other drops them and lets the attraction to $\rho$ set the radial density. If the second works, it replaces the tip-radius machinery for this purpose.

### 3.4 Config

| field | default | meaning |
|---|---|---|
| `repulsion_iterations` | 0 (off) | maximum SCP steps |
| `repulsion_weight` | to tune | $\lambda$ in (2) |
| `repulsion_trust_cells` | to tune | initial $\delta$, in cells |
| `repulsion_tau_width_s` | 0 (off) | width of the τ weighting |

### 3.5 Cost and gate

- **Cost:** each step takes about 1–2 s for the gradient and about 10–20 s for the SOCP. At an estimated 20 steps, that is about 5 min per shape, or about 65 min per design for 128 shapes on 10 cores. Tuning $\lambda$ and $\delta$ will take several designs.
- **Gate:** go on to step 3 only if step 2 lowers the gaps clearly beyond the path-length law of section 1, at short windows as well as long ones. For example, at least 10% better than step 1 at the same path length.

## 4. Step 3: repulsion between shapes, weighted by scan time

### 4.1 What it would add

Most coverage in a window of scan time comes from different TRs: 20 TRs per second at TR = 50 ms. Step 2 spreads samples within a TR only. Full spreading would use a kernel over both k-space and scan time $T$,

$$
E \;=\; \sum_{p, q} \sum_{m, n} w(|T_p - T_q|)\, \big\lVert R_p\, x^{(s_p)}_m - R_q\, x^{(s_q)}_n \big\rVert ,
\qquad (3)
$$

where TR $p$ plays shape $s_p$ rotated by $R_p$. The weight $w$ could be a sum over the window lengths of interest, to match the continuous resolution–time goal: samples close in scan time repel strongly, samples far apart weakly. The result would be blue-noise spacing at every window length.

### 4.2 Why the gain may be small with the current schedule

The TR order is shuffled, and the rotations of each shape are a low-discrepancy subset of SO(3) (super-Fibonacci). Averaged over rotations, the repulsion between two shapes depends only on their radial sample distributions, and the density model already matches those. Only the specific rotations of TRs that are close in time remain. An earlier measurement (2026-10-14) found that TR order does not change the coverage gaps: random and deliberately bad orders gave the same gaps, because each TR's petals already span the sphere.

### 4.3 Cheaper alternative first: choose the rotations

Instead of changing the shapes, choose each TR's rotation to repel the samples of the previous $K$ TRs, weighted by $w$. This is a blue-noise sequence in time, chosen greedily from candidate rotations. It needs no SOCP: each candidate costs one energy evaluation against a window of samples. This tests directly whether time-local spreading between TRs improves the short-window gaps. Do this before the full version.

### 4.4 Full version

Optimize one shape at a time, keeping the others fixed. For shape $s$, the energy includes its samples in all its TRs, against the rotated samples of every TR within the reach of $w$. Take one SCP step as in (2), then move on to the next shape, and repeat for several rounds.

- **Size:** a 1 s window holds about $2.8 \times 10^5$ samples, so pairwise sums need fast summation of the $\lVert x \rVert$ kernel. Options are NFFT-based fast summation, as in SPARKLING implementations, or Barnes–Hut.
- **Cost:** hours per design.
- **Rotations:** the rotations $R_p$ are fixed from the seed during the shape updates. Combining this with 4.3 (alternating rotation choice and shape updates) is possible but harder to analyze.

## 5. Order of work and decisions

1. **Step 1:** one branch, with the defaults unchanged (all zero). Run the sweep of 2.6, then choose new defaults in a follow-up PR.
2. **Within-TR redundancy metric:** together with step 1, so that the sweep can use it.
3. **Step 2:** its own branch, with `repulsion_iterations` = 0 by default. Tune on 8–16 shapes before running full designs. Decide by the gate of 3.5.
4. **Step 3:** 4.3 first, then 4.4 only if 4.3 helps.

Open decisions:

- **Navigator tolerance** (`crossing_tolerance_cells`): how close to $k = 0$ must the crossings stay for the navigator and B0 analysis?
- **Tip anchors in step 2:** keep the blue-noise tips, or let the density alone set the radii?
- **Acoustics and PNS:** every step makes the gradients busier. Each chosen design needs `check --band` and `check --asc` (Cima.X) before scanning.
- **Measured trajectories:** with eddy currents, the actual trajectory (from a gradient impulse response) is what matters. This plan designs the nominal trajectory. The recon uses the measured one.

## 6. Measured results

Fill this in as the work goes. Add one row for each design that is run, with the machine and the PR it belongs to, so that later estimates use measured numbers. The first row is the baseline (10-core Mac, 64 GB).

| step / design | PR | per-shape solve | calibration | full design (128 shapes) | `coverage` | tips/TR | path/TR | gap 4 mm / 1 s | gap 2 mm / 5 s | gap 1 mm / 30 s |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline: fill per shape, margin 1.1 | #16 | ~13 s | ~2 min | ~6 min | ~2.7 min | 40–43 | 57 $k_{max}$ | 2.1 | 2.1 | 4.2 |
| step 1: crossings 1 cell | this PR | 13.5 s | ~3.3 min | 6.2 min | 2.9 min | 42–45 | 58 $k_{max}$ | 2.1 | 2.0 | 4.2 |
| step 1: crossings 1 cell, `tip_width` 0.1 | this PR | 15.9 s | ~4.1 min | 7.5 min | 2.9 min | 42–45 | 58 $k_{max}$ | 2.1 | 2.1 | 4.1 |
| step 1: crossings 1 cell, `tip_width` 0.25 | this PR | 15.8 s | ~4.0 min | 7.4 min | 2.9 min | 42–45 | 59 $k_{max}$ | 2.1 | 2.0 | 3.8 |
| step 1: crossings 1 cell, `tip_width` 0.5 | this PR | 15.6 s | ~3.9 min | 7.3 min | 2.5 min | 42–46 | 59 $k_{max}$ | 2.1 | 2.0 | 3.7 |


Step 1 timing: the per-shape solve is the mean over 128 shapes from the `shapes` log, measured while the previous design's `coverage` ran on one core. The calibration is the full design time less the solves (their total over 10 workers). Other step 1 measurements, same designs in the same order: mean $|g|$ 15.1, 15.1, 15.4, 15.6 mT/m (baseline 14.8); peak slew 100 T/m/s in all; gradient spectrum peak 460–480 Hz (baseline 460 Hz); gap at 8 mm / 0.1 s 2.9, 2.9, 3.1, 3.1 cells (baseline 3.1); $(|k|, \tau)$ empty cells 1.4% in all; within-TR redundancy 4.9%, 5.1%, 5.3%, 5.3% (baseline 5.5%). A relaxed solve of one full-size shape at fixed timing takes about 9 s against 4 s for the hairpin, and lowers slew energy by 9% (13% after two re-linearizations).

| step / design | PR | per-shape solve | calibration | full design (128 shapes) | `coverage` | tips/TR | path/TR | gap 4 mm / 1 s | gap 2 mm / 5 s | gap 1 mm / 30 s |
|---|---|---|---|---|---|---|---|---|---|---|
| step 2: … | | | | | | | | | | |
| step 3: … | | | | | | | | | | |

## References

- C. Lazarus, P. Weiss, N. Chauffert, F. Mauconduit, L. El Gueddari, C. Destrieux, I. Zemmoura, A. Vignaud, P. Ciuciu. SPARKLING: variable-density k-space filling curves for accelerated T2*-weighted MRI. *Magn Reson Med* 81:3643–3661, 2019.
- G. R. Chaithya, P. Weiss, G. Daval-Frérot, A. Massire, A. Vignaud, P. Ciuciu. Optimizing full 3D SPARKLING trajectories for high-resolution magnetic resonance imaging. *IEEE Trans Med Imaging* 41:2105–2117, 2022.
- C. Boyer, N. Chauffert, P. Ciuciu, J. Kahn, P. Weiss. On the generation of sampling schemes for magnetic resonance imaging. *SIAM J Imaging Sci* 9:2039–2072, 2016.
- T. Lipp, S. Boyd. Variations and extension of the convex–concave procedure. *Optim Eng* 17:263–287, 2016.
- M. Alexa. Super-Fibonacci spirals: fast, low-discrepancy sampling of SO(3). *CVPR* 2022.
