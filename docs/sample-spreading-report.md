# Sample spreading: experiment report

Status: complete for the experiments of 2026-10-15. The overall challenge, as this work has come to frame it, is to find the efficient front of the trade-off between a gradient cost (a function of the slew waveform, for example filtered to specific frequencies) and coverage as each application defines it (section 5). This report records every experiment run for step 2 of [sample-spreading-plan.md](sample-spreading-plan.md) (repulsion within each shape) and the follow-up questions it raised. For each experiment it gives the question, why it was asked, the setup, the measured results, and the conclusion. The plan's section 6 keeps a shorter summary of the step 2 sweep; this report is the full record.

## 1. Goals and how they are measured

**The goal.** One acquisition should support a 4D reconstruction (3D space × time since excitation τ) time-averaged at 1 mm, and a 5D reconstruction (adding scan time) at lower resolution, where the resolution and the temporal scale are chosen retrospectively and trade off continuously. So the design is judged by how quickly a window of scan time covers the k-space ball of each resolution, for every window length, not by any fixed frame.

**Metrics** (from `relaxed-rosettes coverage`, default windows and resolutions):

- **Gap**: the 95th-percentile distance from a point of the k-space ball of a resolution to the nearest sample in a window of scan time, in Nyquist cells (1/FOV), the median over window starts. Smaller is better. Windows run from 0.1 s (2 TRs) to 30 s; resolutions from 8 mm to 1 mm.
- **(|k|, τ) empty cells**: the fraction of cells of 20 |k| shells × 90 τ bins of 0.5 ms that no sample of the scan visits, overall, before τ = 5 ms, and in the outer half of k-space. This is the sampling of the relaxation dimension.
- **Within-TR redundancy**: the fraction of a shape's samples within one cell of an earlier pass of the same shape.
- **Cost**: mean slew energy per shape (the optimizer's objective, $\sum_i \lVert g_{i+1} - g_i \rVert^2$, in (mT/m)²), mean and peak $|g|$, and peak slew. Acoustics and PNS follow the slew energy and the gradient spectrum.
- **Path per TR**, in units of $k_{max}$. Gaps scale as (path)$^{-1/3}$ for any 3D sampling (plan, section 1), so a change in path explains a change in gap of a third of its size. A change that spreads the samples should do better than that.

**The gate** for step 2 (plan, section 3.5): at least 10% smaller gaps than step 1 at the same path length, at short windows as well as long ones.

## 2. Common setup

- **Designs** have 16 shapes (not the default 128), seed 0, and the default config except where stated. Gaps of 16-shape designs are larger than those of 128-shape designs, so compare only within this report.
- **Repulsion**: `relaxed_rosettes.repulsion` (PR #23): energy distance to a target radial density, minimized by trust-region SCP that keeps every constraint of the solved shape. Unless stated, 20 steps, `repulsion_target = initial` (each shape's own radial histogram), initial trust region 2 cells.
- **Weight**: `repulsion_weight` sets $\lambda$ relative to each shape's starting slew energy (at 1, a full first step along $\nabla F$ is worth about that slew energy).
- **Machine**: 10-core Mac, 64 GB, at most 5 worker processes (the machine is shared). Every multi-worker run had a watchdog that kills it if the process tree exceeds 16 GB.
- **Code**: experiments 3.1–3.3 ran on the `feature/repulsion` branch before PR #23 merged; 3.4 onward on `main` at `81c687d` (PR #23). Experiments 3.4–3.7 used scratch drivers, described in each section, for variants that the package does not expose.
- **Design labels** (A, B, …) are used in the tables of section 5 and the appendix (section 7).
- **Raw outputs**: the coverage outputs, logs, drivers and transcribed console output are in [sample-spreading-report/](sample-spreading-report/README.md); the waveforms (shapes files) are kept outside the repository, as that README describes.

## 3. Experiments

### 3.0 Engineering measurements that set the cost

These are not coverage experiments, but they decided what could be run.

**Question.** A first 16-shape run of the repulsion (5 workers) used about 350 GB and crashed the machine. Why, and what does a step really cost?

**Findings** (one full-size shape, one process, under a memory limit):

| measurement | result |
|---|---|
| cvxpy problem compiled once with `Parameter`s, k as `cumsum(g)` (16 shapes, 5 workers) | used all memory (about 350 GB) and crashed the machine; not measured alone |
| same with k as a sparse variable (PR #21), still with `Parameter`s (one shape) | killed at 12.7 GB in the first SCP step: the parameters themselves were the problem (their parameter-to-data maps grow as $n^2$, $n$ = 5000 raster points) |
| problem built from constants at every step, sparse k | 0.38 GB peak; 2.2 s per step |
| step 1 solves before → after sparse k (PR #21) | hairpin 3.7 → 0.7 s, relaxed 8.9 → 1.7 s, relaxed with tube 15.7 → 2.3 s; peak 0.8 → 0.3 GB; same waveforms within 2e-4 mT/m |
| energy and gradient, $M$ = 14 000 samples, one thread | 0.64 s (row blocks) → 0.37 s (symmetric blocks, each pair once); identical to 2e-12 |
| BLAS threads (Apple Accelerate) | CPU time = wall time with or without `VECLIB_MAXIMUM_THREADS=1`: no oversubscription |

**Conclusion.** Each SCP step builds its problem from constants (PR #23). Measure one full-size shape under a memory limit before any parallel run. A 20-step repulsion takes about 45–66 s per shape, about 3–4 min for 16 shapes on 5 workers (the prototype took 4–7 min per shape).

### 3.1 Weight sweep from the relaxed design (A–D)

**High-level question.** Does spreading each shape's samples improve coverage beyond what its extra path length explains, at short and long windows, at an acceptable gradient cost?

**Immediate question.** Which repulsion weight, and how much slew energy does it cost? The prototype (weight 1, 10 steps) gained 9–14% at 0.1 s and 13% at 1 mm / 30 s, for 10–15% more slew energy; its energy was still falling.

**Setup.** Base A: step 1 with crossings 1 cell and `tip_width` 0.5 (the step 1 settings of the prototype). B, C, D: A plus repulsion at weight 0.5, 1 and 3, with the trust region doubled on every accepted step (the first rule).

**Results** (1 mm, gap in cells):

| window | A | B (w 0.5) | C (w 1) | D (w 3) |
|---|---|---|---|---|
| 0.1 s | 43.0 | 39.2 | 38.5 | 37.7 |
| 1 s | 18.4 | 18.2 | 17.7 | 17.2 |
| 5 s | 9.2 | 9.2 | 8.9 | 8.5 |
| 30 s | 3.8 | 3.9 | 3.7 | 3.5 |
| slew energy | 1415 | +0.9% | +6.0% | +18.5% |
| energy distance to target | — | −78% | −80% | −83% |

Peak slew 100 T/m/s and target violations ≤ 3e-9 in all. Almost every shape accepted exactly 10 of its 20 steps.

**Conclusion.** Short windows gain 9–15%, about as in the prototype, for any weight. Long windows gain only with weight (8% at weight 3). The weight 3 run used 18.5% of a slew-energy budget of about 30%. The constant 10/20 acceptance suggested a problem with the trust-region rule (3.2).

### 3.2 Trust-region diagnosis (one shape)

**Question.** Why does every shape accept about half its steps? If the trust region alternates between an accepted size and a rejected one, half the compute is wasted and fewer real steps are taken.

**Setup.** Shape 0 of A, weight 3, 20 steps; the trust region and the ratio of actual to predicted merit decrease at each step. Then the same with the ratio rule: halve below a ratio of 0.25, keep between 0.25 and 0.75, double above 0.75 (accept at ≥ 0.1, as before).

**Results.**

| rule | accepted | pattern | final F | final slew energy |
|---|---|---|---|---|
| double on accept | 10/20 | from step 12: 1 cell accepted at ratio 0.12–0.18, then 2 cells rejected at ratio −0.56 to −0.69, repeated | 0.3383148 | 1746 |
| ratio rule | 15/20 | settles at 0.5–1 cell; two rejections at 4 cells were `optimal_inaccurate` solves | 0.3382969 | 1733 |

**Conclusion.** The doubling rule wasted about half the steps. The linear model of $F$ is good only up to about 1–2 cells, the spacing of consecutive samples at full speed. The ratio rule reaches a lower $F$ at lower slew energy; PR #23 uses it. Solves that end `optimal_inaccurate` (at large trust regions) are treated as failed, since their waveform may miss a constraint.

### 3.3 Weight 3 with the ratio rule (E)

**Question.** Does the ratio rule turn the extra accepted steps into better coverage?

**Setup.** A plus repulsion at weight 3 with the ratio rule.

**Results** (1 mm): 0.1 s 37.1, 0.2 s 30.5, 1 s 17.0, 5 s 8.4, 30 s 3.5 (D: 37.7, 30.8, 17.2, 8.5, 3.5). 14–18 of 20 steps accepted. Slew energy +18.4%, path per TR 59.3 → 61.8 $k_{max}$ (+4%, which explains about 1.4% of a gap change). 0.1 s gaps: 8 mm 3.3 → 2.8, 2 mm 15.3 → 12.2.

**Conclusion.** A small improvement over D at the same cost. Against the gate: 0.1 s windows pass (14–20%); 0.2 s and longer reach 8–9% at 1 mm, just short. The prototype's 3.3 at 1 mm / 30 s is not reproduced. This is the result recorded in PR #23.

### 3.4 Repulsion from the hairpin design (F, G)

**High-level question.** Does step 1 (relaxed targets) earn its place? On its own it added little: in the 128-shape designs it improved only the 1 mm / 30 s gap (4.2 → 3.7) and lowered slew energy at fixed timing by 9%.

**Immediate question.** Every repulsion run so far started from the relaxed design. Is relaxation what gives repulsion its freedom, or does repulsion work as well from the plain hairpin design (point targets)?

**Setup.** F: the default config (no relaxation), 16 shapes. G: F plus repulsion at weight 3, ratio rule, with the point targets kept as constraints. Compared with A and E.

**Results** (gap in cells):

| window | F hairpin | G hairpin + rep | A relaxed | E relaxed + rep |
|---|---|---|---|---|
| 8 mm / 0.1 s | 3.1 | 2.9 | 3.3 | 2.8 |
| 2 mm / 0.1 s | 13.6 | 12.3 | 15.3 | 12.2 |
| 1 mm / 0.1 s | 42.5 | 38.4 | 43.0 | 37.1 |
| 1 mm / 1 s | 19.3 | 16.7 | 18.4 | 17.0 |
| 1 mm / 5 s | 9.9 | 8.0 | 9.2 | 8.4 |
| 1 mm / 30 s | 4.3 | 3.2 | 3.8 | 3.5 |
| tips/TR | 40–42 | 40–42 | 42–45 | 42–45 |
| path/TR ($k_{max}$) | 55.9 | 59.3 | 59.3 | 61.8 |
| slew energy | 1252 | 1815 (+45%) | 1415 | 1674 (+18%) |

**Conclusion.**

- Repulsion does not need relaxation. From the hairpin it gains more, and more at longer windows: at 1 mm −10% at 0.1 s, −13% at 1 s, −19% at 5 s, −26% at 30 s, with path +6% (about 2% of gap). This passes the gate at every window, and 3.2 at 1 mm / 30 s matches the prototype.
- It costs more slew energy: +45% over its own start, +28% over the relaxed start, and 8% more in absolute terms than E.
- Without repulsion, relaxation trades short windows for long ones (2 mm / 0.1 s 13.6 → 15.3, 1 mm / 30 s 4.3 → 3.8), a side effect not measured before.
- The two starts differ in more than the waveform (constraints, tip count, timing, $\lambda$ scale), so this run alone cannot say why. Repulsion moved samples a median of 5 cells (hairpin) and 10 cells (relaxed), 95th percentile 18–21 cells, out of 128: with 20 steps it is a local refinement, so the outcome depends on the start by construction.

### 3.5 Test 1: same problem, different start (H)

**High-level question.** Is the repulsion sensitive to its initial conditions, or do the constraints decide the outcome?

**Immediate question.** In 3.4, the hairpin start ended better at long windows. Is that because of the starting waveform, or because of the problem (point targets, fewer tips, hairpin timing)?

**Setup.** The relaxed config's targets (42–45 tips, relaxed timing) and constraints, as in E. The starting waveform is the hairpin solved on those same targets (feasible for all 16 shapes; 8–13% more slew energy than the relaxed solutions). Per shape, $\lambda$ and the target histogram were fixed to E's values (a scratch driver), so that only the start differs.

**Results** (1 mm, gap in cells):

| window | E relaxed start + rep | H hairpin start, same problem + rep |
|---|---|---|
| 0.1 s | 37.1 | 37.0 |
| 1 s | 17.0 | 16.9 |
| 5 s | 8.4 | 8.4 |
| 30 s | 3.5 | 3.4 |
| slew energy | 1674 | 1739 (+4%) |
| path/TR ($k_{max}$) | 61.8 | 61.7 |

The two starts are a median 7.1 cells apart (95th percentile 24.2); the two results 6.1 cells (22.2). Shape 8 did not move (0/20 steps accepted): its hairpin reaches $|k|$ = 1.0057, outside the relaxed problem's $|k| \le 1$ ball, so no step was feasible. The other 15 shapes accepted 16–18 steps.

**Conclusion.** Not sensitivity to the start. With the same problem, different starting waveforms reach the same coverage, though not the same waveform. The advantage of G in 3.4 comes from its problem: exact point targets, fewer tips, and the hairpin timing.

### 3.6 Free tips from the hairpin (I, J)

**High-level question.** Are fixed blue-noise targets the right way to pick the points, or should the repulsion find them? The targets do two jobs: sequence requirements (k = 0 crossings for the navigator, B0 and the τ basis; the spoil moment) and a coverage heuristic (tip directions and radii).

**Immediate question.** If the tips are no longer constraints during repulsion (crossings, spoil and hardware limits kept), does the energy place the petals better than the blue-noise tips do?

**Setup.** Start: F (the hairpin design, through its blue-noise tips). Repulsion at weight 3 with the tip targets removed from the constraints (a scratch driver that replaces `waveform_constraints` in the SCP step). I: 20 steps. J: 60 steps.

**Results** (gap in cells):

| window | F hairpin | G rep, tips fixed | I rep, tips free, 20 steps | J rep, tips free, 60 steps |
|---|---|---|---|---|
| 8 mm / 0.1 s | 3.1 | 2.9 | 2.9 | 2.9 |
| 2 mm / 0.1 s | 13.6 | 12.3 | 12.3 | 12.2 |
| 1 mm / 0.1 s | 42.5 | 38.4 | 37.9 | 36.6 |
| 1 mm / 1 s | 19.3 | 16.7 | 19.4 | 19.7 |
| 1 mm / 5 s | 9.9 | 8.0 | 12.1 | 13.7 |
| 1 mm / 30 s | 4.3 | 3.2 | 7.4 | 10.1 |
| (\|k\|, τ) empty cells: all / outer half | 4.1% / 6.8% | 3.9% / 6.6% | 8.5% / 15.7% | 10.7% / 19.8% |
| worst τ gap in a shell (outermost shell) | 2.5 ms | 2.5 ms | 23.0 ms | 45.0 ms |
| energy distance to the target | — | −75% | −85% | −87% |
| slew energy | 1252 | 1815 (+45%) | 1362 (+9%) | 1338 (+7%) |
| path/TR ($k_{max}$) | 55.9 | 59.3 | 57.4 | 57.3 |
| tip displacement: median / max per shape (cells) | — | 0 | 14.6 / 22.1 | 17.4 / 30.5 |
| mean tip radius change | — | 0 | −0.9 cells | −2.2 cells |

**Why the long windows got worse: the edge empties.** Samples near $k_{max}$ (16 shapes pooled):

| design | samples within 2 cells of $k_{max}$ | within 5 cells | largest \|k\| (median over shapes) |
|---|---|---|---|
| F hairpin | 0.44% | 1.27% | 1.0003 |
| G tips fixed | 0.47% | 1.31% | 1.0001 |
| I tips free, 20 steps | 0.01% | 0.05% | 0.9485 (6.6 cells inside) |
| J tips free, 60 steps | 0.00% | 0.00% | 0.9094 (11.6 cells inside) |

At the start (F), the energy's own descent direction points inward for only 53% of the samples within 2 cells of $k_{max}$: the energy distance neither pulls the edge in nor holds it. The pull comes from the slew-energy term of the merit: for the same petal duration, a shorter excursion costs less slew energy. With the tips fixed, the spreading cost 45% more slew energy; with them free, the energy distance fell further for only 7–9%, largely by not going out as far.

**Conclusion.**

- Freeing the tips makes the 1 mm long windows much worse (30 s: 7.4 at 20 steps, 10.1 at 60, against 3.2 with the tips fixed), and more steps make it worse. Short windows improve slightly (1 mm / 0.1 s 36.6 at 60 steps, the best of all designs).
- The energy distance and the 1 mm coverage move in opposite directions here: more steps lower the distance and raise the long-window gaps. The distance does not represent what the 1 mm reconstruction needs. With the `initial` target in 40 shells of about 3.2 cells, and the full-range $|x - y|$ kernel, it is nearly blind to the outermost few cells, where 1 mm coverage is decided (the last 1–2% of the samples).
- The fixed blue-noise tips do real work that the energy does not replace: they make the path reach $k_{max}$. Fixes, from cheapest: keep the reach constraint $u \cdot k \ge r$ (directions free, radius required); a finer radial target near the edge; a distance that sees the 1 mm scale (section 4).

### 3.7 Recon-filter density as the target (K, L, M)

**High-level question.** Should the densities of the reconstruction kernels drive the repulsion? The `filter` tip density is the SNR-optimal density for a set of Hann-filtered reconstructions from 1 to 8 mm, $D \propto \sqrt{\sum_r \pi(r) w_r^2}$.

**Immediate question.** All repulsion so far kept each shape's own radial histogram (`initial`), so the samples spread only at the radius they already have. With the filter density as the target (`model`), does repulsion move samples in radius in a way that helps the reconstructions, and how does that trade against the coverage gaps, which reward uniform filling?

**Setup.** K: `tip_density = filter`, no relaxation (hairpin), 16 shapes. L: K plus repulsion with the `initial` target. M: K plus repulsion with the `model` target. Weight 3, 20 steps. Scored on the gaps and on each resolution's predicted noise $\int w_r^2 / D$, estimated from the radial sample density (this ignores angular non-uniformity).

**Results** (gap in cells; noise relative to K):

| | F hairpin (`power`) | K `filter` base | L K + rep, `initial` | M K + rep, `model` |
|---|---|---|---|---|
| tips/TR | 40–42 | 45–48 | 45–48 | 45–48 |
| 8 mm / 0.1 s | 3.1 | 2.9 | 2.7 | 2.8 |
| 2 mm / 0.1 s | 13.6 | 12.2 | 11.2 | 11.3 |
| 1 mm / 0.1 s | 42.5 | 47.0 | 45.0 | 43.0 |
| 4 mm / 1 s | 2.1 | 1.9 | 1.9 | 1.9 |
| 1 mm / 1 s | 19.3 | 26.7 | 25.1 | 23.2 |
| 1 mm / 5 s | 9.9 | 16.7 | 15.2 | 13.1 |
| 1 mm / 30 s | 4.3 | 8.2 | 6.8 | 5.5 |
| (\|k\|, τ) empty cells: all / outer half | 4.1% / 6.8% | 12.0% / 22.7% | 12.1% / 22.8% | 11.7% / 21.7% |
| weighted predicted noise (the `filter` prior over 64 resolutions, 1–8 mm) | 1.12 | 1.00 | 1.00 | 0.96 |
| predicted noise at 8 / 4 / 2 / 1.5 / 1 mm | 1.00 / 1.01 / 1.15 / 1.25 / 0.97 | 1 | 1.01 / 1.01 / 1.01 / 1.01 / 1.00 | 1.30 / 1.28 / 1.20 / 1.09 / 0.87 |
| slew energy | 1252 | 1829 | 2325 (+27%) | 3003 (+64%) |
| path/TR ($k_{max}$) | 55.9 | 58.9 | 61.2 | 64.2 |

Repulsion: energy distance −67% (L) and −68% (M); 12–19 of 20 steps accepted; peak slew 100 T/m/s. The predicted noise is $\sum_{\text{shells}} w_r^2 V^2 / n$ from the radial histogram of the 16 shapes' samples (128 shells; `scripts/predicted_noise.py`), so it ignores angular holes.

**Conclusion.**

- The `filter` density does what it is designed for: 11% lower weighted noise than the `power` hairpin (F), and better coarse and short-window gaps (2 mm / 0.1 s 12.2 against 13.6). It pays at 1 mm: the 30 s gap doubles (4.3 → 8.2) and the outer half of the (|k|, τ) plane is 23% empty, because it puts fewer samples at the edge by design. It also has 46% more slew energy (more tips per TR).
- The `model` target moves the samples toward the filter density, which the base cannot reach (every petal passes the center, so the tips can only give a majorant of the target, oversampling the center; `density.py`). Samples move outward: 1 mm gaps improve (30 s 8.2 → 5.5), noise rises 20–30% at coarse resolutions and falls 13% at 1 mm, and the weighted noise falls 4%. The cost is large: +64% slew energy, 2.4 times the hairpin's.
- The `initial` target changes the noise by less than 1%, as expected (it keeps the radial profile), and gives modest gap gains for +27% slew energy.
- **The two scores disagree about which design is best.** By the 1 mm gaps, M is worse than the plain hairpin; by the weighted noise, it is the best design measured. A recon-density target moves the design toward its own optimum, but whether that is better depends entirely on the coverage representation, which must therefore be set by the application (sections 5–6).

## 4. Summary of conclusions



1. **Within-shape repulsion works at short windows** (0.1 s gaps 10–20% smaller in every variant), and at long windows only when the problem keeps the path anchored (3.4).
2. **The starting waveform matters little; the constraints decide** (3.5). With 20 steps the repulsion is a local refinement (median 5–10 cells).
3. **Relaxed targets do not help repulsion.** Repulsion from the plain hairpin design passes the gate at every window (3.4), but at more slew energy.
4. **The fixed blue-noise tips are needed** (3.6): without them the slew-energy term pulls the petals in, the samples leave the edge of k-space, and the 1 mm long-window gaps get 2–3 times worse.
5. **The energy distance is a weak and sometimes wrong-signed stand-in for coverage.** It fell 75–87% in every run while the gaps changed by −26% to +135%, and in 3.6 lowering it made the 1 mm coverage worse.
6. **Long windows are built across TRs.** Within-shape spreading gives at most about 25% at 1 mm / 30 s; the larger lever is likely between TRs (plan, step 3).
7. **The scores can disagree** (3.7): the `filter` density with a `model` repulsion target has the lowest weighted predicted noise of all designs (−14% against the `power` hairpin), but a 28% larger 1 mm / 30 s gap (5.5 against 4.3) and 2.4 times its slew energy. Which design is better depends on the coverage representation.
8. **The real goal is an efficient front, not one design** (section 5): the trade-off between a gradient cost (a function of slew, for example filtered to specific frequencies) and coverage as each application defines it. The experiments so far are points; the next work should define both axes and trace the front.

**Direction: a multi-scale, recon-kernel distance, weighted by scan time.** Results 2, 4 and 5 all point at the objective rather than the optimizer. A distance built from the reconstruction kernels, with the kernel scale tied to the window length (coarse kernels for short windows, fine ones for long windows), would match the continuous resolution–time goal. Its natural home is the cross-TR problem: one shape has about 14 000 samples, while the 1 mm ball has about 9 million Nyquist cells and needs about 600 TRs (30 s) to fill, so a 1 mm-scale kernel within one shape sees almost no close pairs. Within a shape, only the coarse scales (about 4–8 mm) are reachable, and the edge problem has cheaper fixes (a reach constraint, a finer radial target). None of the results tests this distance yet; they show that the current one is misaligned. Section 6.1 shows that, with its time windows set by a target resolution–time curve, this distance is the smooth counterpart of the time-to-coverage measure, and section 6.2 how the same curve fixes the scale pairing of the reconstruction kernels.

## 5. Cost against coverage: the efficient front

**The design problem, restated.** The challenge is to find the efficient (Pareto) front of the trade-off between two quantities, and then to choose a point on it for each application:

- **A gradient cost**: some function of the slew waveform. Plain slew energy is the crudest choice. Slew energy filtered to specific frequencies (for example a band around the coil's mechanical resonances, for acoustics) or a PNS model output would represent what the cost is really about.
- **Coverage, as the application defines it**: which resolutions and which windows of scan time matter, and how much each one matters. A time-averaged 1 mm 4D reconstruction, a 5D reconstruction at a few millimetres, and a navigator-driven correction each weight the coverage curve differently.

**Decision (2026-10-15): the cost axis is slew energy for now**, $\sum_i \lVert g_{i+1} - g_i \rVert^2$ over the TR: the squared 2-norm of the discrete slew waveform, proportional to $\int \lVert dg/dt \rVert^2 dt$, and the optimizer's objective today. (The 2-norm and its square order designs the same way, so they give the same front.) Band-filtered slew energy and the PNS model are left for later. The hardware limits stay hard constraints: $\lVert g \rVert \le$ 28 mT/m and $\lVert \text{slew} \rVert \le$ 100 T/m/s per sample (25 T/m/s in the boundary windows).

A design is efficient if no other design has better coverage at the same cost, or lower cost at the same coverage. The work so far has produced individual points (below), not the front. Every method in this report (timing margins, relaxed targets, repulsion, choice of tips) is a way of moving along or toward this front, and should be judged by how far it moves it, not by its own objective (such as the energy distance, section 3.6).

In crude terms, the trade-off is "loud and fast against slow and quiet". Two knobs set it today: `segment_time_margin` (petal time beyond the fastest straight path: fewer petals, lower $|g|$ and slew) and `repulsion_weight` (spreading against slew energy). The hardware limits are hard constraints. The designs so far (16 shapes):

| design | mean slew energy | gap 1 mm / 30 s | gap 1 mm / 0.1 s |
|---|---|---|---|
| F hairpin | 1252 | 4.3 | 42.5 |
| J tips free, 60 steps | 1338 | 10.1 | 36.6 |
| I tips free, 20 steps | 1362 | 7.4 | 37.9 |
| A relaxed | 1415 | 3.8 | 43.0 |
| B relaxed + rep w 0.5 | 1427 | 3.9 | 39.2 |
| C relaxed + rep w 1 | 1499 | 3.7 | 38.5 |
| E relaxed + rep w 3 (ratio rule) | 1674 | 3.5 | 37.1 |
| H hairpin start, relaxed problem + rep | 1739 | 3.4 | 37.0 |
| G hairpin + rep w 3 | 1815 | 3.2 | 38.4 |
| K `filter` base | 1829 | 8.2 | 47.0 |
| L `filter` + rep `initial` | 2325 | 6.8 | 45.0 |
| M `filter` + rep `model` | 3003 | 5.5 | 43.0 |

K–M score better on predicted noise (3.7) than this table suggests: they sit on a different front, the one for the noise representation of coverage. Among the designs that keep the tips fixed, each further step down in the 1 mm / 30 s gap costs more slew energy: the expected shape of a Pareto front, though mixed with differences in the problem (constraints, tip count, timing). Three things limit this as a front today:

- **The coverage side is not one number.** It is gaps at every resolution and window, so a trade-off needs weights over them (for example the `filter` model's prior over resolutions), and the repulsion currently trades slew energy against the energy distance, not against coverage.
- **Slew energy is a crude stand-in for loudness and PNS.** Acoustics depend on the spectrum near the coil's mechanical resonances (`check --band`); PNS on the slew waveform through the SAFE model (`check --asc`). A frequency-weighted penalty would be a better cost.
- **The point on the front is an application choice** (comfort, PNS limits, scan time), so the deliverable is the front, not one design.

**What finding the front needs.**

1. **A cost function.** For now, slew energy (decision above). Later, chosen with the scanner in mind: band-filtered slew energy (`check --band`) and the PNS model (`check --asc`).
2. **An application-defined coverage measure**: a weighting of the gaps (or of a recon-kernel distance, section 4) over resolutions and window lengths, for each application of interest.
3. **One problem swept over the knobs** (margin, repulsion weight, and later the cross-TR choices), each design scored on both axes. The efficient designs form the front; the others are discarded.
4. **Optimizers that target the front directly**: an objective of the form cost + μ × coverage loss, swept over μ, with both terms the ones chosen in 1 and 2. The current merit (slew energy + λ × energy distance) has this form, but with stand-ins for both terms.

## 6. The coverage axis: candidate representations

With slew energy as the cost axis (section 5), the open question is how to represent coverage. The candidates so far, grouped by what they measure. *Optimizable* means smooth in the sample positions, so that an optimizer can use it as an objective; the others can only score designs.

| # | representation | what it measures | status | optimizable | limitations |
|---|---|---|---|---|---|
| **A. Geometric coverage of windows of scan time** | | | | | |
| 1 | gap at (resolution, window) | 95th-percentile distance to the nearest sample, in cells, on a grid of 7 resolutions × 8 windows (median or maximum over window starts) | the main yardstick of this report | no | one number only with an application weighting over the grid |
| 2 | fraction of cells covered | the same grid, the fraction of Nyquist cells with a sample | printed by `coverage` | no | more sensitive to small holes than the gap; saturates at long windows |
| 3 | time to coverage $W(\text{res})$ | the window length needed to reach a target gap at each resolution: the continuous resolution–time curve | measured once (about 31 s at 1 mm, 5 s at 2 mm, 1 s at 4 mm, default design of 2026-10-14) | no | derived from 1; reads directly as "how long until a reconstruction at this resolution" |
| **B. The relaxation dimension** | | | | | |
| 4 | $(\lvert k \rvert, \tau)$ empty cells, longest τ gap per shell | how evenly the scan samples time since excitation at each radius | printed by `coverage` | no | exposed the edge failure of 3.6 (τ gaps of 23–45 ms in the outermost shell) |
| **C. Radial density and the noise it implies** | | | | | |
| 5 | match to a target radial density | the `exponent` model (a resolution–time power law) or the `filter` model (SNR-optimal for Hann-filtered recons, $D \propto \sqrt{\sum_r \pi(r) w_r^2}$) | sets the tip radii; the target in 3.7 | yes (as a target) | radial only; says nothing about holes |
| 6 | predicted noise of filtered recons, $\int w_r^2 / D$ per resolution | the SNR side of each retrospective reconstruction | used in 3.7, with $D$ from the radial histogram | yes, with a smooth estimate of $D$ | the radial estimate ignores angular holes |
| **D. Distribution distances** | | | | | |
| 7 | energy distance to a target density ($\lvert x - y \rvert$ kernel) | how far the samples' distribution is from the target, at all scales at once | the repulsion objective (PR #23) | yes | shown to be misaligned: it fell 75–87% while the gaps barely moved, and in 3.6 lowering it made the 1 mm coverage worse |
| 8 | multi-scale recon-kernel distance, weighted by scan time | an MMD-type distance built from the reconstruction kernels, with the kernel scale tied to the window length (coarse for short windows, fine for long ones), and pairs across TRs weighted by $w(\lvert T_p - T_q \rvert)$ (plan, eq. 3) | proposed (section 4) | yes | untested; belongs in the cross-TR problem; needs fast summation at scale |
| **E. Proxies and diagnostics** | | | | | |
| 9 | path length per TR | speed: gaps scale as (path)$^{-1/3}$ | used to separate "faster" from "better spread" (the gate) | yes | not a coverage measure itself |
| 10 | within-TR redundancy | retracing within one TR | printed by `coverage` | no | a diagnostic; stayed at 3–6% in every design |
| **Not yet discussed** | | | | | |
| 11 | point-spread function | sidelobe energy of each window's PSF at each resolution | — | yes (expensive) | closer to image quality than geometric gaps; costs an NUFFT for each window and resolution |
| 12 | conditioning of the reconstruction | for example the convergence rate of CG-SENSE for a window | — | hardly | closest to the real reconstruction; very expensive, and coil-dependent |

**What any choice still needs.**

- **An application weighting.** For each application, which (resolution, window) pairs matter and how much. The `filter` model's prior $\pi(r)$ over resolutions is one half of it; a prior over window lengths would complete it.
- **A split between the objective and the score.** Representations 1–4 are good for scoring the front but cannot be optimized directly; 5–8 (and 11) can, and must be checked against 1–4. The energy distance (7) failed exactly that check (3.6).

### 6.1 Time to coverage (3) and the recon-kernel distance (8) are one object

They are two views of the resolution–time curve: 3 measures it, and 8, set up as below, optimizes it.

- **3** asks, for each resolution, how long a window must be before its samples cover that resolution's k-space ball to a target gap. The answer is the curve $W(\text{res})$.
- **8** is a sum over resolutions of a kernel distance in which the kernel scale and the time window are tied:

$$
E \;=\; \sum_{\text{res}} \alpha_{\text{res}} \sum_{p, q} w_{\text{res}}\big(\lvert T_p - T_q \rvert\big) \sum_{m, n} \kappa_{\text{res}}\big(x_m - x_n\big) \;-\; \text{(attraction to the target)} ,
$$

  where $\kappa_{\text{res}}$ is a kernel about one Nyquist cell wide at that resolution (for example the Hann-filtered PSF of that reconstruction), which penalizes clumps and holes at that scale, and $w_{\text{res}}$ is a window in scan time that says which TRs' samples must look well spread together at that resolution.

**The connection.** Choose the width of $w_{\text{res}}$ to be $W(\text{res})$. The term for each resolution then asks that every window of length $W(\text{res})$ sample that resolution's ball evenly at that resolution's scale, which is exactly the statement whose failure 3 measures. So:

- a target curve $W^*(\text{res})$ sets the time windows of 8;
- the weights $\alpha_{\text{res}}$ are the application's priority over resolutions (the $\pi(r)$ of the `filter` model);
- minimizing $E$ is a smooth stand-in for achieving $W^*(\text{res})$;
- 3 then checks whether it did.

The `exponent` density (5) is a third view of the same object: it chooses the radial density so that $W(\text{res}) \propto \text{res}^{-\alpha}$, but only on average over radius, and blind to holes. 8 adds the angular holes and the structure across TRs that 5 ignores.

**Where 3 and 8 can disagree.**

- **Worst case against average.** 3 uses a 95th percentile, which is about the worst holes; a kernel distance measures an average mismatch. Lowering an average does not guarantee the tail, so 3 stays as the check.
- **Threshold against smooth trade-off.** 3 needs a target gap to define "covered"; 8 has no threshold and rewards spreading at every scale, weighted by $\alpha$.

**Practical pairing.** Write the application's need as a target curve $W^*(\text{res})$ with weights $\alpha_{\text{res}}$, optimize 8 built from them, and score the result with 3 (and 4 for the τ dimension).

### 6.2 The target curve and the reconstruction kernels

The target curve also fixes the **scale pairing** of the reconstruction kernels, but not their shape, and the two must be defined together with an assumption about acceleration.

- **What the curve defines.** $W^*(\text{res})$ says that a window of length $T$ supports a reconstruction at $\text{res}(T) = W^{*-1}(T)$. For a continuous 5D reconstruction, that gives a one-parameter family of space–time kernels, $K_T(k, t) = w_T(t)\, h_{\text{res}(T)}(k)$: a temporal window of length $T$ with a k-space filter cut off at $1/(2\,\text{res}(T))$. Short windows go with coarse kernels and long windows with fine ones: the same pairing that ties $w_{\text{res}}$ to $\kappa_{\text{res}}$ in 8. If the acquisition is designed for $W^*$, the matching reconstruction kernels lie on that curve, and the design objective and the reconstruction use the same family.
- **What it does not define.** The kernel *shape* (Hann or another taper) is a choice of SNR against sharpness; the curve sets bandwidths, not profiles. Nor does it forbid points off the curve: a window longer than $W(\text{res})$ is oversampled (better SNR), and a shorter one is undersampled (it relies on the coils and regularization). The curve is the locus where the data are just sufficient: the efficient pairing, not the only one.
- **"Sufficient" depends on the reconstruction.** $W(\text{res})$ here counts Nyquist cells. With multi-coil data and a subspace or low-rank model (for example the τ basis), an effective acceleration $R$ lowers the density needed, so the curve the reconstruction really needs is about $W(\text{res})/R$, with $R$ itself depending on resolution and coil geometry.

**So it runs both ways.** The application chooses its reconstruction kernels (shapes, and a priority over resolutions and windows) and its acceleration model; together they set $W^*(\text{res})$ and the weights $\alpha$; those define the design objective (8) and the score (3). Conversely, a target curve says which kernels the data support.

[coverage-target.md](coverage-target.md) designs how an application states its coverage requirements, $W^*(\text{res})$ among them: which dimensions each resolves or averages, which inputs it picks, which are reconstruction settings with defaults, and how the requirements trace the front.

## 7. Appendix: full results

*Generated from the coverage outputs and shapes files (`sample-spreading-report/scripts/report_tables.py`); see section 2 for the setup.*

| design | tips/TR | path/TR (k_max) | mean \|g\| in ADC (mT/m) | peak \|g\| (mT/m) | mean slew energy | within-TR redundancy | empty (\|k\|, τ) cells: all / < 5 ms / outer half |
|---|---|---|---|---|---|---|---|
| A relaxed base | 42–45 | 59.3 | 15.5 | 27.4 | 1415 | 5.9% | 3.4% / 13.0% / 5.8% |
| B relaxed + rep w0.5 (doubling) | 42–45 | 59.1 | 15.5 | 28.0 | 1427 | 4.1% | 3.9% / 13.0% / 6.8% |
| C relaxed + rep w1 (doubling) | 42–45 | 60.0 | 15.7 | 28.0 | 1499 | 3.6% | 3.9% / 13.0% / 6.8% |
| D relaxed + rep w3 (doubling) | 42–45 | 61.6 | 16.2 | 28.0 | 1676 | 3.3% | 3.6% / 14.0% / 6.0% |
| E relaxed + rep w3 (ratio) | 42–45 | 61.8 | 16.2 | 28.0 | 1674 | 3.3% | 3.7% / 14.0% / 6.3% |
| F hairpin base | 40–42 | 55.9 | 14.7 | 26.1 | 1252 | 5.5% | 4.1% / 15.5% / 6.8% |
| G hairpin + rep w3 | 40–42 | 59.3 | 15.5 | 28.0 | 1815 | 3.9% | 3.9% / 15.0% / 6.6% |
| H hairpin on relaxed targets + rep (test 1) | 42–45 | 61.7 | 16.2 | 28.0 | 1739 | 3.5% | 3.8% / 14.0% / 6.4% |
| I hairpin + rep, free tips, 20 steps | 40–42 | 57.4 | 15.0 | 27.6 | 1362 | 3.9% | 8.5% / 17.0% / 15.7% |
| J hairpin + rep, free tips, 60 steps | 40–42 | 57.3 | 15.0 | 28.0 | 1338 | 3.9% | 10.7% / 21.5% / 19.8% |
| K filter base | 45–48 | 58.9 | 15.4 | 26.7 | 1829 | 5.6% | 12.0% / 20.5% / 22.7% |
| L filter + rep initial | 45–48 | 61.2 | 16.0 | 28.0 | 2325 | 4.1% | 12.1% / 21.5% / 22.8% |
| M filter + rep model | 45–48 | 64.2 | 16.8 | 28.0 | 3003 | 3.3% | 11.7% / 21.5% / 21.7% |

**A relaxed base** — gap (95th percentile, cells; median over window starts)

| window (s) | 8 mm | 6 mm | 4 mm | 3 mm | 2 mm | 1.5 mm | 1 mm |
|---|---|---|---|---|---|---|---|
| 0.1 | 3.3 | 4.4 | 6.8 | 9.5 | 15.3 | 22.4 | 43.0 |
| 0.2 | 2.3 | 3.0 | 4.7 | 6.4 | 10.6 | 16.3 | 33.6 |
| 0.5 | 1.5 | 1.9 | 3.0 | 4.1 | 6.8 | 10.5 | 24.3 |
| 1 | 1.1 | 1.4 | 2.1 | 2.9 | 4.6 | 7.2 | 18.4 |
| 2 | 0.8 | 1.0 | 1.5 | 2.0 | 3.3 | 5.0 | 13.8 |
| 5 | 0.7 | 0.7 | 1.0 | 1.3 | 2.0 | 3.1 | 9.2 |
| 10 | 0.7 | 0.7 | 0.8 | 1.0 | 1.4 | 2.1 | 6.6 |
| 30 | 0.7 | 0.7 | 0.7 | 0.7 | 0.9 | 1.2 | 3.8 |

**B relaxed + rep w0.5 (doubling)** — gap (95th percentile, cells; median over window starts)

| window (s) | 8 mm | 6 mm | 4 mm | 3 mm | 2 mm | 1.5 mm | 1 mm |
|---|---|---|---|---|---|---|---|
| 0.1 | 2.9 | 3.9 | 6.0 | 8.2 | 13.1 | 19.3 | 39.2 |
| 0.2 | 2.1 | 2.9 | 4.4 | 6.1 | 9.9 | 15.0 | 32.4 |
| 0.5 | 1.4 | 1.9 | 2.9 | 3.9 | 6.4 | 9.7 | 23.8 |
| 1 | 1.1 | 1.4 | 2.1 | 2.8 | 4.6 | 6.8 | 18.2 |
| 2 | 0.8 | 1.0 | 1.5 | 2.0 | 3.2 | 4.8 | 13.8 |
| 5 | 0.7 | 0.7 | 1.0 | 1.3 | 2.0 | 3.0 | 9.2 |
| 10 | 0.7 | 0.7 | 0.8 | 1.0 | 1.4 | 2.1 | 6.6 |
| 30 | 0.7 | 0.7 | 0.7 | 0.7 | 0.9 | 1.2 | 3.9 |

**C relaxed + rep w1 (doubling)** — gap (95th percentile, cells; median over window starts)

| window (s) | 8 mm | 6 mm | 4 mm | 3 mm | 2 mm | 1.5 mm | 1 mm |
|---|---|---|---|---|---|---|---|
| 0.1 | 2.8 | 3.8 | 5.9 | 8.0 | 12.7 | 18.7 | 38.5 |
| 0.2 | 2.1 | 2.8 | 4.4 | 6.0 | 9.6 | 14.6 | 31.6 |
| 0.5 | 1.4 | 1.9 | 2.8 | 3.9 | 6.3 | 9.5 | 23.3 |
| 1 | 1.0 | 1.4 | 2.1 | 2.8 | 4.5 | 6.6 | 17.7 |
| 2 | 0.8 | 1.0 | 1.5 | 2.0 | 3.2 | 4.7 | 13.4 |
| 5 | 0.7 | 0.7 | 1.0 | 1.3 | 2.0 | 3.0 | 8.9 |
| 10 | 0.7 | 0.7 | 0.8 | 1.0 | 1.4 | 2.1 | 6.4 |
| 30 | 0.7 | 0.7 | 0.7 | 0.7 | 0.9 | 1.2 | 3.7 |

**D relaxed + rep w3 (doubling)** — gap (95th percentile, cells; median over window starts)

| window (s) | 8 mm | 6 mm | 4 mm | 3 mm | 2 mm | 1.5 mm | 1 mm |
|---|---|---|---|---|---|---|---|
| 0.1 | 2.8 | 3.8 | 5.8 | 7.9 | 12.4 | 18.1 | 37.7 |
| 0.2 | 2.1 | 2.8 | 4.4 | 5.9 | 9.5 | 14.2 | 30.8 |
| 0.5 | 1.4 | 1.9 | 2.8 | 3.9 | 6.2 | 9.2 | 22.5 |
| 1 | 1.0 | 1.4 | 2.0 | 2.8 | 4.4 | 6.5 | 17.2 |
| 2 | 0.8 | 1.0 | 1.5 | 2.0 | 3.1 | 4.6 | 12.8 |
| 5 | 0.7 | 0.7 | 1.0 | 1.3 | 2.0 | 2.9 | 8.5 |
| 10 | 0.7 | 0.7 | 0.8 | 1.0 | 1.4 | 2.0 | 6.1 |
| 30 | 0.7 | 0.7 | 0.7 | 0.7 | 0.9 | 1.2 | 3.5 |

**E relaxed + rep w3 (ratio)** — gap (95th percentile, cells; median over window starts)

| window (s) | 8 mm | 6 mm | 4 mm | 3 mm | 2 mm | 1.5 mm | 1 mm |
|---|---|---|---|---|---|---|---|
| 0.1 | 2.8 | 3.7 | 5.8 | 7.8 | 12.2 | 18.0 | 37.1 |
| 0.2 | 2.1 | 2.8 | 4.3 | 5.8 | 9.3 | 13.9 | 30.5 |
| 0.5 | 1.4 | 1.9 | 2.8 | 3.9 | 6.1 | 9.0 | 22.2 |
| 1 | 1.0 | 1.4 | 2.0 | 2.8 | 4.4 | 6.5 | 17.0 |
| 2 | 0.8 | 1.0 | 1.5 | 2.0 | 3.1 | 4.6 | 12.8 |
| 5 | 0.7 | 0.7 | 1.0 | 1.3 | 2.0 | 2.9 | 8.4 |
| 10 | 0.7 | 0.7 | 0.8 | 1.0 | 1.4 | 2.0 | 6.0 |
| 30 | 0.7 | 0.7 | 0.7 | 0.7 | 0.9 | 1.2 | 3.5 |

**F hairpin base** — gap (95th percentile, cells; median over window starts)

| window (s) | 8 mm | 6 mm | 4 mm | 3 mm | 2 mm | 1.5 mm | 1 mm |
|---|---|---|---|---|---|---|---|
| 0.1 | 3.1 | 4.2 | 6.2 | 8.5 | 13.6 | 20.8 | 42.5 |
| 0.2 | 2.2 | 3.0 | 4.5 | 6.2 | 10.1 | 15.5 | 33.7 |
| 0.5 | 1.5 | 1.9 | 2.9 | 4.0 | 6.5 | 10.1 | 24.8 |
| 1 | 1.1 | 1.4 | 2.1 | 2.9 | 4.6 | 7.1 | 19.3 |
| 2 | 0.8 | 1.0 | 1.5 | 2.0 | 3.3 | 4.9 | 14.5 |
| 5 | 0.7 | 0.7 | 1.0 | 1.3 | 2.1 | 3.1 | 9.9 |
| 10 | 0.7 | 0.7 | 0.8 | 1.0 | 1.5 | 2.2 | 7.3 |
| 30 | 0.7 | 0.7 | 0.7 | 0.7 | 0.9 | 1.3 | 4.3 |

**G hairpin + rep w3** — gap (95th percentile, cells; median over window starts)

| window (s) | 8 mm | 6 mm | 4 mm | 3 mm | 2 mm | 1.5 mm | 1 mm |
|---|---|---|---|---|---|---|---|
| 0.1 | 2.9 | 3.9 | 6.0 | 7.9 | 12.3 | 17.9 | 38.4 |
| 0.2 | 2.2 | 2.9 | 4.5 | 6.1 | 9.4 | 13.6 | 30.2 |
| 0.5 | 1.5 | 1.9 | 2.9 | 3.9 | 6.2 | 9.1 | 21.9 |
| 1 | 1.1 | 1.4 | 2.1 | 2.8 | 4.5 | 6.6 | 16.7 |
| 2 | 0.8 | 1.0 | 1.5 | 2.0 | 3.2 | 4.6 | 12.3 |
| 5 | 0.7 | 0.7 | 1.0 | 1.3 | 2.0 | 2.9 | 8.0 |
| 10 | 0.7 | 0.7 | 0.8 | 1.0 | 1.4 | 2.1 | 5.7 |
| 30 | 0.7 | 0.7 | 0.7 | 0.7 | 0.9 | 1.2 | 3.2 |

**H hairpin on relaxed targets + rep (test 1)** — gap (95th percentile, cells; median over window starts)

| window (s) | 8 mm | 6 mm | 4 mm | 3 mm | 2 mm | 1.5 mm | 1 mm |
|---|---|---|---|---|---|---|---|
| 0.1 | 2.9 | 3.8 | 5.8 | 7.7 | 12.0 | 17.5 | 37.0 |
| 0.2 | 2.1 | 2.8 | 4.3 | 5.9 | 9.3 | 13.8 | 30.6 |
| 0.5 | 1.4 | 1.9 | 2.8 | 3.9 | 6.1 | 9.1 | 22.2 |
| 1 | 1.1 | 1.4 | 2.0 | 2.8 | 4.4 | 6.5 | 16.9 |
| 2 | 0.8 | 1.0 | 1.5 | 2.0 | 3.1 | 4.6 | 12.7 |
| 5 | 0.7 | 0.7 | 1.0 | 1.3 | 2.0 | 2.9 | 8.4 |
| 10 | 0.7 | 0.7 | 0.8 | 1.0 | 1.4 | 2.0 | 5.9 |
| 30 | 0.7 | 0.7 | 0.7 | 0.7 | 0.9 | 1.2 | 3.4 |

**I hairpin + rep, free tips, 20 steps** — gap (95th percentile, cells; median over window starts)

| window (s) | 8 mm | 6 mm | 4 mm | 3 mm | 2 mm | 1.5 mm | 1 mm |
|---|---|---|---|---|---|---|---|
| 0.1 | 2.9 | 3.9 | 5.9 | 7.8 | 12.3 | 17.8 | 37.9 |
| 0.2 | 2.2 | 2.9 | 4.5 | 6.0 | 9.4 | 13.6 | 31.3 |
| 0.5 | 1.5 | 1.9 | 2.9 | 4.0 | 6.2 | 9.0 | 24.0 |
| 1 | 1.1 | 1.4 | 2.1 | 2.8 | 4.5 | 6.5 | 19.4 |
| 2 | 0.8 | 1.0 | 1.5 | 2.0 | 3.2 | 4.6 | 15.6 |
| 5 | 0.7 | 0.8 | 1.0 | 1.3 | 2.0 | 2.9 | 12.1 |
| 10 | 0.7 | 0.7 | 0.8 | 1.0 | 1.5 | 2.0 | 10.2 |
| 30 | 0.7 | 0.7 | 0.7 | 0.7 | 0.9 | 1.2 | 7.4 |

**J hairpin + rep, free tips, 60 steps** — gap (95th percentile, cells; median over window starts)

| window (s) | 8 mm | 6 mm | 4 mm | 3 mm | 2 mm | 1.5 mm | 1 mm |
|---|---|---|---|---|---|---|---|
| 0.1 | 2.9 | 3.8 | 5.8 | 7.8 | 12.2 | 17.5 | 36.6 |
| 0.2 | 2.2 | 2.9 | 4.4 | 6.0 | 9.4 | 13.5 | 30.4 |
| 0.5 | 1.5 | 1.9 | 2.9 | 4.0 | 6.4 | 9.1 | 23.7 |
| 1 | 1.1 | 1.4 | 2.1 | 2.8 | 4.6 | 6.5 | 19.7 |
| 2 | 0.8 | 1.0 | 1.5 | 2.1 | 3.3 | 4.6 | 16.4 |
| 5 | 0.7 | 0.8 | 1.0 | 1.3 | 2.1 | 2.9 | 13.7 |
| 10 | 0.7 | 0.7 | 0.8 | 1.0 | 1.5 | 2.1 | 12.2 |
| 30 | 0.7 | 0.7 | 0.7 | 0.7 | 0.9 | 1.2 | 10.1 |

**K filter base** — gap (95th percentile, cells; median over window starts)

| window (s) | 8 mm | 6 mm | 4 mm | 3 mm | 2 mm | 1.5 mm | 1 mm |
|---|---|---|---|---|---|---|---|
| 0.1 | 2.9 | 3.8 | 5.6 | 7.3 | 12.2 | 20.7 | 47.0 |
| 0.2 | 2.1 | 2.8 | 4.1 | 5.5 | 9.2 | 15.7 | 40.0 |
| 0.5 | 1.4 | 1.8 | 2.7 | 3.6 | 6.0 | 10.6 | 32.4 |
| 1 | 1.0 | 1.3 | 1.9 | 2.6 | 4.3 | 7.5 | 26.7 |
| 2 | 0.8 | 1.0 | 1.4 | 1.8 | 3.0 | 5.3 | 22.0 |
| 5 | 0.7 | 0.7 | 0.9 | 1.2 | 1.9 | 3.3 | 16.7 |
| 10 | 0.7 | 0.7 | 0.7 | 0.9 | 1.4 | 2.3 | 12.8 |
| 30 | 0.7 | 0.7 | 0.7 | 0.7 | 0.8 | 1.3 | 8.2 |

**L filter + rep initial** — gap (95th percentile, cells; median over window starts)

| window (s) | 8 mm | 6 mm | 4 mm | 3 mm | 2 mm | 1.5 mm | 1 mm |
|---|---|---|---|---|---|---|---|
| 0.1 | 2.7 | 3.6 | 5.4 | 7.0 | 11.2 | 18.7 | 45.0 |
| 0.2 | 2.0 | 2.7 | 4.1 | 5.4 | 8.7 | 14.2 | 38.3 |
| 0.5 | 1.4 | 1.8 | 2.7 | 3.6 | 5.7 | 9.4 | 31.0 |
| 1 | 1.0 | 1.3 | 1.9 | 2.5 | 4.1 | 6.8 | 25.1 |
| 2 | 0.8 | 1.0 | 1.4 | 1.8 | 2.9 | 4.8 | 20.4 |
| 5 | 0.7 | 0.7 | 0.9 | 1.2 | 1.9 | 3.0 | 15.2 |
| 10 | 0.7 | 0.7 | 0.7 | 0.9 | 1.3 | 2.1 | 11.3 |
| 30 | 0.7 | 0.7 | 0.7 | 0.7 | 0.8 | 1.2 | 6.8 |

**M filter + rep model** — gap (95th percentile, cells; median over window starts)

| window (s) | 8 mm | 6 mm | 4 mm | 3 mm | 2 mm | 1.5 mm | 1 mm |
|---|---|---|---|---|---|---|---|
| 0.1 | 2.8 | 3.7 | 5.6 | 7.4 | 11.3 | 17.9 | 43.0 |
| 0.2 | 2.1 | 2.8 | 4.1 | 5.5 | 8.6 | 13.4 | 36.2 |
| 0.5 | 1.4 | 1.8 | 2.7 | 3.6 | 5.6 | 8.9 | 28.7 |
| 1 | 1.0 | 1.3 | 1.9 | 2.6 | 4.0 | 6.3 | 23.2 |
| 2 | 0.8 | 1.0 | 1.4 | 1.8 | 2.9 | 4.5 | 18.4 |
| 5 | 0.7 | 0.7 | 1.0 | 1.2 | 1.8 | 2.8 | 13.1 |
| 10 | 0.7 | 0.7 | 0.8 | 0.9 | 1.3 | 2.0 | 9.4 |
| 30 | 0.7 | 0.7 | 0.7 | 0.7 | 0.8 | 1.2 | 5.5 |
