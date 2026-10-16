# B0 over scan time

Status: plan; nothing here is implemented yet. The first steps are a bound computed from simulated data (section 4) and a measurement on the phantom session of [basis-quality-test-plan.md](basis-quality-test-plan.md) (section 2); the model is built into the reconstruction only if the measurements of section 6 call for it. Record the measured results in section 9, in the same PR as each step, and correct the text where the plan was wrong.

## Question

The reconstruction models off-resonance as a field map $f(r)$ that is fixed over the scan (`relaxed_rosettes.recon.b0`):

$$y_v(k, \tau) = \sum_r s_v(r)\, x(r)\, e^{2\pi i\,(k \cdot r + f(r)\,\tau)} .$$

The map is estimated from all TRs together, so it is the average over scan time $T$. A B0 that changes over $T$ leaves a phase error $2\pi\, \Delta f(r, T)\, \tau$ in every sample. With the default ADC window (τ up to about 45 ms), 1 Hz gives 0.28 rad at the end of the readout. In a reconstruction averaged over $T$ the error blurs and adds artefacts; in one that resolves $T$ (navigator, BOLD) it is false signal at the frequencies of the change, and a respiratory change can alias into the BOLD band like the cardiac harmonics of [example-requirements-design.md](example-requirements-design.md), section 6.1.

How large are the changes over $T$, how should they be modelled, and how can they be estimated from the data the trajectory already acquires?

## Short answer

Model the change as a few modes, $\Delta f(r, T) = \sum_k u_k(r)\, w_k(T)$, and find the modes of each subject in that subject's data. The k ≈ 0 crossings and the samples around them see the whole object about 40 times per TR, at known τ, with the coil sensitivities as a coarse spatial encoding. Fit a deliberately rich model over $T$ to them, take its principal modes, and refit with those modes alone. A subject's respiration then needs only the modes it has, not the larger set that describes respiration across all subjects.

The design uses the population instead: priors from physiology and from earlier subjects give the space that every subject's modes lie in, and the trajectory must keep that whole space observable. Check that with a Cramér–Rao bound before the scan; add a B0 term to the trajectory objective only if the bound falls short.

Measure first: the bound, then the phantom (drift), then a person (respiration). Build the change into the reconstruction only if it is above the threshold of section 6. Modes whose spatial content is of order 0 or 1 (a global frequency, linear gradients) cost nothing in the forward model.

## 1. Sources and scope

| source | time scale | spatial form | in scope |
|---|---|---|---|
| scanner drift (gradient heating, passive shim iron) | minutes | mostly a global frequency and linear gradients | yes |
| respiration (the chest's susceptibility changes with lung volume) | about 0.2–0.4 Hz | smooth, larger inferiorly; low order over the head | yes |
| head motion (the head's susceptibility field moves with it, and changes with its orientation) | any | the full map, in the head's frame | no: add with a motion model |
| cardiac, local (near vessels) | about 1 Hz | local | no: below what a low-order model sees |

**Head motion is important but out of scope here.** A rigid motion moves the map with the head and changes it with the orientation of the head relative to $B_0$. It belongs with a future motion model, which will need its own estimate of the head's pose over $T$, and which the modes below cannot replace. The estimators of section 3 will read a motion as a change of B0; the τ-linearity test and the residual check of section 6 should flag it.

The sizes in the brain at 2.89 T (the default `system.B0_T`) are not known for this scanner; the measurements of section 6 give them. The drift should be visible on the phantom; respiration needs a person.

## 2. The model

$$f(r, T) = f_0(r) + \Delta f(r, T), \qquad \Delta f(r, T) = \sum_{k=1}^{K} u_k(r)\, w_k(T),$$

- $f_0(r)$: the static map at full resolution, as now.
- $u_k(r)$: the spatial modes of this subject, each in the span of a **rich spatial basis** $Y_s(r)$: real spherical harmonics in the scanner frame up to order 2 or 3 (9 or 16 terms), or a population basis from priors (section 5). $u_k = \sum_s a_{ks}\, Y_s$.
- $w_k(T)$: the temporal modes, as cubic B-splines in $T$, with a spacing that resolves the change: about 30 s for drift, 0.5 s or less for respiration.
- $K$: the number of modes, from the data (section 3, stage 2). Expected: 1–3 drift modes on the phantom; those plus one or two respiration modes in a person.

**What a mode costs in the forward model** depends on its spatial content:

- **Order 0** (a global frequency) is a phase $e^{2\pi i\, \Delta f_0(T_m)\, \tau_m}$ on each sample $m$. It is diagonal on the data, so it does not change the normal operator $E^H W E$; only the right-hand side $E^H W y$ changes. No cost.
- **Order 1** ($g(T) \cdot r$, with $g$ in Hz/m) is a shift of the trajectory: $e^{2\pi i\,(k \cdot r + g \cdot r\, \tau)} = e^{2\pi i\,(k + g\, \tau) \cdot r}$. Correct the trajectory with $k_m + g(T_m)\, \tau_m$ before the NUFFT. No extra NUFFTs, and a Toeplitz kernel can be computed once from the corrected trajectory. The gradient-delay fit (basis-quality plan, section 3) already estimates a static background gradient $G$ in this form; order 1 is that gradient over $T$.
- **Order 2 and up** adds phase that is neither diagonal on the data nor a shift. It needs the time segmentation to cover the range of $f_0(r) + \Delta f(r, T)$ over both τ and $T$. With $K$ modes, one option is to segment over the $K$ mode amplitudes $w_k(T)\, \tau$ as well as τ. Measure that cost before adding it.

So split each mode into its order-0 and order-1 part, which is free, and the rest. If the rest is below the threshold of section 6, the reconstruction keeps its cost.

## 3. Estimators

### A. Modes from the crossings (the lead)

**The samples.** Not only those within a radius of $k = 0$: every sample in the neighbourhood of a crossing, weighted by the information it carries (section 4). The derivative of a sample with respect to a field term $Y_s$ is

$$\frac{\partial y_{vm}}{\partial \beta_s} = 2\pi i\, \tau_m\, \mathcal{F}[s_v\, x\, Y_s]\,(k_m)\, e^{2\pi i\, (f_0\, \tau_m)} ,$$

so it grows with τ (against the decay of the signal), and it is large where the spectrum of $s_v\, x\, Y_s$ is. For order 0 that is at $k = 0$. For order 1 it is the derivative of the object's spectrum, near zero at $k = 0$ and largest a few cells out (about 1 / the size of the object); the coils break part of that symmetry. So most of the information on the linear terms is on either side of each crossing, and a fixed radius of 0.5 cell would throw it away. The Gauss–Newton fit below weights the samples correctly if they are included.

**The reference.** The time-averaged reconstruction ($x$, $s_v$, $f_0$). The forward model is evaluated at the actual $k$ of each sample, so crossings that miss $k = 0$ by a fraction of a cell need no special case.

**Three stages.**

1. **Rich fit.** For each short window of $T$ (a few TRs), fit the coefficients $\beta_s$ of the rich basis by Gauss–Newton, with a complex scale for the window so that a change of the signal's magnitude (BOLD, inflow) is not read as phase. Start each window from the previous one: at τ = 45 ms the phase passes π at about 11 Hz, so the linearization holds only near the answer. Each window's fit is noisy but unbiased.
2. **Mode search.** An SVD over $T$ of the rich coefficients $\beta_s(T)$, whitened by their covariance from stage 1. The number of modes $K$ is the number of singular values above the noise edge (Marchenko–Pastur), as in the basis-quality plan (section 5). The right singular vectors give $w_k(T)$, the left ones $u_k(r)$.
3. **Refit.** Fit the $K$ temporal modes as splines, with $u_k$ fixed, over all the samples. Far fewer parameters than stage 1, so the estimate is better than the rich fit it came from. Optionally, update $u_k$ and $w_k$ in turn.

**Telling the field from other changes.** The field adds a phase that is zero at τ = 0 and grows linearly with τ. A change of magnitude (BOLD, inflow) is not phase; motion and a change of the object's phase add phase that is not proportional to τ. So test each mode for linearity in τ: fit the crossing phases of the mode with a slope in τ and with a free offset, and compare. A mode that needs the offset is not (only) field; in vivo, it is a candidate for the motion model.

**Limit: observability.** The search finds the best modes inside the span of the rich basis, among the directions the data can see. A spatial order that the coils and the neighbourhoods of the crossings cannot see will not appear. The bound of section 4 says which orders are observable, and so how rich stage 1 can be.

**Limit: structure the model imposes.** The estimate is smooth in space and low-rank over $T$ by construction: every voxel's field change is a combination of the same $K$ time courses, with spatial patterns from the rich basis. So the spatial patterns of the modes, and the shared time courses of different regions, are partly the model's, not measurements. Two consequences:

- **A change outside the span is moved, not lost.** A local change (for example near the sinuses, from swallowing) that the basis cannot represent is projected onto the modes, and appears as a smooth change over the whole head, where it corrupts the correction. Independent changes in two regions can be merged into one mode, which then gives them the same time course.
- **Do not read the modes as findings about the field.** A conclusion about how the field changes in space (for example the spatial pattern of the respiratory field, or whether two regions change together) needs evidence from outside the model: the residual (section 6, metric 4), estimator B, whose maps are not low-order in space, or the simulations of section 6, where the truth is known.

The rank at the noise edge, not at the number of modes expected, and the residual check limit this; the simulations of section 6 (metric 1) measure it.

### B. A field map for each window of T

The existing `field_map_hz` on windows of $W$ TRs, at 8 mm, from the gridding of the navigator (basis-quality plan, candidate B). Each window must also be split into windows of τ, so it needs more TRs than the navigator: probably $W$ ≥ 10 (0.5 s), which is marginal for respiration. It can replace stage 1 of A (the SVD of stage 2 then runs over the maps, inside a mask of the object), and it is a check of A. Use it as the estimator if A fails.

### C. Joint estimation in the reconstruction

$\Delta f = \sum_k u_k(r)\, w_k(T)$ updated in turn with the image, or Gauss–Newton on both. The most general and the most expensive: it puts a nonlinear outer loop around the reconstruction. Only if A and B fail.

## 4. The bound before the scan

The field model has few parameters (16 per window of $T$ for spherical harmonics up to order 3, or a few per mode), so the Fisher information $J^H J$ (section 3, the derivative of each sample) is small, and the forward model can compute it from simulated data of P4 and P1. Its inverse, the Cramér–Rao bound, answers before the scan:

- Can the crossings estimate each spatial order to well below the threshold of section 6 (about 0.35 Hz at the end of the readout)? This sets how rich stage 1 can be.
- At what resolution in $T$? Respiration needs about 0.5 s or finer. Sampling in $T$ is not the limit (40 crossings every 50 ms); the spread of τ, the neighbourhoods in $k$, and the noise are.
- Which samples carry the information: the weights for estimator A, and, if the bound falls short, what is missing (τ near T2\*, the samples a few cells out, the directions of the passes).

The bound depends on the object and the coil maps (use the simulated head phantom and its coils), and on a linearization at the current field. It is a prediction for a head-like object, not a guarantee for each subject.

## 5. Priors and the design

**Two different bases.** The design and the reconstruction need different things:

- The **reconstruction** fits the modes of one subject, found in that subject's data (section 3). A subject's respiration is described with the modes it has: one or two, where a model of respiration across all subjects needs more.
- The **design** must work for every subject before their data exist. Each subject's modes lie in the span of the population: the space that priors from physiology (smooth fields, larger inferiorly, the known spatial pattern of the respiratory field) and modes pooled from earlier subjects describe. The trajectory must keep that whole space observable.

So the design targets the population span, and the reconstruction finds the subject's modes inside it. The rich basis of stage 1 is that span: spherical harmonics up to order 2–3 now, a population basis once enough in vivo sessions have been analysed. Build the population basis from the richest estimates (stage 1, or estimator B), not from the subjects' modes: modes found inside a basis cannot show what is outside it, so a population basis made from them would only repeat the basis it started from.

**If the bound of section 4 falls short** for the population span, B0 becomes one more requirement in the objective of the design ([example-requirements-design.md](example-requirements-design.md), section 3): a fixed model whose conditioning is $J^H J$, as the frame potential measures the conditioning of the image models. The shapes set what is invariant under rotation (the τ of the crossings, and the radius of the samples around them); the schedule sets the directions of the passes through $k = 0$, which the order-1 terms need in all three directions within each window of $T$ (one TR has about 40 passes in different directions, so this probably holds already). The last resort is a navigator segment (basis-quality plan, candidate C).

## 6. Data, metrics and decision rule

**Data.**

- **Simulation.** `relaxed-rosettes simulate` already supports an off-resonance that changes over $T$ for each compartment (`Sine`, `Blocks` on `Property.OFF_RESONANCE`), so a global change is a `Sine` on every compartment. Linear gradients are not supported yet. For the analytic ellipsoids and blobs they are exact: a gradient $g$ shifts the sample's $k$ to $k + g\,\tau$ (section 2). Add a field term to the phantom config for that: a sum of modes, each a spherical-harmonic pattern with a time course.
- **Phantom (drift).** The session of the basis-quality plan (section 2): P4, P1, and the optional long P4 (about 200 s), which matters most here because drift grows over minutes. A static phantom has no respiration and no motion, so its change over $T$ is the drift.
- **In vivo (respiration).** The first in vivo session of the project, with a respiratory belt recorded, so that the estimate can be compared with breathing. Respiration cannot be tested on the phantom.

**Metrics.**

1. **Recovery (simulation).** Known modes: a drift (1 Hz and 5 Hz/m over 250 s) and a respiration mode with a spatial pattern of order 1–2 (0.5 Hz at 0.3 Hz), at the measured noise level. Whether the search finds $K$ correctly, the angle between the true and estimated spatial modes, the error of $w_k(T)$, and the residual phase at τ = 45 ms. Compare with the bound of section 4. Two cases test the structure the model imposes (section 3): a local change outside the span of the rich basis (how much of it the modes spread over the head), and independent changes of the same size in two regions (whether the estimate gives them a shared time course).
2. **Rank and size (phantom, in vivo).** $K$ at the noise edge, and each mode's range and spectrum over $T$. On the phantom, more than the drift modes (1–3) is an artefact of the estimator.
3. **τ-linearity.** For each mode, whether its phase is proportional to τ (section 3).
4. **Residual.** After the refit, the phase of the crossings that the modes do not explain, against τ, $T$, the shape and the rotation (as in the basis-quality plan, section 5).

**Threshold:** a change matters if $2\pi\, \Delta f\, \tau_{max}$ exceeds 0.1 rad, which is about 0.35 Hz for τ up to 45 ms. Set the thresholds before looking at the data.

- **Below the threshold** over the whole scan, on the phantom and in vivo: keep the static map. Record the sizes.
- **Above, with modes of order 0–1** (the rest below the threshold): add those parts to the forward model (section 8, steps 4–5).
- **Above, with more than order 1:** measure the extra segmentation cost first.
- **A mode that fails the τ-linearity test, or a residual correlated with motion** (in vivo): leave it for the motion model.

## 7. Relation to the other plans

- **The scan-time basis** ([basis-quality-test-plan.md](basis-quality-test-plan.md)). A changing B0 puts a phase $e^{2\pi i\, \Delta f(r, T)\, \tau}$ into the data, which is not separable in τ and $T$, so it adds components to $v_k(T)$. Estimate and remove it before the basis is estimated, so that the basis holds the object's change and not the field's. An error of the field estimate is a phase with the time courses of the field modes, shared by every voxel the modes cover; removed with the field, it leaves those shared time courses in the data, and a T basis learned from them can impose the same correlations on the image. For BOLD, that is false functional connectivity. The simulations of section 6 (metric 1) should also check the image's T basis after the field is removed. The field modes and the image's basis come from the same crossing data, by the same kind of SVD: the field from the phase that grows with τ, then the basis from what remains. On the static phantom, drift is the content the basis test expects; with B0 removed first, what remains of the drift is the change of the signal's magnitude.
- **The requirements** ([example-requirements-design.md](example-requirements-design.md), sections 3 and 3.1). "Data consistency (coils, B0)" means $f(r, T)$ of this plan, once it is built; until then, $f_0(r)$. B0 is a conditional term of the design objective (section 5).
- **The navigator tolerance** ([sample-spreading-plan.md](sample-spreading-plan.md), open questions). Estimator A evaluates the forward model at the actual $k$ and uses the neighbourhoods of the crossings, so it does not need exact crossings; the bound against `crossing_tolerance_cells` is one answer to that question.
- **Reconstruction speed** ([recon-optimization-plan.md](recon-optimization-plan.md), section 3.3). The order-0 and order-1 parts keep the Toeplitz form; higher orders add segments.

## 8. Implementation

Small steps, each with tests on simulated data:

1. **Phantom field term:** modes of spherical-harmonic patterns with time courses in the phantom config; order 0 and 1 exact by the shift of $k$.
2. **Bound:** the Fisher information of the rich basis for each window of $T$, from simulated data of a design, written to an `.npz` file (no plotting in this package). Section 4.
3. **Mode estimator:** a function in `relaxed_rosettes.recon` for the three stages of estimator A, and a CLI command that writes the modes, the rank, the τ-linearity test and the residuals to an `.npz` file. It reuses the crossing selection of the basis-quality plan (its step 1), extended to the neighbourhoods.
4. **Forward model, order 0:** the phase of each sample, in `SenseOperator`, from the modes.
5. **Forward model, order 1:** the trajectory corrected with $g(T)\, \tau$, as the delay correction of `export-traj` is applied.
6. **Config:** the rich basis (`b0.scan_time_order`, 0 = off), the window and spline spacing in $T$ (`b0.scan_time_spacing_s`), in `B0Config`.

Steps 1–3 are the measurement; steps 4–6 follow only if section 6 calls for them.

## 9. Results

None yet.
