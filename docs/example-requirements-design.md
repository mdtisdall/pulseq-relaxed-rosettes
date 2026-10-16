# Designing for the example requirements

Status: working document, started 2026-10-15. Its goal is a conclusion about the design (section 9): the objective of the shapes, the objective of the schedule across TRs, and the reconstructions that go with them, for one concrete set of requirements. Record each decision in section 8 with its date, and each measurement in the section it tests. The conclusions so far come from discussion, not from measurement; claims that are hypotheses are marked as such.

It builds on [coverage-target.md](coverage-target.md), which defines how an application states its requirements, and on [sample-spreading-report.md](sample-spreading-report.md), sections 4–6, which restate the design problem as an efficient front between slew energy and coverage.

Numbers below use the defaults: field of view 256 mm, TR 50 ms, 14 000 ADC samples per TR over about 45 ms, 5000 TRs (250 s), 128 shapes.

## 1. The requirements

The data have five dimensions: three of space, the **time since excitation τ** within each TR (the 4th: relaxation), and **scan time T** (the 5th: the time of the TR within the scan; motion and physiology). Each requirement states, for each dimension, whether it is fixed, averaged, or as fine as possible ([coverage-target.md](coverage-target.md), section 1).

| requirement | space | τ (4th) | T (5th) | k-space ball |
|---|---|---|---|---|
| navigator | fixed, about 8 mm | averaged | fixed, the shortest window (about 0.15 s, 3 TRs) | about 17 000 cells |
| relaxation 4D | fixed, 1.5 mm | resolved | averaged | about 2.6 million cells |
| static | fixed, 1 mm | averaged | averaged | about 8.8 million cells |
| BOLD | as fine as possible | as fine as possible | fixed, 0.5 s (10 TRs) | — |

The navigator's window is a placeholder. For BOLD, the measured resolution–time curve of the default design (about 1 s at 4 mm) suggests about 5 mm at 0.5 s with τ averaged; resolving τ costs resolution (section 2).

**Resolution in T is a kernel width, not a window.** The reconstructions that resolve T (navigator, BOLD) model a continuous volume in T, with each sample entering where it falls, weighted by a smooth temporal basis of the required width. They need not make separate images from fixed windows.

## 2. What the requirements imply for coverage

From [coverage-target.md](coverage-target.md), sections 2–4:

- **Two curves.** The navigator and the static image anchor the curve with τ averaged, $W^*_{\text{avg}}(\text{res})$. The relaxation 4D image anchors the curve with τ resolved, $W^*_{\tau}(\text{res}; L)$, which costs about $L$ times the samples, for a τ basis of rank $L$.
- **BOLD is a priority, not an anchor.** It is read off the two curves at $T$ = 0.5 s, with a preference for a small $L$ (2–4 τ values to separate T2\* from $S_0$).
- **Averaging needs no correlation.** An image averaged over τ is clean only if the τ distribution of the samples is about the same at every $|k|$; the same holds for T.
- **Variation over T, for BOLD.** In a continuous model there are no window boundaries. The concern is the temporal spectrum of the sampling: coherent structure in the BOLD band (about 0.01–0.1 Hz) or at the respiratory and cardiac frequencies adds false signal; incoherent variation raises the noise floor; variation pushed above about 1/(0.5 s) (blue noise in T) mostly falls outside the band. The current TR order is a random shuffle: no coherent structure, but white variation.

## 3. The objective: conditioning of each reconstruction

**Principle.** Design the trajectory so that the reconstructions that will be run are well-conditioned. Each requirement defines a reconstruction model; the reconstruction fits it, and the design objective measures how well the samples determine it. So the reconstruction and the trajectory objective come from the same source.

### 3.1 The reconstruction models

Each requirement is a linear model $x(r, \tau, T) = \sum c \, \phi_l(\tau)\, b_j(T)$ on a spatial grid at its resolution, fitted by data consistency (coils, B0) plus regularization: CG-SENSE extended with τ and T bases. B0 is the static field map $f_0(r)$ for now. A change of B0 over T (scanner drift, respiration) would make it $f(r, T)$; [b0-scan-time-plan.md](b0-scan-time-plan.md) measures that change first, and adds it as a few modes found in the data only if it is large enough to matter.

| requirement | spatial grid | τ basis | T basis |
|---|---|---|---|
| navigator | 8 mm | constant ($L$ = 1) | splines about 0.15 s apart |
| relaxation 4D | 1.5 mm | rank-$L$ subspace | constant |
| static | 1 mm | constant | constant |
| BOLD | from the curves at 0.5 s | small $L$ (2–4) | splines 0.5 s apart |

BOLD analysis can work on the spline coefficients directly instead of on windowed images.

**Averaging is a model choice.** The static reconstruction with $L$ = 1 treats the real τ variation of the signal as model error; the no-correlation condition of section 2 keeps that error incoherent. The alternative is to reconstruct with the full τ basis and average afterwards: more expensive, and it removes the condition.

### 3.2 The frame potential

For a linear model, the samples are rows $a_m$ of the forward operator $A$. The reconstruction is best conditioned when $A^H A$ is close to a multiple of the identity (a tight frame). The standard measure of the distance from that is the **frame potential**,

$$
\text{FP} \;=\; \lVert A A^H \rVert_F^2 \;=\; \sum_{m,n} \big| K(z_m, z_n) \big|^2 ,
\qquad (1)
$$

with $z = (k, \tau, T)$ a sample and $K$ the reproducing kernel of the model space: the product of

- the k-space kernel of the ball at the requirement's resolution (sinc-like, made local and smooth by a Hann taper);
- the τ-basis kernel $\sum_l \phi_l(\tau)\, \phi_l(\tau')$, or 1 if τ is averaged;
- the T-basis kernel, about the requirement's width in T, or 1 if T is averaged.

The trajectory objective is

$$
\text{slew energy} \;+\; \mu \sum_{\text{req}} \alpha_{\text{req}}\, \text{FP}_{\text{req}} ,
\qquad (2)
$$

with the priorities $\alpha$ of the requirements, swept over $\mu$ to trace the front.

- **It is measure 8 of the report with its kernel chosen.** The kernel is the squared model kernel of each reconstruction, not a free choice. That settles the kernel-shape question: the Hann taper matters because it makes $K$ local and smooth, and so the objective cheap and well-behaved.
- **In Fourier terms it is weighted aliasing energy.** For translation-invariant kernels, the kernel sum equals the energy of the point-spread function weighted by the kernel's Fourier transform. So each term is the aliasing energy of that reconstruction's PSF, counted only where it lands inside the model: inside the field of view, within the τ basis, and at the temporal frequencies the T basis resolves. For BOLD, the weight in temporal frequency can be shaped to the BOLD band and the physiological frequencies, so the k–T spectral score of section 2 becomes part of the objective.
- **It replaces the energy distance** of the repulsion (PR #23), which the report found misaligned with coverage.
- **The whole-scan terms (static, relaxation 4D) couple all pairs** of about 70 million samples. Direct pair sums are impossible; the Fourier form can be evaluated with a NUFFT (FINUFFT, already used by the reconstruction). The navigator and BOLD terms are local in T and much cheaper.

**The cost axis may depend on the requirement.** Scanner noise activates auditory cortex and limits comfort in fMRI, so BOLD argues for adopting band-filtered slew energy (acoustics) sooner than the others do. PNS matters for all.

**Caveats.**

- Coils are ignored: these are Nyquist-level frames ($R$ = 1). A coil-aware kernel is the principled way to bring in $R(\text{res})$ later.
- The frame potential is an average-case measure. Measure 3 (the worst holes) stays as the score.
- Each term alone favours uniform density within its ball. *Hypothesis:* the weighted sum of nested terms gives the centre-heavy density that the curves need. Its relation to the SNR-optimal `filter` density is not worked out.
- Nonlinear reconstructions (for example low-rank 5D) are outside this argument.

## 4. What the shapes control, and what the schedule controls

**A rotation preserves $|k|$, τ, and the slew energy of every sample.** It changes only the direction in k-space. So:

- **the shapes** set everything about radius, τ, their joint distribution, and the cost;
- **the schedule** (which shape, at which rotation, in which TR) sets everything about direction and about timing in T, and through how often it uses each shape, part of the radial density.

| requirement | set by the shapes | set by the schedule |
|---|---|---|
| navigator (about 3 TRs) | almost all: the 8 mm ball is about 17 000 cells and one TR has 14 000 samples weighted to the centre, so each shape must cover the coarse ball nearly by itself, in all directions | the 2–3 neighbouring TRs fill each other's angular holes (a local term) |
| BOLD (10 TRs) | the radial density, and the spread of τ at each radius (the small $L$) | the angular fill across the 10 TRs, and the k–T spectrum (no periodicity; variation above the band): entirely a schedule property |
| relaxation 4D (whole scan) | that every radius is sampled in every τ group: a property of the $(\lvert k \rvert, \tau)$ distribution, which rotations cannot change; it can hold over the library, with shapes whose petal orders in τ complement each other | the angular fill of each τ group over the whole scan (a global term) |
| static (whole scan, about 600 TRs to fill) | the radial density, and no $\lvert k \rvert$–τ correlation (again over the library) | the angular fill over the whole scan, and the early-stop property (section 5) |
| cost (slew energy) | all of it: slew energy does not change under rotation | none; PNS and acoustics depend somewhat on which gradient axis carries the slew |
| spoiling (section 5.1) | the size of the spoil moment, and its direction relative to the shape's petals | the direction of each TR's spoil moment, so where the spoiled magnetization of every earlier TR lies when the next readouts sample k-space |

By window length: windows of about one TR (navigator) are set by the shapes; windows of a few to tens of TRs (BOLD) by the shapes in radius and τ and by the schedule in direction and T; whole-scan windows (static, relaxation 4D) by the library in radius and τ and by the schedule in angular fill.

**Consequences.**

1. **The cost axis of the front is a shape problem.** The schedule improves coverage at no slew cost, so optimizing it moves designs toward the front for free. This argues for doing the schedule first. It is not free in spoiling, though: the rotation that places a TR's samples also places its spoiled magnetization (section 5.1).
2. **The shape objective gets only the terms the shapes control:** the coverage terms on $(\lvert k \rvert, \tau)$, the coarse angular coverage of each shape for the navigator, and slew energy. Within-shape repulsion at fine scales is of little use, since fine angular fill comes from the schedule; this agrees with the report's finding that only coarse scales (4–8 mm) are reachable within one shape.
3. **Some requirements belong to the library, not to single shapes:** the $(\lvert k \rvert, \tau)$ coverage for relaxation 4D and the no-correlation condition for the static image hold over a set of shapes. So the library is designed as a set. A window short enough to contain only a few shapes (BOLD's 10 TRs) sees only part of that complementarity, so the schedule should also mix the shapes in time. The schedule's choice of how often to use each shape feeds back into the radial density.

## 5. The schedule: greedy across TRs

**Approach.** Shapes form a library (about 128), optimized offline. For each TR in turn, pick the shape and rotation that most reduce objective (2), given all earlier TRs. This is selection from candidates, not waveform optimization. It extends step 3, section 4.3 of [sample-spreading-plan.md](sample-spreading-plan.md) (greedy rotations against recent TRs) with the requirement terms.

**Why.**

- **Size.** The full problem (shapes and schedule together) is very large; greedy selection splits it into a small slow part (the shapes) and a fast sequential part (the schedule).
- **Early stopping.** Greedy minimization of a kernel discrepancy is kernel herding. In that literature, prefixes of a herded sequence approach the target faster than random ones (about $1/N$ against $1/\sqrt{N}$, under its assumptions). Today's schedule is a shuffled super-Fibonacci set, which is not nested, so its prefixes are random. Early stopping matters only for the requirements averaged over T (static, relaxation 4D); for the navigator and BOLD, the value of greedy is consistency over the scan.

**Cost per step.** The objective is a sum of kernel terms, so adding a TR changes it only through its interaction with what is already there. Each term is kept as an accumulated k-space grid:

- terms averaged over T: one grid of all earlier samples convolved with $\lvert K \rvert^2$;
- terms local in T: with an exponential kernel in T, the grid decays at each TR and the new samples are added, O(1) per TR. This makes each TR differ from recent ones: blue noise in T;
- τ-resolved terms: one grid for each of the $L$ τ groups.

Scoring a candidate is then a lookup of its samples in the grids; subsampling candidates or a GPU should make it practical. Grids at 1 mm are tens to hundreds of MB, so measure memory on one configuration before any parallel run.

**Risks.**

1. **Quasi-periodicity.** Herded sequences can become quasi-periodic; for BOLD, periodic structure is the worst case (coherent aliasing in the band). Mitigation: score a random subset of candidates at each step, and check the k–T spectrum of every schedule.
2. **Myopia.** Late TRs have only small holes to fill, and greedy never revisits a choice. A final swap pass could improve the full scan at some cost to the prefixes.
3. **The schedule partly sets the density** through how often it uses each shape; check the coverage curves.
4. **The gain may not show in measure 3.** An earlier measurement (plan, section 4.2) found that TR order did not change the median gaps. The gains to expect are in prefix quality, the k–T spectrum and conditioning.

### 5.1 Spoiling: the rotation also places the spoiled magnetization

**How the sequence spoils.** Each shape ends at its spoil moment, $5\,k_{max}$ (`spoil_moment_kmax`) along a direction fixed relative to its petals, and the TR's rotation turns the whole waveform, spoiler included. So TR $p$ leaves its transverse magnetization dephased by $s_p = R_p\, m_{\text{shape}(p)}$, a vector of length $5\,k_{max}$ whose direction the schedule chooses. RF spoiling adds a phase to each excitation (`excitation.rf_phase`: random by default, or the quadratic 117° scheme).

**Where the spoiled energy goes.** Later pulses split that magnetization into coherence pathways (an extended phase graph in 3D, with continuous dephasing vectors instead of integer states). A pathway that reaches TR $q$ has net dephasing

$$
M \;=\; \sum_{i=p}^{q-1} \varepsilon_i\, R_i\, m_{\text{shape}(i)}, \qquad \varepsilon_i \in \{+1, -1, 0\},
$$

with $+1$ for a TR spent transverse, $-1$ after a pulse conjugates it (spin-echo-like), and $0$ while it is stored longitudinally (stimulated echoes). During the readout of TR $q$ its signal is the object's spectrum at $M + k_q(t)$. So it is sampled, as a spurious echo, when the readout passes near $-M$: in practice when $\lvert M \rvert$ is less than about $k_{max}$ plus the few cells over which the object's spectrum is still strong. A single TR's spoiler ($5\,k_{max}$) is far outside the readout ball. Sums over several TRs are not: they are a random walk of $5\,k_{max}$ steps in directions that the schedule chooses, and a walk can return near the origin.

**The pathways that matter**, at the defaults (flip angle 15°, TR 50 ms):

- **Transverse chains** ($\varepsilon = +1$ throughout): each pulse keeps $\cos^2(\alpha/2) \approx 0.98$ of the coherence, and T2 and intravoxel dephasing reduce it by roughly half or more per TR. Strong, but short-lived: 1–3 TRs. Two consecutive TRs return to the ball if their spoilers are nearly antiparallel ($\lvert u_p + u_{p+1} \rvert < 0.2$, an angle above about 168°): about 1% of pairs for independent random directions.
- **Conjugated pathways** ($\varepsilon$ changes sign): weighted by $\sin^2(\alpha/2) \approx 0.02$ per conjugation. Two TRs return if their spoilers are nearly parallel (within about 11°), again about 1% of random pairs.
- **Stimulated echoes** (with $\varepsilon = 0$ stretches): each conversion between transverse and longitudinal costs a factor of order $\sin\alpha$, but storage decays only with T1 (about 4% per TR), so these last tens of TRs, and there are many of them.

**Why incoherent spoiling may not carry over.** In 3D gradient-echo imaging with incoherent spoiling, the spoiler directions are independent of the imaging and effectively random: the returns near the origin are rare, uncorrelated, and, with RF spoiling, add incoherently. Here the directions are not independent of the imaging. The schedule chooses each rotation to place the samples (section 5), so the spoiler directions inherit the schedule's structure:

- a greedy coverage objective is blind to spoiling, and can choose consecutive rotations whose spoilers are nearly parallel or antiparallel;
- quasi-periodic greedy sequences and cycled roles (section 6.1) make the spoiler sequence periodic, so returns near the origin can repeat at a fixed period: coherent spurious signal, and for BOLD a periodic fluctuation (section 2);
- the 117° quadratic RF scheme is designed for a constant spoiler direction; with rotating spoilers its benefit is not established, and the random scheme makes residual pathways add incoherently but varies the steady state from TR to TR.

**Effect on the requirements.** Spurious pathways bias every reconstruction, with different signatures: for the static image, a contrast change and incoherent artefact; for relaxation 4D, pathways whose τ dependence is not the FID's (stimulated echoes do not decay with T2\* in the same way), so they corrupt the relaxation fit; for BOLD, a fluctuation over T that follows the schedule, which must stay out of the band, like the sampling variation of section 2.

**What to add to the design.**

- **A spoiling term in the schedule's objective.** Keep the low-order pathways (transverse chains over the last few TRs, conjugated pairs, and stimulated echoes with one storage period over the last tens of TRs) as a list of dephasing vectors with estimated amplitudes. For a candidate rotation, update them with its spoiler and score the amplitude-weighted energy of those that fall near the readout ball, for example $\sum a\, S(\lvert M \rvert - k_{max})$ with $S$ a model of the object's spectral decay. This is O(pathways) per candidate, cheap beside the coverage lookup. A hard constraint is the simpler alternative: no transverse chain of up to 3 TRs and no conjugated pair within a margin of the ball.
- **Spoiler parameters in the shape design.** The size of the spoil moment trades slew energy (the cost axis) against how far the walk must go to return. Its direction relative to the petals is a free parameter of each shape; a library with varied spoiler directions gives the schedule rotations that are good for both coverage and spoiling.
- **The RF phase scheme**, chosen with the schedule, by simulation (experiment 8).

## 6. The shapes

From section 4, the shape objective is: slew energy, plus the frame-potential terms restricted to what one shape (or the library) controls:

- **the $(\lvert k \rvert, \tau)$ distribution of the library**: every radius in every τ group for relaxation 4D, and no $\lvert k \rvert$–τ correlation for the static image;
- **coarse angular coverage of each shape**, for the navigator;
- **the radial density** that the curves need.

Open: how to make the library complementary in $(\lvert k \rvert, \tau)$ (for example, assign each shape an order of tip radii in τ so that the set covers the plane evenly), and whether shapes should be re-optimized for the statistics of the schedule, alternating the two.

### 6.1 The number of shapes

Rotations do not change $\lvert k \rvert(t)$, so the $(\lvert k \rvert, \tau)$ plane is sampled only by the shapes. The number of shapes is therefore about the $(\lvert k \rvert, \tau)$ terms; the angular terms are met by rotations at any number of shapes. The `Config` docstring records a measurement on that plane: 128 shapes halve the empty cells (20 shells × 90 bins of 0.5 ms) compared with 32, and 256 add little.

**The subspace model lowers what the plane needs.** The reconstructions resolve τ only through a basis of rank $L$ (section 3.1). So at each radius, what matters is the projection of the τ distribution of the samples onto $L$ basis functions, not its values in 0.5 ms bins:

- **relaxation 4D** needs, at each radius up to 1.5 mm, enough samples in each of $L$ τ groups;
- **averaging over τ** (static, navigator) needs, at each radius, the same $L$ projections: a finite condition (the first $L$ "moments" of the τ distribution in the basis), not an even distribution in every bin.

So measure 4 overstates the need, and the right count of shapes is the one that meets these $L$-dimensional conditions. *Hypothesis:* a designed, complementary library meets them with far fewer shapes than a random one.

**The relevant number is the size of a complete set, not the total.** Call a set of shapes *complete* if their union meets the $(\lvert k \rvert, \tau)$ conditions. The windows that need those conditions bound its size $m$:

- BOLD resolves τ (small $L$) within 10 TRs, so a complete set for its $L$ must fit in 10 TRs: $m \le 10$;
- the navigator averages over τ within about 3 TRs, so the no-correlation condition for its coarse ball should hold within about 3 shapes, or within each shape;
- relaxation 4D and the static image average over T, so any complete set is enough, but early stopping (section 5) favours completing sets early and often.

This suggests a **library of complete groups**: $S = G \times m$ shapes in $G$ groups, each group complete, with the schedule playing a whole group within each short window. A window then has the same $(\lvert k \rvert, \tau)$ statistics wherever it starts, which is also what the k–T spectrum of BOLD needs (section 2).

**What more shapes buy, beyond one complete group.**

- Diversity in $(\lvert k \rvert, \tau)$ beyond rank $L$: little, if the subspace model holds; it is insurance if the signal is not well described by rank $L$.
- Radial resolution of the density: with about 43 tips per shape, even 32 shapes give about 1400 tip radii, plenty for a smooth radial density over 128 shells at 1 mm.
- Choice for the greedy schedule: small, since the rotations already give a continuous choice.
- Less repetition of identical waveforms, which may matter for eddy-current and PNS behaviour that depend on the waveform.

**What more shapes cost.**

- Shape optimization time and memory, linear in $S$.
- Each shape's actual trajectory may need measuring (for example with a gradient impulse response, plan, open decisions); fewer shapes, fewer to calibrate.
- Sequence size: every distinct waveform is stored in the `.seq` file.

**Diminishing returns, not an optimum.** In theory, more shapes are never worse: a library of $S$ shapes can repeat one of them and so do anything a library of $S - 1$ can, so the best achievable objective (2) cannot increase with $S$. The gain saturates, though:

- the $(\lvert k \rvert, \tau)$ terms stop improving once the rank-$L$ conditions are met; beyond that, more shapes are insurance against a signal that rank $L$ does not describe well;
- more shapes can share the coverage obligations, so each shape can be cheaper in slew energy. For the windowed requirements (BOLD, navigator), the sharing is limited to the shapes within one window, so this gain saturates near the group size $m$; for the whole-scan requirements it continues further, but weakly.

The costs grow about linearly in $S$ (optimization, calibration, sequence size), and a jointly designed library of many shapes is a harder optimization, more likely to stop at a poor local optimum. In practice, more can then be worse. So there is a "good enough" $S$ for each design: where the measured gain on the scores of the requirements drops below their measurement noise, or stops being worth the added cost. Experiment 3 of section 7.2 measures that curve for random libraries.

**Fitting with neighbours.** Part of the difficulty is that each shape must fit with the shapes it shares windows with. If every shape had to fit with *any* $m - 1$ others, the number of combinations would be $\binom{S}{m}$: polynomial in $S$, but exponential in the window size $m$. Three things keep the demand manageable:

- **Only the windows that occur matter.** The schedule is chosen, not random, so the constraints are on the windows of the actual sequence: about one for each TR, so linear in the number of TRs. The combinatorial difficulty moves into choosing the order, which is what the greedy schedule (section 5) handles.
- **Without structure, the shapes still conflict.** In an arbitrary order, each shape has many different sets of neighbours over the scan, and it must fit with all of them. That pushes each shape toward being complete by itself, which is expensive.
- **Roles remove the conflict.** Give each shape one of $m$ *roles* in $(\lvert k \rvert, \tau)$, design the roles to be complete together, and let the schedule cycle through the roles: role 1, 2, …, $m$, 1, 2, …. Then *any* $m$ consecutive TRs contain every role exactly once, wherever the window starts. Each shape needs to fit only its role: the demand is linear in $m$, and the greedy schedule keeps its freedom in which shape of the role, and in the rotation. Roles can be nested for the shorter windows: for example, $m = 9$ as 3 × 3, so that every 3 consecutive TRs meet the navigator's no-correlation condition, and every 9 meet BOLD's rank-$L$ condition.

The price of roles is a period of $m$ TRs in the $(\lvert k \rvert, \tau)$ sampling: 0.5 s, or 2 Hz, for $m$ = 10. That is above the BOLD band, where section 2 wants the variation, but a physiological signal near the period can alias into the band: for example, a cardiac harmonic near 2 Hz would alias to $\lvert f - 2\ \text{Hz} \rvert$, which can be below 0.1 Hz. Check this in the k–T spectrum; jittering the role order between cycles would trade it against the completeness of every window.

*Hypothesis:* for these requirements the good-enough $S$ is a few complete groups of $m \le 10$ shapes, so tens of shapes rather than 128, provided the library is designed as complementary groups. A random library needs more shapes for the same coverage, which is what the 32 against 128 measurement shows.

## 7. Plan of in silico experiments

Each experiment tests a claim of sections 3–6. Record each result here, in the same PR as the experiment, and correct the sections it tests. Coils ($R(\text{res})$), PNS and acoustics are left out until the objective is validated (experiment 1).

### 7.1 Shared tools

The scan-time basis of [basis-quality-test-plan.md](basis-quality-test-plan.md) is about T, not τ.

- **τ basis** (`relaxed_rosettes.recon.basis`): the SVD of decay curves over the ADC window, for a range of R2\*. It gives $L$ and $\phi_l(\tau)$. Off-resonance stays in the field map and time segments of the reconstruction; its change over T, if needed, in the modes of [b0-scan-time-plan.md](b0-scan-time-plan.md).
- **Frame-potential evaluator** for each requirement: the NUFFT (PSF) form for the terms averaged over T, the accumulated-grid form for the terms local in T (section 5).
- **k–T spectral score**: for each k-region at a requirement's resolution, the power spectrum of its sample occupancy over T, integrated over the BOLD band.
- **Numerical phantom** $x(r, \tau, T)$ (`relaxed_rosettes.recon.phantom`, `relaxed-rosettes simulate`): analytic ellipsoids and Gaussian blobs, each with its own density, R2\* and off-resonance, which can vary over scan time (a BOLD R2\* change in blocks, cardiac and respiratory sines). The signal is exact at every sample, written as ISMRMRD. Needed only for experiments 1, 6 and 7.

### 7.2 Experiments

1. **Does the frame potential predict reconstruction quality?** The gate for the rest: section 3 replaces the energy distance because it was misaligned, so the new objective needs the check the old one failed.
   - Setup: designs A–M of the report and a few 128-shape designs. For each, the frame potential of each requirement, and the matching linear reconstruction (section 3.1) of phantom data with noise; error and noise against the ground truth.
   - Decision: the rank correlation of frame potential with reconstruction error, against that of measure 3 and of the energy distance. If the frame potential does not predict the error better, revise the objective before optimizing it.
   - Status: the static requirement can run with the existing code (sub-experiment 1a). The other three need reconstructions that do not exist yet: a τ-subspace CG-SENSE (relaxation 4D) and a CG-SENSE with a spline basis in T (navigator, BOLD). Add them after 1a, if 1a does not already revise the objective.

   **1a. The static requirement only.** Started 2026-10-15. The first gate: the static requirement is the one where the report found the energy distance misaligned with coverage, and its model (1 mm, τ and T averaged) is what `relaxed-rosettes sense` reconstructs now.
   - Setup, as run (2026-10-15 to 2026-10-16; driver `run1a.py` and results in `data/experiment-1a-2026-10-15/`, local, not tracked): designs A–M of the report (16 shapes, 5000 TRs, 250 s), and as a separate group P1 (128 shapes) and its band-penalty variants P1pen1.1, P1pen1.5 and P1pen2.1 (16 shapes), all with 1500 TRs (75 s). For each: the static frame-potential ratio; measure 3 as the 1 mm gap (95th percentile, cells) of the whole scan, plus windows of 15, 30, 60 and 125 s; the energy distance; and the static reconstruction of the phantom, once without noise and once with noise. Error: the NRMSE of the noiseless reconstruction against the ground truth. Noise: the RMS of the difference between the noisy and the noiseless reconstructions, relative to the norm of the ground truth. The real τ variation (R2\*) stays in the data, as the model error that the static model averages over (section 3.1). All runs at commit 489f030; the trajectories of A, F, M and P1 are identical under a629a48.
   - Changes from the plan, and why:
     - P4 is dropped: it is a 4 mm design and cannot fill the 1 mm ball.
     - The P1 group has 4 designs, too few for rank correlations. Its results are reported, not correlated.
     - The A–M shape files predate the config fields of PR #32. Their config digests were re-stamped on copies. The new fields default to no band penalty, so the waveforms are unchanged.
     - Reconstruction: not `simulate` and `sense`, but the driver: unweighted least squares plus Tikhonov (1e-3, relative to $A^H A / M$), one coil of sensitivity 1, no B0, CG from 0 for 30 iterations, with the noisy and the noiseless data as two channels of one CG. `sense` estimates coil maps from the data, prewhitens, and uses DCF-weighted least squares; none of these is the linear model that the frame potential scores.
     - Ground truth: the phantom's analytic Fourier transform, averaged over the τ of the ADC samples, limited to the 1 mm ball on the Cartesian grid of the field of view, and transformed back: the image that perfect sampling of the ball would give.
     - Phantom: the default, with off-resonance 0 (already the default), and with its scan-time modulations (vessel sines, BOLD blocks) removed, so that the static truth does not depend on T.
     - Energy distance: the mean over shapes, to one common target for all designs: the `exponent` model (resolution–time exponent 2.5) of the default config.
   - Before the runs in parallel: peak memory at most 14.1 GB per process; 3 processes at a time.
   - Decision: as for experiment 1, with the static term alone: the rank correlations with the error, and with the noise, of the static frame-potential ratio, of measure 3 at 1 mm, and of the energy distance. If the frame potential does not predict better, revise the objective before the other reconstructions are built.
   - Results (2026-10-16). A–M:

     | design | FP ratio | gap (cells) | energy distance | error | noise |
     |---|---|---|---|---|---|
     | A | 232 | 1.21 | 0.0150 | 0.318 | 0.0419 |
     | B | 268 | 1.29 | 0.0119 | 0.347 | 0.0361 |
     | C | 274 | 1.23 | 0.0121 | 0.341 | 0.0371 |
     | D | 284 | 1.15 | 0.0123 | 0.332 | 0.0355 |
     | E | 282 | 1.15 | 0.0122 | 0.315 | 0.0340 |
     | F | 715 | 1.43 | 0.0127 | 0.142 | 0.0222 |
     | G | 749 | 1.09 | 0.0107 | 0.149 | 0.0207 |
     | H | 284 | 1.14 | 0.0123 | 0.296 | 0.0324 |
     | H0 | 715 | 1.36 | 0.0126 | 0.143 | 0.0222 |
     | I | 748 | 3.95 | 0.0104 | 0.151 | 0.0218 |
     | J | 734 | 7.28 | 0.0103 | 0.147 | 0.0220 |
     | K | 670 | 2.88 | 0.0083 | 0.144 | 0.0212 |
     | L | 699 | 2.33 | 0.0073 | 0.159 | 0.0203 |
     | M | 427 | 1.84 | 0.0141 | 0.152 | 0.0265 |

     The P1 group: error 0.148–0.169, noise 0.0326–0.0388.

     Spearman rank correlations, A–M (14 designs):

     | score | vs error | vs noise |
     |---|---|---|
     | frame potential | −0.80 | −0.86 |
     | gap (measure 3) | −0.46 | −0.45 |
     | energy distance | 0.09 | 0.63 |

   - Reading:
     - The gate fails as posed. A higher frame-potential ratio goes with a *lower* error (−0.80) and a lower noise (−0.86). The gap has the wrong sign too (−0.46).
     - The error splits by waveform family, not by any of the three scores: the relaxed designs (A–E, H) 0.30–0.35; the hairpin and filter designs (F, G, H0, I–M) 0.14–0.16. H0 (hairpins on the relaxed targets) falls with the hairpins, so the waveform style matters, not the targets. I and J (gaps of 3.95 and 7.28 cells) reconstruct as well as G (1.09 cells).
     - The extra error of the relaxed family is below 0.25 $k_{max}$: there its share of the error² is about 0.09 (A, E) against 0.011 (H0, F); above it, the same. In those shells the sample counts and the τ distributions are the same in both families (mean, standard deviation and 5–95% range of τ within 1–2 ms), so neither the radial density nor the $\lvert k \rvert$–τ correlation explains it.

   **1a, control: R2\* = 0.** The same reconstructions of the phantom with every R2\* set to 0, so that the static model is exact and only the sampling (and Tikhonov) error remains. P1 group skipped.

     | design | error (R2\*) | error (R2\* = 0) | noise (R2\* = 0) |
     |---|---|---|---|
     | A | 0.318 | 0.110 | 0.0266 |
     | B | 0.347 | 0.120 | 0.0198 |
     | C | 0.341 | 0.116 | 0.0220 |
     | D | 0.332 | 0.121 | 0.0198 |
     | E | 0.315 | 0.120 | 0.0199 |
     | F | 0.142 | 0.135 | 0.0134 |
     | G | 0.149 | 0.126 | 0.0164 |
     | H | 0.296 | 0.123 | 0.0186 |
     | H0 | 0.143 | 0.133 | 0.0137 |
     | I | 0.151 | 0.132 | 0.0143 |
     | J | 0.147 | 0.133 | 0.0136 |
     | K | 0.144 | 0.125 | 0.0173 |
     | L | 0.159 | 0.133 | 0.0133 |
     | M | 0.152 | 0.132 | 0.0175 |

   - Spearman, A–M, at R2\* = 0: frame potential +0.81 against the error and −0.84 against the noise; gap +0.55 and −0.56; energy distance −0.21 and +0.38.
   - Reading: the relaxed family's extra error is τ-averaging model error. It falls from 0.30–0.35 to 0.11–0.12 at R2\* = 0, while the hairpin and filter designs change little (0.14–0.16 to 0.125–0.135). The static frame potential cannot see this error, because its model has τ averaged, and here it is the larger part. With it removed, the frame potential ranks the sampling error with the right sign, better than the gap, but over a narrow range (0.110–0.135), and it ranks the noise with the opposite sign: the high-ratio designs have less noise.
   - Check at 4 mm (Tier 1, section 7.3), the static model at 4 mm on the same designs:

     | design | FP ratio (4 mm) | error (R2\*) | error (R2\* = 0) |
     |---|---|---|---|
     | A | 41.9 | 0.287 | 0.055 |
     | E | 50.6 | 0.289 | 0.062 |
     | F | 121.4 | 0.091 | 0.069 |
     | H0 | 119.9 | 0.089 | 0.067 |

     It reproduces both effects, the family split and its disappearance at R2\* = 0, about 12 times faster than 1 mm.
   - Decision and next: keep the static frame potential as the sampling term. The objective needs a second term, for the τ-averaging model error of an averaged model. This bears on the no-correlation condition of section 2 and on experiment 6: the per-shell τ statistics did not detect the difference, so the cause must be local, among samples near each other in k. *Hypothesis*, to test at Tier 0 or 1 before the term is designed.
2. **Do nested terms give a centre-heavy density?** Tests the hypothesis of section 3.2.
   - Setup: a radial problem, with no waveforms: choose $D(\lvert k \rvert)$ to minimize $\sum \alpha\, \text{FP}$ over the four requirements' balls at a fixed number of samples. Compare with the `exponent` and `filter` densities and the measured curve (about $\text{res}^{-2.5}$).
   - Cheap. It shows whether the requirements imply a radial density by themselves.
3. **How many random shapes meet the $(\lvert k \rvert, \tau)$ conditions?** The "good enough" curve of section 6.1 for a random library; no optimization.
   - Setup: random libraries of 8, 16, 32, 64 and 128 shapes; windows of 1, 3, 10 and all TRs. At each radius, the samples in each of the $L$ τ groups (relaxation 4D), and the spread across radii of the $L$ basis projections of the τ distribution (no correlation, for averaging).
   - Decision: where the curves flatten. If one complete set needs far more than 10 shapes, BOLD cannot be complete within its window, and roles are needed, not optional.
4. **Greedy schedule against the shuffle, with the current shapes.** Tests sections 4 and 5: coverage at no slew cost, and the early-stop property.
   - Setup: the 128 shapes, a dense set of SO(3) rotations as candidates; pure greedy, greedy on a random subset of candidates at each step, and the shuffle. Memory check on one configuration first.
   - Measures: the static term and measure 3 at 25%, 50% and 100% of the scan; the static frame potential against the number of TRs on a log–log plot (herding predicts a steeper slope than the shuffle); the k–T spectrum in the BOLD band (pure greedy may be quasi-periodic); the navigator's gaps at short windows; the time to score the candidates of one TR.
5. **A library with roles.** Tests section 6.1: $m$ roles complete together make every window of $m$ TRs complete, at lower slew energy than shapes that are each complete.
   - Setup: assign tip radii to τ slots across $m$ = 9 or 10 roles (3 × 3 nested), with the existing target and tip code, then the SCP shape solve.
   - Measures: the conditions of experiment 3 for every window of $m$ TRs, against a random library of the same size; the slew energy of each shape (the claim that shared obligations make shapes cheaper).
6. **Is "no correlation" the right condition for averaging?** Tests sections 2 and 3.1: averaging is a model choice, and the condition has an $L$-projection form.
   - Setup: the static 1 mm image of the phantom, reconstructed with $L$ = 1 directly, and with the full $L$ and then averaged. Vary the $\lvert k \rvert$–τ correlation of the library on purpose (for example, tip radii ordered by τ).
   - Decision: whether the artefact level follows the $L$-projection mismatch of experiment 3, and whether reconstructing with the full basis is worth its cost.
7. **Aliasing from the period of the roles.** Tests the risk of section 6.1: a cardiac harmonic near 2 Hz aliasing into the BOLD band.
   - Setup: the phantom with cardiac and respiratory components; roles cycled at 2 Hz, roles with the order jittered between cycles, the greedy schedule, and the shuffle. Score with the k–T spectrum in the band; if affordable, the continuous-T BOLD reconstruction and the false signal in the band.
   - Decision: whether roles need jitter, and what jitter costs in completeness.

8. **Spoiling of a schedule.** Tests section 5.1: does a schedule optimized for coverage spoil as well as incoherent spoiling does?
   - Setup: schedules from the shuffle, greedy coverage, greedy with the spoiling term, and roles; RF phase random and quadratic 117°. (a) Pathway analysis: the distribution of $\lvert M \rvert$ of the low-order pathways, and how often and with what amplitude they fall within the readout ball. (b) Simulation: the analytic phantom (section 7.1) gives the object's spectrum at any $k$, so the spurious signal of each pathway, its amplitude times the phantom signal at $M + k_q(t)$, can be added to a simulated scan exactly; reconstruct and measure the artefact, the steady-state fluctuation over T, and its spectrum in the BOLD band. An isochromat (Bloch) simulation of one voxel checks the pathway truncation.
   - Decision: whether the spoiling term (or constraint) is needed, the spoil moment size, and the RF phase scheme.

**Order.** Experiment 1 first, because 2, 4 and 5 optimize the frame potential. Experiments 2 and 3 are cheap and independent and can run beside 1. Experiment 4 needs only the evaluator; 5–7 need the τ basis and the phantom. Experiment 8(a) needs only the schedules, so it can run with experiment 4; 8(b) needs the phantom and the pathway simulation. If 1–5 leave a gap, alternate shapes and schedule (section 6).

### 7.3 Tiers of cost

A full 1 mm run of 1a takes about 20 min and 13 GB. Most questions do not need it. Ask each question at the cheapest tier that can answer it.

- **Tier 0: exact small problems.** 3D at 16³–24³, or 2D at 32²–64², with $A$ explicit and its SVD: the exact error, noise and model error, in seconds. It tests whether an objective measures the right thing, not whether a design is good.
- **Tier 1: the real designs, reduced.** The static model at 4 mm on the samples in its ball: about 100 s and 3.7 GB per run. Fewer TRs where the question allows. 1a's 4 mm check shows that it reproduces the effects found at 1 mm. A Toeplitz normal operator is a planned speed-up for full runs, not yet measured.
- **Tier 2: full 1 mm runs**, only to confirm a conclusion before it goes into the plan or the objective.

## 8. Decisions

| date | decision |
|---|---|
| 2026-10-15 | The cost axis is slew energy for now; band-filtered slew energy and PNS later (report, section 5). |
| 2026-10-15 | The example requirements of section 1 are the design target of this document. |
| 2026-10-15 | Resolution in T means a smooth temporal basis in a continuous model, not fixed windows. |
| 2026-10-15 | Experiment 1 starts with the static requirement alone (sub-experiment 1a), with the existing CG-SENSE; the other reconstructions come after it. |
| 2026-10-15 | B0 over T: scanner drift and respiration are in scope; head motion is important but waits for a motion model. The model is the static map plus a few modes, found in each subject's data inside a rich spatial basis (spherical harmonics up to order 2–3, later a population basis); the design targets that population span, the reconstruction the subject's modes. Bound the estimate and measure the change on the phantom scans before building it ([b0-scan-time-plan.md](b0-scan-time-plan.md)). |
| 2026-10-16 | From 1a: the static frame potential stays as the sampling term. The objective needs a term for the τ-averaging model error of an averaged model (*hypothesis*: a local effect among samples near each other in k; test at Tier 0 or 1). |
| 2026-10-16 | Experiments run at the cheapest tier that answers them (section 7.3). |

## 9. Conclusion

Not reached. The direction so far: design the schedule greedily against the frame potentials of the requirements' reconstructions, a term for the τ-averaging model error of the averaged models (experiment 1a), and a spoiling term (section 5.1), then design the shapes for the $(\lvert k \rvert, \tau)$ terms and slew energy as $m$ complementary roles that the schedule cycles through, so that every window of $m$ TRs is complete, probably with tens of shapes rather than 128. The experiments of section 7 decide whether that holds.
