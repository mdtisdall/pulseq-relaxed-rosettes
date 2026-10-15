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

Each requirement is a linear model $x(r, \tau, T) = \sum c \, \phi_l(\tau)\, b_j(T)$ on a spatial grid at its resolution, fitted by data consistency (coils, B0) plus regularization: CG-SENSE extended with τ and T bases.

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

By window length: windows of about one TR (navigator) are set by the shapes; windows of a few to tens of TRs (BOLD) by the shapes in radius and τ and by the schedule in direction and T; whole-scan windows (static, relaxation 4D) by the library in radius and τ and by the schedule in angular fill.

**Consequences.**

1. **The cost axis of the front is a shape problem.** The schedule improves coverage at no slew cost, so optimizing it moves designs toward the front for free. This argues for doing the schedule first.
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

The costs grow about linearly in $S$ (optimization, calibration, sequence size), and a jointly designed library of many shapes is a harder optimization, more likely to stop at a poor local optimum. In practice, more can then be worse. So there is a "good enough" $S$ for each design: where the measured gain on the scores of the requirements drops below their measurement noise, or stops being worth the added cost. Experiment 2 of section 7 measures that curve.

**Fitting with neighbours.** Part of the difficulty is that each shape must fit with the shapes it shares windows with. If every shape had to fit with *any* $m - 1$ others, the number of combinations would be $\binom{S}{m}$: polynomial in $S$, but exponential in the window size $m$. Three things keep the demand manageable:

- **Only the windows that occur matter.** The schedule is chosen, not random, so the constraints are on the windows of the actual sequence: about one for each TR, so linear in the number of TRs. The combinatorial difficulty moves into choosing the order, which is what the greedy schedule (section 5) handles.
- **Without structure, the shapes still conflict.** In an arbitrary order, each shape has many different sets of neighbours over the scan, and it must fit with all of them. That pushes each shape toward being complete by itself, which is expensive.
- **Roles remove the conflict.** Give each shape one of $m$ *roles* in $(\lvert k \rvert, \tau)$, design the roles to be complete together, and let the schedule cycle through the roles: role 1, 2, …, $m$, 1, 2, …. Then *any* $m$ consecutive TRs contain every role exactly once, wherever the window starts. Each shape needs to fit only its role: the demand is linear in $m$, and the greedy schedule keeps its freedom in which shape of the role, and in the rotation. Roles can be nested for the shorter windows: for example, $m = 9$ as 3 × 3, so that every 3 consecutive TRs meet the navigator's no-correlation condition, and every 9 meet BOLD's rank-$L$ condition.

The price of roles is a period of $m$ TRs in the $(\lvert k \rvert, \tau)$ sampling: 0.5 s, or 2 Hz, for $m$ = 10. That is above the BOLD band, where section 2 wants the variation, but a physiological signal near the period can alias into the band: for example, a cardiac harmonic near 2 Hz would alias to $\lvert f - 2\ \text{Hz} \rvert$, which can be below 0.1 Hz. Check this in the k–T spectrum; jittering the role order between cycles would trade it against the completeness of every window.

*Hypothesis:* for these requirements the good-enough $S$ is a few complete groups of $m \le 10$ shapes, so tens of shapes rather than 128, provided the library is designed as complementary groups. A random library needs more shapes for the same coverage, which is what the 32 against 128 measurement shows.

## 7. Plan of experiments

1. **Greedy schedule with the current shapes** (section 5). Keep the 128 shapes; candidates are a dense set of SO(3) rotations; choose against the requirement terms. Compare with today's shuffle on:
   - the static term and measure 3 at 25%, 50% and 100% of the scan (early stopping);
   - the k–T spectrum in the BOLD band (periodicity);
   - the navigator's gaps at short windows.

   Memory check on one configuration first.
2. **The $(\lvert k \rvert, \tau)$ distribution of the current library**: measure how well it meets the relaxation 4D and no-correlation conditions in the $L$-dimensional form of section 6.1, as a function of the number of shapes (for example 8, 16, 32, 64, 128 random shapes) and of the number of consecutive TRs, before redesigning shapes.
3. **A complementary library with roles** (sections 6, 6.1): $m$ roles complete together, cycled by the schedule, if step 2 shows a deficit, or shows that random libraries need many more shapes than one complete set. Check the k–T spectrum for aliasing from the period of the roles.
4. **Alternating shapes and schedule**, if 1–3 leave a gap.

## 8. Decisions

| date | decision |
|---|---|
| 2026-10-15 | The cost axis is slew energy for now; band-filtered slew energy and PNS later (report, section 5). |
| 2026-10-15 | The example requirements of section 1 are the design target of this document. |
| 2026-10-15 | Resolution in T means a smooth temporal basis in a continuous model, not fixed windows. |

## 9. Conclusion

Not reached. The direction so far: design the schedule greedily against the frame potentials of the requirements' reconstructions, then design the shapes for the $(\lvert k \rvert, \tau)$ terms and slew energy as $m$ complementary roles that the schedule cycles through, so that every window of $m$ TRs is complete, probably with tens of shapes rather than 128. Section 7 decides whether that holds.
