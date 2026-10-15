# Coverage requirements: what an application specifies

Status: design; nothing here is implemented yet. It follows [sample-spreading-report.md](sample-spreading-report.md), sections 5–6, which restate the design problem as an efficient front between slew energy and coverage. Those sections need a target curve $W^*(\text{res})$: how long a stretch of scan time may be before its samples must cover the k-space ball of each resolution. This note generalizes that curve to requirements that resolve some dimensions and average over others, says what the application picks and what the reconstruction settings are, and how the requirements trace the front.

## 1. The dimensions and the four example requirements

The data have five dimensions: three of space (k-space, up to a resolution), the **time since excitation τ** within each TR (the 4th: the relaxation axis, across the ADC window of about 45 ms), and **scan time T** (the 5th: the time of the TR within the scan, 0 to about 250 s by default; motion and physiology).

A requirement says, for each dimension, one of three things:

- **fixed**: a target resolution in that dimension;
- **averaged**: integrated over the whole range;
- **as fine as possible**: maximized, with a priority, at the cost of the other dimensions.

The four example requirements:

| requirement | space | τ (4th) | T (5th) |
|---|---|---|---|
| navigator | fixed, coarse (about 8 mm) | averaged | fixed, the shortest window |
| relaxation 4D | fixed, 1.5 mm | resolved | averaged |
| static | fixed, 1 mm | averaged | averaged |
| BOLD | as fine as possible | as fine as possible | fixed, 0.5 s |

A requirement whose dimensions are all fixed or averaged is a **constraint**: an anchor of a curve (section 3). A requirement with a dimension "as fine as possible" is a **priority**: it adds no anchor, but says which part of the curves matters and how to spend the samples there (section 4).

**Resolution in T is a kernel width, not a window.** A reconstruction that resolves T need not make separate images from fixed windows. The better model is a continuous volume in T, with each sample entering where it falls, weighted by a smooth temporal kernel or basis of the required width (0.5 s for BOLD). The windows of measure 3 in the report are a proxy for that kernel width.

## 2. One curve costs slew in its level, not its shape

For requirements that average over τ, the curve $W^*(\text{res})$ is as in the report: the scan-time width within which the samples must cover the ball of radius $1/(2\,\text{res})$ to a target gap.

The requirements at different resolutions are coupled. Let $D(k)$ be the volumetric density of the samples, normalized to integrate to 1 over the ball of $k_{max}$, and $r$ the rate at which the scan places samples, in Nyquist cells per second. A cell at radius $k$ then receives samples at rate $r\,D(k)$. A ball is covered when its least-sampled cells are, and the achievable densities do not increase with radius ([sampling-density.md](sampling-density.md), section 1), so the least-sampled cells are at the edge:

$$
W(\text{res}) \;\approx\; \frac{c}{r\, D\big(1/(2\,\text{res})\big)} ,
\qquad (1)
$$

with $c$ set by the target gap. So:

- **The shape of $W^*$ fixes the radial density.** For example, $W^* \propto \text{res}^{-a}$ gives $D \propto k^{-a}$: the `exponent` model, with $a$ = `resolution_time_exponent`.
- **The level of $W^*$ fixes the rate** (section 3, equation 2). The rate is set by the path speed, and so by $|g|$ and the slew energy. A curve is not wrong if it is too fast; it is expensive, or infeasible within the hardware limits.
- **$W^*$ must not get shorter at finer resolutions.** A finer ball contains a coarser one.

Equation (1) assumes a density without angular holes, so the measured curve (measure 3) stays the check. The report's measurement on the default design (about 31 s at 1 mm, 5 s at 2 mm, 1 s at 4 mm) is close to $\text{res}^{-2.5}$, the default exponent.

## 3. Constraints: one curve for each set of resolved dimensions

**Resolving τ costs samples.** The reconstruction represents τ with a subspace basis of rank $L$. A requirement that resolves τ needs every neighbourhood of k-space at its resolution sampled at τ values spread well enough to fit $L$ coefficients. A practical form: split τ into $L$ groups of equal basis energy, and require coverage in each group. That costs about $L$ times the samples of the τ-averaged requirement at the same resolution and width in T.

So the constraints form two curves:

- **$W^*_{\text{avg}}(\text{res})$, τ averaged**: anchored by the navigator (coarse, shortest window) and the static image (1 mm, whole scan).
- **$W^*_{\tau}(\text{res}; L)$, τ resolved**: anchored by the relaxation 4D image (1.5 mm, whole scan). With one anchor, its shape comes from a preset (section 6).

One density must serve both curves. At each radius it must meet the larger of the two demands, so the rate that the requirements need is about

$$
r \;\approx\; c \int_0^{k_{max}} 4\pi k^2 \max\!\left( \frac{1}{W^*_{\text{avg}}(k)},\; \frac{L}{W^*_{\tau}(k)} \right) dk ,
\qquad (2)
$$

with $W^*(k)$ the target at the resolution whose edge is $k$. The τ-resolved curve adds one more condition that (2) does not show: within each width in T, the τ values at each radius must be spread over the $L$ groups, not only numerous.

**Averaging is not free.** An averaged image is clean only if the samples' distribution in the averaged dimension is about the same at every $|k|$. If outer k-space is sampled mostly at late τ, the contrast varies across k-space, which rings or blurs; the same holds for T and motion. So every averaging requirement carries a weak condition: no correlation between $|k|$ and τ (or T) at its resolution.

## 4. Priorities: BOLD

BOLD fixes the width in T (0.5 s) and asks for as much as possible in space and τ. The samples within 0.5 s (about 10 TRs) can be spent on spatial cells or on τ groups, roughly as $\text{res}^{-3} \times L \lesssim$ (samples within the width) $/ c$. The two curves of section 3, read at $T$ = 0.5 s, give the outcome: $W^*_{\text{avg}}$ the resolution at $L$ = 1, $W^*_{\tau}$ the resolution at larger $L$. So BOLD adds no anchor. It adds:

- **a priority at $T$ = 0.5 s**, spread over resolutions (as the prior $\pi(r)$ of the `filter` model spreads it over resolutions for the whole scan);
- **a preference for small $L$**: BOLD needs a few τ values (2–4) to separate T2\* from $S_0$, so within 0.5 s it favours coarse cells each sampled at a few τ values over fine cells at one τ.

**Variation of the sampling over T.** Because the BOLD signal changes by about 1%, the concern is aliasing that varies with T. In the continuous model of section 1 there are no window boundaries, so the question is not whether every window looks alike. Variation of the sampling pattern over T puts aliasing at the temporal frequencies of that variation:

- coherent structure in the BOLD band (about 0.01–0.1 Hz) or near the respiratory and cardiac frequencies adds a false signal or aliases a confound into the band;
- incoherent, broadband variation raises the noise floor across all frequencies, the band included;
- variation pushed above about 1/(0.5 s) (blue noise in T) falls mostly outside the band the model resolves. This is the target.

Today the TR order is shuffled at random (`relaxed_rosettes.rotations`): no periodicity, so no coherent aliasing, but white variation, so the band gets its share. The lever is the cross-TR order (plan, step 3).

## 5. What the application picks, and what we pick

**The application** picks the requirements of section 1: for each, the state of each dimension, the fixed widths, and the priorities of best-effort dimensions. The values come from the application owner. "Resolution" in space means the k-space cutoff, $1/(2\,\text{res})$; a tapered filter (Hann) has a coarser effective resolution, so either the documentation states this convention or a later version converts an effective resolution into a cutoff.

**The reconstruction settings** have defaults; an expert can change them.

| setting | what it does | default |
|---|---|---|
| kernel shape | the k-space filter of each reconstruction ($\kappa_{\text{res}}$ in measure 8, $h_{\text{res}}$ in the kernels $K_T$) | Hann. The curves set bandwidths, not profiles, so the shape changes the design little |
| temporal kernel | the smooth weight in T of each requirement that resolves T | to be chosen; smooth, not a box (section 4) |
| τ basis rank $L$ | the τ resolution of τ-resolved requirements | from the reconstruction's τ basis |
| coverage threshold | the target gap (cells) and its percentile, which define "covered" in measure 3 and the constant $c$ in (1) | 95th percentile; target gap to be chosen |
| acceleration $R(\text{res})$ | how much the coils and the subspace model lower the density needed: about $W^*(\text{res})/R(\text{res})$ of Nyquist coverage | 1 (Nyquist). Measure it later, for example with the phantom test of [basis-quality-test-plan.md](basis-quality-test-plan.md) |

The curves stay in Nyquist cells and $R$ is applied separately, so a requirement keeps its meaning across coils and reconstructions.

## 6. Interpolation, validation and presets

**Interpolation.** Along each curve, monotone cubic (PCHIP) in $\log \text{res}$, $\log W$. With two anchors this is a power law. Outside the anchors, extend with the end slopes, and clamp to [one TR, the scan duration]. A curve with one anchor takes its shape from a preset.

**Validation** (with the other checks of `Config`):

- resolutions positive, and none finer than `geometry.resolution_m`;
- on each curve, widths in T non-increasing as the resolution gets coarser (section 2);
- widths in T at least one TR, `readout.tr_s`, and at most the scan duration, `num_trs` × `readout.tr_s`;
- every requirement has one state for each dimension; a "fixed" state has a width; priorities are positive;
- a warning, not an error, when the rate that (2) needs is far beyond the rate of the current design.

**Presets.** The existing density models become generators of curve shapes, not alternatives to the requirements:

- **`exponent`**: $W^* \propto \text{res}^{-a}$ over [`density_resolution_min_m`, `density_resolution_max_m`];
- **`filter`**: the shape from (1) with $D \propto \sqrt{\sum_r \pi(r)\, w_r^2}$, the SNR-optimal density for Hann reconstructions weighted by $\pi(r)$.

A preset gives a shape; one anchor of the curve sets its level.

## 7. Use: tracing the front for one application

The level of the requirements is the parameter that traces the front. For a scale $s > 0$, every fixed width in T is multiplied by $s$, and the requirement is that each curve is met at $s$ times its target (an ε-constraint):

- **Definition.** The front for the application is the cheapest slew energy at which a design meets the scaled requirements, as a function of $s$. $s < 1$ is faster than asked; $s > 1$ is a relaxation. "Meeting your requirements costs X; relaxing them by 1.5 costs Y" is how the result is reported.
- **Optimization.** Measure 3 is not smooth, so it cannot be the optimizer's constraint directly. For each $s$, the optimizer minimizes slew energy + μ × measure 8, built from the requirements:
  - each requirement gives terms with a smooth kernel in T of width $s\,W^*(\text{res})/R(\text{res})$ (constant if T is averaged), a kernel in τ (constant if τ is averaged, $L$ groups wide if resolved), and a kernel in k at its resolution (spread over resolutions with the priority if space is best effort);
  - the priorities weight the terms.

  A smooth kernel in T that penalizes samples close in both k and T also pushes the sampling toward blue noise in (k, T), the target of section 4. A sweep over μ gives candidate designs; the scores below decide which meet the requirements; the cheapest of those is the point of the front at $s$.
- **Scores.** Measure 3 (gap within windows) for each constraint, within each τ group if τ is resolved. Measure 4 ($(|k|, \tau)$ empty cells) is a radial-only diagnostic of the τ-resolved score. For requirements that resolve T, a k–T spectral score: for each k-region at the requirement's resolution, the power spectrum of its sample occupancy over T, integrated over the band of interest (the k–T point-spread function).
- **Feasibility.** Equation (2) estimates, before any optimization, the rate that a value of $s$ needs, and so whether it is in reach.
- **Reconstruction.** The kernels of each reconstruction follow from the same requirements: for example $K_T(k, t) = w_T(t)\, h_{\text{res}(T)}(k)$ along the τ-averaged curve, with $\text{res}(T)$ the inverse of $s\, W^*_{\text{avg}}$.

A single scale keeps the balance between requirements that the application asked for. When that balance is too expensive, the priorities say which requirements to relax first. Doing this automatically (a scale for each requirement) is left for later.

## 8. Proposed configuration

Requirements as an array of tables of `Config`, in SI units with the unit in the field name. The widths and priorities below are placeholders.

```toml
[[coverage.requirement]]
name = "navigator"
resolution_m = 8e-3
tau = "averaged"
scan_time_s = 0.15                  # placeholder: the shortest window

[[coverage.requirement]]
name = "relaxation_4d"
resolution_m = 1.5e-3
tau = "resolved"                    # rank L from coverage.tau_basis_rank
scan_time_s = "averaged"

[[coverage.requirement]]
name = "static"
resolution_m = 1e-3
tau = "averaged"
scan_time_s = "averaged"

[[coverage.requirement]]
name = "bold"
resolution_m = "best"
tau = "best"
scan_time_s = 0.5
priority = 1.0

[coverage]
tau_basis_rank = 4                  # placeholder: from the reconstruction's basis
gap_cells = 1.0                     # placeholder: target gap still to be chosen
gap_percentile = 95.0
kernel = "hann"
acceleration = 1.0
```

A field is a number (fixed), `"averaged"`, or `"best"`. A whole-scan width is `"averaged"`, not the scan duration, so the requirement keeps its meaning when the scan length changes.

## 9. Open questions

- The fixed widths: the navigator's shortest window, and whether 0.5 s is the BOLD width.
- The τ basis rank $L$ for the relaxation 4D requirement, and for BOLD.
- The temporal kernel shape, and the band of the k–T spectral score for BOLD.
- The target gap that defines "covered", and whether the 95th percentile is the right tail.
- How to measure $R(\text{res})$, and whether it can be predicted from the coil geometry.
- Whether the front needs a separate scale for each requirement, or whether one scale and the priorities are enough.
