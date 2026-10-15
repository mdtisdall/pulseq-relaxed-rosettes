# Testing the scan-time basis on a static phantom

Status: plan; nothing here is implemented yet. It uses the minimal real-data scan (design A) that already checks the static reconstruction. Record the measured results in section 7, in the same PR as each step, and correct the text where the plan was wrong.

## Question

The 5D reconstruction is planned as a low-rank model in scan time $T$:

$$x(r, \tau, T) = \sum_{k=1}^{K} u_k(r, \tau)\, v_k(T).$$

If the temporal basis $v_k(T)$ is estimated first and then held fixed, the reconstruction is one linear solve: the Toeplitz kernels for each pair of basis functions are computed once, and there is no outer loop alternating between $u$ and $v$. That is the main saving in reconstruction time (estimated at roughly the number of outer iterations, 3–10×; not measured), and it removes the bilinear step, which is the least stable part of the solve.

Can a good enough $v_k(T)$ come from data that the current trajectory already acquires, or does every shape need a dedicated navigator segment?

## Short answer

Test the two navigators the trajectory already gives, on a static phantom, where the true temporal content is known: a constant signal plus slow system drift. A basis is good if every component above the noise level is explained by drift, and none is correlated with the shape or the rotation of the TR. A simulated modulation of the marker vial checks that the test can also detect real change. Add a navigator segment to the shapes only if both candidates fail.

## 1. Why a single TR is not a navigator by itself

Measured on a 128-shape export (scale-free, so it holds for any resolution): the 95th-percentile gap from a point at radius $|k|$ to the nearest sample of one TR grows about 0.43 cells per cell of radius, as for about 80 straight spokes. One TR is at Nyquist only to $|k| \approx 1$ cell. Beyond that, the per-TR aliasing depends on the rotation, so it changes every TR. An SVD reads such changes as temporal signal. A navigator must therefore either use only what one TR samples completely (around $k = 0$), or average enough TRs that the rotation no longer matters.

## 2. Data

- **Scan:** design A (4 mm, TR 40 ms, 10000 ADC samples in 2 segments, 1500 TRs, about 60 s), the phantom with the marker vial, as already planned for the static reconstruction check. The noise scan gives the noise covariance for the noise floor in section 4.
- **Longer repeat (optional):** the same design with about 5000 TRs (about 200 s). Drift from gradient heating grows over minutes. A longer scan shows whether drift needs more than one or two components.
- **Design settings to record:** `crossing_tolerance_cells` and `tip_width`. With a crossing tolerance above 0, the crossings pass up to that distance from $k = 0$, in a direction that rotates with the TR. That is a rotation-dependent offset by design, and candidate A must then be judged with it in mind.

## 3. Candidates

### A. Samples near k = 0 (free)

$|R k| = |k|$, so the samples within a radius $\varepsilon$ of $k = 0$ are known per shape, independent of the rotation. Use $\varepsilon$ = 0.5 cell (try 0.25 and 1). Each TR has about 40 such passes, one per petal, at known $\tau$.

The $\tau$ of the crossings differs between shapes, because the petal timing differs. Rows of the navigator matrix must compare the same $\tau$ across TRs, so for each TR and whitened channel, interpolate the crossing samples onto a common grid of $\tau$ (for example 16 points over the range shared by all shapes; linear in magnitude and unwrapped phase). The navigator matrix has one column per TR and (channels × $\tau$ points) rows. Use the whitened channels before coil compression: more rows, and no dependence on the compression.

### B. Low-resolution pre-reconstruction

Grid non-overlapping windows of $W$ TRs at 8 mm ($n$ = 32) with the existing gridding (Pipe–Menon DCF), coil-combined with the time-averaged maps, using one early $\tau$ window (for example the first 5 ms) so that decay and B0 phase change little. Try $W$ = 10, 25 and 50 TRs (0.4, 1 and 2 s). The navigator matrix has one column per window and one row per voxel inside a mask of the object. Repeat with CG-SENSE for $W$ = 25 if gridding fails, since SENSE suppresses part of the aliasing.

### C. Navigator segment (only if A and B fail)

The same short trajectory, at the same $\tau$, in every shape, sampling a small ball around $k = 0$ completely. Its size and its cost in imaging time are a design question for the shape optimizer and `coverage`; it needs a new scan, so it is a second phase of this test.

## 4. Metrics

For each candidate's navigator matrix:

1. **Rank above noise.** Singular values after whitening, compared with the largest singular value of a pure noise matrix of the same size (the Marchenko–Pastur edge, about $\sigma(\sqrt m + \sqrt n)$; check it by applying the same pipeline to the noise scan, or to data with the TR order shuffled within each shape). The count of components above the edge is the measured rank.
2. **Attribution.** Regress each component above the edge, $v_k(T)$, on:
   - drift: a polynomial in $T$ up to degree 3;
   - shape: one indicator per shape (an ANOVA over the shape index);
   - rotation: the 9 entries of the rotation matrix, and their products up to degree 2;
   - for candidate A, the mean $\tau$ of the crossings used.

   Report the fraction of each component's variance explained by each group. Drift is real and stays in the basis; shape, rotation and $\tau$ terms are contamination.
3. **Detection (positive control, by simulation).** Take the time-averaged CG-SENSE image, coil maps and field map of the phantom. Multiply the marker vial by $1 + a\,m(T)$, with $m$ a 0.05 Hz sinusoid plus a step at mid-scan. Generate data with the forward operator (`SenseOperator`) at the measured noise level, and run each candidate on it. Report the smallest $a$ (try 0.5%, 1%, 2%, 5%) at which the recovered basis spans $m$ (the correlation between $m$ and its projection onto the basis above 0.9).

## 5. Decision rule

Set the thresholds before looking at the data:

- A candidate **passes** if, in every component above the noise edge, shape, rotation and $\tau$ terms together explain less than 10% of the variance, and the simulated modulation is detected at $a \le 2\%$.
- **A passes:** take the basis from the crossings. No navigator; no pre-reconstruction.
- **Only B passes:** take the basis from the pre-reconstruction, at the smallest $W$ that passes. This sets the finest temporal resolution of the basis.
- **Neither passes:** design candidate C, measure its cost with `coverage`, and repeat the test on a new design A scan with it.

If contamination is found, its attribution says where it comes from. Rotation terms in candidate A point at gradient delays or eddy currents, which move the real crossings away from $k = 0$ by a fraction of a cell in a direction that rotates with the TR. They would also corrupt the trajectory of the static reconstruction, so they are worth knowing about either way.

## 6. Implementation

Small steps, each with tests on simulated data:

1. **Crossing navigator:** a function in `relaxed_rosettes.recon` that selects the samples within $\varepsilon$ of $k = 0$ for each shape, reads them for every TR from the ISMRMRD data, whitens them, and interpolates them onto the common $\tau$ grid.
2. **Window navigator:** windows of TRs for `Selection`, and gridding of each window, reusing `gridding` and the time-averaged coil maps.
3. **Analysis:** SVD, noise edge, regressions and the simulated modulation, written to an `.npz` file (no plotting in this package), through a CLI command (for example `relaxed-rosettes basis-test`).

The phantom data and its results stay out of the repository; the code and this plan go through PRs as usual.

## 7. Results

Not run yet.
