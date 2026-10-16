# Collapsing a dimension with a quadratic norm

Status: analysis, with a 1D toy that checks its main claims (section 7, [`toy_x_t.py`](quadratic-norm-report/toy_x_t.py)). Nothing is implemented in the reconstruction. Started 2026-10-16.

## Question

A requirement can leave a dimension out of the output: the static image averages over τ and $T$ ([coverage-target.md](coverage-target.md), section 1). An average is linear, so it is one linear reconstruction. But often the quantity of interest is not the mean over a dimension but its size: the power of the fluctuations over $T$ in a band, at each voxel, with no interest in their time course. The usual route is a model of the dimension (a low-rank or subspace model in $T$), a reconstruction of the time courses, and then the norm. That route imposes the model's structure on the result (for a low-rank model, shared time courses; [b0-scan-time-plan.md](b0-scan-time-plan.md), section 3).

Can a dimension of no interest be collapsed with a norm instead, without a model of it? For which norms does the reconstruction separate and become simpler?

## Short answer

Yes, for quadratic norms: any $q(r) = \sum_{t,t'} w(t,t')\, x_t(r)^*\, x_{t'}(r)$ with $W$ positive semi-definite. That covers the energy, the variance, and the power in a band over $T$ (the quantity behind ALFF), and, across two voxels, the cross-power and coherence of functional connectivity.

1. **The reconstruction separates.** In the eigenbasis of $W$, $q$ is a weighted sum of squared components (Parseval). When the sampling of each TR is drawn independently of $T$, every component is a static reconstruction of the data weighted by its basis function, with the same averaged normal operator: one Toeplitz kernel set, solved as a batch. For the power in a band, the basis is the Slepian sequences of multitaper spectral estimation, and the number of components is about twice the bandwidth times the scan time: about 45 for the BOLD band in 250 s.
2. **Errors that average out of a mean add up in a norm.** Aliasing and noise that cancel in the static image put a positive bias on $q$. With patterns independent across TRs, the aliasing of a static object is white over temporal frequency, so a band of $J$ of the $N$ temporal frequencies gets the fraction $J/N$ of it. The static image must be subtracted from the data first, *resolved in τ*: its aliasing otherwise dominates (in the toy, by a factor of 260), and so does the aliasing of its decay within each readout, which repeats in every TR at different k (a factor of 11 in the toy with readout timing).
3. **The bias can be removed without a model.** Leave out the products of each TR with itself (a U-statistic): with independent patterns, the products of different TRs are unbiased for both noise and aliasing. What remains is the diagonal of the quadratic form, $-\sum_t w(t,t)\,|x_t|^2$: for a band, $J/N$ times the residual energy at that voxel, under 1% for BOLD. The correction is the complete-basis energy map, so both come from one pass.
4. **The assumption is smoothness in T, not a rank.** A product of two samples estimates $|x|^2$ whenever both see the same $x$: within the coherence time of $x$, not only within one TR. Physiology changes slowly against a 50 ms TR, so once the static image is subtracted, the energy and the variance of the residual are band powers over $[-B_{max}, B_{max}]$ too, and the U-statistic applies to them, with an exact scale correction $1/(1 - J/N)$. Only content above $B_{max}$ needs products within one TR, where the bias is a convolution with the mean squared point-spread function of one TR, to be deconvolved as in speckle interferometry and in CMB power spectra from a masked sky. Its conditioning is set by the differences that one TR's samples cover, not by their number.
5. **The figure of merit** is the leakage per component, $\rho \approx M/S - 1$ for $M$ cells and $S$ effective samples of one TR inside the ball, times $J/N \approx 2 B\, T_R$. The number of samples in one TR enters only here, as variance: never as a limit on what can be estimated. The bias does not depend on the scan length; the variance falls with it. Band power maps look practical at 2–4 mm without coil acceleration, and need coils and spatial averaging at 1 mm (section 5).
6. **The schedule decides where the leakage goes.** Independent rotations make it white. A periodic schedule puts all of it at multiples of $1/(\text{period})$, as UNFOLD does on purpose: zero in the band if the harmonics avoid it, concentrated if one falls in it. The k–T spectral score of [example-requirements-design.md](example-requirements-design.md) (section 2) is exactly this leakage, so it is the right objective for such a requirement.

## 1. The problem

**Data.** TR $t = 1, \dots, N$ acquires $y_t = A_t\, x_t + n_t$, with $x_t(r)$ the image at the time of TR $t$ and $A_t$ the encoding of that TR (its samples, the coil maps, the NUFFT, the field map). Each $A_t$ samples far less than the image needs: one TR has 14 000 samples, the 1 mm ball 8.8 million cells.

**A TR is not an instant.** Sample $m$ of TR $p$ is at absolute time $t_m = T_p + \tau_m$: the readout spreads each TR over about 45 ms of the 50 ms TR. In the plane of τ and absolute time, each TR is a diagonal segment $t - \tau = T_p$; in k and $t$, it is a stroke along which k changes quickly. The signal depends on both: on τ through relaxation and the static field, on $t$ through physiology. Sections 2–3.4 treat $x_t$ as constant over one readout. That holds for the slow change over $t$ (section 3.5 gives the conditions) but not for the change over τ, which section 3.1 must remove first.

**Target.** A quadratic form over $T$ at each voxel,

$$q_W(r) = \sum_{t, t'} w(t, t')\; x_t(r)^*\, x_{t'}(r) = \mathbf{x}(r)^H W\, \mathbf{x}(r), \qquad W \succeq 0,$$

with $\mathbf{x}(r)$ the time course at voxel $r$. Examples:

| quantity | $W$ | use |
|---|---|---|
| energy | $I$ | root-sum-of-squares over $T$ |
| variance | $I - \mathbf{1}\mathbf{1}^T / N$ | temporal standard deviation, tSNR |
| power in a band | the projector onto the band | ALFF, the mean amplitude of the spectrum in 0.01–0.08 Hz (Zang et al., 2007), which the band power approximates as its square; fALFF, its ratio to the whole spectrum (Zou et al., 2008) |
| cross-power between voxels $r, r'$ | the band projector, with $\mathbf{x}(r)^H W\, \mathbf{x}(r')$ | coherence, functional connectivity (Sun et al., 2004) |

A form with $W$ not positive semi-definite (a difference of bands) is a difference of two of these. Other modalities already image such quantities instead of time courses: functional ultrasound maps the power of the blood signal over slow time in each pixel (Macé et al., 2011), and SOFI maps temporal cumulants of fluorescence fluctuations (Dertinger et al., 2009). Both have every frame fully sampled; the problem here is that each TR is not.

**It is linear in the lifted variable.** $q_W(r) = \operatorname{tr}(W\, \mathbf{x}(r)\, \mathbf{x}(r)^H)$, so it is a linear functional of the covariance-like matrix $\mathbf{x}\mathbf{x}^H$. Estimating linear functionals of a second-moment matrix from compressed linear measurements is the problem of compressive covariance sensing (Romero et al., 2016; Ariananda and Leus, 2012) and of covariance estimation from quadratic samples (Chen, Chi and Goldsmith, 2015); lifting quadratic measurements to a linear problem in $\mathbf{x}\mathbf{x}^H$ is also the idea of PhaseLift (Candès et al., 2013). Two things make the MRI case easier than those: the encoding of each TR is known and differs between TRs, and the target is only a few functionals per voxel, not the whole matrix.

## 2. Separation in the eigenbasis of $W$

Write $W = \sum_j \lambda_j\, \psi_j \psi_j^H$ with orthonormal $\psi_j(t)$. Then

$$q_W(r) = \sum_j \lambda_j\, |c_j(r)|^2, \qquad c_j(r) = \sum_t \psi_j(t)^*\, x_t(r).$$

**The components decouple when the sampling does not depend on $T$.** A joint least-squares fit of the $c_j$ has the normal operator with blocks $\sum_t \psi_j(t)^*\, \psi_{j'}(t)\, A_t^H D A_t$, with $D$ the density weighting. If the pattern of each TR is drawn independently of $t$ (random rotations, the shuffled schedule), $A_t^H D A_t$ fluctuates around its mean $\bar A / N$ independently of $\psi$, and the blocks are $\delta_{jj'}\, \bar A / N$ plus a fluctuation. So, to that approximation,

$$\hat c_j = N \bar A^{-1} \sum_t \psi_j(t)^*\, A_t^H D\, y_t ,$$

a static reconstruction (the operator $\bar A$ that `relaxed-rosettes sense` inverts now) of the data weighted by $\psi_j(t)$. All components share one normal operator, so one set of Toeplitz kernels (Fessler et al., 2005) and one preconditioner, and they can be solved together as a batch of right-hand sides (block CG; O'Leary, 1980). The fluctuation that this drops is not an error of the method but its leakage (section 3.1).

**For a band, the basis is the Slepian sequences.** The eigenvectors of the band projector restricted to $N$ samples are the discrete prolate spheroidal sequences (Slepian, 1978); about $2 B\, N T_R$ of them have eigenvalues near 1, for a band of width $B$ (both signs of frequency). The estimate $\sum_j |\hat c_j|^2$ is then the multitaper estimate of the band power (Thomson, 1982; Percival and Walden, 1993), computed from the k-space data instead of from time courses, and it inherits multitaper's control of out-of-band leakage over the finite scan. The quadratic-estimator view of spectrum estimation (Mullis and Scharf, 1991) is the same: every such estimator is $\mathbf{x}^H Q\, \mathbf{x}$ for some $Q$, and its eigenvectors are the tapers.

**Counts at the defaults** (TR 50 ms, 5000 TRs, 250 s): the BOLD band 0.01–0.1 Hz has $J \approx 2 \times 0.09 \times 250 = 45$ components. The band power is then about 45 static reconstructions with one shared operator, with no rank chosen from the data and no structure in $T$ imposed: the basis is set by the question, not fitted.

**The limit of a complete basis** ($W = I$, the energy) needs no basis at all: by Parseval again, $\sum_{j=1}^{N} |\hat c_j|^2 = \sum_t |z_t|^2$ with $z_t = N \bar A^{-1} A_t^H D\, y_t$, the per-TR image. So the energy is one pass over the TRs: grid each TR, apply one fixed filter, add the squared magnitude into a single 3D array.

## 3. Bias, and how to remove it without a model

The per-TR estimate $z_t = R_t\, y_t$ (with $R_t = N \bar A^{-1} A_t^H D$) is unbiased over the random pattern, $\mathbb E[R_t A_t] \approx I$, but each one is badly aliased. A mean over $T$ lets that aliasing cancel; a squared norm does not.

### 3.1 Leakage: aliasing over temporal frequency

Write $R_t A_t = I + \Delta_t$. The error of a component is $e_j = \sum_t \psi_j(t)^*\, \Delta_t\, x_t$. With the $\Delta_t$ independent across TRs and zero-mean,

$$\mathbb E\, |e_j(r)|^2 = \sum_t |\psi_j(t)|^2\; \mathbb E\, |(\Delta_t\, x_t)(r)|^2 .$$

For a Fourier or Slepian basis $|\psi_j(t)|^2 \approx 1/N$, so every component gets the same share: the average aliasing energy of one TR, divided by $N$. **Independent patterns make the aliasing white over temporal frequency.** A band of $J$ components gets $J/N$ of it, and a narrow band gains $N/J$ over a single TR. This is the temporal side of the incoherence argument of compressed sensing (Lustig et al., 2007): random undersampling turns aliasing into noise-like interference, here spread over temporal frequency.

For random, uniform sampling of $S$ of $M$ cells, the aliasing energy of one TR is $\rho\, \lVert x_t \rVert^2$ with $\rho = M/S - 1$, spread evenly over space. The toy confirms the predicted bias to within its noise (section 7, table 1).

**The static image must be subtracted first, resolved in τ.** The leakage is driven by the whole image, and the static image is about 100 times the BOLD fluctuation. Its aliasing changes every TR, so it lands in the band like everything else. Subtract its prediction, $y_t - A_t\, \hat{\bar x}$, from the data before anything else. $\hat{\bar x}$ is the static reconstruction from all TRs, determined far better than any component. In the toy, without the subtraction, the band power estimate is 260 times the truth (table 2). The static image must also carry its τ-dependence. Within each readout the signal decays (by 60% over 45 ms at R2\* = 20/s), and that decay is the same in every TR but lands at different k in each, so after a τ-averaged subtraction it remains in every TR's residual and aliases into every temporal frequency like the static image itself. In the toy with readout timing (table 4), subtracting only the τ-averaged static image leaves a band power 11 times the truth and a U-statistic whose standard deviation exceeds the signal; subtracting a static model resolved in τ (a quadratic in τ) brings both back to within 3% of the truth. In the 3D reconstruction that model is the relaxation 4D image, the static image with a rank-$L$ τ basis ([example-requirements-design.md](example-requirements-design.md), section 3.1), at the resolution of the collapsed quantity. The same holds for other large, known components: drift, and the field modes of [b0-scan-time-plan.md](b0-scan-time-plan.md). This is what the SVD clutter filter of functional ultrasound does before power Doppler (Demené et al., 2015): it removes the tissue signal before the power of the blood signal is measured (Macé et al., 2011).

**Deterministic schedules move the leakage instead of whitening it.** If the patterns repeat with a period of $m$ TRs, $\Delta_t$ is periodic, and all the leakage lands on the harmonics of $1/(m T_R)$ (table 3). UNFOLD (Madore et al., 1999) uses exactly this: it shifts the aliasing to the edge of the temporal spectrum and filters it out. k-t BLAST and k-t SENSE (Tsao et al., 2003) generalize it to lattice patterns in k–t. For a band power, a periodic schedule whose harmonics avoid the band has no leakage in it at all. One whose harmonic falls in the band concentrates the leakage there. This is the risk that experiment 7 of [example-requirements-design.md](example-requirements-design.md) names for the roles of the shapes.

### 3.2 Noise

$\mathbb E\, |\hat c_j|^2$ also includes the noise energy of the component, $\sum_t |\psi_j(t)|^2\, \mathbb E |R_t n_t|^2$. It is the noise bias of any squared magnitude, as in root-sum-of-squares coil combination (Roemer et al., 1990; Constantinides et al., 1997) and the Rician bias of magnitude images (Gudbjartsson and Patz, 1995). It can be computed from the noise covariance (Kellman and McVeigh, 2005), for example with pseudo-replicas (Robson et al., 2008) or a stochastic trace estimate (Hutchinson, 1990), and subtracted. Section 3.3 removes it without that.

### 3.3 The U-statistic: leave out each TR's product with itself

Expand the estimate:

$$\hat q_W = \sum_{t,t'} w(t,t')\, z_t^*\, z_{t'} = \underbrace{\sum_{t \ne t'} w(t,t')\, z_t^*\, z_{t'}}_{\hat q_U} + \sum_t w(t,t)\, |z_t|^2 .$$

For $t \ne t'$, $z_t$ and $z_{t'}$ come from independent patterns and independent noise, so $\mathbb E[z_t^*\, z_{t'}] = x_t^*\, x_{t'}$: **$\hat q_U$ has no noise bias and no leakage bias.** Leaving out the diagonal is the construction of a U-statistic (Hoeffding, 1948). It is also the reason that cosmology estimates power spectra from cross-spectra between detectors or data splits whose noise is uncorrelated (Hinshaw et al., 2003; Polenta et al., 2005; Tristram et al., 2005), and that MRI measures SNR from the difference of two acquisitions rather than from one image (Reeder et al., 2005; Dietrich et al., 2007). And it is how DICS estimates source power and coherence in a band from the cross-spectral density of MEG data, without time courses (Gross et al., 2001). In MRI, activation statistics have been written directly in terms of k-space data (Rowe et al., 2009), but for fully sampled frames. We have not found a precedent for estimating band power or coherence maps from undersampled k-space with cross-products of independently sampled TRs. For a Fourier or Slepian basis $w(t,t) = J/N$, so

$$\hat q_U = \hat q_W - \frac{J}{N} \sum_t |z_t|^2 :$$

the correction is the complete-basis energy of section 2, from the same pass.

**What remains** is the diagonal of the true form, $\mathbb E\, \hat q_U = q_W - \sum_t w(t,t)\, |x_t|^2$. For a band that is $J/N$ times the energy of the residual (after the subtraction of section 3.1) at that voxel. With $J/N \approx 2 B\, T_R = 0.009$ for BOLD, it is under 1% of the residual energy. In the toy it is −1.8% where all the residual is in the band, and −9% where an out-of-band oscillation four times larger shares the voxel (tables 1 and 2). When all of the residual is in the band, $\sum_t w(t,t)\, |x_t|^2 = (J/N)\, q_W$ exactly, so $\hat q_U / (1 - J/N)$ is unbiased (table 1: the bias is $-J/N$ = −1.75%). Otherwise it can be corrected with an estimate of the residual energy, or made smaller by subtracting the large out-of-band components (respiration, the field modes) too.

**Cross-power between voxels works the same way.** $\hat q_U(r, r') = \sum_{t \ne t'} w(t,t')\, z_t(r)^*\, z_{t'}(r')$. Leaving out the same-TR products removes the correlation that one TR's aliasing puts between $r$ and $r'$, which is exactly the false connectivity that undersampling can create. So band-limited coherence maps can be estimated without a model in $T$ and without imposing shared time courses. The diagonal bias that remains comes from components that $r$ and $r'$ really share (in the toy, the global oscillation; table 2).

### 3.4 Energy, variance, and the coherence time of $x$

The U-statistic leaves out the products of each TR with itself. For the energy ($W = I$) that is everything, and $\hat q_U = 0$. But that is an artefact of treating every TR as a separate time. A product $z_t^*\, z_{t'}$ estimates $|x|^2$ whenever $x_t \approx x_{t'}$: within the coherence time of $x$, about $1/B_{max}$ if its spectrum over $T$ ends at $B_{max}$. Physiology changes slowly against a 50 ms TR: the BOLD band ends near 0.1 Hz, respiration near 0.5 Hz, the cardiac harmonics near 3 Hz. So, once the static image is subtracted:

- **The energy of a band-limited residual is a band power.** $\sum_t |x_t|^2 = q_{W_B}$ with $W_B$ the projector onto $[-B_{max}, B_{max}]$, and $J/N = 2 B_{max} T_R$ (0.3 for 3 Hz). The U-statistic applies, with the exact scale correction of section 3.3, $\hat q_U / (1 - J/N)$. The products it uses are those of pairs of different TRs within the coherence time, whose patterns are independent. The variance is the residual energy minus the mean, the same way.
- **The samples within one TR are also within the coherence time.** They span at most 45 ms, so their products are valid estimates of $|x|^2$ too. They are left out of $\hat q_U$ not because they are at the wrong time but because they share one pattern, so their aliasing is correlated. Leaving them out costs only the fraction $J/N$ of the pairs.

The assumption is that $x$ is band-limited in $T$: a smoothness assumption, set by physiology, not a rank fitted to the data.

**Content above $B_{max}$**, or an energy without the assumption, needs the products within one TR, whose aliasing is correlated. Their bias is still known from the trajectory:

- **Deconvolution.** Under the same random-phase assumption, the per-TR energy is the true energy convolved with the mean squared point-spread function of one TR, plus the noise floor: $\mathbb E \sum_t |z_t|^2 = K \circledast \sum_t |x_t|^2 + \text{noise}$, with $K = \mathbb E\, |\mathrm{PSF}_t|^2$. $K$ is known from the trajectory, so the energy map is a non-negative deconvolution in 3D. Speckle interferometry recovers diffraction-limited power spectra from short exposures through random atmospheric phase by the same averaging of squared transforms (Labeyrie, 1970). The pseudo-$C_\ell$ estimators of CMB power spectra deconvolve the coupling kernel of an incomplete sky the same way (Hivon et al., 2002).
- **Pairs of samples in one TR.** In k-space, $\sum_t |x_t|^2$ needs, for each separation $q$, the products $X_t(k)\, X_t(k - q)^*$ of samples of the same TR. Weighting each pair by the inverse of its probability under the random rotation gives an unbiased estimate (Horvitz and Thompson, 1952), and leaving out the products of a sample with itself removes the noise bias. What it needs is the set of differences that one TR's samples cover, as nested and co-prime arrays estimate second-order statistics at more lags than they have sensors (Pal and Vaidyanathan, 2010; Vaidyanathan and Pal, 2011).

**The conditioning is set by the differences, not by the number of samples.** The Fourier transform of $K$ is the autocorrelation of one TR's sampling, averaged over the rotations: how often one TR has a pair of samples at separation $q$. It must not be small at the separations the energy map needs. For uniform random sampling $K$ is a unit peak plus a flat plateau $\rho / M$ (table 1, last column), whose transform is 1 at every $q \neq 0$: perfectly conditioned at any $\rho$, and the correction is the subtraction of the plateau. A rosette TR has ~40 passes through the centre in different directions, so its differences cover the ball densely; the 3D check (section 8) measures how evenly. **The number of samples in one TR enters the variance, not the conditioning**, through $\rho$, as for every form (section 5).

### 3.5 Each TR is a band in time

A sample is at $t_m = T_p + \tau_m$, not at $T_p$ (section 1). Four consequences:

- **The basis is evaluated at each sample's time.** $\hat c_j = N \bar A^{-1} \sum_m \psi_j(t_m)^*\, a_m\, d_m\, y_m$ over the samples $m$, with $a_m$ the encoding row and $d_m$ the density weight: a weight per sample, not per TR. Everything in sections 2–3.4 holds with that change. For a band with $2\pi B\, d \ll 1$, with $d$ the readout duration, the difference is small: 1% in the toy for a band up to 0.4 Hz (table 4).
- **Within one TR, a change over $t$ looks like a change over τ.** Along the diagonal $t - \tau = T_p$, a component $e^{2\pi i f t}$ is $e^{2\pi i f T_p}\, e^{2\pi i f \tau}$: within the TR it is a frequency $f$ along τ, indistinguishable from an off-resonance of $f$ Hz; to first order, any change of $x$ during the readout is a term linear in τ, whose imaginary part looks like a field change and whose real part like a change of R2\*. Across TRs the factor $e^{2\pi i f T_p}$ separates them, modulo $1/T_R$. In the Fourier domain of (τ, $t$), with $\nu$ the frequency along τ, the sampling $\sum_p \delta(t - \tau - p T_R)$ over $0 \le \tau \le d$ replicates the spectrum at the shifts $(\nu, f) \to (\nu - n/T_R,\, f + n/T_R)$, each blurred along $\nu$ over a width $1/d$: those components alias onto each other. For BOLD ($f$ < 0.1 Hz) the within-readout change is negligible ($2\pi f d$ < 0.03). For the cardiac cycle (1–3 Hz) it is not ($2\pi f d$ up to 0.85): pulsation during the readout is read as a τ-linear term. This is the confound that the τ-linearity test of [b0-scan-time-plan.md](b0-scan-time-plan.md) must keep in mind: a change of B0 and a fast change of the object both give phase that grows with τ within a TR, and only the TR-to-TR behaviour tells them apart.
- **The lags between samples are continuous.** Pairs of samples of different TRs have lags $|T_p - T_{p'}| + (\tau_m - \tau_n)$, which with a duty cycle of $d / T_R$ = 0.9 cover nearly every lag, not only multiples of $T_R$. The coherence-time argument of section 3.4 is therefore in continuous time, and the U-statistic leaves out the pairs within one TR (lags under 45 ms), whose sampling is dependent: for a band, a fraction of about $2B\, d^2 / T_R$ of the form, slightly less than the $2B\, T_R$ of the instantaneous model.
- **The TR itself is a period.** Whatever the rotations, every TR repeats the same timing in τ, and each shape the same $|k|$ at each τ. So part of the leakage is periodic with $T_R$ and lands at multiples of 20 Hz, as for the periodic schedules of section 3.1: far above the physiological bands, but a reason to keep the τ-resolved subtraction exact.

## 4. Phase and magnitude

$q_W$ is a form in the complex $x_t$. A phase that changes over $T$ is therefore part of the power: a field change over $T$ ([b0-scan-time-plan.md](b0-scan-time-plan.md)) adds $x\,(e^{i\phi_t} - 1)$ to the residual, and its band power with it. Correct the field first, or keep the phase in the result as signal. A form in the magnitude, such as the power of $|x_t|$, is not quadratic in $x_t$ and does not separate. That is the price of the separation. Multi-shot diffusion faces the same choice between magnitude averaging and modelling the shot phase (Chen et al., 2013; Eichner et al., 2015).

For the energy alone ($W = I$), the phase does not matter at each voxel: $\sum_t |x_t|^2$ ignores any phase that varies over $T$. That is the reason root-sum-of-squares combinations are used across coils, shots and echoes.

## 5. The figure of merit, at this project's sizes

With the static image subtracted and patterns independent across TRs, for a voxel whose residual energy is near the average:

- leakage bias of $\hat q_W$, relative to the band power: $\approx \rho\, (J/N) / f$, with $f$ the fraction of the residual energy in the band. $\hat q_U$ removes it.
- standard deviation of the leakage, relative to the band power: about $\rho\, (J/N) / (f \sqrt{J})$. It falls as the scan gets longer ($J$ grows with it) while the bias does not change.
- $J/N \approx 2 B\, T_R$: 0.009 for $B$ = 0.09 Hz and $T_R$ = 50 ms.

$\rho$ for one TR at each resolution, if all 14 000 samples counted (an upper bound for the samples, so a lower bound for $\rho$; the rosette puts more samples near the centre, which helps coarse resolutions): 0.2 at 8 mm (17 000 cells), 9 at 4 mm, 78 at 2 mm, 630 at 1 mm. Coils divide it by roughly the usable acceleration.

| resolution | $\rho$ (no coils) | relative std per voxel, $f = 0.5$, 250 s |
|---|---|---|
| 4 mm | 9 | 0.02 |
| 2 mm | 78 | 0.2 |
| 1 mm | 630 | 1.7 |

So band power and coherence maps look practical at 2–4 mm from the leakage alone, and at 1 mm need the coils, spatial averaging, or a longer scan. The same formula holds for the energy of the residual band-limited to 3 Hz ($J$ = 1500, $f$ = 1): relative std $\approx 0.008\, \rho$, so 0.07 at 4 mm, 0.6 at 2 mm, 5 at 1 mm per voxel. With aliasing that is white in space, averaging $v$ voxels of the map divides these by $\sqrt v$: the energy map at a coarser resolution than the image is cheaper than the image's own resolution, which is the sense in which the fine detail of the map, not the map, is what the samples of one TR limit. These numbers assume aliasing that is white in space, as in the toy. The 3D rosette's point-spread function is not, so its leakage is structured, and the numbers need a 3D check (section 8). The intrinsic variance of a band power of a random process, about $1/\sqrt{J}$ per voxel, comes on top. It is the same for any method.

## 6. Relation to low-rank models and to this project

| | low-rank model in $T$ | quadratic collapse |
|---|---|---|
| what is fitted | a subspace of time courses, from the data | nothing in $T$; the basis is set by $W$ |
| structure imposed | shared time courses; can create correlations between regions | none |
| bias | model error when the rank is too low | removable ($\hat q_U$), up to $J/N$ of the residual energy |
| cost | coupled solve over the subspace | a batch of static solves with one operator |
| output | time courses | only the quadratic quantities |
| needs | low rank | independent patterns (or a designed k–T spectrum), and the static image subtracted |

Subspace and low-rank reconstructions (Liang, 2007; Chiew et al., 2015) gain from the low rank of real data, but they can only represent signals inside the subspace they find. Chiew et al. (2018) report that recovery is poor when the signals of interest have low variance, and that in simulation two regions with a 1 s latency difference collapse onto one common component. That reconstruction and processing can induce artificial correlations is established for linear operations (Karaman et al., 2014), and leakage between slices in multiband reconstruction produces false activation (Todd et al., 2016). We found no study that measures whether low-rank reconstruction inflates functional connectivity; that it can, by giving regions shared time courses, is our inference from the model, and a test of it belongs with section 8. The quadratic collapse is the complement: it cannot give time courses, and it gains nothing from low rank, but it needs no subspace and imposes none. The two can be combined: subtract what a model explains well (the static image, drift, field modes, respiration), then collapse what remains.

**As a kind of requirement.** In [coverage-target.md](coverage-target.md) a dimension is fixed, averaged, or as fine as possible. "Collapsed by a quadratic form" would be a fourth. Its conditions differ from those of an average:

- the static requirement at the same resolution (for $\bar A$ and the subtraction);
- per-TR coverage, through $\rho(\text{res})$, which sets the variance;
- independence of the patterns across TRs, or a k–T spectrum with no power in the band: the k–T spectral score of [example-requirements-design.md](example-requirements-design.md) (section 2) is the leakage of section 3.1, so it is the exact objective for this requirement, not a proxy;
- for broadband forms only, the set of differences of one TR's samples (section 3.4).

The shuffled schedule satisfies the independence. A greedy or periodic schedule (sections 5 and 6.1 of the requirements design) must be checked against the band, as table 3 shows.

**Other collapsed dimensions.** The same machinery applies to any dimension whose samples come from independent patterns. That is true of $T$, whose TRs are acquired with independent rotations, and of repetitions or shots in general. It is not true of τ: every TR samples all of its τ in one pattern, so a form over τ has only within-TR products, and needs the deconvolution of section 3.4. The root-sum-of-squares over echoes of multi-echo imaging is such a form.

## 7. Toy results

Tables 1–3 treat each TR as an instant; table 4 does not. A 1D image of $M$ = 128 cells, $N$ = 2000 TRs, each sampling $S$ of the 128 DFT coefficients, uniformly at random (or from a cycle of patterns). The per-TR estimate is the density-compensated adjoint, $z_t = (M/S)\, F^H P_t^H y_t$. The band is temporal bins 5–39 ($J$ = 35, $J/N$ = 1.75%). 30 trials per row. Script: [`toy_x_t.py`](quadratic-norm-report/toy_x_t.py), seeded; the tables are its output.

**Table 1: leakage against the undersampling of one TR.** A 40-cell region with all of its fluctuation in the band (1% RMS), no static image. Bias and standard deviation are relative to the true band power.

| $S$ | $\rho$ | $\hat q_W$ bias | predicted, $J \rho / M \cdot \bar E$ | $\hat q_U$ bias | $\hat q_U$ std | energy / $(E + \rho \sum E / M)$ |
|---|---|---|---|---|---|---|
| 8 | 15 | +0.079 | +0.080 | −0.017 | 0.065 | 0.984 |
| 16 | 7 | +0.033 | +0.037 | −0.021 | 0.043 | 0.986 |
| 32 | 3 | +0.015 | +0.016 | −0.018 | 0.029 | 0.991 |
| 64 | 1 | +0.005 | +0.005 | −0.018 | 0.016 | 0.996 |
| 96 | 0.33 | +0.002 | +0.002 | −0.017 | 0.010 | 0.998 |

The leakage bias follows the prediction of section 3.1. $\hat q_U$ removes it, leaving the diagonal term $-J/N$ = −1.75% of section 3.3. Its standard deviation grows with $\rho$. The last column checks the energy model of section 3.4: a unit peak plus a plateau $\rho / M$.

**Table 2: an object.** A static image, a global oscillation outside the band (2% of the image, at bin 300, standing in for respiration), and independent band fluctuations (1% RMS) in two regions A and B. $S$ = 32 ($\rho$ = 3). Band power is relative to the true band power in A; cross-power is between the centres of A and B.

| case | $\hat q_W$ in A | $\hat q_U$ in A (std) | $\hat q_U$ outside A, B | A–B cross-power, true 0.028: $\hat q_U$ mean (std) |
|---|---|---|---|---|
| static image not subtracted | 260 | −158 (50) | −78 | 26 (9.7) |
| static image subtracted | 1.10 | 0.91 (0.08) | −0.03 | 0.022 (0.017) |
| subtracted, with noise | 2.91 | 0.98 (0.41) | −0.03 | 0.032 (0.088) |

Without the subtraction, the static image's leakage swamps everything (section 3.1). With it, $\hat q_W$ is biased by the leakage of the global oscillation (+10% in A, and +11% of A's power outside the regions in the plain estimate). $\hat q_U$ removes that and the noise bias, which nearly triples $\hat q_W$ in the noisy case. The −9% that remains in A is the diagonal term: $J/N$ times the residual energy, which the global oscillation makes four times the band power at A. The cross-power has the same diagonal bias from the shared oscillation, and its standard deviation shows that one pair of voxels at this $N$ is noisy.

**Table 3: where the leakage goes.** A static image only, no subtraction; the mean power over the temporal bins at the voxels outside the object, relative to the mean over all bins.

| schedule | in the band (bins 5–39) | at multiples of $N$ / period | other bins |
|---|---|---|---|
| independent random | 0.99 | — | 1.00 |
| 10 patterns, cycled | 0.00 | 222 | 0.00 |
| 100 patterns, cycled | 0.61 | 20 | 0.00 |

Independent patterns make the leakage white. A cycle of 10 patterns puts it all at bins 200, 400, …, outside the band. A cycle of 100 puts it at multiples of 20, and bin 20 is in the band, so the band gets 1/35 of its bins at 20 times the mean (0.58 expected, 0.61 measured).

**Table 4: readout timing.** Each TR's 32 samples spread over 45 ms of a 50 ms TR, at absolute times $T_p + \tau_s$; the static image decays with R2\* = 20/s; the region's fluctuation is in the band 0.05–0.4 Hz of absolute time. The static model of each k coefficient, over all the TRs that sampled it, is its mean (τ-averaged) or a quadratic in τ (τ-resolved). The basis is evaluated at each TR's start or at each sample's time. 6 trials; relative to the true band power of the region.

| static model | basis time | $\hat q_U$ in the region (std) | $\hat q_U$ in the rest of the object | $\hat q_W$ in the region |
|---|---|---|---|---|
| τ-averaged | TR | 2.06 (1.33) | 0.85 | 12.3 |
| τ-averaged | sample | 0.51 (1.52) | −0.65 | 10.7 |
| τ-resolved | TR | 0.966 (0.022) | 0.000 | 0.988 |
| τ-resolved | sample | 0.977 (0.026) | 0.000 | 0.999 |

The decay that a τ-averaged static model leaves in every TR dominates both estimators. With the τ-resolved model, the U-statistic is within its trials' standard error of the expected diagonal bias, and per-sample times move it by about 1%.

## 8. Open questions and next steps

1. **3D, with the real trajectory.** The leakage kernel of the rosette (its mean squared point-spread function per TR, at each resolution), $\rho(\text{res})$ from the samples of one TR inside each ball, and the band power and coherence of the phantom (`relaxed-rosettes simulate`, with BOLD blocks and the cardiac and respiratory changes it already has), against the truth.
2. **Coils.** How much the coils reduce $\rho$ for one TR at each resolution: the $R(\text{res})$ question of [coverage-target.md](coverage-target.md).
3. **The variance and the frame potential.** The variance of $\hat q_U$ is a fourth-order sum over pairs of TRs. *Hypothesis:* under a Gaussian model it reduces to a frame potential with the band as the temporal kernel, which would make the frame-potential objective of the requirements design (section 3.2) the exact design objective for this requirement as well.
4. **Fast physiology within the readout.** Cardiac pulsation changes the object within one 45 ms readout and is read as a τ-linear term (section 3.5). Whether it should be subtracted with its own model (for example from a pulse recording) before a band power or coherence is estimated.
5. **Regularization.** The separated solves of section 2 are unregularized static reconstructions. Regularization trades bias for variance and changes $\mathbb E[R_t A_t]$, so the U-statistic's unbiasedness then holds for the regularized (blurred) image. Its effect on the band power and the coherence needs the 3D check.
6. **Where it goes in the requirements.** Whether "collapsed by a quadratic form" becomes a fourth kind of requirement in [coverage-target.md](coverage-target.md), and whether BOLD is better stated as a band power (and coherence) requirement than as time courses at 0.5 s.

## References

- Ariananda DD, Leus G (2012). Compressive wideband power spectrum estimation. *IEEE Trans Signal Process* 60(9):4775–4789. doi:10.1109/TSP.2012.2201153
- Candès EJ, Strohmer T, Voroninski V (2013). PhaseLift: exact and stable signal recovery from magnitude measurements via convex programming. *Comm Pure Appl Math* 66(8):1241–1274. doi:10.1002/cpa.21432
- Chen NK, Guidon A, Chang HC, Song AW (2013). A robust multi-shot scan strategy for high-resolution diffusion weighted MRI enabled by multiplexed sensitivity-encoding (MUSE). *NeuroImage* 72:41–47. doi:10.1016/j.neuroimage.2013.01.038
- Chen Y, Chi Y, Goldsmith AJ (2015). Exact and stable covariance estimation from quadratic sampling via convex programming. *IEEE Trans Inf Theory* 61(7):4034–4059. doi:10.1109/TIT.2015.2429594
- Chiew M, Smith SM, Koopmans PJ, Graedel NN, Blumensath T, Miller KL (2015). k-t FASTER: acceleration of functional MRI data acquisition using low rank constraints. *Magn Reson Med* 74(2):353–364. doi:10.1002/mrm.25395
- Chiew M, Graedel NN, Miller KL (2018). Recovering task fMRI signals from highly under-sampled data with low-rank and temporal subspace constraints. *NeuroImage* 174:97–110. doi:10.1016/j.neuroimage.2018.02.062
- Constantinides CD, Atalar E, McVeigh ER (1997). Signal-to-noise measurements in magnitude images from NMR phased arrays. *Magn Reson Med* 38(5):852–857. doi:10.1002/mrm.1910380524
- Demené C, Deffieux T, Pernot M, et al. (2015). Spatiotemporal clutter filtering of ultrafast ultrasound data highly increases Doppler and fUltrasound sensitivity. *IEEE Trans Med Imaging* 34(11):2271–2285. doi:10.1109/TMI.2015.2428634
- Dertinger T, Colyer R, Iyer G, Weiss S, Enderlein J (2009). Fast, background-free, 3D super-resolution optical fluctuation imaging (SOFI). *Proc Natl Acad Sci USA* 106(52):22287–22292. doi:10.1073/pnas.0907866106
- Dietrich O, Raya JG, Reeder SB, Reiser MF, Schoenberg SO (2007). Measurement of signal-to-noise ratios in MR images: influence of multichannel coils, parallel imaging, and reconstruction filters. *J Magn Reson Imaging* 26(2):375–385. doi:10.1002/jmri.20969
- Eichner C, Cauley SF, Cohen-Adad J, et al. (2015). Real diffusion-weighted MRI enabling true signal averaging and increased diffusion contrast. *NeuroImage* 122:373–384. doi:10.1016/j.neuroimage.2015.07.074
- Fessler JA, Lee S, Olafsson VT, Shi HR, Noll DC (2005). Toeplitz-based iterative image reconstruction for MRI with correction for magnetic field inhomogeneity. *IEEE Trans Signal Process* 53(9):3393–3402. doi:10.1109/TSP.2005.853152
- Gross J, Kujala J, Hämäläinen M, Timmermann L, Schnitzler A, Salmelin R (2001). Dynamic imaging of coherent sources: studying neural interactions in the human brain. *Proc Natl Acad Sci USA* 98(2):694–699. doi:10.1073/pnas.98.2.694
- Gudbjartsson H, Patz S (1995). The Rician distribution of noisy MRI data. *Magn Reson Med* 34(6):910–914. doi:10.1002/mrm.1910340618
- Hinshaw G, et al. (2003). First-year Wilkinson Microwave Anisotropy Probe (WMAP) observations: the angular power spectrum. *Astrophys J Suppl* 148(1):135–159. doi:10.1086/377225
- Hivon E, Górski KM, Netterfield CB, Crill BP, Prunet S, Hansen F (2002). MASTER of the cosmic microwave background anisotropy power spectrum: a fast method for statistical analysis of large and complex cosmic microwave background data sets. *Astrophys J* 567(1):2–17. doi:10.1086/338126
- Hoeffding W (1948). A class of statistics with asymptotically normal distribution. *Ann Math Stat* 19(3):293–325. doi:10.1214/aoms/1177730196
- Horvitz DG, Thompson DJ (1952). A generalization of sampling without replacement from a finite universe. *J Am Stat Assoc* 47(260):663–685. doi:10.1080/01621459.1952.10483446
- Hutchinson MF (1990). A stochastic estimator of the trace of the influence matrix for Laplacian smoothing splines. *Commun Stat Simul Comput* 19(2):433–450. doi:10.1080/03610919008812866
- Karaman M, Nencka AS, Bruce IP, Rowe DB (2014). Quantification of the statistical effects of spatiotemporal processing of nontask fMRI data. *Brain Connect* 4(9):649–661. doi:10.1089/brain.2014.0278
- Kellman P, McVeigh ER (2005). Image reconstruction in SNR units: a general method for SNR measurement. *Magn Reson Med* 54(6):1439–1447. doi:10.1002/mrm.20713
- Labeyrie A (1970). Attainment of diffraction limited resolution in large telescopes by Fourier analysing speckle patterns in star images. *Astron Astrophys* 6:85–87. (ADS 1970A&A.....6...85L)
- Liang Z-P (2007). Spatiotemporal imaging with partially separable functions. *Proc IEEE ISBI*, 988–991. doi:10.1109/ISBI.2007.357020
- Lustig M, Donoho D, Pauly JM (2007). Sparse MRI: the application of compressed sensing for rapid MR imaging. *Magn Reson Med* 58(6):1182–1195. doi:10.1002/mrm.21391
- Macé E, Montaldo G, Cohen I, Baulac M, Fink M, Tanter M (2011). Functional ultrasound imaging of the brain. *Nat Methods* 8(8):662–664. doi:10.1038/nmeth.1641
- Madore B, Glover GH, Pelc NJ (1999). Unaliasing by Fourier-encoding the overlaps using the temporal dimension (UNFOLD), applied to cardiac imaging and fMRI. *Magn Reson Med* 42(5):813–828. doi:10.1002/(SICI)1522-2594(199911)42:5<813::AID-MRM1>3.0.CO;2-S
- Mullis CT, Scharf LL (1991). Quadratic estimators of the power spectrum. In: Haykin S (ed.), *Advances in Spectrum Analysis and Array Processing*, Vol. I, ch. 1, 1–57. Prentice-Hall.
- O'Leary DP (1980). The block conjugate gradient algorithm and related methods. *Linear Algebra Appl* 29:293–322. doi:10.1016/0024-3795(80)90247-5
- Pal P, Vaidyanathan PP (2010). Nested arrays: a novel approach to array processing with enhanced degrees of freedom. *IEEE Trans Signal Process* 58(8):4167–4181. doi:10.1109/TSP.2010.2049264 (erratum 58(9):4973, doi:10.1109/TSP.2010.2061930)
- Percival DB, Walden AT (1993). *Spectral Analysis for Physical Applications: Multitaper and Conventional Univariate Techniques.* Cambridge University Press. doi:10.1017/CBO9780511622762
- Polenta G, Marinucci D, Balbi A, et al. (2005). Unbiased estimation of an angular power spectrum. *J Cosmol Astropart Phys* 2005(11):001. doi:10.1088/1475-7516/2005/11/001
- Reeder SB, Wintersperger BJ, Dietrich O, et al. (2005). Practical approaches to the evaluation of signal-to-noise ratio performance with parallel imaging: application with cardiac imaging and a 32-channel cardiac coil. *Magn Reson Med* 54(3):748–754. doi:10.1002/mrm.20636
- Robson PM, Grant AK, Madhuranthakam AJ, Lattanzi R, Sodickson DK, McKenzie CA (2008). Comprehensive quantification of signal-to-noise ratio and g-factor for image-based and k-space-based parallel imaging reconstructions. *Magn Reson Med* 60(4):895–907. doi:10.1002/mrm.21728
- Roemer PB, Edelstein WA, Hayes CE, Souza SP, Mueller OM (1990). The NMR phased array. *Magn Reson Med* 16(2):192–225. doi:10.1002/mrm.1910160203
- Romero D, Ariananda DD, Tian Z, Leus G (2016). Compressive covariance sensing: structure-based compressive sensing beyond sparsity. *IEEE Signal Process Mag* 33(1):78–93. doi:10.1109/MSP.2015.2486805
- Rowe DB, Hahn AD, Nencka AS (2009). Functional magnetic resonance imaging brain activation directly from k-space. *Magn Reson Imaging* 27(10):1370–1381. doi:10.1016/j.mri.2009.05.048
- Slepian D (1978). Prolate spheroidal wave functions, Fourier analysis, and uncertainty — V: the discrete case. *Bell Syst Tech J* 57(5):1371–1430. doi:10.1002/j.1538-7305.1978.tb02104.x
- Sun FT, Miller LM, D'Esposito M (2004). Measuring interregional functional connectivity using coherence and partial coherence analyses of fMRI data. *NeuroImage* 21(2):647–658. doi:10.1016/j.neuroimage.2003.09.056
- Thomson DJ (1982). Spectrum estimation and harmonic analysis. *Proc IEEE* 70(9):1055–1096. doi:10.1109/PROC.1982.12433
- Todd N, Moeller S, Auerbach EJ, Yacoub E, Flandin G, Weiskopf N (2016). Evaluation of 2D multiband EPI imaging for high-resolution, whole-brain, task-based fMRI studies at 3T: sensitivity and slice leakage artifacts. *NeuroImage* 124(Pt A):32–42. doi:10.1016/j.neuroimage.2015.08.056
- Tristram M, Macías-Pérez JF, Renault C, Santos D (2005). XSPECT, estimation of the angular power spectrum by computing cross-power spectra with analytical error bars. *Mon Not R Astron Soc* 358(3):833–842. doi:10.1111/j.1365-2966.2005.08760.x
- Tsao J, Boesiger P, Pruessmann KP (2003). k-t BLAST and k-t SENSE: dynamic MRI with high frame rate exploiting spatiotemporal correlations. *Magn Reson Med* 50(5):1031–1042. doi:10.1002/mrm.10611
- Vaidyanathan PP, Pal P (2011). Sparse sensing with co-prime samplers and arrays. *IEEE Trans Signal Process* 59(2):573–586. doi:10.1109/TSP.2010.2089682
- Zang YF, He Y, Zhu CZ, et al. (2007). Altered baseline brain activity in children with ADHD revealed by resting-state functional MRI. *Brain Dev* 29(2):83–91. doi:10.1016/j.braindev.2006.07.002
- Zou QH, Zhu CZ, Yang Y, et al. (2008). An improved approach to detection of amplitude of low-frequency fluctuation (ALFF) for resting-state fMRI: fractional ALFF. *J Neurosci Methods* 172(1):137–141. doi:10.1016/j.jneumeth.2008.04.012
