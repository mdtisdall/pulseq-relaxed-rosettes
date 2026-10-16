# pulseq-relaxed-rosettes

A relaxed rosette k-space trajectory pulse sequence, written in Python with [PyPulseq](https://github.com/pulseq/pypulseq).

## Setup

The nix devShell supplies Python, uv, and gh. With direnv, it loads when you enter the directory. Otherwise:

```
nix develop
uv sync
```

The reconstruction commands need the `recon` extra: `uv sync --extra recon` (or `--all-extras`).

## Usage

Every design parameter is in one TOML config file, including the random seed. Write the defaults to a file, edit it, then check it:

```
uv run relaxed-rosettes config init design.toml
uv run relaxed-rosettes config show design.toml
```

`config show` validates the config and prints it with every default filled in, together with its digest. Override single values with `--set`:

```
uv run relaxed-rosettes config show design.toml --set readout.tr_s=0.025 --set readout.adc_num_samples=7000
```

Generate the k-space targets of every shape (petal tips, k = 0 crossings, and the spoil moment, with the time of each):

```
uv run relaxed-rosettes targets design.toml -o targets.npz
```

The `.npz` file has `k` (shapes × targets × 3, in units of k_max), `t_s` (shapes × targets, seconds from the start of the TR block), and the config that made them.

Optimize the gradient waveform of every shape (minimum slew energy, with the gradients off during the RF pulse and at the end of the TR, within the gradient and slew limits, through every target):

```
uv run relaxed-rosettes shapes design.toml -o shapes.npz
```

Each shape is a convex problem, solved with cvxpy and Clarabel, in parallel (`--jobs`, default: the CPU count). Clarabel and the linear-algebra libraries may start several threads in each worker process for little gain; with `RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1` in the environment, `--jobs N` uses N cores (measured: one solve used 3.3 cores without the limits, 1 with them, in the same time).

To keep the gradients out of the coil's acoustic resonances, set `gradients.penalty_bands_hz` (`[[low, high], ...]` in Hz) or pass the gradient system file, which replaces them with its resonances:

```
uv run relaxed-rosettes shapes design.toml --asc MP_GradSys_....asc -o shapes.npz
```

The optimizer then adds `gradients.band_penalty_weight` (default 100) times the slew energy at the band frequencies, each band widened by `gradients.penalty_band_margin_hz` (default 50 Hz) on both sides, because a spectrum measured over a 50 ms window spreads each frequency by about ±40 Hz. The band energy is a sum of squares of linear functions of the slew (Parseval, over one TR's DFT oversampled 4×), kept in a reduced basis (the SVD of the band frequencies, the Slepian directions of the bands), so each shape stays a convex quadratic problem. The penalty removes what the targets do not need: the main frequency of a shape is set by its target timing, so keep it away from the bands with `shapes.segment_time_margin` (slower petals, fewer per TR). The table that `shapes` prints has the fraction of each shape's gradient energy in the bands. The penalty costs solve time (one full-size shape: about 5 s instead of 0.6 s) and pushes the waveforms toward the gradient limit, which raises PNS: check every design. With the resonances in the config, the shapes, design, trajectory, and ISMRMRD files all contain them. Measured on the phantom designs (128 shapes; `docs/basis-quality-test-plan.md`): the 1 mm design at margin 1.5 has 0.6% of its gradient energy in the bands (2.3% in the worst shape), against 38% without the penalty, with 30–33 instead of 40–43 petals per TR, better coverage at 1 mm and slightly worse at 4 mm, and PNS up from 62% to 80%; the 4 mm design needs weight 300 at margin 2.0 (1.0%, worst 2.2%).

The `.npz` file has `g_mT_per_m` (shapes × raster points × 3, sampled at the raster centers, as a pypulseq arbitrary gradient with `first = last = 0`), the targets, and the config.

Build the `.seq` file from the optimized shapes:

```
uv run relaxed-rosettes sequence shapes.npz -o relaxed_rosettes.seq
```

Each TR is one block: a hard pulse with the gradients off, one shape rotated by a rotation event, and the ADC (none in the dummy TRs at the start). The rotations are a super-Fibonacci spiral on SO(3), split among the shapes, in shuffled order. Next to the `.seq` file, the design file `relaxed_rosettes.npz` has everything it was built from: the schedule of each TR (`shape_index`, `rotation_matrix` with physical gradient = R @ shape gradient, `rf_phase_rad` (also the ADC phase), `acquired`), the shape waveforms and targets, and the config. The other commands read it.

Check the sequence:

```
uv run relaxed-rosettes check relaxed_rosettes.seq --asc MP_GPA_....asc --band 500:700
```

It reports pypulseq's timing check; the peak gradient and slew per physical axis over every TR; whether the ADC dwell meets Nyquist along the path at the peak gradient; the gradient spectrum peak and its level in each `--band` (Hz; e.g. the scanner's acoustic resonances; by default the resonances in `--asc`, else the design's penalty bands); and, with the scanner's gradient `.asc` file, the peak PNS (SAFE model). The spectrum and PNS cover the first `--trs` acquired TRs (default 100). `--json` prints the report as JSON. It exits with status 1 if timing, Nyquist, or PNS fails.

Export the k-space trajectory at the ADC samples, for reconstruction:

```
uv run relaxed-rosettes export-traj relaxed_rosettes.seq -o traj.npz
```

All acquired TRs sample the same times, and each plays one shape rotated, so the file is compact: `k_shape_per_m` (shapes × samples × 3, logical axes), `adc_time_in_tr_s` and `time_since_excitation_s` (samples), and for each acquired TR `tr_index`, `tr_start_s`, `shape_index`, `rotation_matrix`, and `rf_phase_rad`. The k-space position (1/m) of sample `s` in acquired TR `a` is `rotation_matrix[a] @ k_shape_per_m[shape_index[a], s]`; `relaxed_rosettes.trajectory.load_trajectory(path)[1].k_per_m()` computes it for all (or some) TRs. Before writing, the export checks itself against pypulseq's k-space of the `.seq` file.

`--gradient-delays-s DX,DY,DZ` writes the trajectory as played with a delay on each physical gradient axis (seconds; for example from `relaxed-rosettes delays`, below): component `i` of k is the nominal one at `t - DX`, `t - DY` or `t - DZ`. Rotations mix the logical axes, so the file then also has `k_shape_delayed_per_m` (3 × shapes × samples × 3: each shape's k at the ADC times shifted by each axis's delay) and `gradient_delay_s`; component `i` of k in TR `a` is `rotation_matrix[a][i] @ k_shape_delayed_per_m[i, shape_index[a], s]`, and `k_per_m()` handles both forms. The check against pypulseq is of the nominal trajectory. `gg_shape_T2s_per_m2` stays nominal.

For the concomitant (Maxwell) field, the file also has `gg_shape_T2s_per_m2` (shapes × samples × 3 × 3, logical axes): `∫ g gᵀ dt` from the start of the TR, in T² s/m². The lowest-order concomitant field is quadratic in the gradient, so the physical tensor of acquired TR `a` is `R @ gg_shape_T2s_per_m2[shape_index[a], s] @ R.T` with `R = rotation_matrix[a]` (`Trajectory.gg_T2s_per_m2()`), and `relaxed_rosettes.trajectory.concomitant_phase_rad(config, gg, r_m)` gives its phase at physical positions (z along B0, from the isocenter; same sign as the encoding phase `2π k·r`). The phase is a second-order polynomial in position, for a reconstruction with an expanded encoding model. It comes from the nominal gradients: eddy currents and the gradient system response are not included.

Measure the resolution–time trade-off: for windows of scan time of several lengths, at several start times, how much of k-space each resolution needs is sampled:

```
uv run relaxed-rosettes coverage relaxed_rosettes.seq --windows 0.5,2,10,30 --resolutions-mm 8,4,2,1
```

It prints two tables (window length × resolution, median and worst over start times): the *coverage*, the fraction of k-space cells (spacing 1/FOV) inside the resolution's k-space ball that hold a sample; and the *gap*, the 95th percentile distance from that ball to the nearest sample, in cells, which shows how far a reconstruction must fill in between samples. `--tau-bins N` reports the worst of N bins of time since excitation; `--json` prints the arrays.

It also reports the sampling of the (|k|, time since excitation) plane over the whole scan: the fraction of (|k| shell × τ bin) cells with no sample (overall, early in the readout, and in the outer half of k-space) and the longest τ gap in any shell. Rotations do not change |k|(t), so only the shapes sample this plane; more shapes (`shapes.num_shapes`), not more TRs, fill it in.

Check the gradient spoiling of the schedule:

```
uv run relaxed-rosettes spoiling relaxed_rosettes.seq
uv run relaxed-rosettes spoiling relaxed_rosettes.seq --rotations random --rf-phase quadratic117
```

Each TR ends at its shape's spoil moment, turned by the TR's rotation, so the schedule decides where the spoiled magnetization of earlier TRs lies when later readouts sample k-space. `relaxed_rosettes.spoiling` follows one voxel through every TR (dummies included) with an extended phase graph whose states are continuous 3D dephasing vectors: the RF pulse with the TR's phase, the TR's readout path, T1 and T2 relaxation, diffusion, and the rotated spoil moment. A coherence dephased by `k` is sampled where the readout passes `-k`; the report sums the spurious amplitude (relative to each TR's FID) of the coherences whose closest approach to the TR's samples is within 1, 3 and 10 cells, split into transverse chains, conjugated and stored (stimulated) pathways; the fraction of TRs whose spurious amplitude within 3 cells exceeds 1% and 10%; and, for a phantom (`--phantom`, default the built-in one; `--no-phantom` to skip), the spurious over expected signal energy of each TR, from the 32 strongest near coherences. `--tissue T1,T2[,D]` (repeatable) sets the voxel (default grey matter, 1.5 s, 90 ms, 0.8 µm²/ms, and CSF, 4 s, 1.5 s, 3 µm²/ms); `--rotations random` replaces the schedule's rotations with independent random ones (incoherent spoiling, a reference) and `--rf-phase` the RF phase scheme. States below `--threshold` (default 1e-5 of M0) are dropped and at most `--max-states` kept; a `*` marks a tissue that reached the cap. Intravoxel off-resonance (T2') is ignored, which overestimates transverse pathways. `-o` writes the per-TR arrays.

Convert the scanner's raw data to ISMRMRD, the input of the reconstruction (needs the `recon` extra):

```
uv run relaxed-rosettes to-ismrmrd meas_MID00042_relaxed_rosettes.dat relaxed_rosettes.seq -o raw.h5
```

It reads the Siemens `meas.dat` file with twixtools and checks that its header has the MD5 signature of the `.seq` file, so raw data cannot be paired with the wrong design (`--no-signature-check` converts anyway, with a warning). It writes one acquisition for each acquired TR, in order, with the ADC segments (the interpreter splits an ADC longer than `system.adc_samples_limit` samples) joined, and one for each noise scan (`ACQ_IS_NOISE_MEASUREMENT`). In a TR acquisition, `scan_counter` is the acquired TR number, `user_int[0]` its index in the full schedule and `user_int[1]` its shape; `traj` is the k-space position in cycles per FOV on the physical gradient axes (`--no-trajectory` leaves it out). The header has the design config, its digest, and the `.seq` signature as user parameters. No patient information is copied. A scan with fewer TRs than the design (e.g. stopped early) needs `--allow-partial`. The interpreter applies the ADC phase offset, so the data are already demodulated by the RF phase of each TR.

`relaxed_rosettes.recon.rawdata.load_rawdata(path)` reads the file: the noise samples, the TR index, shape, and time stamp of each TR, and the data of any TRs on demand (`read`). `check_trajectory(config, trajectory)` fails unless the raw data and a trajectory file come from the same design.

### Simulated scans

A simulated scan of an analytic phantom, in the same ISMRMRD layout (needs the `recon` extra):

```
uv run relaxed-rosettes phantom init phantom.toml
uv run relaxed-rosettes phantom show phantom.toml
uv run relaxed-rosettes simulate phantom.toml relaxed_rosettes.seq -o sim.h5
```

The phantom (`relaxed_rosettes.recon.phantom`) is a set of compartments (`[[compartment]]`): uniform ellipsoids and Gaussian blobs, each with a proton density, R2*, and off-resonance. A compartment `inside` another replaces that one's tissue within its shape; otherwise compartments add. `[[sine]]` and `[[blocks]]` (periodic, raised-cosine edges) tables change a compartment's density (relative), R2* (1/s), or off-resonance (Hz) over scan time. The signal is the closed-form Fourier transform of each compartment times `ρ exp(-R2* τ + 2πi Δf τ)` at each sample's τ and scan time, so it is exact (no NUFFT, no time segmentation). Coils have constant complex sensitivities (`coil_sensitivity_real`, `coil_sensitivity_imag`); the noise is white, with standard deviation `noise_relative` times the signal at k = 0, τ = 0 in each of the real and imaginary parts, and `noise_readouts` noise-only readouts at the ADC dwell. The data are seeded from `seed`. The header also has the phantom TOML and its digest. `--gradient-delays-s DX,DY,DZ` plays each physical gradient axis late by that delay (seconds): the signal follows the delayed k, while the trajectory stored in the file stays nominal, as the scanner's would; the header records the delays. The default phantom is a head-sized ellipsoid with ventricles, small spheres of 1.5–4 mm radius, a Gaussian blob, a vessel pulsing at 1.1 and 2.2 Hz, and a BOLD region whose R2* falls by 0.6/s in 20 s blocks every 40 s; it has no off-resonance. A full default design takes a few minutes.

### Frame potential of the coverage requirements

How well a design conditions the linear reconstruction of each coverage requirement (needs the `recon` extra; see [docs/example-requirements-design.md](docs/example-requirements-design.md), section 3):

```
uv run relaxed-rosettes requirements init requirements.toml
uv run relaxed-rosettes frame-potential relaxed_rosettes.seq -r requirements.toml
```

A requirement (`[[requirement]]`) is a resolution, a τ model (`tau_rank = 0`: averaged; `L`: a rank-`L` decay basis over `[r2star_min_per_s, r2star_max_per_s]`), a scan-time model (`scan_time_width_s = 0`: averaged; `W`: hat functions `W` apart), and an optional Hann `taper` of the samples by `|k|`. Its model is voxels of that resolution over the field of view, times those bases, fitted to the samples inside its k-space ball (no coils, no off-resonance). The command reports, for each requirement, the frame potential `||A Aᴴ||²` of the forward operator as the scale-free ratio `N ||A Aᴴ||² / trace(A Aᴴ)²` (`N` coefficients): the mean squared eigenvalue of `Aᴴ A` over the squared mean, 1 for a tight frame, and the coefficient of variation of the eigenvalues. The ball does not fill the band of the cubic grid, so the ratio has an offset that is the same for every design. It is computed as a triangle-weighted energy of a type-1 NUFFT of the samples onto a grid of twice the size (one channel per pair of τ basis functions; one NUFFT per hat function for terms local in scan time). The default requirements are the four examples of the design document, with placeholder values; `--only NAME` evaluates one, `--json` prints the numbers.

`relaxed_rosettes.recon.basis.decay_basis` builds an orthonormal basis of `exp(-R2* τ)` over a range of R2*, on the τ of the ADC samples, for reconstructions that resolve τ: the leading singular vectors of the normalized curves, with the rank fixed or the smallest that represents every curve to a relative tolerance.

### Gradient delays

Estimate a delay on each physical gradient axis from the data (needs the `recon` extra; [docs/basis-quality-test-plan.md](docs/basis-quality-test-plan.md), section 3):

```
uv run relaxed-rosettes delays raw.h5 traj.npz -o delays.npz
uv run relaxed-rosettes export-traj relaxed_rosettes.seq -o traj-corrected.npz --gradient-delays-s DX,DY,DZ
```

Every petal passes through k = 0, where the signal has an echo. For each crossing of each TR, the command measures the peak of the coil-combined magnitude (prewhitened root sum of squares; a parabola through the largest sample and its neighbours) and fits, over all crossings, the echo time shift

`Δt = gᵀ Q D g / gᵀ Q g + gᵀ Q k₀ / (γ gᵀ Q g)`, with `D = diag(δx, δy, δz)` and `k₀ = k_offset - γ G τ`,

for the delays, a static k-space offset of the echo (object phase), and a background gradient `G` (linear off-resonance). `g` is the physical gradient at the crossing, from the trajectory. `Q` is the shape of the object's spectrum at its peak (the identity for a sphere), fitted from the widths of the echo peaks: without it, the delays of an elongated object come out biased by about 10% of their differences. The fit is iterated around the trajectory delayed by the current estimate until it converges, and drops outliers (more than 5 scaled MADs). The trajectory must be the nominal one (no delays).

It prints the delays with their standard errors, the offsets, the eigenvalues of `Q`, the residual, a bound on the k error that the delays cause (`γ max|δ| max|g|`, in cells of 1/FOV), the miss distance perpendicular to `g` at the crossings (which a time shift cannot absorb), and the `--gradient-delays-s` option for `export-traj`. The `.npz` file has these and every crossing's TR, shape, gradient, nominal and measured times, residual, and whether it was used. Equal delays on all axes are a shift of the ADC relative to the gradients. On simulated scans of P4 and P1 (default phantom, delays 3, −2 and 5 µs), the delays come out within 0.025 µs (0.01 µs without delays), in about 5 s; the standard errors are statistical only and are smaller than that.

### Time-averaged gridding reconstruction

Reconstruction parameters are in their own TOML config, separate from the design (one scan can be reconstructed many ways); its digest is stored in every output:

```
uv run relaxed-rosettes recon-config init recon.toml
uv run relaxed-rosettes recon-config show recon.toml --set data.resolution_m=0.004
```

A first image, averaged over the whole scan and the whole readout:

```
uv run relaxed-rosettes grid raw.h5 traj.npz -c recon.toml -o grid.npz
```

It checks that the raw data and the trajectory come from the same design, then:

- prewhitens the coils with the noise scans (the noise covariance is scaled from the noise dwell to the ADC dwell), and compresses them to `coils.virtual_coils` virtual coils (SVD of the whitened data of `coils.calibration_trs` TRs spread over the scan);
- uses the samples inside the k-space ball of `data.resolution_m` and the window `[data.tau_min_s, data.tau_max_s]` of time since excitation, from the first `data.max_trs` TRs (0: all), streamed `data.batch_trs` TRs at a time;
- weights them with Pipe-Menon density compensation (`dcf.iterations`; the kernel is the sinc² transform of a triangle window on a 2n³ grid, which is non-negative, so the iteration converges);
- grids them with FINUFFT (`nufft.eps`, `nufft.upsampling`, `nufft.threads`).

The `.npz` file has `rss` (n³, root sum of squares), `coil_images` (with `--coil-images`), `coil_matrix` (prewhitening then compression), `noise_covariance`, `singular_values`, `dcf_change`, and the recon and design config digests. Image axis `i` is physical gradient axis `i` (pypulseq x, y, z), and index `j` is at `(j - n // 2) * resolution_m` from the isocenter. With the 45 ms readout, off-resonance blurs this image; a narrower τ window reduces the blur.

### Static CG-SENSE reconstruction

```
uv run relaxed-rosettes sense raw.h5 traj.npz -c recon.toml -o sense.npz
```

One image from the whole scan, with coil maps and off-resonance, all estimated from the scan itself:

- The prewhitened, compressed, selected data are read once and kept in memory (about `8 * coils.virtual_coils + 20` bytes per selected sample: ~3.8 GB at 2 mm, ~6.5 GB at 1 mm with 12 virtual coils).
- Coil maps: the samples inside the k-space ball of `coil_maps.resolution_m` (default 8 mm), gridded on the reconstruction grid, then Inati's iterative method (box filter `coil_maps.smoothing_m`). The maps have unit norm over the coils.
- Field map (`b0.segments > 0`): low-resolution (`b0.resolution_m`) coil-combined images of `b0.windows` windows of time since excitation, each at the mean τ of its samples (empty windows are skipped); the frequency is a least-squares fit of the phase steps between consecutive windows, smoothed over `b0.smoothing_m` (weighted by magnitude), then clipped to the magnitude-weighted 0.5–99.5% range so the background's noise does not count. Each step must be below half a cycle.
- Off-resonance in the forward model: `exp(2πi f τ)` by `b0.segments` time segments, with least-squares coefficients over the field map's histogram (one NUFFT per segment).
- CG on `(Eᴴ W E + λ I) x = Eᴴ W y` (`cg.iterations`, `cg.tolerance`, `cg.tikhonov`), with `W` the density compensation (`cg.weighting = "dcf"`, which scales `Eᴴ W E` to about 1) or the identity (`"none"`).

The `.npz` file has `image` (complex), `magnitude`, `field_map_hz`, `segment_tau_s`, `segmentation_rms_error`, `coil_maps` (with `--coil-maps`), `cg_residual`, `coil_matrix`, and the recon and design config digests. The signal model is `sum_r x(r) exp(2πi (k·r + f(r) τ))`, the sign of the encoding phase, with τ from the RF center; the field map absorbs anything that adds phase linearly in τ. The image axes are those of `grid`.

The recon uses NumPy and FINUFFT on the CPU. FINUFFT cannot share a process with PyTorch on macOS (both wheels bundle an OpenMP runtime), so the recon does not use PyTorch.

## Tip-radius density

Where the petal tips are, in radius, decides how samples spread over k-space. `shapes.tip_density` chooses how the radii are distributed:

- `power` (default): radii `u ** tip_radius_power`, `u` blue noise.
- `exponent`: derived so that a window of scan time covers the k-space ball of resolution `res` in a time proportional to `res ** -resolution_time_exponent` (2 to below 3), over the resolutions from `density_resolution_min_m` to `density_resolution_max_m`.
- `filter`: derived to minimize the noise of Hann-filtered reconstructions at log-spaced resolutions in the same range, weighted `res ** filter_prior_power` per octave.

The derived models calibrate the trajectory speed on a few shapes (`calibration_shapes`, `calibration_iterations`). [docs/sampling-density.md](docs/sampling-density.md) derives both models and compares the designs. [docs/lattice-sampling.md](docs/lattice-sampling.md) discusses optimal-lattice (BCC, D4, D5) sampling references and their effect on these numbers.

## Dependencies

PyPulseq is pinned to a commit of the [mcencini fork](https://github.com/mcencini/pypulseq) (`experimental` branch), because it has the rotation extension (`pp.make_rotation`) and the official release does not yet ([pulseq/pypulseq#372](https://github.com/pulseq/pypulseq/pull/372)).
