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

Each shape is a convex problem, solved with cvxpy and Clarabel, in parallel (`--jobs`, default: the CPU count). The `.npz` file has `g_mT_per_m` (shapes × raster points × 3, sampled at the raster centers, as a pypulseq arbitrary gradient with `first = last = 0`), the targets, and the config.

Build the `.seq` file from the optimized shapes:

```
uv run relaxed-rosettes sequence shapes.npz -o relaxed_rosettes.seq
```

Each TR is one block: a hard pulse with the gradients off, one shape rotated by a rotation event, and the ADC (none in the dummy TRs at the start). The rotations are a super-Fibonacci spiral on SO(3), split among the shapes, in shuffled order. Next to the `.seq` file, the design file `relaxed_rosettes.npz` has everything it was built from: the schedule of each TR (`shape_index`, `rotation_matrix` with physical gradient = R @ shape gradient, `rf_phase_rad` (also the ADC phase), `acquired`), the shape waveforms and targets, and the config. The other commands read it.

Check the sequence:

```
uv run relaxed-rosettes check relaxed_rosettes.seq --asc MP_GPA_....asc --band 500:700
```

It reports pypulseq's timing check; the peak gradient and slew per physical axis over every TR; whether the ADC dwell meets Nyquist along the path at the peak gradient; the gradient spectrum peak and its level in each `--band` (Hz; e.g. the scanner's acoustic resonances); and, with the scanner's gradient `.asc` file, the peak PNS (SAFE model). The spectrum and PNS cover the first `--trs` acquired TRs (default 100). `--json` prints the report as JSON. It exits with status 1 if timing, Nyquist, or PNS fails.

Export the k-space trajectory at the ADC samples, for reconstruction:

```
uv run relaxed-rosettes export-traj relaxed_rosettes.seq -o traj.npz
```

All acquired TRs sample the same times, and each plays one shape rotated, so the file is compact: `k_shape_per_m` (shapes × samples × 3, logical axes), `adc_time_in_tr_s` and `time_since_excitation_s` (samples), and for each acquired TR `tr_index`, `tr_start_s`, `shape_index`, `rotation_matrix`, and `rf_phase_rad`. The k-space position (1/m) of sample `s` in acquired TR `a` is `rotation_matrix[a] @ k_shape_per_m[shape_index[a], s]`; `relaxed_rosettes.trajectory.load_trajectory(path)[1].k_per_m()` computes it for all (or some) TRs. Before writing, the export checks itself against pypulseq's k-space of the `.seq` file.

For the concomitant (Maxwell) field, the file also has `gg_shape_T2s_per_m2` (shapes × samples × 3 × 3, logical axes): `∫ g gᵀ dt` from the start of the TR, in T² s/m². The lowest-order concomitant field is quadratic in the gradient, so the physical tensor of acquired TR `a` is `R @ gg_shape_T2s_per_m2[shape_index[a], s] @ R.T` with `R = rotation_matrix[a]` (`Trajectory.gg_T2s_per_m2()`), and `relaxed_rosettes.trajectory.concomitant_phase_rad(config, gg, r_m)` gives its phase at physical positions (z along B0, from the isocenter; same sign as the encoding phase `2π k·r`). The phase is a second-order polynomial in position, for a reconstruction with an expanded encoding model. It comes from the nominal gradients: eddy currents and the gradient system response are not included.

Measure the resolution–time trade-off: for windows of scan time of several lengths, at several start times, how much of k-space each resolution needs is sampled:

```
uv run relaxed-rosettes coverage relaxed_rosettes.seq --windows 0.5,2,10,30 --resolutions-mm 8,4,2,1
```

It prints two tables (window length × resolution, median and worst over start times): the *coverage*, the fraction of k-space cells (spacing 1/FOV) inside the resolution's k-space ball that hold a sample; and the *gap*, the 95th percentile distance from that ball to the nearest sample, in cells, which shows how far a reconstruction must fill in between samples. `--tau-bins N` reports the worst of N bins of time since excitation; `--json` prints the arrays.

It also reports the sampling of the (|k|, time since excitation) plane over the whole scan: the fraction of (|k| shell × τ bin) cells with no sample (overall, early in the readout, and in the outer half of k-space) and the longest τ gap in any shell. Rotations do not change |k|(t), so only the shapes sample this plane; more shapes (`shapes.num_shapes`), not more TRs, fill it in.

Convert the scanner's raw data to ISMRMRD, the input of the reconstruction (needs the `recon` extra):

```
uv run relaxed-rosettes to-ismrmrd meas_MID00042_relaxed_rosettes.dat relaxed_rosettes.seq -o raw.h5
```

It reads the Siemens `meas.dat` file with twixtools and checks that its header has the MD5 signature of the `.seq` file, so raw data cannot be paired with the wrong design (`--no-signature-check` converts anyway, with a warning). It writes one acquisition for each acquired TR, in order, with the ADC segments (the interpreter splits an ADC longer than `system.adc_samples_limit` samples) joined, and one for each noise scan (`ACQ_IS_NOISE_MEASUREMENT`). In a TR acquisition, `scan_counter` is the acquired TR number, `user_int[0]` its index in the full schedule and `user_int[1]` its shape; `traj` is the k-space position in cycles per FOV on the physical gradient axes (`--no-trajectory` leaves it out). The header has the design config, its digest, and the `.seq` signature as user parameters. No patient information is copied. A scan with fewer TRs than the design (e.g. stopped early) needs `--allow-partial`. The interpreter applies the ADC phase offset, so the data are already demodulated by the RF phase of each TR.

`relaxed_rosettes.recon.rawdata.load_rawdata(path)` reads the file: the noise samples, the TR index, shape, and time stamp of each TR, and the data of any TRs on demand (`read`). `check_trajectory(config, trajectory)` fails unless the raw data and a trajectory file come from the same design.

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
