# Recon optimization plan

Status: plan. No step is implemented. The steps start only after a real scan has shown that the recon is correct: the minimal phantom test with design P4 ([basis-quality-test-plan.md](basis-quality-test-plan.md), section 2), against a product 3D GRE. Each step is then measured on that real dataset before the next one starts. The h5py read leak that the first measurement found is fixed (PR #22; section 5).

The gains in this plan are estimates from the code and from one profile of a simulation (section 1). As each step is implemented, record the measured time, memory, and image change in section 7, in the same PR as the step, and correct the estimates in the text where they were wrong.

## Question

`sense` on a simulated 64 s scan of design P4 (4 mm), with 2 virtual coils, took 18 minutes. With 12 virtual coils, which a 32-channel scan gets by default, it would take about 1–1.5 hours (CG scales with the coils; the estimation stages mostly do not), and 2 mm or 1 mm reconstructions far longer. Experiment 1a of [example-requirements-design.md](example-requirements-design.md) (section 7) runs `sense` at 1 mm, without the off-resonance term, on about 15 designs (A–M of the sample-spreading report, P1, P4), with and without noise. Where does the time and memory go, and what can be cut without changing the images?

## Short answer

Almost all the time (97%) is in FINUFFT, and it runs at a quarter of its speed because the calls are small. In order:

1. **Measure** (section 2): stage timings and peak memory in every output, and a fixed benchmark.
2. **Exact plumbing** (section 3.1): larger NUFFT calls, reused FINUFFT plans, segments × coils in one call. Same images, about 2–4× faster.
3. **Cheaper estimation** (section 3.2): coil maps and the field map on a coarse grid, one density compensation shared by all τ windows, and the number of off-resonance segments chosen from the field map. Removes about a third of the run time and most of the CG cost for a well-shimmed object.
4. **Toeplitz CG** (section 3.3): the normal operator as FFTs on a 2n grid, with no NUFFT per iteration. CG drops from minutes to seconds at 4 mm. Without the off-resonance term, as in experiment 1a, it needs a single kernel (1.1 GB at 1 mm). Memory-heavy with many segments at fine resolution.
5. **Looser NUFFT accuracy** (section 3.4), and **memory trims** (section 4).

## 1. Baseline

`main` at `574cfaa` (PR #21), before the leak fix of PR #22. Design P4: 4 mm, FOV 256 mm (64³), TR 40 ms, 10000 ADC samples, 1500 TRs, 128 shapes (58–62 echoes per TR). Simulated scan (an ad hoc script, before `relaxed-rosettes simulate` existed): a body blob and a marker blob, 4 coils with correlated noise, uniform 20 Hz off-resonance; recon with 2 virtual coils and the other recon defaults (8 segments, 12 B0 windows, 10 DCF iterations, 20 CG iterations, `eps` 1e-4, `upsampfac` 2). Apple M1 Max, 10 cores.

**Measured, whole scan (15.0M samples):**

| step | time | result |
|---|---|---|
| `to-ismrmrd` | 8 s | |
| `grid` | 84 s | correlation 0.999 with the phantom; 4.6% error after a scale fit |
| `sense` | 1058 s | correlation 1.000; 1.0% error; field map 20.17 Hz (true 20) |

`sense` ran at about 190% CPU on 10 cores, with a peak footprint of 3.8 GB, of which about 1.7 GB was the h5py read leak (section 5, since fixed).

**Measured, profile of `sense` on the first 300 TRs (3.0M samples), 213 s:**

| stage | time | share |
|---|---|---|
| CG (20 iterations × 8 segments × forward and adjoint) | 120 s | 57% |
| field map (12 τ windows, each with its own density compensation and gridding) | 77 s | 36% |
| coil maps (one density compensation and gridding) | 7 s | 3% |
| density compensation of all the data, for CG | ~7 s | 3% |
| reading, prewhitening, compression, caching | < 1 s | |

FINUFFT was 206 s of the 213 s: 1005 type-1 and 942 type-2 calls, about 1.1M points each, at 10M point-transforms per second. A single 2M-point call on the same machine runs at about 40M per second, so the small calls cost about 4×.

The field-map stage is not limited by points: each window holds few. It runs 12 windows × 11 density-compensation passes × 2 transforms on the 2n³ = 128³ grid (256³ after FINUFFT's upsampling), so its FFTs most likely dominate (inferred from the call counts, not measured separately; step 0 should split FFT from spreading time). The 8 mm data do not need that grid.

The CG cost per iteration is 2 × segments × virtual coils × samples point-transforms. It grows linearly with each.

## 2. Step 0: measure

- Record per-stage wall time and peak memory (`resource.getrusage` peak RSS, and the footprint where available) in the `.npz` of `grid` and `sense`, and print them.
- A benchmark: P4 and P1, built from `main`, with `relaxed-rosettes simulate` of a fixed analytic phantom (with off-resonance, for the B0 stages), and, once it exists, the real phantom scan. Record the commit, machine, phantom, and recon config with each run. Re-measure the baseline there first: the baseline of section 1 predates the leak fix and `simulate`.
- Acceptance for every later step: exact steps change the image by less than 1e-4 relative (NUFFT tolerance); approximate steps by less than 1% relative, with the field map within 0.5 Hz in the object, on both the simulation and the real scan.

## 3. Runtime

### 3.1 Exact plumbing (expected 2–4×)

- **Larger calls.** Concatenate the cached batches into chunks of about 10M points for every NUFFT, instead of one call per 128-TR batch. FINUFFT's throughput at 1.1M points per call is a quarter of its large-call throughput.
- **Reuse plans.** The points of a chunk never change during CG, density compensation, or gridding. Keep a type-1 and a type-2 `finufft.Plan` per chunk and call `setpts` once, so the spreading sort and setup are done once, not once per call. Memory: the sorted indices (8 B per point) and the upsampled grid per transform; at 1 mm and 12 coils the upsampled grid is about 1 GB per transform, so limit the number of transforms per call there.
- **One call per segment set.** In the forward and adjoint with off-resonance, stack the segments × virtual coils into one many-vector call (`n_trans = segments × coils`), or at least the coils of all segments for one chunk.
- **Threads.** With the above, the 10 cores should be busy; today `sense` averages about 2.

### 3.2 Cheaper estimation (expected to remove most of the field-map stage)

- **Coarse grids.** Grid the coil-map and field-map data on the grid of their own resolution (8 mm: 32³, with density compensation on 64³), then interpolate the maps to the reconstruction grid (the maps are smooth). At 1 mm this replaces 512³ FFTs with 64³ ones.
- **One density compensation for all τ windows.** Compute it once for the low-resolution data and use each window's subset of the weights. Only the phase of each window's image enters the field map, so a per-window scale does not matter; check that the shape error does not bias the map (acceptance above).
- **Segments from the field map.** Choose the smallest number of segments whose segmentation error is below a tolerance (e.g. 1e-3 RMS), instead of a fixed 8. The number needed grows with (frequency range × readout length): the simulated phantom (20–27 Hz over 32 ms) needs 1–2, not 8, which divides the CG cost by 4–8. Keep `b0.segments` as a maximum.
- **Fewer density-compensation iterations for the low-resolution images** (e.g. 5): they only feed the coil maps and field map.

### 3.3 Toeplitz normal operator (expected: CG from minutes to seconds at 4 mm)

`Eᴴ W E` is a convolution. Without off-resonance, its kernel on a 2n³ grid is one type-1 NUFFT of the weights, computed once. Each CG iteration then costs one forward and one inverse FFT of the 2n³ grid per virtual coil, and no pass over the samples.

With L segments, `Eᴴ W E = sum over (l, l') of P_l* T_ll' P_l'`, where `P_l` applies the coil maps and `exp(2πi f tau_l)`, and `T_ll'` is the convolution with the kernel of the weights times `conj(b_l(tau)) b_l'(tau)`. There are L(L+1)/2 kernels (Hermitian pairs), each one type-1 NUFFT to build, each a complex 2n³ grid in memory:

| resolution | 2n³ grid | one kernel | kernels at L = 8 (36) | at L = 3 (6) |
|---|---|---|---|---|
| 4 mm | 128³ | 16 MB | 0.6 GB | 0.1 GB |
| 2 mm | 256³ | 134 MB | 4.8 GB | 0.8 GB |
| 1 mm | 512³ | 1.1 GB | 39 GB | 6.4 GB |

So: Toeplitz without off-resonance or with the few segments that section 3.2 should give; NUFFT CG (with the plumbing of 3.1) as the fallback when the kernels do not fit. The right-hand side `Eᴴ W y` still needs one pass of adjoint NUFFTs.

### 3.4 NUFFT accuracy (expected 1.5–2× on what remains)

`eps` 1e-4 → 1e-3 narrows FINUFFT's spreading kernel (about 5 → 4 points per dimension), which cuts spreading work by about 2×; `upsampfac` 1.25 shrinks the upsampled grid by 4× in volume at a wider kernel. Both are config values; measure them on the benchmark and change the defaults only if the images pass the acceptance test. CG iterates to a tolerance and the density compensation is approximate, so neither needs 1e-4.

## 4. Memory

- **The read leak** (section 5): fixed in PR #22.
- **Cache.** It stores, per selected sample, 8 B per virtual coil of data plus k (12 B), τ (4 B), and the sample index (4 B). τ follows from the sample index, and k from the TR's shape, rotation, and sample index (one 3 × 3 product per point, per batch). Keeping only the index saves 16 of the 20 B of overhead: about 1 GB at 1 mm. With plan reuse (section 3.1) the plans hold the sorted points instead.
- **Off-resonance phase images.** `exp(2πi f tau_l)` is stored for every segment (L complex n³ images, about 1 GB at 1 mm and L = 8). Compute each on the fly.
- **Virtual coils from the singular values.** Keep the coils that hold, e.g., 99% of the whitened data's energy (with `coils.virtual_coils` as a maximum). Every cost scales with the number kept; 6–8 instead of 12 is common for head coils.
- **Coil maps on the coarse grid** (section 3.2), interpolated per use, if the full-grid maps (12 × n³ complex: 1.6 GB at 1 mm) are a problem.

## 5. The h5py read leak (fixed)

The baseline measurement found it. `RawData.read` and `load_rawdata` read with `.fields(...)`. In h5py 3.16 (HDF5 2.0), a read of some fields of a compound with variable-length members never frees the buffers of the variable-length fields that were not requested (`dset_rw` in `h5py/_proxy.templ.pyx` reads with the dataset's full type and reclaims only on write). Our files have two such fields, `traj` and `data`, so each read pass leaked about one copy of the raw data. PR #22 reads whole records instead. An upstream report and fix in h5py are separate.

## 6. Not in this plan

- A CUDA backend (CuPy and cuFINUFFT behind the `Nufft` interface).
- Changes to the reconstruction model (regularization, the 5D basis).

## 7. Measurements

Record each step here: commit, dataset, recon config, stage times, peak memory, and the image and field-map change against the previous step.

| step | dataset | `grid` | `sense` | peak memory | image change | field map change |
|---|---|---|---|---|---|---|
| baseline (`574cfaa`, before PR #22) | P4 simulation (ad hoc script), 2 virtual coils | 84 s | 1058 s | 3.8 GB (1.7 GB of it the read leak) | — | — |
