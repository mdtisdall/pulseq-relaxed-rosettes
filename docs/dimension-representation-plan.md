# How the reconstruction represents τ and T: plan of a chapter series

Status: plan, started 2026-10-16. Nothing in the series is written yet. Record each decision in section 7 with its date.

## Purpose

The design work is in three gated stages: (a) before the scan, choose the design to test; (b) before the scan, check that the chosen design is implemented correctly; (c) after the scan, check that the measured data match the expectations, and decide the next steps. This series serves stage (a). Part of its purpose is to show which on-scanner tests are the simplest and most informative to run first.

A requirement can leave a dimension out of its output, or resolve it ([coverage-target.md](coverage-target.md), section 1). How the reconstruction handles each dimension sets what the trajectory must do. The existing documents handle dimensions in different ways: [example-requirements-design.md](example-requirements-design.md) averages, by a model of rank 1 or a fixed basis; [quadratic-norm-report.md](quadratic-norm-report.md) takes a 2-norm; [basis-quality-test-plan.md](basis-quality-test-plan.md) fits a low-rank model of the signal in T, and [b0-scan-time-plan.md](b0-scan-time-plan.md) a low-rank model of the field in T. Before the design objective is fixed, the series develops each way on an equal footing, and then compares and combines them.

## 1. Two questions about each dimension

Every way of handling a dimension answers two separate questions:

1. **Representation:** what the reconstruction estimates along the dimension. It is always a basis of some rank:
   - **rank 1, fixed:** the constant;
   - **rank L, fixed:** a small basis chosen before the reconstruction, for example the τ basis of decays (`relaxed_rosettes.recon.basis`);
   - **dense, fixed:** a basis rich enough for the content of the signal, for example splines or Slepian sequences in T;
   - **rank K, fitted:** components fitted to the data, $x = \sum_k u_k(r, \tau)\, v_k(T)$. They couple T with space and τ, so this representation does not act on one dimension alone.
2. **Output:** what the requirement reports about the dimension:
   - **coefficients:** the estimate along the dimension itself (resolved);
   - **linear:** a linear functional of it, for example the mean;
   - **quadratic:** a quadratic form of it, for example a 2-norm, a band power, or a cross-spectrum.

Nothing here is free of a model. One TR is undersampled by a factor of hundreds (about 630 at 1 mm, [quadratic-norm-report.md](quadratic-norm-report.md), section 8), so every representation assumes something: a rank, smoothness in T, or a band limit. The distinctions that matter are fixed against fitted, and sparse against dense relative to the signal.

### 1.1 Signal dimensions and operator components

The two questions apply in full only to the signal dimensions. The forward operator (the trajectory, the coils, the B0 phase, and the NUFFT) has components of two other kinds.

| kind | examples | role in the series |
|---|---|---|
| signal dimensions | τ, T | the object's signal changes along them, and what a requirement reports about them is a scientific choice; they get cells (section 2) |
| nuisance operator components | the coils; B0 over T | parameters of the forward model, not dimensions of the object; the reconstruction estimates or collapses them, and every cell carries them |
| design operator component | the trajectory | the part of the operator that we choose; the design pressures act on its parameters (section 1.2) |

Who controls each operator component, and when:

| component | chosen by | when | known or estimated |
|---|---|---|---|
| the trajectory | us | before the scan, for each design | known, up to the gradient impulse response |
| the coils | the coil designer; we choose among the available coils | before the scan | estimated for each subject |
| B0 over T | the subject (shimming in part) | during the scan | estimated |

**The coils.** They fit the grid formally. Coil $v$ measures $A(s_v x)$, so the coil images are $c_v(r) = s_v(r)\, x(r)$:

- **fitted (SENSE):** a rank-1 representation over the coils at each voxel, with the basis $s(r)$ fitted before the reconstruction (calibration, ESPIRiT), and the single coefficient $x$ as the output;
- **collapsed:** a dense representation, one free image for each coil, with a quadratic output (root-sum-of-squares), as in [quadratic-norm-report.md](quadratic-norm-report.md), section 4.1.

But they differ from τ and T in kind:

1. **No requirement wants a coil-resolved output.** The coils are always removed, so the output question is empty and only the representation is a choice. The rank-1 structure over the coils is exact: the only error is the error of the coil maps, not the model error of experiment 1a.
2. **The choice changes the operator, so it acts on every cell.** Fitted coils add encoding and lower ρ by about the usable acceleration $R(\text{res})$; collapsed coils multiply the unknowns by the number of coils and add no encoding. So the choice scales every pressure, and adds few pressures of its own on the shapes and the schedule.
3. **Coils hardly interact with the schedule.** The maps are the same for every sample. The exception is their estimation (calibration, ESPIRiT), which needs coverage near the centre of k-space.
4. **Collapsed coils make the final output quadratic**, even where the output over τ and T is linear: a static image combined by root-sum-of-squares is linear over τ and T and quadratic over the coils. It then needs the removal of bias of chapter 2 (outputs), whatever cell handles τ and T.

**B0 over T** is a nuisance operator component too. It changes the phase of the encoding over T, and [b0-scan-time-plan.md](b0-scan-time-plan.md) fits it as a few modes: a low-rank model of the operator, not of the signal.

Motion is out of scope (section 7, decisions). With motion, the coil maps and the field change over T relative to the object, and the nuisance operator components would interact with the schedule.

### 1.2 The trajectory as an operator component

The trajectory already has a representation: a library of shapes, a rotation for each TR, and a schedule.

- **The shapes** are a basis over $(|k|, \tau)$. Their number is the rank of that basis: the question of how many shapes are enough ([example-requirements-design.md](example-requirements-design.md), section 6.1) is a choice of rank, like the choice of $L$ or $K$.
- **The rotations** cover the directions of k.
- **The schedule** says which shape and which rotation each TR uses, so it sets the timing in T.
- **The split between shapes and schedule is a factorization.** A rotation preserves $|k|$, τ, and the slew energy of every sample ([example-requirements-design.md](example-requirements-design.md), section 4), so the shapes set radius, τ and cost, and the schedule sets direction and timing. The roles of the shapes (same document, section 6.1) are further structure on that factorization.
- **The measured trajectory** (from a gradient impulse response) is the estimated part, as the coil maps are for the coils. Its error is a question for stage (b), not for this series.

**The trajectory is fixed before the object is seen.** The shapes, the rotations and the schedule are designed offline and do not depend on the subject or on the data. The form of the representation (shapes, rotations, schedule) is fixed for the series; its values are what the design chooses. Optimization during the scan, from the data acquired so far, is out of scope, like motion (section 7, decisions).

**What this does to the pressures.** A pressure is the change of a cell's error (a bias, a variance, a model error) along a direction of the trajectory's representation, within the constraints that limit that direction (slew, PNS, spoiling). The vocabulary of the pressures (section 4) is the list of those directions.

**Spoiling** belongs here: the spurious echoes are terms of the forward model that depend on the trajectory, through the spoil moment of each shape and the rotation of each TR ([example-requirements-design.md](example-requirements-design.md), section 5.1).

### 1.3 The example requirements, and their names

The series uses the four example requirements of [example-requirements-design.md](example-requirements-design.md), section 1: the static image, relaxation 4D, BOLD, and the requirement those documents call the "navigator". "Navigator" has several meanings in MRI and in this repository, so the series calls that requirement the **fast coarse image**:

- **What it is:** a reconstruction from the ordinary rosette data, not a separate readout. Space fixed at about 8 mm (a ball of about 17 000 cells), τ averaged, T at the shortest width (about 0.15 s, 3 TRs; a placeholder).
- **What it is for:** changes over T that are fast and coarse in space: the field (B0 over T) and physiology, and a source of the temporal basis for the fitted low-rank representation. It also contains information on motion, although motion is out of scope here (section 7, decisions).
- **What it does to the design:** it anchors the coarse, short end of the τ-averaged resolution–time curve, and it sets the main pressure on each single shape: a shape must cover the coarse ball almost by itself, in all directions ([example-requirements-design.md](example-requirements-design.md), section 4).

It is not the same as the "navigators" of [basis-quality-test-plan.md](basis-quality-test-plan.md): the samples at the k = 0 crossings (candidate A), the gridded windows of TRs (candidate B), and a dedicated segment (candidate C). Those are estimators of a temporal basis or of the field. [b0-scan-time-plan.md](b0-scan-time-plan.md) uses the second of them.

## 2. The cells

A **cell** is one representation with one output. Each cell has a name and a chapter (section 3); the series refers to a cell by its chapter.

Cells are general across dimensions. Each cell chapter says which signal dimensions it applies to (τ, T), and the synthesis assigns a cell to each dimension of each requirement.

The set of cells holds the comparisons that the synthesis needs: chapters 8 and 9 differ only in the representation, and chapters 5 and 8 mostly in the output. Chapter 9 is the only cell that no current requirement uses: it is there to show the effect of an imposed structure (shared time courses, a lost lag) on a quadratic output, against chapter 8.

Cells also combine in sequence. The quadratic forms (chapter 8) need the τ-resolved static image (chapter 4 over τ, chapter 3 over T) to be subtracted before they estimate a band power: subtract what a model explains, then take the quadratic form of what remains.

## 3. The chapters

One document for each chapter, numbered in reading order. File names carry the number and the name, for example `03-rank1-average.md`, so that they sort in that order.

| chapter | name | content |
|---|---|---|
| 0 | Foundations | the grid of section 1 (representation × output; signal dimensions and operator components), the notation, the shared toy, the template of the cell chapters |
| 1 | Representations | primer for the representation axis: the conditioning of a basis-weighted operator, fixed against fitted bases, rank against density, coupling with space; the nuisance operator components (section 1.1): coils fitted or collapsed, B0 over T as modes; and the trajectory's representation (section 1.2): the shapes as a basis over $(|k|, \tau)$ and their number as its rank, the rotations, the schedule, the factorization between shapes and schedule, the roles |
| 2 | Outputs | primer for the output axis: linear against quadratic functionals, why errors that cancel in a linear output add up in a quadratic one, and how products between independent sets remove that bias |
| 3 | Rank 1 average | rank 1, fixed × coefficient (truncate and average). Used by: the static image and the fast coarse image over τ; the static image and relaxation 4D over T. Now: [example-requirements-design.md](example-requirements-design.md), section 3.1; measured in 3D in experiment 1a |
| 4 | Fixed basis, resolved | rank L, fixed × coefficients. Used by: relaxation 4D and BOLD over τ. Now: [example-requirements-design.md](example-requirements-design.md), section 3.1 |
| 5 | Fixed basis, then average | rank L, fixed × linear (resolve, then average). Used by: an alternative for the static image. Now: one sentence in [example-requirements-design.md](example-requirements-design.md), section 3.1; experiment 6, not run |
| 6 | Splines in T | dense, fixed (splines) × coefficients. Used by: the fast coarse image and BOLD over T. Now: [example-requirements-design.md](example-requirements-design.md), section 3.1 |
| 7 | Fitted low rank | rank K, fitted × coefficients. Used by: BOLD time courses. Now: [basis-quality-test-plan.md](basis-quality-test-plan.md) |
| 8 | Quadratic forms | dense, fixed (decay basis and Slepian) × quadratic. Used by: the magnitude-like image, band power, cross-spectrum. Now: [quadratic-norm-report.md](quadratic-norm-report.md) |
| 9 | Fitted low rank, quadratic | rank K, fitted × quadratic (optional, section 7). Used by: no current requirement; the comparison with chapter 8. Now: none |
| 10 | Synthesis | the cells combined (section 5) |
| 11 | Checks | the checks and gates of stage (a) (section 5) |

**Dependencies:** 0 → 1 and 2 → 3–9 → 10 → 11. The toy of chapter 0 goes through several rounds of review and refinement before any other chapter starts, because every chapter uses it. The cell chapters (3–9) do not depend on each other, so they can be written in parallel, on separate branches. Each chapter goes to `main` on its own branch and PR.

**Order of writing the cell chapters:** chapter 3 first, because it has 3D data (experiment 1a) to anchor the toy against; then chapter 8, whose theory exists; then the others.

**Existing material moves:**

- [quadratic-norm-report.md](quadratic-norm-report.md) splits three ways: its theory of the basis into chapter 1, its estimator (products between sets) into chapter 2, its applications into chapter 8.
- [example-requirements-design.md](example-requirements-design.md) keeps the decisions and the measurements. The theory of the rank 1 average (frame potential, the τ-averaging mismatch) moves to chapter 3, and the documents link to each other, so that the same content is not in two places.
- [example-requirements-design.md](example-requirements-design.md) also keeps the design machinery (the greedy schedule, the shape solve, roles in practice). The theory of the trajectory's representation (its sections 4 and 6.1) is stated in chapter 1, with links both ways.
- [coverage-target.md](coverage-target.md) gets any new kind of requirement from chapter 10.

## 4. Rules for the cell chapters (3–9)

**Template.** Each cell chapter has these sections, in this order: the dimensions it applies to; the assumptions about the signal; the estimator; its bias; its variance; its cost; its failure modes; the toy; the design pressures.

**Short, and no repetition of the primers.** A cell chapter contains only what comes from the combination of its representation and its output. If it starts to derive something that holds for every cell with the same representation, or the same output, that material belongs in chapter 1 or 2.

**General theory first.** The theory is stated for $x(r, \tau, T)$ in 5D, with samples off the grid, rotations, and the library of shapes. The toy is a special case of it.

**The toy explains; it does not predict.**

- All chapters use the same 1D toy signal model, which chapter 0 defines.
- The toy shows mechanisms, for example model error, leakage and its removal by products, a phase drift, shared time courses, a lost lag, and the correlation with a task stimulus. A toy result is not a prediction of the effect on a rosette.
- Some mechanisms have no 1D form. A rotation in 1D is only a sign flip, so angular fill, roles across directions, and the placement of spoiled magnetization cannot be shown. Where the toy cannot show a mechanism, the chapter says so.
- 1D can make an effect larger or smaller than it is in 3D. In 1D the DFT makes every k its own problem; in 3D the point-spread functions of off-grid samples overlap, and the aliasing is less white.

**Design pressures, in 5D.** Each cell chapter ends with the pressures that its cell puts on the trajectory:

- They are for the 5D case and are derived from the chapter's general theory. The toy can be referred to as an illustration, not as evidence.
- Each pressure comes from a stated expression in the theory (a bias term, a variance term, a conditioning measure). The pressure section says which part of that expression the shapes control and which part the schedule controls. A rotation preserves $|k|$, τ, and the slew energy of every sample ([example-requirements-design.md](example-requirements-design.md), section 4).
- All chapters use one vocabulary: the directions of the trajectory's representation (section 1.2).
  - each shape: the $(|k|, \tau)$ distribution, the coverage of one TR ($\rho(\text{res})$), the spoil moment;
  - the library: the number of shapes, completeness, no correlation between $|k|$ and τ, roles;
  - the schedule: angular fill, the k–T spectrum, independence across TRs, early stopping, compatibility with a split of the TRs into sets, the directions of the spoil moments.
- Each pressure names the direction it acts on and the constraint that limits that direction: slew energy, PNS, or spoiling.
- Each pressure is stated in a form that can be checked: the quantity, the direction in which it pushes the design, and how strongly, according to the theory.
- Each pressure has a status: derived; derived under a named assumption; or a hypothesis.
- The pressures are stated with the coils fitted (SENSE). Each chapter says how they scale with $R(\text{res})$, and does not repeat its theory for collapsed coils.
- The cell chapters do not plan checks. Chapter 11 does.

The axes help to attribute the pressures. Pressures on conditioning come mainly from the representation (a frame potential whose kernel is the basis). Pressures on model error come from a rank lower than the signal's, as the output sees it: the τ-averaging mismatch $E_\tau$ of experiment 1a belongs to chapter 3. Pressures for independence across TRs and for compatible splits come from the quadratic output.

**Depth of the variance theory.** The variance and the conditioning rank designs, and they are the least developed part for every cell. Each chapter develops bias and variance far enough to rank designs, and lists what remains open.

**Known thin places.**

- Chapter 3: the frame potential has a theoretical basis. The τ-averaging mismatch $E_\tau$ is a heuristic that ranked the error of experiment 1a well. The chapter derives it from the model error of a rank-1 representation, or labels it empirical.
- Chapters 7 and 9: the reconstruction is nonlinear. Its theory (incoherence and recovery guarantees) gives sufficient conditions, which are usually loose, so their pressures can be qualitative. The chapters say so where that applies.
- Chapter 8: the hypothesis that the variance of the products reduces to a frame potential with the band as its kernel ([quadratic-norm-report.md](quadratic-norm-report.md), section 11.3). If it holds, chapters 3 and 8 share one form of pressure.

## 5. The synthesis and the checks (chapters 10 and 11)

**Chapter 10, synthesis.**

- Combine the pressures of the cell chapters into one table of cells × pressures, and find where they conflict. Two conflicts are already known: greedy (herded) schedules against the independence across TRs that the quadratic forms (chapter 8) need, and cycled roles against splits into interleaved halves.
- Treat the combinations of cells across dimensions, including those that do not factor (chapters 7 and 9, which couple T with space), and sequences (subtract a model, then take a quadratic form of the residual).
- Treat collapsed coils as a final quadratic step on top of any cell (section 1.1, item 4).
- Restate the example requirements (fast coarse image, relaxation 4D, static, BOLD) as a cell for each dimension, with any new kind of requirement ([coverage-target.md](coverage-target.md), section 9).

**Chapter 11, checks.**

- For each pressure that chapter 10 keeps, a check of its size on the real trajectory, at the cheapest tier that can answer it ([example-requirements-design.md](example-requirements-design.md), section 7.3; normally Tier 1).
- A measurement of $R(\text{res})$, by which the coils scale every pressure: the coil-aware kernel that [example-requirements-design.md](example-requirements-design.md) (section 3.2, caveats) defers.
- Checks of the trajectory's representation itself: how many shapes are enough, and whether roles help ([example-requirements-design.md](example-requirements-design.md), experiments 3 and 5).
- The gates of stage (a): which results select or reject a design.
- Results that contradict a chapter go back into that chapter.

## 6. Writing the cell chapters in parallel

Once chapters 0, 1 and 2 are merged, the cell chapters (3–9) can be written at the same time, on separate branches or by separate agents, one chapter each. The risk is drift in notation and in depth. Chapter 0 (notation, template, toy) and one reviewer for all cell chapters control it.

## 7. Open decisions

Settle these in chapter 0. The numbers of the chapters are final only when the list of cells is: decision 1 can add a cell, and decision 2 can remove one.

1. **The toy: details.** The starting point is decided (see the decisions below): the running example of [quadratic-norm-report.md](quadratic-norm-report.md) (section 1.1), with task-based BOLD added. Open within it:
   - **The task.** The stimulus time series (for example a block or event design), the response model (a fixed response function, or a basis with its derivatives), and how lags enter: one lag for each region, or a lag that the analysis estimates.
   - **Where task BOLD falls in the cells.** The correlation of a voxel with a known stimulus has a numerator that is linear in the signal: a projection onto the stimulus, or a fit with the stimulus as a fixed basis over T. Its normalization by the voxel's power is quadratic. So task BOLD may need a cell with a linear output over T, which section 3 does not have yet (it would go between chapters 6 and 7), or a fixed basis over T in chapter 4.
   - **The mechanism of experiment 1a.** Whether to add a design parameter that shows it (a local, signal-weighted τ mismatch near the centre of k-space), for example a relaxed and a hairpin petal family.
2. **Chapter 9.** Whether to write it. No current requirement uses it.
3. **The end point of chapter 10.** Whether the synthesis stops at the table of pressures, as directions of the trajectory's representation, or goes on to a design objective. An objective would overlap with the greedy schedule and the shape solve of [example-requirements-design.md](example-requirements-design.md), sections 5 and 6.
4. **Location.** Where the chapters go, for example `docs/representation/`.

| date | decision |
|---|---|
| 2026-10-16 | Frame the series by two questions about each dimension, its representation and its output, not by ways to collapse it. |
| 2026-10-16 | One chapter for each cell in use, with the theory of each axis in two primers, then a synthesis chapter and a separate checks chapter. |
| 2026-10-16 | The 1D toy explains mechanisms and does not predict effects on a rosette. |
| 2026-10-16 | The toy starts from the running example of [quadratic-norm-report.md](quadratic-norm-report.md) and adds task-based BOLD: the correlation, lags included, of each voxel with a given stimulus time series. It goes through several rounds of review and refinement before the other chapters start. |
| 2026-10-16 | The design pressures are for 5D, derived from theory. They are checked in chapter 11, not in the cell chapters. |
| 2026-10-16 | τ and T are signal dimensions and get cells. The coils and B0 over T are nuisance operator components: properties of the forward model that every cell carries, treated in chapter 1, with the pressures stated for fitted coils (section 1.1). |
| 2026-10-16 | The requirement called "navigator" in [example-requirements-design.md](example-requirements-design.md) stays, and the series calls it the fast coarse image, with its purpose stated (section 1.3). |
| 2026-10-16 | Motion is out of scope. The series is to inform the simplest and most informative first on-scanner tests, and motion is unlikely to help with that. |
| 2026-10-16 | The chapters are numbered 0–11 in reading order. Cells are named, and referred to by their chapter, not by numbers of their own (section 3). |
| 2026-10-16 | The trajectory is the design operator component. Its representation (shapes, rotations, schedule) is treated in chapter 1, and the pressures are directions of it (section 1.2). The form of the representation is fixed for the series, and its values are designed before the object is seen. Optimization during the scan is out of scope, a later exploration like motion. |
