# A reference of theory for sampling and reconstruction design: plan of a document series

Status: plan, started 2026-10-16. Nothing in the series is written yet. Record each decision in section 11 with its date.

## Purpose

The series is a reference of theory. Later work on the design of a sampling scheme, or of a reconstruction, can rely on it, and come back to it for each new problem.

- **It centralizes existing theory by citing it.** It does not derive again what a source already gives. Each result is stated, with its source, its assumptions, and its notation mapped to the series' notation (section 5).
- **It is separate from design.** It does not choose a design, a reconstruction, or an application. Design documents (for example [example-requirements-design.md](example-requirements-design.md)) cite results from the reference; the reference does not make design decisions.
- **It is abstract, and its examples are concrete.** The theory is stated for a signal over any set of dimensions (section 1). A 2D toy (x–t: one dimension of space, and time) and a 5D problem from this project illustrate it (section 4). They explain the theory; they are not its goal.

## The framework

The series states its theory in the framework of Bayesian (optimal) experimental design:

| element | what it is |
|---|---|
| **design variables** | what is chosen before the measurement: in the abstract, the parameters of the joint distribution of the sample coordinates (section 1) |
| **priors** | what is known before the measurement: distributions over every unknown (section 1): the signal and its changes, the unknown parameters of the operator, and the noise |
| **outputs and loss** | what is wanted from the data, and how the error of an estimate of it is measured |
| **estimator** | what is done with the data: a reconstruction, which represents the signal in some way and reports the outputs (section 2) |
| **expected loss** | the loss of the estimator, averaged over the priors and the noise, as a function of the design |

The best design minimizes the expected loss for a given estimator. The reference gives the theory of how the expected loss depends on the design, the priors and the estimator. It does not minimize it for any particular problem: that is design work.

The reference covers families of estimators, not the best possible estimator. A representation sits across two elements: it is a choice of the estimator, and it builds in a prior (a rank, smoothness, a band limit).

## 1. The abstraction

**The signal** is a function over a product of dimensions. The dimensions are alike: the theory does not depend on what a dimension stands for. The examples (section 4) give concrete ones.

**The measurement.** Each sample is a linear functional of the signal, plus noise. In the simplest case, a sample is a point evaluation in a chosen domain of each dimension: the signal domain, or its conjugate. Which domain is sampled is a property of the measurement of each dimension, not of the dimension: a point evaluation in one domain is an integral against a complex exponential in the other. Measurements that are not point evaluations in either domain are other linear functionals in the same theory. Section 4.2 writes the MRI case out.

**The coordinate distribution.** The design sets the joint distribution of the samples' coordinates. The results of the theory depend on its structure:

- how the coordinates are distributed in each dimension;
- whether they are independent across groups of samples (for example, across the repetitions of a measurement);
- whether they are separable across dimensions;
- how they are linked within one sample (for example, one coordinate that is a function of another, or two coordinates that map onto one physical variable).

**What is known and what is unknown.** The forward model has:

- **the design:** the sampling, known and chosen; the design variables parameterize its coordinate distribution;
- **the unknowns:** the signal, and the unknown parameters of the operator. Each unknown is classed by how it enters the forward model:
  - **linearly**, as the signal does: the measurement is linear in it, which the grid and most of the cited theory assume;
  - **bilinearly**, as a parameter that multiplies the signal;
  - **nonlinearly**, for example through a phase.

  How an unknown enters affects the estimators, their conditioning, and whether a joint estimate is a convex problem. The grid applies to any unknown that has dimensions, not only to the signal.

**Nuisance parameters are relative to an output.** For a given output, the nuisance parameters are the unknowns it does not ask for. The same unknown can be an output in one problem and a nuisance parameter in another, so the reference does not label any unknown as nuisance in advance. An estimator handles each nuisance parameter in one of three ways: it estimates it (profiles it), collapses it with a representation from the grid, or averages over its prior (marginalizes it).

## 2. The grid

Every way of handling a dimension answers two questions.

1. **Representation:** what the estimator estimates along the dimension. It is always a basis of some rank:
   - **fixed, rank 1:** the constant;
   - **fixed, rank L:** a small basis chosen before the reconstruction;
   - **fixed, dense:** a basis rich enough for the content of the signal;
   - **fitted, rank K:** components fitted to the data. A fitted representation can couple the dimension with others, so it does not always act on one dimension alone.
2. **Output:** what is reported about the dimension:
   - **coefficients:** the estimate along the dimension itself;
   - **linear:** a linear functional of it, for example the mean, or the projection onto a given time series;
   - **quadratic:** a quadratic form of it, for example a 2-norm, the power in a band, or a cross-spectrum.

A **cell** is one representation with one output, for an abstract dimension. Whether a cell suits a particular dimension of a particular problem is a question for that problem. The grid has 12 pairs. Some are degenerate: with a rank-1 representation, the coefficient and a linear output are the same up to a scale. Which pairs share a chapter is an open decision (section 10).

**Combinations.** A real estimator uses one cell for each dimension, and cells combine:

- **across dimensions:** a cell for each dimension, which factors unless a fitted representation couples them;
- **in sequence:** subtract what one estimate explains, then estimate an output of the residual with another cell. For example, a quadratic output in one dimension can need the subtraction of a resolved estimate first.

Nothing here is free of a model. When each repetition of the measurement samples far less than the representation needs, every representation assumes something: a rank, smoothness, or a band limit. The distinctions that matter are fixed against fitted, and the rank against the content of the signal.

## 3. Pressures and conditions

- **The formal quantity** is the change of the expected loss with the coordinate distribution, for a given estimator and priors.
- **A pressure** is an informal claim about that quantity: that the expected loss changes when the distribution changes in a named way (for example, more independence across groups of samples, or less correlation between two coordinates), with a sign, a rough strength, and a status. Pressures are soft: they can be traded against each other. The reference does not combine them into a design objective; that is design work. The word is meant in the sense of selection pressure: a tendency, not a force or a gradient.
- **A condition** is hard: where it fails, the estimator is invalid, not only worse. A condition is a property of the pair (design, estimator), because some can be met by a choice of the estimator after the measurement. For example, products between independent sets of samples need each set to hold the whole structure that the representation needs; that depends on how the samples are split, as well as on the design.

Conditions filter designs; pressures rank the designs that remain.

Each pressure and each condition states:

- its status: cited, adapted, or new (section 5);
- the priors it depends on;
- which properties of the coordinate distribution it concerns;
- how it changes with the unknown parameters of the operator (for example, with the conditioning that they add to the measurement).

## 4. The examples

Two examples make the abstraction concrete. Each states its signal model, the unknown parameters of its operator, its priors, its design variables and its noise explicitly. The outputs are not fixed in advance: each cell chapter chooses, from each example, the outputs and the loss that best illustrate its theory. A cell can be a poor fit for an example; that is part of what the example shows. A claim seen only in an example is a hypothesis, not a result of the reference.

### 4.1 The 2D toy, x–t (simulated)

It starts from the running example of [quadratic-norm-report.md](quadratic-norm-report.md), section 1.1, and adds a change locked to a given stimulus time series, with a lag.

- **Design variables:** the analogues of the shapes in one spatial dimension (petals and their tips), their number, the sign of k as the rotation, the schedule, and the number of repetitions.
- **Priors:** a stated distribution over objects and their changes over t, and over the noise. The toy's objects are drawn from it. Which changes it includes is chosen to illustrate the mechanisms of the theory, not to copy an application (open decision 3).
- **Outputs and loss:** chosen by each cell chapter, with its estimator.
- **Expected loss:** averaged over draws from the priors and the noise.

The toy explains and does not predict effects in the 5D problem. With one spatial dimension, some mechanisms have no form: a rotation is only a sign flip. One spatial dimension can also make an effect larger or smaller than three do: the DFT makes every k its own problem.

### 4.2 The 5D rosette problem (described, not simulated)

This project's problem: a relaxed rosette trajectory, with the signal over space, the time since excitation τ, and the scan time T.

- **Design variables:** the trajectory's parameterization: a library of shapes, a rotation for each repetition (TR), and a schedule. The shapes set $(|k|, \tau)$ and the slew energy, which a rotation preserves; the rotations set the direction of k; the schedule sets the timing in T ([example-requirements-design.md](example-requirements-design.md), sections 4 and 6.1). The spoil moment of each shape is a design variable too. TR, flip angle, RF phase scheme, ADC window and dwell, scan length, and the coil are held fixed (the defaults of `relaxed_rosettes.config.Config`, and the coil of the session).
- **Priors:** the object's spectrum, the range of R2\*, the BOLD band and the physiological bands, a task stimulus, the population span of the field over T, the noise and its covariance, the coil geometry.
- **Outputs:** chosen by each cell chapter. Outputs that the project's design documents already name keep their names here. One name is new: the **fast coarse image**, a reconstruction from the ordinary data at about 8 mm, with τ averaged, over a short window of T (about 0.15 s). [example-requirements-design.md](example-requirements-design.md) calls it the "navigator", a name with several other meanings; it is not the navigators of [basis-quality-test-plan.md](basis-quality-test-plan.md).
- **The measurement:** a sample is a point in k for the spatial dimensions, and a point in τ and T for the others. With signal $x(r, \tau, T)$,
  $$y_m = \int x(r, \tau, T)\, e^{-2\pi i\, k_m \cdot r}\, \delta(\tau - \tau_m)\, \delta(T - T_m)\, dr\, d\tau\, dT \;+\; n_m ,$$
  a point evaluation at $(k_m, \tau_m, T_m)$; in the Fourier domain of T, the same sample is an integral against $e^{2\pi i f T_m}$. Within one sample, k is a function of τ (the shape), and τ and T map onto one absolute time $t = T + \tau$. Spatial encoding by the excitation pattern would be a measurement that is not a point evaluation in either domain of space.
- **Unknown parameters of the operator:** the coil sensitivities, which enter bilinearly (often nuisance parameters, either estimated as fitted rank 1, as in SENSE, or collapsed as dense with a quadratic output, as in root-sum-of-squares; the output itself for coil calibration); and the field over T, which enters nonlinearly through a phase (modelled as a few modes in [b0-scan-time-plan.md](b0-scan-time-plan.md); a nuisance parameter for an image, the output itself for a field map).
- **Signal effects:** the decay over τ (R2\*) and its change with BOLD, which make a rank-L representation over τ a natural choice here.
- **MR physics outside the abstraction:** spoiling and the steady state enter the abstract theory only as a model error of the operator; their content is stated here ([example-requirements-design.md](example-requirements-design.md), section 5.1).
- **Where it has been measured:** for each element, the measurements so far, for example the static image in experiment 1a.

The 5D problem is not simulated in the reference. Measuring it is design work, or verification at the cheapest tier (section 6).

## 5. Sources

**Cite, do not derive again.** Each result in the reference has a status:

- **cited:** stated as in its source;
- **adapted:** a known result restated for this setting (for example, multitaper estimation applied to point evaluations in a mixed domain), with its source and the adaptation shown;
- **new:** no source found. It is marked as new, states its nearest precedents, and gets the most scrutiny in verification. For example, [quadratic-norm-report.md](quadratic-norm-report.md) found no precedent for estimating power from products between independently sampled sets of repetitions in undersampled k-space.

**The annotated bibliography** is a document of its own in the series. Each entry has a stable anchor, so that chapters can link to it, and these fields:

- the full citation, with its DOI, checked against the source;
- what the series uses from it, with its location in the source (section, theorem, or equation);
- the assumptions under which the result holds there;
- the notation map: each symbol of the source to the series' symbol;
- the status of the result in the series;
- the chapters that use it.

**The series' notation** is defined once, in chapter 0, as a table of symbols. Every notation map points into it. Results marked new have no entry; the chapter that states them links to the entries of their nearest precedents.

`scripts/check_links.py` checks the file of a relative link, not its anchor, so links to entries are checked only as far as the bibliography's file.

## 6. The documents

One document for each chapter, numbered in reading order. File names carry the number and a short name, for example `00-problem.md`, so that they sort in that order.

| chapter | content |
|---|---|
| 0. The problem | the framework; the abstraction of section 1; the table of the series' notation; the definitions of pressure and condition; the two examples (section 4), each with its signal model, the unknown parameters of its operator, priors, design variables and noise; the outputs are left to the cell chapters |
| 1. Representations | opens with the grid (section 2): both axes, cells, combinations. Then the representation axis: the conditioning of a basis-weighted operator, fixed against fitted bases, rank against the content of the signal, coupling between dimensions; unknowns that enter bilinearly or nonlinearly; how an estimator handles nuisance parameters (estimated, collapsed, or marginalized) |
| 2. Outputs | the output axis, with the grid as chapter 1 states it: coefficients, linear and quadratic outputs; why errors that cancel in a linear output add up in a quadratic one; how products between independent sets of samples remove that bias |
| cells | one chapter for each cell, or for each group of degenerate cells (section 10). Numbered once the groups are settled |
| combination | combinations of cells across dimensions (including fitted representations that couple them) and in sequence |
| verification | numerical checks that each result holds in the toy and, at the cheapest tier that can answer it, in the 5D problem ([example-requirements-design.md](example-requirements-design.md), section 7.3); concentrated on results marked new |
| bibliography | the annotated bibliography (section 5), unnumbered, grown with each chapter |

## 7. Rules for the chapters

**Template.** Each cell chapter has these sections, in this order:

1. its place in the framework: the priors it assumes, and the properties of the coordinate distribution its results concern;
2. for the outputs it chooses from the examples, the nuisance parameters, and how its estimator handles each;
3. the assumptions about the signal;
4. the estimator;
5. its bias;
6. its variance;
7. its cost;
8. its failure modes;
9. its pressures and conditions (section 3);
10. the examples: the toy, then the 5D problem. For each, the chapter chooses the outputs and the loss that best illustrate its theory, and says why.

**Cite, do not derive again.** Each result has a status and, unless it is new, an entry in the bibliography.

**No repetition of the primers.** A cell chapter contains only what comes from the combination of its representation and its output. Material that holds for every cell with the same representation, or the same output, belongs in chapter 1 or 2.

**Abstract first, examples last.** The theory is stated in the abstract, only as generally as the two examples need, and each result is shown in both. The examples do not make design decisions.

**Depth of the variance theory.** The variance is the least developed part for every cell. Each chapter takes bias and variance far enough to state its pressures, and lists what remains open.

**Known thin places**, from the existing documents:

- the frame potential of a rank-1 representation has a theoretical basis. The mismatch between the local and the global distribution of a coordinate along an averaged dimension is a heuristic (in the 5D example, the τ-averaging mismatch of [example-requirements-design.md](example-requirements-design.md), section 3.2, which ranked the error of experiment 1a well), to derive or to label empirical;
- fitted representations give nonlinear estimators, whose theory (incoherence, recovery guarantees) gives sufficient conditions that are usually loose;
- the hypothesis that the variance of products between independent sets reduces to a frame potential with the band as its kernel ([quadratic-norm-report.md](quadratic-norm-report.md), section 11.3).

## 8. Writing order and dependencies

0 → 1 → 2 → the cell chapters → combination → verification. The bibliography grows with each chapter.

- Chapter 0, and its toy, go through several rounds of review and refinement before any other chapter starts.
- Chapter 2 uses the grid as chapter 1 states it, so the primers are written in order.
- The cell chapters do not depend on each other. Once chapters 0–2 are merged, they can be written at the same time, on separate branches. Chapter 0 (notation, examples), chapter 1 (the grid), the template of section 7, and one reviewer for all cell chapters keep them consistent.
- Each chapter goes to `main` on its own branch and PR.

## 9. Relation to the existing documents

- **Design documents keep the design.** [example-requirements-design.md](example-requirements-design.md) keeps the requirements, the decisions, the measurements, and the design machinery (the greedy schedule, the shape solve, roles). [coverage-target.md](coverage-target.md) keeps the form of the requirements. They cite the reference where they use its results.
- **Theory in existing documents.** [quadratic-norm-report.md](quadratic-norm-report.md) is mostly theory, for chapters 1, 2 and the quadratic cells; [example-requirements-design.md](example-requirements-design.md), section 3, has the theory of the frame potential and the τ-averaging mismatch. What becomes of those documents once the reference holds their theory is an open decision (section 10).

## 10. Open decisions

Settle these in chapter 0, unless stated otherwise.

1. **The groups of cells.** Which pairs of the grid share a chapter: for example, the degenerate rank-1 pairs; or the three fixed representations (rank 1, rank L, dense) as one family with the rank as a parameter. This sets the number of cell chapters and the numbering.
2. **Estimator settings.** Whether regularization, the number of iterations, and the split into sets are part of an estimator (so of a cell), or parameters that each result states.
3. **The toy:**
   - which effects of the running example to keep: only those that illustrate a mechanism of the theory. Effects that belong to an application, such as the decay over τ, go only into the 5D example;
   - the task: the stimulus time series, the response model, and how lags enter;
   - the prior: which parameters are drawn, over what ranges, and how many draws make the expected loss stable enough;
   - whether to add a design variable that shows a mismatch between the local and the global distribution of a coordinate, where the signal is strong (the mechanism of experiment 1a in the 5D example), for example two petal families.
4. **The existing theory documents.** Whether [quadratic-norm-report.md](quadratic-norm-report.md) and section 3 of [example-requirements-design.md](example-requirements-design.md) are kept with a pointer to the reference, reduced, or removed, once the reference holds their theory. Settle before the first chapter that takes over their content.
5. **Location.** Where the series goes, for example `docs/reference/`.

## 11. Decision log

| date | decision |
|---|---|
| 2026-10-16 | The series is a reference of theory for later sampling and reconstruction design. It is not tied to an application. |
| 2026-10-16 | It cites existing theory and does not derive it again. Each result is marked cited, adapted, or new. |
| 2026-10-16 | It is separate from design: it does not choose a design, a reconstruction, or an application. Design documents cite it. |
| 2026-10-16 | The framework is Bayesian (optimal) experimental design: design variables, priors, outputs and loss, estimator, expected loss. Chapter 0 states it, and the other chapters refer to it. |
| 2026-10-16 | The theory is abstract: a signal over a product of dimensions, all alike; samples as linear functionals, in the simplest case point evaluations in a domain chosen for each dimension; the design as the joint distribution of the sample coordinates. |
| 2026-10-16 | Two questions about each dimension, representation and output, form the grid. A cell is one representation with one output, for an abstract dimension. |
| 2026-10-16 | The operator has nuisance components, which every estimator carries, and a design component, the sampling. (Reversed below: nuisance is relative to an output.) |
| 2026-10-16 | Two examples, each with every element of the framework: the 2D toy, x–t (simulated, objects drawn from a prior) and the 5D rosette problem (described, not simulated). They explain; a claim seen only in an example is a hypothesis. |
| 2026-10-16 | The toy starts from the running example of [quadratic-norm-report.md](quadratic-norm-report.md) and adds task-based BOLD. It goes through several rounds of review before the other chapters start. |
| 2026-10-16 | "Pressure": an informal, soft claim about how the expected loss changes with the coordinate distribution. "Condition": a hard property of the pair (design, estimator). Conditions filter designs; pressures rank them. The reference does not combine pressures into a design objective. |
| 2026-10-16 | Sources are kept in an annotated bibliography, with the notation map of each source in its entry. The series' notation is defined once, in chapter 0. |
| 2026-10-16 | Motion and design during the measurement are out of scope; both are later work. |
| 2026-10-16 | In the 5D example, the requirement called "navigator" in [example-requirements-design.md](example-requirements-design.md) is an output called the fast coarse image. |
| 2026-10-16 | Reversed: the series no longer serves stage (a) of the design work (choosing a design to test before the scan), nor the choice of the first on-scanner tests. Those are design work. |
| 2026-10-16 | Reversed: cells are no longer assigned to τ or T, nor "used by" the example requirements; the requirements are outputs of the 5D example, not goals. The synthesis and checks chapters became the combination and verification chapters. |
| 2026-10-16 | Reversed: the vocabulary of pressures specific to the rosette (shapes, library, schedule) moved from the theory to the 5D example; the theory's pressures concern the coordinate distribution. |
| 2026-10-16 | The toy is 2D: one dimension of space, and time (x–t). |
| 2026-10-16 | Effects that belong to an application (such as the decay over τ, R2\*, the coils) appear only in the examples, mainly the 5D example; the theory and the toy include only what illustrates a mechanism of the theory. |
| 2026-10-16 | The examples do not fix their outputs in advance. Each cell chapter chooses, from each example, the outputs and loss that best illustrate its theory. Names that the design documents already use are kept, and the 5D example names the fast coarse image. |
| 2026-10-16 | Reversed: no unknown is a nuisance parameter in advance. The forward model has the design (known, chosen) and unknowns (the signal and the operator's parameters), classed by how they enter it: linearly, bilinearly, or nonlinearly. For a given output, the nuisance parameters are the unknowns it does not ask for; an estimator estimates, collapses, or marginalizes each. |
