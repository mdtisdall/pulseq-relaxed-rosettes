# Review of the reference-theory plan: issues to settle before writing

Status: working document, started 2026-10-16. It lists the issues found in a review of `docs/reference-theory-plan.md` (branch `docs/collapse-plan`, commit `5ff637b`, not yet on `main`), so that they can be settled one at a time. For each issue, record the decision and its date, and then apply the fix to the plan on its own branch.

The plan is referred to by its path, not by a link, because it is not on `main` yet: a relative link to it would fail the link check of this branch.

## How to use this document

- Go through the issues in the order of section 1.
- For each issue: discuss it, record the decision in its **Decision** line, and set its **Status** to `decided` (or `dropped`, with the reason).
- Apply the decided fixes to the plan on `docs/collapse-plan`, and set the status to `applied`, with the commit.
- The section numbers below (§) are those of the plan at commit `5ff637b`.

Status values: `open`, `decided`, `applied`, `dropped`.

## 1. Order

1. A1–A4: they change the content of chapters 0–2.
2. D: the linking structure decides the files that chapter 0 creates.
3. B1–B7: wording and consistency fixes.
4. C1–C6: additions to the open decisions of the plan.

## A. Gaps in the foundations of the theory

### A1. Model error has no place in the abstraction

- **Problem.** §4.2 says that spoiling enters "as a model error of the operator", and §7 lists the local–global mismatch as a thin place, but §1–§3 never define model error. The main result of the rank-1 average (experiment 1a) is a model error: the true signal lies outside the representation. The framework (§"The framework") says that a representation "builds in a prior", but does not connect that to the expected loss.
- **Proposed fix.** Add to §1 the distinction between the **true model** (the priors and the forward model over which the expected loss is computed) and the **estimator's assumed model** (its representation and its operator). Model error is the difference between them, in the signal or in the operator. §3 then lets a pressure or a condition concern model error as well as noise and conditioning.
- **Status:** decided (2026-10-16)
- **Decision:** two models, and the gap between them, with these names:

  - **Data-generating model:** how the data arise. The forward operator, with the priors over everything it acts on: the signal and its changes, the unknown parameters of the operator, and the noise. The expected loss is computed over it. It is the richest model that a problem states, not the truth, which is never known.
  - **Reconstruction model:** what the estimator assumes. Its representation (a cell of the grid), its assumed operator (for example a static field map, or no spurious echoes), and any prior that its regularization builds in. "Inversion model" is the equivalent term in the inverse-problems literature; "reconstruction model" is used because the reference is about MRI reconstruction.
  - **Modelling error:** the difference between the two models, in the signal (the part of the signal outside the representation) or in the operator (the difference between the true and the assumed operator).

  Using the same model to generate synthetic data and to invert them is the **inverse crime**, a term from Colton and Kress (*Inverse Acoustic and Electromagnetic Scattering Theory*), as discussed by Wirgin (2004, arXiv:math-ph/0401050): it hides the modelling error and makes results look better than they will be on real data. The reference therefore keeps the two models separate, in the theory and in both examples.

  **How the error of an estimate splits.** With data from the data-generating model, an estimator built on the reconstruction model, and an output, the error has four parts:

  | part | what it is | depends on the design |
  |---|---|---|
  | noise | the estimator applied to the noise | yes (conditioning) |
  | aliasing within the reconstruction model | the error that remains when the signal is in the representation and the operator is as assumed | yes (what the frame potential measures) |
  | propagated modelling error, signal | the part of the signal outside the representation, carried into the estimate by the operator and the estimator | yes |
  | propagated modelling error, operator | the estimator applied to the difference between the true and the assumed operator, acting on the signal | often |

  The modelling error itself (for example the distance from the signal to the best estimate in the representation) does not depend on the design. How much of it propagates into the estimate does, when the sampling is incomplete. Experiment 1a is the example: the decay over τ that a rank-1 representation leaves out propagated into the average according to the mix of τ in each cell of k-space, which the frame potential could not see.

  **How an estimator can handle modelling error** (the same options as for a nuisance parameter, §1 of the plan):

  - ignore it, and accept its propagated part as a bias;
  - enlarge the reconstruction model, so that the error becomes part of the representation or of the estimated operator;
  - **marginalize over it:** treat the modelling error as an additional, correlated noise term, whose mean and covariance are estimated by sampling the prior of the data-generating model, and include it in the likelihood. This is the approximation error approach of Kaipio and Somersalo (*Statistical and Computational Inverse Problems*, Springer, 2005; and "Statistical inverse problems: discretization, model reduction and inverse crimes", *J. Comput. Appl. Math.* 198(2):493–504, 2007).

  **Changes to the plan:**

  - The framework: the data-generating model defines the expected loss; the reconstruction model belongs to the estimator; the sentence "a representation builds in a prior" becomes a statement about the reconstruction model.
  - §1: the two models, modelling error, and the four parts of the error, with only the propagated parts depending on the design.
  - §3: each pressure and each condition states which part of the error it concerns: noise, aliasing, or propagated modelling error (signal or operator).
  - §4: each example states its data-generating model; each cell chapter states its reconstruction model for the outputs it chooses. The 5D example's spoiling and steady state become modelling error of the operator, in these terms.
  - Bibliography: entries for Colton and Kress, Wirgin (2004), and Kaipio and Somersalo (2005, 2007), with the notation map: their "model used to generate the data" and "accurate model" to the data-generating model; their "model used for the inversion" and "approximate" or "reduced model" to the reconstruction model; their "approximation error" and "modelling error" to modelling error. Each citation is checked against its source when the entry is written: the edition and page of Colton and Kress where the term is introduced, the year of the Kaipio and Somersalo book (2004 or 2005), and the DOIs.

### A2. "It is always a basis of some rank" is false as stated

- **Problem.** §2 says that a representation "is always a basis of some rank". Sparsity (compressed sensing), total variation, unions of subspaces, and learned priors are representations that are not bases of a rank. Agents will apply the grid literally.
- **Proposed fix.** Either state the scope: the reference covers linear, low-rank representations, and lists the others as out of scope; or add a class to the representation axis for regularized or sparse representations.
- **Status:** decided (2026-10-16)
- **Decision:** the reference is about **linear estimators**, and says so in its title, its introduction, and its rules.

  **The scope.** Linear estimators: estimators that are linear in the data, given their reconstruction model (A1). This includes estimators whose basis is fitted from the data first and then held fixed, as in `docs/basis-quality-test-plan.md`, where the temporal basis is estimated first so that the reconstruction is one linear solve. The fitting of the basis is then a modelling error of the reconstruction model: the basis is chosen with the data.

  **Known gaps**, listed in the introduction with their main citations, to be filled when a design or reconstruction in the project uses them:

  - non-quadratic priors: sparsity (Lustig, Donoho and Pauly, 2007), total variation, the nuclear norm; their sampling theory (incoherence; variable density, Puy, Vandergheynst and Wiaux, 2011; multilevel sampling, Adcock, Hansen, Poon and Roman, 2017; variable density along continuous trajectories, Chauffert, Ciuciu, Kahn and Weiss, 2014; structured acquisition, Boyer, Bigot and Weiss, 2019); and their partial tools for risk (degrees of freedom and Stein's unbiased risk estimate for the lasso and generalized lasso, Tibshirani and Taylor, 2012);
  - learned priors;
  - joint (alternating) fitting of a low-rank model, which gives a nonlinear estimator.

  These citations are checked against their sources when the known gaps are written.

  **The reconstruction model** (A1) has three parts: the representation (what is estimated), the prior over its coefficients (none or quadratic, in scope; non-quadratic, a known gap), and the assumed operator. The reference does not classify priors. A result states what it assumes about the data-generating prior (for example, a band limit) and what class of estimator it holds for.

  **The representation axis of the grid** has two families:

  - **fixed, rank L:** one family, with the rank as its parameter. Its regimes are set by two comparisons, L against the content of the signal (modelling error) and L against what the sampling can determine (conditioning and variance):

    | regime | rank | what dominates |
    |---|---|---|
    | rank 1 | L = 1 | modelling error, unless the signal is constant; the coefficient and a linear output coincide |
    | below the content | 1 < L < content | a trade between modelling error and variance |
    | dense | L ≥ content | variance and conditioning; modelling error is negligible |

    "Dense" is relative to the signal: for example, about 45 Slepian components for a band in 5000 repetitions. The largest possible rank would be no model of the dimension, which an undersampled measurement cannot determine.
  - **fitted then fixed, rank K:** the same two comparisons, plus the effect of fitting the basis from the data.

  So the grid has 2 × 3 cells (two families of representation, three outputs). This answers part of open decision 1 of the plan (the groups of cells); rank 1 stays a named special case inside the fixed family.

  **Two rules**, added to §7 of the plan:

  1. **Scope rule.** Results concern linear estimators, given the reconstruction model. A fitted basis is treated as fixed, and its fitting as a modelling error. A result outside the scope is not stated as a result of the reference; it goes into the known gaps.
  2. **Abstraction test.** A concept is kept in the reference only if (a) at least one result that the reference states uses it, and (b) both examples can instantiate it. Otherwise it goes on a list of deferred extensions. (This also bears on D1: the results index can be added when the first chapter has results.)

  **Changes to the plan:**

  - Title: for example, "A reference of theory for sampling and reconstruction design with linear estimators: plan of a document series".
  - Purpose: the scope sentence, the reason (it covers the reconstructions that the project uses, and the theory is cleanest there), and the known gaps.
  - §2: "It is always a basis of some rank" is replaced by the two families above; "fitted, rank K" becomes "fitted then fixed, rank K"; the reconstruction model's prior is named, not classified.
  - §7: the two rules. In the known thin places, the entry on nonlinear estimators for fitted representations is replaced by "the effect of fitting the basis from the same data as the reconstruction".
  - §10: open decision 1 is reduced to whether the two families, or rank 1, need chapters of their own.

### A3. Fixed designs and random designs are not separated

- **Problem.** The framework defines the design as "the parameters of the joint distribution of the sample coordinates", and several results need independence across groups of samples. A real design is one deterministic set of samples (from a seed). The expected loss in the framework averages over the priors and the noise only.
- **Proposed fix.** State, in §1 and in the framework, whether a result is for a **random design** (expectations include the randomness of the design) or for a **fixed design** (conditional on one realization), and how the empirical distribution of a fixed design relates to the distribution that the results use.
- **Status:** open
- **Decision:**

### A4. "Invalid" in the definition of a condition is undefined

- **Problem.** §3 defines a condition as one where, if it fails, "the estimator is invalid", without saying what invalid means. Its example, "each set to hold the whole structure that the representation needs", is vague.
- **Proposed fix.** Define invalid: the unknown is not identifiable from the design, or an assumption of the estimator's derivation fails (for example, independence between sets). Rewrite the example as "each set must determine the representation on its own".
- **Status:** open
- **Decision:**

## D. The linking structure

### D1. A file structure that both people and agents can navigate

- **Problem.** §5 and §6 of the plan describe one bibliography document with anchors, and put the notation and the examples inside chapter 0. `scripts/check_links.py` checks the file of a relative link but not its anchor, so links to entries would be checked only as far as the file. Cell chapters written in parallel would all edit one bibliography and one notation table, and conflict. An agent that needs one symbol, one result, or one source must load a whole chapter.
- **Proposed fix.** This structure, with one entry point and every item one or two links away:

  ```
  docs/reference/
    README.md            entry point: purpose, reading order, the chapters
                         with one-line summaries, conventions (status
                         labels, result IDs), links to the indexes below
    notation.md          the single registry of symbols: symbol, meaning,
                         defining chapter (link). Chapters add rows; they
                         do not redefine symbols
    results.md           index of results by stable ID (for example R01.3):
                         a one-line statement, status (cited, adapted, new),
                         chapter link, source links
    examples/
      toy-x-t.md         the full definition of the toy (signal model,
                         unknowns, priors, design, noise), and its code
      rosette-5d.md      the full description of the 5D problem, with the
                         glossary of output names (C4)
    sources/
      README.md          index of sources
      <author-year-key>.md   one file for each source: citation and DOI,
                         what is used and where, assumptions, notation map,
                         status, the chapters that use it
    00-problem.md … the chapters
  ```

  - **One file for each source:** links to it are checked by the link check; chapters written in parallel add separate files and do not conflict; an agent can load one source alone.
  - **Notation and results in their own files:** a design agent can load `notation.md` and `results.md` alone; chapters add rows to one registry.
  - **Stable result IDs:** design documents cite a result by its ID, which does not change when a heading changes; the index makes the IDs checkable.
  - **The examples as their own files:** everything about the toy or the 5D problem is in one place; cell chapters link to them and add only the outputs they choose; chapter 0 frames the examples and does not contain them.
  - **A header in each chapter**, as a short list (not YAML, so that it reads well on GitHub): depends on (chapters), defines (symbols, linked to `notation.md`), results (IDs), sources (files).
  - **The README is the entry point**, and the lasting content of the plan (conventions, reading order) moves into it (C5).

  Consequences: §5 and §6 of the plan change to this structure; checking anchors in `scripts/check_links.py` becomes optional, and would be a separate change on its own branch.
- **Status:** open
- **Decision:**

## B. Inconsistencies and errors

### B1. The 5D measurement formula leaves out the unknowns that §4.2 lists

- **Problem.** The formula of §4.2 has no coil sensitivity and no field phase, but the next bullet says that the coils enter bilinearly and the field nonlinearly.
- **Proposed fix.** Write the formula with both, for example

  $$y_{m,v} = \int s_v(r)\, x(r, \tau_m, T_m)\, e^{-2\pi i\, (k_m \cdot r \,+\, \Delta f(r, T_m)\, \tau_m)}\, dr \;+\; n_{m,v} .$$
- **Status:** open
- **Decision:**

### B2. The toy lacks two elements that §4 requires of each example

- **Problem.** §4 says that each example states the unknown parameters of its operator and its noise. §4.1 has neither bullet. The phase drift of the running example could be an unknown of the signal or of the operator.
- **Proposed fix.** Add both bullets to §4.1, and decide where the phase drift belongs.
- **Status:** open
- **Decision:**

### B3. The toy's design variables are the rosette's

- **Problem.** §4.1 states the toy's design as "petals and their tips" and "the sign of k as the rotation": the structure of the application, against the rule of §4 that the toy holds only what illustrates a mechanism.
- **Proposed fix.** State the toy's design as a parameterization of the distribution of its (k, t) coordinates, with petals as one instance of it.
- **Status:** open
- **Decision:**

### B4. The 5D example makes a design statement

- **Problem.** §4.2: "which make a rank-L representation over τ a natural choice here" chooses a representation, against "the examples do not make design decisions" (§7).
- **Proposed fix.** State the signal effects, and which representations can represent them, without choosing one.
- **Status:** open
- **Decision:**

### B5. The order of the template's items

- **Problem.** Template item 2 (§7) names the nuisance parameters "for the outputs it chooses", but the outputs are chosen only in item 10.
- **Proposed fix.** Choose the outputs at item 2 and use them at item 10; or move the nuisance parameters into item 10.
- **Status:** open
- **Decision:**

### B6. The decision log does not mark superseded rows

- **Problem.** Some rows of §11 are partly superseded without a mark: "each with every element of the framework" (the outputs are no longer fixed in advance), and "adds task-based BOLD" (now "a change locked to a stimulus"). An agent that reads the log will take them as current.
- **Proposed fix.** Mark each superseded row ("superseded by …"), and state at the top of the log that the body of the plan governs and the log is history.
- **Status:** open
- **Decision:**

### B7. Small errors of wording

- **Problem.**
  - §4.2: "the coil of the session": the reference has no session.
  - §2: "the projection onto a given time series" is the wording of an application.
- **Proposed fix.** "A coil array, stated by each use"; "a given function".
- **Status:** open
- **Decision:**

## C. Underspecified areas

### C1. Templates exist only for the cell chapters

- **Problem.** §7 gives a template for the cell chapters only. Chapters 0, 1 and 2, combination, and verification have none, and chapters 0–2 are written first.
- **Proposed fix.** A template for each of these chapters, or one shared template with the items that each uses.
- **Status:** open
- **Decision:**

### C2. Code and data for the toy and for verification

- **Problem.** The toy is simulated and verification runs numerical checks, but the plan does not say where the code goes (next to each chapter, as with `docs/quadratic-norm-report/`, or in the package), that it follows the project's rules on seeds and reproducibility, or where its outputs go (`data/`, not tracked).
- **Proposed fix.** Decide the place of the code and of its outputs, and state the rules.
- **Status:** open
- **Decision:**

### C3. The cost tiers come from a design document

- **Problem.** The verification chapter (§6) uses the tiers of `docs/example-requirements-design.md`, section 7.3. A reference should not depend on a design document for its own method.
- **Proposed fix.** The verification chapter defines its tiers, and cites the design document only as their origin.
- **Status:** open
- **Decision:**

### C4. The 5D outputs are named by reference only

- **Problem.** §4.2: "outputs that the project's design documents already name keep their names" gives an agent no list.
- **Proposed fix.** A glossary of output names in the 5D example (static image, relaxation 4D, band power, fast coarse image, and so on), each with its source document.
- **Status:** open
- **Decision:**

### C5. The life of the plan

- **Problem.** The plan does not say what becomes of it once chapters exist: whether it becomes the README of the series, stays as history, or is deleted.
- **Proposed fix.** Its lasting content (conventions, reading order) moves into `docs/reference/README.md` (D1); the rest is removed when the series is set up.
- **Status:** open
- **Decision:**

### C6. Missing open decisions

- **Problem.** §10 of the plan does not list A1–A3, C1, C2, or the linking structure (D1).
- **Proposed fix.** Add those that are not settled here before the plan is merged.
- **Status:** open
- **Decision:**
