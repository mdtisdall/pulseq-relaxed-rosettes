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
5. E1: the containment of the reference in its own repository (added after C6; it modifies A1, C2, C4 and D1).

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
- **Status:** decided (2026-10-16)
- **Decision:** fixed terms, each with one meaning, used throughout the reference and nowhere with another meaning. They go into `notation.md` (D1) as glossary entries, with the terms of A1.

  **What is chosen:**

  | term | meaning | 5D example | toy |
  |---|---|---|---|
  | **sampling pattern** | one concrete set of sample coordinates, with the coordinates of each sample; deterministic; what is actually measured | the trajectory of one `.seq` file: its shapes, rotations and schedule, realized from one seed | the coordinates of one simulated measurement |
  | **pattern generator** | a procedure that generates sampling patterns, possibly at random | the generator of the library, the rotations and the schedule, before a seed is fixed | the rule from which each trial's pattern is drawn |
  | **design variables** | the parameters of a pattern generator | the number of shapes, the target density, the spoil moment, the rule of the schedule | the number of shapes, the density of tips, the rule of the schedule |
  | **seed** | what selects one pattern from a pattern generator | the seed in `Config` | the seed of each trial |

  A deterministic pattern generator produces one pattern, and the distinction disappears for it. "Pattern generator" is always written in full: a bare "generator" would be confused with the data-generating model (A1) and with `np.random.Generator`. "Design" alone is used only for the activity (experimental design), never for one of these objects.

  **The distributions of the coordinates:**

  | term | meaning |
  |---|---|
  | **pattern distribution** | the empirical distribution of the coordinates within one sampling pattern; defined for any pattern |
  | **pattern-generator distribution** | the distribution over sampling patterns that a pattern generator produces. Independence across repetitions is a property of it |

  These replace "the joint distribution of the sample coordinates" of the plan.

  **Expected loss:**

  | term | averaged over | conditional on |
  |---|---|---|
  | **expected loss of a pattern** | the noise, and the priors of the data-generating model | the sampling pattern |
  | **expected loss of a pattern generator** | the same, and the pattern-generator distribution | the design variables |

  The expected loss of a pattern generator is the average of the expected loss of its patterns.

  The expected loss of a pattern generator averages over two distributions: the data-generating model's, which is assumed, and the pattern generator's, which is chosen and known exactly. The pattern generator is therefore not called a model: it approximates nothing, and has no modelling error of its own. The difference between the nominal and the measured trajectory (the gradient impulse response) is modelling error of the operator (A1).

  **Kinds of result:**

  | term | meaning |
  |---|---|
  | **pattern result** | holds for a given sampling pattern. For a linear estimator (A2), its bias and variance can be computed exactly for that pattern; the frame potential and the errors of experiment 1a are pattern quantities |
  | **generator result** | holds on average over a pattern generator; for example, leakage is white over temporal frequency when the patterns of the repetitions are drawn independently, and products between independent sets are unbiased |

  A generator result applied to one sampling pattern is an approximation. Each result states its kind; the size of the gap for a given pattern is a question for verification.

  **Related terms:**

  - **repetition:** a group of samples measured together; the unit across which independence is stated (one TR in the 5D example);
  - **split:** the division of the repetitions of a sampling pattern into sets, for the product estimators. A split is part of the estimator, not of the pattern;
  - **pressures and conditions** each state whether they concern a sampling pattern or a pattern generator. A condition is a property of the pair (sampling pattern, estimator) or (pattern generator, estimator) (A4).

  **Changes to the plan:**

  - The framework: "design variables" are the parameters of a pattern generator; the expected loss is defined in its two forms.
  - §1: "the coordinate distribution" is replaced by the pattern distribution and the pattern-generator distribution; "repetition" is defined.
  - §3: each pressure and condition states its kind (pattern or generator); "the change of the expected loss with the coordinate distribution" names which distribution.
  - §4: each example states its pattern generator, its design variables, and how its sampling patterns are drawn.
  - Elsewhere, "design" as a noun for a pattern or a generator is replaced by the term above.

### A4. "Invalid" in the definition of a condition is undefined

- **Problem.** §3 defines a condition as one where, if it fails, "the estimator is invalid", without saying what invalid means. Its example, "each set to hold the whole structure that the representation needs", is vague.
- **Proposed fix.** Define invalid: the unknown is not identifiable from the design, or an assumption of the estimator's derivation fails (for example, independence between sets). Rewrite the example as "each set must determine the representation on its own".
- **Status:** decided (2026-10-16)
- **Decision:** a condition is a yes-or-no property of one of two kinds. With the scope of A2 and the terms of A1 and A3:

  1. **Estimability.** The output must be estimable from the data, under the reconstruction model. For a linear model, a linear function of the coefficients is **estimable** if it lies in the row space of the model's operator (the operator applied to the representation): some linear function of the data has it as its expectation (Scheffé, *The Analysis of Variance*, 1959; Searle, *Linear Models*, 1971). If the output is not estimable, no estimator in the scope recovers it; with regularization, the estimate is set by the regularization prior, not by the data. Estimability is a pattern result: for a linear model, it can be checked exactly for a given sampling pattern.
  2. **A property that the estimator is built to have.** Some estimators are defined by a property that holds only under an assumption: for example, products between independent sets are unbiased only if the errors of the sets are independent. If the assumption fails, the estimator loses the property it exists for, and the bias that it was built to remove returns. These conditions are usually generator results.

  **Conditions and pressures.** The failure of a condition is binary: the output is estimable or not; the property holds or not. Nearness to failure is a pressure: an output that is barely estimable is badly conditioned, and its variance is large, which is measured continuously. So the τ-averaging mismatch of experiment 1a is a pressure, not a condition: the average was estimable, with a large propagated modelling error.

  **The example of §3, rewritten:** "For the product estimators: (a) the output must be estimable from each set of the split on its own, under the reconstruction model; (b) the errors of the sets must be independent, which needs their sampling patterns drawn independently by the pattern generator, and a split that does not align with any periodicity of the pattern generator."

  **Changes to the plan:**

  - §3: the definition of a condition, its two kinds, and the rule that failure is binary and nearness to failure is a pressure;
  - §3: the rewritten example;
  - bibliography: entries for estimability (Scheffé, 1959; Searle, 1971), with their notation mapped to the series'. The citations (editions, pages) are checked when the entries are written.

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
- **Status:** decided (2026-10-16)
- **Decision:** the proposed structure, started lighter, following the abstraction test of A2:

  ```
  docs/reference/
    README.md        entry point: purpose and scope (linear estimators, A2),
                     reading order, the chapters with one-line summaries,
                     conventions, links to the files below
    notation.md      the glossary (the terms of A1, A3 and A4) and the table
                     of symbols; each row links to the chapter that defines it
    examples/
      toy-x-t.md     the toy: data-generating model, pattern generator, noise;
                     and its code
      rosette-5d.md  the 5D problem, with its glossary of output names (C4)
    sources/
      README.md      index of sources
      <key>.md       one file for each source: citation and DOI, what is used
                     and where, assumptions, notation map, status, the chapters
                     that use it
    00-problem.md …  the chapters, each with a short header (a list, not YAML):
                     depends on (chapters), defines (terms and symbols, linked
                     to notation.md), sources (files)
  ```

  - **Deferred:** the results index (`results.md`) and stable result IDs, until the first chapter has results to index. The README states the convention for IDs, so that it is ready.
  - **The location** is `docs/reference/`; this settles open decision 5 of the plan.
  - **Unchanged from the proposal:** one file for each source (links checked by `scripts/check_links.py`; chapters written in parallel do not conflict; an agent loads one source alone); the examples in their own files; the notation in one registry; the README as the entry point for people and agents, which takes the lasting content of the plan (C5).

  **Changes to the plan:** §5 (one bibliography document with anchors) and §6 (the documents) change to this structure; open decision 5 is removed; checking anchors in `scripts/check_links.py` is not needed.

## B. Inconsistencies and errors

B1–B4 concern the details of the examples. Those details belong to the documents that define the examples (`examples/toy-x-t.md` and `examples/rosette-5d.md`, D1), and are refined when they are written. The plan keeps its §4 only as a rough draft of them, marked as such; B1–B4 are kept here as drafting notes for those documents, not as fixes to the plan. Change to the plan: §4 is marked "rough draft: the examples are defined, and refined, in `docs/reference/examples/`".

### B1. The 5D measurement formula leaves out the unknowns that §4.2 lists

- **Problem.** The formula of §4.2 has no coil sensitivity and no field phase, but the next bullet says that the coils enter bilinearly and the field nonlinearly.
- **Proposed fix.** Write the formula with both, for example

  $$y_{m,v} = \int s_v(r)\, x(r, \tau_m, T_m)\, e^{-2\pi i\, (k_m \cdot r \,+\, \Delta f(r, T_m)\, \tau_m)}\, dr \;+\; n_{m,v} .$$
- **Status:** decided (2026-10-16)
- **Decision:** deferred to the examples (2026-10-16). Drafting note for `examples/rosette-5d.md`: write the measurement with the coil sensitivity and the field phase, for example $y_{m,v} = \int s_v(r)\, x(r, \tau_m, T_m)\, e^{-2\pi i\, (k_m \cdot r + \Delta f(r, T_m)\, \tau_m)}\, dr + n_{m,v}$.

### B2. The toy lacks two elements that §4 requires of each example

- **Problem.** §4 says that each example states the unknown parameters of its operator and its noise. §4.1 has neither bullet. The phase drift of the running example could be an unknown of the signal or of the operator.
- **Proposed fix.** Add both bullets to §4.1, and decide where the phase drift belongs.
- **Status:** decided (2026-10-16)
- **Decision:** deferred to the examples (2026-10-16). Drafting note for `examples/toy-x-t.md`: state the unknowns of the operator and the noise. A candidate: the phase drift of the running example as an unknown of the operator (a phase that multiplies the signal), so that the toy has an instance of modelling error of the operator (A1) and of a nonlinear unknown.

### B3. The toy's design variables are the rosette's

- **Problem.** §4.1 states the toy's design as "petals and their tips" and "the sign of k as the rotation": the structure of the application, against the rule of §4 that the toy holds only what illustrates a mechanism.
- **Proposed fix.** State the toy's design as a parameterization of the distribution of its (k, t) coordinates, with petals as one instance of it.
- **Status:** decided (2026-10-16)
- **Decision:** deferred to the examples (2026-10-16). Drafting note for `examples/toy-x-t.md`: state the toy's design as a pattern generator over the (k, t) coordinates (A3), whose design variables set the pattern-generator distribution; petals are one instance of it.

### B4. The 5D example makes a design statement

- **Problem.** §4.2: "which make a rank-L representation over τ a natural choice here" chooses a representation, against "the examples do not make design decisions" (§7).
- **Proposed fix.** State the signal effects, and which representations can represent them, without choosing one.
- **Status:** decided (2026-10-16)
- **Decision:** deferred to the examples (2026-10-16). Drafting note for `examples/rosette-5d.md`: state the signal effects and which representations can represent them (for example, a fixed rank-L basis over τ, and the modelling error of a lower rank), without choosing one.

### B5. The order of the template's items

- **Problem.** Template item 2 (§7) names the nuisance parameters "for the outputs it chooses", but the outputs are chosen only in item 10.
- **Proposed fix.** Choose the outputs at item 2 and use them at item 10; or move the nuisance parameters into item 10.
- **Status:** decided (2026-10-16)
- **Decision:** the nuisance parameters split into two kinds, and each goes to a different item of the template.

  **Within the cell's dimension.** The output usually asks for less than the representation holds: for example, the representation holds a mean and a fluctuation around it, and the output is the mean, so the fluctuation is a nuisance parameter. This is intrinsic to the cell, not to an example, and the three ways to handle it are the grid's own choices:

  | handling of the part not asked for | what the cell does | in the grid |
  |---|---|---|
  | estimate it, then discard it | represent the mean and the fluctuation, then report the mean | fixed rank L × linear ("resolve, then average") |
  | leave it out of the representation | represent only the mean; the fluctuation becomes modelling error (A1) | rank 1 (truncate and average) |
  | marginalize over it | leave it out, but model its effect as correlated noise from its prior | the approximation error approach (A1) |

  Experiment 1a is this case: the output was the average over τ, the decay was the part not asked for, and the rank-1 cell left it out, so it became propagated modelling error.

  **Outside the cell's dimension:** the unknowns of the operator (for example coils and field) and the content of other dimensions. They are handled where dimensions and operators come together: in the combination chapter in the abstract, and in the examples.

  **The template of the cell chapters (§7 of the plan):**

  - **Item 2 (abstract):** which part of the representation the output does not ask for, and how the cell handles it: estimates it, leaves it out as modelling error, or marginalizes it.
  - **Item 10 (examples):** for each example, the outputs chosen and their loss, and the other unknowns that are nuisance parameters for them (of the operator, and of other dimensions), with how the estimator handles each.

  The abstract part (items 1–9) does not refer to the examples, and the outputs are still chosen at item 10.

  **Under C1** there is no fixed template, so items 2 and 10 become two items of the checklist of required content. Their order and their relation are kept, because each depends on the other:

  - **The within-dimension handling comes first, and in the abstract.** It defines the options (estimate and discard, leave out as modelling error, marginalize) and what each does to the bias and the variance. It belongs to the cell, so it is stated without the examples, by the rule "abstract first, examples last".
  - **The examples come after, and use those options.** For each example, the outputs chosen, their loss, and the nuisance parameters outside the dimension (of the operator, of other dimensions) are stated, and each nuisance parameter is handled by naming one of the options already defined. So the examples refer to the abstract part, never the reverse.
  - **Between them, in the combination chapter,** the abstract treatment of nuisance parameters outside a dimension, which the examples also use.

  **Also for chapter 2 (outputs):** the output axis of the grid and the handling of nuisance parameters are the same question. Each output type says which part of the representation is wanted and which is a nuisance parameter. Chapter 2 states this.

### B6. The decision log does not mark superseded rows

- **Problem.** Some rows of §11 are partly superseded without a mark: "each with every element of the framework" (the outputs are no longer fixed in advance), and "adds task-based BOLD" (now "a change locked to a stimulus"). An agent that reads the log will take them as current.
- **Proposed fix.** Mark each superseded row ("superseded by …"), and state at the top of the log that the body of the plan governs and the log is history.
- **Status:** decided (2026-10-16)
- **Decision:** replace the decision log with a section **"Rejected choices"**, whose purpose is to stop later work, by people or agents, from returning to a choice that was considered and turned down. It is not a changelog: current decisions are stated only in the body, open decisions only in their section, and the history of changes is in git and in the PR descriptions.

  **Structure.**

  - Grouped by topic, not by date, so that a reader who is about to make a choice finds the earlier rejection under the same topic.
  - Each entry has four fields:
    - **Rejected:** the choice, stated in the words in which it is likely to be proposed again, so that it is recognized;
    - **Instead:** what the reference does, with a link to the section of the body that states it;
    - **Why:** the reason, in one or two sentences;
    - **Reopen if:** what would justify reconsidering it (or "no foreseeable case").
  - The date of the rejection, after the entry.
  - An entry is removed only when its choice is adopted after all; the body then states it.
  - When the lasting content of the plan moves to `docs/reference/README.md` (C5), this section moves with it.

  **The first entries**, from the discussion so far (the wording is final when the plan is edited):

  *Purpose and scope*

  - **Rejected:** the reference serves stage (a) of the design work, or chooses the first on-scanner tests. **Instead:** a reference of theory, separate from design (Purpose). **Why:** choosing a design is design work; mixing it in ties the theory to one application. **Reopen if:** no foreseeable case.
  - **Rejected:** the synthesis chooses a design objective. **Instead:** it stops at conditions and the table of pressures. **Why:** a unified objective needs research of its own. **Reopen if:** a later project takes up that research.
  - **Rejected:** non-quadratic priors (sparsity, TV, the nuclear norm), learned priors, and joint low-rank fitting in scope now. **Instead:** linear estimators; these are known gaps (A2). **Why:** the project's reconstructions are linear, and the theory is cleanest there. **Reopen if:** a design or reconstruction in the project uses one of them.
  - **Rejected:** motion, and design during the measurement, in scope. **Instead:** both are later work. **Why:** neither is needed by the results planned. **Reopen if:** a design adapts to the data, or the motion model is taken up.
  - **Rejected:** a taxonomy of priors, or generalizing over all priors. **Instead:** each result states what it assumes about the data-generating prior (A2). **Why:** the data-generating prior is part of a problem, not of the reference. **Reopen if:** no foreseeable case.

  *Framework and terms*

  - **Rejected:** framing the series as ways to "collapse" a dimension. **Instead:** two questions for each dimension, representation and output (the grid). **Why:** collapsing mixes the representation with the output, and hides the choices of each.
  - **Rejected:** dimensions of different kinds, for example "encoded" (space) against "sampled directly" (time). **Instead:** all dimensions are alike; which domain is sampled is a property of the measurement (§1). **Why:** space sampled in k and time sampled in T are the same structure on opposite sides of a Fourier transform.
  - **Rejected:** some unknowns (for example coils, field) are nuisance parameters in advance. **Instead:** nuisance is relative to an output (§1). **Why:** the same unknown is an output in one problem and a nuisance parameter in another.
  - **Rejected:** the names "true model" or "evaluation model", "forward model" against "fitting model", "approximate model". **Instead:** data-generating model, reconstruction model, modelling error (A1). **Why:** "true" claims too much; both models have a forward operator; "approximate" suggests that the reconstruction model's simplicity is a defect, not a choice.
  - **Rejected:** "scheme", a bare "generator", or a "pattern-generating model". **Instead:** sampling pattern, pattern generator (A3). **Why:** a bare "generator" collides with the data-generating model and `np.random.Generator`; a pattern generator is chosen and exact, so it is not a model.
  - **Rejected:** "force" or "preference" for what a cell asks of the sampling. **Instead:** pressure (soft) and condition (hard) (§3, A4). **Why:** "force" suggests a formal vector at a point; "preference" sounds subjective and optional.
  - **Rejected:** the name "navigator" for the 8 mm, short-window output of the 5D example. **Instead:** fast coarse image. **Why:** "navigator" has several other meanings in MRI and in this repository.

  *The grid*

  - **Rejected:** cells assigned to τ or T, or "used by" the example requirements. **Instead:** cells are for an abstract dimension; whether a cell fits a dimension is a question for a problem (§2). **Why:** the reference is not tied to an application.
  - **Rejected:** rank 1, rank L and dense as different kinds of representation. **Instead:** one family, fixed rank L, with regimes (A2). **Why:** the theory is the same, with the rank as a parameter.
  - **Rejected:** "a representation is always a basis of some rank", with the prior folded into the representation. **Instead:** the reconstruction model is a representation, a prior and an assumed operator (A2). **Why:** sparsity and TV are priors over a representation, not representations.

  *The examples*

  - **Rejected:** a 1D toy. **Instead:** a 2D toy, x–t. **Why:** time is a dimension of the problem.
  - **Rejected:** outputs fixed in advance for the examples. **Instead:** each cell chapter chooses the outputs that best illustrate it. **Why:** fixed outputs make the examples into goals.
  - **Rejected:** effects of an application (decay over τ, R2\*, coils) in the toy. **Instead:** in the 5D example only. **Why:** the toy keeps only what illustrates a mechanism of the theory.
  - **Rejected:** simulating the 5D example in the reference. **Instead:** described, not simulated; measurement is design work or verification. **Why:** cost, and the separation from design.
  - **Rejected:** the details of the examples in the plan. **Instead:** a rough draft; the examples are defined in `docs/reference/examples/` (B1–B4).

  *The structure of the documents*

  - **Rejected:** chapters named by a mix of letters and numbers, or cells with numbers of their own. **Instead:** chapters numbered in reading order; cells named. **Why:** two numbering systems need translating in every reference.
  - **Rejected:** one bibliography file with anchors, and the notation inside chapter 0. **Instead:** one file for each source; `notation.md` (D1). **Why:** the link check does not check anchors; parallel chapters would conflict in one file; an agent can load one source or the notation alone.
  - **Rejected:** a results index before any results exist. **Instead:** deferred until the first chapter has results (D1). **Why:** the abstraction test (A2). **Reopen if:** the first chapter with results is written.
  - **Rejected:** a decision log as a changelog. **Instead:** this section. **Why:** a changelog repeats the body, and agents read superseded rows as current.

  **Change to the plan:** §11 (decision log) is replaced by this section; the decisions in the old log that are current are already stated in the body, or are added to it where they are not.

### B7. Small errors of wording

- **Problem.**
  - §4.2: "the coil of the session": the reference has no session.
  - §2: "the projection onto a given time series" is the wording of an application.
- **Proposed fix.** "A coil array, stated by each use"; "a given function".
- **Status:** decided (2026-10-16)
- **Decision:** as proposed. §4.2: "a coil array, stated by each use" (in the rough draft of the examples, B1–B4). §2: "a given function". Throughout: "design" is not used as a noun for a sampling pattern or a pattern generator (A3).

## C. Underspecified areas

### C1. Templates exist only for the cell chapters

- **Problem.** §7 gives a template for the cell chapters only. Chapters 0, 1 and 2, combination, and verification have none, and chapters 0–2 are written first.
- **Proposed fix.** A template for each of these chapters, or one shared template with the items that each uses.
- **Status:** decided (2026-10-16)
- **Decision:** no templates in the plan, including the template of the cell chapters in §7. The documents are written in sequence, and their structure is decided while writing, starting with chapter 0, and refined as the series grows.

  **Instead, a checklist of required content.** Each item is something that a chapter must contain, wherever it fits. Where one item depends on another, the checklist records the order and the reason, so that the dependency is kept even without a template (as for the nuisance parameters, B5).

  - the header: depends on, defines, sources (D1);
  - for each result: its status (cited, adapted, new); what it assumes about the data-generating prior; the class of estimator it holds for (A2); whether it is a pattern result or a generator result (A3);
  - for each pressure: the part of the error it concerns (A1), its direction and sign, and whether it concerns a sampling pattern or a pattern generator; for each condition: its kind (A4), and the same pair or generator;
  - for a cell, in the abstract and before the examples: the part of the representation that the output does not ask for, and how the cell handles it (B5);
  - the abstract theory never refers to the examples; the examples come after it, use the options it defines, and make no design choices;
  - in the examples: the outputs chosen, their loss, and the nuisance parameters outside the dimension, each handled by one of the options defined in the abstract (B5).

  **A template only before parallel writing.** Before the cell chapters are written at the same time (§8 of the plan), the structure of the first cell chapter can be adopted as the template for the others, so that they stay consistent. That template is based on a real chapter.

  **Change to the plan:** §7's template is replaced by this checklist; §8 notes the template taken from the first cell chapter.

### C2. Code and data for the toy and for verification

- **Problem.** The toy is simulated and verification runs numerical checks, but the plan does not say where the code goes (next to each chapter, as with `docs/quadratic-norm-report/`, or in the package), that it follows the project's rules on seeds and reproducibility, or where its outputs go (`data/`, not tracked).
- **Proposed fix.** Decide the place of the code and of its outputs, and state the rules.
- **Status:** decided (2026-10-16)
- **Decision:** code sits next to the document that uses it, as `docs/quadratic-norm-report/` does: the code of the toy in `docs/reference/examples/`, and the code of verification next to the verification chapter. It follows the project's rules: randomness from an `np.random.Generator` seeded explicitly, every parameter stated, results reproducible from them. Its outputs go to `data/` (not tracked); only the figures that a chapter shows are committed, next to it. Code moves into the package only if design work needs to reuse it.

### C3. The cost tiers come from a design document

- **Problem.** The verification chapter (§6) uses the tiers of `docs/example-requirements-design.md`, section 7.3. A reference should not depend on a design document for its own method.
- **Proposed fix.** The verification chapter defines its tiers, and cites the design document only as their origin.
- **Status:** decided (2026-10-16)
- **Decision:** the verification chapter defines its own tiers of cost, and does not cite the design document for them. The tiers themselves (exact small problems; the real sampling patterns, reduced; full runs only to confirm a conclusion) are used as they are. Change to the plan: §6 drops the reference to `docs/example-requirements-design.md`, section 7.3.

### C4. The 5D outputs are named by reference only

- **Problem.** §4.2: "outputs that the project's design documents already name keep their names" gives an agent no list.
- **Proposed fix.** A glossary of output names in the 5D example (static image, relaxation 4D, band power, fast coarse image, and so on), each with its source document.
- **Status:** decided (2026-10-16)
- **Decision:** deferred to the examples, like B1–B4. Drafting note for `examples/rosette-5d.md`: a glossary of the names of the outputs of the 5D example (for example the static image, relaxation 4D, band power, the fast coarse image), each with the design document that defines it.

### C5. The life of the plan

- **Problem.** The plan does not say what becomes of it once chapters exist: whether it becomes the README of the series, stays as history, or is deleted.
- **Proposed fix.** Its lasting content (conventions, reading order) moves into `docs/reference/README.md` (D1); the rest is removed when the series is set up.
- **Status:** decided (2026-10-16)
- **Decision:** settled by D1. When the series is set up, the lasting content of the plan (the scope, the conventions, the reading order, and the rejected choices of B6) moves into `docs/reference/README.md`, and the plan is then removed.

### C6. Missing open decisions

- **Problem.** §10 of the plan does not list A1–A3, C1, C2, or the linking structure (D1).
- **Proposed fix.** Add those that are not settled here before the plan is merged.
- **Status:** decided (2026-10-16)
- **Decision:** after this review, the open decisions of the plan are only those still open:

  1. whether rank 1, or either family of the grid (fixed rank L; fitted then fixed rank K), needs a chapter of its own (A2);
  2. estimator settings: whether regularization, the number of iterations, and the split into sets are part of an estimator, or parameters that each result states;
  3. the details of the toy: which effects of the running example to keep, the task, the prior, and whether to show the mechanism of experiment 1a;
  4. the fate of the existing theory documents (`docs/quadratic-norm-report.md`, and section 3 of `docs/example-requirements-design.md`).

  Everything decided in this review goes into the body of the plan.

## E. The repository of the reference

### E1. The reference lives in its own repository, and is self-contained

- **Problem.** The reference will be rewritten, and will live, in a repository of its own, separate from this project. Several decisions above assume that it sits in `docs/reference/` of this project, next to the design documents it draws on.
- **Status:** decided (2026-10-16)
- **Decision:**

  **The containment rule.** The documents of the reference (the chapters, `README.md`, `notation.md`, the examples, the sources) are self-contained in the reference's repository. Their only citations outside it are published sources. They do not link to this project's documents or code. The plan is a working document and may point into this project.

  **Consequences:**

  - **Paths.** The structure of D1 is the root of the new repository, not `docs/reference/`.
  - **The 5D example is described in full inside the reference:** its sampling, its outputs, its physics. It does not point to this project's design documents. Its glossary of output names (C4) defines each name in the reference itself. A measurement from this project (for example experiment 1a) is not cited: if it matters to the reference, the reference reproduces it in its own verification, or leaves it out.
  - **Theory from this project's documents is restated, not linked.** `docs/quadratic-norm-report.md` and section 3 of `docs/example-requirements-design.md` are unpublished, so they are not cited. Their content enters the reference stated in full, sourced to published literature where a source exists. Where none does (for example the products between independent sets, and the τ-averaging mismatch), the result is marked **new** and stated with its own argument. From the reference's side, this settles open decision 4 of C6: the reference never depends on those documents. What this project does with them afterwards is its own decision.
  - **The code is self-contained.** The code of the toy and of verification does not import `relaxed_rosettes`. C2's "moves into the package if design work needs to reuse it" becomes "moves into the reference repository's own package". If this project wants it, this project depends on the reference, not the reverse.
  - **The direction of citation.** This project's design documents cite the reference (by URL, at a fixed commit). The reference does not cite this project.
  - **README and notation.** They are part of the reference, so the rejected choices that move into the README (B6, C5) are worded without links into this project.

  **Changes to earlier decisions:**

  - **A1:** the example of experiment 1a stays in the plan as motivation; in the reference, the propagated modelling error is shown by the examples and by the reference's own verification.
  - **C2:** as above (the reference's own package).
  - **C4:** the glossary defines the names in the reference, without the design documents.
  - **C6:** open decision 4 changes to: what this project does with its theory documents once the reference holds their content (a decision for this project, not for the reference).
  - **D1:** the structure is at the root of the new repository.

  **Changes to the plan:** a containment rule in the rules (§7); the paths of §5 and §6; the open decisions (§10, from C6); and a rejected choice: "Rejected: chapters link to the project's design documents or code. Instead: the reference is self-contained, with published sources as its only external citations. Why: the reference must stand on its own, and depend on no application."

