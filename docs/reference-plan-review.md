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
- **Status:** open
- **Decision:**

### A2. "It is always a basis of some rank" is false as stated

- **Problem.** §2 says that a representation "is always a basis of some rank". Sparsity (compressed sensing), total variation, unions of subspaces, and learned priors are representations that are not bases of a rank. Agents will apply the grid literally.
- **Proposed fix.** Either state the scope: the reference covers linear, low-rank representations, and lists the others as out of scope; or add a class to the representation axis for regularized or sparse representations.
- **Status:** open
- **Decision:**

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
