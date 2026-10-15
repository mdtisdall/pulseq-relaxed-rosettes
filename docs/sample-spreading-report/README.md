# Supporting data for the sample-spreading report

Raw outputs behind [../sample-spreading-report.md](../sample-spreading-report.md), kept so that later work can review or rescore them. All designs have 16 shapes and seed 0 (report, section 2).

## Files

- `coverage/<label>.cov.txt`: the full output of `relaxed-rosettes coverage` for each design (coverage and gaps at every window and resolution, median and maximum over window starts; (|k|, τ) sampling; within-TR redundancy).
- `logs/<label>.shapes.log`: the `relaxed-rosettes shapes` output of a base design (per-shape tips, peak gradient and slew, target error, solve time).
- `logs/<label>.repel.log`: the per-shape output of a repulsion driver (energy distance and slew energy before and after, steps accepted, constraint checks, time).
- `console/`: console output of one-off checks, transcribed from the session because it was not written to files (trust-region histories, the `optimal_inaccurate` check, the edge analysis, the engineering measurements of report section 3.0).
- `scripts/`: the drivers and pipelines that produced everything. They contain absolute paths to the session's temporary directories; change those to rerun them.

## Designs

| label | what it is | made by |
|---|---|---|
| A | relaxed base: crossings 1 cell, `tip_width` 0.5 | `relaxed-rosettes shapes` (`scripts/sweep_A-D_pipeline.sh`) |
| B, C, D | A + repulsion, weight 0.5 / 1 / 3, **first trust-region rule** (double on every accepted step) | `scripts/repel.py`; the doubling rule was never committed (it is the ratio rule's predecessor, see report 3.2) |
| E | A + repulsion, weight 3, ratio rule (as in PR #23) | `scripts/repel.py` |
| F | hairpin base: the default config (no relaxation) | `relaxed-rosettes shapes` (`scripts/F-G_pipeline.sh`) |
| G | F + repulsion, weight 3 | `scripts/repel.py` |
| H0 | hairpins solved on A's targets (only in the waveform folder) | `scripts/H_hairpins_on_relaxed_targets.py` |
| H | H0 + repulsion under A's relaxed constraints, with each shape's target and λ fixed to E's | `scripts/H_repel_fixed_target_lambda.py` |
| I, J | F + repulsion with the tip targets dropped, 20 / 60 steps | `scripts/I-J_repel_free_tips.py` |
| K, L, M | `tip_density = filter` base; + repulsion with the `initial` target; + repulsion with the `model` target | `scripts/K-M_pipeline.sh` |

**The embedded configs do not always describe the variant.** B–D and E share one config but used different trust-region rules; H's config is E's, but its start, target and λ came from the driver; I and J carry F's config with repulsion fields, but their tips were not constraints. Use the driver named above to reproduce a design.

## Waveforms

The shapes files (`<label>.shapes.npz`, about 2 MB each, 16 shapes with their targets and config) are not in the repository. They are in `data/sample-spreading-2026-10-15/` of the main checkout on the machine that ran the experiments (`data/` is not tracked; see `.gitignore`). Load them with `relaxed_rosettes.optimize.load_shapes`, and rebuild a sequence with `relaxed-rosettes sequence <file> -o <name>.seq` (the `.seq` and design files were not kept).
