# pulseq-relaxed-rosettes

A relaxed rosette k-space trajectory pulse sequence, written in Python with [PyPulseq](https://github.com/pulseq/pypulseq).

## Setup

The nix devShell supplies Python, uv, and gh. With direnv, it loads when you enter the directory. Otherwise:

```
nix develop
uv sync
```

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

Each TR is one block: a hard pulse with the gradients off, one shape rotated by a rotation event, and the ADC (none in the dummy TRs at the start). The rotations are a super-Fibonacci spiral on SO(3), split among the shapes, in shuffled order. Next to the `.seq` file, `relaxed_rosettes.npz` has the schedule of each TR: `shape_index`, `rotation_matrix` (physical gradient = R @ shape gradient), `rf_phase_rad` (also the ADC phase), `acquired`, and the config.

## Dependencies

PyPulseq is pinned to a commit of the [mcencini fork](https://github.com/mcencini/pypulseq) (`experimental` branch), because it has the rotation extension (`pp.make_rotation`) and the official release does not yet ([pulseq/pypulseq#372](https://github.com/pulseq/pypulseq/pull/372)).
