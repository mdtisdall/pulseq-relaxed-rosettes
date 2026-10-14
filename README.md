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

## Dependencies

PyPulseq is pinned to a commit of the [mcencini fork](https://github.com/mcencini/pypulseq) (`experimental` branch), because it has the rotation extension (`pp.make_rotation`) and the official release does not yet ([pulseq/pypulseq#372](https://github.com/pulseq/pypulseq/pull/372)).
