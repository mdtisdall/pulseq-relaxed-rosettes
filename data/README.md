# Local data

This directory holds two kinds of local files. Git ignores both; only this
README is in the repository.

## Gradient system files

Put the Siemens gradient system `.asc` files of the scanner in this directory.
`relaxed-rosettes check --asc PATH` needs them for the PNS prediction (SAFE
model) and the acoustic resonance bands. Without them, there is no PNS check.

The files are Siemens-confidential. `.gitignore` ignores `data/*.asc`, so git
does not commit them. Do not force-add them, and do not rename them to an
extension other than `.asc`.

### Files

A gradient system has two files. Put both of them here:

- `MP_GradSys_<GPA>_<engine>_<coil>.asc`. It has the gradient limits and
  the acoustic resonances. pypulseq and the
  safe_pns_prediction README give names of the form `MP_GPA_<GPA>_<coil>.asc`,
  which other software versions may use.
- `MP_GradSys_<GPA>_<engine>_<coil>_GSWD_SAFETY.asc`. It has the SAFE PNS
  model parameters (`GradPatSup.Phys.PNS.flGSWD*`). The first file includes it
  with a `$INCLUDE` line.

Give `--asc` the first file. The PNS prediction also reads each file
that a `$INCLUDE` line names, from the same directory, so keep the two files
together. If the included file is not there, the prediction stops with an
error that names it.

### Where to get the files

They are on the scanner host. The IDEA installation can also have them. The IDEA files can
replace the scanner files if the gradient system and the software version are
the same.
Record the source of the files, the software version, and the `imprint` output
of the scanner.

### Worktrees

A new worktree in `.worktrees/` does not get these files. Give the path to the
files in the main checkout, for example `--asc ../../data/<file>.asc`.

## Results

Experiment results, designs, trajectories and `.seq` files go in
`data/<topic>-<date>/` of the main checkout, for example
`data/phantom-2026-10-15/`. They can be large (hundreds of MB), so they stay
local. Keep only `.seq` files that pass every check (`check --asc`: timing,
Nyquist, PNS, and the acoustic resonance bands), and delete the others so
that they cannot be run by accident. Designs made with the acoustic bands
carry the coil's resonance frequencies in their config, so treat their
files like the `.asc` files when sharing them.
