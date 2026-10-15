"""Command-line interface: ``relaxed-rosettes <command>``."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pypulseq as pp

from relaxed_rosettes import __version__
from relaxed_rosettes.checks import (
    acoustic_resonances,
    acquired_time_range,
    format_report,
    gradient_report,
    pns_report,
    spectrum_report,
    timing_report,
)
from relaxed_rosettes.config import Config, ConfigError
from relaxed_rosettes.coverage import (
    format_ktau,
    format_metrics,
    format_redundancy,
    ktau_metrics,
    window_metrics,
    within_tr_redundancy,
)
from relaxed_rosettes.optimize import (
    Shape,
    ShapeOptimizationError,
    load_shapes,
    optimize_shapes,
    save_shapes,
    shape_metrics,
)
from relaxed_rosettes.recon.config import ReconConfig
from relaxed_rosettes.sequence import (
    Schedule,
    build_sequence,
    load_design,
    make_schedule,
    read_seq_definition,
    read_seq_signature,
    save_design,
)
from relaxed_rosettes.system import make_opts
from relaxed_rosettes.targets import ShapeTargets, make_targets, save_targets
from relaxed_rosettes.trajectory import (
    load_trajectory,
    make_trajectory,
    save_trajectory,
    verify_trajectory,
)


def _load_config(args: argparse.Namespace) -> Config:
    config = Config.load(args.config) if args.config else Config()
    return config.with_overrides(args.set)


def _add_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "config", nargs="?", type=Path, help="TOML config file (default: built-in defaults)"
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="SECTION.KEY=VALUE",
        help="override one config value (repeatable), e.g. --set readout.tr_s=0.025",
    )


def _cmd_config_init(args: argparse.Namespace) -> int:
    path: Path = args.path
    if path.exists() and not args.force:
        print(f"error: {path} exists (use --force to overwrite)", file=sys.stderr)
        return 1
    path.write_text(Config().to_toml())
    print(f"wrote {path}")
    return 0


def _cmd_config_show(args: argparse.Namespace) -> int:
    config = _load_config(args)
    print(f"# config digest: {config.digest()}")
    print(config.to_toml(), end="")
    return 0


def _load_recon_config(args: argparse.Namespace) -> ReconConfig:
    config = ReconConfig.load(args.recon_config) if args.recon_config else ReconConfig()
    return config.with_overrides(args.set)


def _add_recon_config_args(parser: argparse.ArgumentParser, *, positional: bool) -> None:
    if positional:
        parser.add_argument(
            "recon_config",
            nargs="?",
            type=Path,
            help="recon TOML config file (default: built-in defaults)",
        )
    else:
        parser.add_argument(
            "-c",
            "--recon-config",
            type=Path,
            help="recon TOML config file (default: built-in defaults)",
        )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="SECTION.KEY=VALUE",
        help="override one recon config value (repeatable), e.g. --set data.resolution_m=0.004",
    )


def _cmd_recon_config_init(args: argparse.Namespace) -> int:
    path: Path = args.path
    if path.exists() and not args.force:
        print(f"error: {path} exists (use --force to overwrite)", file=sys.stderr)
        return 1
    path.write_text(ReconConfig().to_toml())
    print(f"wrote {path}")
    return 0


def _cmd_recon_config_show(args: argparse.Namespace) -> int:
    config = _load_recon_config(args)
    print(f"# recon config digest: {config.digest()}")
    print(config.to_toml(), end="")
    return 0


def _echo_range(targets: list[ShapeTargets]) -> str:
    counts = [t.num_echoes for t in targets]
    low, high = min(counts), max(counts)
    return str(low) if low == high else f"{low}-{high} (mean {np.mean(counts):.1f})"


def _cmd_targets(args: argparse.Namespace) -> int:
    config = _load_config(args)
    targets = make_targets(config)
    save_targets(args.output, config, targets)
    crossings = [t.crossing_times_s for t in targets]
    first, last = min(c[0] for c in crossings), max(c[-1] for c in crossings)
    print(
        f"wrote {args.output}: {len(targets)} shapes, {_echo_range(targets)} echoes each, "
        f"crossings from {first * 1e3:.2f} to {last * 1e3:.2f} ms (config {config.digest()})"
    )
    return 0


def _cmd_shapes(args: argparse.Namespace) -> int:
    config = _load_config(args)
    shapes = optimize_shapes(config, jobs=args.jobs)
    save_shapes(args.output, config, shapes)
    print(
        f"{'shape':>5} {'echoes':>6} {'max |g| mT/m':>13} {'max slew T/m/s':>15} "
        f"{'k error':>9} {'solve s':>8}"
    )
    for i, shape in enumerate(shapes):
        m = shape_metrics(config, shape)
        print(
            f"{i:5d} {shape.targets.num_echoes:6d} {m.max_grad_mT_per_m:13.2f} "
            f"{m.max_slew_T_per_m_per_s:15.2f} {m.max_target_error_kmax:9.1e} "
            f"{shape.solve_time_s:8.2f}"
        )
    print(
        f"wrote {args.output}: {len(shapes)} shapes, "
        f"{_echo_range([s.targets for s in shapes])} echoes each (config {config.digest()})"
    )
    return 0


def _cmd_sequence(args: argparse.Namespace) -> int:
    config, shapes = load_shapes(args.shapes)
    schedule = make_schedule(config)
    seq = build_sequence(config, shapes, schedule)
    output: Path = args.output
    seq.write(str(output))
    design_path = output.with_suffix(".npz")
    save_design(design_path, config, schedule, shapes)
    print(
        f"wrote {output}: {schedule.num_trs} TRs ({config.scan.num_dummy_trs} dummy), "
        f"{seq.duration()[0]:.1f} s; design in {design_path} (config {config.digest()})"
    )
    return 0


def _load_seq_and_design(
    seq_path: Path,
) -> tuple[pp.Sequence, Config, Schedule, list[Shape]]:
    config, schedule, shapes = load_design(seq_path.with_suffix(".npz"))
    seq = pp.Sequence(system=make_opts(config))
    seq.read(str(seq_path))
    digest = read_seq_definition(seq_path, "ConfigDigest")
    if digest != config.digest():
        raise ValueError(
            f"{seq_path} has config digest {digest!r}, but its design file has {config.digest()!r}"
        )
    return seq, config, schedule, shapes


def _parse_band(text: str) -> tuple[float, float]:
    low, sep, high = text.partition(":")
    try:
        band = (float(low), float(high))
    except ValueError:
        band = (0.0, -1.0)
    if not sep or not 0 <= band[0] < band[1]:
        raise argparse.ArgumentTypeError(f"band {text!r} is not LOW:HIGH in Hz")
    return band


def _cmd_check(args: argparse.Namespace) -> int:
    seq, config, schedule, shapes = _load_seq_and_design(args.seq)
    time_range = acquired_time_range(config, args.trs)
    bands = list(args.band)
    if args.asc and not bands:
        bands = acoustic_resonances(args.asc)
    report: dict[str, Any] = {
        "config_digest": config.digest(),
        "timing": timing_report(seq),
        "gradients": gradient_report(config, schedule, shapes),
        "spectrum": spectrum_report(seq, bands, time_range),
    }
    if args.asc:
        report["pns"] = pns_report(config, schedule, shapes, args.asc)
    print(json.dumps(report, indent=2) if args.json else format_report(report))
    failed = not report["timing"]["ok"] or not report["gradients"]["nyquist_ok"]
    failed |= "pns" in report and not report["pns"]["ok"]
    return 1 if failed else 0


def _cmd_export_trajectory(args: argparse.Namespace) -> int:
    seq, config, schedule, shapes = _load_seq_and_design(args.seq)
    trajectory = make_trajectory(config, schedule, shapes)
    # The .seq file stores waveforms and rotations as rounded text, so the
    # scanner's k-space differs slightly from the design's; allow 1e-3 of a
    # k-space pixel (1 / FOV).
    pixel = 1.0 / config.geometry.fov_m
    error = verify_trajectory(seq, trajectory, args.verify_trs) / pixel
    if error > 1e-3:
        print(
            f"error: the trajectory differs from pypulseq's k-space by {error:.3g} "
            "k-space pixels (1/FOV)",
            file=sys.stderr,
        )
        return 1
    save_trajectory(args.output, config, trajectory)
    print(
        f"wrote {args.output}: {len(trajectory.tr_index)} TRs x "
        f"{len(trajectory.adc_time_in_tr_s)} samples; matches pypulseq's k-space of "
        f"{args.seq} within {error:.1e} k-space pixels over "
        f"{min(args.verify_trs, len(trajectory.tr_index))} TRs"
    )
    return 0


def _cmd_to_ismrmrd(args: argparse.Namespace) -> int:
    try:
        from relaxed_rosettes.recon.rawdata import write_ismrmrd
        from relaxed_rosettes.recon.twix import read_twix
    except ImportError as e:
        print(f"error: to-ismrmrd needs the recon extra ({e})", file=sys.stderr)
        return 1
    _, config, schedule, shapes = _load_seq_and_design(args.seq)
    signature = read_seq_signature(args.seq)
    twix = read_twix(args.meas, config)
    if not twix.has_signature(signature):
        message = f"the header of {args.meas} does not have the signature {signature} of {args.seq}"
        if not args.no_signature_check:
            print(f"error: {message} (--no-signature-check to convert anyway)", file=sys.stderr)
            return 1
        print(f"warning: {message}", file=sys.stderr)
    trajectory = make_trajectory(config, schedule, shapes)
    report = write_ismrmrd(
        args.output,
        config,
        trajectory,
        twix,
        signature,
        allow_partial=args.allow_partial,
        embed_trajectory=not args.no_trajectory,
    )
    print(
        f"wrote {args.output}: {report.trs_written} of {report.trs_expected} TRs, "
        f"{report.coils} coils, {report.noise_acquisitions} noise readouts "
        f"(config {config.digest()}, signature {signature})"
    )
    if report.tr_spacing_error_s > config.readout.tr_s / 2:
        print(
            f"warning: the time stamps of consecutive TRs differ from the TR by up to "
            f"{report.tr_spacing_error_s * 1e3:.1f} ms",
            file=sys.stderr,
        )
    return 0


def _cmd_phantom_init(args: argparse.Namespace) -> int:
    from relaxed_rosettes.recon.phantom import PhantomConfig

    path: Path = args.path
    if path.exists() and not args.force:
        print(f"error: {path} exists (use --force to overwrite)", file=sys.stderr)
        return 1
    path.write_text(PhantomConfig().to_toml())
    print(f"wrote {path}")
    return 0


def _cmd_phantom_show(args: argparse.Namespace) -> int:
    from relaxed_rosettes.recon.phantom import PhantomConfig

    phantom = PhantomConfig.load(args.phantom) if args.phantom else PhantomConfig()
    print(f"# phantom digest: {phantom.digest()}")
    print(phantom.to_toml(), end="")
    return 0


def _cmd_simulate(args: argparse.Namespace) -> int:
    try:
        from relaxed_rosettes.recon.phantom import PhantomConfig, simulate_noise, simulate_trs
        from relaxed_rosettes.recon.rawdata import (
            PHANTOM_DIGEST,
            PHANTOM_TOML,
            write_simulated_ismrmrd,
        )
    except ImportError as e:
        print(f"error: simulate needs the recon extra ({e})", file=sys.stderr)
        return 1
    phantom = PhantomConfig.load(args.phantom)
    _, config, schedule, shapes = _load_seq_and_design(args.seq)
    signature = read_seq_signature(args.seq)
    trajectory = make_trajectory(config, schedule, shapes)
    coils = write_simulated_ismrmrd(
        args.output,
        config,
        trajectory,
        signature,
        simulate_trs(phantom, trajectory, batch_trs=args.batch_trs),
        simulate_noise(phantom, config.readout.adc_num_samples),
        extra_strings={PHANTOM_TOML: phantom.to_toml(), PHANTOM_DIGEST: phantom.digest()},
        embed_trajectory=not args.no_trajectory,
    )
    print(
        f"wrote {args.output}: {len(trajectory.tr_index)} TRs, {coils} coils, "
        f"{phantom.noise_readouts} noise readouts (config {config.digest()}, "
        f"phantom {phantom.digest()}, signature {signature})"
    )
    return 0


def _cmd_grid(args: argparse.Namespace) -> int:
    try:
        from relaxed_rosettes.recon.gridding import reconstruct, save_gridding
        from relaxed_rosettes.recon.rawdata import load_rawdata
    except ImportError as e:
        print(f"error: grid needs the recon extra ({e})", file=sys.stderr)
        return 1
    recon = _load_recon_config(args)
    raw = load_rawdata(args.raw)
    design, trajectory = load_trajectory(args.trajectory)
    result = reconstruct(raw, design, trajectory, recon)
    save_gridding(args.output, result, coil_images=args.coil_images)
    n = result.rss.shape[0]
    print(
        f"wrote {args.output}: {n}^3 at {recon.data.resolution_m * 1e3:g} mm from "
        f"{result.num_trs} TRs, {result.num_samples} samples, "
        f"{result.coils.num_virtual_coils} virtual coils; DCF change "
        f"{result.dcf_change[-1] if len(result.dcf_change) else 0.0:.3g} "
        f"(recon config {recon.digest()})"
    )
    return 0


def _cmd_sense(args: argparse.Namespace) -> int:
    try:
        from relaxed_rosettes.recon.rawdata import load_rawdata
        from relaxed_rosettes.recon.sense import reconstruct, save_sense
    except ImportError as e:
        print(f"error: sense needs the recon extra ({e})", file=sys.stderr)
        return 1
    recon = _load_recon_config(args)
    raw = load_rawdata(args.raw)
    design, trajectory = load_trajectory(args.trajectory)
    result = reconstruct(raw, design, trajectory, recon)
    save_sense(args.output, result, coil_maps=args.coil_maps)
    n = result.image.shape[0]
    b0 = "no off-resonance term"
    if result.field_map_hz is not None and result.segmentation is not None:
        f = result.field_map_hz
        b0 = (
            f"field map {f.min():.0f} to {f.max():.0f} Hz, "
            f"{len(result.segmentation.segment_tau_s)} segments "
            f"(RMS error {result.segmentation.rms_error:.1e})"
        )
    print(
        f"wrote {args.output}: {n}^3 at {recon.data.resolution_m * 1e3:g} mm from "
        f"{result.num_trs} TRs, {result.num_samples} samples; {b0}; "
        f"{len(result.residual) - 1} CG iterations, residual {result.residual[-1]:.1e} "
        f"(recon config {recon.digest()})"
    )
    return 0


def _parse_floats(text: str) -> list[float]:
    try:
        values = [float(v) for v in text.split(",")]
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a comma-separated list") from None
    if not values or min(values) <= 0:
        raise argparse.ArgumentTypeError(f"{text!r}: values must be > 0")
    return values


def _cmd_coverage(args: argparse.Namespace) -> int:
    _, config, schedule, shapes = _load_seq_and_design(args.seq)
    trajectory = make_trajectory(config, schedule, shapes)
    resolutions = np.array(sorted(args.resolutions_mm, reverse=True)) * 1e-3
    metrics = window_metrics(
        config,
        trajectory,
        sorted(args.windows),
        resolutions,
        num_starts=args.starts,
        tau_bins=args.tau_bins,
    )
    ktau = ktau_metrics(config, trajectory, shells=args.shells, tau_bin_s=args.tau_bin_ms * 1e-3)
    used = np.unique(trajectory.shape_index)
    redundancy = within_tr_redundancy(
        [trajectory.k_shape_per_m[i] for i in used], 1.0 / config.geometry.fov_m
    )
    if args.json:
        report = {
            "config_digest": config.digest(),
            "ktau": {
                "shells": args.shells,
                "tau_bin_s": ktau.tau_bin_s,
                "empty_fraction": ktau.empty_fraction,
                "empty_fraction_early": ktau.empty_fraction_early,
                "early_s": ktau.early_s,
                "empty_fraction_outer_half": ktau.empty_fraction_outer_half,
                "max_tau_gap_s": ktau.max_tau_gap_s.tolist(),
            },
            "within_tr_redundancy": redundancy.tolist(),
            "windows_s": metrics.windows_s,
            "resolutions_m": metrics.resolutions_m.tolist(),
            "tau_bins": metrics.tau_bins,
            "coverage": metrics.coverage.tolist(),
            "gap_percentile": metrics.gap_percentile,
            "gap_cells": metrics.gap_cells.tolist(),
        }
        print(json.dumps(report, indent=2))
    else:
        print(format_metrics(metrics))
        print()
        print(format_ktau(ktau))
        print(format_redundancy(redundancy))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="relaxed-rosettes",
        description="Design and build the relaxed rosette pulse sequence.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    config = commands.add_parser("config", help="write or show a design config")
    config_commands = config.add_subparsers(dest="config_command", required=True)

    init = config_commands.add_parser("init", help="write a config file with the defaults")
    init.add_argument("path", type=Path, help="TOML file to write")
    init.add_argument("--force", action="store_true", help="overwrite an existing file")
    init.set_defaults(func=_cmd_config_init)

    show = config_commands.add_parser(
        "show", help="validate a config and print it with all defaults filled in"
    )
    _add_config_args(show)
    show.set_defaults(func=_cmd_config_show)

    targets = commands.add_parser(
        "targets", help="generate the k-space targets and times of every shape"
    )
    _add_config_args(targets)
    targets.add_argument("-o", "--output", type=Path, required=True, help="output .npz file")
    targets.set_defaults(func=_cmd_targets)

    shapes = commands.add_parser("shapes", help="optimize the gradient waveform of every shape")
    _add_config_args(shapes)
    shapes.add_argument("-o", "--output", type=Path, required=True, help="output .npz file")
    shapes.add_argument(
        "--jobs", type=int, default=None, help="worker processes (default: CPU count)"
    )
    shapes.set_defaults(func=_cmd_shapes)

    sequence = commands.add_parser(
        "sequence",
        help="build the .seq file from optimized shapes, and its design file (.npz)",
    )
    sequence.add_argument("shapes", type=Path, help=".npz file from `relaxed-rosettes shapes`")
    sequence.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="output .seq file; the design file goes next to it, with the suffix .npz",
    )
    sequence.set_defaults(func=_cmd_sequence)

    check = commands.add_parser(
        "check", help="check a .seq file: timing, gradients, Nyquist, spectrum, PNS"
    )
    check.add_argument("seq", type=Path, help=".seq file, with its design .npz next to it")
    check.add_argument(
        "--asc",
        type=Path,
        help="Siemens gradient system file (MP_GradSys_*.asc, with its *_GSWD_SAFETY.asc "
        "next to it) for the PNS check; without --band, its acoustic resonances "
        "are the spectrum bands",
    )
    check.add_argument(
        "--band",
        type=_parse_band,
        action="append",
        default=[],
        metavar="LOW:HIGH",
        help="frequency band (Hz) to report the gradient spectrum in (repeatable)",
    )
    check.add_argument(
        "--trs",
        type=int,
        default=100,
        help="acquired TRs for the spectrum check (default: 100); PNS covers every TR",
    )
    check.add_argument("--json", action="store_true", help="print the report as JSON")
    check.set_defaults(func=_cmd_check)

    export = commands.add_parser(
        "export-traj", help="export the k-space trajectory at the ADC samples (.npz)"
    )
    export.add_argument("seq", type=Path, help=".seq file, with its design .npz next to it")
    export.add_argument("-o", "--output", type=Path, required=True, help="output .npz file")
    export.add_argument(
        "--verify-trs",
        type=int,
        default=2,
        help="acquired TRs to check against pypulseq's k-space (default: 2)",
    )
    export.set_defaults(func=_cmd_export_trajectory)

    recon_config = commands.add_parser("recon-config", help="write or show a recon config")
    recon_config_commands = recon_config.add_subparsers(dest="recon_config_command", required=True)
    recon_init = recon_config_commands.add_parser(
        "init", help="write a recon config file with the defaults"
    )
    recon_init.add_argument("path", type=Path, help="TOML file to write")
    recon_init.add_argument("--force", action="store_true", help="overwrite an existing file")
    recon_init.set_defaults(func=_cmd_recon_config_init)
    recon_show = recon_config_commands.add_parser(
        "show", help="validate a recon config and print it with all defaults filled in"
    )
    _add_recon_config_args(recon_show, positional=True)
    recon_show.set_defaults(func=_cmd_recon_config_show)

    grid = commands.add_parser(
        "grid",
        help="time-averaged gridding reconstruction of ISMRMRD raw data (recon extra)",
    )
    grid.add_argument("raw", type=Path, help="ISMRMRD file from `relaxed-rosettes to-ismrmrd`")
    grid.add_argument(
        "trajectory", type=Path, help="trajectory .npz from `relaxed-rosettes export-traj`"
    )
    _add_recon_config_args(grid, positional=False)
    grid.add_argument("-o", "--output", type=Path, required=True, help="output .npz file")
    grid.add_argument(
        "--coil-images", action="store_true", help="also store the image of each virtual coil"
    )
    grid.set_defaults(func=_cmd_grid)

    sense = commands.add_parser(
        "sense",
        help="static CG-SENSE reconstruction with coil maps and off-resonance (recon extra)",
    )
    sense.add_argument("raw", type=Path, help="ISMRMRD file from `relaxed-rosettes to-ismrmrd`")
    sense.add_argument(
        "trajectory", type=Path, help="trajectory .npz from `relaxed-rosettes export-traj`"
    )
    _add_recon_config_args(sense, positional=False)
    sense.add_argument("-o", "--output", type=Path, required=True, help="output .npz file")
    sense.add_argument("--coil-maps", action="store_true", help="also store the coil maps")
    sense.set_defaults(func=_cmd_sense)

    to_ismrmrd = commands.add_parser(
        "to-ismrmrd",
        help="convert Siemens raw data (meas.dat) of the sequence to ISMRMRD (recon extra)",
    )
    to_ismrmrd.add_argument("meas", type=Path, help="Siemens raw data file (meas.dat)")
    to_ismrmrd.add_argument(
        "seq", type=Path, help=".seq file that was played, with its design .npz next to it"
    )
    to_ismrmrd.add_argument(
        "-o", "--output", type=Path, required=True, help="output ISMRMRD file (.h5; must not exist)"
    )
    to_ismrmrd.add_argument(
        "--allow-partial",
        action="store_true",
        help="convert a scan that has fewer TRs than the design (e.g. stopped early)",
    )
    to_ismrmrd.add_argument(
        "--no-trajectory",
        action="store_true",
        help="do not store the k-space trajectory in each acquisition",
    )
    to_ismrmrd.add_argument(
        "--no-signature-check",
        action="store_true",
        help="convert even if the raw data header does not have the .seq signature",
    )
    to_ismrmrd.set_defaults(func=_cmd_to_ismrmrd)

    phantom = commands.add_parser(
        "phantom", help="write or show a phantom for `relaxed-rosettes simulate`"
    )
    phantom_commands = phantom.add_subparsers(dest="phantom_command", required=True)
    phantom_init = phantom_commands.add_parser(
        "init", help="write a phantom file with the default phantom"
    )
    phantom_init.add_argument("path", type=Path, help="TOML file to write")
    phantom_init.add_argument("--force", action="store_true", help="overwrite an existing file")
    phantom_init.set_defaults(func=_cmd_phantom_init)
    phantom_show = phantom_commands.add_parser(
        "show", help="validate a phantom and print it with all defaults filled in"
    )
    phantom_show.add_argument(
        "phantom", nargs="?", type=Path, help="phantom TOML file (default: built-in phantom)"
    )
    phantom_show.set_defaults(func=_cmd_phantom_show)

    simulate = commands.add_parser(
        "simulate",
        help="simulate a scan of an analytic phantom, as ISMRMRD raw data (recon extra)",
    )
    simulate.add_argument("phantom", type=Path, help="phantom TOML file")
    simulate.add_argument(
        "seq", type=Path, help=".seq file to simulate, with its design .npz next to it"
    )
    simulate.add_argument(
        "-o", "--output", type=Path, required=True, help="output ISMRMRD file (.h5; must not exist)"
    )
    simulate.add_argument(
        "--batch-trs", type=int, default=64, help="TRs simulated at a time (default: 64)"
    )
    simulate.add_argument(
        "--no-trajectory",
        action="store_true",
        help="do not store the k-space trajectory in each acquisition",
    )
    simulate.set_defaults(func=_cmd_simulate)

    coverage = commands.add_parser(
        "coverage",
        help="k-space coverage of windows of scan time, and (|k|, tau) sampling",
    )
    coverage.add_argument("seq", type=Path, help=".seq file, with its design .npz next to it")
    coverage.add_argument(
        "--windows",
        type=_parse_floats,
        default=[0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
        metavar="S,S,...",
        help="window lengths in seconds (default: 0.1,0.2,0.5,1,2,5,10,30)",
    )
    coverage.add_argument(
        "--resolutions-mm",
        type=_parse_floats,
        default=[8.0, 6.0, 4.0, 3.0, 2.0, 1.5, 1.0],
        metavar="MM,MM,...",
        help="resolutions in mm (default: 8,6,4,3,2,1.5,1)",
    )
    coverage.add_argument(
        "--starts", type=int, default=8, help="window start times to sample (default: 8)"
    )
    coverage.add_argument(
        "--tau-bins",
        type=int,
        default=1,
        help="report the worst of this many bins of time since excitation (default: 1)",
    )
    coverage.add_argument(
        "--shells", type=int, default=20, help="|k| shells for the (|k|, tau) sampling"
    )
    coverage.add_argument(
        "--tau-bin-ms",
        type=float,
        default=0.5,
        help="tau bin width (ms) for the (|k|, tau) sampling (default: 0.5)",
    )
    coverage.add_argument("--json", action="store_true", help="print the metrics as JSON")
    coverage.set_defaults(func=_cmd_coverage)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        status: int = args.func(args)
    except (ConfigError, ShapeOptimizationError, OSError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return status


if __name__ == "__main__":
    sys.exit(main())
