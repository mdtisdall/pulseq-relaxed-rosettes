"""Command-line interface: ``relaxed-rosettes <command>``."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pypulseq as pp

from relaxed_rosettes import __version__
from relaxed_rosettes.checks import (
    acquired_time_range,
    format_report,
    gradient_report,
    pns_report,
    spectrum_report,
    timing_report,
)
from relaxed_rosettes.config import Config, ConfigError
from relaxed_rosettes.optimize import (
    Shape,
    ShapeOptimizationError,
    load_shapes,
    optimize_shapes,
    save_shapes,
    shape_metrics,
)
from relaxed_rosettes.sequence import (
    Schedule,
    build_sequence,
    load_design,
    make_schedule,
    save_design,
)
from relaxed_rosettes.system import make_opts
from relaxed_rosettes.targets import make_targets, save_targets
from relaxed_rosettes.trajectory import make_trajectory, save_trajectory, verify_trajectory


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


def _cmd_targets(args: argparse.Namespace) -> int:
    config = _load_config(args)
    targets = make_targets(config)
    save_targets(args.output, config, targets)
    crossings = [t.crossing_times_s for t in targets]
    first, last = min(c[0] for c in crossings), max(c[-1] for c in crossings)
    print(
        f"wrote {args.output}: {len(targets)} shapes, {targets[0].num_echoes} echoes each, "
        f"crossings from {first * 1e3:.2f} to {last * 1e3:.2f} ms (config {config.digest()})"
    )
    return 0


def _cmd_shapes(args: argparse.Namespace) -> int:
    config = _load_config(args)
    shapes = optimize_shapes(config, jobs=args.jobs)
    save_shapes(args.output, config, shapes)
    print(f"{'shape':>5} {'max |g| mT/m':>13} {'max slew T/m/s':>15} {'k error':>9} {'solve s':>8}")
    for i, shape in enumerate(shapes):
        m = shape_metrics(config, shape)
        print(
            f"{i:5d} {m.max_grad_mT_per_m:13.2f} {m.max_slew_T_per_m_per_s:15.2f} "
            f"{m.max_target_error_kmax:9.1e} {shape.solve_time_s:8.2f}"
        )
    print(f"wrote {args.output}: {len(shapes)} shapes (config {config.digest()})")
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
    digest = seq.get_definition("ConfigDigest")
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
    report: dict[str, Any] = {
        "config_digest": config.digest(),
        "timing": timing_report(seq),
        "gradients": gradient_report(config, schedule, shapes),
        "spectrum": spectrum_report(seq, args.band, time_range),
    }
    if args.asc:
        report["pns"] = pns_report(seq, args.asc, time_range)
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
        "--asc", type=Path, help="Siemens gradient hardware .asc file, for the PNS check"
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
        help="acquired TRs for the PNS and spectrum checks (default: 100)",
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
