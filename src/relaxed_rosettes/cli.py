"""Command-line interface: ``relaxed-rosettes <command>``."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from relaxed_rosettes import __version__
from relaxed_rosettes.config import Config, ConfigError
from relaxed_rosettes.optimize import (
    ShapeOptimizationError,
    load_shapes,
    optimize_shapes,
    save_shapes,
    shape_metrics,
)
from relaxed_rosettes.sequence import build_sequence, make_schedule, save_schedule
from relaxed_rosettes.targets import make_targets, save_targets


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
    schedule_path = output.with_suffix(".npz")
    save_schedule(schedule_path, config, schedule)
    print(
        f"wrote {output}: {schedule.num_trs} TRs ({config.scan.num_dummy_trs} dummy), "
        f"{seq.duration()[0]:.1f} s; schedule in {schedule_path} (config {config.digest()})"
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
        help="build the .seq file from optimized shapes, and its TR schedule (.npz)",
    )
    sequence.add_argument("shapes", type=Path, help=".npz file from `relaxed-rosettes shapes`")
    sequence.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="output .seq file; the schedule goes next to it, with the suffix .npz",
    )
    sequence.set_defaults(func=_cmd_sequence)

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
