"""Assembly of the pulseq sequence from the optimized shapes.

Each TR is one block: a hard pulse with the gradients off, one shape's
gradient waveform rotated by a rotation event, and (in acquired TRs) the ADC.
Dummy TRs come first and have no ADC. The RF and ADC phases follow the
configured scheme. Every shape starts and ends at g = 0, so the TRs need no
ramps between them.
"""

from __future__ import annotations

import copy
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pypulseq as pp
from numpy.typing import NDArray

from relaxed_rosettes.config import GAMMA_HZ_PER_T, Config, RfPhase
from relaxed_rosettes.optimize import Shape
from relaxed_rosettes.rotations import acquired_schedule, dummy_schedule
from relaxed_rosettes.system import make_opts
from relaxed_rosettes.targets import ShapeTargets

SCHEDULE_STREAM = 1
"""Mixed into the seed for the schedule's random stream, which is then
independent of the per-shape target streams."""

RF_SPOIL_INCREMENT_DEG = 117.0


@dataclass(frozen=True)
class Schedule:
    """What each TR plays, in order: dummy TRs, then acquired TRs."""

    shape_index: NDArray[np.int64]
    rotation_matrix: NDArray[np.float64]
    rf_phase_rad: NDArray[np.float64]
    acquired: NDArray[np.bool_]

    @property
    def num_trs(self) -> int:
        return len(self.shape_index)


def rf_phases(config: Config, n: int, rng: np.random.Generator) -> NDArray[np.float64]:
    """RF (and ADC) phase of each of ``n`` TRs, in radians."""
    if config.excitation.rf_phase is RfPhase.RANDOM:
        return rng.uniform(0.0, 2.0 * np.pi, n)
    k = np.arange(n)
    return np.deg2rad(np.mod(RF_SPOIL_INCREMENT_DEG * k * (k + 1) / 2, 360.0))


def make_schedule(config: Config) -> Schedule:
    """The TR schedule of a config."""
    rng = np.random.default_rng([config.seed, SCHEDULE_STREAM])
    dummy_shapes, dummy_rotations = dummy_schedule(config, rng)
    shapes, rotations = acquired_schedule(config, rng)
    matrices = rotations.as_matrix()
    if dummy_rotations is not None:
        matrices = np.concatenate([dummy_rotations.as_matrix(), matrices])
    n_dummy = config.scan.num_dummy_trs
    acquired = np.arange(n_dummy + config.scan.num_trs) >= n_dummy
    return Schedule(
        shape_index=np.concatenate([dummy_shapes, shapes]).astype(np.int64),
        rotation_matrix=matrices,
        rf_phase_rad=rf_phases(config, len(acquired), rng),
        acquired=acquired,
    )


def check_adc_segments(config: Config, system: pp.Opts) -> tuple[int, int]:
    """ADC segments and samples per segment; fail if the scanner cannot split it.

    A Siemens ADC longer than ``adc_samples_limit`` samples is split into
    equal segments, each a multiple of the gradient raster and of
    ``adc_samples_divisor`` samples.
    """
    r = config.readout
    segments, per_segment = pp.calc_adc_segments(r.adc_num_samples, r.adc_dwell_s, system)
    if segments * per_segment != r.adc_num_samples:
        raise ValueError(
            f"readout.adc_num_samples = {r.adc_num_samples} cannot be split into ADC "
            f"segments (pypulseq suggests {segments} x {per_segment} = "
            f"{segments * per_segment})"
        )
    return int(segments), int(per_segment)


def build_sequence(config: Config, shapes: list[Shape], schedule: Schedule) -> pp.Sequence:
    """The pulseq sequence of the schedule."""
    system = make_opts(config)
    check_adc_segments(config, system)
    rf = pp.make_block_pulse(
        flip_angle=np.deg2rad(config.excitation.flip_angle_deg),
        duration=config.excitation.rf_duration_s,
        delay=config.system.rf_dead_time_s,
        system=system,
        use="excitation",
    )
    adc = pp.make_adc(
        num_samples=config.readout.adc_num_samples,
        dwell=config.readout.adc_dwell_s,
        delay=config.adc_start_s,
        system=system,
    )
    grads = [
        [
            pp.make_arbitrary_grad(
                channel=channel,
                waveform=shape.g_mT_per_m[:, axis] * 1e-3 * GAMMA_HZ_PER_T,
                first=0,
                last=0,
                delay=0,
                system=system,
            )
            for axis, channel in enumerate("xyz")
        ]
        for shape in shapes
    ]

    seq = pp.Sequence(system=system)
    for tr in range(schedule.num_trs):
        phase = float(schedule.rf_phase_rad[tr])
        rf_tr = copy.copy(rf)
        rf_tr.phase_offset = phase
        events = [
            rf_tr,
            *grads[schedule.shape_index[tr]],
            pp.make_rotation(schedule.rotation_matrix[tr]),
        ]
        if schedule.acquired[tr]:
            adc_tr = copy.copy(adc)
            adc_tr.phase_offset = phase
            events.append(adc_tr)
        seq.add_block(*events)

    fov = config.geometry.fov_m
    seq.set_definition("Name", "relaxed_rosettes")
    seq.set_definition("FOV", [fov, fov, fov])
    seq.set_definition("TR", config.readout.tr_s)
    seq.set_definition("ConfigDigest", config.digest())
    seq.set_definition("NumDummyTRs", config.scan.num_dummy_trs)

    ok, report = seq.check_timing()
    if not ok:
        raise ValueError("sequence timing check failed:\n" + "".join(str(r) for r in report))
    return seq


def read_seq_signature(path: Path) -> str:
    """The MD5 signature of a ``.seq`` file (lowercase hex), checked against its content.

    pypulseq signs a ``.seq`` file with the MD5 of everything before the
    newline that precedes ``[SIGNATURE]``. The Siemens interpreter records the
    signature in the raw data header, which links the raw data to this file.
    """
    text = path.read_bytes()
    start = text.rfind(b"\n[SIGNATURE]")
    if start < 0:
        raise ValueError(f"{path} has no [SIGNATURE] section")
    match = re.search(rb"^Hash\s+([0-9a-fA-F]{32})\s*$", text[start:], re.MULTILINE)
    if match is None:
        raise ValueError(f"{path}: the [SIGNATURE] section has no MD5 hash")
    signature = match.group(1).decode().lower()
    if hashlib.md5(text[:start]).hexdigest() != signature:
        raise ValueError(f"{path}: the content does not match its signature")
    return signature


def save_design(path: Path, config: Config, schedule: Schedule, shapes: list[Shape]) -> None:
    """Write everything a ``.seq`` file was built from to an ``.npz`` file.

    Schedule arrays (one entry per TR, dummies first): ``shape_index``,
    ``rotation_matrix`` (TRs, 3, 3; physical = R @ shape gradient),
    ``rf_phase_rad`` (also the ADC phase), ``acquired``. Shape arrays:
    ``g_mT_per_m`` (shapes, raster points, 3), the targets ``k`` (shapes,
    targets, 3; units of k_max) and ``t_s``. And ``config_toml``,
    ``config_digest``.
    """
    np.savez(
        path,
        shape_index=schedule.shape_index,
        rotation_matrix=schedule.rotation_matrix,
        rf_phase_rad=schedule.rf_phase_rad,
        acquired=schedule.acquired,
        g_mT_per_m=np.stack([s.g_mT_per_m for s in shapes]),
        k=np.stack([s.targets.k for s in shapes]),
        t_s=np.stack([s.targets.t_s for s in shapes]),
        config_toml=config.to_toml(),
        config_digest=config.digest(),
    )


def load_design(path: Path) -> tuple[Config, Schedule, list[Shape]]:
    """Read a file written by ``save_design``."""
    with np.load(path) as data:
        config = Config.from_toml(str(data["config_toml"]))
        if str(data["config_digest"]) != config.digest():
            raise ValueError(f"{path}: config digest does not match its config")
        schedule = Schedule(
            shape_index=data["shape_index"],
            rotation_matrix=data["rotation_matrix"],
            rf_phase_rad=data["rf_phase_rad"],
            acquired=data["acquired"],
        )
        shapes = [
            Shape(
                g_mT_per_m=g,
                targets=ShapeTargets(k=k, t_s=t),
                status="optimal",
                solve_time_s=float("nan"),
            )
            for g, k, t in zip(data["g_mT_per_m"], data["k"], data["t_s"], strict=True)
        ]
    return config, schedule, shapes
