"""ISMRMRD raw data of a relaxed rosette scan: writing from twix or a simulation, and loading.

The reconstruction reads only ISMRMRD. ``write_ismrmrd`` writes one
acquisition for each acquired TR, in order, with its ADC segments joined, and
one for each noise scan (flag ``ACQ_IS_NOISE_MEASUREMENT``).
``write_simulated_ismrmrd`` writes simulated data in the same layout. In a TR
acquisition:

- ``scan_counter`` is the acquired TR number (0, 1, ...), ``user_int[0]`` its
  index in the full schedule (dummy TRs included), and ``user_int[1]`` the
  shape it plays;
- ``acquisition_time_stamp`` and ``physiology_time_stamp[0]`` are the Siemens
  MDH time stamps (ticks of ``TIMESTAMP_TICK_S``);
- ``traj`` (unless left out) is the k-space position of each sample in cycles
  per FOV (``k * fov``, the unit BART uses), on the physical gradient axes
  (pypulseq x, y, z after the rotation event);
- ``position`` and the direction vectors come from the MDH slice data, as
  ``siemens_to_ismrmrd`` sets them.

In a simulation, the time stamps are the TR start times, ``position`` is 0,
and the directions are the gradient axes.

The XML header has the design config (TOML) and its digest, and the
``.seq`` signature, as user parameters, so ``load_rawdata`` can check a file
against a trajectory. No patient information is copied from the twix header.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import ismrmrd
import ismrmrd.xsd
import numpy as np
from ismrmrd.util import quaternion_to_directions
from numpy.typing import NDArray

from relaxed_rosettes.config import GAMMA_HZ_PER_T, Config
from relaxed_rosettes.recon.twix import TIMESTAMP_TICK_S, TwixScan
from relaxed_rosettes.sequence import check_adc_segments
from relaxed_rosettes.system import make_opts
from relaxed_rosettes.trajectory import Trajectory

CONFIG_TOML = "relaxed_rosettes_config_toml"
CONFIG_DIGEST = "relaxed_rosettes_config_digest"
SEQ_SIGNATURE = "pulseq_signature_md5"
PHANTOM_TOML = "relaxed_rosettes_phantom_toml"
PHANTOM_DIGEST = "relaxed_rosettes_phantom_digest"
SIMULATED_GRADIENT_DELAYS = "relaxed_rosettes_simulated_gradient_delays_s"
TRAJECTORY_UNITS = "trajectory_units"
ADC_SEGMENTS = "adc_segments_per_tr"
TIMESTAMP_TICK = "time_stamp_tick_s"

TRAJECTORY_UNITS_VALUE = "cycles/FOV, physical gradient axes"

_NOISE_FLAG = 1 << (ismrmrd.ACQ_IS_NOISE_MEASUREMENT - 1)

ComplexArray = NDArray[np.complex64]

_CHUNK_ROWS = 32
"""Acquisitions read from the file at a time.

Reads take whole records (``dset[a:b]``), never ``dset.fields(...)``: h5py
3.16 does not free the variable-length members (``traj``, ``data``) that a
field read leaves out, so each pass over a file would leak them."""


def _record_slices(rows: NDArray[np.int64]) -> Iterator[slice]:
    """Slices of at most ``_CHUNK_ROWS`` rows that cover sorted, unique ``rows``."""
    for run in np.split(rows, np.flatnonzero(np.diff(rows) != 1) + 1):
        for start in range(0, len(run), _CHUNK_ROWS):
            chunk = run[start : start + _CHUNK_ROWS]
            yield slice(int(chunk[0]), int(chunk[-1]) + 1)


@dataclass(frozen=True)
class ConversionReport:
    """What ``write_ismrmrd`` wrote."""

    trs_written: int
    trs_expected: int
    noise_acquisitions: int
    coils: int
    tr_spacing_error_s: float
    """Largest difference between the time stamp spacing of consecutive TRs
    and the TR (0 with fewer than two TRs). One tick is 2.5 ms."""


def write_ismrmrd(
    path: Path,
    config: Config,
    trajectory: Trajectory,
    twix: TwixScan,
    signature: str,
    *,
    allow_partial: bool = False,
    embed_trajectory: bool = True,
) -> ConversionReport:
    """Write the twix scan of a design to a new ISMRMRD file.

    Fails if the scan has more TRs than the design, or fewer unless
    ``allow_partial`` (e.g. a scan stopped early; then only the TRs it has
    are written).
    """
    expected = len(trajectory.tr_index)
    found = len(twix.readouts)
    if found > expected or (found < expected and not allow_partial):
        raise ValueError(
            f"the raw data has {found} TRs, but the design acquires {expected}"
            + (" (use allow_partial for a scan stopped early)" if found < expected else "")
        )
    if found == 0:
        raise ValueError("the raw data has no TRs")
    segments = len(twix.readouts[0].segments)

    frequency = twix.header_value("Dicom", "lFrequency")
    field_strength = twix.header_value("Dicom", "flMagneticFieldStrength")
    model = twix.header_value("Dicom", "ManufacturersModelName")
    xml = _xml_header(
        config,
        signature,
        segments,
        coils=twix.num_coils,
        vendor="Siemens",
        frequency_hz=frequency if isinstance(frequency, int) else None,
        field_strength_T=float(field_strength) if isinstance(field_strength, float | int) else None,
        model=model if isinstance(model, str) else None,
    )
    with ismrmrd.Dataset(str(path), "dataset", mode="w-") as dataset:
        dataset.write_xml_header(xml)
        for noise in twix.noise:
            data = noise.data()
            acq = ismrmrd.Acquisition.from_array(data)
            acq.setHead(_head_from_mdh(acq.getHead(), noise.mdb.mdh))
            acq.set_flag(ismrmrd.ACQ_IS_NOISE_MEASUREMENT)
            acq.sample_time_us = 0.0 if noise.dwell_s is None else noise.dwell_s * 1e6
            dataset.append_acquisition(acq)

        fov = config.geometry.fov_m
        for a, readout in enumerate(twix.readouts):
            traj = None
            if embed_trajectory:
                k = trajectory.k_per_m(np.array([a]))[0]
                traj = (k * fov).astype(np.float32)
            acq = ismrmrd.Acquisition.from_array(readout.data(), traj)
            acq.setHead(_head_from_mdh(acq.getHead(), readout.mdh))
            acq.scan_counter = a
            acq.sample_time_us = config.readout.adc_dwell_s * 1e6
            acq.user_int[0] = int(trajectory.tr_index[a])
            acq.user_int[1] = int(trajectory.shape_index[a])
            if a == found - 1:
                acq.set_flag(ismrmrd.ACQ_LAST_IN_MEASUREMENT)
            dataset.append_acquisition(acq)

    stamps = np.array([r.mdh.TimeStamp for r in twix.readouts], dtype=np.float64)
    spacing_error = np.abs(np.diff(stamps) * TIMESTAMP_TICK_S - config.readout.tr_s)
    return ConversionReport(
        trs_written=found,
        trs_expected=expected,
        noise_acquisitions=len(twix.noise),
        coils=twix.num_coils,
        tr_spacing_error_s=float(spacing_error.max()) if len(spacing_error) else 0.0,
    )


def _head_from_mdh(head: Any, mdh: Any) -> Any:
    """Copy the MDH fields that ``siemens_to_ismrmrd`` copies."""
    head.measurement_uid = int(mdh.MeasUID)
    head.acquisition_time_stamp = int(mdh.TimeStamp)
    head.physiology_time_stamp[0] = int(mdh.PMUTimeStamp)
    position = mdh.SliceData.SlicePos
    head.position[:] = [position.Sag, position.Cor, position.Tra]
    w, x, y, z = (float(q) for q in mdh.SliceData.Quaternion)
    # siemens_to_ismrmrd reorders the quaternion to (x, y, z, w) and passes the
    # phase direction where ISMRMRD expects the read direction.
    phase_dir, read_dir, slice_dir = quaternion_to_directions([x, y, z, w])
    head.read_dir[:] = read_dir
    head.phase_dir[:] = phase_dir
    head.slice_dir[:] = slice_dir
    head.patient_table_position[:] = [mdh.PTABPosX, mdh.PTABPosY, mdh.PTABPosZ]
    return head


def write_simulated_ismrmrd(
    path: Path,
    config: Config,
    trajectory: Trajectory,
    signature: str,
    readouts: Iterable[ComplexArray],
    noise: list[ComplexArray],
    *,
    extra_strings: dict[str, str] | None = None,
    embed_trajectory: bool = True,
) -> int:
    """Write simulated data of a design to a new ISMRMRD file; return the coil count.

    ``readouts`` gives the (coils, samples) data of every acquired TR of the
    trajectory, in order; ``noise`` the noise readouts, at the ADC dwell.
    ``extra_strings`` are added to the header as user parameters.
    """
    expected = len(trajectory.tr_index)
    segments, _ = check_adc_segments(config, make_opts(config))
    fov = config.geometry.fov_m
    coils = 0
    with ismrmrd.Dataset(str(path), "dataset", mode="w-") as dataset:
        acquisitions = []
        for data in noise:
            acq = ismrmrd.Acquisition.from_array(data)
            _set_simulated_head(acq, 0.0)
            acq.set_flag(ismrmrd.ACQ_IS_NOISE_MEASUREMENT)
            acq.sample_time_us = config.readout.adc_dwell_s * 1e6
            acquisitions.append(acq)
        a = -1
        for a, data in enumerate(readouts):
            if a >= expected:
                raise ValueError(f"more simulated TRs than the design's {expected}")
            if a == 0:
                coils = data.shape[0]
                xml = _xml_header(
                    config,
                    signature,
                    segments,
                    coils=coils,
                    vendor="relaxed_rosettes simulation",
                    frequency_hz=None,
                    field_strength_T=config.system.B0_T,
                    model=None,
                    extra_strings=extra_strings,
                )
                dataset.write_xml_header(xml)
                for noise_acq in acquisitions:
                    dataset.append_acquisition(noise_acq)
            traj = None
            if embed_trajectory:
                k = trajectory.k_per_m(np.array([a]))[0]
                traj = (k * fov).astype(np.float32)
            acq = ismrmrd.Acquisition.from_array(data, traj)
            _set_simulated_head(acq, float(trajectory.tr_start_s[a]))
            acq.scan_counter = a
            acq.sample_time_us = config.readout.adc_dwell_s * 1e6
            acq.user_int[0] = int(trajectory.tr_index[a])
            acq.user_int[1] = int(trajectory.shape_index[a])
            if a == expected - 1:
                acq.set_flag(ismrmrd.ACQ_LAST_IN_MEASUREMENT)
            dataset.append_acquisition(acq)
        if a != expected - 1:
            raise ValueError(f"simulated {a + 1} TRs, but the design acquires {expected}")
    return coils


def _set_simulated_head(acq: Any, time_s: float) -> None:
    acq.acquisition_time_stamp = round(time_s / TIMESTAMP_TICK_S)
    acq.read_dir[:] = [1.0, 0.0, 0.0]
    acq.phase_dir[:] = [0.0, 1.0, 0.0]
    acq.slice_dir[:] = [0.0, 0.0, 1.0]


def _xml_header(
    config: Config,
    signature: str,
    segments: int,
    *,
    coils: int,
    vendor: str,
    frequency_hz: int | None,
    field_strength_T: float | None,
    model: str | None,
    extra_strings: dict[str, str] | None = None,
) -> str:
    x = ismrmrd.xsd
    fov_mm = config.geometry.fov_m * 1e3
    matrix = round(config.geometry.fov_m / config.geometry.resolution_m)
    space = x.encodingSpaceType(
        matrixSize=x.matrixSizeType(x=matrix, y=matrix, z=matrix),
        fieldOfView_mm=x.fieldOfViewMm(x=fov_mm, y=fov_mm, z=fov_mm),
    )
    if frequency_hz is None:
        frequency_hz = round(GAMMA_HZ_PER_T * config.system.B0_T)
    header = x.ismrmrdHeader(
        experimentalConditions=x.experimentalConditionsType(H1resonanceFrequency_Hz=frequency_hz),
        acquisitionSystemInformation=x.acquisitionSystemInformationType(
            systemVendor=vendor,
            systemModel=model,
            systemFieldStrength_T=field_strength_T,
            receiverChannels=coils,
        ),
        encoding=[
            x.encodingType(
                encodedSpace=space,
                reconSpace=space,
                encodingLimits=x.encodingLimitsType(),
                trajectory=x.trajectoryType.OTHER,
            )
        ],
        sequenceParameters=x.sequenceParametersType(
            TR=[config.readout.tr_s * 1e3],
            flipAngle_deg=[config.excitation.flip_angle_deg],
            sequence_type="relaxed_rosettes",
        ),
        userParameters=x.userParametersType(
            userParameterLong=[x.userParameterLongType(name=ADC_SEGMENTS, value=segments)],
            userParameterDouble=[
                x.userParameterDoubleType(name=TIMESTAMP_TICK, value=TIMESTAMP_TICK_S)
            ],
            userParameterString=[
                x.userParameterStringType(name=CONFIG_TOML, value=config.to_toml()),
                x.userParameterStringType(name=CONFIG_DIGEST, value=config.digest()),
                x.userParameterStringType(name=SEQ_SIGNATURE, value=signature),
                x.userParameterStringType(name=TRAJECTORY_UNITS, value=TRAJECTORY_UNITS_VALUE),
                *(
                    x.userParameterStringType(name=name, value=value)
                    for name, value in (extra_strings or {}).items()
                ),
            ],
        ),
    )
    xml: str = ismrmrd.xsd.ToXML(header)
    return xml


@dataclass(frozen=True)
class RawData:
    """An ISMRMRD file written by ``write_ismrmrd``.

    The headers and noise are read when the file is loaded; the TR data are
    read on demand with ``read``.
    """

    path: Path
    config: Config
    seq_signature: str
    num_coils: int
    tr_index: NDArray[np.int64]
    """(TRs,): index of each TR in the full schedule."""
    shape_index: NDArray[np.int64]
    """(TRs,): shape played in each TR."""
    time_stamp_s: NDArray[np.float64]
    """(TRs,): scanner time stamp of each TR."""
    noise: ComplexArray
    """(coils, samples): all noise samples, joined."""
    noise_dwell_s: float | None
    """Dwell time of the noise samples, if known."""
    _rows: NDArray[np.int64]

    @property
    def num_trs(self) -> int:
        return len(self.tr_index)

    def read(self, trs: slice | NDArray[np.int64] | None = None) -> ComplexArray:
        """(TRs, coils, samples) data of the TRs (default: all)."""
        rows = self._rows[slice(None) if trs is None else trs]
        if len(rows) == 0:
            return np.zeros((0, self.num_coils, 0), dtype=np.complex64)
        unique, first, inverse = np.unique(rows, return_index=True, return_inverse=True)
        out: ComplexArray | None = None
        with h5py.File(self.path, "r") as f:
            dset = f["dataset/data"]
            positions = iter(first)
            for chunk in _record_slices(unique):
                for flat in dset[chunk]["data"]:
                    tr = flat.view(np.complex64).reshape(self.num_coils, -1)
                    if out is None:
                        out = np.empty((len(rows), *tr.shape), np.complex64)
                    out[next(positions)] = tr
        assert out is not None
        repeats = np.flatnonzero(first[inverse] != np.arange(len(rows)))
        out[repeats] = out[first[inverse[repeats]]]
        return out

    def check_trajectory(self, config: Config, trajectory: Trajectory) -> None:
        """Fail unless the raw data and the trajectory come from the same design.

        A partial scan matches the first TRs of the trajectory.
        """
        if config.digest() != self.config.digest():
            raise ValueError(
                f"{self.path} has config digest {self.config.digest()}, "
                f"but the trajectory has {config.digest()}"
            )
        expected = trajectory.tr_index[: self.num_trs]
        if self.num_trs > len(trajectory.tr_index) or not np.array_equal(self.tr_index, expected):
            raise ValueError(f"{self.path}: the TRs do not match the trajectory's")


def load_rawdata(path: Path) -> RawData:
    """Read the headers and noise of a file written by ``write_ismrmrd``."""
    with h5py.File(path, "r") as f:
        if "dataset/xml" not in f or "dataset/data" not in f or len(f["dataset/data"]) == 0:
            raise ValueError(f"{path} is not an ISMRMRD file with acquisitions")
        xml = f["dataset/xml"][0]
        dset = f["dataset/data"]
        chunks, noise_data = [], []
        for chunk in _record_slices(np.arange(len(dset))):
            records = dset[chunk]
            # A copy, so that the variable-length members of the chunk are freed.
            chunks.append(records["head"].copy())
            noise_data.extend(records["data"][(records["head"]["flags"] & _NOISE_FLAG) != 0])
    heads = np.concatenate(chunks)
    is_noise = (heads["flags"] & _NOISE_FLAG) != 0
    noise_rows = np.flatnonzero(is_noise)
    header = ismrmrd.xsd.CreateFromDocument(xml)
    params = header.userParameters
    strings = {p.name: p.value for p in params.userParameterString} if params else {}
    for name in (CONFIG_TOML, CONFIG_DIGEST, SEQ_SIGNATURE):
        if name not in strings:
            raise ValueError(
                f"{path} has no user parameter {name!r}; convert the raw data with "
                "`relaxed-rosettes to-ismrmrd`"
            )
    config = Config.from_toml(strings[CONFIG_TOML])
    if strings[CONFIG_DIGEST] != config.digest():
        raise ValueError(f"{path}: config digest does not match its config")
    doubles = {p.name: p.value for p in params.userParameterDouble}
    tick_s = float(doubles.get(TIMESTAMP_TICK, TIMESTAMP_TICK_S))

    rows = np.flatnonzero(~is_noise)
    tr_heads = heads[rows]
    if not np.array_equal(tr_heads["scan_counter"], np.arange(len(rows))):
        raise ValueError(f"{path}: the TR acquisitions are not in order")
    coils = {int(c) for c in heads["active_channels"]}
    if len(coils) != 1:
        raise ValueError(f"{path}: the coil count is not the same in every acquisition")
    num_coils = coils.pop()

    noise = (
        np.concatenate(
            [flat.view(np.complex64).reshape(num_coils, -1) for flat in noise_data], axis=1
        )
        if len(noise_rows)
        else np.zeros((num_coils, 0), dtype=np.complex64)
    )
    noise_dwell = {float(t) for t in heads["sample_time_us"][noise_rows]}
    return RawData(
        path=path,
        config=config,
        seq_signature=strings[SEQ_SIGNATURE],
        num_coils=num_coils,
        tr_index=tr_heads["user_int"][:, 0].astype(np.int64),
        shape_index=tr_heads["user_int"][:, 1].astype(np.int64),
        time_stamp_s=tr_heads["acquisition_time_stamp"].astype(np.float64) * tick_s,
        noise=noise,
        noise_dwell_s=(
            noise_dwell.pop() * 1e-6 if len(noise_dwell) == 1 and 0.0 not in noise_dwell else None
        ),
        _rows=rows.astype(np.int64),
    )
