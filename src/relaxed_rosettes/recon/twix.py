"""Siemens raw data (``meas.dat``) of a relaxed rosette scan, for conversion to ISMRMRD.

A ``meas.dat`` file holds one or more measurements. The last is the scan; the
earlier ones are adjustments, of which only the noise scans (``NOISEADJSCAN``)
are kept. The Siemens interpreter splits an ADC longer than
``adc_samples_limit`` samples into equal segments, each its own readout, so
the readouts of the scan are grouped into one readout per acquired TR, in
order. The interpreter applies the ADC phase offset, so the data of each TR
are already demodulated by its RF phase.

The data are read from the file only when ``TwixReadout.data`` or
``NoiseScan.data`` is called, so a large file needs little memory.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import twixtools
from numpy.typing import NDArray

from relaxed_rosettes.config import Config
from relaxed_rosettes.sequence import check_adc_segments
from relaxed_rosettes.system import make_opts

TIMESTAMP_TICK_S = 2.5e-3
"""Unit of the Siemens MDH ``TimeStamp`` and ``PMUTimeStamp``."""

ComplexArray = NDArray[np.complex64]


@dataclass(frozen=True)
class TwixReadout:
    """The ADC segments of one acquired TR, in order."""

    segments: list[Any]
    """twixtools ``Mdb`` of each segment."""

    @property
    def mdh(self) -> Any:
        """MDH of the first segment."""
        return self.segments[0].mdh

    def data(self) -> ComplexArray:
        """(coils, samples) data of the TR: the segments joined in time."""
        return np.concatenate([np.atleast_2d(m.data) for m in self.segments], axis=1)


@dataclass(frozen=True)
class NoiseScan:
    """One noise readout, from any measurement in the file."""

    mdb: Any
    dwell_s: float | None
    """Dwell time from the protocol of its measurement, if the header has it."""

    def data(self) -> ComplexArray:
        """(coils, samples) noise samples."""
        return np.atleast_2d(self.mdb.data)


@dataclass(frozen=True)
class TwixScan:
    """The readouts of a relaxed rosette scan and its header."""

    readouts: list[TwixReadout]
    noise: list[NoiseScan]
    header: dict[str, Any]
    """Parsed protocol of the scan measurement (twixtools ``hdr``)."""
    header_text: str
    """All raw protocol buffers of the scan measurement, joined."""
    num_coils: int

    def has_signature(self, signature: str) -> bool:
        """Whether the protocol has the ``.seq`` signature (hex, any case) in it."""
        return signature.lower() in self.header_text.lower()

    def header_value(self, *keys: str | int) -> Any:
        """A value of the parsed protocol, or None if it is missing."""
        value: Any = self.header
        for key in keys:
            try:
                value = value[key]
            except (KeyError, IndexError, TypeError):
                return None
        return value


def read_twix(path: Path, config: Config) -> TwixScan:
    """Read the readouts of a scan of ``config`` from a ``meas.dat`` file.

    Fails if a readout does not have the samples of one ADC segment, if the
    readouts do not make whole TRs, or if the coil count changes.
    """
    measurements = twixtools.read_twix(
        str(path),
        parse_pmu=False,
        parse_geometry=False,
        keep_syncdata=False,
        verbose=False,
    )
    if not measurements:
        raise ValueError(f"{path} has no measurements")
    scan = measurements[-1]

    noise = [
        NoiseScan(mdb=m, dwell_s=_dwell_s(meas.get("hdr", {})))
        for meas in measurements
        for m in meas["mdb"]
        if m.is_flag_set("NOISEADJSCAN")
    ]

    segments, per_segment = check_adc_segments(config, make_opts(config))
    imaging = [m for m in scan["mdb"] if m.is_image_scan()]
    for i, m in enumerate(imaging):
        if m.mdh.SamplesInScan != per_segment:
            raise ValueError(
                f"{path}: readout {i} has {m.mdh.SamplesInScan} samples; the design's ADC "
                f"has {segments} segment(s) of {per_segment}"
            )
    if len(imaging) % segments:
        raise ValueError(
            f"{path}: {len(imaging)} readouts is not a whole number of TRs "
            f"({segments} ADC segments each)"
        )
    readouts = [
        TwixReadout(segments=imaging[i : i + segments]) for i in range(0, len(imaging), segments)
    ]
    coils = {int(m.mdh.UsedChannels) for m in imaging}
    if len(coils) > 1:
        raise ValueError(f"{path}: the coil count changes between readouts: {sorted(coils)}")

    header = scan.get("hdr", {})
    header_text = "\n".join(v for k, v in header.items() if k.endswith("_raw"))
    return TwixScan(
        readouts=readouts,
        noise=noise,
        header=header,
        header_text=header_text,
        num_coils=coils.pop() if coils else 0,
    )


def _dwell_s(header: dict[str, Any]) -> float | None:
    try:
        dwell_ns = header["MeasYaps"]["sRXSPEC"]["alDwellTime"][0]
    except (KeyError, IndexError, TypeError):
        return None
    return float(dwell_ns) * 1e-9
