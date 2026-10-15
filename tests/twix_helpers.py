"""Test helpers: small designs built through the CLI, and synthetic meas.dat files."""

import struct
from pathlib import Path

import numpy as np
import twixtools
from numpy.typing import NDArray
from twixtools.mdb import Mdb_local

from relaxed_rosettes.cli import main
from relaxed_rosettes.config import Config
from relaxed_rosettes.recon.twix import TIMESTAMP_TICK_S

NOISE_DWELL_NS = 5000


def build_design(directory: Path, config: Config) -> tuple[Path, Path]:
    """Build a design through the CLI: the .seq file (design .npz next to it) and trajectory."""
    config_path = directory / "design.toml"
    config_path.write_text(config.to_toml())
    shapes_path = directory / "shapes.npz"
    seq_path = directory / "design.seq"
    traj_path = directory / "traj.npz"
    assert main(["shapes", str(config_path), "-o", str(shapes_path), "--jobs", "1"]) == 0
    assert main(["sequence", str(shapes_path), "-o", str(seq_path)]) == 0
    assert main(["export-traj", str(seq_path), "-o", str(traj_path)]) == 0
    return seq_path, traj_path


def noise_mixing(rng: np.random.Generator, coils: int) -> NDArray[np.complex128]:
    """A coil noise mixing matrix (noise covariance 2 M Mᴴ)."""
    random = rng.standard_normal((coils, coils)) + 1j * rng.standard_normal((coils, coils))
    mixing: NDArray[np.complex128] = np.eye(coils, dtype=np.complex128) + 0.4 * random
    return mixing


def _header(buffers: dict[str, str]) -> NDArray[np.bytes_]:
    """A twix measurement header with the given protocol buffers."""
    body = b"".join(
        name.encode() + b"\x00" + struct.pack("<I", len(text) + 1) + text.encode() + b"\x00"
        for name, text in buffers.items()
    )
    raw = struct.pack("<II", 8 + len(body), len(buffers)) + body
    return np.frombuffer(raw, dtype="S1")


def _ascconv(lines: list[str]) -> str:
    return "### ASCCONV BEGIN ###\n" + "\n".join(lines) + "\n### ASCCONV END ###\n"


def _mdb(data: NDArray[np.complex64], scan_counter: int, time_stamp: int) -> Mdb_local:
    mdb = Mdb_local(data)
    mdb.mdh.ScanCounter = scan_counter
    mdb.mdh.TimeStamp = time_stamp
    mdb.mdh.PMUTimeStamp = time_stamp + 7
    mdb.mdh.MeasUID = 42
    mdb.mdh.SliceData.Quaternion[:] = [1.0, 0.0, 0.0, 0.0]
    mdb.mdh.SliceData.SlicePos.Tra = 12.5
    return mdb


def write_meas(
    path: Path,
    signature: str,
    data: NDArray[np.complex64],
    noise: NDArray[np.complex64],
    *,
    segments: int,
    tr_s: float,
) -> None:
    """A meas.dat file with a noise measurement and a scan.

    ``data`` is (TRs, coils, samples), split into ``segments`` readouts per
    TR; ``noise`` is (noise readouts, coils, samples). The scan header has
    the ``.seq`` signature.
    """
    noise_mdbs = []
    for i, readout in enumerate(noise):
        mdb = _mdb(readout, i + 1, 100)
        mdb.add_flag("NOISEADJSCAN")
        noise_mdbs.append(mdb)
    noise_scan = {
        "hdr_str": _header({"MeasYaps": _ascconv([f"sRXSPEC.alDwellTime[0] = {NOISE_DWELL_NS}"])}),
        "mdb": noise_mdbs,
    }
    segment = data.shape[2] // segments
    tr_ticks = round(tr_s / TIMESTAMP_TICK_S)
    mdbs = [
        _mdb(
            data[t, :, s * segment : (s + 1) * segment],
            segments * t + s + 1,
            1000 + t * tr_ticks,
        )
        for t in range(data.shape[0])
        for s in range(segments)
    ]
    scan = {
        "hdr_str": _header(
            {
                "Dicom": '<ParamLong."lFrequency">  { 123200000  }\n'
                '<ParamDouble."flMagneticFieldStrength">  { <Precision> 6  2.89 }\n',
                "MeasYaps": _ascconv([f'sWipMemBlock.tFree = "{signature}"']),
            }
        ),
        "mdb": mdbs,
    }
    twixtools.write_twix([noise_scan, scan], str(path))
