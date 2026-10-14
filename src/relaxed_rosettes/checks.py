"""Checks of a built sequence, reported as plain data (no plots).

- Gradients in physical axes: peak amplitude and slew per axis and of the
  vector, over every TR (exact, from the design file).
- Nyquist along the path: the longest ADC dwell that the peak gradient allows.
- pypulseq timing check of the ``.seq`` file.
- PNS (the SAFE model) over every TR, with a Siemens gradient system
  ``.asc`` file (``relaxed_rosettes.pns``).
- The gradient spectrum in given frequency bands (by default the coil's
  acoustic resonances from the ``.asc`` file), over a window of acquired TRs,
  because it samples the waveforms of the whole window.
"""

from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pypulseq as pp

from relaxed_rosettes.config import GAMMA_HZ_PER_T, Config
from relaxed_rosettes.optimize import Shape
from relaxed_rosettes.pns import design_pns
from relaxed_rosettes.sequence import Schedule

SPECTRUM_WINDOW_S = 0.05
"""Window of pypulseq's gradient spectrogram (its default)."""


def gradient_report(config: Config, schedule: Schedule, shapes: list[Shape]) -> dict[str, Any]:
    """Peak gradient and slew in physical axes over every TR, and Nyquist margin."""
    raster = config.system.grad_raster_time_s
    max_g_axis = np.zeros(3)
    max_slew_axis = np.zeros(3)
    max_g = max_slew = 0.0
    for index, shape in enumerate(shapes):
        rotations = schedule.rotation_matrix[schedule.shape_index == index]
        if len(rotations) == 0:
            continue
        # Include the half-raster steps from first = 0 and to last = 0.
        g = np.vstack([np.zeros((1, 3)), shape.g_mT_per_m, np.zeros((1, 3))])
        steps = np.diff(g, axis=0)
        durations = np.full(len(steps), raster)
        durations[[0, -1]] = raster / 2
        slew = steps / durations[:, None] * 1e-3  # T/m/s
        max_g = max(max_g, float(np.linalg.norm(g, axis=1).max()))
        max_slew = max(max_slew, float(np.linalg.norm(slew, axis=1).max()))
        g_physical = np.einsum("rij,nj->rni", rotations, g)
        slew_physical = np.einsum("rij,nj->rni", rotations, slew)
        max_g_axis = np.maximum(max_g_axis, np.abs(g_physical).max(axis=(0, 1)))
        max_slew_axis = np.maximum(max_slew_axis, np.abs(slew_physical).max(axis=(0, 1)))

    max_dwell = 1.0 / (GAMMA_HZ_PER_T * max_g * 1e-3 * config.geometry.fov_m)
    return {
        "max_grad_mT_per_m": max_g,
        "max_grad_per_axis_mT_per_m": max_g_axis.tolist(),
        "max_slew_T_per_m_per_s": max_slew,
        "max_slew_per_axis_T_per_m_per_s": max_slew_axis.tolist(),
        "system_max_grad_mT_per_m": config.system.max_grad_mT_per_m,
        "system_max_slew_T_per_m_per_s": config.system.max_slew_T_per_m_per_s,
        "adc_dwell_s": config.readout.adc_dwell_s,
        "nyquist_max_dwell_s": max_dwell,
        "nyquist_ok": config.readout.adc_dwell_s <= max_dwell,
    }


def acquired_time_range(config: Config, num_trs: int) -> list[float]:
    """Time range of the first ``num_trs`` acquired TRs."""
    n = min(num_trs, config.scan.num_trs)
    start = config.scan.num_dummy_trs * config.readout.tr_s
    return [start, start + n * config.readout.tr_s]


def timing_report(seq: pp.Sequence) -> dict[str, Any]:
    """pypulseq's timing check."""
    ok, errors = seq.check_timing()
    return {"ok": bool(ok), "num_errors": len(errors), "errors": [str(e) for e in errors[:10]]}


def read_asc(path: Path) -> dict[str, Any]:
    """A Siemens gradient system ``.asc`` file, with its ``$INCLUDE`` files.

    pypulseq's ``readasc`` ignores ``$INCLUDE`` lines, but the gradient
    system file (``MP_GradSys_*.asc``) keeps the SAFE PNS model parameters in
    an included ``*_GSWD_SAFETY.asc`` file. Each included file is read from
    the same directory and merged in; its values take precedence, as with a
    textual include at the end of the file.
    """
    from pypulseq.utils.siemens.readasc import readasc

    asc: dict[str, Any] = readasc(str(path))[0]
    for line in path.read_text(errors="replace").splitlines():
        match = re.match(r"\s*\$INCLUDE\s+(\S+)", line)
        if match is None:
            continue
        included = path.parent / match.group(1)
        if not included.exists():
            raise FileNotFoundError(
                f"{path} includes {included.name}, which is not in {path.parent}"
            )
        _merge(asc, read_asc(included))
    return asc


def _merge(base: dict[Any, Any], update: dict[Any, Any]) -> None:
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = value


def pns_hardware(path: Path) -> SimpleNamespace:
    """SAFE PNS hardware description of a gradient system ``.asc`` file."""
    from pypulseq.utils.siemens.asc_to_hw import asc_to_hw

    asc = read_asc(path)
    try:
        asc["GradPatSup"]["Phys"]["PNS"]
        asc["asGPAParameters"][0]["sGCParameters"]["flGScaleFactorX"]
    except KeyError as e:
        raise ValueError(
            f"{path} (with its includes) has no {e}: give the gradient system file, "
            "MP_GradSys_*.asc, with its *_GSWD_SAFETY.asc next to it"
        ) from None
    hardware: SimpleNamespace = asc_to_hw(asc)
    return hardware


def acoustic_resonances(path: Path) -> list[tuple[float, float]]:
    """Frequency bands (Hz) of the coil's acoustic resonances in an ``.asc`` file."""
    gc = read_asc(path).get("asGPAParameters", {}).get(0, {}).get("sGCParameters", {})
    frequencies = gc.get("aflAcousticResonanceFrequency", {})
    widths = gc.get("aflAcousticResonanceBandwidth", {})
    bands = []
    for i, f in sorted(frequencies.items()):
        width = float(widths.get(i, 0.0))
        if f > 0 and width > 0:
            bands.append((float(f) - width / 2, float(f) + width / 2))
    return bands


def pns_report(
    config: Config,
    schedule: Schedule,
    shapes: list[Shape],
    hardware: Path | SimpleNamespace,
) -> dict[str, Any]:
    """Peak PNS (fraction of the stimulation threshold) over every TR.

    ``hardware`` is a Siemens gradient system ``.asc`` file (read with its
    ``$INCLUDE`` files), or a SAFE hardware description
    (``pypulseq.utils.safe_pns_prediction.safe_example_hw()``). The SAFE model
    runs on the physical waveform of every TR, dummies included
    (``relaxed_rosettes.pns``).
    """
    hw = pns_hardware(hardware) if isinstance(hardware, Path) else hardware
    result = design_pns(config, schedule, shapes, hw)
    worst = result.worst_tr
    return {
        "ok": result.peak < 1.0,
        "max_pns": result.peak,
        "max_pns_per_axis": result.peak_per_axis.tolist(),
        "worst_tr": worst,
        "worst_tr_shape": int(schedule.shape_index[worst]),
        "trs": schedule.num_trs,
    }


def spectrum_report(
    seq: pp.Sequence, bands_hz: list[tuple[float, float]], time_range: list[float]
) -> dict[str, Any]:
    """Gradient spectrum peaks, overall and in each band, over a time range.

    Amplitudes are relative to the largest spectral peak of any axis.
    """
    if time_range[1] - time_range[0] < SPECTRUM_WINDOW_S:
        raise ValueError(
            f"the spectrum needs at least {SPECTRUM_WINDOW_S * 1e3:g} ms of TRs "
            f"(got {(time_range[1] - time_range[0]) * 1e3:g} ms)"
        )
    max_frequency = max([2000.0, *(high for _, high in bands_hz)])
    spectra, _, frequencies, _ = seq.calculate_gradient_spectrum(
        max_frequency=max_frequency,
        window_width=SPECTRUM_WINDOW_S,
        time_range=time_range,
        plot=False,
    )
    spectra = np.abs(np.stack(spectra))  # (axes, frequencies)
    peak = float(spectra.max())
    axis, index = np.unravel_index(int(np.argmax(spectra)), spectra.shape)
    bands: list[dict[str, Any]] = []
    for low, high in bands_hz:
        inside = (frequencies >= low) & (frequencies <= high)
        if not inside.any():
            bands.append({"band_hz": [low, high], "relative_peak": None})
            continue
        band = spectra[:, inside]
        a, i = np.unravel_index(int(np.argmax(band)), band.shape)
        bands.append(
            {
                "band_hz": [low, high],
                "relative_peak": float(band.max()) / peak if peak > 0 else 0.0,
                "peak_frequency_hz": float(frequencies[inside][i]),
                "peak_axis": "xyz"[a],
            }
        )
    return {
        "peak_frequency_hz": float(frequencies[index]),
        "peak_axis": "xyz"[axis],
        "bands": bands,
        "time_range_s": time_range,
    }


def format_report(report: dict[str, Any]) -> str:
    """Human-readable text of a ``check`` report."""
    lines = [f"config digest: {report['config_digest']}"]
    t = report["timing"]
    lines.append(f"timing: {'ok' if t['ok'] else 'FAIL'} ({t['num_errors']} errors)")
    lines += [f"  {e}" for e in t["errors"]]
    g = report["gradients"]
    axes = "/".join(f"{v:.2f}" for v in g["max_grad_per_axis_mT_per_m"])
    slews = "/".join(f"{v:.1f}" for v in g["max_slew_per_axis_T_per_m_per_s"])
    lines += [
        f"max |g|: {g['max_grad_mT_per_m']:.2f} mT/m (x/y/z {axes}; "
        f"system {g['system_max_grad_mT_per_m']:g})",
        f"max slew: {g['max_slew_T_per_m_per_s']:.1f} T/m/s (x/y/z {slews}; "
        f"system {g['system_max_slew_T_per_m_per_s']:g})",
        f"Nyquist: {'ok' if g['nyquist_ok'] else 'FAIL'}: dwell {g['adc_dwell_s'] * 1e6:.2f} us, "
        f"at most {g['nyquist_max_dwell_s'] * 1e6:.2f} us at peak |g|",
    ]
    if "pns" in report:
        p = report["pns"]
        per_axis = "/".join(f"{v:.0%}" for v in p["max_pns_per_axis"])
        lines.append(
            f"PNS: {'ok' if p['ok'] else 'FAIL'}: peak {p['max_pns']:.1%} (x/y/z {per_axis}) "
            f"over all {p['trs']} TRs; worst: TR {p['worst_tr']} (shape {p['worst_tr_shape']})"
        )
    s = report["spectrum"]
    lines.append(f"gradient spectrum peak: {s['peak_frequency_hz']:.0f} Hz ({s['peak_axis']})")
    for b in s["bands"]:
        low, high = b["band_hz"]
        if b["relative_peak"] is None:
            lines.append(f"  band {low:g}-{high:g} Hz: no spectrum samples")
        else:
            lines.append(
                f"  band {low:g}-{high:g} Hz: {b['relative_peak']:.1%} of peak "
                f"at {b['peak_frequency_hz']:.0f} Hz ({b['peak_axis']})"
            )
    return "\n".join(lines)
