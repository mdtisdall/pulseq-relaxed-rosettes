from pathlib import Path

import numpy as np
import pytest
from pypulseq.utils.safe_pns_prediction import safe_example_hw, safe_tau_lowpass

from relaxed_rosettes.checks import acoustic_resonances, pns_hardware, read_asc
from relaxed_rosettes.config import Config
from relaxed_rosettes.optimize import optimize_shapes
from relaxed_rosettes.pns import design_pns, rc_lowpass
from relaxed_rosettes.sequence import build_sequence, make_schedule

SMALL = Config().with_overrides(
    [
        "readout.tr_s=0.01",
        "readout.adc_num_samples=2000",
        "shapes.echoes_per_tr=4",
        "shapes.num_shapes=3",
        "scan.num_trs=6",
        "scan.num_dummy_trs=2",
    ]
)


@pytest.mark.parametrize("tau_ms", [0.03, 0.2, 2.0])
def test_rc_lowpass_matches_pypulseq(tau_ms: float) -> None:
    x = np.random.default_rng(0).standard_normal(20_000)
    expected = safe_tau_lowpass(x, tau_ms, 1e-5 * 1000)
    np.testing.assert_allclose(rc_lowpass(x, tau_ms, 1e-5), expected, atol=1e-12)


def test_design_pns_matches_pypulseq() -> None:
    """The fast model on the design equals pypulseq's PNS of the built sequence."""
    shapes = optimize_shapes(SMALL, jobs=1)
    schedule = make_schedule(SMALL)
    seq = build_sequence(SMALL, shapes, schedule)
    hw = safe_example_hw()
    _, norm, components, _ = seq.calculate_pns(hw, do_plots=False)
    result = design_pns(SMALL, schedule, shapes, hw)
    assert result.peak == pytest.approx(float(norm.max()), rel=1e-3)
    np.testing.assert_allclose(result.peak_per_axis, components.max(axis=0), rtol=1e-3)
    # Chunks carry the filter history: the chunk size does not matter.
    chunked = design_pns(SMALL, schedule, shapes, hw, chunk_trs=3)
    np.testing.assert_allclose(chunked.peak_per_tr, result.peak_per_tr, rtol=1e-9, atol=1e-12)


def write(path: Path, lines: list[str]) -> Path:
    path.write_text("\n".join(lines) + "\n")
    return path


def test_read_asc_follows_include(tmp_path: Path) -> None:
    write(tmp_path / "inc.asc", ["GradPatSup.Phys.PNS.flGSWDTauX[0] = 1.5", "a.b = 5"])
    main = write(
        tmp_path / "main.asc",
        [
            "### ASCCONV BEGIN ###",
            "asGPAParameters[0].sGCParameters.flGScaleFactorX = 0.25",
            "a.b = 1",
            "$INCLUDE inc.asc",
            "### ASCCONV END ###",
        ],
    )
    asc = read_asc(main)
    assert asc["asGPAParameters"][0]["sGCParameters"]["flGScaleFactorX"] == pytest.approx(0.25)
    assert asc["GradPatSup"]["Phys"]["PNS"]["flGSWDTauX"][0] == pytest.approx(1.5)
    assert asc["a"]["b"] == 5  # the included file wins, as a textual include


def test_read_asc_reports_a_missing_include(tmp_path: Path) -> None:
    main = write(tmp_path / "main.asc", ["a.b = 1", "$INCLUDE missing_SAFETY.asc"])
    with pytest.raises(FileNotFoundError, match="missing_SAFETY.asc"):
        read_asc(main)


def test_pns_hardware_needs_the_safety_file(tmp_path: Path) -> None:
    main = write(tmp_path / "main.asc", ["asGPAParameters[0].sGCParameters.flGScaleFactorX = 0.3"])
    with pytest.raises(ValueError, match="GSWD_SAFETY"):
        pns_hardware(main)


def test_acoustic_resonances(tmp_path: Path) -> None:
    gc = "asGPAParameters[0].sGCParameters"
    main = write(
        tmp_path / "main.asc",
        [
            f"{gc}.aflAcousticResonanceFrequency[0] = 500",
            f"{gc}.aflAcousticResonanceBandwidth[0] = 80",
            f"{gc}.aflAcousticResonanceFrequency[1] = 1200",
            f"{gc}.aflAcousticResonanceBandwidth[1] = 200",
            f"{gc}.aflAcousticResonanceFrequency[2] = 0",
            f"{gc}.aflAcousticResonanceBandwidth[2] = 0",
        ],
    )
    assert acoustic_resonances(main) == [(460.0, 540.0), (1100.0, 1300.0)]
