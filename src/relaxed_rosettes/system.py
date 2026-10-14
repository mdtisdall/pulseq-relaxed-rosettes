"""The scanner system description that pypulseq uses."""

from __future__ import annotations

import pypulseq as pp

from relaxed_rosettes.config import Config


def make_opts(config: Config) -> pp.Opts:
    """``pypulseq.Opts`` for the system section of a config."""
    s = config.system
    return pp.Opts(
        max_grad=s.max_grad_mT_per_m,
        grad_unit="mT/m",
        max_slew=s.max_slew_T_per_m_per_s,
        slew_unit="T/m/s",
        rf_dead_time=s.rf_dead_time_s,
        rf_ringdown_time=s.rf_ringdown_time_s,
        adc_dead_time=s.adc_dead_time_s,
        grad_raster_time=s.grad_raster_time_s,
        B0=s.B0_T,
    )
