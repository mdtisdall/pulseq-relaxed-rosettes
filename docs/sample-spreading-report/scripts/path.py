"""Path per TR (k_max), mean |g| and peak |g| of each design. Usage: path.py shapes.npz..."""
import sys, numpy as np
from relaxed_rosettes.optimize import load_shapes
from relaxed_rosettes.repulsion import sample_operator
for f in sys.argv[1:]:
    config, shapes = load_shapes(f)
    ops = sample_operator(config)
    paths = [np.linalg.norm(np.diff(ops(config, s.g_mT_per_m), axis=0), axis=1).sum() for s in shapes]
    gs = np.concatenate([np.linalg.norm(s.g_mT_per_m, axis=1) for s in shapes])
    t = (np.arange(len(shapes[0].g_mT_per_m)) + 0.5) * config.system.grad_raster_time_s
    adc = (t >= config.adc_start_s) & (t <= config.adc_end_s)
    gadc = np.concatenate([np.linalg.norm(s.g_mT_per_m[adc], axis=1) for s in shapes])
    print(f"{f.split('/')[-1]:22s} path/TR {np.mean(paths):5.1f} k_max (min {min(paths):.1f}, max {max(paths):.1f}); mean |g| in ADC {gadc.mean():.1f} mT/m; peak |g| {gs.max():.1f}")
