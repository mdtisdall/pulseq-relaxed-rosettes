"""Solve one full-size shape (hairpin, relaxed, relaxed+tube); save g, time, peak RSS. Usage: solve.py out.npz"""
import resource, sys, time
import numpy as np
from relaxed_rosettes.config import Config
from relaxed_rosettes.optimize import optimize_shape, solve_shape
from relaxed_rosettes.targets import make_targets, PROVISIONAL_PETAL_TIME_FACTOR
base = Config().with_overrides(["shapes.num_shapes=1", "shapes.echoes_per_tr=40"])
relax = base.with_overrides(["shapes.crossing_tolerance_cells=1", "shapes.tip_width=0.5"])
tube = relax.with_overrides(["shapes.tube_cells=2"])
targets = make_targets(base)[0]
out = {}
for name, cfg, kw in [("hairpin", base, {}), ("relaxed", relax, {}), ("tube", tube, None)]:
    t = time.perf_counter()
    sh = solve_shape(cfg, targets) if kw is None else optimize_shape(cfg, targets)
    out[name] = sh.g_mT_per_m
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30  # bytes on macOS
    print(f"{name}: {time.perf_counter() - t:.1f} s, peak RSS so far {rss:.2f} GB, slewE {np.sum(np.diff(sh.g_mT_per_m, axis=0)**2):.3f}", flush=True)
np.savez(sys.argv[1], **out)
