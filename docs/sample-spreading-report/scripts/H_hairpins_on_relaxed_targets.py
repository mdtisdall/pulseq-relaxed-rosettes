"""Hairpin waveform on each relaxed-design target set (point targets, same timing)."""
import dataclasses, time
import numpy as np
from relaxed_rosettes.optimize import ShapeOptimizationError, load_shapes, optimize_shape, save_shapes
from relaxed_rosettes.repulsion import slew_energy
config, shapes = load_shapes("/private/tmp/claude-501/-Users-dylan-dev-pulseq-relaxed-rosettes/a8ca02b4-9385-42df-9fbc-2bd8229d967a/scratchpad/sweep/base16.shapes.npz")
out, t = [], time.perf_counter()
for i, s in enumerate(shapes):
    try:
        hp = optimize_shape(config, s.targets, relax=False)
    except ShapeOptimizationError as e:
        print(f"shape {i}: hairpin INFEASIBLE ({e})"); out.append(None); continue
    print(f"shape {i}: {s.targets.num_echoes} tips, slewE relaxed {slew_energy(s.g_mT_per_m):.0f}, hairpin {slew_energy(hp.g_mT_per_m):.0f}")
    out.append(dataclasses.replace(hp, status=s.status))
print(f"{sum(o is None for o in out)} infeasible, {time.perf_counter() - t:.0f} s")
if all(o is not None for o in out):
    save_shapes("/private/tmp/claude-501/-Users-dylan-dev-pulseq-relaxed-rosettes/6d224db4-ae46-4fa9-8933-eb00150e8494/scratchpad/test1/hp_on_rel.shapes.npz", config, out)
