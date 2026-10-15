"""Shape 0 of base16, 20 steps at a weight; print trust region and ratio per step. Usage: trust_check.py weight"""
import sys
from relaxed_rosettes.optimize import load_shapes
from relaxed_rosettes.repulsion import repel_shape
S = "/private/tmp/claude-501/-Users-dylan-dev-pulseq-relaxed-rosettes/a8ca02b4-9385-42df-9fbc-2bd8229d967a/scratchpad/sweep"
base, shapes = load_shapes(f"{S}/base16.shapes.npz")
config = base.with_overrides(["shapes.repulsion_iterations=20", f"shapes.repulsion_weight={sys.argv[1]}"])
_, hist = repel_shape(config, shapes[0])
for i, h in enumerate(hist):
    print(f"{i:2d} trust {h.trust_cells:5.2f}  ratio {h.ratio:7.3f}  {'accept' if h.accepted else 'REJECT'}  F {h.energy:.7f}  slewE {h.slew_energy:.0f}  {h.solve_time_s:.1f}s")
