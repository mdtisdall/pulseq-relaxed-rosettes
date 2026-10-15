import cvxpy as cp
from relaxed_rosettes import repulsion
from relaxed_rosettes.optimize import load_shapes
S = "/private/tmp/claude-501/-Users-dylan-dev-pulseq-relaxed-rosettes/a8ca02b4-9385-42df-9fbc-2bd8229d967a/scratchpad/sweep"
base, shapes = load_shapes(f"{S}/base16.shapes.npz")
config = base.with_overrides(["shapes.repulsion_iterations=6", "shapes.repulsion_weight=3"])
orig_solve = cp.Problem.solve
def solve(self, *a, **k):
    try:
        out = orig_solve(self, *a, **k)
        print("  solver status:", self.status)
        return out
    except cp.error.SolverError as e:
        print("  SolverError:", str(e)[:200]); raise
cp.Problem.solve = solve
_, hist = repulsion.repel_shape(config, shapes[0])
for i, h in enumerate(hist): print(i, h.trust_cells, h.ratio, h.accepted)
