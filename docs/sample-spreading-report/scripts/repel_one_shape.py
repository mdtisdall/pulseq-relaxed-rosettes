import resource, time
from relaxed_rosettes.config import Config
from relaxed_rosettes.optimize import shape_metrics, solve_shape
from relaxed_rosettes.repulsion import repel_shape
from relaxed_rosettes.targets import make_targets
config = Config().with_overrides(["shapes.num_shapes=1", "shapes.echoes_per_tr=40",
    "shapes.crossing_tolerance_cells=1", "shapes.tip_width=0.5", "shapes.repulsion_iterations=5"])
shape = solve_shape(config, make_targets(config)[0])
gb = lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30
print(f"step 1 solve {shape.solve_time_s:.1f} s, peak RSS {gb():.2f} GB", flush=True)
t = time.perf_counter()
new, hist = repel_shape(config, shape)
total = time.perf_counter() - t
for i, h in enumerate(hist):
    print(f"step {i}: trust {h.trust_cells:.0f} cells, accepted {h.accepted}, F {h.energy:.6f}, slewE {h.slew_energy:.0f}, solve {h.solve_time_s:.1f} s")
solves = sum(h.solve_time_s for h in hist)
m = shape_metrics(config, new)
print(f"total {total:.1f} s: solves {solves:.1f} s, rest {total - solves:.1f} s; peak RSS {gb():.2f} GB")
print(f"max slew {m.max_slew_T_per_m_per_s:.1f} T/m/s, target violation {m.max_target_error_kmax:.1e}")
