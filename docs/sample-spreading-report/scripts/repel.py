"""Repel every shape of a step 1 shapes file. Usage: repel.py in.npz out.npz override..."""
import sys, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from relaxed_rosettes.optimize import load_shapes, save_shapes, shape_metrics
from relaxed_rosettes.repulsion import energy_distance, repel_shape, repulsion_target, sample_operator, slew_energy

def one(args):
    config, shape, i = args
    t = time.perf_counter()
    new, hist = repel_shape(config, shape)
    ops = sample_operator(config)
    x0, x1 = ops(config, shape.g_mT_per_m), ops(config, new.g_mT_per_m)
    target = repulsion_target(config, x0)
    m = shape_metrics(config, new)
    return new, (i, energy_distance(x0, target), energy_distance(x1, target),
                 slew_energy(shape.g_mT_per_m), slew_energy(new.g_mT_per_m),
                 sum(h.accepted for h in hist), len(hist), m.max_target_error_kmax,
                 m.max_slew_T_per_m_per_s, time.perf_counter() - t,
                 float(np.mean([h.solve_time_s for h in hist])))

if __name__ == "__main__":
    src, dst, *overrides = sys.argv[1:]
    base, shapes = load_shapes(src)
    config = base.with_overrides(overrides)
    t = time.perf_counter()
    with ProcessPoolExecutor(5) as ex:
        out = list(ex.map(one, [(config, s, i) for i, s in enumerate(shapes)]))
    print("shape D0 D1 dD% slewE0 slewE1 dS% accepted/steps viol maxslew time_s mean_step_s")
    rows = [r for _, r in out]
    for (i, d0, d1, s0, s1, a, n, v, ms, ts, st) in rows:
        print(f"{i} {d0:.4e} {d1:.4e} {100*(d1/d0-1):+.1f} {s0:.0f} {s1:.0f} {100*(s1/s0-1):+.1f} {a}/{n} {v:.1e} {ms:.1f} {ts:.0f} {st:.1f}")
    d = np.array([[r[1], r[2], r[3], r[4]] for r in rows])
    print(f"mean dD {100*np.mean(d[:,1]/d[:,0]-1):+.1f}%  mean dSlewE {100*np.mean(d[:,3]/d[:,2]-1):+.1f}%  wall {time.perf_counter()-t:.0f}s")
    save_shapes(dst, config, [s for s, _ in out])
