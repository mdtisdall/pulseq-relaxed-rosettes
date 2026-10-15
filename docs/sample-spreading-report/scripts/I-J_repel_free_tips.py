"""Repel the hairpin design with the tip targets dropped: keep k = 0 crossings, the spoil
moment and the hardware limits. Usage: repel_free.py in.npz out.npz iterations"""
import sys, time
from concurrent.futures import ProcessPoolExecutor
import cvxpy as cp
import numpy as np
from relaxed_rosettes import repulsion
from relaxed_rosettes import optimize as opt
from relaxed_rosettes.repulsion import energy_distance, repel_shape, repulsion_target, sample_operator, slew_energy

def free_tip_constraints(config, targets, g, relax, tip_directions=None):
    n = config.raster_points_per_tr
    slew = g[1:] - g[:-1]
    k, constraints = opt._kspace_variable(config, g)
    constraints += [
        g[: opt.num_zero_samples(config)] == 0,
        g[n - 1] == 0,
        cp.norm(g, 2, axis=1) <= config.gradients.max_grad_mT_per_m * (1 - opt.LIMIT_MARGIN),
        cp.norm(slew, 2, axis=1) <= opt.slew_limits(config) * (1 - opt.LIMIT_MARGIN),
    ]
    k_targets = opt._edge_kspace(config, k, g, opt.target_edges(config, targets))
    constraints += [k_targets[1:-1:2] == 0, k_targets[-1] == targets.spoil]  # no tips
    return k, constraints

def one(args):
    config, shape, i = args
    repulsion.waveform_constraints = free_tip_constraints
    t = time.perf_counter()
    new, hist = repel_shape(config, shape)
    ops = sample_operator(config)
    x0, x1 = ops(config, shape.g_mT_per_m), ops(config, new.g_mT_per_m)
    target = repulsion_target(config, x0)
    k0 = opt.target_kspace(config, shape.g_mT_per_m, shape.targets)
    k1 = opt.target_kspace(config, new.g_mT_per_m, shape.targets)
    r0, r1 = np.linalg.norm(k0[0:-1:2], axis=1), np.linalg.norm(k1[0:-1:2], axis=1)
    cross = np.linalg.norm(k1[1:-1:2], axis=1).max()
    cell = opt.cells_to_kmax(config, 1.0)
    tipmove = np.linalg.norm(k1[0:-1:2] - k0[0:-1:2], axis=1) / cell
    m = opt.shape_metrics(config, new)
    return new, (i, energy_distance(x0, target), energy_distance(x1, target), slew_energy(shape.g_mT_per_m),
                 slew_energy(new.g_mT_per_m), sum(h.accepted for h in hist), len(hist), m.max_slew_T_per_m_per_s,
                 cross, float(np.median(tipmove)), float(tipmove.max()), float(np.mean(r1 - r0)), time.perf_counter() - t)

if __name__ == "__main__":
    src, dst, iters = sys.argv[1], sys.argv[2], int(sys.argv[3])
    base, shapes = opt.load_shapes(src)
    config = base.with_overrides([f"shapes.repulsion_iterations={iters}", "shapes.repulsion_weight=3"])
    t = time.perf_counter()
    with ProcessPoolExecutor(5) as ex:
        out = list(ex.map(one, [(config, s, i) for i, s in enumerate(shapes)]))
    print("shape D0 D1 dD% slewE0 slewE1 dS% accepted max_slew max_crossing_kmax tip_move_median_cells tip_move_max_cells mean_tip_radius_change time_s")
    rows = [r for _, r in out]
    for r in rows:
        print(f"{r[0]} {r[1]:.3e} {r[2]:.3e} {100*(r[2]/r[1]-1):+.1f} {r[3]:.0f} {r[4]:.0f} {100*(r[4]/r[3]-1):+.1f} {r[5]}/{r[6]} {r[7]:.1f} {r[8]:.1e} {r[9]:.1f} {r[10]:.1f} {r[11]:+.4f} {r[12]:.0f}")
    d = np.array([[r[1], r[2], r[3], r[4]] for r in rows])
    print(f"mean dD {100*np.mean(d[:,1]/d[:,0]-1):+.1f}%  mean dSlewE {100*np.mean(d[:,3]/d[:,2]-1):+.1f}%  wall {time.perf_counter()-t:.0f}s")
    # Saved with the hairpin config; the tips are not met any more, so do not trust shape_metrics' target error.
    opt.save_shapes(dst, config, [s for s, _ in out])
