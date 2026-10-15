"""Repel the hairpins under the relaxed constraints, with each shape's target and lambda
fixed to the relaxed run's (weight 3 from the relaxed start)."""
import time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from relaxed_rosettes import repulsion
from relaxed_rosettes.optimize import adc_samples, cells_to_kmax, load_shapes, sample_kspace, save_shapes, shape_metrics
from relaxed_rosettes.repulsion import energy, energy_distance, initial_target, repel_shape, sample_operator, slew_energy

def one(args):
    config, rel, hp, i = args
    ops = sample_operator(config)
    x_rel, x_hp = ops(config, rel.g_mT_per_m), ops(config, hp.g_mT_per_m)
    target = initial_target(np.linalg.norm(x_rel, axis=1))
    repulsion.repulsion_target = lambda c, x: target  # same target as the relaxed run
    delta = cells_to_kmax(config, config.shapes.repulsion_trust_cells)
    lam_rel = 3.0 * slew_energy(rel.g_mT_per_m) / (np.abs(energy(x_rel, target)[1]).sum() * delta)
    weight = lam_rel * np.abs(energy(x_hp, target)[1]).sum() * delta / slew_energy(hp.g_mT_per_m)
    c = config.with_overrides([f"shapes.repulsion_weight={float(weight)!r}"])
    ball = np.linalg.norm(sample_kspace(config, hp.g_mT_per_m)[adc_samples(config)], axis=1).max()
    t = time.perf_counter()
    new, hist = repel_shape(c, hp)
    m = shape_metrics(c, new)
    x_new = ops(config, new.g_mT_per_m)
    cell = cells_to_kmax(config, 1.0)
    return new, (i, weight, ball, energy_distance(x_hp, target), energy_distance(x_new, target),
                 energy_distance(x_rel, target), slew_energy(hp.g_mT_per_m), slew_energy(new.g_mT_per_m),
                 sum(h.accepted for h in hist), m.max_target_error_kmax, m.max_slew_T_per_m_per_s,
                 float(np.median(np.linalg.norm(x_new - x_hp, axis=1)) / cell), time.perf_counter() - t)

if __name__ == "__main__":
    base, rels = load_shapes("/private/tmp/claude-501/-Users-dylan-dev-pulseq-relaxed-rosettes/a8ca02b4-9385-42df-9fbc-2bd8229d967a/scratchpad/sweep/base16.shapes.npz")
    _, hps = load_shapes("/private/tmp/claude-501/-Users-dylan-dev-pulseq-relaxed-rosettes/6d224db4-ae46-4fa9-8933-eb00150e8494/scratchpad/test1/hp_on_rel.shapes.npz")
    config = base.with_overrides(["shapes.repulsion_iterations=20", "shapes.repulsion_weight=3"])
    t = time.perf_counter()
    with ProcessPoolExecutor(5) as ex:
        out = list(ex.map(one, [(config, r, h, i) for i, (r, h) in enumerate(zip(rels, hps))]))
    print("shape weight start_max|k| D_hp D_end D_rel_start slewE_hp slewE_end accepted viol maxslew median_move_cells time_s")
    for _, r in out:
        print(f"{r[0]} {r[1]:.2f} {r[2]:.6f} {r[3]:.3e} {r[4]:.3e} {r[5]:.3e} {r[6]:.0f} {r[7]:.0f} {r[8]}/20 {r[9]:.1e} {r[10]:.1f} {r[11]:.1f} {r[12]:.0f}")
    print(f"wall {time.perf_counter() - t:.0f} s")
    save_shapes("/private/tmp/claude-501/-Users-dylan-dev-pulseq-relaxed-rosettes/6d224db4-ae46-4fa9-8933-eb00150e8494/scratchpad/test1/hp_on_rel_rep.shapes.npz", config, [s for s, _ in out])
