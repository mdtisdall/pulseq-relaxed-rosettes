"""Current vs symmetric-block energy on the full-size relaxed shape. Usage: bench_energy.py shapes.npz"""
import sys, time, resource
import numpy as np
from relaxed_rosettes.config import Config
from relaxed_rosettes import repulsion
from relaxed_rosettes.repulsion import energy, initial_target, sample_operator, shell_distance

def energy_sym(x, target, block):
    m = len(x)
    r = np.linalg.norm(x, axis=1)
    phi, dphi = shell_distance(r[:, None], target.radii[None, :])
    attraction = phi @ target.weights
    radial = (dphi @ target.weights) / np.maximum(r, 1e-300)
    sq = np.sum(x**2, axis=1)
    rep = 0.0
    push = np.zeros_like(x)
    starts = range(0, m, block)
    for a in starts:
        xa, sa = x[a:a + block], sq[a:a + block]
        for b in starts:
            if b < a:
                continue
            xb, sb = x[b:b + block], sq[b:b + block]
            d2 = sa[:, None] + sb[None, :] - 2 * (xa @ xb.T)
            if a == b:
                np.fill_diagonal(d2, 0.0)
            d = np.sqrt(np.maximum(d2, 0.0))
            inv = np.divide(1.0, d, out=np.zeros_like(d), where=d > 1e-12)
            push[a:a + block] += xa * inv.sum(axis=1)[:, None] - inv @ xb
            if a == b:
                rep += float(d.sum())
            else:
                rep += 2 * float(d.sum())
                push[b:b + block] += xb * inv.sum(axis=0)[:, None] - inv.T @ xa
    value = float(attraction.mean()) - rep / (2 * m**2)
    return value, radial[:, None] * x / m - push / m**2

config = Config().with_overrides(["shapes.num_shapes=1", "shapes.echoes_per_tr=40"])
g = np.load(sys.argv[1])["relaxed"]
x = sample_operator(config)(config, g)
target = initial_target(np.linalg.norm(x, axis=1))
print(f"M = {len(x)}")
def bench(name, fn, reps=3):
    ts = []
    for _ in range(reps):
        t = time.perf_counter(); out = fn(); ts.append(time.perf_counter() - t)
    print(f"{name}: best {min(ts):.2f} s, median {sorted(ts)[1]:.2f} s", flush=True)
    return out
ref = None
for chunk in (256, 512, 1024):
    repulsion.CHUNK = chunk
    out = bench(f"current, CHUNK {chunk}", lambda: energy(x, target))
    ref = ref or out
for block in (256, 512, 1024, 2048):
    f, gr = bench(f"symmetric, block {block}", lambda: energy_sym(x, target, block))
    print(f"   |dF| {abs(f - ref[0]):.1e}, max |dgrad| / max |grad| {np.abs(gr - ref[1]).max() / np.abs(ref[1]).max():.1e}")
print(f"peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30:.2f} GB")
