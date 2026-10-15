"""Predicted relative noise variance of Hann-filtered recons, sum_shells w_r^2 V^2 / n,
from the radial sample histogram (isotropic approximation; equal sample counts).
Usage: noise.py ref=path.shapes.npz label=path.shapes.npz ..."""
import sys
import numpy as np
from relaxed_rosettes.density import hann
from relaxed_rosettes.optimize import load_shapes
from relaxed_rosettes.repulsion import sample_operator

SHELLS = 128
RES_MM = [8, 6, 4, 3, 2, 1.5, 1]

def noise(path):
    c, shapes = load_shapes(path)
    ops = sample_operator(c)
    r = np.concatenate([np.linalg.norm(ops(c, s.g_mT_per_m), axis=1) for s in shapes])
    edges = np.linspace(0, 1, SHELLS + 1)
    n = np.histogram(np.minimum(r, 1), bins=edges)[0].astype(float)
    mid = (edges[1:] + edges[:-1]) / 2
    vol = 4 / 3 * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    out = []
    design = c.geometry.resolution_m * 1e3
    for res in RES_MM:
        w2 = hann(mid, design / res) ** 2
        need = w2 > 0
        out.append(np.inf if np.any(n[need] == 0) else float(np.sum(w2[need] * vol[need] ** 2 / n[need])))
    return np.array(out), len(r)

items = [a.split("=", 1) for a in sys.argv[1:]]
ref, count = noise(items[0][1])
print("| design | samples | " + " | ".join(f"{r} mm" for r in RES_MM) + " |")
print("|---|---|" + "---|" * len(RES_MM))
for label, path in items:
    v, m = noise(path)
    print(f"| {label} | {m} | " + " | ".join(f"{x:.2f}" for x in v / ref) + " |")
