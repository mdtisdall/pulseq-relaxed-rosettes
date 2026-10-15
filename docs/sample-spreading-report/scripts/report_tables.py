"""Markdown tables from coverage outputs and shapes files. Usage: tables.py label=path_prefix ...
(path_prefix + '.cov.txt' and '.shapes.npz')."""
import re, sys
from pathlib import Path
import numpy as np
from relaxed_rosettes.optimize import load_shapes
from relaxed_rosettes.repulsion import sample_operator, slew_energy

def gaps(path):
    text = Path(path).read_text().splitlines()
    i = next(n for n, l in enumerate(text) if l.startswith("gap:"))
    head = text[i + 1].split()[2:]
    res = [h for h in head if h != "mm"]
    rows = {}
    for line in text[i + 2:]:
        if not line.strip():
            break
        parts = line.split()
        rows[parts[0]] = [float(v) for v in re.findall(r"(?<!\()\b(\d+\.\d+)\b(?!\))", line)][: len(res)]
        rows[parts[0]] = [float(x) for x in re.sub(r"\([^)]*\)", "", line).split()[1:]]
    empty = re.search(r"empty cells: ([\d.]+)% overall, ([\d.]+)% before 5 ms, ([\d.]+)% in the outer", "\n".join(text))
    red = re.search(r"within-TR redundancy: ([\d.]+)%", "\n".join(text))
    return res, rows, empty.groups(), red.group(1)

def shape_stats(path):
    config, shapes = load_shapes(path)
    ops = sample_operator(config)
    paths = [np.linalg.norm(np.diff(ops(config, s.g_mT_per_m), axis=0), axis=1).sum() for s in shapes]
    t = (np.arange(config.raster_points_per_tr) + 0.5) * config.system.grad_raster_time_s
    adc = (t >= config.adc_start_s) & (t <= config.adc_end_s)
    g_adc = np.concatenate([np.linalg.norm(s.g_mT_per_m[adc], axis=1) for s in shapes])
    tips = [s.targets.num_echoes for s in shapes]
    return (f"{min(tips)}–{max(tips)}", np.mean(paths), g_adc.mean(),
            max(np.linalg.norm(s.g_mT_per_m, axis=1).max() for s in shapes),
            np.mean([slew_energy(s.g_mT_per_m) for s in shapes]))

designs = [a.split("=", 1) for a in sys.argv[1:]]
print("| design | tips/TR | path/TR (k_max) | mean \\|g\\| in ADC (mT/m) | peak \\|g\\| (mT/m) | mean slew energy | within-TR redundancy | empty (\\|k\\|, τ) cells: all / < 5 ms / outer half |")
print("|---|---|---|---|---|---|---|---|")
for label, p in designs:
    tips, path, g, gpk, se = shape_stats(p + ".shapes.npz")
    _, _, empty, red = gaps(p + ".cov.txt")
    print(f"| {label} | {tips} | {path:.1f} | {g:.1f} | {gpk:.1f} | {se:.0f} | {red}% | {empty[0]}% / {empty[1]}% / {empty[2]}% |")
print()
for label, p in designs:
    res, rows, _, _ = gaps(p + ".cov.txt")
    print(f"**{label}** — gap (95th percentile, cells; median over window starts)\n")
    print("| window (s) | " + " | ".join(f"{r} mm" for r in res) + " |")
    print("|---|" + "---|" * len(res))
    for w, vals in rows.items():
        print(f"| {w} | " + " | ".join(f"{v:.1f}" for v in vals[: len(res)]) + " |")
    print()
