import sys, time, numpy as np
from relaxed_rosettes.config import Config
from relaxed_rosettes.repulsion import energy, initial_target, sample_operator
config = Config().with_overrides(["shapes.num_shapes=1", "shapes.echoes_per_tr=40"])
x = sample_operator(config)(config, np.load(sys.argv[1])["relaxed"])
target = initial_target(np.linalg.norm(x, axis=1))
energy(x, target)
w, c = time.perf_counter(), time.process_time()
for _ in range(3): energy(x, target)
print(f"wall {(time.perf_counter()-w)/3:.2f} s, cpu {(time.process_time()-c)/3:.2f} s per call")
