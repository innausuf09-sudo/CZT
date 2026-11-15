"""E6: inference latency, throughput and footprint on a common CPU (2 threads)."""
import pickle, time, json, io
import numpy as np
import torch
from zdt.data import load_unsw, unsw_matrix
torch.set_num_threads(2)
d = pickle.load(open("results/e2_fold0_models.pkl", "rb"))
df, _ = load_unsw(verbose=False)
X, _ = unsw_matrix(df)
Xb = X[d["te_idx"]][:5000]
rows = []
for name, m in d["models"].items():
    m.proba(Xb[:256])  # warm-up
    t0 = time.perf_counter(); m.proba(Xb); tb = time.perf_counter() - t0
    singles = []
    for i in range(200):
        t0 = time.perf_counter(); m.proba(Xb[i:i + 1]); singles.append(time.perf_counter() - t0)
    if name == "CZT-Hybrid":
        size = sum(len(pickle.dumps(mm)) for mm in m.members)
    else:
        size = len(pickle.dumps(m))
    nparam = sum(p.numel() for p in m.net.parameters()) if hasattr(m, "net") else (
        sum(p.numel() for p in m.enc.parameters()) if hasattr(m, "enc") else None)
    rows.append(dict(model=name, batch_us_per_sample=1e6 * tb / len(Xb), throughput_per_s=len(Xb) / tb,
                     single_ms_p50=1e3 * np.median(singles), single_ms_p95=1e3 * np.percentile(singles, 95),
                     size_kb=size / 1024, n_params=nparam))
    print(rows[-1], flush=True)
json.dump(rows, open("results/e6_latency.json", "w"), indent=1, default=float)
