"""Summary of the random family partitions (revision R2.6).

Run from code/:  python -m analysis.partitions_report
Reads results/e9_triage_partK.json (K = 0..7) and results/e9_triage.json (original G1/G2 split).
Writes tables/s_table_e9_partitions.csv and prints a summary.
"""
import glob
import json

import numpy as np
import pandas as pd
from scipy import stats

SC = ("original", "central", "uninformative_0.5")


def ci(x):
    x = np.asarray(x, float)
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return x.mean(), x.mean() - h, x.mean() + h


rows = []
files = [("G1/G2 (main)", "results/e9_triage.json")] + [(f"K={p.split('part')[1].split('.')[0]}", p)
                                                       for p in sorted(glob.glob("results/e9_triage_part*.json"))]
for lab, path in files:
    d = json.load(open(path))
    GR = d["meta"]["groups"]
    for s in SC:
        ev = d["eval"]
        v = lambda p, f: np.concatenate([np.asarray(ev[f"{s}|{g}"]["reps"][p][f], float) for g in GR])
        for info in ("flag", "nov"):
            r = dict(partition=lab, groups=" | ".join(", ".join(x) for x in GR.values()), setting=s, info=info)
            for a, b, nm in ((f"CZT-T/{info}", "Risk", "index - Risk"), (f"Risk-T/{info}", "Risk", "tuned Risk - Risk"),
                             (f"CZT-T/{info}", f"Risk-T/{info}", "index - tuned Risk")):
                for f in ("zcv", "kcv"):
                    m, lo, hi = ci(v(a, f) - v(b, f))
                    r[f"{nm} {f.upper()}"] = f"{m:+.3f} [{lo:+.3f}, {hi:+.3f}]"
                    r[f"_{nm}_{f}"], r[f"_{nm}_{f}_lo"], r[f"_{nm}_{f}_hi"] = m, lo, hi
            rows.append(r)
P = pd.DataFrame(rows)
# identical random draws (same family split) are counted once
P = P[~P.duplicated(subset=["groups", "setting", "info"])].reset_index(drop=True)
print("distinct partitions:", P.partition.nunique(), sorted(P.partition.unique()))
P[[c for c in P.columns if not c.startswith("_")]].to_csv("tables/s_table_e9_partitions.csv", index=False)
P.to_csv("tables/s_table_e9_partitions_numeric.csv", index=False)
for s in SC:
    for info in ("flag", "nov"):
        x = P[(P["setting"] == s) & (P["info"] == info)]
        for nm in ("index - Risk", "index - tuned Risk"):
            z = x[f"_{nm}_zcv"]
            print(f"{s:18s} {info:4s} {nm:20s} ZCV range {z.min():+.3f} to {z.max():+.3f}; sig>0 {int((x[f'_{nm}_zcv_lo'] > 0).sum())}, "
                  f"sig<0 {int((x[f'_{nm}_zcv_hi'] < 0).sum())} of {len(x)}; KCV range {x[f'_{nm}_kcv'].min():+.3f} to {x[f'_{nm}_kcv'].max():+.3f}")
