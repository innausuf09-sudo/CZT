"""E20 (revision, second round): evaluation of the tuned triage policies on fresh random streams.

The preliminary analysis with the earlier policy definitions (E8) and the main runs (E9) used the same evaluation seeds
(1,000 to 1,019). Here every policy selected in E9 is re-evaluated, without any re-tuning, on 20 new streams per family
group (seeds 9,000 to 9,019) that no earlier analysis has used, in all 35 distinct catalogue settings. For comparison,
the candidate chosen by the knee-point rule of the earlier design (smallest Euclidean norm of the min-max normalised
objectives on the tuning streams) is evaluated on the same fresh streams.

Run from code/:  python -m experiments.exp_triage_fresh
Output: results/e20_fresh.json
"""
import json
import os
import time
from multiprocessing import Pool as MPPool

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np

SEED_FRESH = 9000
DUP = ("mk1.0_mz0.0_mb0.5", "mk0.8_mz0.2_mb0.5")


def knee(points):
    F = np.array([(1 - z, 1 - k, 0.0 if c is None or np.isnan(c) else c) for z, k, c in points], float)
    Fn = (F - F.min(0)) / (F.max(0) - F.min(0) + 1e-12)
    return int(np.argmin(np.linalg.norm(Fn, axis=1)))


def job(task):
    import experiments.exp_triage_v2 as V
    scen, g, tuned = task
    sc = V.SCEN[scen]
    As = V.streams(V.pools()[("EV", g)], sc, V.RHO_EVAL, range(SEED_FRESH, SEED_FRESH + V.REPS))
    P = {"Risk": ("Risk", {})}
    for (kind, info), t in tuned.items():
        P[f"{kind}-T/{info}"] = (V.policy_of(kind), t["params"])
        k = knee(t["tune_points"])
        P[f"{kind}-T/{info}[knee]"] = (V.policy_of(kind), V.params_of(kind, t["X"][k], info))
    out = {n: {f: [] for f in V.KEEP} for n in P}
    for A in As:
        for n, (pol, par) in P.items():
            r = V.run_named(A, pol, par)
            for f in V.KEEP:
                out[n][f].append(float(r[f]))
    return (scen, g), out


def main():
    import experiments.exp_triage_v2 as V
    t0 = time.time()
    d = json.load(open("results/e9_triage.json"))
    T = d["tuned"]
    tasks = []
    for scen in V.SCEN:
        if scen in DUP:
            continue
        for g in V.GROUPS:
            tuned = {(k, i): T[f"{scen}|{g}|{k}|{i}"] for k in V.TKINDS for i in V.INFOS}
            tasks.append((scen, g, tuned))
    res = {}
    with MPPool(max(1, (os.cpu_count() or 2) - 2)) as mp:
        for i, (key, out) in enumerate(mp.imap_unordered(job, tasks)):
            res["|".join(key)] = out
            if i % 10 == 0:
                print(f"  {i + 1}/{len(tasks)} {key} [{time.time() - t0:.0f}s]", flush=True)
    json.dump(dict(seeds=[SEED_FRESH, SEED_FRESH + V.REPS - 1], eval=res), open("results/e20_fresh.json", "w"))
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
