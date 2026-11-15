"""E22 (revision, third round): matched-load tuning in all 35 distinct catalogue settings.

The main runs tune at load 1.0 on the other family group and evaluate at load 1.2. E17 repeated the tuning at the
evaluation load in the three anchor settings only. Here the index, tuned Risk and the tuned linear rule are re-tuned at
load 1.2 (other family group, same budget, seed and selection rule) in every setting and both information conditions, and
evaluated on the main evaluation streams together with untuned Risk.

Run from code/:  python -m experiments.exp_triage_matched
Output: results/e22_matched.json
"""
import json
import os
import time
from multiprocessing import Pool as MPPool

import experiments.exp_triage_robust as R


def main():
    t0 = time.time()
    T = [("D-load", s, g, k, info, {"rho_tune": 1.2, "n_ckpt": R.E.POP})
         for s in R.SCEN for g in R.E.GROUPS for k in ("CZT", "Risk", "Linear") for info in R.E.INFOS]
    nproc = max(1, (os.cpu_count() or 2) - 2)
    print(f"{len(T)} tasks, {nproc} processes", flush=True)
    res = []
    with MPPool(nproc) as mp:
        for i, (task, r) in enumerate(mp.imap_unordered(R.task_fn, T)):
            res.append(dict(stage=task[0], scenario=task[1], group=task[2], kind=task[3], info=task[4],
                            opt={k: v for k, v in task[5].items()}, **r))
            if i % 25 == 0:
                print(f"  {i + 1}/{len(T)} {task[:5]} [{time.time() - t0:.0f}s]", flush=True)
    json.dump(dict(scenarios=list(R.SCEN), results=res), open("results/e22_matched.json", "w"), default=float)
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
