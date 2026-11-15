"""E19 (revision, second round): CLAD with a wider learning-rate grid on the original P2 splits.

The inner search of E10 chose the larger of its two learning rates (1e-3) in every fold with all features, so the
grid is extended to {1e-2, 3e-3, 1e-3, 3e-4} with the same widths, depths and dropout values (16 configurations),
the same inner hold-out of two known training families (never the outer zero-day family), Platt calibration on the
validation split and the 1% validation-FPR threshold.

Run from code/:  python -m experiments.exp_clad_wide
Output: results/e19_clad_wide.json
"""
import json
import os
import time
from multiprocessing import Pool

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np

GRID = [dict(lr=lr, d_model=w, depth=dp, dropout=do)
        for lr in (1e-2, 3e-3, 1e-3, 3e-4) for (w, dp) in ((256, 3), (128, 2)) for do in (0.0, 0.2)]


def job(f):
    import torch
    import experiments.exp_p2_refit as R
    from experiments.exp_detection_robust import fit_eval
    from zdt.clad import CLAD
    from zdt.data import UNSW_FAMILIES
    torch.set_num_threads(1)
    t0 = time.time()
    R.CLAD_GRID = GRID
    seed = UNSW_FAMILIES.index(f)
    D = R.data()
    X, fam, y, s = D["X"], D["fam"], D["y"], D["S"][f]
    tr, va, te, grp = s["tr"], s["va"], s["te"], s["grp"]
    hp, search = R.clad_search(X[tr], y[tr], fam[tr], X[va], y[va], fam[va], f, seed)
    res = fit_eval({"CLAD-wide": CLAD(seed, **hp)}, X, y, fam, tr, va, te, grp, seed)
    return dict(family=f, hp=hp, search=search, det=res, time_s=time.time() - t0)


def main():
    from zdt.data import UNSW_FAMILIES
    t0 = time.time()
    out = []
    with Pool(max(1, min(9, (os.cpu_count() or 2) - 1))) as p:
        for r in p.imap_unordered(job, UNSW_FAMILIES):
            out.append(r)
            print(f"done {r['family']} lr={r['hp']['lr']} AUROC {r['det']['CLAD-wide']['auroc_zd']:.3f} "
                  f"ZDR {r['det']['CLAD-wide']['zdr']:.3f} [{time.time() - t0:.0f}s]", flush=True)
    json.dump(dict(grid=GRID, folds=out), open("results/e19_clad_wide.json", "w"), indent=1, default=float)
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
