"""E21 (revision, second round): training-seed variability of the leading P2 detectors.

The refit of E10 used one training seed per fold (the fold index). Here the six leading detectors and the hybrid are
refitted on the same P2 splits with two further training seeds (fold index + 100 and + 200), with the same calibration
and 1% validation-FPR threshold, so that the seed-to-seed spread can be compared with the gaps between detectors.

Run from code/:  python -m experiments.exp_detector_seeds
Output: results/e21_detector_seeds.json
"""
import json
import os
import time
from multiprocessing import Pool

os.environ.setdefault("OMP_NUM_THREADS", "1")


def job(task):
    import torch
    from experiments.exp_detection_robust import fit_eval, leading_models
    from experiments.exp_p2_refit import data
    from zdt.data import UNSW_FAMILIES
    torch.set_num_threads(1)
    f, off = task
    t0 = time.time()
    seed = UNSW_FAMILIES.index(f) + off
    D = data()
    X, fam, y, s = D["X"], D["fam"], D["y"], D["S"][f]
    res = fit_eval(leading_models(seed), X, y, fam, s["tr"], s["va"], s["te"], s["grp"], seed)
    return dict(family=f, offset=off, seed=seed, det=res, time_s=time.time() - t0)


def main():
    from zdt.data import UNSW_FAMILIES
    t0 = time.time()
    tasks = [(f, o) for o in (100, 200) for f in UNSW_FAMILIES]
    out = []
    with Pool(max(1, (os.cpu_count() or 2) - 2)) as p:
        for r in p.imap_unordered(job, tasks):
            out.append(r)
            print(f"done {r['family']} +{r['offset']} hybrid {r['det']['CZT-Hybrid']['auroc_zd']:.3f} "
                  f"DNN {r['det']['DNN']['auroc_zd']:.3f} [{time.time() - t0:.0f}s]", flush=True)
    json.dump(dict(folds=out), open("results/e21_detector_seeds.json", "w"), indent=1, default=float)
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
