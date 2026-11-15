"""E18 (revision, second round): detection robustness on the P2 splits and the P5 temporal split.

Tasks
  P2 'plus'      original P2 splits: Isolation Forest and autoencoder (benign-only anomaly baselines), an energy-based
                 open-set novelty score from a multi-class MLP, XGBoost and LightGBM tuned with the same inner
                 known-family scheme as CLAD (8 configurations each), and CLAD at the published 200 epochs.
  P2 'filtered'  leading detectors after removing every feature whose single-feature AUROC (benign vs attack) on the
                 TRAINING split of the fold exceeds 0.95 (host-signature filter fitted on training data only).
  P2 'majority'  leading detectors after cleaning with majority-label resolution of conflicting feature groups
                 (instead of removing every record of a conflicting group); splits drawn with the P2 scheme and seed.
  P5 'filtered'  NF-UNSW-NB15-v3 January -> February with the same training-data feature filter.
Run from code/:  python -m experiments.exp_detection_robust
Output: results/e18_detection_robust.json
"""
import json
import os
import time
from multiprocessing import Pool

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
from sklearn.metrics import roc_auc_score

FILTER_AUC = 0.95
LEADING = ["XGBoost", "LightGBM", "DNN", "ResSE-CNN1D", "B-ProtoMeta", "CZT-Hybrid"]
XGB_GRID = [dict(max_depth=d, learning_rate=lr) for d in (4, 6, 8, 10) for lr in (0.05, 0.1)]
LGB_GRID = [dict(num_leaves=n, learning_rate=lr) for n in (15, 31, 63, 127) for lr in (0.05, 0.1)]


def single_feature_filter(X, y):
    keep = []
    for j in range(X.shape[1]):
        if np.ptp(X[:, j]) == 0:
            keep.append(j); continue
        a = roc_auc_score(y, X[:, j])
        if max(a, 1 - a) <= FILTER_AUC:
            keep.append(j)
    return keep


def inner_search(make, grid, Xtr, ytr, famtr, Xva, famva, f):
    from zdt.data import UNSW_FAMILIES
    inner = [g for g in UNSW_FAMILIES if g != f and g in set(famtr)][:2]
    sc = []
    for hp in grid:
        au = []
        for g in inner:
            m = famtr != g
            mod = make(hp).fit(Xtr[m], ytr[m])
            mv = (famva == "Normal") | (famva == g)
            au.append(roc_auc_score((famva[mv] == g).astype(int), mod.raw_score(Xva[mv])))
        sc.append(float(np.mean(au)))
    return grid[int(np.argmax(sc))], sc


def metrics(p, thr, grp):
    from zdt.metrics import tpr_at_fpr
    isZ, isK, isN = grp == "zeroday", grp == "known", grp == "normal"
    mz = isN | isZ
    r = dict(test_fpr=float((p[isN] >= thr).mean()), known_recall=float((p[isK] >= thr).mean()) if isK.any() else np.nan)
    if isZ.any():
        r.update(auroc_zd=roc_auc_score(isZ[mz].astype(int), p[mz]), zdr=float((p[isZ] >= thr).mean()),
                 tpr_fpr1_zd=tpr_at_fpr(isZ[mz].astype(int), p[mz], 0.01))
    return r


def fit_eval(models, X, y, fam, tr, va, te, grp, seed):
    from zdt.metrics import threshold_at_fpr
    out = {}
    for name, m in models.items():
        if name == "CZT-Hybrid":
            continue
        m.fit(X[tr], y[tr], X[va], y[va], fam=fam[tr]); m.calibrate(X[va], y[va])
    if "CZT-Hybrid" in models:
        from zdt.models import Hybrid
        h = Hybrid(seed, members=[models["LightGBM"], models["ResSE-CNN1D"], models["B-ProtoMeta"]])
        h.fit(X[tr], y[tr], X[va], y[va]); h.calibrate(X[va], y[va]); models["CZT-Hybrid"] = h
    for name, m in models.items():
        pva = m.proba(X[va])
        thr = threshold_at_fpr(pva[y[va] == 0], 0.01)
        out[name] = metrics(m.proba(X[te]), thr, grp)
    return out


def leading_models(seed):
    from zdt.models import REGISTRY
    d = {n: REGISTRY[n](seed) for n in LEADING if n != "CZT-Hybrid"}
    d["CZT-Hybrid"] = None
    return d


def job(task):
    import torch
    from zdt.anomaly import AutoEncoder, EnergyNet, IForest
    from zdt.clad import CLAD
    from zdt.data import UNSW_FAMILIES
    from zdt.models import XGB, LGBM
    from experiments.exp_p2_refit import clad_search, data, ecdf_ref
    torch.set_num_threads(1)
    kind, f = task
    t0 = time.time()
    if kind.startswith("P5"):
        from experiments.exp_nfv3_temporal import CACHE, split, FAMS
        z = np.load(CACHE); X, famr, period = z["X"], z["fam"], z["period"]
        fam = np.where(famr == "Benign", "Normal", famr); y = (fam != "Normal").astype(int)
        seed = 0 if f is None else FAMS.index(f) + 1
        tr, va, te, grp = split(famr, period, f, seed)
        keep = single_feature_filter(X[tr], y[tr])
        Xf = X[:, keep]
        mods = leading_models(seed)
        hp, _ = clad_search(Xf[tr], y[tr], fam[tr], Xf[va], y[va], fam[va], f, seed)
        mods["CLAD"] = CLAD(seed, **hp)
        res = fit_eval(mods, Xf, y, fam, tr, va, te, grp, seed)
        meta = json.load(open("results/e13_nfv3_meta.json"))
        return dict(kind=kind, family=f, removed=[meta["features"][j] for j in range(X.shape[1]) if j not in keep],
                    det=res, time_s=time.time() - t0)
    seed = UNSW_FAMILIES.index(f)
    if kind == "P2-majority":
        D = majority_data()
    else:
        D = data()
    X, fam, y, s = D["X"], D["fam"], D["y"], D["S"][f]
    tr, va, te, grp = s["tr"], s["va"], s["te"], s["grp"]
    if kind == "P2-plus":
        mods = {"IsolationForest": IForest(seed), "Autoencoder": AutoEncoder(seed)}
        bx, sx = inner_search(lambda h: _xgb(seed, h), XGB_GRID, X[tr], y[tr], fam[tr], X[va], fam[va], f)
        bl, sl = inner_search(lambda h: _lgb(seed, h), LGB_GRID, X[tr], y[tr], fam[tr], X[va], fam[va], f)
        mods["XGBoost-tuned"] = _xgb(seed, bx); mods["LightGBM-tuned"] = _lgb(seed, bl)
        hp, _ = clad_search(X[tr], y[tr], fam[tr], X[va], y[va], fam[va], f, seed)
        mods["CLAD-200"] = CLAD(seed, epochs=200, warmup=20, **hp)
        en = EnergyNet(seed)
        mods["EnergyMLP"] = en
        res = fit_eval(mods, X, y, fam, tr, va, te, grp, seed)
        # energy novelty score: zero-day vs known among test records, mapped by validation known attacks
        va_att = va[y[va] == 1]
        F = ecdf_ref(en.energy(X[va_att]))
        nv = F(en.energy(X[te]))
        isZ, isK = grp == "zeroday", grp == "known"
        mk = isZ | isK
        res["_novelty_energy_auroc"] = float(roc_auc_score(isZ[mk].astype(int), nv[mk]))
        np.save(f"results/e10_refit/nov_energy_{f}.npy", nv.astype(np.float32))
        return dict(kind=kind, family=f, det=res, tuned=dict(xgb=bx, lgb=bl, clad=hp), time_s=time.time() - t0)
    keep = list(range(X.shape[1]))
    if kind == "P2-filtered":
        keep = single_feature_filter(X[tr], y[tr])
    Xf = X[:, keep]
    mods = leading_models(seed)
    res = fit_eval(mods, Xf, y, fam, tr, va, te, grp, seed)
    return dict(kind=kind, family=f, det=res, removed=[D["names"][j] for j in range(X.shape[1]) if j not in keep],
                n_train=len(tr), time_s=time.time() - t0)


def _xgb(seed, h):
    from zdt.models import XGB
    import xgboost as xgb
    m = XGB(seed)
    orig = m.fit

    def fit(X, y, Xva=None, yva=None, fam=None):
        m.m = xgb.XGBClassifier(n_estimators=300, subsample=0.9, colsample_bytree=0.9, n_jobs=1, tree_method="hist",
                                random_state=seed, eval_metric="logloss", **h).fit(X, y)
        return m
    m.fit = fit
    return m


def _lgb(seed, h):
    from zdt.models import LGBM
    import lightgbm as lgb
    m = LGBM(seed)

    def fit(X, y, Xva=None, yva=None, fam=None):
        m.m = lgb.LGBMClassifier(n_estimators=300, subsample=0.9, subsample_freq=1, colsample_bytree=0.9, n_jobs=1,
                                 random_state=seed, verbose=-1, **h).fit(X, y)
        return m
    m.fit = fit
    return m


_M = {}


def majority_data():
    """Cleaning variant: exact duplicates removed, conflicting feature groups resolved to their majority label
    (ties: the group is removed); splits drawn with the P2 scheme and seed."""
    if not _M:
        import pandas as pd
        from zdt.data import DATA_DIR, unsw_matrix
        from experiments.exp_p2_refit import replay_splits
        a = pd.read_csv(os.path.join(DATA_DIR, "UNSW_NB15_training-set.csv"))
        b = pd.read_csv(os.path.join(DATA_DIR, "UNSW_NB15_testing-set.csv"))
        for d in (a, b):
            d.columns = [c.strip().lstrip("﻿") for c in d.columns]
        df = pd.concat([a, b], ignore_index=True).drop(columns=["id"])
        df["attack_cat"] = df["attack_cat"].str.strip()
        feats = [c for c in df.columns if c not in ("attack_cat", "label")]
        df["_k"] = pd.util.hash_pandas_object(df[feats], index=False).values
        cnt = df.groupby(["_k", "attack_cat"]).size().rename("n").reset_index().sort_values(["_k", "n"], ascending=[True, False])
        first = cnt.groupby("_k").nth(0).set_index("_k")
        second = cnt.groupby("_k").nth(1).set_index("_k")["n"]
        tie = second.reindex(first.index).fillna(0).values == first["n"].values
        winner = first.loc[~tie, "attack_cat"]
        df = df.drop_duplicates(subset=["_k"]).copy()
        df = df[df["_k"].isin(winner.index)]
        df["attack_cat"] = winner.loc[df["_k"]].values
        df["label"] = (df["attack_cat"] != "Normal").astype(int)
        df = df.drop(columns=["_k"]).reset_index(drop=True)
        X, names = unsw_matrix(df)
        fam = df["attack_cat"].values.astype(str)
        _M.update(X=X, names=names, fam=fam, y=(fam != "Normal").astype(int), S=replay_splits(fam), n=len(df),
                  counts={k: int(v) for k, v in zip(*np.unique(fam, return_counts=True))})
    return _M


def main():
    from zdt.data import UNSW_FAMILIES
    from experiments.exp_nfv3_temporal import FAMS
    tasks = [("P2-plus", f) for f in UNSW_FAMILIES] + [("P2-filtered", f) for f in UNSW_FAMILIES] + \
            [("P2-majority", f) for f in UNSW_FAMILIES] + [("P5-filtered", f) for f in [None] + FAMS]
    t0 = time.time()
    res = []
    with Pool(6) as mp:
        for r in mp.imap_unordered(job, tasks):
            res.append(r)
            h = r["det"].get("CZT-Hybrid") or r["det"].get("XGBoost-tuned")
            print(f"done {r['kind']} {r['family']} AUROC {h.get('auroc_zd', float('nan')):.3f} [{time.time() - t0:.0f}s]", flush=True)
            json.dump(dict(results=res), open("results/e18_detection_robust.json", "w"), indent=1, default=float)
    M = majority_data()
    json.dump(dict(results=res, majority_cleaning=dict(rows=M["n"], counts=M["counts"])),
              open("results/e18_detection_robust.json", "w"), indent=1, default=float)
    print("done", time.time() - t0, flush=True)


if __name__ == "__main__":
    main()
