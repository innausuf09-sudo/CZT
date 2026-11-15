"""E23 / protocol P6 (revision R2.4): site-disjoint test between NF-UNSW-NB15-v3 and NF-CICIDS2018-v3.

Both corpora use the same 53 NetFlow v3 features, so no feature mapping is needed. NF-CICIDS2018-v3
(SHA-1 ddcad6d826b73a128a447a0189137efd6c8f70c9) is reduced with tools/sample_nfv3.py (seed 20261109: up to
60,000 flows per attack class and about 1,000,000 benign flows; see data/nfv3/NF-CICIDS2018-v3_sample_info.txt).

Preprocessing as in P5: identifiers removed, negative values clipped, log(1 + x), exact duplicates and
label-conflicting feature vectors removed within each corpus, non-finite values replaced by the largest finite
value of the feature in the training corpus.

Tasks
- cross: train and validate on the source corpus (benign 10,000 / 2,000, each attack class at most 2,000 / 300,
  validation a random 15%), test on the target corpus (5,000 benign and at most 500 flows of every attack class).
  No target attack class is seen in training, and the threshold is the 1% FPR threshold of the source validation set.
  Directions UNSW -> CIC and CIC -> UNSW; feature variants full, nottl (MIN_TTL, MAX_TTL, DNS_TTL_ANSWER removed)
  and filtered (every feature with source-training single-feature AUROC above 0.95 removed); seeds 0, 1, 2.
- within: the same caps, but training and test drawn from disjoint records of one corpus (reference).
Detectors: XGBoost, LightGBM, DNN, ResSE-CNN1D, B-ProtoMeta, CZT-Hybrid and CLAD (lr 0.001, width 128, depth 2,
no dropout, the configuration selected most often in P2).

Run from code/:  python -m experiments.exp_site_disjoint
Output: results/e23_site_disjoint.json, results/e23_meta.json
"""
import json
import os
import time
from multiprocessing import Pool

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

UNSW = "data/nfv3/NF-UNSW-NB15-v3.csv"
CIC = "data/nfv3/NF-CICIDS2018-v3_sample.csv"
CACHE = "results/e23_cache.npz"
DROP = ["IPV4_SRC_ADDR", "IPV4_DST_ADDR", "L4_SRC_PORT", "L4_DST_PORT", "FLOW_START_MILLISECONDS",
        "FLOW_END_MILLISECONDS", "DNS_QUERY_ID"]
TTL = ["MIN_TTL", "MAX_TTL", "DNS_TTL_ANSWER"]
CLAD_HP = dict(lr=1e-3, d_model=128, depth=2, dropout=0.0)


def load(path):
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    df["Attack"] = df["Attack"].astype(str).str.strip()
    feats = [c for c in df.columns if c not in DROP + ["Label", "Attack"]]
    n0 = len(df)
    df = df.drop_duplicates(subset=feats + ["Attack"]); n1 = len(df)
    g = df.groupby(feats, dropna=False)["Attack"].transform("nunique")
    df = df[g == 1]
    X = np.log1p(df[feats].astype(np.float64).clip(lower=0).values)
    return X, df["Attack"].to_numpy(dtype=str), feats, dict(raw=n0, after_exact_dedup=n1, final=len(df),
                                                             counts={k: int(v) for k, v in df["Attack"].value_counts().items()})


def build_cache():
    Xu, fu, feats, mu = load(UNSW)
    Xc, fc, feats_c, mc = load(CIC)
    assert feats == feats_c
    meta = dict(features=feats, unsw=mu, cic=mc)
    np.savez_compressed(CACHE, Xu=Xu.astype(np.float32), fu=fu, Xc=Xc.astype(np.float32), fc=fc)
    json.dump(meta, open("results/e23_meta.json", "w"), indent=1)


def finite(X, ref):
    X = X.copy()
    for c in range(X.shape[1]):
        bad = ~np.isfinite(X[:, c])
        if bad.any():
            ok = ref[:, c][np.isfinite(ref[:, c])]
            X[bad, c] = ok.max() if len(ok) else 0.0
    return X


def sample(fam, rng, cap_b, cap_a, exclude=None):
    idx = []
    for c in np.unique(fam):
        w = np.where(fam == c)[0]
        if exclude is not None:
            w = np.setdiff1d(w, exclude)
        rng.shuffle(w)
        idx.append(w[: cap_b if c == "Benign" else cap_a])
    return np.concatenate(idx)


def job(task):
    import torch
    from experiments.exp_detection_robust import leading_models, single_feature_filter
    from zdt.clad import CLAD
    from zdt.metrics import threshold_at_fpr, tpr_at_fpr
    from zdt.models import Hybrid
    torch.set_num_threads(1)
    kind, src, variant, seed = task
    t0 = time.time()
    z = np.load(CACHE)
    meta = json.load(open("results/e23_meta.json"))
    D = {"UNSW": (z["Xu"], z["fu"]), "CIC": (z["Xc"], z["fc"])}
    Xs, fs = D[src]
    tgt = src if kind == "within" else ("CIC" if src == "UNSW" else "UNSW")
    Xt, ft = D[tgt]
    rng = np.random.default_rng(seed)
    pool = sample(fs, rng, 12000, 2300)
    trva_tr, trva_va = [], []
    for c in np.unique(fs[pool]):
        w = pool[fs[pool] == c]; nv = int(0.15 * len(w))
        cv = 2000 if c == "Benign" else 300
        trva_va.append(w[:nv][:cv]); trva_tr.append(w[nv:][: 10000 if c == "Benign" else 2000])
    tr, va = np.concatenate(trva_tr), np.concatenate(trva_va)
    te = sample(ft, rng, 5000, 500, exclude=pool if kind == "within" else None)
    Xtr, Xva = finite(Xs[tr], Xs[tr]), finite(Xs[va], Xs[tr])
    Xte = finite(Xt[te], Xs[tr])
    ytr, yva, yte = (fs[tr] != "Benign").astype(int), (fs[va] != "Benign").astype(int), (ft[te] != "Benign").astype(int)
    keep = list(range(Xtr.shape[1]))
    if variant == "nottl":
        keep = [i for i, n in enumerate(meta["features"]) if n not in TTL]
    elif variant == "filtered":
        keep = single_feature_filter(Xtr, ytr)
    Xtr, Xva, Xte = Xtr[:, keep], Xva[:, keep], Xte[:, keep]
    famtr = np.where(fs[tr] == "Benign", "Normal", fs[tr])
    models = {k: v for k, v in leading_models(seed).items() if k != "CZT-Hybrid"}
    for m in models.values():
        m.fit(Xtr, ytr, Xva, yva, fam=famtr); m.calibrate(Xva, yva)
    h = Hybrid(seed, members=[models["LightGBM"], models["ResSE-CNN1D"], models["B-ProtoMeta"]])
    h.fit(Xtr, ytr, Xva, yva); h.calibrate(Xva, yva); models["CZT-Hybrid"] = h
    models["CLAD"] = CLAD(seed, **CLAD_HP).fit(Xtr, ytr).calibrate(Xva, yva)
    out = dict(kind=kind, source=src, target=tgt, variant=variant, seed=seed, n_train=len(tr), n_val=len(va),
               n_test=len(te), test_counts={k: int(v) for k, v in zip(*np.unique(ft[te], return_counts=True))},
               removed=[meta["features"][j] for j in range(len(meta["features"])) if j not in keep], det={})
    for name, m in models.items():
        pva, p = m.proba(Xva), m.proba(Xte)
        thr = threshold_at_fpr(pva[yva == 0], 0.01)
        out["det"][name] = dict(auroc=float(roc_auc_score(yte, p)), tpr_fpr1=float(tpr_at_fpr(yte, p, 0.01)),
                                detection=float((p[yte == 1] >= thr).mean()), test_fpr=float((p[yte == 0] >= thr).mean()),
                                per_class={c: float((p[ft[te] == c] >= thr).mean()) for c in np.unique(ft[te]) if c != "Benign"})
    out["time_s"] = time.time() - t0
    return out


def main():
    t0 = time.time()
    if not os.path.exists(CACHE):
        build_cache()
        print(f"cache built [{time.time() - t0:.0f}s]", flush=True)
    T = [("cross", s, v, sd) for s in ("UNSW", "CIC") for v in ("full", "nottl", "filtered") for sd in (0, 1, 2)]
    T += [("within", s, "full", sd) for s in ("UNSW", "CIC") for sd in (0, 1, 2)]
    res = []
    with Pool(max(1, (os.cpu_count() or 2) - 2)) as p:
        for r in p.imap_unordered(job, T):
            res.append(r)
            print(f"done {r['kind']} {r['source']}->{r['target']} {r['variant']} s{r['seed']} "
                  f"XGB {r['det']['XGBoost']['auroc']:.3f} hybrid {r['det']['CZT-Hybrid']['auroc']:.3f} "
                  f"FPR {r['det']['CZT-Hybrid']['test_fpr']:.3f} [{time.time() - t0:.0f}s]", flush=True)
    json.dump(dict(results=res), open("results/e23_site_disjoint.json", "w"), indent=1, default=float)
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
