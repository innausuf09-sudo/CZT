"""E10 (revision): refit of the leading P2 detectors on the original family hold-out splits.

Purposes
- R2.4: the same detectors without the TTL-type features (sttl, dttl, ct_state_ttl).
- R2.5: the CLAD contrastive comparator (zdt/clad.py), hyperparameters searched on
  training families only (inner hold-out of known families, never the outer zero-day family).
- Label-free novelty scores for the triage (revision instructions, item 1.1), all fitted
  without zero-day labels:
    proto      squared distance to the nearest known-attack prototype of B-ProtoMeta
    epistemic  MC-dropout standard deviation of B-ProtoMeta's benign probability
    disagree   standard deviation of the calibrated log-odds of the three hybrid members
    knn        mean distance to the 5 nearest training attack records (known families only)
    combined   mean of the normalised proto, disagree and knn scores
  Each raw score is mapped to [0, 1] by the empirical CDF of the validation known-attack
  records (labels available at training time).
- R1.1 / R1.5: validation scores of every class, validation priors, split indices and
  overlap checks.

The splits are replayed with the RNG stream of experiments/exp_unsw.py (fh mode) and
checked against the stored labels in results/e2_scores.npz.

Run from code/:  python -m experiments.exp_p2_refit
Output: results/e10_refit/{variant}_{family}.npz, results/e10_refit.json, results/e10_splits.npz
"""
import json
import os
import sys
import time
from multiprocessing import Pool

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
from scipy.special import logit
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

OUT = "results/e10_refit/"
TTL = ["sttl", "dttl", "ct_state_ttl"]
LEADING = ["XGBoost", "LightGBM", "DNN", "ResSE-CNN1D", "B-ProtoMeta", "CZT-Hybrid"]
CLAD_GRID = [dict(lr=lr, d_model=w, depth=dp, dropout=do)
             for lr in (1e-3, 3e-4) for (w, dp) in ((256, 3), (128, 2)) for do in (0.0, 0.2)]


def replay_splits(fam_all):
    """Exact replay of the E2 split draws (experiments/exp_unsw.py, mode fh, not quick)."""
    from zdt.data import UNSW_FAMILIES
    rng = np.random.default_rng(1)
    normal_idx = np.where(fam_all == "Normal")[0]
    rng.shuffle(normal_idx)
    nN = len(normal_idx)
    n_tr, n_va = normal_idx[: int(.7 * nN)], normal_idx[int(.7 * nN): int(.85 * nN)]
    n_te = normal_idx[int(.85 * nN):]
    cap_n_tr, cap_n_va, cap_n_te = 10000, 2000, 3000
    cap_fam_tr, cap_fam_va, cap_test = 2000, 300, 3000
    S = {}
    for f in UNSW_FAMILIES:
        tr_idx = [rng.choice(n_tr, cap_n_tr, replace=False)]
        va_idx = [rng.choice(n_va, cap_n_va, replace=False)]
        known_te = []
        for g in UNSW_FAMILIES:
            if g == f:
                continue
            gi = np.where(fam_all == g)[0].copy()
            rng.shuffle(gi)
            a, b = int(.7 * len(gi)), int(.85 * len(gi))
            tr_idx.append(gi[:a][:cap_fam_tr]); va_idx.append(gi[a:b][:cap_fam_va]); known_te.append(gi[b:])
        tr_idx, va_idx = np.concatenate(tr_idx), np.concatenate(va_idx)
        known_te = np.concatenate(known_te)
        known_te = rng.choice(known_te, min(cap_test, len(known_te)), replace=False)
        zd = np.where(fam_all == f)[0]
        zd = rng.choice(zd, min(cap_test, len(zd)), replace=False)
        nte = rng.choice(n_te, cap_n_te, replace=False)
        te_idx = np.concatenate([nte, known_te, zd])
        grp = np.array(["normal"] * len(nte) + ["known"] * len(known_te) + ["zeroday"] * len(zd))
        S[f] = dict(tr=tr_idx, va=va_idx, te=te_idx, grp=grp)
    return S


_D = {}


def data():
    if not _D:
        from zdt.data import load_unsw, unsw_matrix
        df, _ = load_unsw(verbose=False)
        X, names = unsw_matrix(df)
        fam = df["attack_cat"].values.astype(str)
        _D.update(X=X, names=names, fam=fam, y=(fam != "Normal").astype(int), S=replay_splits(fam))
    return _D


def ecdf_ref(ref):
    ref = np.sort(ref)
    return lambda s: np.searchsorted(ref, s, side="right") / len(ref)


def clad_search(Xtr, ytr, famtr, Xva, yva, famva, f, seed):
    """Inner search on training families only: two known families are held out in turn
    (never the outer zero-day family f); score = mean AUROC (benign vs held-out family) on validation."""
    from zdt.clad import CLAD
    from zdt.data import UNSW_FAMILIES
    inner = [g for g in UNSW_FAMILIES if g != f][:2]
    res = []
    for hp in CLAD_GRID:
        au = []
        for g in inner:
            m = famtr != g
            mod = CLAD(seed, **hp).fit(Xtr[m], ytr[m])
            mv = (famva == "Normal") | (famva == g)
            au.append(roc_auc_score((famva[mv] == g).astype(int), mod.raw_score(Xva[mv])))
        res.append(float(np.mean(au)))
    k = int(np.argmax(res))
    return CLAD_GRID[k], dict(inner_families=inner, grid=CLAD_GRID, inner_auroc=res)


def job(task):
    import torch
    from zdt.clad import CLAD
    from zdt.data import UNSW_FAMILIES
    from zdt.metrics import threshold_at_fpr
    from zdt.models import train_all, _p2z
    torch.set_num_threads(1)
    variant, f = task
    D = data()
    X = D["X"]
    if variant == "nottl":
        keep = [i for i, n in enumerate(D["names"]) if n not in TTL]
        X = X[:, keep]
    fam, y, s = D["fam"], D["y"], D["S"][f]
    tr, va, te, grp = s["tr"], s["va"], s["te"], s["grp"]
    seed = UNSW_FAMILIES.index(f)
    t0 = time.time()
    fitted, times = train_all(X[tr], y[tr], X[va], y[va], fam[tr], seed=seed, names=LEADING, log=lambda *a: None)
    hp, search = clad_search(X[tr], y[tr], fam[tr], X[va], y[va], fam[va], f, seed)
    t1 = time.time()
    clad = CLAD(seed, **hp).fit(X[tr], y[tr])
    clad.calibrate(X[va], y[va])
    fitted["CLAD"] = clad
    times["CLAD"] = time.time() - t1
    out = dict(groups=grp, fam=fam[te].astype(str), fam_va=fam[va].astype(str), y_va=y[va])
    for name, m in fitted.items():
        out[f"{name}|p"] = m.proba(X[te]).astype(np.float32)
        out[f"{name}|pu"] = m.uncal_proba(X[te]).astype(np.float32)
        pva = m.proba(X[va])
        out[f"{name}|p_va"] = pva.astype(np.float32)
        out[f"{name}|pva_normal"] = pva[y[va] == 0].astype(np.float32)
        out[f"{name}|thr"] = np.float32(threshold_at_fpr(pva[y[va] == 0], 0.01))
    # ---------------- label-free novelty scores (no zero-day label used anywhere)
    bml, hyb = fitted["B-ProtoMeta"], fitted["CZT-Hybrid"]
    ni = bml.classes.index("Normal")
    att = [i for i, c in enumerate(bml.classes) if c != "Normal"]

    def proto(Xq):
        Xs = torch.tensor(bml._t(Xq))
        bml.enc.eval()
        with torch.no_grad():
            z = torch.cat([bml.enc(Xs[i:i + 8192]) for i in range(0, len(Xs), 8192)])
            d = torch.cdist(z, bml.protos[att]) ** 2 / z.shape[1]
        return d.min(1).values.numpy()

    def epistemic(Xq):
        torch.manual_seed(seed)
        bml._pnormal(Xq)
        return bml.last_epistemic.copy()

    def disagree(Xq):
        return np.std([_p2z(m.proba(Xq)) for m in hyb.members], 0)

    rng = np.random.default_rng(seed)
    tr_att = tr[y[tr] == 1]
    sc = StandardScaler().fit(X[tr_att])
    ref = X[rng.choice(tr_att, min(8000, len(tr_att)), replace=False)]
    nn_ = NearestNeighbors(n_neighbors=5).fit(sc.transform(ref))

    def knn(Xq):
        return nn_.kneighbors(sc.transform(Xq))[0].mean(1)

    va_att = va[y[va] == 1]
    nov_names = ["proto", "epistemic", "disagree", "knn"]
    for nm, fn in zip(nov_names, (proto, epistemic, disagree, knn)):
        F = ecdf_ref(fn(X[va_att]))
        out[f"nov|{nm}|raw"] = fn(X[te]).astype(np.float32)
        out[f"nov|{nm}"] = F(out[f"nov|{nm}|raw"]).astype(np.float32)
    out["nov|combined"] = np.mean([out[f"nov|{k}"] for k in ("proto", "disagree", "knn")], 0).astype(np.float32)
    np.savez_compressed(f"{OUT}{variant}_{f}.npz", **out)
    summ = dict(variant=variant, family=f, times=times, clad_hp=hp, clad_search=search, time_s=time.time() - t0,
                n_features=int(X.shape[1]))
    isZ, isK, isN = grp == "zeroday", grp == "known", grp == "normal"
    mz = isN | isZ
    from zdt.metrics import tpr_at_fpr
    summ["detectors"] = {}
    for name in fitted:
        p, thr = out[f"{name}|p"], float(out[f"{name}|thr"])
        summ["detectors"][name] = dict(auroc_zd=roc_auc_score(isZ[mz].astype(int), p[mz]),
                                       tpr_fpr1_zd=tpr_at_fpr(isZ[mz].astype(int), p[mz], 0.01),
                                       zdr=float((p[isZ] >= thr).mean()), test_fpr=float((p[isN] >= thr).mean()),
                                       known_recall=float((p[isK] >= thr).mean()))
    alert = out["CZT-Hybrid|p"] >= float(out["CZT-Hybrid|thr"])
    summ["novelty_auroc"] = {}
    for k in nov_names + ["combined"]:
        v = out[f"nov|{k}"]
        mk = isZ | isK
        ma = mk & alert
        summ["novelty_auroc"][k] = dict(
            all=roc_auc_score(isZ[mk].astype(int), v[mk]),
            alerts=roc_auc_score(isZ[ma].astype(int), v[ma]) if len(np.unique(isZ[ma])) == 2 else None)
    return summ


def main():
    os.makedirs(OUT, exist_ok=True)
    from zdt.data import UNSW_FAMILIES
    from zdt.io import load_scores
    D = data()
    # ---------------- split checks (R1.5, R1.9)
    store = load_scores("results/e2_scores.npz")
    checks, arrays = {}, {}
    for f, s in D["S"].items():
        same = bool(np.array_equal(D["fam"][s["te"]], store[f]["fam"].astype(str))
                    and np.array_equal(s["grp"], store[f]["groups"].astype(str)))
        tr, va, te = set(s["tr"]), set(s["va"]), set(s["te"])
        Xh = lambda idx: {hash(D["X"][i].tobytes()) for i in idx}
        htr, hva, hte = Xh(s["tr"]), Xh(s["va"]), Xh(s["te"])
        vf = D["fam"][s["va"]]
        checks[f] = dict(matches_stored_scores=same, n_train=len(s["tr"]), n_val=len(s["va"]), n_test=len(s["te"]),
                         index_overlap=dict(tr_va=len(tr & va), tr_te=len(tr & te), va_te=len(va & te)),
                         feature_vector_overlap=dict(tr_va=len(htr & hva), tr_te=len(htr & hte), va_te=len(hva & hte)),
                         zeroday_in_train_or_val=int(((D["fam"][s["tr"]] == f).sum() + (vf == f).sum())),
                         val_prior_attack=float((vf != "Normal").mean()),
                         val_counts={k: int(v) for k, v in zip(*np.unique(vf, return_counts=True))},
                         train_counts={k: int(v) for k, v in zip(*np.unique(D["fam"][s["tr"]], return_counts=True))},
                         test_counts={g: int((s["grp"] == g).sum()) for g in ("normal", "known", "zeroday")})
        for k in ("tr", "va", "te"):
            arrays[f"{f}|{k}"] = s[k]
        print(f, checks[f]["matches_stored_scores"], checks[f]["index_overlap"], checks[f]["feature_vector_overlap"],
              flush=True)
    np.savez_compressed("results/e10_splits.npz", **arrays)
    if not all(c["matches_stored_scores"] for c in checks.values()):
        print("WARNING: replayed splits differ from the stored scores", flush=True)
    tasks = [(v, f) for v in ("full", "nottl") for f in UNSW_FAMILIES]
    t0 = time.time()
    with Pool(max(1, (os.cpu_count() or 2) - 2)) as mp:
        res = []
        for r in mp.imap_unordered(job, tasks):
            res.append(r)
            print(f"done {r['variant']} {r['family']} {r['time_s']:.0f}s hybrid AUROC "
                  f"{r['detectors']['CZT-Hybrid']['auroc_zd']:.3f} CLAD {r['detectors']['CLAD']['auroc_zd']:.3f} "
                  f"nov(all) {r['novelty_auroc']['combined']['all']:.3f} [{time.time() - t0:.0f}s]", flush=True)
    json.dump(dict(split_checks=checks, folds=res, ttl_features=TTL), open("results/e10_refit.json", "w"), indent=1,
              default=float)


if __name__ == "__main__":
    main()
