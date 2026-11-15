"""E1 (i.i.d. stratified 5-fold CV) and E2 (leave-one-family-out zero-day
protocol) on de-duplicated UNSW-NB15."""
import sys, json, time, pickle
from zdt.io import save_scores, load_scores
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from zdt.data import load_unsw, unsw_matrix, UNSW_FAMILIES, UNSW_SEVERITY, stratified_cap
from zdt.models import train_all, MODEL_ORDER
from zdt.metrics import full_metrics, threshold_at_fpr, ece, tpr_at_fpr
from sklearn.metrics import roc_auc_score, average_precision_score

OUT = "results/"
mode = sys.argv[1] if len(sys.argv) > 1 else "all"
quick = "--quick" in sys.argv
log = lambda *a: print(*a, flush=True)

df, dstats = load_unsw()
X_all, feat_names = unsw_matrix(df)
fam_all = df["attack_cat"].values.astype(str)
y_all = (fam_all != "Normal").astype(int)
json.dump({"dedup": dstats, "features": feat_names,
           "class_counts": df["attack_cat"].value_counts().to_dict()}, open(OUT + "unsw_meta.json", "w"), indent=1)

# ------------------------------------------------------------------ E1: i.i.d.
if mode in ("all", "iid"):
    rng = np.random.default_rng(0)
    n = 6000 if quick else 24000
    idx = rng.choice(len(y_all), n, replace=False)
    X, y, fam = X_all[idx], y_all[idx], fam_all[idx]
    skf = StratifiedKFold(5, shuffle=True, random_state=0)
    rows, times_all = [], []
    import os
    if os.path.exists(OUT + "e1_iid.json"):
        rows = json.load(open(OUT + "e1_iid.json"))
    done1 = set(r["fold"] for r in rows)
    for k, (tr, te) in enumerate(skf.split(X, fam)):
        log(f"[E1] fold {k}")
        if k in done1:
            continue
        tr, va = train_test_split(tr, test_size=0.15, stratify=fam[tr], random_state=k)
        fitted, times = train_all(X[tr], y[tr], X[va], y[va], fam[tr], seed=k, log=log)
        for name, m in fitted.items():
            p = m.proba(X[te]); pu = m.uncal_proba(X[te])
            r = full_metrics(y[te], p); r.update(model=name, fold=k, train_time=times[name],
                                                 ece_uncal=ece(pu, y[te]))
            rows.append(r)
        # keep training curves of deep nets from fold 0
        if k == 0:
            hist = {n_: getattr(m, "history", None) for n_, m in fitted.items() if n_ != "CZT-Hybrid"}
            json.dump(hist, open(OUT + "e1_histories.json", "w"), default=float)
        json.dump(rows, open(OUT + "e1_iid.json", "w"), indent=1, default=float)

# ------------------------------------------------------------------ E2: family hold-out
if mode in ("all", "fh"):
    rng = np.random.default_rng(1)
    normal_idx = np.where(fam_all == "Normal")[0]
    rng.shuffle(normal_idx)
    nN = len(normal_idx)
    n_tr, n_va = normal_idx[: int(.7 * nN)], normal_idx[int(.7 * nN): int(.85 * nN)]
    n_te = normal_idx[int(.85 * nN):]
    cap_n_tr, cap_n_va, cap_n_te = (3000, 1000, 1000) if quick else (10000, 2000, 3000)
    cap_fam_tr, cap_fam_va, cap_test = (500, 150, 800) if quick else (2000, 300, 3000)
    fams = UNSW_FAMILIES[:2] if quick else UNSW_FAMILIES
    import os
    rows, store = [], {}
    if os.path.exists(OUT + "e2_fh.json") and os.path.exists(OUT + "e2_scores.npz") and "--fresh" not in sys.argv:  # resume support
        rows = json.load(open(OUT + "e2_fh.json"))
        store = load_scores(OUT + "e2_scores.npz")
        done = set(store) & set(r["family"] for r in rows)
        rows = [r for r in rows if r["family"] in done]
        store = {k: v for k, v in store.items() if k in done}
    for f in fams:
        log(f"[E2] held-out family = {f}")
        t0 = time.time()
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
        if f in store:  # already computed; random draws above were replayed to keep the RNG stream identical
            log("   (resumed: skipping completed fold)")
            continue
        fitted, times = train_all(X_all[tr_idx], y_all[tr_idx], X_all[va_idx], y_all[va_idx],
                                  fam_all[tr_idx], seed=UNSW_FAMILIES.index(f), log=log)
        store[f] = {"groups": grp, "fam": fam_all[te_idx], "y": y_all[te_idx], "scores": {}}
        for name, m in fitted.items():
            pva = m.proba(X_all[va_idx])
            thr = threshold_at_fpr(pva[y_all[va_idx] == 0], 0.01)
            p = m.proba(X_all[te_idx]); pu = m.uncal_proba(X_all[te_idx])
            store[f]["scores"][name] = {"p": p.astype(np.float32), "pu": pu.astype(np.float32), "thr": thr,
                                        "pva_normal": pva[y_all[va_idx] == 0].astype(np.float32)}
            isN, isK, isZ = grp == "normal", grp == "known", grp == "zeroday"
            mz = isN | isZ
            yz = isZ[mz].astype(int)
            rows.append(dict(model=name, family=f, n_zeroday=int(isZ.sum()), thr=thr,
                             zdr=float((p[isZ] >= thr).mean()), known_recall=float((p[isK] >= thr).mean()),
                             test_fpr=float((p[isN] >= thr).mean()),
                             auroc_zd=roc_auc_score(yz, p[mz]), auprc_zd=average_precision_score(yz, p[mz]),
                             tpr_fpr1_zd=tpr_at_fpr(yz, p[mz], 0.01),
                             auroc_known=roc_auc_score(isK[isN | isK].astype(int), p[isN | isK]),
                             ece_cal=ece(p[mz], yz), ece_uncal=ece(pu[mz], yz),
                             ece_all_cal=ece(p, y_all[te_idx]), ece_all_uncal=ece(pu, y_all[te_idx]),
                             mean_p_zd=float(p[isZ].mean()), train_time=times[name]))
            log(f"      {name:15s} ZDR={rows[-1]['zdr']:.3f} AUROC_zd={rows[-1]['auroc_zd']:.3f} "
                f"known={rows[-1]['known_recall']:.3f} FPR={rows[-1]['test_fpr']:.3f}")
        json.dump(rows, open(OUT + "e2_fh.json", "w"), indent=1, default=float)
        save_scores(store, OUT + "e2_scores.npz")
        if f == fams[0]:
            pickle.dump({"models": fitted, "va_idx": va_idx, "tr_idx": tr_idx, "te_idx": te_idx,
                         "feat_names": feat_names}, open(OUT + "e2_fold0_models.pkl", "wb"))
        log(f"  fold time {time.time() - t0:.0f}s")
