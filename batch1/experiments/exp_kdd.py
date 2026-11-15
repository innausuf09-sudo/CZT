"""E3: NSL-KDD native novel-attack protocol (KDDTrain+ -> KDDTest+, 17 attack
types unseen in training).  E4: cross-dataset transfer UNSW-NB15 <-> NSL-KDD on
a harmonised 18-feature schema with k-shot target adaptation."""
import sys, json, time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from zdt.data import (load_kdd, kdd_matrix, load_unsw, unsw_harmonised, kdd_harmonised,
                      stratified_cap)
from zdt.models import train_all, REGISTRY, Hybrid, TorchModel, BML
from zdt.metrics import full_metrics, threshold_at_fpr, ece

OUT = "results/"
mode = sys.argv[1] if len(sys.argv) > 1 else "all"
log = lambda *a: print(*a, flush=True)
tr_df, te_df = load_kdd()

# ------------------------------------------------------------------ E3
if mode in ("all", "novel"):
    Xtr_all, names = kdd_matrix(tr_df)
    Xte, _ = kdd_matrix(te_df)
    cat_tr = tr_df["category"].values
    ytr_all = (cat_tr != "Normal").astype(int)
    yte = (te_df["category"].values != "Normal").astype(int)
    novel = te_df["novel"].values
    cat_te = te_df["category"].values
    meta = {"train_rows": len(tr_df), "test_rows": len(te_df),
            "train_counts": tr_df["category"].value_counts().to_dict(),
            "test_counts": te_df["category"].value_counts().to_dict(),
            "novel_rows": int(novel.sum()), "novel_types": sorted(te_df.loc[novel, "attack"].unique().tolist()),
            "n_train_types": int(tr_df["attack"].nunique()), "n_test_types": int(te_df["attack"].nunique()),
            "novel_by_category": te_df.loc[novel, "category"].value_counts().to_dict(), "features": names}
    json.dump(meta, open(OUT + "kdd_meta.json", "w"), indent=1)
    caps = {"Normal": 12000, "DoS": 6000, "Probe": 4000, "R2L": 10 ** 6, "U2R": 10 ** 6}
    import os
    rows = json.load(open(OUT + "e3_kdd_novel.json")) if os.path.exists(OUT + "e3_kdd_novel.json") else []
    done3 = set(r["seed"] for r in rows)
    for seed in range(3):
        log(f"[E3] seed {seed}")
        if seed in done3 and sum(r["seed"] == seed for r in rows) == 13:
            continue
        rows = [r for r in rows if r["seed"] != seed]
        rng = np.random.default_rng(100 + seed)
        idx = np.concatenate([rng.choice(np.where(cat_tr == c)[0], min(caps[c], (cat_tr == c).sum()), replace=False)
                              for c in caps])
        tr, va = train_test_split(idx, test_size=0.15, stratify=cat_tr[idx], random_state=seed)
        fitted, times = train_all(Xtr_all[tr], ytr_all[tr], Xtr_all[va], ytr_all[va], cat_tr[tr], seed=seed, log=log)
        for name, m in fitted.items():
            pva = m.proba(Xtr_all[va])
            thr = threshold_at_fpr(pva[ytr_all[va] == 0], 0.01)
            p = m.proba(Xte); pu = m.uncal_proba(Xte)
            r = full_metrics(yte, p)
            isN = yte == 0
            r.update(model=name, seed=seed, thr=thr, ece_uncal=ece(pu, yte),
                     fpr_thr=float((p[isN] >= thr).mean()),
                     dr_known_thr=float((p[(yte == 1) & ~novel] >= thr).mean()),
                     dr_novel_thr=float((p[novel] >= thr).mean()),
                     dr_known_05=float((p[(yte == 1) & ~novel] >= .5).mean()),
                     dr_novel_05=float((p[novel] >= .5).mean()),
                     auroc_novel=roc_auc_score(novel[isN | novel].astype(int), p[isN | novel]),
                     auroc_known=roc_auc_score(((yte == 1) & ~novel)[isN | ((yte == 1) & ~novel)].astype(int),
                                               p[isN | ((yte == 1) & ~novel)]),
                     train_time=times[name])
            for c in ["DoS", "Probe", "R2L", "U2R"]:
                mk = novel & (cat_te == c)
                r[f"dr_novel_{c}"] = float((p[mk] >= thr).mean()) if mk.any() else np.nan
                mk = (~novel) & (cat_te == c)
                r[f"dr_known_{c}"] = float((p[mk] >= thr).mean()) if mk.any() else np.nan
            rows.append(r)
            log(f"      {name:15s} AUROC={r['auroc']:.3f} F1={r['f1']:.3f} DRknown={r['dr_known_thr']:.3f} "
                f"DRnovel={r['dr_novel_thr']:.3f}")
        json.dump(rows, open(OUT + "e3_kdd_novel.json", "w"), indent=1, default=float)

# ------------------------------------------------------------------ E4
if mode in ("all", "transfer"):
    udf, _ = load_unsw(verbose=False)
    U = unsw_harmonised(udf); Ucat = udf["attack_cat"].values.astype(str)
    kdf = __import__("pandas").concat([tr_df, te_df], ignore_index=True)
    K = kdd_harmonised(kdf); Kcat = kdf["category"].values.astype(str)
    domains = {"UNSW-NB15": (U, Ucat), "NSL-KDD": (K, Kcat)}
    MODELS = ["LR", "RF", "XGBoost", "LightGBM", "DNN", "ResSE-CNN1D", "B-ProtoMeta", "CZT-Hybrid"]
    KS = [0, 10, 50, 100, 250, 500, 1000]
    import os
    rows = json.load(open(OUT + "e4_transfer.json")) if os.path.exists(OUT + "e4_transfer.json") else []
    for src, tgt in [("UNSW-NB15", "NSL-KDD"), ("NSL-KDD", "UNSW-NB15")]:
        Xs, cs = domains[src]; Xt, ct = domains[tgt]
        ys, yt = (cs != "Normal").astype(int), (ct != "Normal").astype(int)
        for seed in range(3):
            log(f"[E4] {src} -> {tgt} seed {seed}")
            if sum((r["src"] == src) and (r["seed"] == seed) for r in rows) == 7 * 8:
                continue
            rows = [r for r in rows if not ((r["src"] == src) and (r["seed"] == seed))]
            rng = np.random.default_rng(200 + seed)
            att = np.where(ys == 1)[0]
            sidx = np.concatenate([rng.choice(np.where(ys == 0)[0], 10000, replace=False),
                                   att[stratified_cap(cs[att], 2500, rng)]])
            tr, va = train_test_split(sidx, test_size=0.15, stratify=cs[sidx], random_state=seed)
            perm = rng.permutation(len(yt))
            te, pool = perm[:10000], perm[10000:]
            fitted, times = train_all(Xs[tr], ys[tr], Xs[va], ys[va], cs[tr], seed=seed,
                                      names=MODELS, log=log)
            src_state = {n: m for n, m in fitted.items()}
            # in-domain reference on held-out source validation
            for k in KS:
                if k == 0:
                    models = src_state
                else:
                    # stratified k-shot sample with both classes present
                    kk = pool[:k] if k >= 50 else None
                    if kk is None or len(np.unique(yt[kk])) < 2:
                        pos = pool[yt[pool] == 1][: max(1, k // 2)]; neg = pool[yt[pool] == 0][: k - len(pos)]
                        kk = np.concatenate([pos, neg])
                    Xk, yk, ck = Xt[kk], yt[kk], ct[kk]
                    models = {}
                    rep = int(min(50, max(1, round(0.25 * len(tr) / k))))
                    for n in MODELS:
                        if n == "CZT-Hybrid":
                            continue
                        if n in ("DNN", "ResSE-CNN1D"):
                            import copy
                            m = copy.deepcopy(src_state[n]).finetune(Xk, yk)
                        elif n == "B-ProtoMeta":
                            import copy
                            m = copy.deepcopy(src_state[n]).adapt(Xk, ck)
                        else:
                            Xa = np.concatenate([Xs[tr]] + [Xk] * rep); ya = np.concatenate([ys[tr]] + [yk] * rep)
                            ca = np.concatenate([cs[tr]] + [ck] * rep)
                            m = REGISTRY[n](seed).fit(Xa, ya, fam=ca)
                            m.cal = None
                        if m.cal is None:
                            m.calibrate(Xs[va], ys[va])
                        models[n] = m
                    h = Hybrid(seed, members=[models["LightGBM"], models["ResSE-CNN1D"], models["B-ProtoMeta"]])
                    h.calibrate(Xs[va], ys[va])
                    models["CZT-Hybrid"] = h
                for n, m in models.items():
                    p = m.proba(Xt[te])
                    rows.append(dict(src=src, tgt=tgt, seed=seed, k=k, model=n,
                                     auroc=roc_auc_score(yt[te], p), auprc=average_precision_score(yt[te], p),
                                     f1=f1_score(yt[te], (p >= .5).astype(int)), ece=ece(p, yt[te]),
                                     src_auroc=roc_auc_score(ys[va], m.proba(Xs[va])) if k == 0 else np.nan))
                log(f"   k={k:5d} " + " ".join(f"{r['model'][:6]}={r['auroc']:.3f}" for r in rows[-len(models):]))
            json.dump(rows, open(OUT + "e4_transfer.json", "w"), indent=1, default=float)
