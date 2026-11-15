"""E13 (revision R2.4): NF-UNSW-NB15-v3 temporal split (train January, test February).

Data: data/nfv3/NF-UNSW-NB15-v3.csv (University of Queensland NetFlow v3 release;
SHA-1 2cba4879ead17c51e3268a01c06af41186cd90e4). Captures on 22-23 January 2015 and
18 February 2015 (flow start timestamps, UTC).

Preprocessing (leakage-averse, as for UNSW-NB15)
- removed identifiers: IPV4_SRC_ADDR, IPV4_DST_ADDR, L4_SRC_PORT, L4_DST_PORT,
  FLOW_START_MILLISECONDS, FLOW_END_MILLISECONDS (timestamps used only for the split),
  DNS_QUERY_ID (transaction identifier)
- remaining 46 numeric features, negative values clipped to 0, log(1 + x)
- exact duplicates (features + class) and feature-identical rows with conflicting classes
  removed, separately within each period
- variant 'nottl' also removes MIN_TTL, MAX_TTL and DNS_TTL_ANSWER

Protocols
- T-known: all nine families seen in January; test on February benign and attacks.
- T-zero-day: for each family f, f removed from the January training and validation data;
  test on February benign versus February f (zero-day) and February other families (known).
Caps per fold as in P2: benign 10,000 / 2,000 / 3,000 (train / validation / test), each attack
family at most 2,000 / 300 in January, at most 3,000 test records per group in February.
Validation is a random 15% of January (benign and each family separately).
Detectors: XGBoost, LightGBM, DNN, ResSE-CNN1D, B-ProtoMeta, CZT-Hybrid and CLAD (search on
training families only), Platt calibration and the 1% validation-FPR threshold as in P2.

Run from code/:  python -m experiments.exp_nfv3_temporal
Output: results/e13_nfv3_temporal.json, results/e13_nfv3_meta.json
"""
import json
import os
import time
from multiprocessing import Pool

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

SRC = "data/nfv3/NF-UNSW-NB15-v3.csv"
CACHE = "results/e13_nfv3_cache.npz"
DROP = ["IPV4_SRC_ADDR", "IPV4_DST_ADDR", "L4_SRC_PORT", "L4_DST_PORT", "FLOW_START_MILLISECONDS",
        "FLOW_END_MILLISECONDS", "DNS_QUERY_ID"]
TTL = ["MIN_TTL", "MAX_TTL", "DNS_TTL_ANSWER"]
FAMS = ["Generic", "Exploits", "Fuzzers", "DoS", "Reconnaissance", "Analysis", "Backdoor", "Shellcode", "Worms"]
LEADING = ["XGBoost", "LightGBM", "DNN", "ResSE-CNN1D", "B-ProtoMeta", "CZT-Hybrid"]


def clean(df, feats):
    n0 = len(df)
    df = df.drop_duplicates(subset=feats + ["Attack"])
    n1 = len(df)
    g = df.groupby(feats, dropna=False)["Attack"].transform("nunique")
    df = df[g == 1]
    return df, dict(raw=n0, after_exact_dedup=n1, final=len(df))


def build_cache():
    df = pd.read_csv(SRC, low_memory=False)
    meta = {"rows": len(df)}
    day = pd.to_datetime(df["FLOW_START_MILLISECONDS"], unit="ms", utc=True).dt.strftime("%Y-%m-%d")
    meta["days"] = {d: int(n) for d, n in day.value_counts().sort_index().items()}
    df["period"] = np.where(day.str.startswith("2015-01"), "jan", np.where(day.str.startswith("2015-02"), "feb", "other"))
    df["Attack"] = df["Attack"].astype(str).str.strip()
    meta["counts_by_period_raw"] = {p: {k: int(v) for k, v in df.loc[df.period == p, "Attack"].value_counts().items()}
                                    for p in ("jan", "feb", "other")}
    feats = [c for c in df.columns if c not in DROP + ["Label", "Attack", "period"]]
    parts, meta["cleaning"] = [], {}
    for p in ("jan", "feb"):
        d, st = clean(df[df.period == p], feats)
        meta["cleaning"][p] = st
        parts.append(d)
    df = pd.concat(parts)
    meta["counts_by_period_clean"] = {p: {k: int(v) for k, v in df.loc[df.period == p, "Attack"].value_counts().items()}
                                      for p in ("jan", "feb")}
    X = np.log1p(df[feats].astype(np.float64).clip(lower=0).values).astype(np.float32)
    bad = ~np.isfinite(X)  # infinite rate features in the release: replaced by the largest finite January value
    jan = (df["period"] == "jan").values
    meta["non_finite_values"] = dict(per_feature={feats[c]: int(bad[:, c].sum()) for c in np.where(bad.any(0))[0]},
                                     rows=int(bad.any(1).sum()),
                                     handling="replaced by the largest finite January value of the feature")
    for c in np.where(bad.any(0))[0]:
        X[bad[:, c], c] = np.max(X[jan & np.isfinite(X[:, c]), c])
    meta["features"] = feats
    meta["removed_identifiers"] = DROP
    np.savez_compressed(CACHE, X=X, fam=np.asarray(df["Attack"].to_numpy(dtype=str)), period=np.asarray(df["period"].to_numpy(dtype=str)))
    json.dump(meta, open("results/e13_nfv3_meta.json", "w"), indent=1)
    return meta


def split(fam, period, holdout, seed):
    rng = np.random.default_rng(seed)
    tr, va, te, grp = [], [], [], []
    for c in ["Benign"] + FAMS:
        if c == holdout:
            continue
        jan = np.where((fam == c) & (period == "jan"))[0]
        rng.shuffle(jan)
        nv = int(0.15 * len(jan))
        ctr, cva = (10000, 2000) if c == "Benign" else (2000, 300)
        va.append(jan[:nv][:cva]); tr.append(jan[nv:][:ctr])
    feb_b = np.where((fam == "Benign") & (period == "feb"))[0]
    feb_k = np.where((fam != "Benign") & (fam != str(holdout)) & (period == "feb"))[0]
    te_b = rng.choice(feb_b, min(3000, len(feb_b)), replace=False)
    te_k = rng.choice(feb_k, min(3000, len(feb_k)), replace=False)
    te_parts, grp = [te_b, te_k], ["normal"] * len(te_b) + ["known"] * len(te_k)
    if holdout is not None:
        feb_z = np.where((fam == holdout) & (period == "feb"))[0]
        te_z = rng.choice(feb_z, min(3000, len(feb_z)), replace=False)
        te_parts.append(te_z); grp += ["zeroday"] * len(te_z)
    return np.concatenate(tr), np.concatenate(va), np.concatenate(te_parts), np.array(grp)


def job(task):
    import torch
    from experiments.exp_p2_refit import clad_search
    from zdt.clad import CLAD
    from zdt.metrics import threshold_at_fpr, tpr_at_fpr
    from zdt.models import train_all
    torch.set_num_threads(1)
    variant, holdout = task
    z = np.load(CACHE)
    X, fam, period = z["X"], z["fam"], z["period"]
    meta = json.load(open("results/e13_nfv3_meta.json"))
    if variant == "nottl":
        keep = [i for i, n in enumerate(meta["features"]) if n not in TTL]
        X = X[:, keep]
    seed = 0 if holdout is None else FAMS.index(holdout) + 1
    tr, va, te, grp = split(fam, period, holdout, seed)
    y = (fam != "Benign").astype(int)
    t0 = time.time()
    fitted, _ = train_all(X[tr], y[tr], X[va], y[va], np.where(fam[tr] == "Benign", "Normal", fam[tr]), seed=seed,
                          names=LEADING, log=lambda *a: None)
    fz = np.where(fam == "Benign", "Normal", fam)
    hp, _ = clad_search(X[tr], y[tr], fz[tr], X[va], y[va], fz[va], holdout, seed)
    fitted["CLAD"] = CLAD(seed, **hp).fit(X[tr], y[tr]).calibrate(X[va], y[va])
    isN, isK, isZ = grp == "normal", grp == "known", grp == "zeroday"
    out = dict(variant=variant, holdout=holdout, n_train=len(tr), n_val=len(va), n_test=len(te),
               test_counts={g: int((grp == g).sum()) for g in ("normal", "known", "zeroday")}, clad_hp=hp, det={})
    for name, m in fitted.items():
        pva = m.proba(X[va])
        thr = threshold_at_fpr(pva[y[va] == 0], 0.01)
        p = m.proba(X[te])
        r = dict(test_fpr=float((p[isN] >= thr).mean()), known_recall=float((p[isK] >= thr).mean()),
                 auroc_known=roc_auc_score(isK[isN | isK].astype(int), p[isN | isK]))
        if isZ.any():
            mz = isN | isZ
            r.update(auroc_zd=roc_auc_score(isZ[mz].astype(int), p[mz]), zdr=float((p[isZ] >= thr).mean()),
                     tpr_fpr1_zd=tpr_at_fpr(isZ[mz].astype(int), p[mz], 0.01))
        out["det"][name] = r
    out["time_s"] = time.time() - t0
    return out


def main():
    if not os.path.exists(CACHE):
        meta = build_cache()
        print(json.dumps({k: meta[k] for k in ("rows", "days", "cleaning", "counts_by_period_clean")}, indent=1), flush=True)
    meta = json.load(open("results/e13_nfv3_meta.json"))
    jan, feb = meta["counts_by_period_clean"]["jan"], meta["counts_by_period_clean"]["feb"]
    hold = [f for f in FAMS if feb.get(f, 0) >= 50]  # a held-out family needs February test records only
    tasks = [(v, h) for v in ("full", "nottl") for h in [None] + hold]
    t0 = time.time()
    res = []
    with Pool(6) as mp:
        for r in mp.imap_unordered(job, tasks):
            res.append(r)
            h = r["det"]["CZT-Hybrid"]
            print(f"done {r['variant']} {r['holdout']} hybrid known AUROC {h['auroc_known']:.3f} "
                  f"zd AUROC {h.get('auroc_zd', float('nan')):.3f} [{time.time() - t0:.0f}s]", flush=True)
    json.dump(dict(holdout_families=hold, folds=res), open("results/e13_nfv3_temporal.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
