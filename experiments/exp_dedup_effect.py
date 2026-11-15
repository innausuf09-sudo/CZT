"""E1b: effect of de-duplication on i.i.d. cross-validation (same protocol as P1)."""
import json
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from zdt.data import load_unsw, unsw_matrix
from zdt.models import REGISTRY
from zdt.metrics import full_metrics

rows = []
for dedup in [False, True]:
    df, st = load_unsw(dedup=dedup, verbose=False)
    X, _ = unsw_matrix(df)
    fam = df["attack_cat"].values.astype(str); y = (fam != "Normal").astype(int)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(y), 24000, replace=False)
    X, y, fam = X[idx], y[idx], fam[idx]
    # fraction of test records that have an exact duplicate in the training portion (fold 0)
    skf = StratifiedKFold(5, shuffle=True, random_state=0)
    for k, (tr, te) in enumerate(skf.split(X, fam)):
        if k == 0:
            trset = set(map(bytes, X[tr].round(6).astype(np.float32)))
            overlap = float(np.mean([bytes(r) in trset for r in X[te].round(6).astype(np.float32)]))
        tr, va = train_test_split(tr, test_size=0.15, stratify=fam[tr], random_state=k)
        for name in ["LR", "RF", "XGBoost", "LightGBM", "DNN"]:
            m = REGISTRY[name](k).fit(X[tr], y[tr], X[va], y[va], fam=fam[tr]).calibrate(X[va], y[va])
            r = full_metrics(y[te], m.proba(X[te]))
            r.update(model=name, fold=k, dedup=dedup, test_overlap=overlap)
            rows.append(r)
            print(dedup, k, name, round(r["accuracy"], 4), round(r["auroc"], 4), flush=True)
    json.dump(rows, open("results/e1b_dedup_effect.json", "w"), indent=1, default=float)
