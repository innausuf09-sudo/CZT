"""Single-feature separability of NF-UNSW-NB15-v3 February flows (testbed artefacts, revision R2.4).

Run from code/ after experiments/exp_nfv3_temporal.py has built results/e13_nfv3_cache.npz:
    python -m analysis.nfv3_artefacts
Writes results/e13_ttl_check.json: AUROC of every single feature for attack versus benign on a seeded sample of
60,000 February flows, and the TTL value distribution per class.
"""
import json

import numpy as np
from sklearn.metrics import roc_auc_score

z = np.load("results/e13_nfv3_cache.npz")
X, fam, per = z["X"], z["fam"], z["period"]
F = json.load(open("results/e13_nfv3_meta.json"))["features"]
rng = np.random.default_rng(0)
idx = rng.choice(np.where(per == "feb")[0], 60000, replace=False)
y = (fam[idx] != "Benign").astype(int)
auc = {}
for j, f in enumerate(F):
    a = roc_auc_score(y, X[idx, j])
    auc[f] = round(float(max(a, 1 - a)), 4)
ttl = {}
for f in ("MIN_TTL", "MAX_TTL"):
    j = F.index(f)
    ttl[f] = {}
    for c in np.unique(fam[idx]):
        v = np.round(np.expm1(X[idx][fam[idx] == c, j])).astype(int)
        u, n = np.unique(v, return_counts=True)
        ttl[f][str(c)] = {int(a): int(b) for a, b in zip(u, n)}
json.dump(dict(sample=60000, seed=0, single_feature_auroc=dict(sorted(auc.items(), key=lambda kv: -kv[1])), ttl_values=ttl),
          open("results/e13_ttl_check.json", "w"), indent=1)
print(list(sorted(auc.items(), key=lambda kv: -kv[1]))[:5])
