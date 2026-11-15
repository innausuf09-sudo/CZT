"""Run as: python -m experiments.retrain_fold0

Replays the P2 split for the first held-out family (Generic) and retrains the
three hybrid members so that permutation importance can be computed."""
import pickle
import numpy as np
from zdt.io import load_scores
from zdt.data import load_unsw, unsw_matrix, UNSW_FAMILIES
from zdt.models import LGBM, CNN1D, BML, Hybrid
df, _ = load_unsw(verbose=False)
X, names = unsw_matrix(df)
fam = df["attack_cat"].values.astype(str); y = (fam != "Normal").astype(int)
rng = np.random.default_rng(1)
nidx = np.where(fam == "Normal")[0]; rng.shuffle(nidx); nN = len(nidx)
n_tr, n_va, n_te = nidx[:int(.7*nN)], nidx[int(.7*nN):int(.85*nN)], nidx[int(.85*nN):]
f = UNSW_FAMILIES[0]
tr = [rng.choice(n_tr, 10000, replace=False)]; va = [rng.choice(n_va, 2000, replace=False)]; kt = []
for g in UNSW_FAMILIES:
    if g == f: continue
    gi = np.where(fam == g)[0].copy(); rng.shuffle(gi)
    a, b = int(.7*len(gi)), int(.85*len(gi))
    tr.append(gi[:a][:2000]); va.append(gi[a:b][:300]); kt.append(gi[b:])
tr, va = np.concatenate(tr), np.concatenate(va); kt = np.concatenate(kt)
kt = rng.choice(kt, min(3000, len(kt)), replace=False)
zd = np.where(fam == f)[0]; zd = rng.choice(zd, min(3000, len(zd)), replace=False)
nte = rng.choice(n_te, 3000, replace=False)
te = np.concatenate([nte, kt, zd])
st = load_scores("results/e2_scores.npz")[f]
assert (st["fam"] == fam[te]).all(), "split replay mismatch"
ms = [LGBM(0).fit(X[tr], y[tr]), CNN1D(0).fit(X[tr], y[tr], X[va], y[va]), BML(0).fit(X[tr], y[tr], fam=fam[tr])]
for m in ms: m.calibrate(X[va], y[va])
h = Hybrid(0, members=ms).fit(X[tr], y[tr], X[va], y[va]); h.calibrate(X[va], y[va])
pickle.dump({"models": {"LightGBM": ms[0], "ResSE-CNN1D": ms[1], "B-ProtoMeta": ms[2], "CZT-Hybrid": h},
             "te_idx": te, "va_idx": va, "tr_idx": tr, "feat_names": names}, open("results/e2_fold0_models.pkl", "wb"))
print("saved")
