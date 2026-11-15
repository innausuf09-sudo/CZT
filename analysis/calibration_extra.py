"""Additional calibration diagnostics (second review round).

P2 (stored refit scores, results/e10_refit/full_<family>.npz), on the simulated deployment mixture 97/2/1:
- calibration intercept and slope (weighted logistic regression of the label on the log-odds of the score);
- mean probability given to benign, known-family and unseen-family records (class-wise calibration);
- the deployment attack prior estimated without labels by the EM procedure of Saerens et al. (2002), and the ECE after
  a prior correction that uses this estimate instead of the true prior.

Run from code/:  python -m analysis.calibration_extra
Writes tables/s_table_calibration_extra.csv
"""
import json

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression

from analysis.calibration_prior import FAMS, MIX, PI_D, wece, correct

MODELS = ["XGBoost", "LightGBM", "DNN", "CZT-Hybrid", "CLAD"]


def intercept_slope(p, y, w):
    x = logit(np.clip(p, 1e-6, 1 - 1e-6)).reshape(-1, 1)
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(x, y, sample_weight=w / w.mean())
    return float(lr.intercept_[0]), float(lr.coef_[0, 0])


def em_prior(p, pi_v, w=None, it=200):
    w = np.ones_like(p) if w is None else w / w.mean()
    pi = pi_v
    for _ in range(it):
        q = correct(p, pi_v, pi)
        new = float(np.average(q, weights=w))
        if abs(new - pi) < 1e-7:
            break
        pi = new
    return pi


rows = []
for f in FAMS:
    z = np.load(f"results/e10_refit/full_{f}.npz")
    g = z["groups"].astype(str); y = (g != "normal").astype(int)
    pi_v = float(z["y_va"].astype(int).mean())
    w = np.array([MIX[x] / (g == x).sum() for x in g])
    for m in MODELS:
        p = np.clip(z[f"{m}|p"].astype(float), 1e-6, 1 - 1e-6)
        pc = correct(p, pi_v, PI_D)
        pi_hat = em_prior(p, pi_v, w)
        pe = correct(p, pi_v, pi_hat)
        r = dict(family=f, model=m, pi_v=pi_v, pi_hat=pi_hat)
        for lab, q in (("unc", p), ("cor", pc), ("em", pe)):
            a, b = intercept_slope(q, y, w)
            r.update({f"ece_{lab}": wece(q, y, w), f"int_{lab}": a, f"slope_{lab}": b})
            for cl in ("normal", "known", "zeroday"):
                r[f"mean_{cl}_{lab}"] = float(q[g == cl].mean())
        rows.append(r)
P = pd.DataFrame(rows)
S = P.groupby("model").mean(numeric_only=True)
S.to_csv("tables/s_table_calibration_extra.csv")
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
print(S[["pi_v", "pi_hat", "ece_unc", "ece_cor", "ece_em", "int_unc", "slope_unc", "int_cor", "slope_cor",
         "mean_normal_unc", "mean_normal_cor", "mean_known_unc", "mean_known_cor", "mean_zeroday_unc", "mean_zeroday_cor"]].round(3).to_string())
print("EM prior estimate range over folds and models:", round(P.pi_hat.min(), 4), round(P.pi_hat.max(), 4))
