"""Calibration under the deployment prior (revision R1.1, R1.5 independent check, R2.3).

Inputs: results/e10_refit/full_<family>.npz (refitted detectors on the original P2 splits).
For each fold:
- validation attack prior pi_v (share of attack records in the validation split);
- calibration on the simulated deployment mixture: test records re-weighted to
  97% benign, 2% known-family attacks, 1% unseen-family attacks (weights mix_g / n_g);
- three score versions:
    uncorrected       Platt-scaled probability fitted on the validation split
    prior-corrected   odds multiplied by [pi_d / (1 - pi_d)] / [pi_v / (1 - pi_v)], pi_d = 0.03
    recalibrated      Platt refit on the validation split re-weighted to the deployment
                      benign/attack ratio (97/3; validation holds known families only, so
                      no zero-day label is used)
- independent check (R1.5): calibration on test benign and known-family records at the
  validation mixture (records never used for fitting, early stopping or calibration).
Metrics: expected calibration error (15 equal-width bins, weighted), Brier score.
Run from code/:  python -m analysis.calibration_prior
"""
import json

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression

FAMS = ["Generic", "Exploits", "Fuzzers", "DoS", "Reconnaissance", "Analysis", "Backdoor", "Shellcode", "Worms"]
MODELS = ["XGBoost", "LightGBM", "DNN", "ResSE-CNN1D", "B-ProtoMeta", "CZT-Hybrid", "CLAD"]
MIX = {"normal": 0.97, "known": 0.02, "zeroday": 0.01}
PI_D = 0.03


def wece(p, y, w, bins=15):
    w = w / w.sum()
    e = np.linspace(0, 1, bins + 1)
    k = np.clip(np.digitize(p, e[1:-1]), 0, bins - 1)
    out = 0.0
    for b in range(bins):
        m = k == b
        if w[m].sum() > 0:
            out += w[m].sum() * abs(np.average(p[m], weights=w[m]) - np.average(y[m], weights=w[m]))
    return float(out)


def reliability(p, y, w, bins=10):
    e = np.linspace(0, 1, bins + 1)
    k = np.clip(np.digitize(p, e[1:-1]), 0, bins - 1)
    return [(float(np.average(p[k == b], weights=w[k == b])), float(np.average(y[k == b], weights=w[k == b])),
             float(w[k == b].sum() / w.sum())) for b in range(bins) if w[k == b].sum() > 0]


rows, priors, rel, dec = [], {}, {}, []
COST_RATIOS = (10, 30, 100)  # cost of a missed attack relative to one false escalation
ASSUMED = (0.003, 0.03, 0.10)  # assumed deployment prior used in the correction (true prior 0.03)


def logloss(q, y, w):
    q = np.clip(q, 1e-6, 1 - 1e-6)
    return float(-np.average(y * np.log(q) + (1 - y) * np.log(1 - q), weights=w))


def correct(p, pi_v, pi_a):
    o = p / (1 - p) * (pi_a / (1 - pi_a)) / (pi_v / (1 - pi_v))
    return o / (1 + o)
for f in FAMS:
    z = np.load(f"results/e10_refit/full_{f}.npz")
    g = z["groups"].astype(str)
    y = (g != "normal").astype(int)
    yv = z["y_va"].astype(int)
    pi_v = float(yv.mean())
    priors[f] = dict(val_attack_prior=pi_v, n_val=int(len(yv)), val_attack=int(yv.sum()),
                     test_attack_prior_benign_unseen=float((g == "zeroday").sum() / ((g == "zeroday").sum() + (g == "normal").sum())))
    w_dep = np.array([MIX[x] / (g == x).sum() for x in g])
    kn = g != "zeroday"
    w_val = np.where(g[kn] == "normal", (1 - pi_v) / (g[kn] == "normal").sum(), pi_v / (g[kn] == "known").sum())
    for m in MODELS:
        p = np.clip(z[f"{m}|p"].astype(float), 1e-6, 1 - 1e-6)
        pv = np.clip(z[f"{m}|p_va"].astype(float), 1e-6, 1 - 1e-6)
        odds = p / (1 - p) * (PI_D / (1 - PI_D)) / (pi_v / (1 - pi_v))
        p_prior = odds / (1 + odds)
        sw = np.where(yv == 1, PI_D / yv.sum(), (1 - PI_D) / (1 - yv).sum())
        lr = LogisticRegression(C=1e6, max_iter=2000).fit(logit(pv).reshape(-1, 1), yv, sample_weight=sw * len(yv))
        p_rec = expit(lr.coef_[0, 0] * logit(p) + lr.intercept_[0])
        for lab, q in (("uncorrected", p), ("prior-corrected", p_prior), ("recalibrated", p_rec),
                       ("corrected, assumed prior 0.003", correct(p, pi_v, 0.003)), ("corrected, assumed prior 0.10", correct(p, pi_v, 0.10))):
            # Bayes decision: escalate when q >= 1 / (1 + cost ratio); expected cost per event on the deployment mixture
            for cr in COST_RATIOS:
                esc = q >= 1.0 / (1.0 + cr)
                cost = np.average(np.where(y == 1, (~esc) * cr, esc * 1.0), weights=w_dep)
                dec.append(dict(family=f, model=m, scores=lab, cost_ratio=cr, cost_per_1000=float(1000 * cost),
                                escalated_share=float(np.average(esc, weights=w_dep))))
            rows.append(dict(family=f, model=m, scores=lab, ece_deploy=wece(q, y, w_dep), logloss_deploy=logloss(q, y, w_dep),
                             mean_p_benign=float(np.average(q[y == 0], weights=w_dep[y == 0])),
                             missing_p_attack=float(1 - np.average(q[y == 1], weights=w_dep[y == 1])),
                             brier_deploy=float(np.average((q - y) ** 2, weights=w_dep)),
                             mean_p_deploy=float(np.average(q, weights=w_dep)),
                             ece_indep_valmix=wece(q[kn], y[kn], w_val) if lab == "uncorrected" else np.nan))
            if m == "CZT-Hybrid":
                rel.setdefault(lab, []).append((reliability(q, y, w_dep)))
T = pd.DataFrame(rows)
T.to_csv("tables/s_table_calibration_prior_folds.csv", index=False)
S = T.groupby(["model", "scores"])[["ece_deploy", "logloss_deploy", "brier_deploy", "mean_p_deploy", "mean_p_benign",
                                    "missing_p_attack", "ece_indep_valmix"]].mean().reset_index()
DC = pd.DataFrame(dec)
# oracle: best threshold on the test mixture itself (lower bound of achievable cost for the given ranking)
DC.to_csv("tables/s_table_calibration_decision_folds.csv", index=False)
DS = DC.groupby(["model", "scores", "cost_ratio"])[["cost_per_1000", "escalated_share"]].mean().reset_index()
DS.to_csv("tables/s_table_calibration_decision.csv", index=False)
S.to_csv("tables/s_table_calibration_prior.csv", index=False)
with open("tables/s_table_calibration_prior.md", "w") as fh:
    fh.write(S.to_markdown(index=False, floatfmt=".4f"))
json.dump(dict(priors=priors, deployment_attack_prior=PI_D, mix=MIX, reliability_hybrid=rel),
          open("results/e14_calibration_prior.json", "w"), indent=1)
print(pd.DataFrame(priors).T)
print(S.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
print(DS[DS.model == "CZT-Hybrid"].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
