"""Matched-load tuning in all 35 settings (E22): constraint violations, gains over Risk and grid-level comparisons.

Run from code/:  python -m analysis.matched_report
Writes tables/s_table_e22_matched.csv, results/e22_summary.json
"""
import json

import numpy as np
import pandas as pd

E = json.load(open("results/e22_matched.json"))["results"]
E9 = json.load(open("results/e9_triage.json"))["eval"]
GR = ["G1", "G2"]
SC = sorted({r["scenario"] for r in E})
KN = {"CZT": "Index", "Risk": "Tuned Risk", "Linear": "Tuned linear"}
rng = np.random.default_rng(2026)


def get(s, k, info, who, f):
    out = []
    for g in GR:
        m = [r for r in E if r["scenario"] == s and r["group"] == g and r["kind"] == k and r["info"] == info]
        out.append(np.asarray(m[0]["eval"][who][f], float))
    return np.array(out)  # groups x reps


def main9(s, p, f):
    return np.array([np.asarray(E9[f"{s}|{g}"]["reps"][p][f], float) for g in GR])


rows = []
for s in SC:
    for info in ("flag", "nov"):
        risk_z, risk_k = get(s, "CZT", info, "Risk", "zcv"), get(s, "CZT", info, "Risk", "kcv")
        for k in KN:
            z, kc, sla = get(s, k, info, "tuned", "zcv"), get(s, k, info, "tuned", "kcv"), get(s, k, info, "tuned", "sla_viol_critical")
            z0, k0 = main9(s, f"{k}-T/{info}", "zcv"), main9(s, f"{k}-T/{info}", "kcv")
            rows.append(dict(setting=s, info=info, policy=KN[k], dzcv=float((z - risk_z).mean()), dkcv=float((kc - risk_k).mean()),
                             sla=float(np.nanmean(sla)), dzcv_main=float((z0 - risk_z).mean()), dkcv_main=float((k0 - risk_k).mean())))
D = pd.DataFrame(rows)
D.to_csv("tables/s_table_e22_matched.csv", index=False)
pd.set_option("display.width", 220)
S = D.groupby(["info", "policy"]).agg(
    zcv_gain_median=("dzcv", "median"), zcv_gain_main_median=("dzcv_main", "median"),
    kcv_loss_gt002=("dkcv", lambda x: int((x < -0.02).sum())), kcv_loss_gt002_main=("dkcv_main", lambda x: int((x < -0.02).sum())),
    kcv_median=("dkcv", "median"), sla_gt002=("sla", lambda x: int((x > 0.02).sum())), sla_median=("sla", "median"))
print(S.round(3).to_string())


def grid(a, b, info, f, B=4000):
    Dd = np.array([get(s, a, info, "tuned", f) - get(s, b, info, "tuned", f) for s in SC])  # settings x groups x reps
    n = Dd.shape[2]; est = Dd.mean(); bs = np.empty(B)
    for i in range(B):
        idx = rng.integers(n, size=(2, n)); bs[i] = np.mean([Dd[:, g, idx[g]].mean() for g in range(2)])
    lo, hi = np.percentile(bs, [2.5, 97.5]); return est, lo, hi


G = []
for info in ("flag", "nov"):
    for a, b in (("CZT", "Risk"), ("CZT", "Linear"), ("Linear", "Risk")):
        r = dict(comparison=f"{KN[a]} minus {KN[b]}", info=info)
        for f in ("zcv", "kcv", "cv"):
            e, lo, hi = grid(a, b, info, f); r.update({f: e, f + "_lo": lo, f + "_hi": hi})
        G.append(r)
G = pd.DataFrame(G)
print(G.round(3).to_string(index=False))
json.dump(dict(summary=S.reset_index().to_dict("records"), grid=G.to_dict("records")), open("results/e22_summary.json", "w"), indent=1)
