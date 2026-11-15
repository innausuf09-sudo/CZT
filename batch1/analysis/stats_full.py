"""Full statistical output for P2 (revision R1.8, R2.6).

For each metric (unseen-family AUROC, ZDR at the validation threshold, test TPR at 1% FPR):
- per-fold paired values of every detector (tables/s_table_p2_folds_<metric>.csv)
- Wilcoxon signed-rank tests of CZT-Hybrid against each of the other 12 detectors:
  zero differences discarded (zero_method='wilcox'); exact null distribution when there are
  no ties or zeros among the differences, otherwise the normal approximation with tie
  correction (scipy 'auto' rule made explicit); raw p, Holm-adjusted p over the 12
  comparisons of the metric, number of non-zero pairs, W/L counts and the smallest p
  attainable with that number of non-zero pairs (2 / 2^n).
- Friedman test over the 13 detectors and the Nemenyi critical difference.
Run from code/:  python -m analysis.stats_full
"""
import json

import numpy as np
import pandas as pd
from scipy import stats

R = pd.DataFrame(json.load(open("results/e2_fh.json")))
MODELS = ["LR", "RF", "XGBoost", "LightGBM", "SVM", "DNN", "ResSE-CNN1D", "LSTM", "GRU", "FT-Transformer", "FG-GNN",
          "B-ProtoMeta", "CZT-Hybrid"]
METRICS = {"auroc_zd": "AUROC", "zdr": "ZDR (validation threshold)", "tpr_fpr1_zd": "TPR at 1% test FPR"}
Q_NEMENYI_13 = 3.313  # studentised range / sqrt(2), alpha = 0.05, k = 13


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for r, i in enumerate(o):
        run = max(run, (len(p) - r) * p[i])
        adj[i] = min(1.0, run)
    return adj


rows, summary = [], {}
for key, lab in METRICS.items():
    P = R.pivot(index="family", columns="model", values=key)[MODELS]
    P.round(4).to_csv(f"tables/s_table_p2_folds_{key}.csv")
    h = P["CZT-Hybrid"]
    res = []
    for m in MODELS[:-1]:
        d = (h - P[m]).values
        nz = d[d != 0]
        ties = len(np.unique(np.abs(nz))) < len(nz)
        method = "exact" if (len(nz) == len(d) and not ties) else "approx"
        if len(nz) == 0:
            p = 1.0
        else:
            p = stats.wilcoxon(h.values, P[m].values, zero_method="wilcox", method=method).pvalue
        res.append(dict(metric=lab, comparison=f"CZT-Hybrid vs {m}", median_diff=float(np.median(d)),
                        wins=int((d > 0).sum()), losses=int((d < 0).sum()), zeros=int((d == 0).sum()),
                        n_nonzero=len(nz), ties_in_abs_diff=bool(ties), method=method, p_raw=float(p),
                        p_min_attainable=float(2 / 2 ** len(nz)) if len(nz) else 1.0))
    adj = holm([r["p_raw"] for r in res])
    for r, a in zip(res, adj):
        r["p_holm"] = float(a)
    rows += res
    ranks = P.rank(axis=1, ascending=False).mean()
    fr = stats.friedmanchisquare(*[P[m].values for m in MODELS])
    cd = Q_NEMENYI_13 * np.sqrt(len(MODELS) * (len(MODELS) + 1) / (6 * len(P)))
    summary[key] = dict(friedman_chi2=float(fr.statistic), friedman_p=float(fr.pvalue), nemenyi_cd=float(cd),
                        n_folds=len(P), mean_ranks=ranks.round(3).to_dict())
T = pd.DataFrame(rows)
T.to_csv("tables/s_table_wilcoxon_full.csv", index=False)
with open("tables/s_table_wilcoxon_full.md", "w") as f:
    f.write(T.to_markdown(index=False, floatfmt=".4g"))
json.dump(summary, open("tables/s_table_friedman_full.json", "w"), indent=1)
print(T.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "mean_ranks"} for k, v in summary.items()}, indent=1))
