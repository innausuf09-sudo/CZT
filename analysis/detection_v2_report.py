"""Tables for the revision detection results (E10 refit, no-TTL, CLAD; E13 NetFlow v3 temporal; E14 calibration).

Run from code/:  python -m analysis.detection_v2_report
Writes the manuscript tables T_new.md, T_dep.md, T_dec.md (folder in zdt/paths.py) and tables/s_table_e10_*.csv, s_table_e13_*.csv
"""
import json
import os

import numpy as np
import pandas as pd
from zdt.paths import MAN_TABLES, MAN_FIGS

MAN = MAN_TABLES
ORDER = ["XGBoost", "LightGBM", "DNN", "ResSE-CNN1D", "B-ProtoMeta", "CZT-Hybrid", "CLAD"]

e10 = json.load(open("results/e10_refit.json"))
rows = []
for r in e10["folds"]:
    for m, v in r["detectors"].items():
        rows.append(dict(variant=r["variant"], family=r["family"], model=m, **v))
R = pd.DataFrame(rows)
R.to_csv("tables/s_table_e10_refit_folds.csv", index=False)
P2 = R.groupby(["variant", "model"])[["auroc_zd", "zdr", "tpr_fpr1_zd", "test_fpr"]].mean().unstack(0)
nov = pd.DataFrame([dict(variant=r["variant"], family=r["family"], score=k, auroc_all=v["all"], auroc_alerts=v["alerts"])
                    for r in e10["folds"] for k, v in r["novelty_auroc"].items()])
nov.to_csv("tables/s_table_e10_novelty_auroc.csv", index=False)
gen = R[R.family == "Generic"].pivot(index="model", columns="variant", values="zdr")

T13 = None
if os.path.exists("results/e13_nfv3_temporal.json"):
    e13 = json.load(open("results/e13_nfv3_temporal.json"))
    rows = []
    for r in e13["folds"]:
        for m, v in r["det"].items():
            rows.append(dict(variant=r["variant"], holdout=r["holdout"] or "none", model=m, **v))
    T13 = pd.DataFrame(rows)
    T13.to_csv("tables/s_table_e13_nfv3_folds.csv", index=False)

L = ["| Detector | P2 AUROC | P2 ZDR | P2 AUROC, no TTL | P2 ZDR, no TTL | P5 known AUROC | P5 zero-day AUROC | P5 ZDR | P5 test FPR | P5 zero-day AUROC, no TTL |",
     "|---|---|---|---|---|---|---|---|---|---|"]
for m in ORDER:
    a = P2.loc[m]
    cells = [f"{a[('auroc_zd', 'full')]:.3f}", f"{a[('zdr', 'full')]:.3f}", f"{a[('auroc_zd', 'nottl')]:.3f}", f"{a[('zdr', 'nottl')]:.3f}"]
    if T13 is not None:
        k = T13[(T13.model == m) & (T13.variant == "full") & (T13.holdout == "none")]
        z = T13[(T13.model == m) & (T13.variant == "full") & (T13.holdout != "none")]
        zn = T13[(T13.model == m) & (T13.variant == "nottl") & (T13.holdout != "none")]
        cells += [f"{k.auroc_known.mean():.3f}", f"{z.auroc_zd.mean():.3f}", f"{z.zdr.mean():.3f}", f"{z.test_fpr.mean():.3f}",
                  f"{zn.auroc_zd.mean():.3f}"]
    else:
        cells += ["n/a"] * 5
    L.append(f"| {m} | " + " | ".join(cells) + " |")
open(MAN + "T_new.md", "w").write("\n".join(L) + "\n")
print("\n".join(L))
print("Generic ZDR full vs no-TTL:\n", gen.round(3))
print(nov.groupby(["variant", "score"])[["auroc_all", "auroc_alerts"]].mean().round(3))
if T13 is not None:
    print(T13[T13.variant == "full"].pivot_table(index="holdout", columns="model", values="auroc_zd").round(3))
    print(T13[T13.variant == "full"].pivot_table(index="holdout", columns="model", values="zdr").round(3))
    print(T13[T13.variant == "full"].pivot_table(index="holdout", columns="model", values="test_fpr").round(3))

# deployment calibration table
C = pd.read_csv("tables/s_table_calibration_prior.csv")
L = ["| Detector | ECE uncorrected | ECE corrected | ECE recalibrated | Log-loss uncorrected | Log-loss corrected | Mean probability uncorrected | ECE independent check |",
     "|---|---|---|---|---|---|---|---|"]
for m in ORDER:
    g = C[C.model == m].set_index("scores")
    L.append(f"| {m} | {g.loc['uncorrected', 'ece_deploy']:.4f} | {g.loc['prior-corrected', 'ece_deploy']:.4f} | "
             f"{g.loc['recalibrated', 'ece_deploy']:.4f} | {g.loc['uncorrected', 'logloss_deploy']:.3f} | "
             f"{g.loc['prior-corrected', 'logloss_deploy']:.3f} | {g.loc['uncorrected', 'mean_p_deploy']:.3f} | "
             f"{g.loc['uncorrected', 'ece_indep_valmix']:.3f} |")
open(MAN + "T_dep.md", "w").write("\n".join(L) + "\n")
print("\n".join(L))
DS = pd.read_csv("tables/s_table_calibration_decision.csv")
lab = {"uncorrected": "uncorrected", "prior-corrected": "corrected, true prior 0.03", "recalibrated": "recalibrated",
       "corrected, assumed prior 0.003": "corrected, assumed 0.003", "corrected, assumed prior 0.10": "corrected, assumed 0.10"}
L = ["| Scores | " + " | ".join(f"{m} cost ratio {c}" for m in ("CZT-Hybrid", "XGBoost") for c in (10, 30, 100)) + " |",
     "|" + "---|" * 7]
for k, v in lab.items():
    cells = []
    for m in ("CZT-Hybrid", "XGBoost"):
        for c in (10, 30, 100):
            r = DS[(DS.model == m) & (DS.scores == k) & (DS.cost_ratio == c)].iloc[0]
            cells.append(f"{r.cost_per_1000:.0f} ({100 * r.escalated_share:.0f}%)")
    L.append(f"| {v} | " + " | ".join(cells) + " |")
open(MAN + "T_dec.md", "w").write("\n".join(L) + "\n")
print("\n".join(L))
