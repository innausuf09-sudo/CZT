"""Extra supplement tables for the revision (protocol sizes, triage ablations, index against tuned Risk in every setting,
mean outcomes in the anchor settings).

Run from code/:  python -m analysis.supp_tables
Writes tables/s_table_sizes.csv, s_table_e9_ablation_full.csv, s_table_e9_index_vs_tuned.csv, s_table_e9_anchor_means.csv
"""
import json

import numpy as np
import pandas as pd
from scipy import stats

ANCH = {"original": "Original", "central": "Imperfect", "uninformative_0.5": "Uninformative"}
DUP = ("mk1.0_mz0.0_mb0.5", "mk0.8_mz0.2_mb0.5")


def ci(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return x.mean(), h


# ---------------------------------------------------------------- protocol sizes
e10 = json.load(open("results/e10_refit.json"))
e13 = json.load(open("results/e13_nfv3_temporal.json"))
sp1 = np.load("repro/splits_p1.npz")
sp3 = np.load("repro/splits_p3.npz")
rows = [dict(Protocol="P1 (UNSW-NB15, i.i.d.)", Folds="5 (stratified)", Train=len(sp1["fold0|train"]), Validation=len(sp1["fold0|val"]),
             Test=len(sp1["fold0|test"]), Note="24,000-record stratified sample; same sizes in every fold")]
tr = [c["n_train"] for c in e10["split_checks"].values()]
va = [c["n_val"] for c in e10["split_checks"].values()]
te = [c["n_test"] for c in e10["split_checks"].values()]
rows.append(dict(Protocol="P2 (UNSW-NB15, family hold-out)", Folds="9 (one per family)", Train=f"{min(tr):,} to {max(tr):,}",
                 Validation=f"{min(va):,} to {max(va):,}", Test=f"{min(te):,} to {max(te):,}",
                 Note="test: up to 3,000 benign, 3,000 known-family and 3,000 unseen-family records"))
rows.append(dict(Protocol="P3 (NSL-KDD, unseen types)", Folds="3 seeds", Train=len(sp3["seed0|train"]), Validation=len(sp3["seed0|val"]),
                 Test=22544, Note="official KDDTest+ file; 664 test records have an identical feature vector in KDDTrain+"))
rows.append(dict(Protocol="P4 (cross-dataset, 18 features)", Folds="2 directions x 3 seeds", Train="10,000 benign + up to 2,500 per class",
                 Validation="15% of the source sample", Test="10,000", Note="target sample drawn with the same seed"))
f5 = [r for r in e13["folds"] if r["variant"] == "full"]
rows.append(dict(Protocol="P5 (NF-UNSW-NB15-v3, January to February)", Folds=f"{len(f5)} (none held out, and each family)",
                 Train=f"{min(r['n_train'] for r in f5):,} to {max(r['n_train'] for r in f5):,}",
                 Validation=f"{min(r['n_val'] for r in f5):,} to {max(r['n_val'] for r in f5):,}",
                 Test=f"{min(r['n_test'] for r in f5):,} to {max(r['n_test'] for r in f5):,}",
                 Note="training and validation from January, test from February"))
pd.DataFrame(rows).to_csv("tables/s_table_sizes.csv", index=False)

# ---------------------------------------------------------------- triage tables from E9
d = json.load(open("results/e9_triage.json"))
GR = list(d["meta"]["groups"])
ev = d["eval"]


def rep(s, p, f):
    return np.concatenate([np.asarray(ev[f"{s}|{g}"]["reps"][p][f], float) for g in GR])


ABL = [("", "tuned index"), (" no severity (beta=0)", "no severity term"), (" no score (gamma=0)", "no score term"),
       (" no decay", "no evidence decay"), (" no deadline (kappa=0)", "no deadline term"),
       (" no imputation (s_u=0.5)", "imputed severity 0.5"), ("[prior]", "prior-corrected score")]
rows = []
for s, lab in ANCH.items():
    for info in ("flag", "nov"):
        base = f"CZT-T/{info}"
        for suf, nm in ABL:
            p = base + suf
            if p not in ev[f"{s}|{GR[0]}"]["reps"]:
                continue
            r = dict(Setting=lab, Information="flag" if info == "flag" else "novelty score", Variant=nm)
            for f, fl in (("zcv", "ZCV"), ("kcv", "KCV"), ("sla_viol_critical", "Critical SLA misses"), ("wait_zd", "Zero-day wait (min)")):
                x = rep(s, p, f)
                r[fl] = f"{np.nanmean(x):.3f}"
                if suf:
                    m, h = ci(x - rep(s, base, f))
                    r[fl] += f" ({m:+.3f} ± {h:.3f})"
            rows.append(r)
pd.DataFrame(rows).to_csv("tables/s_table_e9_ablation_full.csv", index=False)

D = pd.read_csv("tables/s_table_e9_paired.csv")
rows = []
for info in ("flag", "nov"):
    x = D[(D.policy == f"CZT-T/{info}") & (D.versus == f"Risk-T/{info}") & (D.group == "pooled") & ~D.scenario.isin(DUP)]
    for r in x.itertuples():
        sig = "better" if r.zcv_lo > 0 else ("worse" if r.zcv_hi < 0 else "no detected difference")
        rows.append({"Setting": r.scenario, "Information": "flag" if info == "flag" else "novelty score",
                     "Information AUROC": r.inf_flag if info == "flag" else r.inf_nov,
                     "Index minus tuned Risk, ZCV": f"{r.zcv:+.3f} [{r.zcv_lo:+.3f}, {r.zcv_hi:+.3f}]",
                     "KCV": f"{r.kcv:+.3f} [{r.kcv_lo:+.3f}, {r.kcv_hi:+.3f}]",
                     "Critical SLA": f"{r.sla_viol_critical:+.3f}", "ZCV verdict": sig})
I = pd.DataFrame(rows)
I.to_csv("tables/s_table_e9_index_vs_tuned.csv", index=False)
print(I.groupby("Information")["ZCV verdict"].value_counts())

M = pd.read_csv("tables/s_table_e9_means.csv")
M = M[M.scenario.isin(ANCH)].copy()
M["Setting"] = M.scenario.map(ANCH)
M = M[["Setting", "policy", "zcv", "kcv", "cv", "wait_zd", "sla_viol_critical", "n_alerts"]]
M.columns = ["Setting", "Policy", "ZCV", "KCV", "CV", "Zero-day wait (min)", "Critical SLA misses", "Alerts per shift"]
M.to_csv("tables/s_table_e9_anchor_means.csv", index=False)
print("ok")
