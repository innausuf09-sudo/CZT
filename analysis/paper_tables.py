"""Compact Tables 4 to 10 of the paper, generated from results/."""
import json, os
import numpy as np
import pandas as pd
from zdt.models import MODEL_ORDER
OUT = "tables/"
os.makedirs(OUT, exist_ok=True)
pm = lambda m, s, d=3: f"{m:.{d}f} ± {s:.{d}f}"
def md(df, name):
    L = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    L += ["| " + " | ".join(str(v) for v in r) + " |" for r in df.values]
    open(OUT + name + ".md", "w").write("\n".join(L) + "\n"); df.to_csv(OUT + name + ".csv", index=False)
# T4 P1
e1 = pd.DataFrame(json.load(open("results/e1_iid.json")))
g, s = e1.groupby("model").mean(numeric_only=True).reindex(MODEL_ORDER), e1.groupby("model").std(numeric_only=True).reindex(MODEL_ORDER)
md(pd.DataFrame({"Detector": MODEL_ORDER, "F1": [pm(g.loc[m, "f1"], s.loc[m, "f1"]) for m in MODEL_ORDER],
                 "AUROC": [pm(g.loc[m, "auroc"], s.loc[m, "auroc"]) for m in MODEL_ORDER],
                 "AUPRC": [pm(g.loc[m, "auprc"], s.loc[m, "auprc"]) for m in MODEL_ORDER],
                 "TPR at 1% FPR": [pm(g.loc[m, "tpr_at_fpr1"], s.loc[m, "tpr_at_fpr1"]) for m in MODEL_ORDER],
                 "ECE": [pm(g.loc[m, "ece"], s.loc[m, "ece"]) for m in MODEL_ORDER]}), "T4_p1")
# T5 P2
e2 = pd.DataFrame(json.load(open("results/e2_fh.json")))
g, s = e2.groupby("model").mean(numeric_only=True).reindex(MODEL_ORDER), e2.groupby("model").std(numeric_only=True).reindex(MODEL_ORDER)
mn, med = e2.groupby("model").min(numeric_only=True).reindex(MODEL_ORDER), e2.groupby("model").median(numeric_only=True).reindex(MODEL_ORDER)
rk = e2.pivot(index="family", columns="model", values="auroc_zd")[MODEL_ORDER].rank(axis=1, ascending=False).mean()
md(pd.DataFrame({"Detector": MODEL_ORDER, "ZDR": [pm(g.loc[m, "zdr"], s.loc[m, "zdr"]) for m in MODEL_ORDER],
                 "Median ZDR": [f"{med.loc[m, 'zdr']:.3f}" for m in MODEL_ORDER],
                 "Known recall": [f"{g.loc[m, 'known_recall']:.3f}" for m in MODEL_ORDER],
                 "Test FPR": [f"{g.loc[m, 'test_fpr']:.3f}" for m in MODEL_ORDER],
                 "AUROC": [pm(g.loc[m, "auroc_zd"], s.loc[m, "auroc_zd"]) for m in MODEL_ORDER],
                 "Worst AUROC": [f"{mn.loc[m, 'auroc_zd']:.3f}" for m in MODEL_ORDER],
                 "TPR at 1% FPR": [f"{g.loc[m, 'tpr_fpr1_zd']:.3f}" for m in MODEL_ORDER],
                 "Mean rank": [f"{rk[m]:.2f}" for m in MODEL_ORDER]}), "T5_p2")
# T6 P3
e3 = pd.DataFrame(json.load(open("results/e3_kdd_novel.json")))
g, s = e3.groupby("model").mean(numeric_only=True).reindex(MODEL_ORDER), e3.groupby("model").std(numeric_only=True).reindex(MODEL_ORDER)
md(pd.DataFrame({"Detector": MODEL_ORDER, "AUROC (all)": [pm(g.loc[m, "auroc"], s.loc[m, "auroc"]) for m in MODEL_ORDER],
                 "F1 at 0.5": [f"{g.loc[m, 'f1']:.3f}" for m in MODEL_ORDER],
                 "Test FPR": [f"{g.loc[m, 'fpr_thr']:.3f}" for m in MODEL_ORDER],
                 "DR seen types": [f"{g.loc[m, 'dr_known_thr']:.3f}" for m in MODEL_ORDER],
                 "DR unseen types": [pm(g.loc[m, "dr_novel_thr"], s.loc[m, "dr_novel_thr"]) for m in MODEL_ORDER],
                 "AUROC unseen": [f"{g.loc[m, 'auroc_novel']:.3f}" for m in MODEL_ORDER]}), "T6_p3")
# T7 P4
e4 = pd.DataFrame(json.load(open("results/e4_transfer.json")))
M4 = ["LR", "RF", "XGBoost", "LightGBM", "DNN", "ResSE-CNN1D", "B-ProtoMeta", "CZT-Hybrid"]
rows = []
for m in M4:
    r = [m]
    for src in ["UNSW-NB15", "NSL-KDD"]:
        for k in [0, 100, 1000]:
            r.append(f"{e4[(e4.src == src) & (e4.model == m) & (e4.k == k)].auroc.mean():.3f}")
    rows.append(r)
md(pd.DataFrame(rows, columns=["Detector", "U→K k=0", "U→K k=100", "U→K k=1000", "K→U k=0", "K→U k=100", "K→U k=1000"]), "T7_p4")
# T8 calibration + cost
lat = pd.DataFrame(json.load(open("results/e6_latency.json"))).set_index("model").reindex(MODEL_ORDER)
c2 = e2.groupby("model")[["ece_uncal", "ece_cal"]].mean().reindex(MODEL_ORDER)
c1 = e1.groupby("model")[["ece"]].mean().reindex(MODEL_ORDER)
md(pd.DataFrame({"Detector": MODEL_ORDER, "ECE P1": [f"{v:.3f}" for v in c1.ece], "ECE P2 raw": [f"{v:.3f}" for v in c2.ece_uncal],
                 "ECE P2 Platt": [f"{v:.3f}" for v in c2.ece_cal],
                 "Train (s)": [f"{v:.1f}" for v in e2.groupby('model').train_time.mean().reindex(MODEL_ORDER)],
                 "Batch (µs/record)": [f"{v:.1f}" for v in lat.batch_us_per_sample],
                 "Single p50 (ms)": [f"{v:.2f}" for v in lat.single_ms_p50],
                 "Size (kB)": [f"{v:,.0f}" for v in lat.size_kb]}), "T8_calib_cost")
# T9 triage
T = json.load(open("results/e7_triage.json"))
DN = {"CZT-Whittle": "CZT-D", "CZT-Whittle-T": "CZT-T", "CZT-Whittle-NSGA": "CZT-TE"}
rows = []
for rho in ["1.2", "1.5"]:
    for p, v in T["load"][rho].items():
        rows.append([rho, DN.get(p, p), f"{v['n_alerts'][0]:.0f}", pm(*v["cv"]), pm(*v["zcv"]), f"{v['wait_zd'][0]:.1f} ± {v['wait_zd'][1]:.1f}",
                     pm(*v["sla_viol"]), f"{v['fp_time_frac'][0]:.3f}"])
md(pd.DataFrame(rows, columns=["ρ", "Policy", "Alerts", "CV", "ZCV", "Zero-day wait (min)", "SLA violation", "FP time share"]), "T9_triage")
# T10 ablation
A = T["ablation"]
REN = [("Risk baseline (p x catalogue severity)", "Risk baseline"), ("CZT-Whittle, default weights", "CZT-D (default weights)"),
       ("  default, without calibration", "  D without calibration"), ("  default, without evidence decay", "  D without decay"),
       ("  default, without deadline term", "  D without deadline term"), ("  default, without detector score (gamma = 0)", "  D without score (γ = 0)"),
       ("CZT-Whittle-NSGA (tuned)", "CZT-TE (tuned)"), ("  tuned, without calibration", "  TE without calibration"),
       ("  tuned, escalation fixed at q = 0.99", "  TE with q = 0.99 (= CZT-T)"), ("  tuned, imputed severity s_u = 0.5", "  TE with s_u = 0.5"),
       ("  tuned, without evidence decay", "  TE without decay"), ("  tuned, without detector score (gamma = 0)", "  TE without score (γ = 0)")]
md(pd.DataFrame([[n.replace("  ", "↳ "), f"{A[k]['zcv'][0]:.3f}", f"{A[k]['wait_zd'][0]:.1f}", f"{A[k]['sla_viol'][0]:.3f}"] for k, n in REN],
                columns=["Configuration", "ZCV", "Wait (min)", "SLA viol."]), "T10_ablation")
print("ok")
