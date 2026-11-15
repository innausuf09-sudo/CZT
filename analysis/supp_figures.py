"""Supplementary figures (styles follow the original project figures: violin,
radar, training curves, significance matrix) and supplementary tables."""
import json, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from zdt.style import setup, panel, MODEL_COLOR, FAMILY_OF, FAM_COLOR
from zdt.models import MODEL_ORDER
from analysis import analyze as A

setup()
R, F, T = "results/", "figures/", "tables/"
os.makedirs(F, exist_ok=True); os.makedirs(T, exist_ok=True)
e2 = A.e2_load()


def save(fig, n):
    fig.savefig(F + n + ".png"); plt.close(fig)


# S1: violin + strip of per-family ZDR and AUROC
fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.6), sharex=True, gridspec_kw={"hspace": 0.12})
for ax, key, lab, l in [(axes[0], "zdr", "ZDR at frozen 1% FPR", "a"), (axes[1], "auroc_zd", "AUROC (unseen family)", "b")]:
    data = [e2[e2.model == m][key].values for m in MODEL_ORDER]
    vp = ax.violinplot(data, showextrema=False, widths=0.8)
    for b, m in zip(vp["bodies"], MODEL_ORDER):
        b.set_facecolor(MODEL_COLOR[m]); b.set_alpha(0.35); b.set_edgecolor("none")
    rng = np.random.default_rng(0)
    for i, (d, m) in enumerate(zip(data, MODEL_ORDER)):
        ax.scatter(i + 1 + rng.uniform(-0.12, 0.12, len(d)), d, s=9, color=MODEL_COLOR[m], edgecolor="white", lw=0.4, zorder=3)
        ax.plot([i + 0.75, i + 1.25], [np.median(d)] * 2, color="#222", lw=1.4, zorder=4)
    ax.set_ylabel(lab); panel(ax, l, -0.09)
axes[1].set_xticks(range(1, len(MODEL_ORDER) + 1)); axes[1].set_xticklabels(MODEL_ORDER, rotation=35, ha="right")
save(fig, "s08_violin_per_family")

# S2: radar of normalised profile for six detectors
mets = {"AUROC P1": ("e1", "auroc"), "TPR@1% P1": ("e1", "tpr_at_fpr1"), "AUROC P2": ("e2", "auroc_zd"),
        "ZDR P2": ("e2", "zdr"), "Worst-family AUROC": ("e2min", "auroc_zd"), "1 - ECE P2": ("e2ece", "ece_cal"),
        "Speed (-log latency)": ("lat", "batch_us_per_sample")}
e1 = pd.DataFrame(json.load(open(R + "e1_iid.json"))).groupby("model").mean(numeric_only=True)
lat = pd.DataFrame(json.load(open(R + "e6_latency.json"))).set_index("model")
vals = {}
for name, (src, col) in mets.items():
    if src == "e1": v = e1[col]
    elif src == "e2": v = e2.groupby("model")[col].mean()
    elif src == "e2min": v = e2.groupby("model")[col].min()
    elif src == "e2ece": v = 1 - e2.groupby("model")[col].mean()
    else: v = -np.log10(lat[col])
    v = v.reindex(MODEL_ORDER)
    vals[name] = (v - v.min()) / (v.max() - v.min())
V = pd.DataFrame(vals)
show = ["LR", "XGBoost", "LightGBM", "DNN", "ResSE-CNN1D", "B-ProtoMeta", "CZT-Hybrid"]
ang = np.linspace(0, 2 * np.pi, len(mets), endpoint=False).tolist(); ang += ang[:1]
fig = plt.figure(figsize=(5.6, 5.0)); ax = fig.add_subplot(111, polar=True)
for m in show:
    r = V.loc[m].tolist(); r += r[:1]
    ax.plot(ang, r, color=MODEL_COLOR[m], lw=2.2 if m == "CZT-Hybrid" else 1.2, label=m)
    ax.fill(ang, r, color=MODEL_COLOR[m], alpha=0.06)
ax.set_xticks(ang[:-1]); ax.set_xticklabels(list(mets), fontsize=7); ax.set_yticklabels([]); ax.set_ylim(0, 1.05)
ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.12), fontsize=7)
save(fig, "s09_radar_profile")

# S3: training curves (fold 0 of P1)
H = json.load(open(R + "e1_histories.json"))
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), gridspec_kw={"wspace": 0.3})
for m in ["DNN", "ResSE-CNN1D", "LSTM", "GRU", "FT-Transformer", "FG-GNN"]:
    h = np.array(H[m]); axes[0].plot(h[:, 0], h[:, 1], color=MODEL_COLOR[m], lw=1.2, ls="--")
    axes[0].plot(h[:, 0], h[:, 2], color=MODEL_COLOR[m], lw=1.6, label=m)
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Weighted BCE"); axes[0].set_title("solid: validation, dashed: training", fontsize=7.5, loc="left")
axes[0].legend(fontsize=6); panel(axes[0], "a", -0.2)
h = np.array(H["B-ProtoMeta"]); axes[1].plot(h[:, 0], h[:, 1], color=MODEL_COLOR["B-ProtoMeta"], lw=1.4)
axes[1].set_xlabel("Episode"); axes[1].set_ylabel("Episodic prototype loss"); panel(axes[1], "b", -0.2)
save(fig, "s10_training_curves")

# S4: pairwise Wilcoxon significance matrix on P2 AUROC
P = e2.pivot(index="family", columns="model", values="auroc_zd")[MODEL_ORDER]
k = len(MODEL_ORDER); M = np.full((k, k), np.nan); D = np.zeros((k, k))
for i, a in enumerate(MODEL_ORDER):
    for j, b in enumerate(MODEL_ORDER):
        if i != j:
            D[i, j] = np.median(P[a] - P[b])
            M[i, j] = stats.wilcoxon(P[a], P[b]).pvalue
fig, ax = plt.subplots(figsize=(6.4, 5.4))
lim = np.nanmax(np.abs(D))
im = ax.imshow(D, cmap="RdBu", vmin=-lim, vmax=lim)
for i in range(k):
    for j in range(k):
        if i != j and M[i, j] < 0.05:
            ax.text(j, i, "*" if M[i, j] >= 0.01 else "**", ha="center", va="center", fontsize=8)
ax.set_xticks(range(k)); ax.set_xticklabels(MODEL_ORDER, rotation=60, ha="right", fontsize=7)
ax.set_yticks(range(k)); ax.set_yticklabels(MODEL_ORDER, fontsize=7); ax.grid(False)
cb = fig.colorbar(im, fraction=0.045); cb.set_label("Median AUROC difference (row - column)", fontsize=7)
save(fig, "s11_pairwise_significance")

# Supplementary tables: per-family full metrics, dedup effect, all triage sweeps
cols = ["model", "family", "n_zeroday", "thr", "zdr", "known_recall", "test_fpr", "auroc_zd", "auprc_zd", "tpr_fpr1_zd",
        "auroc_known", "ece_uncal", "ece_cal", "train_time"]
e2[cols].to_csv(T + "s_table_p2_per_family.csv", index=False)
d = pd.DataFrame(json.load(open(R + "e1b_dedup_effect.json")))
g = d.groupby(["dedup", "model"])[["accuracy", "f1", "auroc", "auprc", "tpr_at_fpr1", "test_overlap"]].agg(["mean", "std"])
g.to_csv(T + "s_table_dedup_effect.csv")
print(g.round(3))
tri = json.load(open(R + "e7_triage.json"))
rows = []
for sweep in ["load", "staff", "halflife"]:
    for x, pols in tri[sweep].items():
        for p, v in pols.items():
            rows.append({"sweep": sweep, "value": x, "policy": p, **{k_: v[k_][0] for k_ in v}, **{k_ + "_ci": v[k_][1] for k_ in v}})
pd.DataFrame(rows).to_csv(T + "s_table_triage_sweeps.csv", index=False)
print("supp ok")
