"""Paper figures 8 to 10 (compact two-column layout)."""
import json, pickle
from zdt.io import load_scores
import numpy as np
import matplotlib.pyplot as plt
from zdt.style import setup, panel, MODEL_COLOR
from zdt.metrics import reliability, ece
setup()
R, F = "results/", "figures/"

def calib_row():
    store = load_scores(R + "e2_scores.npz")
    y = np.concatenate([st["y"] for st in store.values()])
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.15), gridspec_kw={"wspace": 0.38})
    for i, (ax, m) in enumerate(zip(axes, ["XGBoost", "ResSE-CNN1D", "B-ProtoMeta", "CZT-Hybrid"])):
        for key, col, lab, mk in [("pu", "#999999", "raw", "o"), ("p", MODEL_COLOR[m], "Platt", "s")]:
            p = np.concatenate([st["scores"][m][key] for st in store.values()])
            mp, fo, _ = reliability(p, y, 10)
            ax.plot(mp, fo, marker=mk, ms=3, color=col, lw=1.2, label=f"{lab}, ECE {ece(p, y):.3f}")
        ax.plot([0, 1], [0, 1], ls=":", color="#666", lw=0.8)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
        ax.set_title(f"({'abcd'[i]}) {m}", fontsize=7.8, loc="left", fontweight="bold"); ax.legend(fontsize=5.5, loc="lower right", handlelength=1.2)
        ax.set_xlabel("Predicted P(attack)", fontsize=7)
        if i == 0: ax.set_ylabel("Observed frequency", fontsize=7)
    fig.savefig(F + "fig08_calibration.png"); plt.close(fig)

def triage_row():
    T = json.load(open(R + "e7_triage.json"))
    pols = ["FIFO", "Severity", "EDF", "Risk", "CZT-Whittle", "CZT-Whittle-T", "CZT-Whittle-NSGA"]
    DN = {"CZT-Whittle": "CZT-D", "CZT-Whittle-T": "CZT-T", "CZT-Whittle-NSGA": "CZT-TE"}
    pcol = {"FIFO": "#999999", "Severity": "#0072B2", "EDF": "#56B4E9", "Risk": "#009E73", "CZT-Whittle": "#E69F00",
            "CZT-Whittle-T": "#CC79A7", "CZT-Whittle-NSGA": "#D55E00"}
    pmk = {"FIFO": "o", "Severity": "s", "EDF": "v", "Risk": "^", "CZT-Whittle": "D", "CZT-Whittle-T": "X", "CZT-Whittle-NSGA": "P"}
    rhos = sorted(T["load"], key=float); x = np.array(rhos, float)
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.35), gridspec_kw={"wspace": 0.42})
    for ax, (key, lab, l) in zip(axes, [("cv", "Captured value CV", "a"), ("zcv", "Zero-day value ZCV", "b"),
                                        ("wait_zd", "Zero-day wait (min)", "c"), ("sla_viol", "SLA violation rate", "d")]):
        for p in pols:
            mu = np.array([T["load"][r][p][key][0] for r in rhos]); ci = np.array([T["load"][r][p][key][1] for r in rhos])
            ax.plot(x, mu, marker=pmk[p], ms=3, color=pcol[p], lw=1.9 if p in DN else 1.1, label=DN.get(p, p))
            ax.fill_between(x, mu - ci, mu + ci, color=pcol[p], alpha=0.12, lw=0)
        ax.set_xlabel("Offered load ρ", fontsize=7); ax.set_title(f"({l}) {lab}", fontsize=7.4, loc="left", fontweight="bold")
        ax.tick_params(labelsize=6.5)
    h, lb = axes[0].get_legend_handles_labels()
    fig.legend(h, lb, loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=7, fontsize=6.6)
    fig.savefig(F + "fig09_triage_load.png"); plt.close(fig)



def detector_link():
    import pandas as pd
    from scipy import stats
    from zdt.style import FAMILY_OF, MARKERS
    T = json.load(open(R + "e7_triage.json")); D = T["detectors"]
    e2 = pd.DataFrame(json.load(open(R + "e2_fh.json"))).groupby("model")["auroc_zd"].mean()
    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    pts = {m: (e2[m], D[m]["CZT-Whittle-NSGA"]["zcv"]) for m in D}
    for m, (x_, (y_, ci)) in pts.items():
        ax.errorbar(x_, y_, yerr=ci, fmt=MARKERS[FAMILY_OF[m]], color=MODEL_COLOR[m], ms=4.5, lw=0.8, capsize=1.5)
    dense = sorted([m for m in pts if pts[m][0] > 0.94], key=lambda m: -pts[m][1][0])
    for k, m in enumerate(dense):
        ly = 0.645 - k * 0.034
        ax.annotate(m, (pts[m][0], pts[m][1][0]), xytext=(0.917, ly), textcoords="data", fontsize=5.8, va="center",
                    ha="left", arrowprops=dict(arrowstyle="-", lw=0.4, color="#888", shrinkA=0, shrinkB=3))
    for m in [m for m in pts if m not in dense]:
        ax.annotate(m, (pts[m][0], pts[m][1][0]), xytext=(5, 0), textcoords="offset points", fontsize=5.8, va="center")
    r = stats.spearmanr([e2[m] for m in D], [D[m]["CZT-Whittle-NSGA"]["zcv"][0] for m in D])
    ax.text(0.97, 0.06, f"Spearman r = {r.correlation:.2f}", transform=ax.transAxes, fontsize=7, va="bottom", ha="right")
    ax.set_xlabel("Unseen-family AUROC (P2)", fontsize=7.5); ax.set_ylabel("Zero-day value ZCV (CZT-TE)", fontsize=7.5)
    ax.tick_params(labelsize=6.5)
    fig.savefig(F + "fig10_detector_triage_link.png"); plt.close(fig)

