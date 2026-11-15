"""Horizontal workflow figure of the CZT framework (two-column journal width)."""
import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from zdt.style import setup
from zdt.paths import MAN_TABLES, MAN_FIGS

setup()
os.makedirs("figures", exist_ok=True)
fig, ax = plt.subplots(figsize=(7.2, 3.7))
ax.set_xlim(0, 100); ax.set_ylim(0, 52); ax.axis("off")

STAGES = [
    ("1. Data preparation", "#0072B2", "#EAF3FA",
     [("Public data", "UNSW-NB15, NSL-KDD,\nNF-UNSW-NB15-v3"),
      ("Cleaning", "remove duplicates and\nlabel conflicts (Eqs 1-2)"),
      ("Train-only statistics", "no identifiers; log(1+x),\none-hot, train-only scaler")]),
    ("2. Zero-day protocols", "#E69F00", "#FFF5E3",
     [("P1  i.i.d. 5-fold CV", "optimistic reference"),
      ("P2  family hold-out", "9 folds; also without\nTTL-type features"),
      ("P3  unseen types", "17 test-only\nNSL-KDD attack types"),
      ("P4  cross-dataset", "18 shared features,\nk-shot adaptation"),
      ("P5  temporal", "NetFlow v3, January\nto February")]),
    ("3. Detection", "#009E73", "#E7F6F1",
     [("13 detectors + CLAD", "classical, deep, graph,\nmeta-learning, contrastive"),
      ("CZT-Hybrid", "LightGBM, ResSE-CNN1D,\nB-ProtoMeta; log-odds pool"),
      ("Calibration", "Platt scaling, prior\ncorrection, 1% FPR gate")]),
    ("4. SOC triage", "#D55E00", "#FDECE3",
     [("Observables", "score, catalogue match or\nlabel-free novelty score"),
      ("Catalogue model", "match rates for known,\nnovel and benign alerts"),
      ("Policies, equal budget", "index, tuned Risk, linear,\npolicy gradient"),
      ("Outcomes", "zero-day and known value,\nwait time, SLA misses")]),
]
W, GAP, X0 = 22.0, 3.2, 0.6
for s, (title, ec, fc, boxes) in enumerate(STAGES):
    x = X0 + s * (W + GAP)
    ax.add_patch(FancyBboxPatch((x, 1), W, 49.5, boxstyle="round,pad=0.2,rounding_size=1.5", fc="white", ec=ec, lw=1.1))
    ax.add_patch(FancyBboxPatch((x, 45.2), W, 5.3, boxstyle="round,pad=0.2,rounding_size=1.5", fc=ec, ec=ec, lw=0))
    ax.text(x + W / 2, 47.85, title, ha="center", va="center", fontsize=6.5, color="white", fontweight="bold")
    n = len(boxes)
    h = (40.6 - (n - 1) * 1.6) / n
    for i, (bt, bd) in enumerate(boxes):
        y = 43.2 - (i + 1) * h - i * 1.6
        ax.add_patch(FancyBboxPatch((x + 1.2, y), W - 2.4, h, boxstyle="round,pad=0.2,rounding_size=1.0",
                                    fc=fc, ec=ec, lw=0.7))
        ax.text(x + W / 2, y + h - 1.6, bt, ha="center", va="top", fontsize=6.0, fontweight="bold", color="#111")
        ax.text(x + W / 2, y + h / 2 - 1.5, bd, ha="center", va="center", fontsize=5.5, color="#222", linespacing=1.25)
    if s < 3:
        ax.annotate("", xy=(x + W + GAP - 0.2, 25), xytext=(x + W + 0.3, 25),
                    arrowprops=dict(arrowstyle="-|>", lw=1.4, color="#444"))
ax.text(X0 + 3 * (W + GAP) - GAP / 2, 21.5, r"$\hat p_i$", ha="center", fontsize=7.5, color="#D55E00")
fig.savefig("figures/fig01_workflow.png")
fig.savefig(MAN_FIGS + "fig1.png", dpi=600)
