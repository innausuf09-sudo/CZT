"""Figure 9 of the revision: triage gain against the informativeness of the novelty information.

(a) ZCV minus Risk, (b) ZCV gain over the same policy tuned without novelty information (E17 stage A),
(c) KCV minus Risk. 35 distinct catalogue settings (the two duplicate settings are dropped).

Run from code/ after analysis.triage_v2_report and analysis.robust_report:  python -m analysis.fig9_v3
"""
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

from zdt.style import setup, panel
from zdt.paths import MAN_TABLES, MAN_FIGS

setup()
DUP = ("mk1.0_mz0.0_mb0.5", "mk0.8_mz0.2_mb0.5")
KINDS = ("CZT", "Risk", "Linear", "PG")
LAB = {"CZT": "Index (CZT-T)", "Risk": "Tuned Risk", "Linear": "Tuned linear", "PG": "Policy gradient"}
COL = {"CZT": "#D55E00", "Risk": "#0072B2", "Linear": "#009E73", "PG": "#CC79A7"}
MK = {"CZT": "o", "Risk": "s", "Linear": "^", "PG": "D"}
OFF = {"CZT": -0.009, "Risk": -0.003, "Linear": 0.003, "PG": 0.009}  # horizontal offset for visibility
NAME = {"Index": "CZT", "Tuned Risk": "Risk", "Linear rule": "Linear"}

D = pd.read_csv("tables/s_table_e9_paired.csv")
D = D[~D.scenario.isin(DUP) & (D.group == "pooled") & (D.versus == "Risk")]
A = pd.read_csv("tables/s_table_e17_nonov_numeric.csv")
A["kind"] = A.policy.map(NAME)


def draw(ax, k, info, x, y, lo, hi):
    ax.errorbar(x + OFF[k], y, yerr=[y - lo, hi - y], fmt=MK[k], ms=3.0, lw=0.5, color=COL[k],
                mfc=COL[k] if info == "flag" else "white", mew=0.8, alpha=0.85)


fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.9), sharex=True)
for k in KINDS:
    for info in ("flag", "nov"):
        s = D[D.policy == f"{k}-T/{info}"]
        x = s.inf_flag if info == "flag" else s.inf_nov
        draw(axs[0], k, info, x, s.zcv, s.zcv_lo, s.zcv_hi)
        draw(axs[2], k, info, x, s.kcv, s.kcv_lo, s.kcv_hi)
        a = A[(A.kind == k) & (A["info"] == info)]
        if len(a):
            draw(axs[1], k, info, a.info_auroc, a.dzcv, a.dzcv_lo, a.dzcv_hi)
for ax, ttl in zip(axs, ("ZCV minus Risk", "ZCV gain from novelty info.", "KCV minus Risk")):
    ax.axhline(0, color="#555", lw=0.8)
    ax.set_title(ttl, fontsize=7.5, loc="right")
    ax.set_xlabel("AUROC of novelty information", fontsize=7.5)
for ax, p in zip(axs, "abc"):
    panel(ax, p)
axs[0].set_ylabel("Paired difference (40 replications)", fontsize=7.5)
h = [Line2D([], [], marker=MK[k], ls="", color=COL[k]) for k in KINDS]
l = [LAB[k] for k in KINDS]
h += [Line2D([], [], marker="o", ls="", color="#555"), Line2D([], [], marker="o", ls="", color="#555", mfc="white")]
l += ["no-match flag", "novelty score"]
fig.legend(h, l, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.07), fontsize=7)
fig.tight_layout()
fig.savefig("figures/fig_e9_sens_v3.png")
fig.savefig(MAN_FIGS + "fig9.png", dpi=600)
print("ok", len(D), len(A))
