"""Tables and figures for E9 (revision triage results).

Run from code/:  python -m analysis.triage_v2_report [results/e9_triage.json]
Writes tables/s_table_e9_*.csv|md, manuscript tables T_tri.md, T_vs.md, T_abl.md (folder in zdt/paths.py),
figures/fig_e9_sens.png, fig_e9_front.png, fig_e9_simsens.png (and 600 dpi copies in the manuscript figure folder).
"""
import glob
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

from zdt.style import setup, panel
from zdt.paths import MAN_TABLES, MAN_FIGS

setup()
PATH = sys.argv[1] if len(sys.argv) > 1 else "results/e9_triage.json"
QUICK = "quick" in PATH
d = json.load(open(PATH))
EV, SC, TU = d["eval"], d["scenarios"], d["tuned"]
GR = list(d["meta"]["groups"])
INFOS = ("flag", "nov")
KINDS = ("CZT", "Risk", "Linear", "PG")
LAB = {"CZT": "Index (CZT-T)", "Risk": "Tuned Risk", "Linear": "Tuned linear", "PG": "Policy gradient"}
COL = {"CZT": "#D55E00", "Risk": "#0072B2", "Linear": "#009E73", "PG": "#CC79A7"}
MS = ("zcv", "kcv", "cv", "wait_zd", "sla_viol_critical", "n_alerts")
MAN = ""


def vals(s, p, f, groups=GR):
    return np.concatenate([np.asarray(EV[f"{s}|{g}"]["reps"][p][f], float) for g in groups])


def ci(x):
    x = x[~np.isnan(x)]
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return x.mean(), x.mean() - h, x.mean() + h


def fmt_ci(m, lo, hi, nd=3):
    return f"{m:+.{nd}f} [{lo:+.{nd}f}, {hi:+.{nd}f}]"


# ------------------------------------------------------------------ novelty informativeness per scenario
def informativeness():
    """AUROC of the no-match flag (per scenario, analytic) and of the novelty score (empirical) for
    separating unseen-family alerts from the other alerts, at the 1% FPR threshold and the 97/2/1 mix."""
    from experiments.exp_triage_v2 import load_refit
    store = load_refit()
    out = {}
    for g in GR:
        folds = d["meta"]["groups"][g]
        esc = {k: [] for k in ("normal", "known", "zeroday")}
        nov_z, nov_o, w_o = [], [], []
        for f in folds:
            st = store[f]
            p, thr = st["scores"]["CZT-Hybrid"]["p"], np.sort(st["scores"]["CZT-Hybrid"]["pva_normal"])
            t = thr[int(np.ceil(0.99 * len(thr))) - 1]
            for k in esc:
                m = st["groups"] == k
                esc[k].append(float((p[m] >= t).mean()))
            a = p >= t
            for k, wk in (("normal", 0.97), ("known", 0.02)):
                m = a & (st["groups"] == k)
                nov_o.append(st["nov"][m]); w_o.append(np.full(m.sum(), wk / max(m.sum(), 1)))
            nov_z.append(st["nov"][a & (st["groups"] == "zeroday")])
        e = {k: np.mean(v) for k, v in esc.items()}
        sN, sK = 0.97 * e["normal"], 0.02 * e["known"]
        z, o, w = np.concatenate(nov_z), np.concatenate(nov_o), np.concatenate(w_o)
        auc_nov = roc_auc_score(np.r_[np.ones(len(z)), np.zeros(len(o))], np.r_[z, o],
                                sample_weight=np.r_[np.full(len(z), 1 / len(z)), w / w.sum()])
        out[g] = dict(sN=sN, sK=sK, auc_nov=auc_nov)
    res = {}
    for s, c in SC.items():
        a = []
        for g in GR:
            sN, sK = out[g]["sN"], out[g]["sK"]
            p_unm_other = (sN * (1 - c["m_b"]) + sK * (1 - c["m_k"])) / (sN + sK)
            a.append(0.5 * (1 + (1 - c["m_z"]) - p_unm_other))
        res[s] = dict(flag=float(np.mean(a)), nov=float(np.mean([out[g]["auc_nov"] for g in GR])))
    return res


INF = informativeness()

# ------------------------------------------------------------------ long tables
rows, diffs = [], []
for s in SC:
    c = SC[s]
    base = dict(scenario=s, m_k=c["m_k"], m_z=c["m_z"], m_b=c["m_b"], sev_bias=c["sev_bias"], zd_share=c["mix"][2],
                inf_flag=INF[s]["flag"], inf_nov=INF[s]["nov"])
    for p in EV[f"{s}|{GR[0]}"]["reps"]:
        rows.append({**base, "policy": p, **{f: float(np.nanmean(vals(s, p, f))) for f in MS}})
    for info in INFOS:
        for k in KINDS:
            a = f"{k}-T/{info}"
            for ref in ("Risk",) + ((f"Risk-T/{info}",) if k != "Risk" else ()):
                for grp in ["pooled"] + GR:
                    gs = GR if grp == "pooled" else [grp]
                    r = {**base, "info": info, "policy": a, "versus": ref, "group": grp}
                    for f in ("zcv", "kcv", "cv", "sla_viol_critical"):
                        m, lo, hi = ci(vals(s, a, f, gs) - vals(s, ref, f, gs))
                        r.update({f: m, f + "_lo": lo, f + "_hi": hi})
                    diffs.append(r)
M = pd.DataFrame(rows)
D = pd.DataFrame(diffs)
tag = "_quick" if QUICK else ""
M.to_csv(f"tables/s_table_e9_means{tag}.csv", index=False)
D.to_csv(f"tables/s_table_e9_paired{tag}.csv", index=False)
T = []
for key, v in TU.items():
    s, g, k, info = key.split("|")
    T.append(dict(scenario=s, eval_group=g, tuned_on={"G1": "G2", "G2": "G1"}[g], policy=k, info=info,
                  feasible=v["feasible"], n_feasible=v["n_feasible"], n_candidates=len(v["tune_points"]),
                  kcv_ref_tune=v["kcv_ref_tune"],
                  params=json.dumps({a: (np.round(b, 4).tolist() if isinstance(b, list) else (round(b, 4) if isinstance(b, float) else b))
                                     for a, b in v["params"].items()}),
                  tune_zcv=v["tune_points"][v["selected"]][0], tune_kcv=v["tune_points"][v["selected"]][1],
                  tune_crit=v["tune_points"][v["selected"]][2], time_s=v["time_s"]))
TT = pd.DataFrame(T).sort_values(["scenario", "info", "policy", "eval_group"])
TT.to_csv(f"tables/s_table_e9_tuned{tag}.csv", index=False)

# ------------------------------------------------------------------ manuscript tables (compact: mean +/- 95% half-width)
ANCH = [("original", "Original"), ("central", "Imperfect"), ("uninformative_0.5", "Uninformative")]
ANCH = [a for a in ANCH if f"{a[0]}|{GR[0]}" in EV]
SHORT = {"CZT": "Index", "Risk": "Tuned Risk", "Linear": "Linear", "PG": "PG"}


def pm(x):
    m, lo, hi = ci(x)
    return f"{m:+.3f} ± {(hi - lo) / 2:.3f}".replace("-0.000", "0.000").replace("+0.000", "0.000")


POL = [("FIFO", "FIFO"), ("Severity", "Severity"), ("EDF", "EDF"), ("CZT-default", "Index, default")] + \
      [(f"{k}-T/{i}", f"{SHORT[k]}, {'flag' if i == 'flag' else 'novelty'}") for i in INFOS for k in KINDS]
hdr = "| Policy | " + " | ".join(f"{lab[:5] + '.' if len(lab) > 9 else lab} ZCV | {lab[:5] + '.' if len(lab) > 9 else lab} KCV" for _, lab in ANCH) + " |"
L = [hdr, "|" + "---|" * (1 + 2 * len(ANCH))]
L.append("| Risk (value) | " + " | ".join(f"{np.nanmean(vals(s, 'Risk', 'zcv')):.3f} | {np.nanmean(vals(s, 'Risk', 'kcv')):.3f}" for s, _ in ANCH) + " |")
for p, pl in POL:
    L.append(f"| {pl} | " + " | ".join(f"{pm(vals(s, p, 'zcv') - vals(s, 'Risk', 'zcv'))} | {pm(vals(s, p, 'kcv') - vals(s, 'Risk', 'kcv'))}" for s, _ in ANCH) + " |")
open(f"{MAN_TABLES}T_tri{tag}.md", "w").write("\n".join(L) + "\n")

L = [hdr.replace("Policy", "Versus").replace("Original", "Orig.").replace("Imperfect", "Imp."), "|" + "---|" * (1 + 2 * len(ANCH))]
for info in INFOS:
    for b in ("Risk", "Linear", "PG"):
        a = f"CZT-T/{info}"
        L.append(f"| {SHORT[b] if b != 'Risk' else 'Tuned Risk'}, {'flag' if info == 'flag' else 'novelty'} | " + " | ".join(
            f"{pm(vals(s, a, 'zcv') - vals(s, f'{b}-T/{info}', 'zcv'))} | {pm(vals(s, a, 'kcv') - vals(s, f'{b}-T/{info}', 'kcv'))}" for s, _ in ANCH) + " |")
open(f"{MAN_TABLES}T_vs{tag}.md", "w").write("\n".join(L) + "\n")

AB = [("no severity (beta=0)", "no severity term"), ("no score (gamma=0)", "no score term"), ("no decay", "no evidence decay"),
      ("no deadline (kappa=0)", "no deadline term"), ("no imputation (s_u=0.5)", "imputed severity 0.5"), ("[prior]", "prior-corrected score")]
A2 = ANCH[:2]
L = ["| Variant | " + " | ".join(f"{lab}, {'flag' if i == 'flag' else 'novelty'}" for _, lab in A2 for i in INFOS) + " |",
     "|" + "---|" * (1 + 2 * len(A2))]
L.append("| Tuned index (ZCV) | " + " | ".join(f"{np.nanmean(vals(s, f'CZT-T/{i}', 'zcv')):.3f}" for s, _ in A2 for i in INFOS) + " |")
for v, nm in AB:
    L.append(f"| {nm} | " + " | ".join(pm(vals(s, (f'CZT-T/{i}{v}' if v == '[prior]' else f'CZT-T/{i} {v}'), 'zcv') - vals(s, f'CZT-T/{i}', 'zcv'))
                                        for s, _ in A2 for i in INFOS) + " |")
open(f"{MAN_TABLES}T_abl{tag}.md", "w").write("\n".join(L) + "\n")

# ------------------------------------------------------------------ Figure: gain versus informativeness
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0))
for ax, f, ttl in ((axs[0], "zcv", "Zero-day value (ZCV) minus Risk"), (axs[1], "kcv", "Known-attack value (KCV) minus Risk")):
    for k in KINDS:
        for info, mk in (("flag", "o"), ("nov", "^")):
            sub = D[(D.policy == f"{k}-T/{info}") & (D.versus == "Risk") & (D.group == "pooled")]
            x = sub["inf_flag"] if info == "flag" else sub["inf_nov"]
            ax.errorbar(x, sub[f], yerr=[sub[f] - sub[f + "_lo"], sub[f + "_hi"] - sub[f]], fmt=mk, ms=3.2, lw=0.6,
                        color=COL[k], mfc=COL[k] if info == "flag" else "white", alpha=0.85,
                        label=f"{LAB[k]}" if info == "flag" else None)
    ax.axhline(0, color="#555", lw=0.8)
    ax.set_xlabel("Novelty AUROC (unseen-family vs other alerts)")
    ax.set_title(ttl)
panel(axs[0], "a"); panel(axs[1], "b")
axs[0].set_ylabel("Paired difference (40 replications)")
h, l = axs[0].get_legend_handles_labels()
from matplotlib.lines import Line2D
h += [Line2D([], [], marker="o", ls="", color="#555"), Line2D([], [], marker="^", ls="", color="#555", mfc="white")]
l += ["no-match flag", "novelty score"]
fig.legend(h, l, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.08), fontsize=7)
fig.tight_layout()
fig.savefig(f"figures/fig_e9_sens{tag}.png")
if not QUICK:
    pass  # Fig. 9 is now drawn by analysis.fig9_v3
plt.close(fig)

# ------------------------------------------------------------------ Figure: out-of-sample fronts
SEL = [s for s, _ in ANCH if f"{s}|{GR[0]}" in EV]
fig, axs = plt.subplots(2, len(SEL), figsize=(7.2, 4.9), sharex=True, sharey=True, squeeze=False)
for j, s in enumerate(SEL):
    for i, info in enumerate(INFOS):
        ax = axs[i, j]
        for k in KINDS:
            for g, gm, sm in zip(GR, ("o", "s"), ("*", "D")):
                pts = np.asarray(EV[f"{s}|{g}"]["fronts"][f"{k}/{info}"])
                if g == GR[0]:
                    ax.scatter(pts[:, 1], pts[:, 0], s=5, marker=gm, color=COL[k], alpha=0.3, lw=0,
                               label=f"{LAB[k]}" if (i, j) == (0, 0) else None)
                else:
                    ax.scatter(pts[:, 1], pts[:, 0], s=6, marker=gm, facecolors="none", edgecolors=COL[k], alpha=0.45, lw=0.4)
                ax.scatter(np.nanmean(vals(s, f"{k}-T/{info}", "kcv", [g])), np.nanmean(vals(s, f"{k}-T/{info}", "zcv", [g])),
                           s=55 if sm == "*" else 22, marker=sm, color=COL[k], edgecolor="black", lw=0.5, zorder=5)
        for g, gm in zip(GR, ("o", "s")):
            ax.scatter(np.nanmean(vals(s, "Risk", "kcv", [g])), np.nanmean(vals(s, "Risk", "zcv", [g])), s=28, marker=gm,
                       color="black", zorder=6, label=f"Risk (untuned), {g}" if (i, j) == (0, 0) else None)
        ax.set_title(f"{['Original', 'Imperfect', 'Uninformative'][j]}, {'flag' if info == 'flag' else 'novelty score'}", fontsize=8.5)
        if i == 1:
            ax.set_xlabel("KCV")
        if j == 0:
            ax.set_ylabel("ZCV")
for n, ax in enumerate(axs.flat):
    panel(ax, "abcdef"[n], x=-0.2)
from matplotlib.lines import Line2D
h, l = axs[0, 0].get_legend_handles_labels()
h += [Line2D([], [], marker="o", ls="", color="#777", ms=3), Line2D([], [], marker="s", ls="", mfc="none", mec="#777", ms=3),
      Line2D([], [], marker="*", ls="", color="#777", mec="black", ms=8), Line2D([], [], marker="D", ls="", color="#777", mec="black", ms=4.5)]
l += [f"candidates, evaluated on {GR[0]}", f"candidates, evaluated on {GR[1]}", f"selected policy, {GR[0]}", f"selected policy, {GR[1]}"]
fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.12), fontsize=6.5)
fig.tight_layout()
fig.savefig(f"figures/fig_e9_front{tag}.png")
if not QUICK:
    fig.savefig(f"{MAN_FIGS}fig10.png", dpi=600)
plt.close(fig)

# ------------------------------------------------------------------ Figure and table: simulator sensitivity
S = d.get("sens", {})
if S:
    rows = []
    for key, v in S.items():
        s, g, n = key.split("|")
        for p in v:
            if p == "Risk":
                continue
            rows.append(dict(scenario=s, group=g, setting=n, policy=p,
                             dz=list(np.asarray(v[p]["zcv"]) - np.asarray(v["Risk"]["zcv"])),
                             dk=list(np.asarray(v[p]["kcv"]) - np.asarray(v["Risk"]["kcv"]))))
    R = pd.DataFrame(rows)
    agg = []
    for (s, n, p), sub in R.groupby(["scenario", "setting", "policy"]):
        dz = np.concatenate(sub.dz.values); dk = np.concatenate(sub.dk.values)
        agg.append(dict(scenario=s, setting=n, policy=p, **dict(zip(("dzcv", "dzcv_lo", "dzcv_hi"), ci(dz))),
                        **dict(zip(("dkcv", "dkcv_lo", "dkcv_hi"), ci(dk)))))
    A = pd.DataFrame(agg)
    base = D[(D.versus == "Risk") & (D.group == "pooled")]
    for s in A.scenario.unique():
        for info in INFOS:
            for k in KINDS:
                b = base[(base.scenario == s) & (base.policy == f"{k}-T/{info}")].iloc[0]
                A.loc[len(A)] = dict(scenario=s, setting="base (rho1.2, c4)", policy=f"{k}-T/{info}", dzcv=b.zcv, dzcv_lo=b.zcv_lo,
                                     dzcv_hi=b.zcv_hi, dkcv=b.kcv, dkcv_lo=b.kcv_lo, dkcv_hi=b.kcv_hi)
    A.to_csv(f"tables/s_table_e9_simsens{tag}.csv", index=False)
    order = ["base (rho1.2, c4)", "rho0.8", "rho1.0", "rho1.5", "c3", "c6", "halflife30", "halflife120", "sla_x0.5", "sla_x2",
             "svc_attack20", "benign0.95", "benign0.99"]
    order = [o for o in order if o in set(A.setting)]
    scs = [s for s in ("original", "central", "uninformative_0.5") if s in set(A.scenario)]
    fig, axs = plt.subplots(len(scs), 2, figsize=(7.2, 2.1 * len(scs) + 0.6), sharey=True, squeeze=False)
    for i, s in enumerate(scs):
        for jj, (f, ttl) in enumerate((("dzcv", "ZCV minus Risk"), ("dkcv", "KCV minus Risk"))):
            ax = axs[i, jj]
            for kk, k in enumerate(KINDS):
                for info, mk in (("flag", "o"), ("nov", "^")):
                    sub = A[(A.scenario == s) & (A.policy == f"{k}-T/{info}")].set_index("setting").reindex(order)
                    y = np.arange(len(order)) + (kk - 1.5) * 0.12 + (0.05 if info == "nov" else -0.05)
                    ax.errorbar(sub[f], y, xerr=[sub[f] - sub[f + "_lo"], sub[f + "_hi"] - sub[f]], fmt=mk, ms=2.6, lw=0.5,
                                color=COL[k], mfc=COL[k] if info == "flag" else "white",
                                label=LAB[k] if (i, jj, info) == (0, 0, "flag") else None)
            ax.axvline(0, color="#555", lw=0.8)
            NICE = {"base (rho1.2, c4)": "base (load 1.2, 4 analysts)", "rho0.8": "load 0.8", "rho1.0": "load 1.0", "rho1.5": "load 1.5",
                    "c3": "3 analysts", "c6": "6 analysts", "halflife30": "half-life 30 min", "halflife120": "half-life 120 min",
                    "sla_x0.5": "SLA limits halved", "sla_x2": "SLA limits doubled", "svc_attack20": "attack service 20 min",
                    "benign0.95": "benign share 0.95", "benign0.99": "benign share 0.99"}
            ax.set_yticks(range(len(order))); ax.set_yticklabels([NICE[o] for o in order], fontsize=6.5); ax.invert_yaxis()
            ax.set_title(f"{['Original', 'Imperfect', 'Uninformative'][('original', 'central', 'uninformative_0.5').index(s)]}: {ttl}", fontsize=7.5)
    for n, ax in enumerate(axs.flat):
        panel(ax, "abcdef"[n], x=-0.3 if n % 2 == 0 else -0.08)
    from matplotlib.lines import Line2D
    h, l = axs[0, 0].get_legend_handles_labels()
    h += [Line2D([], [], marker="o", ls="", color="#555"), Line2D([], [], marker="^", ls="", color="#555", mfc="white")]
    l += ["no-match flag", "novelty score"]
    fig.legend(h, l, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.03), fontsize=7)
    fig.tight_layout()
    fig.savefig(f"figures/fig_e9_simsens{tag}.png")
    if not QUICK:
        fig.savefig(f"{MAN_FIGS}fig11.png", dpi=600)
    plt.close(fig)

# ------------------------------------------------------------------ console summary
pd.set_option("display.width", 250)
P = D[(D.group == "pooled")]
for info in INFOS:
    for k in KINDS:
        sub = P[(P.policy == f"{k}-T/{info}") & (P.versus == "Risk")]
        print(f"{k:6s} {info}: vs Risk ZCV>0 sig {int((sub.zcv_lo > 0).sum())}, <0 sig {int((sub.zcv_hi < 0).sum())} of {len(sub)}; "
              f"KCV<0 sig {int((sub.kcv_hi < 0).sum())}; median dZCV {sub.zcv.median():+.3f} median dKCV {sub.kcv.median():+.3f}")
    for k in ("CZT", "Linear", "PG"):
        sub = P[(P.policy == f"{k}-T/{info}") & (P.versus == f"Risk-T/{info}")]
        print(f"   {k} vs tuned Risk ({info}): ZCV better {int((sub.zcv_lo > 0).sum())}, worse {int((sub.zcv_hi < 0).sum())}, n.d. "
              f"{int(((sub.zcv_lo <= 0) & (sub.zcv_hi >= 0)).sum())}")
print("feasible share:", TT.groupby(["policy", "info"]).feasible.mean().round(2).to_dict())
print({s: {k: round(v, 3) for k, v in INF[s].items()} for s in list(INF)[:3]})
