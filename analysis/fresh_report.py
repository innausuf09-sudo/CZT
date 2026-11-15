"""Fresh-stream evaluation (E20): rule-family comparisons on ZCV, KCV and CV under the main selection rule and
the knee rule, in the 35 distinct settings, with Holm-adjusted counts and an equivalence check (margin 0.01).

Run from code/:  python -m analysis.fresh_report
Writes tables/s_table_e20_fresh.csv, tables/s_table_e20_counts.csv, results/e20_summary.json
"""
import json

import numpy as np
import pandas as pd
from scipy import stats
from zdt.paths import MAN_TABLES, MAN_FIGS

E = json.load(open("results/e20_fresh.json"))["eval"]
E9 = json.load(open("results/e9_triage.json"))
GR = ["G1", "G2"]
SC = sorted({k.split("|")[0] for k in E})
MARGIN = 0.01


def rep(s, p, f):
    return np.concatenate([np.asarray(E[f"{s}|{g}"][p][f], float) for g in GR])


def e9rep(s, p, f):
    return np.concatenate([np.asarray(E9["eval"][f"{s}|{g}"]["reps"][p][f], float) for g in GR])


def test(x):
    n = len(x); m = x.mean(); se = x.std(ddof=1) / np.sqrt(n)
    h = stats.t.ppf(0.975, n - 1) * se
    p = 2 * stats.t.sf(abs(m / se), n - 1) if se > 0 else 1.0
    # TOST: equivalent if the 90% interval lies inside (-margin, margin)
    h90 = stats.t.ppf(0.95, n - 1) * se
    eq = (m - h90 > -MARGIN) and (m + h90 < MARGIN)
    return m, m - h, m + h, p, eq


rows = []
for s in SC:
    for info in ("flag", "nov"):
        for rule, suf in (("main", ""), ("knee", "[knee]")):
            for a, b in (("CZT", "Risk"), ("CZT", "Linear"), ("Linear", "Risk"), ("PG", "Risk")):
                r = dict(setting=s, info=info, rule=rule, comparison=f"{a} - {b}")
                for f in ("zcv", "kcv", "cv", "sla_viol_critical"):
                    m, lo, hi, p, eq = test(rep(s, f"{a}-T/{info}{suf}", f) - rep(s, f"{b}-T/{info}{suf}", f))
                    r.update({f: m, f + "_lo": lo, f + "_hi": hi, f + "_p": p, f + "_eq": eq})
                rows.append(r)
        for rule, suf in (("main", ""), ("knee", "[knee]")):
            for a in ("CZT", "Risk", "Linear", "PG"):
                r = dict(setting=s, info=info, rule=rule, comparison=f"{a} - untuned Risk")
                for f in ("zcv", "kcv", "cv", "sla_viol_critical"):
                    m, lo, hi, p, eq = test(rep(s, f"{a}-T/{info}{suf}", f) - rep(s, "Risk", f))
                    r.update({f: m, f + "_lo": lo, f + "_hi": hi, f + "_p": p, f + "_eq": eq})
                rows.append(r)
D = pd.DataFrame(rows)


def holm(p):
    p = np.asarray(p); o = np.argsort(p); n = len(p); adj = np.empty(n); run = 0
    for i, k in enumerate(o):
        run = max(run, min(1, (n - i) * p[k])); adj[k] = run
    return adj


for f in ("zcv", "kcv", "cv"):
    D[f + "_holm"] = np.nan
    for _, idx in D.groupby(["info", "rule", "comparison"]).groups.items():
        D.loc[idx, f + "_holm"] = holm(D.loc[idx, f + "_p"].values)


def verdict(r, f, adj=False):
    sig = (r[f + "_holm"] < 0.05) if adj else (r[f + "_lo"] > 0 or r[f + "_hi"] < 0)
    if not sig:
        return "equivalent" if r[f + "_eq"] else "n.d."
    return "better" if r[f] > 0 else "worse"


for f in ("zcv", "kcv", "cv"):
    D[f + "_v"] = D.apply(lambda r: verdict(r, f), axis=1)
    D[f + "_vholm"] = D.apply(lambda r: verdict(r, f, True), axis=1)


def pareto(r):
    z = (r.zcv_lo > 0) - (r.zcv_hi < 0); k = (r.kcv_lo > 0) - (r.kcv_hi < 0)
    if z >= 0 and k >= 0 and z + k > 0:
        return "first better"
    if z <= 0 and k <= 0 and z + k < 0:
        return "second better"
    return "no detected difference" if z == 0 and k == 0 else "trade-off"


D["pareto"] = D.apply(pareto, axis=1)
D.to_csv("tables/s_table_e20_fresh.csv", index=False)

C = []
for (info, rule, comp), g in D.groupby(["info", "rule", "comparison"], sort=False):
    c = dict(Information=info, Rule=rule, Comparison=comp)
    for f in ("zcv", "kcv", "cv"):
        v = g[f + "_v"].value_counts(); vh = g[f + "_vholm"].value_counts()
        c[f.upper()] = f"{v.get('better', 0)}/{v.get('worse', 0)}/{v.get('equivalent', 0)} (Holm {vh.get('better', 0)}/{vh.get('worse', 0)})"
        c[f.upper() + " median"] = round(float(g[f].median()), 3)
    pv = g.pareto.value_counts()
    c["Both objectives (first/second/trade-off)"] = f"{pv.get('first better', 0)}/{pv.get('second better', 0)}/{pv.get('trade-off', 0)}"
    C.append(c)
C = pd.DataFrame(C)
C.to_csv("tables/s_table_e20_counts.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 20)
print(C.to_string(index=False))

# consistency with the original evaluation streams (seeds 1,000 to 1,019)
chk = []
for s in SC:
    for p in ("Risk", "CZT-T/flag", "Risk-T/flag", "Linear-T/flag", "CZT-T/nov"):
        chk.append(dict(setting=s, policy=p, fresh=rep(s, p, "zcv").mean(), main=e9rep(s, p, "zcv").mean()))
chk = pd.DataFrame(chk)
chk["diff"] = chk.fresh - chk.main
print("fresh minus main streams, ZCV: max abs", round(chk["diff"].abs().max(), 4), "mean", round(chk["diff"].mean(), 4))
an = D[D.setting.isin(["original", "central", "uninformative_0.5"]) & (D.rule == "main")]
print(an[["setting", "info", "comparison", "zcv", "zcv_lo", "zcv_hi", "kcv", "cv", "sla_viol_critical"]].round(3).to_string(index=False))
json.dump(dict(counts=C.to_dict("records"), fresh_vs_main_max_abs=float(chk["diff"].abs().max())),
          open("results/e20_summary.json", "w"), indent=1)

# ---------------------------------------------------------------- main-text table (counts, main rule and knee rule)
def cell(g, f):
    v = g[f + "_v"].value_counts(); vh = g[f + "_vholm"].value_counts()
    return f"{v.get('better', 0)}/{v.get('worse', 0)} ({vh.get('better', 0)}/{vh.get('worse', 0)})"


L = ["| Comparison | Information | ZCV | KCV | Total value (CV) | Median CV difference |", "|---|---|---|---|---|---|"]
NM = {"CZT - Risk": "Index minus tuned Risk", "CZT - Linear": "Index minus tuned linear", "Linear - Risk": "Tuned linear minus tuned Risk",
      "PG - Risk": "Policy gradient minus tuned Risk"}
for rule in ("main", "knee"):
    comps = list(NM) if rule == "main" else ["CZT - Risk", "CZT - Linear"]
    for comp in comps:
        for info in ("flag", "nov"):
            g = D[(D.rule == rule) & (D.comparison == comp) & (D["info"] == info)]
            lab = NM[comp] + (", knee rule" if rule == "knee" else "")
            med = g.cv.median()
            L.append(f"| {lab} | {'flag' if info == 'flag' else 'novelty'} | {cell(g, 'zcv')} | {cell(g, 'kcv')} | {cell(g, 'cv')} | {med:+.3f} |".replace("-0.000", "0.000"))
open(MAN_TABLES + "T_fresh.md", "w").write("\n".join(L) + "\n")
print("\n".join(L))

# ---------------------------------------------------------------- supplement table: per setting, main rule
def fmt(r, f):
    return f"{r[f]:+.3f} [{r[f + '_lo']:+.3f}, {r[f + '_hi']:+.3f}]".replace("-0.000", "0.000")


S = []
for s in SC:
    for info in ("flag", "nov"):
        g = D[(D.setting == s) & (D["info"] == info) & (D.rule == "main")].set_index("comparison")
        a, b = g.loc["CZT - Risk"], g.loc["CZT - Linear"]
        S.append({"Setting": s, "Information": "flag" if info == "flag" else "novelty score",
                  "Index minus tuned Risk, ZCV": fmt(a, "zcv"), "KCV": fmt(a, "kcv"), "CV": fmt(a, "cv"),
                  "Index minus tuned linear, CV": fmt(b, "cv"),
                  "Fresh minus main streams, index ZCV": round(float(rep(s, f"CZT-T/{info}", "zcv").mean() - e9rep(s, f"CZT-T/{info}", "zcv").mean()), 3)})
pd.DataFrame(S).to_csv("tables/s_table_e20_setting.csv", index=False)
