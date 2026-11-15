"""Grid-level estimates of the rule-family comparisons (third review round).

Counting settings with an interval that excludes zero treats the 35 settings as independent, although they share the
random streams. Here each comparison is summarised by one estimate, the mean paired difference over the 35 distinct
settings, with a bootstrap interval that resamples whole replication streams (stratified by family group), so that the
dependence across settings is kept. Main streams (seeds 1,000 to 1,019 per group), fresh streams (9,000 to 9,019) and
both pooled (80 replications) are reported. Primary outcomes: ZCV and KCV (the objectives of the selection rule);
secondary: CV.

Run from code/:  python -m analysis.grid_report
Writes tables/s_table_grid.csv, the manuscript table T_grid.md (folder in zdt/paths.py), results/grid_summary.json
"""
import json

import numpy as np
import pandas as pd
from zdt.paths import MAN_TABLES, MAN_FIGS

E9 = json.load(open("results/e9_triage.json"))["eval"]
E20 = json.load(open("results/e20_fresh.json"))["eval"]
DUP = ("mk1.0_mz0.0_mb0.5", "mk0.8_mz0.2_mb0.5")
SC = sorted({k.split("|")[0] for k in E20})
GR = ["G1", "G2"]
B = 4000
rng = np.random.default_rng(2026)


def arr(src, p, f, rule=""):
    """settings x groups x replications"""
    if src == "main":
        return np.array([[np.asarray(E9[f"{s}|{g}"]["reps"][p][f], float) for g in GR] for s in SC])
    return np.array([[np.asarray(E20[f"{s}|{g}"][p + rule][f], float) for g in GR] for s in SC])


def diff(src, a, b, f, rule=""):
    if src == "pooled":
        return np.concatenate([diff("main", a, b, f), diff("fresh", a, b, f, rule)], axis=2)
    pa = a if a == "Risk" else a
    return arr(src, pa, f, rule if a != "Risk" else "") - arr(src, b, f, rule if b != "Risk" else "")


def grid_ci(D):
    # D: settings x 2 groups x n reps; resample reps within group, same indices for every setting
    n = D.shape[2]
    est = D.mean()
    bs = np.empty(B)
    for i in range(B):
        idx = rng.integers(n, size=(2, n))
        bs[i] = np.mean([D[:, g, idx[g]].mean() for g in range(2)])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return est, lo, hi


COMP = [("CZT-T/{i}", "Risk-T/{i}", "Index minus tuned Risk"), ("CZT-T/{i}", "Linear-T/{i}", "Index minus tuned linear"),
        ("Linear-T/{i}", "Risk-T/{i}", "Tuned linear minus tuned Risk"), ("PG-T/{i}", "Risk-T/{i}", "Policy gradient minus tuned Risk"),
        ("CZT-T/{i}", "Risk", "Index minus untuned Risk"), ("Risk-T/{i}", "Risk", "Tuned Risk minus untuned Risk"),
        ("Linear-T/{i}", "Risk", "Tuned linear minus untuned Risk")]
rows = []
for info in ("flag", "nov"):
    for a, b, lab in COMP:
        a_, b_ = a.format(i=info), b.format(i=info)
        for src in ("main", "fresh", "pooled"):
            r = dict(Comparison=lab, Information=info, Streams=src)
            for f in ("zcv", "kcv", "cv"):
                e, lo, hi = grid_ci(diff(src, a_, b_, f))
                r[f] = e; r[f + "_lo"] = lo; r[f + "_hi"] = hi
            rows.append(r)
        # knee rule, fresh streams only (same comparison, both policies chosen by the knee rule)
        if b_ != "Risk":
            r = dict(Comparison=lab, Information=info, Streams="fresh, knee rule")
            for f in ("zcv", "kcv", "cv"):
                D = arr("fresh", a_, f, "[knee]") - arr("fresh", b_, f, "[knee]")
                e, lo, hi = grid_ci(D); r[f] = e; r[f + "_lo"] = lo; r[f + "_hi"] = hi
            rows.append(r)
G = pd.DataFrame(rows)
G.to_csv("tables/s_table_grid.csv", index=False)


def fmt(r, f):
    return f"{r[f]:+.3f} [{r[f + '_lo']:+.3f}, {r[f + '_hi']:+.3f}]".replace("-0.000", "0.000").replace("+0.000", "0.000")


pd.set_option("display.width", 250)
G2 = G.assign(**{f.upper(): G.apply(lambda r, f=f: fmt(r, f), axis=1) for f in ("zcv", "kcv", "cv")})
print(G2[["Comparison", "Information", "Streams", "ZCV", "KCV", "CV"]].to_string(index=False))
L = ["| Comparison | Information | ZCV | KCV | Total value (CV) |", "|---|---|---|---|---|"]
for _, r in G2[G2.Streams == "pooled"].iterrows():
    if "untuned" in r.Comparison:
        continue
    L.append(f"| {r.Comparison} | {'flag' if r.Information == 'flag' else 'novelty'} | {r.ZCV} | {r.KCV} | {r.CV} |")
for _, r in G2[G2.Streams == "fresh, knee rule"].iterrows():
    if r.Comparison in ("Index minus tuned Risk", "Index minus tuned linear"):
        L.append(f"| {r.Comparison}, knee rule | {'flag' if r.Information == 'flag' else 'novelty'} | {r.ZCV} | {r.KCV} | {r.CV} |")
open(MAN_TABLES + "T_grid.md", "w").write("\n".join(L) + "\n")
json.dump(G.to_dict("records"), open("results/grid_summary.json", "w"), indent=1)
