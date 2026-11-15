"""Tables for E8 (catalogue observation model, tuned baselines, CZT-T ablations).

Run from code/:  python -m analysis.catalogue_tables [results/e8_catalogue.json]
Writes tables/s_table_e8_*.csv and .md.
Paired differences use the 2 x REPS replications (same raw stream, same service
times and same catalogue draws for both policies); intervals are 95% t-intervals
over the paired replication-level differences.
"""
import json
import sys

import numpy as np
import pandas as pd
from scipy import stats

PATH = sys.argv[1] if len(sys.argv) > 1 else "results/e8_catalogue.json"
TAG = "_quick" if "quick" in PATH else ""
d = json.load(open(PATH))
EV, SC = d["eval"], d["scenarios"]
GR = list(d["meta"]["groups"])
MAIN = ["FIFO", "Severity", "EDF", "Risk", "CZT-default", "Risk-T", "Linear-T", "CZT-T", "Risk-TE", "Linear-TE",
        "CZT-TE", "CZT-T[orig]", "Risk-T[orig]", "Linear-T[orig]"]
PAIRS = [("CZT-T", "Risk"), ("CZT-T", "Risk-T"), ("CZT-T", "Linear-T"), ("Risk-T", "Risk"),
         ("CZT-TE", "Risk-TE"), ("CZT-TE", "Linear-TE"), ("CZT-T[orig]", "Risk")]
METRICS = ["zcv", "cv", "kcv", "wait_zd", "sla_viol_critical"]


def vals(scen, pol, f, groups=GR):
    return np.concatenate([np.asarray(EV[f"{scen}|{g}"][pol][f], float) for g in groups])


def ci(x):
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return np.nan, np.nan, np.nan
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return x.mean(), x.mean() - h, x.mean() + h


def write(df, name):
    df.to_csv(f"tables/s_table_e8_{name}{TAG}.csv", index=False)
    with open(f"tables/s_table_e8_{name}{TAG}.md", "w") as fh:
        fh.write(df.to_markdown(index=False, floatfmt=".3f"))
    return df


def scen_cols(s):
    c = SC[s]
    return dict(scenario=s, m_k=c["m_k"], m_z=c["m_z"], m_b=c["m_b"], sev_bias=c["sev_bias"], zd_share=c["mix"][2])


# 1) pooled means per scenario and policy
rows = []
for s in SC:
    for p in MAIN:
        rows.append({**scen_cols(s), "policy": p, **{f: np.nanmean(vals(s, p, f)) for f in
                                                      ["zcv", "cv", "kcv", "wait_zd", "wait_known", "sla_viol_critical",
                                                       "n_alerts", "fp_time_frac"]}})
means = write(pd.DataFrame(rows), "means")

# 2) paired differences (pooled and per group)
rows = []
for s in SC:
    for a, b in PAIRS:
        for grp in ["pooled"] + GR:
            gs = GR if grp == "pooled" else [grp]
            r = {**scen_cols(s), "comparison": f"{a} - {b}", "group": grp}
            for f in METRICS:
                m, lo, hi = ci(vals(s, a, f, gs) - vals(s, b, f, gs))
                r[f] = m; r[f + "_lo"] = lo; r[f + "_hi"] = hi
            rows.append(r)
diffs = write(pd.DataFrame(rows), "paired")

# 3) CZT-T and CZT-TE ablations
rows = []
for s in SC:
    for base in ("CZT-T", "CZT-TE"):
        for p in [base] + [k for k in EV[f"{s}|{GR[0]}"] if k.startswith(base + " ")]:
            r = {**scen_cols(s), "policy": p}
            for f in ("zcv", "cv", "kcv", "sla_viol_critical"):
                r[f] = np.nanmean(vals(s, p, f))
            if p != base:
                m, lo, hi = ci(vals(s, p, "zcv") - vals(s, base, "zcv"))
                r.update(dzcv=m, dzcv_lo=lo, dzcv_hi=hi)
            rows.append(r)
abl = write(pd.DataFrame(rows), "ablation")

# 4) tuned parameters (R1.4)
rows = []
for k, v in d["tuned"].items():
    s, g, kind = k.split("|")
    rows.append(dict(scenario=s, eval_group=g, tuned_on={"G1": "G2", "G2": "G1"}[g], policy=kind, q=v["q"],
                     params=json.dumps({a: (np.round(b, 4).tolist() if isinstance(b, list) else round(b, 4))
                                        for a, b in v["params"].items()}),
                     F_knee=json.dumps(np.round(v["F_knee"], 4).tolist()), front=v["front_size"]))
write(pd.DataFrame(rows).sort_values(["scenario", "policy", "eval_group"]), "tuned")

# console summary
pd.set_option("display.width", 220); pd.set_option("display.max_rows", 500)
key = diffs[(diffs.group == "pooled")][["scenario", "comparison", "zcv", "zcv_lo", "zcv_hi", "kcv", "cv"]]
print(key.round(3).to_string(index=False))
