"""Tables for the second-round robustness checks.

E18 (experiments/exp_detection_robust.py): additional detection comparators, feature filter, majority-label cleaning, P5 filter.
E17 (experiments/exp_triage_robust.py): triage robustness (no-novelty tuning, policy gradient with 91 checkpoints, optimiser seeds,
load and group control, alert precision, prototype-only novelty, prior-corrected scores).

Run from code/:  python -m analysis.robust_report
Writes tables/s_table_e18_*.csv, tables/s_table_e17_*.csv, results/e17_summary.json and prints a summary.
"""
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

MET = ["auroc_zd", "zdr", "tpr_fpr1_zd", "test_fpr"]
MLAB = {"auroc_zd": "AUROC", "zdr": "ZDR", "tpr_fpr1_zd": "TPR at 1% FPR", "test_fpr": "Test FPR"}
ANCH = {"original": "Original", "central": "Imperfect", "uninformative_0.5": "Uninformative"}


def ci(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return float(x.mean()) if len(x) else np.nan, np.nan, np.nan
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return x.mean(), x.mean() - h, x.mean() + h


def fmt_ci(x):
    m, lo, hi = ci(x)
    return f"{m:+.3f} [{lo:+.3f}, {hi:+.3f}]"


def verdict(x):
    m, lo, hi = ci(x)
    return "better" if lo > 0 else ("worse" if hi < 0 else "n.d.")


# ===================================================================== E18
if os.path.exists("results/e18_detection_robust.json"):
    e18 = json.load(open("results/e18_detection_robust.json"))
    rows = []
    for r in e18["results"]:
        for m, v in r["det"].items():
            if m.startswith("_"):
                continue
            rows.append(dict(kind=r["kind"], family=r["family"] or "none", model=m, **{k: v.get(k, np.nan) for k in MET}))
    R = pd.DataFrame(rows)
    R.to_csv("tables/s_table_e18_folds.csv", index=False)
    # P2 plus
    p = R[R.kind == "P2-plus"].groupby("model")[MET].mean()
    ref = pd.read_csv("tables/s_table_e10_refit_folds.csv")
    ref = ref[ref.variant == "full"].groupby("model")[MET].mean()
    order = ["IsolationForest", "Autoencoder", "EnergyMLP", "XGBoost-tuned", "LightGBM-tuned", "CLAD-200"]
    base = ["XGBoost", "LightGBM", "CZT-Hybrid", "CLAD"]
    T = pd.concat([ref.loc[[b for b in base if b in ref.index]].assign(Source="refit (main text)"),
                   p.loc[[o for o in order if o in p.index]].assign(Source="added")])
    T = T.reset_index().rename(columns={"model": "Detector", **MLAB})
    T = T[["Detector", "Source"] + list(MLAB.values())]
    nfam = R[R.kind == "P2-plus"].family.nunique()
    T.to_csv("tables/s_table_e18_plus.csv", index=False, float_format="%.3f")
    nov = [r["det"].get("_novelty_energy_auroc") for r in e18["results"] if r["kind"] == "P2-plus"]
    print(f"E18 P2-plus ({nfam} folds)\n", T.round(3).to_string(index=False))
    print("energy novelty AUROC per fold", np.round(nov, 3), "mean", np.nanmean([x for x in nov if x is not None]))
    # filtered and majority
    rows = []
    # the training-data feature filter removed no feature in any P2 fold (printed below), so its rows repeat the full-feature
    # detectors up to run-to-run variation of neural training and are not tabulated
    for kind, lab in (("full", "conflict removal (main text)"), ("P2-majority", "majority-label cleaning")):
        src = ref if kind == "full" else R[R.kind == kind].groupby("model")[MET].mean()
        for m in ["XGBoost", "LightGBM", "DNN", "ResSE-CNN1D", "B-ProtoMeta", "CZT-Hybrid"]:
            if m in src.index:
                rows.append({"Variant": lab, "Detector": m, **{MLAB[k]: src.loc[m, k] for k in MET}})
    F = pd.DataFrame(rows)
    F.to_csv("tables/s_table_e18_filtered.csv", index=False, float_format="%.3f")
    print(F.round(3).to_string(index=False))
    rem = {r["family"]: r.get("removed") for r in e18["results"] if r["kind"] == "P2-filtered"}
    print("P2 filter removed:", rem)
    if "majority_cleaning" in e18:
        print("majority cleaning rows", e18["majority_cleaning"]["rows"])
    # P5 filtered
    q = R[R.kind == "P5-filtered"]
    if len(q):
        rem5 = {r["family"] or "none": r.get("removed") for r in e18["results"] if r["kind"] == "P5-filtered"}
        rows = []
        for fam, g in q.groupby("family"):
            rr = {"Held-out family": fam, "Removed features": len(rem5.get(fam, []) or [])}
            for m in ["XGBoost", "LightGBM", "CZT-Hybrid", "CLAD"]:
                x = g[g.model == m]
                if len(x):
                    rr[f"{m} AUROC"] = x.auroc_zd.iloc[0] if fam != "none" else np.nan
                    rr[f"{m} test FPR"] = x.test_fpr.iloc[0]
            rows.append(rr)
        P5 = pd.DataFrame(rows)
        P5.to_csv("tables/s_table_e18_p5filtered.csv", index=False, float_format="%.3f")
        print(P5.round(3).to_string(index=False))
        print("P5 filter removed:", rem5)

# ===================================================================== E17
PATH = "results/e17_robust.json"
if os.path.exists(PATH):
    e17 = json.load(open(PATH))
    res = e17["results"]
    e9 = json.load(open("results/e9_triage.json"))
    GR = list(e9["meta"]["groups"])
    pa = pd.read_csv("tables/s_table_e9_paired.csv")
    infoauc = {(r.scenario, "flag"): r.inf_flag for r in pa.itertuples()} | {(r.scenario, "nov"): r.inf_nov for r in pa.itertuples()}

    def e9rep(s, pol, f):
        return np.concatenate([np.asarray(e9["eval"][f"{s}|{g}"]["reps"][pol][f], float) for g in GR])

    def e17rep(stage, s, kind, info, who, f, extra=None):
        out = []
        for g in GR:
            m = [r for r in res if r["stage"] == stage and r["scenario"] == s and r["group"] == g and r["kind"] == kind
                 and r["info"] == info and (extra is None or extra(r))]
            if not m:
                return None
            out.append(np.asarray(m[0]["eval"][who][f], float))
        return np.concatenate(out)

    # check that E17 evaluation streams equal those of E9 (common random numbers)
    chk = []
    for s in ANCH:
        x = e17rep("A", s, "Risk0", "flag", "Risk", "zcv")
        if x is not None:
            chk.append(float(np.max(np.abs(x - e9rep(s, "Risk", "zcv")))))
    print("max |Risk(E17) - Risk(E9)| over anchor settings:", chk)
    summary = dict(crn_check_max_abs_diff=chk)

    KMAP = {"CZT": ("CZT-T", "CZT0", "Index"), "Risk": ("Risk-T", "Risk0", "Tuned Risk"), "Linear": ("Linear-T", "Linear0", "Linear rule")}
    scen = e17.get("scenarios") or sorted({r["scenario"] for r in res if r["stage"] == "A"})
    # ---- A: gain of novelty information over the same policy tuned without it
    rows = []
    for s in scen:
        for k, (e9n, k0, lab) in KMAP.items():
            z0 = e17rep("A", s, k0, "flag", "tuned", "zcv")
            k0v = e17rep("A", s, k0, "flag", "tuned", "kcv")
            if z0 is None:
                continue
            r0 = e17rep("A", s, k0, "flag", "Risk", "zcv")
            for info in ("flag", "nov"):
                z1 = e9rep(s, f"{e9n}/{info}", "zcv")
                k1 = e9rep(s, f"{e9n}/{info}", "kcv")
                m, lo, hi = ci(z1 - z0)
                rows.append(dict(setting=s, policy=lab, info=info, info_auroc=infoauc.get((s, info), np.nan),
                                 dzcv=m, dzcv_lo=lo, dzcv_hi=hi, dkcv=float(np.mean(k1 - k0v)),
                                 nonov_vs_risk=float(np.mean(z0 - r0)), verdict=verdict(z1 - z0)))
    A = pd.DataFrame(rows)
    if len(A):
        A.to_csv("tables/s_table_e17_nonov_numeric.csv", index=False)
        cnt = A.groupby(["policy", "info"]).verdict.value_counts().unstack(fill_value=0)
        print("A: novelty-informed minus no-novelty tuned policy (ZCV), counts over settings\n", cnt)
        rho = {f"{p}/{i}": float(stats.spearmanr(g.info_auroc, g.dzcv).correlation) for (p, i), g in A.groupby(["policy", "info"]) if g.info_auroc.nunique() > 1}
        print("Spearman(info AUROC, gain over no-novelty):", {k: round(v, 3) for k, v in rho.items()})
        summary["A_counts"] = {f"{p}/{i}": {k: int(v) for k, v in row.items()} for (p, i), row in cnt.iterrows()}
        summary["A_spearman"] = rho
        summary["A_nonov_vs_risk"] = A.groupby("policy").nonov_vs_risk.describe()[["min", "50%", "max"]].round(3).to_dict("index")
        tab = A.assign(**{"Information": A["info"].map({"flag": "flag", "nov": "novelty score"}),
                          "Information AUROC": A.info_auroc.round(3),
                          "ZCV gain over no-novelty tuning": [f"{a:+.3f} [{b:+.3f}, {c:+.3f}]" for a, b, c in zip(A.dzcv, A.dzcv_lo, A.dzcv_hi)],
                          "KCV change": A.dkcv.round(3), "No-novelty tuned minus Risk (ZCV)": A.nonov_vs_risk.round(3)})
        tab = tab.rename(columns={"setting": "Setting", "policy": "Policy"})
        tab[["Setting", "Policy", "Information", "Information AUROC", "ZCV gain over no-novelty tuning", "KCV change",
             "No-novelty tuned minus Risk (ZCV)"]].to_csv("tables/s_table_e17_nonov.csv", index=False)

    # ---- B: policy gradient with 91 checkpoints
    rows = []
    for s in scen:
        for info in ("flag", "nov"):
            z = e17rep("B", s, "PG", info, "tuned", "zcv")
            if z is None:
                continue
            kc = e17rep("B", s, "PG", info, "tuned", "kcv")
            rk = e17rep("B", s, "PG", info, "Risk", "zcv")
            tr = e9rep(s, f"Risk-T/{info}", "zcv")
            pg10 = e9rep(s, f"PG-T/{info}", "zcv")
            rows.append(dict(setting=s, info=info, vs_risk=fmt_ci(z - rk), vs_tuned_risk=fmt_ci(z - tr), vs_pg10=fmt_ci(z - pg10),
                             v_risk=verdict(z - rk), v_tr=verdict(z - tr), v_pg10=verdict(z - pg10),
                             dkcv_risk=float(np.mean(kc - e17rep("B", s, "PG", info, "Risk", "kcv")))))
    B = pd.DataFrame(rows)
    if len(B):
        print("B: PG(91) vs tuned Risk\n", B.groupby("info").v_tr.value_counts().unstack(fill_value=0))
        print("B: PG(91) vs PG(10)\n", B.groupby("info").v_pg10.value_counts().unstack(fill_value=0))
        print("B: PG(91) vs Risk\n", B.groupby("info").v_risk.value_counts().unstack(fill_value=0))
        summary["B_vs_tuned_risk"] = {i: g.v_tr.value_counts().to_dict() for i, g in B.groupby("info")}
        summary["B_vs_pg10"] = {i: g.v_pg10.value_counts().to_dict() for i, g in B.groupby("info")}
        summary["B_vs_risk"] = {i: g.v_risk.value_counts().to_dict() for i, g in B.groupby("info")}
        B.rename(columns={"setting": "Setting", "info": "Information", "vs_risk": "PG(91) minus Risk, ZCV",
                          "vs_tuned_risk": "PG(91) minus tuned Risk, ZCV", "vs_pg10": "PG(91) minus PG(10), ZCV",
                          "dkcv_risk": "KCV change vs Risk"})[
            ["Setting", "Information", "PG(91) minus Risk, ZCV", "PG(91) minus tuned Risk, ZCV", "PG(91) minus PG(10), ZCV",
             "KCV change vs Risk"]].to_csv("tables/s_table_e17_pg.csv", index=False, float_format="%.3f")

    # ---- C: optimiser seeds (7 = main run)
    rows = []
    for s in ANCH:
        for k, lab in (("CZT", "Index"), ("Risk", "Tuned Risk"), ("Linear", "Linear rule"), ("PG", "Policy gradient (91 checkpoints)"),
                       ("CZT0", "Index, no novelty"), ("Risk0", "Tuned Risk, no novelty"), ("Linear0", "Linear, no novelty")):
            for info in (("flag", "nov") if not k.endswith("0") else ("flag",)):
                vals, idx = {}, {}
                for seed in (7, 11, 13, 17):
                    if seed == 7:
                        if k.endswith("0"):
                            z = e17rep("A", s, k, "flag", "tuned", "zcv")
                        elif k == "PG":
                            z = e17rep("B", s, "PG", info, "tuned", "zcv")
                        else:
                            z = e9rep(s, f"{k}-T/{info}", "zcv")
                    else:
                        z = e17rep("C", s, k, info, "tuned", "zcv", extra=lambda r, sd=seed: r["opt"].get("seed") == sd)
                    if z is None:
                        continue
                    vals[seed] = float(np.mean(z - e9rep(s, "Risk", "zcv")))
                if len(vals) < 2:
                    continue
                v = np.array(list(vals.values()))
                rows.append({"Setting": ANCH[s], "Policy": lab, "Information": info,
                             **{f"seed {sd}": round(x, 3) for sd, x in vals.items()},
                             "Range": round(float(v.max() - v.min()), 3)})
    C = pd.DataFrame(rows)
    if len(C):
        C.to_csv("tables/s_table_e17_seeds.csv", index=False)
        print("C: ZCV minus Risk by optimiser seed\n", C.to_string(index=False))
        # index minus tuned Risk per seed
        rows = []
        for s in ANCH:
            for info in ("flag", "nov"):
                for seed in (7, 11, 13, 17):
                    r = dict(setting=s, info=info, seed=seed)
                    for fld, pre in (("zcv", ""), ("kcv", "kcv_"), ("cv", "cv_")):
                        if seed == 7:
                            a, b = e9rep(s, f"CZT-T/{info}", fld), e9rep(s, f"Risk-T/{info}", fld)
                        else:
                            f = lambda r_, sd=seed: r_["opt"].get("seed") == sd
                            a, b = e17rep("C", s, "CZT", info, "tuned", fld, f), e17rep("C", s, "Risk", info, "tuned", fld, f)
                        if a is None or b is None:
                            break
                        m, lo, hi = ci(a - b)
                        r.update({pre + "d": m, pre + "lo": lo, pre + "hi": hi})
                    else:
                        rows.append(r)
        CI = pd.DataFrame(rows)
        CI.to_csv("tables/s_table_e17_seeds_index_vs_tuned.csv", index=False)
        print(CI.round(3).to_string(index=False))
        summary["C_index_minus_tuned"] = CI.round(4).to_dict("records")

    # ---- D: load and group control
    rows = []
    for s in ANCH:
        for k, lab in (("CZT", "Index"), ("Risk", "Tuned Risk"), ("Linear", "Linear rule"), ("PG", "Policy gradient (91 checkpoints)")):
            for info in ("flag", "nov"):
                main = e9rep(s, f"{k}-T/{info}", "zcv") if k != "PG" else e17rep("B", s, "PG", info, "tuned", "zcv")
                risk = e9rep(s, "Risk", "zcv")
                r = {"Setting": ANCH[s], "Policy": lab, "Information": info,
                     "Tuned at load 1.0, other group (main)": fmt_ci(main - risk) if main is not None else ""}
                mk = e9rep(s, f"{k}-T/{info}", "kcv") if k != "PG" else e17rep("B", s, "PG", info, "tuned", "kcv")
                r["Tuned at load 1.0, other group (main), KCV"] = round(float(np.mean(mk - e9rep(s, "Risk", "kcv"))), 3) if mk is not None else np.nan
                for st, nm in (("D-load", "Tuned at load 1.2, other group"), ("D-same", "Tuned at load 1.2, same group")):
                    z = e17rep(st, s, k, info, "tuned", "zcv")
                    kc = e17rep(st, s, k, info, "tuned", "kcv")
                    r[nm] = fmt_ci(z - risk) if z is not None else ""
                    r[nm + ", KCV"] = round(float(np.mean(kc - e9rep(s, "Risk", "kcv"))), 3) if kc is not None else np.nan
                rows.append(r)
    D = pd.DataFrame(rows)
    if len(D):
        D.to_csv("tables/s_table_e17_load.csv", index=False)
        print("D:\n", D.to_string(index=False))

    # ---- E: precision
    rows = []
    for b in ("0.995", "0.999"):
        st = f"E-{b}"
        for s in ANCH:
            rk = e17rep(st, s, "Risk0", "flag", "Risk", "zcv")
            if rk is None:
                continue
            up = e17rep(st, s, "Risk0", "flag", "upper", "zcv")
            fpf = e17rep(st, s, "Risk0", "flag", "Risk", "fp_time_frac")
            nal = e17rep(st, s, "Risk0", "flag", "Risk", "n_alerts")
            r = {"Benign share": b, "Setting": ANCH[s], "Risk ZCV": round(float(rk.mean()), 3), "Upper bound ZCV": round(float(up.mean()), 3),
                 "Alerts per shift": round(float(nal.mean()), 1), "Analyst time on false positives": round(float(fpf.mean()), 3)}
            for k, lab in (("CZT", "Index"), ("Risk", "Tuned Risk"), ("Linear", "Linear"), ("PG", "PG")):
                for info in ("flag", "nov"):
                    z = e17rep(st, s, k, info, "tuned", "zcv")
                    if z is not None:
                        r[f"{lab}/{info}"] = fmt_ci(z - rk)
            for k, lab in (("CZT0", "Index"), ("Risk0", "Tuned Risk"), ("Linear0", "Linear")):
                z = e17rep(st, s, k, "flag", "tuned", "zcv")
                if z is not None:
                    r[f"{lab}/none"] = fmt_ci(z - rk)
            rows.append(r)
    E_ = pd.DataFrame(rows)
    if len(E_):
        E_.to_csv("tables/s_table_e17_precision.csv", index=False)
        print("E:\n", E_.to_string(index=False))
        # index vs tuned Risk at each precision
        rows = []
        for b in ("0.995", "0.999"):
            for s in ANCH:
                for info in ("flag", "nov"):
                    a = e17rep(f"E-{b}", s, "CZT", info, "tuned", "zcv")
                    c = e17rep(f"E-{b}", s, "Risk", info, "tuned", "zcv")
                    c0 = e17rep(f"E-{b}", s, "Risk0", "flag", "tuned", "zcv")
                    if a is None or c is None:
                        continue
                    rows.append(dict(benign=b, setting=s, info=info, index_minus_tuned=fmt_ci(a - c), v=verdict(a - c),
                                     info_minus_noinfo_risk=fmt_ci(c - c0) if c0 is not None else ""))
        EI = pd.DataFrame(rows)
        print(EI.to_string(index=False))
        EI.to_csv("tables/s_table_e17_precision_index_vs_tuned.csv", index=False)

    # ---- F: prototype-only novelty; G: prior-corrected scores
    rows = []
    for s in ANCH:
        for k, lab in (("CZT", "Index"), ("Risk", "Tuned Risk"), ("Linear", "Linear rule"), ("PG", "Policy gradient (91 checkpoints)")):
            z = e17rep("F", s, k, "proto", "tuned", "zcv")
            if z is None:
                continue
            base = e9rep(s, f"{k}-T/nov", "zcv") if k != "PG" else e17rep("B", s, "PG", "nov", "tuned", "zcv")
            rows.append({"Setting": ANCH[s], "Policy": lab, "Proto score minus Risk": fmt_ci(z - e9rep(s, "Risk", "zcv")),
                         "Combined score minus Risk": fmt_ci(base - e9rep(s, "Risk", "zcv")) if base is not None else "",
                         "Proto minus combined": fmt_ci(z - base) if base is not None else ""})
    F_ = pd.DataFrame(rows)
    if len(F_):
        F_.to_csv("tables/s_table_e17_proto.csv", index=False)
        print("F:\n", F_.to_string(index=False))
    rows = []
    for s in ANCH:
        for k, lab in (("CZT", "Index"), ("Risk", "Tuned Risk")):
            for info in ("flag", "nov"):
                z = e17rep("G", s, k, info, "tuned", "zcv")
                if z is None:
                    continue
                rk = e17rep("G", s, k, info, "Risk", "zcv")
                rows.append({"Setting": ANCH[s], "Policy": lab, "Information": info,
                             "Tuned on corrected scores minus Risk (corrected scores)": fmt_ci(z - rk),
                             "Tuned on corrected scores minus tuned on uncorrected": fmt_ci(z - e9rep(s, f"{k}-T/{info}", "zcv"))})
    G_ = pd.DataFrame(rows)
    if len(G_):
        G_.to_csv("tables/s_table_e17_prior.csv", index=False)
        print("G:\n", G_.to_string(index=False))
    json.dump(summary, open("results/e17_summary.json", "w"), indent=1, default=float)
