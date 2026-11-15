"""E17 (revision, second round): robustness checks of the E9 triage comparison.

Stages (all use the E9 machinery, selection rule and budget; see experiments/exp_triage_v2.py):
  A  no-novelty baselines: the index, tuned Risk and the linear rule tuned WITHOUT novelty input
     (imputed severity fixed at the default 0.5, linear weight on the novelty feature fixed at 0),
     in all 35 distinct catalogue settings, to separate the gain of novelty information from the gain of tuning.
  B  policy gradient with as many candidates as NSGA-III (91 checkpoints), all 35 settings, both information conditions.
  C  optimiser-seed variability: three further seeds (11, 13, 17) for every tuned policy in the three anchor settings.
  D  load and group control: tuning at the evaluation load 1.2 on the other family group, and at 1.2 on the
     tuning half of the SAME family group, in the anchor settings.
  E  alert precision: benign event shares 0.995 and 0.999 (about 75% and 94% false positives among alerts),
     unseen share of attacks kept at one third, all policies re-tuned at that mix, anchor settings.
  F  prototype-distance novelty score (single component) instead of the combined score, anchor settings.
  G  prior-corrected detector scores used for tuning and evaluation (index and tuned Risk), anchor settings.
Every task stores the evaluation replications of the tuned policy, of untuned Risk (same scores and mix) and the
upper bound obtained when every escalated alert is served on arrival.

Run from code/:  python -m experiments.exp_triage_robust [--quick]
Output: results/e17_robust.json
"""
import json
import os
import sys
import time
from multiprocessing import Pool as MPPool

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.core.problem import ElementwiseProblem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize

import experiments.exp_triage_v2 as E
from zdt.catalogue import Pool, alerts, observe, raw_events, run_policy
from zdt.data import UNSW_SEVERITY, UNSW_FAMILIES
from zdt.pg import train_pg
from zdt.triage import DEFAULT, Stream

QUICK = "--quick" in sys.argv
N_GEN = 5 if QUICK else E.N_GEN
REPS = 4 if QUICK else E.REPS
OUT = "results/e17_robust" + ("_quick" if QUICK else "") + ".json"
DUP = {"mk1.0_mz0.0_mb0.5": "original", "mk0.8_mz0.2_mb0.5": "central"}  # identical settings, counted once
SCEN = {k: v for k, v in E.SCEN_ALL.items() if k not in DUP}
ANCH = ("original", "central", "uninformative_0.5")
if QUICK:
    SCEN = {k: SCEN[k] for k in ANCH}
KINDS = dict(E.KINDS)
KINDS["CZT0"] = dict(policy="Whittle", names=["kappa", "gamma", "beta"], xl=[0.0, 0.25, 0.0], xu=[1.0, 4.0, 2.0])
KINDS["Risk0"] = dict(policy="RiskT", names=["gamma", "beta"], xl=[0.25, 0.0], xu=[4.0, 2.0])
KINDS["Linear0"] = dict(policy="Linear", names=["w"] * 4, xl=[-1.0] * 4, xu=[1.0] * 4)


def params_of(kind, x, info):
    x = [float(v) for v in x]
    if kind == "Linear0":
        return {"w": x[:2] + [0.0] + x[2:], "info": "flag"}
    if kind in ("CZT0", "Risk0"):
        return {**{n: v for n, v in zip(KINDS[kind]["names"], x)}, "s_u": 0.5, "info": "flag"}
    if kind in ("Linear", "PG"):
        return {"w": x, "info": "nov" if info != "flag" else "flag"}
    return {**{n: v for n, v in zip(KINDS[kind]["names"], x)}, "info": "nov" if info != "flag" else "flag"}


def policy_of(kind):
    return "Linear" if kind in ("PG", "Linear0") else KINDS[kind]["policy"]


_P = {}


def pools():
    if not _P:
        store = E.load_refit()
        for f in UNSW_FAMILIES:
            z = np.load(f"results/e10_refit/full_{f}.npz")
            store[f]["nov_proto"] = z["nov|proto"]
        for g in E.GROUPS:
            for half, key in ((0, "TU"), (1, "EV")):
                st = Stream(store, "CZT-Hybrid", UNSW_SEVERITY, half=half, folds=E.GROUPS[g])
                ex = {k: Stream(store, "CZT-Hybrid", UNSW_SEVERITY, pscore=(lambda kk: lambda s: s[kk])(k), half=half,
                                folds=E.GROUPS[g]).p for k in ("nov", "p_prior", "nov_proto")}
                _P[(key, g)] = Pool(st, UNSW_SEVERITY, extra=ex)
    return _P


def make_streams(pool, sc, rho, seeds, opt, info="flag"):
    mix = opt.get("mix", sc["mix"])
    raw = E.raw_rate(pool, rho, mix)
    out = []
    for s in seeds:
        R = raw_events(pool, raw, s, mix=mix)
        A = alerts(pool, R, observe(pool, R, E.cat_of(sc)), E.Q0)
        if info == "proto":
            A["nov"] = A["nov_proto"]
        if opt.get("prior"):
            A["p"] = A["p_prior"]
        out.append(A)
    return out


def mean3(res):
    m = lambda f: float(np.nanmean([r[f] for r in res]))
    return m("zcv"), m("kcv"), m("sla_viol_critical")


class Problem(ElementwiseProblem):
    def __init__(self, kind, info, As):
        sp = KINDS[kind]
        super().__init__(n_var=len(sp["xl"]), n_obj=3, xl=np.array(sp["xl"]), xu=np.array(sp["xu"]))
        self.kind, self.info, self.As = kind, info, As

    def _evaluate(self, x, out, *a, **k):
        z, kc, cr = mean3([run_policy(A, E.C0, policy_of(self.kind), params_of(self.kind, x, self.info)) for A in self.As])
        out["F"] = [1 - z, 1 - kc, 0.0 if np.isnan(cr) else cr]


KEEP = ("zcv", "kcv", "cv", "sla_viol_critical", "wait_zd", "n_alerts", "fp_time_frac")


def task_fn(task):
    stage, scen, g, kind, info, opt = task
    sc = E.SCEN_ALL[scen]
    P = pools()
    tg = g if opt.get("same_group") else E.OTHER[g]
    seed = opt.get("seed", E.SEED_NSGA)
    As = make_streams(P[("TU", tg)], sc, opt.get("rho_tune", E.RHO_TUNE), range(E.SEED_TUNE, E.SEED_TUNE + E.N_TUNE), opt, info)
    kcv_ref = mean3([run_policy(A, E.C0, "Risk", {}) for A in As])[1]
    t0 = time.time()
    if kind == "PG":
        n_shifts = E.POP * N_GEN * E.N_TUNE
        X = train_pg(As, E.C0, n_shifts, {"info": "nov" if info != "flag" else "flag"}, seed=seed, n_ckpt=opt.get("n_ckpt", 10))
        cands = [mean3([run_policy(A, E.C0, "Linear", params_of("PG", x, info)) for A in As]) for x in X]
        X = [list(map(float, x)) for x in X]
    else:
        alg = NSGA3(pop_size=E.POP, ref_dirs=E.REF, crossover=SBX(eta=30, prob=1.0, prob_var=0.5), mutation=PM(eta=20, prob=0.9))
        res = minimize(Problem(kind, info, As), alg, ("n_gen", N_GEN), seed=seed, verbose=False)
        Fr, X = np.atleast_2d(res.F), np.atleast_2d(res.X)
        cands = [(1 - f[0], 1 - f[1], f[2]) for f in Fr]
        X = X.tolist()
    k, feas, nf = E.select(cands, kcv_ref)
    par = params_of(kind, X[k], info)
    Ev = make_streams(P[("EV", g)], sc, E.RHO_EVAL, range(E.SEED_EVAL, E.SEED_EVAL + REPS), opt, info)
    out = {"tuned": {f: [] for f in KEEP}, "Risk": {f: [] for f in KEEP}, "upper": {f: [] for f in KEEP}}
    for A in Ev:
        for name, (pol, pr, c) in (("tuned", (policy_of(kind), par, E.C0)), ("Risk", ("Risk", {}, E.C0)), ("upper", ("FIFO", {}, 10000))):
            r = run_policy(A, c, pol, pr)
            for f in KEEP:
                out[name][f].append(float(r[f]))
    return task, dict(params=par, feasible=feas, n_feasible=nf, n_candidates=len(cands), tune_point=list(map(float, cands[k])),
                      kcv_ref_tune=kcv_ref, eval=out, time_s=time.time() - t0)


def tasks():
    T = []
    for s in SCEN:
        for g in E.GROUPS:
            for k in ("CZT0", "Risk0", "Linear0"):
                T.append(("A", s, g, k, "flag", {}))
            for info in E.INFOS:
                T.append(("B", s, g, "PG", info, {"n_ckpt": E.POP}))
    for s in ANCH:
        for g in E.GROUPS:
            for seed in (11, 13, 17):
                for k in ("CZT", "Risk", "Linear", "PG"):
                    for info in E.INFOS:
                        T.append(("C", s, g, k, info, {"seed": seed, "n_ckpt": E.POP}))
                for k in ("CZT0", "Risk0", "Linear0"):
                    T.append(("C", s, g, k, "flag", {"seed": seed}))
            for k in ("CZT", "Risk", "Linear", "PG"):
                for info in E.INFOS:
                    T.append(("D-load", s, g, k, info, {"rho_tune": 1.2, "n_ckpt": E.POP}))
                    T.append(("D-same", s, g, k, info, {"rho_tune": 1.2, "same_group": True, "n_ckpt": E.POP}))
                    for b, mix in ((0.995, (0.995, 0.005 * 2 / 3, 0.005 / 3)), (0.999, (0.999, 0.001 * 2 / 3, 0.001 / 3))):
                        T.append((f"E-{b}", s, g, k, info, {"mix": mix, "n_ckpt": E.POP}))
                T.append(("F", s, g, k, "proto", {"n_ckpt": E.POP}))
            for b, mix in ((0.995, (0.995, 0.005 * 2 / 3, 0.005 / 3)), (0.999, (0.999, 0.001 * 2 / 3, 0.001 / 3))):
                for k in ("CZT0", "Risk0", "Linear0"):
                    T.append((f"E-{b}", s, g, k, "flag", {"mix": mix}))
            for k in ("CZT", "Risk"):
                for info in E.INFOS:
                    T.append(("G", s, g, k, info, {"prior": True}))
    T.sort(key=lambda t: t[3] != "PG")
    return T


def main():
    t0 = time.time()
    T = tasks()
    nproc = max(1, (os.cpu_count() or 2) - 2)
    print(f"{len(T)} tasks, {nproc} processes", flush=True)
    res = []
    with MPPool(nproc) as mp:
        for i, (task, r) in enumerate(mp.imap_unordered(task_fn, T)):
            res.append(dict(stage=task[0], scenario=task[1], group=task[2], kind=task[3], info=task[4],
                            opt={k: (list(v) if isinstance(v, tuple) else v) for k, v in task[5].items()}, **r))
            if i % 25 == 0:
                print(f"  {i + 1}/{len(T)} {task[:5]} [{time.time() - t0:.0f}s]", flush=True)
                json.dump(dict(partial=True, results=res), open(OUT, "w"))
    json.dump(dict(partial=False, duplicates_removed=DUP, scenarios=list(SCEN), results=res,
                   meta=dict(reps=REPS, n_gen=N_GEN, pop=E.POP, time_s=time.time() - t0)), open(OUT, "w"))
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
