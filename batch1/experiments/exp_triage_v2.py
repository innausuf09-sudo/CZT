"""E9 (revision): triage with a label-free novelty score, a pre-declared selection rule,
tuned baselines, a learned scheduler, full trade-off fronts and simulator sensitivity.

Pre-declared selection rule (fixed in this file before any evaluation run, applied
identically to every tuned policy, using tuning streams only):
    among the candidate parameter vectors (the NSGA-III front, or the policy-gradient
    checkpoints), choose the one with the largest ZCV subject to
      (a) critical SLA misses <= 0.02 and
      (b) known-attack captured value (KCV) >= KCV of untuned Risk - 0.02,
    where both constraints are measured on the same tuning streams. If no candidate is
    feasible, choose the one with the smallest total constraint violation (ties: larger ZCV).

Information conditions
    flag  the catalogue no-match flag decides where the imputed severity s_u is used (as in E8)
    nov   the label-free novelty score in [0, 1] (results/e10_refit, 'combined') replaces the
          flag: s_idx = (1 - nov) * s_obs + nov * s_u; the linear rule and the learned
          scheduler use nov in place of the flag.
The escalation threshold is fixed at q = 0.99 for every policy, so all policies of one
replication face the same alerts.

Detector scores: refitted CZT-Hybrid on the original P2 splits (results/e10_refit/full_*.npz).
Tuning: tuning half of the records of the other family group, rho = 1.0, 8 streams (seeds 5000+).
Evaluation: evaluation half, rho = 1.2, c = 4, 20 replications per group (seeds 1000+), common
random numbers for every policy. NSGA-III: 91 reference directions (3 objectives, 12
partitions), 60 generations, seed 7, SBX(eta 30, p 1.0, per-variable 0.5), PM(eta 20, p 0.9).
Objectives (minimised): 1 - ZCV, 1 - KCV, critical SLA misses. REINFORCE: same number of
simulated shifts (91 x 60 x 8), seed 7.

Run from code/:
  python -m experiments.exp_triage_v2                    # main (G1/G2 split, 37 scenarios, sensitivity)
  python -m experiments.exp_triage_v2 --partition K      # random family partition K (R2.6)
  add --quick for a smoke test
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
from pymoo.util.ref_dirs import get_reference_directions

from experiments.exp_triage_catalogue import SCEN as SCEN_ALL
from zdt.catalogue import Pool, alerts, observe, raw_events, run_policy
from zdt.data import UNSW_FAMILIES, UNSW_SEVERITY
from zdt.pg import train_pg
from zdt.triage import DEFAULT, Stream

ARGS = sys.argv[1:]
QUICK = "--quick" in ARGS
PART = int(ARGS[ARGS.index("--partition") + 1]) if "--partition" in ARGS else None
C0, RHO_TUNE, RHO_EVAL, Q0 = 4, 1.0, 1.2, 0.99
REPS = 4 if QUICK else 20
N_TUNE, N_GEN, SEED_NSGA = 8, (5 if QUICK else 60), 7
SEED_TUNE, SEED_EVAL = 5000, 1000
REF = get_reference_directions("das-dennis", 3, n_partitions=12)
POP = len(REF)
PG_SHIFTS = POP * N_GEN * N_TUNE
SLA_MAX, KCV_LOSS = 0.02, 0.02
NOV_KEY = "combined"

if PART is None:
    GROUPS = {"G1": ["Generic", "Fuzzers", "Reconnaissance", "Backdoor", "Worms"],
              "G2": ["Exploits", "DoS", "Analysis", "Shellcode"]}
else:  # random partition of the nine families into groups of five and four (seed = 100 + K)
    perm = list(np.random.default_rng(100 + PART).permutation(UNSW_FAMILIES))
    GROUPS = {"G1": sorted(perm[:5]), "G2": sorted(perm[5:])}
OTHER = {"G1": "G2", "G2": "G1"}
if PART is not None:
    SCEN = {k: SCEN_ALL[k] for k in ("original", "central", "uninformative_0.5")}
elif QUICK:
    SCEN = {k: SCEN_ALL[k] for k in ("original", "central")}
else:
    SCEN = SCEN_ALL
TAG = ("_quick" if QUICK else "") + (f"_part{PART}" if PART is not None else "")
OUT = f"results/e9_triage{TAG}.json"

KINDS = {  # same ranges as E7 / E8; q fixed at Q0
    "CZT": dict(policy="Whittle", names=["kappa", "gamma", "beta", "s_u"], xl=[0.0, 0.25, 0.0, 0.3], xu=[1.0, 4.0, 2.0, 1.0]),
    "Risk": dict(policy="RiskT", names=["gamma", "beta", "s_u"], xl=[0.25, 0.0, 0.3], xu=[4.0, 2.0, 1.0]),
    "Linear": dict(policy="Linear", names=["w"] * 5, xl=[-1.0] * 5, xu=[1.0] * 5),
}
INFOS = ("flag", "nov")


def params_of(kind, x, info):
    if kind in ("Linear", "PG"):
        return {"w": [float(v) for v in x], "info": info}
    return {**{n: float(v) for n, v in zip(KINDS[kind]["names"], x)}, "info": info}


def policy_of(kind):
    return "Linear" if kind == "PG" else KINDS[kind]["policy"]


# ------------------------------------------------------------------ detector store from the refit
_S = {}


def load_refit():
    store = {}
    for f in UNSW_FAMILIES:
        z = np.load(f"results/e10_refit/full_{f}.npz")
        pv, yv = z["CZT-Hybrid|p_va"].astype(float), z["y_va"]
        pi_v = yv.mean()
        p = z["CZT-Hybrid|p"].astype(float)
        pi_d = 1 - DEFAULT["mix"][0]
        odds = p / np.clip(1 - p, 1e-9, None) * (pi_d / (1 - pi_d)) / (pi_v / (1 - pi_v))
        store[f] = {"groups": z["groups"].astype(str), "fam": z["fam"].astype(str), "nov": z[f"nov|{NOV_KEY}"],
                    "p_prior": odds / (1 + odds),
                    "scores": {"CZT-Hybrid": {"p": p, "pu": z["CZT-Hybrid|pu"], "pva_normal": z["CZT-Hybrid|pva_normal"]}}}
    return store


def pools():
    if not _S:
        store = load_refit()
        for g in GROUPS:
            for half, key in ((0, "TU"), (1, "EV")):
                st = Stream(store, "CZT-Hybrid", UNSW_SEVERITY, half=half, folds=GROUPS[g])
                ex = {k: Stream(store, "CZT-Hybrid", UNSW_SEVERITY, pscore=(lambda kk: lambda s: s[kk])(k), half=half,
                                folds=GROUPS[g]).p for k in ("nov", "p_prior")}
                _S[(key, g)] = Pool(st, UNSW_SEVERITY, extra=ex)
    return _S


def cat_of(sc):
    return {k: sc[k] for k in ("m_k", "m_z", "m_b", "sev_bias", "sev_noise")}


def raw_rate(pool, rho, mix, c=C0):
    R = raw_events(pool, 40000 / DEFAULT["horizon"], 0, mix=mix)
    A = alerts(pool, R, observe(pool, R), Q0)
    return rho * c / A["svc"].mean() / (len(A["t"]) / len(R["t"]))


def streams(pool, sc, rho, seeds, cfg=DEFAULT):
    raw = raw_rate(pool, rho, sc["mix"])
    out = []
    for s in seeds:
        R = raw_events(pool, raw, s, cfg=cfg, mix=sc["mix"])
        out.append(alerts(pool, R, observe(pool, R, cat_of(sc), cfg=cfg), Q0))
    return out


def outcome(As, policy, par, cfg=DEFAULT, c=C0):
    return [run_policy(A, c, policy, par, cfg) for A in As]


def mean3(res):
    m = lambda f: float(np.nanmean([r[f] for r in res]))
    return m("zcv"), m("kcv"), m("sla_viol_critical")


def select(cands, kcv_ref):
    """Pre-declared rule; cands = list of (zcv, kcv, crit)."""
    C = np.array(cands, float)
    C[:, 2] = np.nan_to_num(C[:, 2])
    feas = (C[:, 2] <= SLA_MAX) & (C[:, 1] >= kcv_ref - KCV_LOSS)
    if feas.any():
        k = int(np.flatnonzero(feas)[np.argmax(C[feas, 0])])
    else:
        viol = np.maximum(0, C[:, 2] - SLA_MAX) + np.maximum(0, kcv_ref - KCV_LOSS - C[:, 1])
        best = np.flatnonzero(viol <= viol.min() + 1e-12)
        k = int(best[np.argmax(C[best, 0])])
    return k, bool(feas.any()), int(feas.sum())


class Problem(ElementwiseProblem):
    def __init__(self, kind, info, As):
        sp = KINDS[kind]
        super().__init__(n_var=len(sp["xl"]), n_obj=3, xl=np.array(sp["xl"]), xu=np.array(sp["xu"]))
        self.kind, self.info, self.As = kind, info, As

    def _evaluate(self, x, out, *a, **k):
        z, kc, cr = mean3(outcome(self.As, KINDS[self.kind]["policy"], params_of(self.kind, x, self.info)))
        out["F"] = [1 - z, 1 - kc, 0.0 if np.isnan(cr) else cr]


def tune(task):
    scen, g_eval, kind, info = task
    sc = SCEN[scen]
    As = streams(pools()[("TU", OTHER[g_eval])], sc, RHO_TUNE, range(SEED_TUNE, SEED_TUNE + N_TUNE))
    kcv_ref = mean3(outcome(As, "Risk", {}))[1]
    t0 = time.time()
    if kind == "PG":
        X = train_pg(As, C0, PG_SHIFTS, {"info": info}, seed=SEED_NSGA)
        cands = [mean3(outcome(As, "Linear", params_of("PG", x, info))) for x in X]
        X = [list(map(float, x)) for x in X]
    else:
        alg = NSGA3(pop_size=POP, ref_dirs=REF, crossover=SBX(eta=30, prob=1.0, prob_var=0.5), mutation=PM(eta=20, prob=0.9))
        res = minimize(Problem(kind, info, As), alg, ("n_gen", N_GEN), seed=SEED_NSGA, verbose=False)
        Fr, X = np.atleast_2d(res.F), np.atleast_2d(res.X)
        cands = [(1 - f[0], 1 - f[1], f[2]) for f in Fr]
        X = X.tolist()
    k, feasible, n_feas = select(cands, kcv_ref)
    return task, dict(X=X, tune_points=[list(map(float, c)) for c in cands], selected=k, feasible=feasible,
                      n_feasible=n_feas, kcv_ref_tune=kcv_ref, params=params_of(kind, X[k], info), time_s=time.time() - t0)


KEEP = ("cv", "zcv", "kcv", "wait_zd", "wait_known", "sla_viol", "sla_viol_critical", "n_alerts", "fp_time_frac")
TKINDS = ("CZT", "Risk", "Linear", "PG")


def eval_policies(tuned, tuned_orig):
    P = {"FIFO": ("FIFO", {}), "Severity": ("Severity", {}), "EDF": ("EDF", {}), "Risk": ("Risk", {}),
         "CZT-default": ("Whittle", {"kappa": 0.1, "gamma": 1.0, "beta": 1.0, "s_u": 0.5}),
         "Risk[prior]": ("Risk", {"_p": "p_prior"})}
    for info in INFOS:
        for kind in TKINDS:
            P[f"{kind}-T/{info}"] = (policy_of(kind), tuned[(kind, info)]["params"])
            P[f"{kind}-T/{info}[orig]"] = (policy_of(kind), tuned_orig[(kind, info)]["params"])
        cz = tuned[("CZT", info)]["params"]
        for lab, mod in (("no severity (beta=0)", dict(beta=0.0)), ("no score (gamma=0)", dict(gamma=0.0)),
                         ("no decay", dict(decay=False)), ("no deadline (kappa=0)", dict(kappa=0.0)),
                         ("no imputation (s_u=0.5)", dict(s_u=0.5))):
            P[f"CZT-T/{info} {lab}"] = ("Whittle", {**cz, **mod})
        P[f"CZT-T/{info}[prior]"] = ("Whittle", {**cz, "_p": "p_prior"})
    return P


def run_named(A, pol, par, cfg=DEFAULT, c=C0):
    if "_p" in par:
        A = dict(A); A["p"] = A[par["_p"]]
        par = {k: v for k, v in par.items() if k != "_p"}
    return run_policy(A, c, pol, par, cfg)


def evaluate(task):
    scen, g, tuned, tuned_orig = task
    sc = SCEN[scen]
    As = streams(pools()[("EV", g)], sc, RHO_EVAL, range(SEED_EVAL, SEED_EVAL + REPS))
    P = eval_policies(tuned, tuned_orig)
    out = {name: {f: [] for f in KEEP} for name in P}
    for A in As:
        for name, (pol, par) in P.items():
            r = run_named(A, pol, par)
            for f in KEEP:
                out[name][f].append(float(r[f]))
    # out-of-sample evaluation of every candidate on the fronts (trade-off figure)
    fronts = {}
    for (kind, info), t in tuned.items():
        pts = []
        for x in t["X"]:
            res = outcome(As, policy_of(kind), params_of(kind, x, info))
            pts.append(list(mean3(res)))
        fronts[f"{kind}/{info}"] = pts
    return (scen, g), dict(reps=out, fronts=fronts)


SENS = {  # simulator-parameter sensitivity (no re-tuning)
    "rho0.8": dict(rho=0.8), "rho1.0": dict(rho=1.0), "rho1.5": dict(rho=1.5),
    "c3": dict(c=3), "c6": dict(c=6),
    "halflife30": dict(cfg=dict(half_life=30.0)), "halflife120": dict(cfg=dict(half_life=120.0)),
    "sla_x0.5": dict(cfg=dict(sla=((0.85, 7.5), (0.65, 30.0), (0.40, 120.0), (0.0, 720.0)))),
    "sla_x2": dict(cfg=dict(sla=((0.85, 30.0), (0.65, 120.0), (0.40, 480.0), (0.0, 2880.0)))),
    "svc_attack20": dict(cfg=dict(svc_attack=(20.0, 6.0))),
    "benign0.95": dict(mix=(0.95, 0.0333, 0.0167)), "benign0.99": dict(mix=(0.99, 0.00667, 0.00333)),
}
SENS_SCEN = ("original", "central", "uninformative_0.5")


def sensitivity(task):
    scen, g, tuned, sname = task
    sp = SENS[sname]
    sc = dict(SCEN[scen]); sc["mix"] = sp.get("mix", sc["mix"])
    cfg = dict(DEFAULT); cfg.update(sp.get("cfg", {}))
    c = sp.get("c", C0)
    pool = pools()[("EV", g)]
    raw = raw_rate(pool, sp.get("rho", RHO_EVAL), sc["mix"])
    P = {"Risk": ("Risk", {})}
    for info in INFOS:
        for kind in TKINDS:
            P[f"{kind}-T/{info}"] = (policy_of(kind), tuned[(kind, info)]["params"])
    out = {n: {f: [] for f in KEEP} for n in P}
    for s in range(SEED_EVAL, SEED_EVAL + REPS):
        R = raw_events(pool, raw, s, cfg=cfg, mix=sc["mix"])
        A = alerts(pool, R, observe(pool, R, cat_of(sc), cfg=cfg), Q0)
        for n, (pol, par) in P.items():
            r = run_policy(A, c, pol, par, cfg)
            for f in KEEP:
                out[n][f].append(float(r[f]))
    return (scen, g, sname), out


def main():
    t0 = time.time()
    nproc = max(1, (os.cpu_count() or 2) - 2)
    tasks = [(s, g, k, i) for s in SCEN for g in GROUPS for k in TKINDS for i in INFOS]
    tasks.sort(key=lambda t: t[2] != "PG")  # slowest first
    print(f"{len(SCEN)} scenarios, groups {GROUPS}, {len(tasks)} tuning runs, {nproc} processes", flush=True)
    with MPPool(nproc) as mp:
        tuned = {}
        for i, (task, r) in enumerate(mp.imap_unordered(tune, tasks)):
            tuned[task] = r
            if i % 20 == 0:
                print(f"  tuned {i + 1}/{len(tasks)} {task} feasible={r['feasible']} [{time.time() - t0:.0f}s]", flush=True)
        T = lambda s, g: {(k, i): tuned[(s, g, k, i)] for k in TKINDS for i in INFOS}
        ev = dict(mp.map(evaluate, [(s, g, T(s, g), T("original", g)) for s in SCEN for g in GROUPS]))
        print(f"  evaluated [{time.time() - t0:.0f}s]", flush=True)
        sens = {}
        if PART is None:
            sens = dict(mp.map(sensitivity, [(s, g, T(s, g), n) for s in SENS_SCEN if s in SCEN for g in GROUPS for n in SENS]))
    res = {"meta": dict(groups=GROUPS, partition=PART, reps=REPS, rho_eval=RHO_EVAL, rho_tune=RHO_TUNE, c=C0, q=Q0,
                        n_tune=N_TUNE, n_gen=N_GEN, pop=POP, nsga_seed=SEED_NSGA, pg_shifts=PG_SHIFTS,
                        seed_tune=SEED_TUNE, seed_eval=SEED_EVAL, novelty=NOV_KEY, sla_max=SLA_MAX, kcv_loss=KCV_LOSS,
                        sbx=dict(eta=30, prob=1.0, prob_var=0.5), pm=dict(eta=20, prob=0.9, prob_var="1/n_var"),
                        bounds={k: dict(xl=v["xl"], xu=v["xu"]) for k, v in KINDS.items()},
                        objectives=["1 - ZCV", "1 - KCV", "critical SLA misses"], time_s=time.time() - t0),
           "scenarios": SCEN, "sens_settings": SENS,
           "tuned": {"|".join(k): v for k, v in tuned.items()},
           "eval": {"|".join(k): v for k, v in ev.items()},
           "sens": {"|".join(k): v for k, v in sens.items()}}
    json.dump(res, open(OUT, "w"))
    print(f"done in {time.time() - t0:.0f}s -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
