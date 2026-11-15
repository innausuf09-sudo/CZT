"""E8 (revision): catalogue observation model, tuned baselines and CZT-T ablations.

Addresses R1.2 / R2.2 (sensitivity to the catalogue mechanism and novelty
prevalence), R1.3 / R2.1 (tuned Risk and tuned linear baselines with the same
NSGA-III budget, ablations of CZT-T, common random numbers) and part of R2.6
(known-attack outcomes, group-specific paired effects).

Design
- Detector scores: stored P2 family hold-out scores of CZT-Hybrid (results/e2_scores.npz).
- Leakage-averse cross-fitting as in E7: policies are tuned on the tuning half of
  the records of one family group and evaluated on the evaluation half of the other.
- Each catalogue scenario is a separate world. In each, every tunable policy is
  tuned in that scenario with the same NSGA-III budget (84 reference directions,
  60 generations, seed 7, same objectives and knee rule as E7).
- "transfer" policies use the parameters tuned in the original scenario and are
  evaluated in every scenario (robustness to a misspecified catalogue model).
- Evaluation at offered load rho = 1.2, c = 4 analysts, 20 replications per group.
  Per-replication outcomes are stored so that differences can be paired.

Run from code/:  python -m experiments.exp_triage_catalogue [--quick]
Output: results/e8_catalogue.json
"""
import itertools
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

from zdt.catalogue import CAT_BASE, Pool, alerts, observe, raw_events, run_policy
from zdt.data import UNSW_SEVERITY
from zdt.io import load_scores
from zdt.triage import DEFAULT, Stream

QUICK = "--quick" in sys.argv
OUT = "results/e8_catalogue.json"
C0, RHO_TUNE, RHO_EVAL = 4, 1.0, 1.2
REPS = 5 if QUICK else 20
N_TUNE_SETS, N_GEN, NSGA_SEED = 8, (10 if QUICK else 60), 7
SEED_TUNE, SEED_EVAL = 5000, 1000
GROUPS = {"G1": ["Generic", "Fuzzers", "Reconnaissance", "Backdoor", "Worms"],
          "G2": ["Exploits", "DoS", "Analysis", "Shellcode"]}
OTHER = {"G1": "G2", "G2": "G1"}
MIX_BASE = (0.97, 0.02, 0.01)

# ------------------------------------------------------------------ scenarios
SCEN = {"original": dict(CAT_BASE, mix=MIX_BASE)}
for mk, mz, mb in itertools.product([0.6, 0.8, 1.0], [0.0, 0.2, 0.4], [0.2, 0.5, 0.8]):
    SCEN[f"mk{mk}_mz{mz}_mb{mb}"] = dict(m_k=mk, m_z=mz, m_b=mb, sev_bias=0.0, sev_noise=0.05, mix=MIX_BASE)
CENTRAL = dict(m_k=0.8, m_z=0.2, m_b=0.5, sev_noise=0.05, mix=MIX_BASE)
SCEN["uninformative_0.5"] = dict(m_k=0.5, m_z=0.5, m_b=0.5, sev_bias=0.0, sev_noise=0.05, mix=MIX_BASE)
SCEN["uninformative_0.8"] = dict(m_k=0.8, m_z=0.8, m_b=0.8, sev_bias=0.0, sev_noise=0.05, mix=MIX_BASE)
for b in (0.1, 0.2):
    SCEN[f"central_bias{b}"] = dict(CENTRAL, sev_bias=b)
for zd in (0.005, 0.02):
    mix = (0.97, 0.03 - zd, zd)
    SCEN[f"original_zd{zd}"] = dict(CAT_BASE, mix=mix)
    SCEN[f"central_zd{zd}"] = dict(CENTRAL, sev_bias=0.0, mix=mix)
SCEN["central"] = dict(CENTRAL, sev_bias=0.0)  # duplicate of mk0.8_mz0.2_mb0.5, kept as a named anchor
if QUICK:
    SCEN = {k: SCEN[k] for k in ("original", "central", "uninformative_0.5")}

# ------------------------------------------------------------------ tunable policies (same variable ranges as E7)
TUNABLE = {
    "CZT": dict(policy="Whittle", names=["kappa", "gamma", "beta", "s_u"],
                xl=[0.0, 0.25, 0.0, 0.3], xu=[1.0, 4.0, 2.0, 1.0]),
    "RiskT": dict(policy="RiskT", names=["gamma", "beta", "s_u"],
                  xl=[0.25, 0.0, 0.3], xu=[4.0, 2.0, 1.0]),
    "Linear": dict(policy="Linear", names=["w"] * 5, xl=[-1.0] * 5, xu=[1.0] * 5),
}
Q_RANGE = (0.95, 0.999)


def to_params(kind, x):
    spec = TUNABLE[kind]
    if kind == "Linear":
        return {"w": [float(v) for v in x[:5]]}
    return {n: float(v) for n, v in zip(spec["names"], x)}


# ------------------------------------------------------------------ shared state per worker
_STATE = {}


def state():
    if not _STATE:
        store = load_scores("results/e2_scores.npz")
        for g in GROUPS:
            for half, key in ((0, "TU"), (1, "EV")):
                _STATE[(key, g)] = Pool(Stream(store, "CZT-Hybrid", UNSW_SEVERITY, half=half, folds=GROUPS[g]),
                                        UNSW_SEVERITY)
    return _STATE


def raw_rate(pool, rho, mix, c=C0):
    """Raw event rate giving offered load rho at the reference threshold q = 0.99.
    The rate is then held fixed for every policy, so a policy with a different
    tuned threshold receives a different number of alerts from the same raw stream."""
    R = raw_events(pool, 40000 / DEFAULT["horizon"], 0, mix=mix)
    A = alerts(pool, R, observe(pool, R), 0.99)
    return rho * c / A["svc"].mean() / (len(A["t"]) / len(R["t"]))


def cat_of(sc):
    return {k: sc[k] for k in ("m_k", "m_z", "m_b", "sev_bias", "sev_noise")}


class Problem(ElementwiseProblem):
    def __init__(self, kind, pool, streams):
        sp = TUNABLE[kind]
        super().__init__(n_var=len(sp["xl"]) + 1, n_obj=4, xl=np.array(sp["xl"] + [Q_RANGE[0]]),
                         xu=np.array(sp["xu"] + [Q_RANGE[1]]))
        self.kind, self.pool, self.streams = kind, pool, streams

    def _evaluate(self, x, out, *a, **k):
        par, q = to_params(self.kind, x[:-1]), float(x[-1])
        res = [run_policy(alerts(self.pool, R, ob, q), C0, TUNABLE[self.kind]["policy"], par)
               for R, ob in self.streams]
        m = lambda f: float(np.nanmean([r[f] for r in res]))
        # objectives (all minimised): 1 - CV, 1 - ZCV, SLA violation share, alerts per hour / 100
        out["F"] = [1 - m("cv"), 1 - m("zcv"), m("sla_viol"), m("alerts_per_hour") / 100.0]


def tune(task):
    scen_name, g_eval, kind = task
    sc = SCEN[scen_name]
    S = state()
    pool = S[("TU", OTHER[g_eval])]
    raw = raw_rate(pool, RHO_TUNE, sc["mix"])
    streams = []
    for r in range(N_TUNE_SETS):
        R = raw_events(pool, raw, SEED_TUNE + r, mix=sc["mix"])
        streams.append((R, observe(pool, R, cat_of(sc))))
    ref = get_reference_directions("das-dennis", 4, n_partitions=6)
    alg = NSGA3(pop_size=len(ref), ref_dirs=ref, crossover=SBX(eta=30, prob=1.0, prob_var=0.5),
                mutation=PM(eta=20, prob=0.9))
    t0 = time.time()
    res = minimize(Problem(kind, pool, streams), alg, ("n_gen", N_GEN), seed=NSGA_SEED, verbose=False)
    F, X = np.atleast_2d(res.F), np.atleast_2d(res.X)
    Fn = (F - F.min(0)) / (F.max(0) - F.min(0) + 1e-12)
    knee = int(np.argmin(np.linalg.norm(Fn, axis=1)))
    return task, dict(params=to_params(kind, X[knee][:-1]), q=float(X[knee][-1]), F_knee=F[knee].tolist(),
                      front_size=len(F), time_s=time.time() - t0, x_knee=X[knee].tolist())


KEEP = ("cv", "zcv", "kcv", "wait_zd", "wait_known", "wait_attack", "sla_viol", "sla_viol_critical",
        "n_alerts", "fp_time_frac", "served_frac")


def evaluate(task):
    scen_name, g, tuned, tuned_orig = task
    sc = SCEN[scen_name]
    pool = state()[("EV", g)]
    raw = raw_rate(pool, RHO_EVAL, sc["mix"])
    pols = {"FIFO": ("FIFO", {}, 0.99), "Severity": ("Severity", {}, 0.99), "EDF": ("EDF", {}, 0.99),
            "Risk": ("Risk", {}, 0.99),
            "CZT-default": ("Whittle", {"kappa": 0.1, "gamma": 1.0, "beta": 1.0, "s_u": 0.5}, 0.99)}
    for kind, lab in (("CZT", "CZT"), ("RiskT", "Risk"), ("Linear", "Linear")):
        t = tuned[kind]
        pols[f"{lab}-T"] = (TUNABLE[kind]["policy"], t["params"], 0.99)  # tuned weights, common threshold
        pols[f"{lab}-TE"] = (TUNABLE[kind]["policy"], t["params"], t["q"])  # tuned weights and threshold
        to = tuned_orig[kind]  # parameters tuned in the original scenario (misspecified-catalogue transfer)
        pols[f"{lab}-T[orig]"] = (TUNABLE[kind]["policy"], to["params"], 0.99)
    cz = tuned["CZT"]["params"]
    abl = {"CZT-T no severity (beta=0)": dict(beta=0.0), "CZT-T no score (gamma=0)": dict(gamma=0.0),
           "CZT-T no decay": dict(decay=False), "CZT-T no deadline (kappa=0)": dict(kappa=0.0),
           "CZT-T s_u=0.5": dict(s_u=0.5)}
    for k, mod in abl.items():
        pols[k] = ("Whittle", {**cz, **mod}, 0.99)
        pols[k.replace("CZT-T", "CZT-TE")] = ("Whittle", {**cz, **mod}, tuned["CZT"]["q"])
    out = {p: {f: [] for f in KEEP} for p in pols}
    for r in range(REPS):
        R = raw_events(pool, raw, SEED_EVAL + r, mix=sc["mix"])
        ob = observe(pool, R, cat_of(sc))
        cache = {}
        for name, (pol, par, q) in pols.items():
            if q not in cache:
                cache[q] = alerts(pool, R, ob, q)
            res = run_policy(cache[q], C0, pol, par)
            for f in KEEP:
                out[name][f].append(float(res[f]))
    return task[:2], out


def main():
    t0 = time.time()
    nproc = max(1, (os.cpu_count() or 2) - 2)
    tasks = [(s, g, k) for s in SCEN for g in GROUPS for k in TUNABLE]
    print(f"{len(SCEN)} scenarios, {len(tasks)} tuning runs, {nproc} processes", flush=True)
    with MPPool(nproc) as mp:
        tuned = {}
        for i, (task, r) in enumerate(mp.imap_unordered(tune, tasks)):
            tuned[task] = r
            if i % 10 == 0:
                print(f"  tuned {i + 1}/{len(tasks)} {task} q={r['q']:.4f} F={np.round(r['F_knee'], 3)} "
                      f"[{time.time() - t0:.0f}s]", flush=True)
        ev_tasks = [(s, g, {k: tuned[(s, g, k)] for k in TUNABLE}, {k: tuned[("original", g, k)] for k in TUNABLE})
                    for s in SCEN for g in GROUPS]
        evals = dict(mp.map(evaluate, ev_tasks))
    res = {"meta": dict(reps=REPS, rho_eval=RHO_EVAL, rho_tune=RHO_TUNE, c=C0, n_tune_sets=N_TUNE_SETS,
                        n_gen=N_GEN, pop=84, nsga_seed=NSGA_SEED, seed_tune=SEED_TUNE, seed_eval=SEED_EVAL,
                        sbx=dict(eta=30, prob=1.0, prob_var=0.5), pm=dict(eta=20, prob=0.9, prob_var="1/n_var"),
                        bounds={k: dict(xl=v["xl"] + [Q_RANGE[0]], xu=v["xu"] + [Q_RANGE[1]]) for k, v in TUNABLE.items()},
                        groups=GROUPS, detector="CZT-Hybrid", time_s=time.time() - t0, quick=QUICK),
           "scenarios": SCEN,
           "tuned": {f"{s}|{g}|{k}": v for (s, g, k), v in tuned.items()},
           "eval": {f"{s}|{g}": v for (s, g), v in evals.items()}}
    json.dump(res, open(OUT.replace(".json", "_quick.json") if QUICK else OUT, "w"))
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
