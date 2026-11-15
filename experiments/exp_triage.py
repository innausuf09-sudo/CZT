"""E7: SOC triage simulation driven by family-hold-out detector scores.

Leakage-averse design: the nine held-out families are split into two groups.
The Whittle-type index is tuned with NSGA-III on alerts whose zero-day family
belongs to one group (records of the tuning half only) and evaluated on alerts
whose zero-day family belongs to the other group (records of the evaluation half
only), and vice versa (two-fold cross-fitting over families).  Reported
quantities pool both evaluation groups (2 x 20 replications).
"""
import json, pickle, time
from zdt.io import load_scores
import numpy as np
from scipy.special import logit, expit
from zdt.data import UNSW_SEVERITY
from zdt.triage import Stream, make_alerts, simulate, DEFAULT
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize

log = lambda *a: print(*a, flush=True)
store = load_scores("results/e2_scores.npz")
OUT = "results/"
REPS = 20
C0 = 4
GROUPS = {"G1": ["Generic", "Fuzzers", "Reconnaissance", "Backdoor", "Worms"],
          "G2": ["Exploits", "DoS", "Analysis", "Shellcode"]}
OTHER = {"G1": "G2", "G2": "G1"}


def hybrid_uncal(st):
    """Hybrid built from uncalibrated member probabilities (calibration ablation)."""
    z = np.mean([logit(np.clip(st["scores"][m]["pu"], 1e-6, 1 - 1e-6))
                 for m in ("LightGBM", "ResSE-CNN1D", "B-ProtoMeta")], 0)
    return expit(z)


def S(model="CZT-Hybrid", g="G1", half=1, pscore="p"):
    return Stream(store, model, UNSW_SEVERITY, pscore=pscore, half=half, folds=GROUPS[g])


EV = {g: S(g=g, half=1) for g in GROUPS}
EV_UNCAL = {g: S(g=g, half=1, pscore=hybrid_uncal) for g in GROUPS}
TU = {g: S(g=g, half=0) for g in GROUPS}


def raw_rate_for(stream, rho, c, q=0.99, n=40000, seed=0):
    """Raw event rate (per min) that gives offered analyst load rho under threshold q."""
    A = make_alerts(stream, n / DEFAULT["horizon"], q, np.random.default_rng(seed))
    esc = len(A["t"]) / n
    return rho * c / A["svc"].mean() / esc


RAW = {}


def raw_for(g, rho, cfg_key="base"):
    key = (g, rho)
    if key not in RAW:
        RAW[key] = raw_rate_for(EV[g], rho, C0)
    return RAW[key]


def run(stream, policy, params, raw, c=C0, q=0.99, reps=REPS, seed0=1000, cfg=DEFAULT):
    return [simulate(make_alerts(stream, raw, q, np.random.default_rng(seed0 + r), cfg), c, policy, params, cfg)
            for r in range(reps)]


def summ(res):
    return {k: (float(np.nanmean([r[k] for r in res])), float(1.96 * np.nanstd([r[k] for r in res]) / np.sqrt(len(res))))
            for k in res[0].keys()}


# ------------------------------------------------------------------ NSGA-III (per group)
class TriageProblem(ElementwiseProblem):
    def __init__(self, tune_stream, alerts):
        super().__init__(n_var=5, n_obj=4, xl=np.array([0.0, 0.25, 0.0, 0.3, 0.95]),
                         xu=np.array([1.0, 4.0, 2.0, 1.0, 0.999]))
        self.ts, self.alerts = tune_stream, alerts

    def _evaluate(self, x, out, *a, **k):
        kappa, gamma, beta, s_u, q = x
        res = []
        for A0 in self.alerts:
            keep = np.array([A0["pg"][i] >= self.ts.threshold(A0["fold"][i], q) for i in range(len(A0["t"]))], bool)
            A = {kk: (v[keep] if not kk.startswith("_") else v) for kk, v in A0.items()}
            res.append(simulate(A, C0, "Whittle", {"kappa": kappa, "gamma": gamma, "beta": beta, "s_u": s_u}))
        s = summ(res)
        out["F"] = [1 - s["cv"][0], 1 - s["zcv"][0], s["sla_viol"][0], s["alerts_per_hour"][0] / 100.0]


TUNED, nsga_out = {}, {}
for g_eval in GROUPS:
    g_tune = OTHER[g_eval]
    ts = TU[g_tune]
    raw_t = raw_rate_for(ts, 1.0, C0)
    alerts = [make_alerts(ts, raw_t, 0.95, np.random.default_rng(5000 + r)) for r in range(8)]
    t0 = time.time()
    ref = get_reference_directions("das-dennis", 4, n_partitions=6)
    res = minimize(TriageProblem(ts, alerts), NSGA3(pop_size=len(ref), ref_dirs=ref), ("n_gen", 60), seed=7, verbose=False)
    F, X = res.F, res.X
    Fn = (F - F.min(0)) / (F.max(0) - F.min(0) + 1e-12)
    knee = int(np.argmin(np.linalg.norm(Fn, axis=1)))
    kx = X[knee]
    TUNED[g_eval] = ({"kappa": float(kx[0]), "gamma": float(kx[1]), "beta": float(kx[2]), "s_u": float(kx[3])}, float(kx[4]))
    nsga_out[g_eval] = {"tuned_on": g_tune, "F": F.tolist(), "X": X.tolist(), "knee": knee, "n_ref": len(ref),
                        "n_gen": 60, "time_s": time.time() - t0}
    log(f"NSGA-III (tune {g_tune} -> eval {g_eval}) {time.time()-t0:.0f}s |front|={len(F)} knee x={np.round(kx, 3)} F={np.round(F[knee], 3)}")
json.dump(nsga_out, open(OUT + "e7_nsga3.json", "w"), indent=1)

BASE = {"FIFO": ("FIFO", {}, None), "Severity": ("Severity", {}, None), "EDF": ("EDF", {}, None),
        "Risk": ("Risk", {}, None), "CZT-Whittle": ("Whittle", {"kappa": 0.1, "gamma": 1, "beta": 1}, None)}
ALL = list(BASE) + ["CZT-Whittle-T", "CZT-Whittle-NSGA"]


def spec(name, g):
    if name == "CZT-Whittle-T":  # tuned index weights, default escalation threshold (same alert set as baselines)
        par, q = TUNED[g]
        return "Whittle", par, 0.99
    if name == "CZT-Whittle-NSGA":
        par, q = TUNED[g]
        return "Whittle", par, q
    p, par, q = BASE[name]
    return p, par, (q or 0.99)


def pooled(name, rho, c=C0, raw_rho=None, cfg=DEFAULT, stream_map=EV, override=None):
    """Evaluate one policy on both evaluation groups and pool the 2 x REPS replications."""
    out = []
    for g in GROUPS:
        pol, par, q = override(g) if override else spec(name, g)
        raw = raw_for(g, raw_rho if raw_rho is not None else rho)
        out += run(stream_map[g], pol, par, raw, c, q, REPS, cfg=cfg)
    return summ(out)


results = {"tuned": {g: {**TUNED[g][0], "q": TUNED[g][1]} for g in GROUPS}, "groups": GROUPS}
results["load"] = {}
for rho in [0.6, 0.8, 1.0, 1.2, 1.5]:
    results["load"][rho] = {p: pooled(p, rho) for p in ALL}
    log(f"rho={rho}: " + " ".join(f"{p}:cv={v['cv'][0]:.3f},zcv={v['zcv'][0]:.3f},sla={v['sla_viol'][0]:.3f},n={v['n_alerts'][0]:.0f}"
                                  for p, v in results["load"][rho].items()))
    json.dump(results, open(OUT + "e7_triage.json", "w"), indent=1)

results["staff"] = {}
for c in [2, 3, 4, 6, 8]:
    results["staff"][c] = {p: pooled(p, 1.0, c=c, raw_rho=1.0) for p in ALL}
    log(f"c={c}: " + " ".join(f"{p}:zcv={v['zcv'][0]:.3f},wzd={v['wait_zd'][0]:.0f}" for p, v in results["staff"][c].items()))
json.dump(results, open(OUT + "e7_triage.json", "w"), indent=1)

results["halflife"] = {}
for hl in [15, 30, 60, 120, 240]:
    cfg = dict(DEFAULT); cfg["half_life"] = float(hl)
    results["halflife"][hl] = {p: pooled(p, 1.2, cfg=cfg) for p in ALL}
json.dump(results, open(OUT + "e7_triage.json", "w"), indent=1)

# ablation at rho = 1.2: default index and tuned index, one ingredient removed at a time
def tuned_mod(**mods):
    def f(g):
        par, q = TUNED[g]
        par = dict(par); qq = mods.pop("q", q) if False else mods.get("q", q)
        for k, v in mods.items():
            if k != "q":
                par[k] = v
        return "Whittle", par, qq
    return f


abl = {
    "Risk baseline (p x catalogue severity)": dict(name="Risk"),
    "CZT-Whittle, default weights": dict(name="CZT-Whittle"),
    "  default, without calibration": dict(name="CZT-Whittle", stream_map=EV_UNCAL),
    "  default, without evidence decay": dict(override=lambda g: ("Whittle", {"kappa": .1, "decay": False}, 0.99)),
    "  default, without deadline term": dict(override=lambda g: ("Whittle", {"kappa": 0.0}, 0.99)),
    "  default, without detector score (gamma = 0)": dict(override=lambda g: ("Whittle", {"kappa": .1, "gamma": 0.0}, 0.99)),
    "CZT-Whittle-NSGA (tuned)": dict(name="CZT-Whittle-NSGA"),
    "  tuned, without calibration": dict(name="CZT-Whittle-NSGA", stream_map=EV_UNCAL),
    "  tuned, escalation fixed at q = 0.99": dict(override=tuned_mod(q=0.99)),
    "  tuned, imputed severity s_u = 0.5": dict(override=tuned_mod(s_u=0.5)),
    "  tuned, without evidence decay": dict(override=tuned_mod(decay=False)),
    "  tuned, without detector score (gamma = 0)": dict(override=tuned_mod(gamma=0.0)),
}
results["ablation"] = {}
for k, a in abl.items():
    results["ablation"][k] = pooled(a.get("name", "CZT-Whittle"), 1.2, stream_map=a.get("stream_map", EV),
                                    override=a.get("override"))
    log(f"ablation {k}: cv={results['ablation'][k]['cv'][0]:.3f} zcv={results['ablation'][k]['zcv'][0]:.3f} "
        f"sla={results['ablation'][k]['sla_viol'][0]:.3f}")
json.dump(results, open(OUT + "e7_triage.json", "w"), indent=1)

# detector comparison: identical raw event load, tuned index fed by each detector's calibrated scores
results["detectors"] = {}
for m in ["LR", "RF", "XGBoost", "LightGBM", "SVM", "DNN", "ResSE-CNN1D", "LSTM", "GRU", "FT-Transformer", "FG-GNN",
          "B-ProtoMeta", "CZT-Hybrid"]:
    smap = {g: S(model=m, g=g, half=1) for g in GROUPS}
    results["detectors"][m] = {
        "Risk": pooled("Risk", 1.2, stream_map=smap),
        "CZT-Whittle-NSGA": pooled("CZT-Whittle-NSGA", 1.2, stream_map=smap)}
    v = results["detectors"][m]["CZT-Whittle-NSGA"]
    log(f"detector {m}: cv={v['cv'][0]:.3f} zcv={v['zcv'][0]:.3f} n={v['n_alerts'][0]:.0f}")
json.dump(results, open(OUT + "e7_triage.json", "w"), indent=1)
log("done")
