"""Label-free catalogue observation model and common-random-number alert
streams for the SOC simulator (revision tasks R1.2, R1.3, R2.1, R2.2).

The original simulator (zdt/triage.make_alerts) derived the catalogue match
from the true event group: every known attack matched, every zero-day was
unmatched. Here the catalogue lookup is an explicit noisy observation:

  known attack   matches its own family entry with probability m_k
  zero-day       falsely matches a random known entry with probability m_z
  benign event   matches a random known entry with probability m_b

A matched entry reports the catalogue severity of that entry, which can carry a
per-family systematic bias (drawn once per replication, sd = sev_bias) plus
per-alert noise (sd = sev_noise). Unmatched alerts report the default 0.5.
The original model is the special case m_k = 1, m_z = 0, m_b = 0.5.

Scheduling policies receive only observables: detector score p, observed
severity s_obs, the match flag, arrival time and the SLA deadline (which is set
from s_obs). The true group and true severity are used only for the outcome
measures.

Common random numbers: every raw event (arrival time, group, record, service
time and the catalogue uniforms) is drawn once per replication from separate
substreams of one seed. Alerts for any escalation threshold q are obtained by
filtering this fixed raw stream, so all policies, thresholds and catalogue
scenarios of one replication see the same events and the same service times.
"""
import numpy as np

from .triage import DEFAULT, _lognormal, index_fn, simulate

CAT_BASE = dict(m_k=1.0, m_z=0.0, m_b=0.5, sev_bias=0.0, sev_noise=0.05)
GROUPS3 = np.array(["normal", "known", "zeroday"])


class Pool:
    """Vectorised view of a zdt.triage.Stream."""

    def __init__(self, stream, sev_map, extra=None):
        """extra: dict name -> per-record array aligned with stream (e.g. novelty score,
        prior-corrected probability); copied onto every raw event and alert."""
        self.s = stream
        self.extra = extra or {}
        self.folds = list(stream.folds)
        self.fam_names = [f for f in sev_map if f != "Normal"]
        self.fam_id = {f: i for i, f in enumerate(self.fam_names)}
        self.sev_true = np.array([sev_map[f] for f in self.fam_names])
        # known catalogue entries available when fold f is held out
        self.known_ids = [np.array([self.fam_id[x] for x in self.fam_names if x != f]) for f in self.folds]
        self.rec_fam = np.array([self.fam_id.get(f, -1) for f in stream.fam])
        self.thr_tab = [stream.thr_q[f] for f in self.folds]

    def thresholds(self, q):
        out = np.empty(len(self.folds))
        for k, a in enumerate(self.thr_tab):
            out[k] = a[min(len(a) - 1, int(np.ceil(q * len(a))) - 1)]
        return out


def raw_events(pool, raw_rate, seed, cfg=DEFAULT, mix=None):
    """Draw all raw detector events of one replication (catalogue-independent parts)."""
    st = pool.s
    mix = mix or cfg["mix"]
    r_arr, r_svc, r_cat = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(3)]
    H = cfg["horizon"]
    n = r_arr.poisson(raw_rate * H)
    t = np.sort(r_arr.uniform(0, H, n))
    g = r_arr.choice(3, n, p=mix)
    fk = r_arr.integers(len(pool.folds), size=n)
    u_rec = r_arr.random(n)
    rec = np.empty(n, int)
    for k, f in enumerate(pool.folds):
        for gi, gname in enumerate(GROUPS3):
            m = (fk == k) & (g == gi)
            idx = st.by[(f, gname)]
            rec[m] = idx[(u_rec[m] * len(idx)).astype(int)]
    y = g > 0
    svc = np.where(y, _lognormal(r_svc, *cfg["svc_attack"], n), _lognormal(r_svc, *cfg["svc_benign"], n))
    s_true = st.s_true[rec]
    R = dict(t=t, gi=g, group=GROUPS3[g], fk=fk, rec=rec, y=y, svc=svc, s=s_true,
             p=st.p[rec], pg=st.pg[rec], fold=np.array(pool.folds)[fk],
             **{k: v[rec] for k, v in pool.extra.items()},
             # catalogue uniforms and normals, fixed per replication
             u_match=r_cat.random(n), u_pick=r_cat.random(n), z_noise=r_cat.normal(0, 1, n),
             z_bias=r_cat.normal(0, 1, len(pool.fam_names)))
    R["_tot"] = np.array([s_true[y].sum(), s_true[g == 2].sum()])
    R["_tot_known"] = float(s_true[g == 1].sum())
    R["_raw_n"] = np.array([n, int((g == 2).sum())])
    return R


def observe(pool, R, cat=CAT_BASE, cfg=DEFAULT):
    """Apply the catalogue observation model; returns matched flag, s_obs and deadline."""
    n = len(R["t"])
    g, u = R["gi"], R["u_match"]
    cat_sev = np.clip(pool.sev_true + cat["sev_bias"] * R["z_bias"], 0.05, 1.0)
    p_match = np.select([g == 0, g == 1, g == 2], [cat["m_b"], cat["m_k"], cat["m_z"]])
    matched = u < p_match
    entry = np.full(n, -1)
    own = matched & (g == 1)
    entry[own] = pool.rec_fam[R["rec"][own]]
    other = matched & (g != 1)
    for k in range(len(pool.folds)):
        m = other & (R["fk"] == k)
        ids = pool.known_ids[k]
        entry[m] = ids[(R["u_pick"][m] * len(ids)).astype(int)]
    s_obs = np.full(n, 0.5)
    s_obs[matched] = np.clip(cat_sev[entry[matched]] + cat["sev_noise"] * R["z_noise"][matched], 0.05, 1.0)
    dl = np.full(n, cfg["sla"][-1][1])
    for thr, d in reversed(cfg["sla"]):
        dl = np.where(s_obs >= thr, d, dl)
    return dict(matched=matched, s_obs=s_obs, deadline=R["t"] + dl)


def alerts(pool, R, obs, q):
    """Escalated alerts for threshold q (filter of the fixed raw stream)."""
    keep = R["pg"] >= pool.thresholds(q)[R["fk"]]
    A = {k: v[keep] for k, v in R.items() if not k.startswith(("_", "u_", "z_"))}
    A.update({k: v[keep] for k, v in obs.items()})
    A["_tot"], A["_raw_n"], A["_tot_known"] = R["_tot"], R["_raw_n"], R["_tot_known"]
    return A


def novelty_info(A, Q, params):
    """Weight given to the imputed severity s_u: the catalogue no-match flag (info='flag')
    or the label-free novelty score in [0, 1] (info='nov')."""
    if params.get("info", "flag") == "nov":
        return A["nov"][Q]
    return (~A["matched"][Q]).astype(float)


def index_v2(policy, params, cfg, c, mean_svc):
    """Additional tuned baselines over the same observables as the index.
    Every policy uses only p, s_obs, the novelty information, arrival time and deadline."""
    if policy in ("Whittle", "RiskT") and params.get("info", "flag") == "nov":
        delta = np.log(2.0) / cfg["half_life"]
        Dbar = mean_svc / c
        ga, be, su = params.get("gamma", 1.0), params.get("beta", 1.0), params.get("s_u", 0.5)
        kappa = params.get("kappa", 0.1)
        use_decay, use_deadline = params.get("decay", True), params.get("deadline", True)

        def f(A, Q, now):
            v = A["nov"][Q]
            s_idx = (1 - v) * A["s_obs"][Q] + v * su
            r = (A["p"][Q] ** ga) * (s_idx ** be)
            if policy == "RiskT":
                return r - 1e-9 * A["t"][Q]
            decay = np.exp(-delta * (now - A["t"][Q])) if use_decay else 1.0
            term1 = r * decay * (delta * Dbar / (1 + delta * Dbar))
            tau = A["deadline"][Q] - now
            term2 = kappa * (s_idx ** be) * np.exp(-np.clip(tau, 0, None) / Dbar) * (tau > 0) if use_deadline else 0.0
            return term1 + term2 - 1e-12 * A["t"][Q]
        return f
    if policy == "RiskT":  # tuned risk rule p^gamma * s^beta with the same imputed s_u for unmatched alerts
        ga, be, su = params["gamma"], params["beta"], params["s_u"]

        def f(A, Q, now):
            s_idx = np.where(A["matched"][Q], A["s_obs"][Q], su)
            return (A["p"][Q] ** ga) * (s_idx ** be) - 1e-9 * A["t"][Q]
        return f
    if policy == "Linear":  # tuned linear priority over score, severity, no-match flag, age and deadline urgency
        w = np.asarray(params["w"])
        Dbar = mean_svc / c

        def f(A, Q, now):
            tau = A["deadline"][Q] - now
            urg = np.exp(-np.clip(tau, 0, None) / Dbar) * (tau > 0)
            age = (now - A["t"][Q]) / 60.0
            return (w[0] * A["p"][Q] + w[1] * A["s_obs"][Q] + w[2] * novelty_info(A, Q, params) + w[3] * age
                    + w[4] * urg - 1e-9 * A["t"][Q])
        return f
    return index_fn(policy, params, cfg, c, mean_svc)


def run_policy(A, c, policy, params, cfg=DEFAULT):
    """Simulate one policy on alert set A (outcomes include known-attack measures)."""
    return simulate(A, c, policy, params, cfg, index=index_v2)
