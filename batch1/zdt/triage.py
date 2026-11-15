"""Discrete-event SOC triage simulator with evidence decay, SLA deadlines and a
closed-form Whittle-type priority index (see manuscript Section 3.6)."""
import heapq
import numpy as np

LN2 = np.log(2.0)


class Stream:
    """Pool of scored detector outputs (from the family hold-out protocol) used
    to synthesise alert streams.  Each record carries: group (normal/known/
    zeroday), true family severity s, calibrated score p, fold id."""

    def __init__(self, store, model, sev_map, pscore="p", half=None, rng_split=0, folds=None):
        """pscore: key or function giving the score used by the priority index;
        escalation always uses the model's calibrated score 'p' (Platt scaling is
        monotone, so the escalated set is identical for calibrated/uncalibrated)."""
        recs = []
        self.thr_q = {}
        for fold, st in store.items():
            if folds is not None and fold not in folds:
                continue
            sc = st["scores"][model]
            p = sc[pscore] if isinstance(pscore, str) else pscore(st)
            self.thr_q[fold] = np.sort(sc["pva_normal"])
            for g, fam, pi, pg in zip(st["groups"], st["fam"], p, sc["p"]):
                recs.append((fold, g, fam, float(pi), float(pg)))
        rng = np.random.default_rng(rng_split)
        idx = rng.permutation(len(recs))
        if half is not None:  # disjoint tuning / evaluation pools
            idx = idx[: len(idx) // 2] if half == 0 else idx[len(idx) // 2:]
        self.fold = np.array([recs[i][0] for i in idx])
        self.group = np.array([recs[i][1] for i in idx])
        self.fam = np.array([recs[i][2] for i in idx])
        self.p = np.array([recs[i][3] for i in idx])
        self.pg = np.array([recs[i][4] for i in idx])
        self.s_true = np.array([sev_map[f] for f in self.fam])
        self.folds = sorted(f for f in store if folds is None or f in folds)
        self.by = {(f, g): np.where((self.fold == f) & (self.group == g))[0]
                   for f in self.folds for g in ("normal", "known", "zeroday")}
        self.known_sev = {f: np.array([sev_map[x] for x in sev_map if x not in ("Normal", f)]) for f in self.folds}

    def threshold(self, fold, q):
        a = self.thr_q[fold]
        return float(a[min(len(a) - 1, int(np.ceil(q * len(a))) - 1)])


DEFAULT = dict(mix=(0.97, 0.02, 0.01), svc_attack=(12.0, 4.0), svc_benign=(7.0, 3.0),
               half_life=60.0, horizon=480.0, drain=120.0,
               sla=((0.85, 15.0), (0.65, 60.0), (0.40, 240.0), (0.0, 1440.0)))


def _lognormal(rng, mean, sd, n):
    s2 = np.log(1 + (sd / mean) ** 2)
    return rng.lognormal(np.log(mean) - s2 / 2, np.sqrt(s2), n)


def make_alerts(stream, raw_rate, q, rng, cfg=DEFAULT):
    """Raw detector events arrive as a Poisson process of rate raw_rate (per
    minute) over the horizon; an event becomes an analyst alert when its score
    exceeds the fold-specific escalation threshold (validation-normal quantile q)."""
    H = cfg["horizon"]
    n = rng.poisson(raw_rate * H)
    t = np.sort(rng.uniform(0, H, n))
    groups = rng.choice(["normal", "known", "zeroday"], n, p=cfg["mix"])
    folds = rng.choice(stream.folds, n)
    keep, rec = [], []
    tot_all, tot_zd = 0.0, 0.0
    for i in range(n):
        pool = stream.by[(folds[i], groups[i])]
        j = pool[rng.integers(len(pool))]
        if groups[i] != "normal":  # every raw attack event counts in the denominator of CV / ZCV
            tot_all += stream.s_true[j]
            if groups[i] == "zeroday":
                tot_zd += stream.s_true[j]
        if stream.pg[j] >= stream.threshold(folds[i], q):
            keep.append(i); rec.append(j)
    keep, rec = np.array(keep, int), np.array(rec, int)
    A = dict(t=t[keep], group=groups[keep], p=stream.p[rec], pg=stream.pg[rec], s=stream.s_true[rec],
             fold=folds[keep])
    m = len(keep)
    # observable (enrichment) severity: catalogue severity for known families,
    # default 'medium' 0.5 for zero-day behaviour, a random catalogue severity for false positives
    s_obs = np.empty(m)
    matched = np.ones(m, bool)
    for i in range(m):
        g = A["group"][i]
        if g == "known":
            s_obs[i] = np.clip(A["s"][i] + rng.normal(0, 0.05), 0.05, 1)
        elif g == "zeroday":
            s_obs[i] = 0.5; matched[i] = False
        else:
            if rng.random() < cfg.get("fp_match", 0.5):
                s_obs[i] = rng.choice(stream.known_sev[A["fold"][i]])
            else:
                s_obs[i] = 0.5; matched[i] = False
    A["matched"] = matched
    A["s_obs"] = s_obs
    y = (A["group"] != "normal")
    A["y"] = y
    svc = np.where(y, _lognormal(rng, *cfg["svc_attack"], m), _lognormal(rng, *cfg["svc_benign"], m))
    A["svc"] = svc
    dl = np.empty(m)
    for i in range(m):
        for thr, d in cfg["sla"]:
            if s_obs[i] >= thr:
                dl[i] = d
                break
    A["deadline"] = A["t"] + dl
    A["_tot"] = np.array([tot_all, tot_zd])
    A["_raw_n"] = np.array([n, int((groups == "zeroday").sum())])
    return A


def index_fn(policy, params, cfg, c, mean_svc):
    """Return a function (A, idx_queue, now) -> priority scores (higher first)."""
    delta = LN2 / cfg["half_life"]
    Dbar = mean_svc / c  # expected time between service opportunities
    kappa, gamma, beta = params.get("kappa", 0.1), params.get("gamma", 1.0), params.get("beta", 1.0)
    s_u = params.get("s_u", 0.5)  # severity imputed for alerts with no catalogue match
    use_decay, use_deadline = params.get("decay", True), params.get("deadline", True)

    if policy == "FIFO":
        return lambda A, Q, now: -A["t"][Q]
    if policy == "Severity":
        return lambda A, Q, now: A["s_obs"][Q] - 1e-9 * A["t"][Q]
    if policy == "EDF":
        return lambda A, Q, now: -A["deadline"][Q]
    if policy == "Risk":
        return lambda A, Q, now: A["p"][Q] * A["s_obs"][Q] - 1e-9 * A["t"][Q]
    if policy == "Whittle":
        def f(A, Q, now):
            s_idx = np.where(A["matched"][Q], A["s_obs"][Q], s_u)
            r = (A["p"][Q] ** gamma) * (s_idx ** beta)
            decay = np.exp(-delta * (now - A["t"][Q])) if use_decay else 1.0
            term1 = r * decay * (delta * Dbar / (1 + delta * Dbar))  # 1 - E[exp(-delta*Delta)], Delta~Exp(Dbar)
            tau = A["deadline"][Q] - now
            term2 = kappa * (s_idx ** beta) * np.exp(-np.clip(tau, 0, None) / Dbar) * (tau > 0) \
                if use_deadline else 0.0
            return term1 + term2 - 1e-12 * A["t"][Q]
        return f
    raise ValueError(policy)


def simulate(A, c, policy, params=None, cfg=DEFAULT, index=None):
    """index: optional factory with the signature of index_fn (used for the
    additional tuned baselines of the revision, see zdt/catalogue.py)."""
    params = params or {}
    m = len(A["t"])
    delta = LN2 / cfg["half_life"]
    mean_svc = float(np.mean(A["svc"])) if m else 10.0
    prio = (index or index_fn)(policy, params, cfg, c, mean_svc)
    start = np.full(m, np.nan)
    free = [0.0] * c
    heapq.heapify(free)
    queue = []
    nxt = 0
    end = cfg["horizon"] + cfg["drain"]
    busy_fp = 0.0; busy = 0.0
    while True:
        tfree = heapq.heappop(free)
        # admit arrivals up to the moment a server becomes free
        while nxt < m and A["t"][nxt] <= tfree:
            queue.append(nxt); nxt += 1
        if not queue:
            if nxt >= m:
                break
            tfree = max(tfree, A["t"][nxt])
            while nxt < m and A["t"][nxt] <= tfree:
                queue.append(nxt); nxt += 1
        if tfree >= end:
            break
        Q = np.array(queue)
        k = int(np.argmax(prio(A, Q, tfree)))
        i = queue.pop(k)
        start[i] = tfree
        busy += A["svc"][i]
        if not A["y"][i]:
            busy_fp += A["svc"][i]
        heapq.heappush(free, tfree + A["svc"][i])
    return metrics(A, start, delta, end, busy_fp, busy)


def metrics(A, start, delta, end, busy_fp, busy):
    y = A["y"]; zd = A["group"] == "zeroday"
    served = ~np.isnan(start)
    wait = np.where(served, start - A["t"], end - A["t"])
    val = np.where(served, A["s"] * np.exp(-delta * np.nan_to_num(start - A["t"])), 0.0)
    # end-to-end denominators: all raw attack events, including those the detector did not escalate
    tot, tot_zd = (A["_tot"] if "_tot" in A else (A["s"][y].sum(), A["s"][zd].sum()))
    viol = np.where(served, start > A["deadline"], A["deadline"] < end)
    crit = y & (A["s"] >= 0.85)
    esc_zd = float(zd.sum() / A["_raw_n"][1]) if "_raw_n" in A and A["_raw_n"][1] > 0 else np.nan
    extra = {}
    if "_tot_known" in A:  # known-attack outcomes (revision R2.6), only for the catalogue-model streams
        kn = A["group"] == "known"
        extra = dict(kcv=float(val[kn].sum() / A["_tot_known"]) if A["_tot_known"] > 0 else np.nan,
                     wait_known=float(wait[kn].mean()) if kn.any() else np.nan)
    return dict(**extra,n_alerts=int(len(y)), n_attack=int(y.sum()), n_zd=int(zd.sum()), zd_escalated=esc_zd,
                cv=float(val[y].sum() / tot) if tot > 0 else np.nan,
                zcv=float(val[zd].sum() / tot_zd) if tot_zd > 0 else np.nan,
                wait_attack=float(wait[y].mean()) if y.any() else np.nan,
                wait_zd=float(wait[zd].mean()) if zd.any() else np.nan,
                wait_zd_median=float(np.median(wait[zd])) if zd.any() else np.nan,
                sla_viol=float(viol.mean()) if len(viol) else np.nan,
                sla_viol_attack=float(viol[y].mean()) if y.any() else np.nan,
                sla_viol_critical=float(viol[crit].mean()) if crit.any() else np.nan,
                served_frac=float(served.mean()) if len(served) else np.nan,
                fp_time_frac=float(busy_fp / busy) if busy > 0 else 0.0,
                alerts_per_hour=float(len(y) / (DEFAULT["horizon"] / 60)))
