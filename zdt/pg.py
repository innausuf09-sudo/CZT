"""Learned scheduler (revision R2.1): a softmax policy over the same observables as the
tuned linear rule, trained with REINFORCE in the simulator.

At each service opportunity the policy picks alert i from the queue with probability
softmax(theta . phi_i), where phi_i = (p, s_obs, novelty information, age in hours,
deadline urgency). The episode reward is ZCV + KCV - LAMBDA * max(0, critical SLA misses - 0.02),
fixed before any run. At evaluation the greedy policy (argmax theta . phi) is used, which
has the same functional form as the linear rule, so the comparison isolates the effect of
the training method. The budget is the same number of simulated shifts as NSGA-III
(population x generations x tuning streams).
"""
import heapq

import numpy as np

from .catalogue import novelty_info
from .triage import DEFAULT, LN2, metrics

LAMBDA = 10.0


def features(A, Q, now, Dbar, params):
    tau = A["deadline"][Q] - now
    urg = np.exp(-np.clip(tau, 0, None) / Dbar) * (tau > 0)
    return np.stack([A["p"][Q], A["s_obs"][Q], novelty_info(A, Q, params), (now - A["t"][Q]) / 60.0, urg], 1)


def simulate_pg(A, c, theta, rng, params, cfg=DEFAULT):
    """Stochastic rollout; returns outcome metrics and the score-function gradient sum."""
    m = len(A["t"])
    delta = LN2 / cfg["half_life"]
    mean_svc = float(np.mean(A["svc"])) if m else 10.0
    Dbar = mean_svc / c
    start = np.full(m, np.nan)
    free = [0.0] * c
    heapq.heapify(free)
    queue, nxt = [], 0
    end = cfg["horizon"] + cfg["drain"]
    busy_fp = busy = 0.0
    grad = np.zeros_like(theta)
    while True:
        tfree = heapq.heappop(free)
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
        phi = features(A, Q, tfree, Dbar, params)
        z = phi @ theta
        pr = np.exp(z - z.max()); pr /= pr.sum()
        k = int(rng.choice(len(Q), p=pr)) if len(Q) > 1 else 0
        grad += phi[k] - pr @ phi
        i = queue.pop(k)
        start[i] = tfree
        busy += A["svc"][i]
        if not A["y"][i]:
            busy_fp += A["svc"][i]
        heapq.heappush(free, tfree + A["svc"][i])
    return metrics(A, start, delta, end, busy_fp, busy), grad


def reward(r):
    crit = 0.0 if np.isnan(r["sla_viol_critical"]) else r["sla_viol_critical"]
    return r["zcv"] + r["kcv"] - LAMBDA * max(0.0, crit - 0.02)


def train_pg(alert_sets, c, n_shifts, params, seed=7, lr=0.05, n_ckpt=10):
    """REINFORCE with a moving-average baseline and Adam; one update per pass over the
    tuning alert sets. Returns checkpoints (theta) for the pre-declared selection rule."""
    rng = np.random.default_rng(seed)
    theta = np.zeros(5)
    mom, vel = np.zeros(5), np.zeros(5)
    base, n_upd = None, n_shifts // len(alert_sets)
    ckpts = []
    for u in range(n_upd):
        g_tot, rs, grads = np.zeros(5), [], []
        for A in alert_sets:
            r, g = simulate_pg(A, c, theta, rng, params)
            rs.append(reward(r)); grads.append(g)
        rs = np.array(rs)
        base = rs.mean() if base is None else 0.9 * base + 0.1 * rs.mean()
        for rr, g in zip(rs, grads):
            g_tot += (rr - base) * g
        g_tot /= len(alert_sets)
        mom = 0.9 * mom + 0.1 * g_tot
        vel = 0.999 * vel + 0.001 * g_tot ** 2
        theta = theta + lr * (mom / (1 - 0.9 ** (u + 1))) / (np.sqrt(vel / (1 - 0.999 ** (u + 1))) + 1e-8)
        if (u + 1) % max(1, n_upd // n_ckpt) == 0:
            ckpts.append(theta.copy())
    return ckpts
