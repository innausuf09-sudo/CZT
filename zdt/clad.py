"""CLAD comparator (revision R2.5): contrastive learning for anomaly detection,
re-implemented from J. Wilkie et al., "A Novel Contrastive Loss for Zero-Day
Network Intrusion Detection", IEEE TNSM, 2026 (arXiv 2601.09902), Eqs. (4), (7)-(10).

- Encoder: linear projection to d_model, `depth` blocks of Linear(d_model) + ReLU
  (+ dropout), linear head to the embedding dimension, L2 normalisation.
- Loss (Eq. 7): benign anchors only; positives = other benign samples in the
  batch, negatives = malicious samples in the batch, d(z, z') = (1 - z.z') / 2,
  L = mean_i [ mean_p d(z_i, z_p)^2 + mean_n (1 - d(z_i, z_n))^2 ].
- Score (Eq. 9): s(x) = -(z . mu), mu = normalised benign training centroid (Eq. 8).
- Training as in the paper: AdamW, linear warm-up then cosine annealing, class-
  balanced batch sampling. Values the paper does not give (learning rate, width,
  depth, dropout, epochs) are chosen by a small search on training families only
  (see experiments/exp_p2_refit.py); the score is Platt-calibrated on the same
  validation split as every other detector.
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler

from .models import Base


class _CLADNet(nn.Module):
    def __init__(self, d, d_model=256, depth=3, emb=32, dropout=0.1):
        super().__init__()
        layers = [nn.Linear(d, d_model)]
        for _ in range(depth):
            layers += [nn.Linear(d_model, d_model), nn.ReLU(), nn.Dropout(dropout)]
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(d_model, emb)

    def forward(self, x):
        return F.normalize(self.head(self.body(x)), dim=-1)


def clad_loss(z, y):
    """Eq. (7) of Wilkie et al.; y = 1 for malicious, 0 for benign."""
    b = y == 0
    if b.sum() < 2:
        return z.sum() * 0.0
    zb = z[b]
    d_bb = (1 - zb @ zb.T) / 2
    nb = len(zb)
    pos = (d_bb ** 2).sum(1) / (nb - 1)  # diagonal is zero
    if (~b).any():
        d_bm = (1 - zb @ z[~b].T) / 2
        neg = ((1 - d_bm) ** 2).mean(1)
    else:
        neg = torch.zeros_like(pos)
    return (pos + neg).mean()


class CLAD(Base):
    name = "CLAD"

    def __init__(self, seed=0, lr=1e-3, d_model=256, depth=3, emb=32, dropout=0.1, epochs=40, batch=512,
                 warmup=4, weight_decay=1e-4):
        super().__init__(seed)
        self.hp = dict(lr=lr, d_model=d_model, depth=depth, emb=emb, dropout=dropout, epochs=epochs, batch=batch,
                       warmup=warmup, weight_decay=weight_decay)

    def fit(self, X, y, Xva=None, yva=None, fam=None):
        hp = self.hp
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        self.scaler = StandardScaler().fit(X)
        Xt = torch.tensor(self._t(X))
        yt = torch.tensor(y.astype(np.int64))
        self.net = _CLADNet(X.shape[1], hp["d_model"], hp["depth"], hp["emb"], hp["dropout"])
        opt = torch.optim.AdamW(self.net.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
        steps_per_ep = max(1, len(X) // hp["batch"])
        total, warm = hp["epochs"] * steps_per_ep, hp["warmup"] * steps_per_ep
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
        # class-balanced batch sampling (weighted sampling of benign vs malicious)
        w = np.where(y == 0, 0.5 / max((y == 0).sum(), 1), 0.5 / max((y == 1).sum(), 1))
        w = w / w.sum()
        self.net.train()
        for _ in range(hp["epochs"]):
            for _ in range(steps_per_ep):
                j = torch.tensor(rng.choice(len(X), hp["batch"], p=w))
                loss = clad_loss(self.net(Xt[j]), yt[j])
                opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        self.net.eval()
        with torch.no_grad():
            zb = torch.cat([self.net(Xt[(yt == 0)][i:i + 8192]) for i in range(0, int((yt == 0).sum()), 8192)])
            self.mu = F.normalize(zb.sum(0), dim=0)
        return self

    def raw_score(self, X):
        Xt = torch.tensor(self._t(X))
        with torch.no_grad():
            z = torch.cat([self.net(Xt[i:i + 8192]) for i in range(0, len(Xt), 8192)])
            return (-(z @ self.mu)).numpy().astype(np.float64)
