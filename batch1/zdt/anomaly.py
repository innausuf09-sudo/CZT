"""Additional comparators for the second revision round.

- IForest: Isolation Forest fitted on benign training records only (one-class anomaly baseline).
- AutoEncoder: fully connected autoencoder fitted on benign training records; score = reconstruction error.
- EnergyNet: multi-class MLP over the training classes (benign and known families); the energy
  E(x) = -logsumexp(logits) is used as an open-set novelty score (Liu et al., NeurIPS 2020); its
  attack score (for detection) is 1 - p(benign).
All follow the Base interface of zdt/models.py (raw_score on a log-odds-like scale, Platt calibration).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from .models import Base, _p2z


class IForest(Base):
    name = "IsolationForest"
    needs_scaling = False

    def fit(self, X, y, Xva=None, yva=None, fam=None):
        self.m = IsolationForest(n_estimators=200, random_state=self.seed, n_jobs=1).fit(X[y == 0])
        return self

    def raw_score(self, X):
        return -self.m.score_samples(X)


class _AE(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Linear(64, 16), nn.ReLU())
        self.dec = nn.Sequential(nn.Linear(16, 64), nn.ReLU(), nn.Linear(64, d))

    def forward(self, x):
        return self.dec(self.enc(x))


class AutoEncoder(Base):
    name = "Autoencoder"

    def fit(self, X, y, Xva=None, yva=None, fam=None, epochs=30):
        torch.manual_seed(self.seed)
        Xb = X[y == 0]
        self.scaler = StandardScaler().fit(Xb)
        Xt = torch.tensor(self._t(Xb))
        self.net = _AE(X.shape[1])
        opt = torch.optim.AdamW(self.net.parameters(), lr=1e-3, weight_decay=1e-4)
        for _ in range(epochs):
            perm = torch.randperm(len(Xt))
            for i in range(0, len(Xt), 256):
                j = perm[i:i + 256]
                loss = F.mse_loss(self.net(Xt[j]), Xt[j])
                opt.zero_grad(); loss.backward(); opt.step()
        self.net.eval()
        return self

    def raw_score(self, X):
        Xt = torch.tensor(self._t(X))
        with torch.no_grad():
            r = torch.cat([((self.net(Xt[i:i + 8192]) - Xt[i:i + 8192]) ** 2).mean(1) for i in range(0, len(Xt), 8192)])
        return np.log(r.numpy().astype(np.float64) + 1e-9)


class EnergyNet(Base):
    name = "EnergyMLP"

    def fit(self, X, y, Xva=None, yva=None, fam=None, epochs=20):
        torch.manual_seed(self.seed)
        fam = np.asarray(fam).astype(str)
        self.classes = sorted(np.unique(fam))
        self.ni = self.classes.index("Normal")
        lab = torch.tensor([self.classes.index(c) for c in fam])
        self.scaler = StandardScaler().fit(X)
        Xt = torch.tensor(self._t(X))
        self.net = nn.Sequential(nn.Linear(X.shape[1], 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 128), nn.ReLU(),
                                 nn.Linear(128, len(self.classes)))
        cnt = np.bincount(lab.numpy(), minlength=len(self.classes)).astype(float)
        w = torch.tensor(len(lab) / (len(cnt) * np.maximum(cnt, 1)), dtype=torch.float32)
        opt = torch.optim.AdamW(self.net.parameters(), lr=1e-3, weight_decay=1e-4)
        for _ in range(epochs):
            self.net.train()
            perm = torch.randperm(len(Xt))
            for i in range(0, len(Xt), 512):
                j = perm[i:i + 512]
                loss = F.cross_entropy(self.net(Xt[j]), lab[j], weight=w)
                opt.zero_grad(); loss.backward(); opt.step()
        self.net.eval()
        return self

    def _logits(self, X):
        Xt = torch.tensor(self._t(X))
        with torch.no_grad():
            return torch.cat([self.net(Xt[i:i + 8192]) for i in range(0, len(Xt), 8192)])

    def raw_score(self, X):
        p = F.softmax(self._logits(X), 1)[:, self.ni].numpy()
        return _p2z(1.0 - p)

    def energy(self, X):
        return (-torch.logsumexp(self._logits(X), 1)).numpy()
