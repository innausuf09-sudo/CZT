"""Model zoo: classical, deep, graph and meta-learning detectors with a common
interface.  Every model exposes fit(), raw_score() (a real-valued log-odds-like
score, larger = more malicious) and, after calibrate(), proba()."""
import time
import warnings
import numpy as np
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from scipy.special import logit, expit
import xgboost as xgb
import lightgbm as lgb

torch.set_num_threads(2)
EPS = 1e-6


# ------------------------------------------------------------------ calibration
class Platt:
    """Platt / temperature-bias scaling on the raw log-odds score:
    p = sigma(a * z + b), fitted by maximum likelihood on a validation split."""

    def fit(self, z, y):
        lr = LogisticRegression(C=1e6, max_iter=2000)
        lr.fit(z.reshape(-1, 1), y)
        self.a, self.b = float(lr.coef_[0, 0]), float(lr.intercept_[0])
        return self

    def __call__(self, z):
        return expit(self.a * z + self.b)


class Base:
    name = "base"
    needs_scaling = True

    def __init__(self, seed=0):
        self.seed = seed
        self.cal = None
        self.scaler = None

    # scaling is fitted on training data only (leakage-averse)
    def _fit_scaler(self, X):
        if self.needs_scaling:
            self.scaler = StandardScaler().fit(X)

    def _t(self, X):
        return self.scaler.transform(X).astype(np.float32) if self.scaler is not None else X

    def calibrate(self, Xva, yva):
        self.cal = Platt().fit(self.raw_score(Xva), yva)
        return self

    def proba(self, X):
        z = self.raw_score(X)
        return self.cal(z) if self.cal is not None else expit(z)

    def uncal_proba(self, X):
        return expit(self.raw_score(X))


def _p2z(p):
    p = np.clip(p, EPS, 1 - EPS)
    return logit(p)


# ------------------------------------------------------------------ classical
class LR(Base):
    name = "LR"

    def fit(self, X, y, Xva=None, yva=None, fam=None):
        self._fit_scaler(X)
        self.m = LogisticRegression(C=1.0, max_iter=2000, random_state=self.seed).fit(self._t(X), y)
        return self

    def raw_score(self, X):
        return self.m.decision_function(self._t(X))


class RF(Base):
    name = "RF"
    needs_scaling = False

    def fit(self, X, y, Xva=None, yva=None, fam=None):
        self.m = RandomForestClassifier(n_estimators=200, min_samples_leaf=1, n_jobs=2,
                                        random_state=self.seed).fit(X, y)
        return self

    def raw_score(self, X):
        return _p2z(self.m.predict_proba(X)[:, 1] * 0.98 + 0.01)


class XGB(Base):
    name = "XGBoost"
    needs_scaling = False

    def fit(self, X, y, Xva=None, yva=None, fam=None):
        self.m = xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                                   subsample=0.9, colsample_bytree=0.9, n_jobs=2,
                                   tree_method="hist", random_state=self.seed,
                                   eval_metric="logloss").fit(X, y)
        return self

    def raw_score(self, X):
        return self.m.predict(X, output_margin=True)


class LGBM(Base):
    name = "LightGBM"
    needs_scaling = False

    def fit(self, X, y, Xva=None, yva=None, fam=None):
        self.m = lgb.LGBMClassifier(n_estimators=300, num_leaves=31, learning_rate=0.05,
                                    subsample=0.9, subsample_freq=1, colsample_bytree=0.9,
                                    n_jobs=2, random_state=self.seed, verbose=-1).fit(X, y)
        return self

    def raw_score(self, X):
        return self.m.predict(X, raw_score=True)


class SVM(Base):
    name = "SVM"
    max_train = 6000

    def fit(self, X, y, Xva=None, yva=None, fam=None):
        rng = np.random.default_rng(self.seed)
        if len(X) > self.max_train:
            idx = rng.choice(len(X), self.max_train, replace=False)
            X, y = X[idx], y[idx]
        self._fit_scaler(X)
        self.m = SVC(C=1.0, kernel="rbf", gamma="scale", random_state=self.seed).fit(self._t(X), y)
        return self

    def raw_score(self, X):
        return self.m.decision_function(self._t(X))


# ------------------------------------------------------------------ deep nets
class _MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2),
                                 nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
                                 nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


class _SE(nn.Module):
    def __init__(self, c, r=8):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(c, max(c // r, 4)), nn.ReLU(), nn.Linear(max(c // r, 4), c), nn.Sigmoid())

    def forward(self, x):
        w = self.fc(x.mean(-1))
        return x * w.unsqueeze(-1)


class _ResSE(nn.Module):
    def __init__(self, cin, cout, k=3):
        super().__init__()
        self.c1 = nn.Conv1d(cin, cout, k, padding=k // 2, bias=False)
        self.b1 = nn.BatchNorm1d(cout)
        self.c2 = nn.Conv1d(cout, cout, k, padding=k // 2, bias=False)
        self.b2 = nn.BatchNorm1d(cout)
        self.se = _SE(cout)
        self.skip = nn.Conv1d(cin, cout, 1, bias=False) if cin != cout else nn.Identity()
        self.drop = nn.Dropout(0.15)

    def forward(self, x):
        h = self.drop(F.relu(self.b1(self.c1(x))))
        h = self.se(self.b2(self.c2(h)))
        return F.relu(h + self.skip(x))


class _CNN(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(1, 32, 3, padding=1), nn.BatchNorm1d(32), nn.ReLU())
        self.b1 = _ResSE(32, 32)
        self.b2 = _ResSE(32, 64)
        self.pool = nn.MaxPool1d(2)
        self.head = nn.Sequential(nn.Linear(64 * 2, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 1))

    def forward(self, x):
        h = self.stem(x.unsqueeze(1))
        h = self.pool(self.b1(h))
        h = self.b2(h)
        h = torch.cat([h.mean(-1), h.amax(-1)], 1)
        return self.head(h).squeeze(-1)


class _RNN(nn.Module):
    def __init__(self, d, cell="lstm", hidden=64):
        super().__init__()
        R = nn.LSTM if cell == "lstm" else nn.GRU
        self.inp = nn.Linear(1, 16)
        self.rnn = R(16, hidden, num_layers=1, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        h, _ = self.rnn(F.relu(self.inp(x.unsqueeze(-1))))
        return self.head(h[:, -1]).squeeze(-1)


class _FTT(nn.Module):
    """Feature-tokenizer Transformer (Gorishniy et al., 2021)."""

    def __init__(self, d, dm=32, heads=4, layers=2):
        super().__init__()
        self.w = nn.Parameter(torch.randn(d, dm) * 0.1)
        self.b = nn.Parameter(torch.zeros(d, dm))
        self.cls = nn.Parameter(torch.zeros(1, 1, dm))
        enc = nn.TransformerEncoderLayer(dm, heads, dim_feedforward=64, dropout=0.1,
                                         batch_first=True, norm_first=True)
        self.tr = nn.TransformerEncoder(enc, layers)
        self.head = nn.Sequential(nn.LayerNorm(dm), nn.ReLU(), nn.Linear(dm, 1))

    def forward(self, x):
        tok = x.unsqueeze(-1) * self.w + self.b
        tok = torch.cat([self.cls.expand(len(x), -1, -1), tok], 1)
        return self.head(self.tr(tok)[:, 0]).squeeze(-1)


class _FGNN(nn.Module):
    """Feature-graph isomorphism network: nodes = features, edges = |corr|>0.3
    on the training split; GIN-style updates; mean+max read-out."""

    def __init__(self, d, A, dm=16, layers=2):
        super().__init__()
        self.register_buffer("A", A)
        self.w = nn.Parameter(torch.randn(d, dm) * 0.1)
        self.b = nn.Parameter(torch.zeros(d, dm))
        self.eps = nn.Parameter(torch.zeros(layers))
        self.mlps = nn.ModuleList([nn.Sequential(nn.Linear(dm, dm), nn.ReLU(), nn.Linear(dm, dm))
                                   for _ in range(layers)])
        self.head = nn.Sequential(nn.Linear(2 * dm, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 1))

    def forward(self, x):
        h = x.unsqueeze(-1) * self.w + self.b
        for l, mlp in enumerate(self.mlps):
            h = F.relu(mlp((1 + self.eps[l]) * h + torch.einsum("ij,bjk->bik", self.A, h)))
        return self.head(torch.cat([h.mean(1), h.amax(1)], 1)).squeeze(-1)


class TorchModel(Base):
    epochs = 20
    lr = 1e-3
    batch = 512
    patience = 3

    def build(self, d, X):
        raise NotImplementedError

    def fit(self, X, y, Xva=None, yva=None, fam=None):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self._fit_scaler(X)
        Xt = torch.tensor(self._t(X))
        yt = torch.tensor(y.astype(np.float32))
        self.net = self.build(X.shape[1], self._t(X))
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=1e-4)
        pos_w = torch.tensor(float((y == 0).sum()) / max(float((y == 1).sum()), 1.0))
        lossf = nn.BCEWithLogitsLoss(pos_weight=pos_w)
        best, best_state, bad = np.inf, None, 0
        self.history = []
        if Xva is not None:
            Xv, yv = torch.tensor(self._t(Xva)), torch.tensor(yva.astype(np.float32))
        n = len(Xt)
        for ep in range(self.epochs):
            self.net.train()
            perm = torch.randperm(n)
            tl = 0.0
            for i in range(0, n, self.batch):
                j = perm[i:i + self.batch]
                if len(j) < 2:
                    continue
                opt.zero_grad()
                l = lossf(self.net(Xt[j]), yt[j])
                l.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 5.0)
                opt.step()
                tl += l.item() * len(j)
            tl /= n
            if Xva is not None:
                self.net.eval()
                with torch.no_grad():
                    vl = sum(lossf(self.net(Xv[i:i + 4096]), yv[i:i + 4096]).item() * len(yv[i:i + 4096])
                             for i in range(0, len(Xv), 4096)) / len(Xv)
                self.history.append((ep + 1, tl, vl))
                if vl < best - 1e-4:
                    best, bad = vl, 0
                    best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
                else:
                    bad += 1
                    if bad >= self.patience:
                        break
            else:
                self.history.append((ep + 1, tl, np.nan))
        if best_state is not None:
            self.net.load_state_dict(best_state)
        self.net.eval()
        return self

    def finetune(self, X, y, epochs=10, lr=3e-4):
        """Few-shot adaptation on k labelled target samples (source scaler kept)."""
        if len(np.unique(y)) < 2:
            return self
        Xt, yt = torch.tensor(self._t(X)), torch.tensor(y.astype(np.float32))
        opt = torch.optim.AdamW(self.net.parameters(), lr=lr, weight_decay=1e-4)
        lossf = nn.BCEWithLogitsLoss()
        self.net.train()
        for _ in range(epochs):
            perm = torch.randperm(len(Xt))
            for i in range(0, len(Xt), 64):
                j = perm[i:i + 64]
                if len(j) < 2:
                    continue
                opt.zero_grad()
                lossf(self.net(Xt[j]), yt[j]).backward()
                opt.step()
        self.net.eval()
        return self

    def raw_score(self, X):
        self.net.eval()
        Xt = torch.tensor(self._t(X))
        with torch.no_grad():
            return np.concatenate([self.net(Xt[i:i + 8192]).numpy() for i in range(0, len(Xt), 8192)])


class DNN(TorchModel):
    name = "DNN"

    def build(self, d, X):
        return _MLP(d)


class CNN1D(TorchModel):
    name = "ResSE-CNN1D"

    def build(self, d, X):
        return _CNN(d)


class LSTM(TorchModel):
    name = "LSTM"
    epochs = 12

    def build(self, d, X):
        return _RNN(d, "lstm")


class GRU(TorchModel):
    name = "GRU"
    epochs = 12

    def build(self, d, X):
        return _RNN(d, "gru")


class FTTransformer(TorchModel):
    name = "FT-Transformer"
    epochs = 10
    batch = 1024

    def build(self, d, X):
        return _FTT(d)


class FGNN(TorchModel):
    name = "FG-GNN"

    def build(self, d, X):
        C = np.nan_to_num(np.corrcoef(X[:20000].T))
        A = (np.abs(C) > 0.3).astype(np.float32)
        np.fill_diagonal(A, 0)
        deg = A.sum(1, keepdims=True) + 1.0
        return _FGNN(d, torch.tensor(A / deg))


# ------------------------------------------------- Bayesian prototypical meta-learner
class _Enc(nn.Module):
    def __init__(self, d, e=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 128), nn.ReLU(), nn.Dropout(0.2),
                                 nn.Linear(128, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, e))

    def forward(self, x):
        return self.net(x)


class BML(Base):
    """Bayesian prototypical meta-learner: episodic training (Snell et al., 2017)
    over known families, Monte-Carlo dropout at inference (Gal & Ghahramani, 2016).
    Attack score = log-odds of 1 - p(Normal | x) under a softmax over negative
    squared distances to class prototypes, averaged over T stochastic passes."""
    name = "B-ProtoMeta"
    episodes = 1500
    n_way = 5
    k_shot = 10
    n_query = 20
    T = 10

    def fit(self, X, y, Xva=None, yva=None, fam=None):
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        self._fit_scaler(X)
        Xs = torch.tensor(self._t(X))
        fam = np.asarray(fam if fam is not None else y).astype(str)
        self.classes = sorted(np.unique(fam))
        self.normal_label = "Normal"
        idx_by = {c: np.where(fam == c)[0] for c in self.classes}
        usable = [c for c in self.classes if len(idx_by[c]) >= self.k_shot + 5]
        self.enc = _Enc(X.shape[1])
        opt = torch.optim.Adam(self.enc.parameters(), lr=1e-3)
        self.history = []
        for ep in range(self.episodes):
            way = min(self.n_way, len(usable))
            cls = list(rng.choice(usable, way, replace=False))
            if self.normal_label in usable and self.normal_label not in cls:
                cls[0] = self.normal_label
            sup, qry, ql = [], [], []
            for ci, c in enumerate(cls):
                ii = rng.choice(idx_by[c], min(len(idx_by[c]), self.k_shot + self.n_query), replace=False)
                sup.append(Xs[ii[:self.k_shot]])
                q = ii[self.k_shot:]
                qry.append(Xs[q])
                ql += [ci] * len(q)
            self.enc.train()
            protos = torch.stack([self.enc(s).mean(0) for s in sup])
            zq = self.enc(torch.cat(qry))
            logits = -torch.cdist(zq, protos) ** 2
            loss = F.cross_entropy(logits, torch.tensor(ql))
            opt.zero_grad()
            loss.backward()
            opt.step()
            if ep % 50 == 0:
                self.history.append((ep, loss.item()))
        # full-data prototypes (posterior mean over MC-dropout passes)
        self.support = {c: Xs[rng.choice(idx_by[c], min(len(idx_by[c]), 500), replace=False)] for c in self.classes}
        self.enc.eval()
        with torch.no_grad():
            self.protos = torch.stack([self.enc(self.support[c]).mean(0) for c in self.classes])
        return self

    def adapt(self, X, fam):
        """Few-shot adaptation: add / refresh prototypes from k labelled target samples."""
        Xs = torch.tensor(self._t(X))
        fam = np.asarray(fam).astype(str)
        self.enc.eval()
        with torch.no_grad():
            for c in np.unique(fam):
                zc = self.enc(Xs[fam == c]).mean(0)
                if c in self.classes:
                    i = self.classes.index(c)
                    self.protos[i] = 0.5 * self.protos[i] + 0.5 * zc
                else:
                    self.classes.append(c)
                    self.protos = torch.cat([self.protos, zc.unsqueeze(0)])
        return self

    def _pnormal(self, X, T=None):
        T = T or self.T
        Xs = torch.tensor(self._t(X))
        ni = self.classes.index(self.normal_label)
        outs = []
        self.enc.train()  # MC dropout
        with torch.no_grad():
            for _ in range(T):
                z = torch.cat([self.enc(Xs[i:i + 8192]) for i in range(0, len(Xs), 8192)])
                p = F.softmax(-torch.cdist(z, self.protos) ** 2 / z.shape[1], 1)[:, ni]
                outs.append(p.numpy())
        self.enc.eval()
        P = np.stack(outs)
        self.last_epistemic = P.std(0)
        return P.mean(0)

    def raw_score(self, X):
        return _p2z(1.0 - self._pnormal(X))


# ------------------------------------------------------------------ hybrid
class Hybrid(Base):
    """Calibrated heterogeneous soft-vote: LightGBM (axis-aligned splits),
    ResSE-CNN1D (local convolutional patterns) and B-ProtoMeta (metric-space
    distances).  Members are Platt-calibrated on the validation split and their
    calibrated log-odds are averaged; the result is re-calibrated."""
    name = "CZT-Hybrid"

    def __init__(self, seed=0, members=None):
        super().__init__(seed)
        self.members = members

    def fit(self, X, y, Xva=None, yva=None, fam=None):
        if self.members is None:
            self.members = [LGBM(self.seed).fit(X, y), CNN1D(self.seed).fit(X, y, Xva, yva),
                            BML(self.seed).fit(X, y, fam=fam)]
        for m in self.members:
            if m.cal is None:
                m.calibrate(Xva, yva)
        return self

    def raw_score(self, X):
        return np.mean([_p2z(m.proba(X)) for m in self.members], 0)


REGISTRY = {"LR": LR, "RF": RF, "XGBoost": XGB, "LightGBM": LGBM, "SVM": SVM, "DNN": DNN,
            "ResSE-CNN1D": CNN1D, "LSTM": LSTM, "GRU": GRU, "FT-Transformer": FTTransformer,
            "FG-GNN": FGNN, "B-ProtoMeta": BML}
MODEL_ORDER = list(REGISTRY) + ["CZT-Hybrid"]


def train_all(Xtr, ytr, Xva, yva, famtr, seed=0, names=None, log=print):
    """Train every model (and the hybrid re-using fitted members); returns dict
    name -> fitted, calibrated model and dict of training times (s)."""
    names = names or MODEL_ORDER
    fitted, times = {}, {}
    for n in names:
        if n == "CZT-Hybrid":
            continue
        t0 = time.time()
        m = REGISTRY[n](seed).fit(Xtr, ytr, Xva, yva, fam=famtr)
        times[n] = time.time() - t0
        m.calibrate(Xva, yva)
        fitted[n] = m
        log(f"    {n:15s} trained in {times[n]:7.1f}s")
    if "CZT-Hybrid" in names:
        t0 = time.time()
        h = Hybrid(seed, members=[fitted["LightGBM"], fitted["ResSE-CNN1D"], fitted["B-ProtoMeta"]])
        h.fit(Xtr, ytr, Xva, yva)
        h.calibrate(Xva, yva)
        fitted["CZT-Hybrid"] = h
        times["CZT-Hybrid"] = times["LightGBM"] + times["ResSE-CNN1D"] + times["B-ProtoMeta"] + (time.time() - t0)
    return fitted, times
