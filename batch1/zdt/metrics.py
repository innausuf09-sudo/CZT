"""Evaluation metrics: threshold, ranking, operating-point and calibration."""
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score, roc_curve, brier_score_loss)


def ece(p, y, bins=15):
    """Top-label expected calibration error (Guo et al., 2017) for binary p=P(attack)."""
    p = np.asarray(p); y = np.asarray(y)
    conf = np.maximum(p, 1 - p)
    pred = (p >= 0.5).astype(int)
    acc = (pred == y).astype(float)
    edges = np.linspace(0.5, 1.0, bins + 1)
    e = 0.0
    for i in range(bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1]) if i else (conf >= edges[i]) & (conf <= edges[i + 1])
        if m.any():
            e += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(e)


def reliability(p, y, bins=10):
    """Positive-class reliability curve: mean predicted P(attack) vs observed frequency."""
    edges = np.linspace(0, 1, bins + 1)
    mp, fo, cnt = [], [], []
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1]) if i < bins - 1 else (p >= edges[i]) & (p <= 1)
        if m.sum() > 0:
            mp.append(p[m].mean()); fo.append(y[m].mean()); cnt.append(int(m.sum()))
    return np.array(mp), np.array(fo), np.array(cnt)


def tpr_at_fpr(y, s, fpr_target=0.01):
    fpr, tpr, _ = roc_curve(y, s)
    return float(np.interp(fpr_target, fpr, tpr))


def fpr_at_tpr(y, s, tpr_target=0.95):
    fpr, tpr, _ = roc_curve(y, s)
    return float(np.interp(tpr_target, tpr, fpr))


def full_metrics(y, p, thr=0.5):
    yhat = (p >= thr).astype(int)
    return dict(accuracy=accuracy_score(y, yhat), precision=precision_score(y, yhat, zero_division=0),
                recall=recall_score(y, yhat, zero_division=0), f1=f1_score(y, yhat, zero_division=0),
                auroc=roc_auc_score(y, p), auprc=average_precision_score(y, p),
                tpr_at_fpr1=tpr_at_fpr(y, p, 0.01), fpr_at_tpr95=fpr_at_tpr(y, p, 0.95),
                ece=ece(p, y), brier=brier_score_loss(y, p))


def threshold_at_fpr(p_normal_val, fpr=0.01):
    """Operating threshold frozen on validation benign traffic so that FPR = fpr."""
    return float(np.quantile(p_normal_val, 1 - fpr))
