"""Builds every table (CSV + Markdown) and figure of the manuscript from the
raw experiment outputs in results/."""
import os, json, pickle, sys
from zdt.io import load_scores
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from scipy import stats
from sklearn.metrics import roc_curve, precision_recall_curve
from zdt.style import setup, panel, MODEL_COLOR, FAMILY_OF, FAM_COLOR, MARKERS, OKABE
from zdt.models import MODEL_ORDER
from zdt.metrics import reliability, ece

setup()
R, FIG, TAB = "results/", "figures/", "tables/"
os.makedirs(FIG, exist_ok=True); os.makedirs(TAB, exist_ok=True)
FAMS = ["Generic", "Exploits", "Fuzzers", "DoS", "Reconnaissance", "Analysis", "Backdoor", "Shellcode", "Worms"]
FIGNUM = {}  # filled by caller order


def save(fig, name):
    fig.savefig(FIG + name + ".png"); plt.close(fig)


def md_table(df, path, floatfmt=3):
    def f(v):
        if isinstance(v, (float, np.floating)):
            return "" if np.isnan(v) else f"{v:.{floatfmt}f}"
        return str(v)
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(f(v) for v in r.values) + " |")
    open(path, "w").write("\n".join(lines) + "\n")
    df.to_csv(path.replace(".md", ".csv"), index=False)


def pm(m, s, d=3):
    return f"{m:.{d}f} ± {s:.{d}f}"


# ================================================================= data figure
def fig_data(name):
    from zdt.data import load_unsw, unsw_matrix, load_kdd
    meta = json.load(open(R + "unsw_meta.json"))
    raw = {"Normal": 93000, "Generic": 58871, "Exploits": 44525, "Fuzzers": 24246, "DoS": 16353,
           "Reconnaissance": 13987, "Analysis": 2677, "Backdoor": 2329, "Shellcode": 1511, "Worms": 174}
    ded = meta["class_counts"]
    fig = plt.figure(figsize=(7.2, 5.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.05], hspace=0.55, wspace=0.28)
    ax = fig.add_subplot(gs[0, :])
    cl = list(raw)
    x = np.arange(len(cl))
    ax.bar(x - 0.2, [raw[c] for c in cl], 0.38, color="#9CC8E6", label="Released (257,673 records)", edgecolor="white", lw=0.8)
    ax.bar(x + 0.2, [ded.get(c, 0) for c in cl], 0.38, color="#0072B2", label="After Eqs (1)-(2) (151,239 records)", edgecolor="white", lw=0.8)
    for i, c in enumerate(cl):
        red = 100 * (1 - ded.get(c, 0) / raw[c])
        ax.text(i, max(raw[c], ded.get(c, 0)) * 1.25, f"-{red:.0f}%", ha="center", fontsize=6.8, color="#333")
    ax.set_yscale("log"); ax.set_ylim(80, 4e5)
    ax.set_xticks(x); ax.set_xticklabels(cl, rotation=20, ha="right")
    ax.set_ylabel("Records (log scale)"); ax.legend(loc="upper right", ncol=2)
    ax.set_title("UNSW-NB15 class sizes before and after de-duplication", loc="left")
    panel(ax, "a", -0.07)
    # (b) t-SNE
    ax = fig.add_subplot(gs[1, 0])
    df, _ = load_unsw(verbose=False)
    X, _ = unsw_matrix(df)
    rng = np.random.default_rng(0)
    fam = df["attack_cat"].values
    idx = np.concatenate([rng.choice(np.where(fam == c)[0], min(450, (fam == c).sum()), replace=False) for c in cl])
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler
    E = TSNE(2, perplexity=35, random_state=0, init="pca").fit_transform(StandardScaler().fit_transform(X[idx]))
    cols = ["#BBBBBB"] + OKABE[:7] + ["#882255", "#44AA99"]
    for c, col in zip(cl, cols):
        m = fam[idx] == c
        ax.scatter(E[m, 0], E[m, 1], s=3, color=col, label=c, alpha=0.8, lw=0, rasterized=True)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    ax.set_title("t-SNE of cleaned UNSW-NB15", loc="left")
    ax.legend(markerscale=3, fontsize=5.6, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.02), handletextpad=0.05,
              columnspacing=0.4)
    panel(ax, "b")
    # (c) NSL-KDD train vs test
    ax = fig.add_subplot(gs[1, 1])
    km = json.load(open(R + "kdd_meta.json"))
    cats = ["Normal", "DoS", "Probe", "R2L", "U2R"]
    tr = np.array([km["train_counts"].get(c, 0) for c in cats], float); te = np.array([km["test_counts"].get(c, 0) for c in cats], float)
    nov = np.array([km["novel_by_category"].get(c, 0) for c in cats], float)
    xx = np.arange(len(cats))
    ax.bar(xx - 0.2, 100 * tr / tr.sum(), 0.38, color="#9CC8E6", label="KDDTrain+", edgecolor="white")
    ax.bar(xx + 0.2, 100 * (te - nov) / te.sum(), 0.38, color="#0072B2", label="KDDTest+ (seen types)", edgecolor="white")
    ax.bar(xx + 0.2, 100 * nov / te.sum(), 0.38, bottom=100 * (te - nov) / te.sum(), color="#D55E00",
           label="KDDTest+ (17 unseen types)", edgecolor="white", hatch="////", lw=0.5)
    ax.set_xticks(xx); ax.set_xticklabels(cats); ax.set_ylabel("Share of file (%)")
    ax.set_title("NSL-KDD category mix", loc="left"); ax.legend(fontsize=6.3, loc="upper right")
    panel(ax, "c")
    save(fig, name)


# ================================================================= E1
def e1_tables(tabname, figname):
    df = pd.DataFrame(json.load(open(R + "e1_iid.json")))
    cols = ["accuracy", "precision", "recall", "f1", "auroc", "auprc", "tpr_at_fpr1", "ece", "brier"]
    g = df.groupby("model")[cols]
    mean, sd = g.mean().reindex(MODEL_ORDER), g.std().reindex(MODEL_ORDER)
    out = pd.DataFrame({"Model": MODEL_ORDER})
    names = {"accuracy": "Accuracy", "precision": "Precision", "recall": "Recall", "f1": "F1", "auroc": "AUROC",
             "auprc": "AUPRC", "tpr_at_fpr1": "TPR@1%FPR", "ece": "ECE", "brier": "Brier"}
    for c in cols:
        out[names[c]] = [pm(mean.loc[m, c], sd.loc[m, c]) for m in MODEL_ORDER]
    md_table(out, TAB + tabname + ".md")
    # heatmap
    show = ["accuracy", "precision", "recall", "f1", "auroc", "auprc", "tpr_at_fpr1", "ece"]
    M = mean[show].values
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    Z = M.copy()
    Z[:, -1] = 1 - Z[:, -1] / max(Z[:, -1].max(), 1e-9) * 0.5  # lower ECE -> darker
    norm = (Z - Z.min(0)) / (Z.max(0) - Z.min(0) + 1e-12)
    ax.imshow(norm, cmap="Blues", aspect="auto", vmin=-0.15, vmax=1.1)
    for i in range(M.shape[0]):
        best = None
        for j in range(M.shape[1]):
            v = M[i, j]
            colbest = (M[:, j].min() if show[j] == "ece" else M[:, j].max())
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=7,
                    color="white" if norm[i, j] > 0.6 else "#111", fontweight="bold" if np.isclose(v, colbest) else "normal")
    ax.set_xticks(range(len(show))); ax.set_xticklabels([names[c] + (" (lower is better)" if c == "ece" else "") for c in show], rotation=25, ha="right")
    ax.set_yticks(range(len(MODEL_ORDER))); ax.set_yticklabels(MODEL_ORDER); ax.grid(False)
    for t, m in zip(ax.get_yticklabels(), MODEL_ORDER):
        t.set_color(FAM_COLOR[FAMILY_OF[m]])
    ax.set_title("Protocol P1: i.i.d. 5-fold cross-validation on cleaned UNSW-NB15 (mean of folds)", loc="left", fontsize=9)
    fig.text(0.01, -0.09, "Shading is scaled within each column (darker = better); bold = best in column. "
             "Label colours: blue classical, orange deep, green graph/meta, red hybrid.", fontsize=6.5, color="#444")
    save(fig, figname)
    return mean, sd, df


# ================================================================= E2
def e2_load():
    df = pd.DataFrame(json.load(open(R + "e2_fh.json")))
    return df


def e2_table(tabname):
    df = e2_load()
    cols = ["zdr", "known_recall", "test_fpr", "auroc_zd", "auprc_zd", "tpr_fpr1_zd"]
    g = df.groupby("model")[cols]
    mean, sd = g.mean().reindex(MODEL_ORDER), g.std().reindex(MODEL_ORDER)
    med = g.median().reindex(MODEL_ORDER)
    worst = df.groupby("model")["auroc_zd"].min().reindex(MODEL_ORDER)
    out = pd.DataFrame({"Model": MODEL_ORDER})
    out["ZDR at frozen threshold"] = [pm(mean.loc[m, "zdr"], sd.loc[m, "zdr"]) for m in MODEL_ORDER]
    out["Known-family recall"] = [pm(mean.loc[m, "known_recall"], sd.loc[m, "known_recall"]) for m in MODEL_ORDER]
    out["Test FPR"] = [f"{mean.loc[m, 'test_fpr']:.3f}" for m in MODEL_ORDER]
    out["AUROC (unseen)"] = [pm(mean.loc[m, "auroc_zd"], sd.loc[m, "auroc_zd"]) for m in MODEL_ORDER]
    out["Worst-family AUROC"] = [f"{worst.loc[m]:.3f}" for m in MODEL_ORDER]
    out["AUPRC (unseen)"] = [pm(mean.loc[m, "auprc_zd"], sd.loc[m, "auprc_zd"]) for m in MODEL_ORDER]
    out["TPR@1%FPR (unseen)"] = [pm(mean.loc[m, "tpr_fpr1_zd"], sd.loc[m, "tpr_fpr1_zd"]) for m in MODEL_ORDER]
    ranks = df.pivot(index="family", columns="model", values="auroc_zd")[MODEL_ORDER].rank(axis=1, ascending=False).mean()
    out["Mean rank (AUROC)"] = [f"{ranks[m]:.2f}" for m in MODEL_ORDER]
    md_table(out, TAB + tabname + ".md")
    return mean, sd, ranks


def e2_heatmap(name):
    df = e2_load()
    fams = [f for f in FAMS if f in set(df.family)]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.6), gridspec_kw={"wspace": 0.28})
    for ax, key, title, letter in [(axes[0], "zdr", "Zero-day detection rate\n(frozen 1% FPR threshold)", "a"),
                                   (axes[1], "auroc_zd", "AUROC, benign vs\nunseen family", "b")]:
        P = df.pivot(index="model", columns="family", values=key).reindex(MODEL_ORDER)[fams]
        vmin = 0 if key == "zdr" else 0.5
        im = ax.imshow(P.values, cmap="YlGnBu", vmin=vmin, vmax=1, aspect="auto")
        for i in range(P.shape[0]):
            for j in range(P.shape[1]):
                v = P.values[i, j]
                ax.text(j, i, f"{v:.2f}" if key == "zdr" else f"{v:.2f}", ha="center", va="center", fontsize=5.8,
                        color="white" if (v - vmin) / (1 - vmin) > 0.62 else "#111")
        ax.set_xticks(range(len(fams))); ax.set_xticklabels(fams, rotation=40, ha="right", fontsize=7)
        ax.set_yticks(range(len(MODEL_ORDER)))
        ax.set_yticklabels(MODEL_ORDER if letter == "a" else [], fontsize=7.2)
        ax.grid(False); ax.set_title(title, fontsize=8.2, loc="left", pad=4)
        cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03); cb.ax.tick_params(labelsize=6.5)
        panel(ax, letter, -0.1 if letter == "b" else -0.5, 1.1)
    for t, m in zip(axes[0].get_yticklabels(), MODEL_ORDER):
        t.set_color(FAM_COLOR[FAMILY_OF[m]])
    save(fig, name)


def gap_figure(name, e1mean):
    df = e2_load()
    fh = df.groupby("model")[["auroc_zd", "tpr_fpr1_zd"]].mean().reindex(MODEL_ORDER)
    fw = df.groupby("model")[["auroc_zd", "tpr_fpr1_zd"]].min().reindex(MODEL_ORDER)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.1), sharey=True, gridspec_kw={"wspace": 0.12})
    y = np.arange(len(MODEL_ORDER))[::-1]
    for ax, (a, b, lab, letter) in zip(axes, [("auroc", "auroc_zd", "AUROC", "a"),
                                              ("tpr_at_fpr1", "tpr_fpr1_zd", "TPR at 1% FPR", "b")]):
        for yi, m in zip(y, MODEL_ORDER):
            v1, v2, v3 = e1mean.loc[m, a], fh.loc[m, b], fw.loc[m, b]
            ax.plot([v3, v1], [yi, yi], color="#DDDDDD", lw=2.4, zorder=1, solid_capstyle="round")
            ax.scatter(v1, yi, s=34, color="#0072B2", zorder=3, edgecolor="white", lw=0.8)
            ax.scatter(v2, yi, s=34, color="#D55E00", zorder=3, marker="D", edgecolor="white", lw=0.8)
            ax.scatter(v3, yi, s=30, facecolor="white", edgecolor="#D55E00", zorder=3, marker="D", lw=1.1)
        ax.set_xlabel(lab); ax.set_yticks(y); ax.set_yticklabels(MODEL_ORDER)
        panel(ax, letter, -0.05 if letter == "b" else -0.36)
    axes[0].set_xlim(0.7, 1.005); axes[1].set_xlim(-0.02, 0.8)
    for t, m in zip(axes[0].get_yticklabels(), MODEL_ORDER):
        t.set_color(FAM_COLOR[FAMILY_OF[m]])
    h = [Line2D([], [], marker="o", color="#0072B2", ls="", label="P1 i.i.d. CV (mean of 5 folds)"),
         Line2D([], [], marker="D", color="#D55E00", ls="", label="P2 family hold-out, mean of 9 families"),
         Line2D([], [], marker="D", mfc="white", mec="#D55E00", color="#D55E00", ls="", label="P2 family hold-out, worst family")]
    fig.legend(handles=h, loc="lower center", ncol=3, fontsize=7, bbox_to_anchor=(0.55, -0.04))
    save(fig, name)


def roc_pr_figure(name):
    store = load_scores(R + "e2_scores.npz")
    show = ["LR", "XGBoost", "LightGBM", "DNN", "ResSE-CNN1D", "FT-Transformer", "B-ProtoMeta", "CZT-Hybrid"]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7), gridspec_kw={"wspace": 0.38})
    for m in show:
        ys, ps = [], []
        for f, st in store.items():
            mk = st["groups"] != "known"
            ys.append((st["groups"][mk] == "zeroday").astype(int)); ps.append(st["scores"][m]["p"][mk])
        y, p = np.concatenate(ys), np.concatenate(ps)
        fpr, tpr, _ = roc_curve(y, p)
        prec, rec, _ = precision_recall_curve(y, p)
        lw = 2.2 if m == "CZT-Hybrid" else 1.2
        ls = "-" if FAMILY_OF[m] in ("Classical", "Hybrid (proposed)") else "--"
        axes[0].plot(fpr, tpr, color=MODEL_COLOR[m], lw=lw, ls=ls, label=m)
        axes[1].plot(fpr, tpr, color=MODEL_COLOR[m], lw=lw, ls=ls)
        keep = rec > 0.002
        axes[2].plot(rec[keep], prec[keep], color=MODEL_COLOR[m], lw=lw, ls=ls)
    axes[0].plot([0, 1], [0, 1], color="#999", lw=0.8, ls=":")
    axes[0].set_xlabel("False-positive rate"); axes[0].set_ylabel("True-positive rate"); axes[0].set_title("Pooled ROC", loc="left")
    axes[1].set_xscale("log"); axes[1].set_xlim(1e-3, 1); axes[1].axvline(0.01, color="#666", lw=0.8, ls=":")
    axes[1].set_xlabel("False-positive rate (log)"); axes[1].set_title("Low-FPR region", loc="left")
    axes[1].text(0.011, 0.03, "1% FPR", fontsize=6.5, color="#444")
    axes[2].set_xlabel("Recall"); axes[2].set_ylabel("Precision"); axes[2].set_title("Pooled precision-recall", loc="left")
    for ax, l in zip(axes, "abc"):
        panel(ax, l, -0.22)
    axes[0].legend(fontsize=5.8, loc="lower right", handlelength=1.8)
    save(fig, name)


def cd_diagram(name, tabname):
    df = e2_load()
    P = df.pivot(index="family", columns="model", values="auroc_zd")[MODEL_ORDER]
    N, k = P.shape
    ranks = P.rank(axis=1, ascending=False)
    Rj = ranks.mean()
    chi2 = 12 * N / (k * (k + 1)) * (np.sum(Rj.values ** 2) - k * (k + 1) ** 2 / 4)
    FF = (N - 1) * chi2 / (N * (k - 1) - chi2)
    p_ff = 1 - stats.f.cdf(FF, k - 1, (k - 1) * (N - 1))
    fr = stats.friedmanchisquare(*[P[m].values for m in MODEL_ORDER])
    q = stats.studentized_range.ppf(0.95, k, np.inf) / np.sqrt(2)
    CD = q * np.sqrt(k * (k + 1) / (6 * N))
    # Wilcoxon hybrid vs others, Holm
    rows = []
    for m in MODEL_ORDER:
        if m == "CZT-Hybrid":
            continue
        for key in ["auroc_zd", "zdr"]:
            Q = df.pivot(index="family", columns="model", values=key)
            d = Q["CZT-Hybrid"] - Q[m]
            try:
                w = stats.wilcoxon(Q["CZT-Hybrid"], Q[m], zero_method="wilcox")
                pv = w.pvalue
            except ValueError:
                pv = 1.0
            rows.append(dict(comparison=f"CZT-Hybrid vs {m}", metric=key, median_diff=float(np.median(d)),
                             wins=int((d > 0).sum()), losses=int((d < 0).sum()), p=pv))
    W = pd.DataFrame(rows)
    for key in ["auroc_zd", "zdr"]:
        sub = W[W.metric == key].sort_values("p")
        mcount = len(sub)
        adj, prev = [], 0
        for i, (ix, r) in enumerate(sub.iterrows()):
            v = min(1.0, max(prev, (mcount - i) * r.p)); adj.append((ix, v)); prev = v
        for ix, v in adj:
            W.loc[ix, "p_holm"] = v
    out = pd.DataFrame({"Comparison": [c.replace("CZT-Hybrid vs ", "") for c in W[W.metric == "auroc_zd"].comparison]})
    for key, lab in [("auroc_zd", "AUROC"), ("zdr", "ZDR")]:
        s = W[W.metric == key].reset_index(drop=True)
        out[f"{lab}: median diff."] = [f"{v:+.3f}" for v in s.median_diff]
        out[f"{lab}: W/L"] = [f"{a}/{b}" for a, b in zip(s.wins, s.losses)]
        out[f"{lab}: Holm p"] = [f"{v:.3f}" for v in s.p_holm]
    md_table(out, TAB + tabname + ".md")
    json.dump(dict(N=N, k=k, chi2=chi2, chi2_scipy=fr.statistic, p_chi2=fr.pvalue, FF=FF, p_FF=p_ff, q=q, CD=CD,
                   ranks=Rj.to_dict()), open(TAB + tabname + "_friedman.json", "w"), indent=1)
    # CD diagram
    order = Rj.sort_values()
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    lo, hi = 1, k
    ax.set_xlim(lo - 3.2, hi + 3.2); ax.set_ylim(0.0, 1.25); ax.axis("off")
    ax.plot([lo, hi], [1, 1], color="#333", lw=1)
    for t in range(lo, hi + 1):
        ax.plot([t, t], [1, 1.04], color="#333", lw=1); ax.text(t, 1.08, str(t), ha="center", fontsize=7.5)
    ax.plot([lo, lo + CD], [1.2, 1.2], color="#D55E00", lw=2.2)
    ax.text(lo + CD / 2, 1.13, f"CD = {CD:.2f}", ha="center", fontsize=7.5, color="#D55E00")
    names = list(order.index)
    half = (len(names) + 1) // 2
    for i, m in enumerate(names):
        r = order[m]
        if i < half:
            yy = 0.80 - i * 0.11
            ax.plot([r, r], [1, yy], color=MODEL_COLOR[m], lw=1)
            ax.plot([r, lo - 0.2], [yy, yy], color=MODEL_COLOR[m], lw=1)
            ax.text(lo - 0.28, yy, f"{m} ({r:.2f})", ha="right", va="center", fontsize=7.2,
                    fontweight="bold" if m == "CZT-Hybrid" else "normal")
        else:
            yy = 0.80 - (len(names) - 1 - i) * 0.11
            ax.plot([r, r], [1, yy], color=MODEL_COLOR[m], lw=1)
            ax.plot([r, hi + 0.2], [yy, yy], color=MODEL_COLOR[m], lw=1)
            ax.text(hi + 0.28, yy, f"({r:.2f}) {m}", ha="left", va="center", fontsize=7.2,
                    fontweight="bold" if m == "CZT-Hybrid" else "normal")
    # cliques of non-significant differences
    cl, yb = [], 0.93
    for i in range(len(names)):
        j = i
        while j + 1 < len(names) and order[names[j + 1]] - order[names[i]] <= CD:
            j += 1
        if j > i and not any(a <= i and j <= b for a, b in cl):
            cl.append((i, j))
    for n_, (a, b) in enumerate(cl):
        y0 = 0.955 - 0.03 * n_
        ax.plot([order[names[a]] - 0.03, order[names[b]] + 0.03], [y0, y0], color="#333", lw=2.4, solid_capstyle="round")
    ax.set_title(f"Average rank of unseen-family AUROC over {N} hold-out folds (Friedman p = {fr.pvalue:.2g}; "
                 f"Iman-Davenport p = {p_ff:.2g})", fontsize=8.2, loc="left")
    save(fig, name)
    return dict(CD=CD, chi2=chi2, p=fr.pvalue, FF=FF, pFF=p_ff, ranks=Rj)


def calib_figure(name, tabname):
    store = load_scores(R + "e2_scores.npz")
    df = e2_load()
    show = ["LR", "XGBoost", "LightGBM", "SVM", "ResSE-CNN1D", "B-ProtoMeta", "CZT-Hybrid"]
    fig = plt.figure(figsize=(7.2, 5.4))
    gs = fig.add_gridspec(2, 4, hspace=0.55, wspace=0.45)
    for i, m in enumerate(["XGBoost", "ResSE-CNN1D", "B-ProtoMeta", "CZT-Hybrid"]):
        ax = fig.add_subplot(gs[0, i])
        y = np.concatenate([st["y"] for st in store.values()])
        for key, col, lab, mk in [("pu", "#999999", "raw", "o"), ("p", MODEL_COLOR[m], "Platt", "s")]:
            p = np.concatenate([st["scores"][m][key] for st in store.values()])
            mp, fo, cnt = reliability(p, y, 10)
            ax.plot(mp, fo, marker=mk, ms=3.5, color=col, lw=1.3, label=f"{lab} (ECE {ece(p, y):.3f})")
        ax.plot([0, 1], [0, 1], ls=":", color="#666", lw=0.8)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
        ax.set_title(m, fontsize=8, loc="left"); ax.legend(fontsize=5.6, loc="lower right", handlelength=1.2)
        ax.set_xlabel("Mean predicted P(attack)", fontsize=7)
        if i == 0:
            ax.set_ylabel("Observed attack frequency", fontsize=7)
        panel(ax, "abcd"[i], -0.3)
    ax = fig.add_subplot(gs[1, :])
    g = df.groupby("model")[["ece_uncal", "ece_cal"]].mean().reindex(MODEL_ORDER)
    x = np.arange(len(MODEL_ORDER))
    ax.bar(x - 0.2, g.ece_uncal, 0.38, color="#BBBBBB", label="raw score", edgecolor="white")
    ax.bar(x + 0.2, g.ece_cal, 0.38, color=[MODEL_COLOR[m] for m in MODEL_ORDER], label="Platt-calibrated", edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(MODEL_ORDER, rotation=30, ha="right")
    ax.set_ylabel("ECE (benign + unseen)")
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color="#BBBBBB"), plt.Rectangle((0, 0), 1, 1, color="#0072B2")],
              labels=["raw score", "Platt-calibrated (colour = model)"], loc="upper left")
    ax.set_title("Calibration error under the family hold-out protocol (mean of 9 folds; lower is better)", loc="left", fontsize=8.5)
    panel(ax, "e", -0.07)
    save(fig, name)
    # table: ECE / Brier in P1 and P2
    e1 = pd.DataFrame(json.load(open(R + "e1_iid.json"))).groupby("model")[["ece_uncal", "ece", "brier"]].mean().reindex(MODEL_ORDER)
    g2 = df.groupby("model")[["ece_uncal", "ece_cal", "ece_all_uncal", "ece_all_cal"]].mean().reindex(MODEL_ORDER)
    out = pd.DataFrame({"Model": MODEL_ORDER, "P1 ECE raw": e1.ece_uncal.values, "P1 ECE Platt": e1.ece.values,
                        "P1 Brier Platt": e1.brier.values, "P2 ECE raw (all test)": g2.ece_all_uncal.values,
                        "P2 ECE Platt (all test)": g2.ece_all_cal.values, "P2 ECE raw (benign+unseen)": g2.ece_uncal.values,
                        "P2 ECE Platt (benign+unseen)": g2.ece_cal.values})
    md_table(out, TAB + tabname + ".md")
    return out


def importance_figure(name):
    """Permutation importance (AUROC drop) of LightGBM and CZT-Hybrid on fold-0 unseen-family test data."""
    from zdt.data import load_unsw, unsw_matrix
    from sklearn.metrics import roc_auc_score
    d = pickle.load(open(R + "e2_fold0_models.pkl", "rb"))
    df, _ = load_unsw(verbose=False); X, names = unsw_matrix(df)
    te = d["te_idx"]; st = load_scores(R + "e2_scores.npz")[FAMS[0]]
    mk = st["groups"] != "known"
    Xt, yt = X[te][mk], (st["groups"][mk] == "zeroday").astype(int)
    rng = np.random.default_rng(0)
    res = {}
    for m in ["LightGBM", "CZT-Hybrid"]:
        model = d["models"][m]
        base = roc_auc_score(yt, model.proba(Xt))
        imp = []
        # group one-hot blocks
        groups = {}
        for j, n in enumerate(names):
            key = n.split("=")[0]
            groups.setdefault(key, []).append(j)
        for gname, cols in groups.items():
            drops = []
            for r in range(3):
                Xp = Xt.copy(); perm = rng.permutation(len(Xt)); Xp[:, cols] = Xt[perm][:, cols]
                drops.append(base - roc_auc_score(yt, model.proba(Xp)))
            imp.append((gname, np.mean(drops), np.std(drops)))
        res[m] = sorted(imp, key=lambda t: -t[1])[:12]
    json.dump({m: [[g, float(v), float(sd)] for g, v, sd in imp] for m, imp in res.items()},
              open(R + "e5_permutation_importance.json", "w"), indent=1)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), gridspec_kw={"wspace": 0.75})
    for ax, (m, imp), l in zip(axes, res.items(), "ab"):
        nm = [t[0] for t in imp][::-1]; v = [t[1] for t in imp][::-1]; e = [t[2] for t in imp][::-1]
        ax.barh(nm, v, xerr=e, color=MODEL_COLOR[m], edgecolor="white", error_kw=dict(lw=0.7, capsize=1.5))
        ax.set_xlabel("AUROC drop when permuted")
        ax.set_title(f"{m}: held-out {FAMS[0]}", loc="left", fontsize=8.5)
        panel(ax, l, -0.45)
    save(fig, name)
    return res


# ================================================================= E3
def e3(tabname, figname):
    df = pd.DataFrame(json.load(open(R + "e3_kdd_novel.json")))
    cols = ["auroc", "auprc", "f1", "fpr_thr", "dr_known_thr", "dr_novel_thr", "auroc_novel", "ece"]
    g = df.groupby("model")[cols]; mean, sd = g.mean().reindex(MODEL_ORDER), g.std().reindex(MODEL_ORDER)
    out = pd.DataFrame({"Model": MODEL_ORDER})
    lab = {"auroc": "AUROC (all)", "auprc": "AUPRC (all)", "f1": "F1 at 0.5", "fpr_thr": "Test FPR at frozen threshold",
           "dr_known_thr": "DR seen types", "dr_novel_thr": "DR unseen types", "auroc_novel": "AUROC unseen types", "ece": "ECE"}
    for c in cols:
        out[lab[c]] = [pm(mean.loc[m, c], sd.loc[m, c]) if c != "fpr_thr" else f"{mean.loc[m, c]:.3f}" for m in MODEL_ORDER]
    md_table(out, TAB + tabname + ".md")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.9), gridspec_kw={"wspace": 0.55, "width_ratios": [1.1, 1]})
    ax = axes[0]; y = np.arange(len(MODEL_ORDER))[::-1]
    for yi, m in zip(y, MODEL_ORDER):
        a, b = mean.loc[m, "dr_known_thr"], mean.loc[m, "dr_novel_thr"]
        ax.plot([b, a], [yi, yi], color="#BBBBBB", lw=2.2, zorder=1)
        ax.scatter(a, yi, color="#0072B2", s=30, zorder=3, edgecolor="white")
        ax.scatter(b, yi, color="#D55E00", s=30, marker="D", zorder=3, edgecolor="white")
    ax.set_yticks(y); ax.set_yticklabels(MODEL_ORDER); ax.set_xlabel("Detection rate (threshold frozen at 1% validation FPR)", fontsize=8)
    ax.legend(handles=[Line2D([], [], marker="o", color="#0072B2", ls="", label="attack types seen in training"),
                       Line2D([], [], marker="D", color="#D55E00", ls="", label="17 unseen attack types")],
              loc="upper center", bbox_to_anchor=(0.45, -0.13), ncol=2, fontsize=6.5, frameon=False)
    for t, m in zip(ax.get_yticklabels(), MODEL_ORDER):
        t.set_color(FAM_COLOR[FAMILY_OF[m]])
    panel(ax, "a", -0.45)
    ax = axes[1]
    cats = ["DoS", "Probe", "R2L", "U2R"]
    M = np.array([[df[df.model == m][f"dr_novel_{c}"].mean() for c in cats] for m in MODEL_ORDER])
    im = ax.imshow(M, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6.3, color="white" if M[i, j] > 0.6 else "#111")
    km = json.load(open(R + "kdd_meta.json"))
    ax.set_xticks(range(4)); ax.set_xticklabels([f"{c}\n(n={km['novel_by_category'].get(c, 0)})" for c in cats], fontsize=7)
    ax.set_yticks(range(len(MODEL_ORDER))); ax.set_yticklabels(MODEL_ORDER, fontsize=7); ax.grid(False)
    ax.set_title("Unseen-type detection by category", loc="left", fontsize=8.5)
    fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02)
    panel(ax, "b", -0.45)
    save(fig, figname)
    return mean, sd


# ================================================================= E4
def e4(tabname, figname):
    df = pd.DataFrame(json.load(open(R + "e4_transfer.json")))
    models = ["LR", "RF", "XGBoost", "LightGBM", "DNN", "ResSE-CNN1D", "B-ProtoMeta", "CZT-Hybrid"]
    rows = []
    for m in models:
        r = {"Model": m}
        for (s, t), lab in [(("UNSW-NB15", "NSL-KDD"), "U→K"), (("NSL-KDD", "UNSW-NB15"), "K→U")]:
            sub = df[(df.src == s) & (df.model == m)]
            src = sub[sub.k == 0].src_auroc
            r[f"{lab} source AUROC"] = pm(src.mean(), src.std())
            for k in [0, 100, 1000]:
                v = sub[sub.k == k].auroc
                r[f"{lab} AUROC k={k}"] = pm(v.mean(), v.std())
        rows.append(r)
    md_table(pd.DataFrame(rows), TAB + tabname + ".md")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), sharey=True, gridspec_kw={"wspace": 0.08})
    for ax, (s, t), l in zip(axes, [("UNSW-NB15", "NSL-KDD"), ("NSL-KDD", "UNSW-NB15")], "ab"):
        for m in models:
            sub = df[(df.src == s) & (df.model == m)].groupby("k").auroc
            mu, sd = sub.mean(), sub.std()
            kk = np.array(mu.index, float); kk[kk == 0] = 3
            fam = FAMILY_OF[m]
            ax.plot(kk, mu.values, marker=MARKERS[fam], ms=3.5, color=MODEL_COLOR[m], lw=2.2 if m == "CZT-Hybrid" else 1.2, label=m)
            ax.fill_between(kk, mu - sd, mu + sd, color=MODEL_COLOR[m], alpha=0.10, lw=0)
        ax.set_xscale("log"); ax.set_xticks([3, 10, 50, 100, 250, 1000]); ax.set_xticklabels(["0", "10", "50", "100", "250", "1000"])
        ax.axhline(0.5, color="#999", ls=":", lw=0.8)
        ax.set_xlabel("Labelled target records k"); ax.set_title(f"{s} → {t}", loc="left")
        panel(ax, l, -0.12)
    axes[0].set_ylabel("Target AUROC (mean ± s.d., 3 seeds)")
    axes[1].legend(fontsize=6, ncol=2, loc="lower right")
    save(fig, figname)
    return df


# ================================================================= E6
def e6(tabname, figname):
    lat = pd.DataFrame(json.load(open(R + "e6_latency.json"))).set_index("model").reindex(MODEL_ORDER)
    e2 = e2_load().groupby("model")[["auroc_zd", "train_time"]].mean().reindex(MODEL_ORDER)
    out = pd.DataFrame({"Model": MODEL_ORDER, "Train time P2 (s)": e2.train_time.values,
                        "Batch latency (µs/record)": lat.batch_us_per_sample.values,
                        "Throughput (records/s)": lat.throughput_per_s.values,
                        "Single-record p50 (ms)": lat.single_ms_p50.values, "Single-record p95 (ms)": lat.single_ms_p95.values,
                        "Serialised size (kB)": lat.size_kb.values,
                        "Trainable parameters": lat.n_params.values})
    o2 = out.copy()
    o2["Throughput (records/s)"] = [f"{v:,.0f}" for v in out["Throughput (records/s)"]]
    o2["Serialised size (kB)"] = [f"{v:,.0f}" for v in out["Serialised size (kB)"]]
    o2["Train time P2 (s)"] = [f"{v:.1f}" for v in out["Train time P2 (s)"]]
    o2["Batch latency (µs/record)"] = [f"{v:.1f}" for v in out["Batch latency (µs/record)"]]
    o2["Single-record p50 (ms)"] = [f"{v:.2f}" for v in out["Single-record p50 (ms)"]]
    o2["Single-record p95 (ms)"] = [f"{v:.2f}" for v in out["Single-record p95 (ms)"]]
    o2["Trainable parameters"] = [f"{int(v):,}" if v == v and v is not None else "n/a" for v in out["Trainable parameters"]]
    md_table(o2, TAB + tabname + ".md")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), gridspec_kw={"wspace": 0.3})
    for ax, (col, lab, l) in zip(axes, [("batch_us_per_sample", "Batch inference latency (µs per record, log)", "a"),
                                          ("single_ms_p50", "Single-record latency p50 (ms, log)", "b")]):
        for m in MODEL_ORDER:
            fam = FAMILY_OF[m]
            ax.scatter(lat.loc[m, col], e2.loc[m, "auroc_zd"], s=18 + 60 * np.log10(1 + lat.loc[m, "size_kb"]) / 3,
                       color=MODEL_COLOR[m], marker=MARKERS[fam], edgecolor="white", lw=0.6, zorder=3, alpha=0.9)
            off = {"XGBoost": (-4, -9), "DNN": (3, -8), "LightGBM": (4, 3), "FG-GNN": (4, -2)}.get(m, (3, 2))
            ax.annotate(m, (lat.loc[m, col], e2.loc[m, "auroc_zd"]), fontsize=5.8, xytext=off, textcoords="offset points",
                        ha="right" if m == "XGBoost" else "left")
        # Pareto frontier
        pts = sorted([(lat.loc[m, col], e2.loc[m, "auroc_zd"]) for m in MODEL_ORDER])
        fr, best = [], -1
        for x_, y_ in pts:
            if y_ > best:
                fr.append((x_, y_)); best = y_
        ax.step([p[0] for p in fr], [p[1] for p in fr], where="post", color="#D55E00", lw=1, ls="--", alpha=0.7)
        ax.set_xscale("log"); ax.set_xlabel(lab); panel(ax, l, -0.2)
    axes[0].set_ylabel("Unseen-family AUROC (P2 mean)")
    save(fig, figname)
    return out


# ================================================================= E7
def e7(tab_pol, tab_abl, fig_load, fig_pareto, fig_staff, fig_abl):
    T = json.load(open(R + "e7_triage.json"))
    pols = ["FIFO", "Severity", "EDF", "Risk", "CZT-Whittle", "CZT-Whittle-T", "CZT-Whittle-NSGA"]
    pcol = {"FIFO": "#999999", "Severity": "#0072B2", "EDF": "#56B4E9", "Risk": "#009E73", "CZT-Whittle": "#E69F00",
            "CZT-Whittle-T": "#CC79A7", "CZT-Whittle-NSGA": "#D55E00"}
    pmk = {"FIFO": "o", "Severity": "s", "EDF": "v", "Risk": "^", "CZT-Whittle": "D", "CZT-Whittle-T": "X", "CZT-Whittle-NSGA": "P"}
    DN = {"CZT-Whittle": "CZT-D (default)", "CZT-Whittle-T": "CZT-T (tuned index)", "CZT-Whittle-NSGA": "CZT-TE (tuned index + escalation)"}
    dn = lambda p: DN.get(p, p)
    rhos = sorted(T["load"], key=float)
    # table: rho = 1.0 and 1.2
    rows = []
    for rho in ["1.0", "1.2", "1.5"]:
        for p in pols:
            v = T["load"][rho][p]
            rows.append({"ρ": rho, "Policy": dn(p), "Alerts per shift": f"{v['n_alerts'][0]:.0f}",
                         "Captured value CV": pm(*v["cv"]), "Zero-day value ZCV": pm(*v["zcv"]),
                         "Mean zero-day wait (min)": pm(*v["wait_zd"], 1), "SLA violation rate": pm(*v["sla_viol"]),
                         "Critical-attack SLA violation": pm(*v["sla_viol_critical"]), "FP share of analyst time": f"{v['fp_time_frac'][0]:.3f}"})
    md_table(pd.DataFrame(rows), TAB + tab_pol + ".md")
    # load figure
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2), gridspec_kw={"wspace": 0.3, "hspace": 0.45})
    axes = axes.ravel()
    for ax, (key, lab, l) in zip(axes, [("cv", "Captured value CV", "a"), ("zcv", "Zero-day value ZCV", "b"),
                                          ("wait_zd", "Mean zero-day wait (min)", "c"), ("sla_viol", "SLA violation rate", "d")]):
        for p in pols:
            mu = np.array([T["load"][r][p][key][0] for r in rhos]); ci = np.array([T["load"][r][p][key][1] for r in rhos])
            x = np.array(rhos, float)
            ax.plot(x, mu, marker=pmk[p], ms=3.5, color=pcol[p], lw=2 if "CZT-Whittle-" in p else 1.2, label=dn(p))
            ax.fill_between(x, mu - ci, mu + ci, color=pcol[p], alpha=0.12, lw=0)
        ax.set_xlabel("Offered load ρ"); ax.set_title(lab, fontsize=8.5, loc="left"); panel(ax, l, -0.16)
    axes[0].legend(fontsize=6.5, loc="lower left")
    save(fig, fig_load)
    # Pareto (two cross-fitting groups)
    N = json.load(open(R + "e7_nsga3.json"))
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.8), gridspec_kw={"wspace": 0.75, "width_ratios": [1, 1, 1.1]})
    labs = ["1 - CV", "1 - ZCV", "SLA viol.", "Alerts/h"]
    for ax, g, l in zip(axes[:2], ["G1", "G2"], "ab"):
        F = np.array(N[g]["F"]); kn = N[g]["knee"]
        sc = ax.scatter(1 - F[:, 1], F[:, 2], c=F[:, 3] * 100, cmap="viridis", s=20, edgecolor="white", lw=0.4, vmin=10, vmax=35)
        ax.scatter(1 - F[kn, 1], F[kn, 2], s=90, facecolor="none", edgecolor="#D55E00", lw=1.8, label="knee")
        ax.set_xlabel("ZCV on tuning pool"); ax.set_ylabel("SLA violation rate")
        ax.set_title(f"Tuned on {N[g]['tuned_on']} (for {g})", loc="left", fontsize=8)
        ax.legend(fontsize=6.3, loc="upper left"); panel(ax, l, -0.3)
    cax = axes[1].inset_axes([1.06, 0.08, 0.05, 0.84])
    cb = fig.colorbar(sc, cax=cax); cb.set_label("Alerts per hour", fontsize=7, labelpad=1); cb.ax.tick_params(labelsize=6)
    ax = axes[2]
    for g, col in [("G1", "#0072B2"), ("G2", "#D55E00")]:
        F = np.array(N[g]["F"]); kn = N[g]["knee"]
        Fn = (F - F.min(0)) / (F.max(0) - F.min(0) + 1e-12)
        for i_ in range(len(F)):
            ax.plot(range(4), Fn[i_], color=col, lw=0.5, alpha=0.35)
        ax.plot(range(4), Fn[kn], color=col, lw=2.2, marker="o", ms=4, label=f"knee for {g}")
    ax.set_xticks(range(4)); ax.set_xticklabels(labs, rotation=20); ax.set_ylabel("Normalised objective")
    ax.legend(fontsize=6.3); panel(ax, "c", -0.3)
    save(fig, fig_pareto)
    # staffing and half-life
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), gridspec_kw={"wspace": 0.45})
    cs = sorted(T["staff"], key=int)
    for ax, key, lab, l in [(axes[0], "zcv", "Zero-day value ZCV", "a"), (axes[1], "wait_zd", "Mean zero-day wait (min)", "b")]:
        for p in pols:
            mu = np.array([T["staff"][c][p][key][0] for c in cs]); ci = np.array([T["staff"][c][p][key][1] for c in cs])
            ax.plot(np.array(cs, int), mu, marker=pmk[p], ms=3.5, color=pcol[p], lw=2 if "CZT-Whittle-" in p else 1.2, label=dn(p))
            ax.fill_between(np.array(cs, int), mu - ci, mu + ci, color=pcol[p], alpha=0.12, lw=0)
        ax.set_xlabel("Analysts c (fixed event rate)"); ax.set_title(lab, fontsize=7.8, loc="left"); panel(ax, l, -0.3)
    h_, l_ = axes[0].get_legend_handles_labels()
    fig.legend(h_, l_, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=4, fontsize=6.8)
    ax = axes[2]; hls = sorted(T["halflife"], key=float)
    for p in pols:
        mu = np.array([T["halflife"][h][p]["zcv"][0] for h in hls]); ci = np.array([T["halflife"][h][p]["zcv"][1] for h in hls])
        ax.plot(np.array(hls, float), mu, marker=pmk[p], ms=3.5, color=pcol[p], lw=2 if "CZT-Whittle-" in p else 1.2, label=dn(p))
        ax.fill_between(np.array(hls, float), mu - ci, mu + ci, color=pcol[p], alpha=0.12, lw=0)
    ax.set_xscale("log"); ax.set_xticks([15, 30, 60, 120, 240]); ax.set_xticklabels(["15", "30", "60", "120", "240"])
    ax.set_xlabel("Evidence half-life h (min)"); ax.set_title("Zero-day value ZCV at ρ = 1.2", fontsize=7.8, loc="left"); panel(ax, "c", -0.3)
    save(fig, fig_staff)
    # ablation + detectors
    REN = {"CZT-Whittle, default weights": "CZT-D (default weights)",
           "CZT-Whittle-NSGA (tuned)": "CZT-TE (tuned index + escalation)",
           "  tuned, escalation fixed at q = 0.99": "  TE with escalation reset to q = 0.99 (= CZT-T)"}
    A = {REN.get(k, k).replace("  default,", "  D,").replace("  tuned,", "  TE,"): v for k, v in T["ablation"].items()}
    rows = [{"Configuration (ρ = 1.2, c = 4)": k, "CV": pm(*v["cv"]), "ZCV": pm(*v["zcv"]),
             "Mean zero-day wait (min)": pm(*v["wait_zd"], 1), "SLA violation": pm(*v["sla_viol"]),
             "Alerts per shift": f"{v['n_alerts'][0]:.0f}"} for k, v in A.items()]
    md_table(pd.DataFrame(rows), TAB + tab_abl + ".md")
    D = T["detectors"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6), gridspec_kw={"wspace": 0.35, "width_ratios": [1, 1.1]})
    ax = axes[0]
    keys = list(A.keys())
    vals = [A[k]["zcv"][0] for k in keys]; cis = [A[k]["zcv"][1] for k in keys]
    cols = ["#009E73" if k.startswith("Risk") else ("#E69F00" if k.startswith("CZT-D") else
            ("#D55E00" if k.startswith("CZT-TE") else ("#CC79A7" if "CZT-T)" in k else "#BBBBBB"))) for k in keys]
    yy = np.arange(len(keys))[::-1]
    ax.barh(yy, vals, xerr=cis, color=cols, edgecolor="white", error_kw=dict(lw=0.7, capsize=1.5))
    ax.set_yticks(yy); ax.set_yticklabels([k.strip() for k in keys], fontsize=6.3)
    ax.axvline(A["CZT-TE (tuned index + escalation)"]["zcv"][0], color="#D55E00", ls=":", lw=0.9)
    ax.set_xlabel("Zero-day value ZCV (ρ = 1.2)"); ax.set_title("Index ablation", loc="left", fontsize=8.5); panel(ax, "a", -1.25)
    ax.set_xlim(0.4, max(vals) + 0.06)
    ax = axes[1]
    e2m = e2_load().groupby("model")["auroc_zd"].mean()
    placed = []
    for m in sorted(D, key=lambda m_: -D[m_]["CZT-Whittle-NSGA"]["zcv"][0]):
        v = D[m]["CZT-Whittle-NSGA"]
        x_, y_ = e2m[m], v["zcv"][0]
        ax.errorbar(x_, y_, yerr=v["zcv"][1], fmt=MARKERS[FAMILY_OF[m]], color=MODEL_COLOR[m], ms=5, lw=0.8, capsize=1.5)
        dy = 0
        while any(abs(x_ - px) < 0.012 and abs((y_ + dy * 0.001) - py) < 0.022 for px, py in placed):
            dy -= 22
        ly = y_ + dy * 0.001
        placed.append((x_, ly))
        ax.annotate(m, (x_, y_), xytext=(x_ - 0.004 if x_ > 0.95 else x_ + 0.002, ly), fontsize=5.6,
                    ha="right" if x_ > 0.95 else "left", va="center",
                    arrowprops=dict(arrowstyle="-", lw=0.4, color="#888") if dy else None)
    r_ = stats.spearmanr([e2m[m] for m in D], [D[m]["CZT-Whittle-NSGA"]["zcv"][0] for m in D])
    ax.text(0.03, 0.95, f"Spearman r_s = {r_.correlation:.2f} (13 detectors)", transform=ax.transAxes, fontsize=7, va="top")
    ax.set_xlabel("Detector: unseen-family AUROC (P2)"); ax.set_ylabel("Triage: zero-day value ZCV")
    ax.set_title("Same tuned index, different detectors (ρ = 1.2)", loc="left", fontsize=8.5); panel(ax, "b", -0.25)
    save(fig, fig_abl)
    return T



