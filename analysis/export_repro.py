"""Machine-readable reproduction package (revision R1.9).

Writes into repro/:
- splits_p1.npz       P1 sample indices and the 5 stratified folds (train / validation / test row indices
                      into the cleaned UNSW-NB15 table)
- splits_p2.npz       copy of results/e10_splits.npz (train / validation / test row indices per held-out family)
- splits_p3.npz       P3 training / validation indices into KDDTrain+ for the three seeds
- masks_unsw.npz      copy of results/e15_masks.npz (exact-duplicate and label-conflict masks over the merged release)
- streams_e9.npz      catalogue-independent raw event streams of every E9 tuning and evaluation replication
                      (arrival time, group, fold, record index into the stream pool, service time, catalogue
                      uniforms) for the G1/G2 split and the base event mix
- config.json         every setting used by the revision experiments (detectors, protocols, simulator,
                      catalogue scenarios, NSGA-III, policy gradient, selection rule, seeds)
Run from code/:  python -m analysis.export_repro
"""
import json
import os
import shutil

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split

os.makedirs("repro", exist_ok=True)

# P1
from zdt.data import load_unsw, load_kdd
df, _ = load_unsw(verbose=False)
fam = df["attack_cat"].values.astype(str)
rng = np.random.default_rng(0)
idx = rng.choice(len(fam), 24000, replace=False)
out = {"sample": idx}
for k, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=0).split(idx, fam[idx])):
    tr2, va = train_test_split(tr, test_size=0.15, stratify=fam[idx][tr], random_state=k)
    out[f"fold{k}|train"], out[f"fold{k}|val"], out[f"fold{k}|test"] = idx[tr2], idx[va], idx[te]
np.savez_compressed("repro/splits_p1.npz", **out)

# P2 and masks
shutil.copy("results/e10_splits.npz", "repro/splits_p2.npz")
shutil.copy("results/e15_masks.npz", "repro/masks_unsw.npz")

# P3
tr_df, _ = load_kdd()
cat = tr_df["attack"].map(lambda a: a).values
from zdt.data import KDD_MAP
cat_tr = tr_df["category"].values
caps = {"Normal": 12000, "DoS": 6000, "Probe": 4000, "R2L": 10 ** 6, "U2R": 10 ** 6}
out = {}
for seed in range(3):
    rng = np.random.default_rng(100 + seed)
    idx = np.concatenate([rng.choice(np.where(cat_tr == c)[0], min(caps[c], (cat_tr == c).sum()), replace=False) for c in caps])
    tr, va = train_test_split(idx, test_size=0.15, stratify=cat_tr[idx], random_state=seed)
    out[f"seed{seed}|train"], out[f"seed{seed}|val"] = tr, va
np.savez_compressed("repro/splits_p3.npz", **out)

# E9 streams
import experiments.exp_triage_v2 as E
from zdt.catalogue import raw_events
arr = {}
for g in E.GROUPS:
    for key, seeds, rho in (("TU", range(E.SEED_TUNE, E.SEED_TUNE + E.N_TUNE), E.RHO_TUNE),
                            ("EV", range(E.SEED_EVAL, E.SEED_EVAL + E.REPS), E.RHO_EVAL)):
        pool = E.pools()[(key, g if key == "EV" else E.OTHER[g])]
        raw = E.raw_rate(pool, rho, (0.97, 0.02, 0.01))
        for s in seeds:
            R = raw_events(pool, raw, s, mix=(0.97, 0.02, 0.01))
            pre = f"{key}|evalgroup={g}|seed={s}"
            for k in ("t", "group", "fold", "rec", "svc", "u_match", "u_pick", "z_noise", "z_bias", "p", "nov"):
                arr[f"{pre}|{k}"] = np.asarray(R[k]).astype(str) if k in ("group", "fold") else np.asarray(R[k])
            arr[f"{pre}|raw_rate_per_min"] = np.float64(raw)
np.savez_compressed("repro/streams_e9.npz", **arr)

# config
from zdt.triage import DEFAULT
from zdt.data import UNSW_SEVERITY, KDD_SEVERITY
from experiments.exp_triage_catalogue import SCEN
cfg = dict(
    detectors=dict(XGBoost="300 trees, depth 6, lr 0.1, subsample 0.9, colsample 0.9, hist", LightGBM="300 trees, 31 leaves, lr 0.05, subsample 0.9, colsample 0.9",
                   RF="200 trees", LR="C=1, L2", SVM="RBF C=1 gamma=scale, max 6000 training records",
                   neural="AdamW lr 1e-3 wd 1e-4, batch 512, epochs 20 (LSTM/GRU 12, FT-Transformer 10 with batch 1024), patience 3 on validation loss, grad clip 5, pos_weight = n_benign/n_attack",
                   B_ProtoMeta="encoder 128-128-64 dropout 0.2, Adam 1e-3, 1500 episodes, 5-way, 10-shot, 20 query, classes with < 15 records excluded, prototypes from <= 500 records, MC dropout T=10",
                   CLAD="see zdt/clad.py; grid " + json.dumps(__import__("experiments.exp_p2_refit", fromlist=["CLAD_GRID"]).CLAD_GRID),
                   calibration="Platt scaling on the validation split; threshold = 99th percentile of benign validation scores"),
    protocols=dict(P1="24000 records, StratifiedKFold(5, seed 0), 15% validation", P2="caps train 10000/2000, val 2000/300, test 3000/3000/3000; rng seed 1",
                   P3="Normal 12000, DoS 6000, Probe 4000, all R2L and U2R; seeds 100+s", P5="NF-UNSW-NB15-v3 January -> February, caps as P2, seed 0 or family index + 1"),
    simulator={k: (v if not isinstance(v, tuple) else list(v)) for k, v in DEFAULT.items()},
    severity=dict(UNSW=UNSW_SEVERITY, KDD=KDD_SEVERITY),
    catalogue_scenarios=SCEN,
    e9=dict(groups=E.GROUPS, rho_tune=E.RHO_TUNE, rho_eval=E.RHO_EVAL, c=E.C0, q=E.Q0, n_tune_streams=E.N_TUNE, eval_reps=E.REPS,
            seeds=dict(tune=E.SEED_TUNE, eval=E.SEED_EVAL, nsga=E.SEED_NSGA, partitions="numpy default_rng(100 + K)"),
            nsga3=dict(ref_dirs="das-dennis M=3 p=12 (91)", pop=E.POP, n_gen=E.N_GEN, sbx=dict(eta=30, prob=1.0, prob_var=0.5),
                       pm=dict(eta=20, prob=0.9, prob_var="1/n_var"), objectives=["1 - ZCV", "1 - KCV", "critical SLA misses"]),
            bounds={k: dict(xl=v["xl"], xu=v["xu"]) for k, v in E.KINDS.items()},
            policy_gradient=dict(shifts=E.PG_SHIFTS, lr=0.05, reward="ZCV + KCV - 10 max(0, critical SLA - 0.02)", checkpoints=10),
            selection_rule="max ZCV s.t. critical SLA <= 0.02 and KCV >= KCV(untuned Risk) - 0.02 on tuning streams; else min violation"),
)
json.dump(cfg, open("repro/config.json", "w"), indent=1, default=str)
print(sorted(os.listdir("repro")), {f: os.path.getsize("repro/" + f) // 1024 for f in os.listdir("repro")})
