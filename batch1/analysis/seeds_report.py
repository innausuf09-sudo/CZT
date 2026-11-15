"""Tables for the wider CLAD grid (E19) and the training-seed variability of the leading detectors (E21).

Run from code/:  python -m analysis.seeds_report
Writes tables/s_table_e19_clad_wide.csv, tables/s_table_e21_detector_seeds.csv, results/e21_summary.json
"""
import json

import numpy as np
import pandas as pd

w = json.load(open("results/e19_clad_wide.json"))
rows = [dict(Family=f["family"], **{"Learning rate": f["hp"]["lr"], "Width": f["hp"]["d_model"], "Depth": f["hp"]["depth"],
             "Dropout": f["hp"]["dropout"]}, AUROC=round(f["det"]["CLAD-wide"]["auroc_zd"], 3), ZDR=round(f["det"]["CLAD-wide"]["zdr"], 3),
             **{"TPR at 1% FPR": round(f["det"]["CLAD-wide"]["tpr_fpr1_zd"], 3), "Test FPR": round(f["det"]["CLAD-wide"]["test_fpr"], 3)})
        for f in sorted(w["folds"], key=lambda r: r["family"])]
W = pd.DataFrame(rows)
W.loc[len(W)] = dict(Family="Mean", **{"Learning rate": np.nan, "Width": np.nan, "Depth": np.nan, "Dropout": np.nan},
                     AUROC=round(W.AUROC.mean(), 3), ZDR=round(W.ZDR.mean(), 3),
                     **{"TPR at 1% FPR": round(W["TPR at 1% FPR"].mean(), 3), "Test FPR": round(W["Test FPR"].mean(), 3)})
W.to_csv("tables/s_table_e19_clad_wide.csv", index=False)

e10 = json.load(open("results/e10_refit.json"))
e21 = json.load(open("results/e21_detector_seeds.json"))
MODELS = ["XGBoost", "LightGBM", "DNN", "ResSE-CNN1D", "B-ProtoMeta", "CZT-Hybrid"]
recs = []
for f in e10["folds"]:
    if f["variant"] != "full":
        continue
    for m in MODELS:
        d = f["detectors"][m]
        recs.append(dict(model=m, seed="fold index", family=f["family"], auroc=d["auroc_zd"], zdr=d["zdr"], tpr=d["tpr_fpr1_zd"]))
for f in e21["folds"]:
    for m in MODELS:
        d = f["det"][m]
        recs.append(dict(model=m, seed=f"fold index + {f['offset']}", family=f["family"], auroc=d["auroc_zd"], zdr=d["zdr"], tpr=d["tpr_fpr1_zd"]))
R = pd.DataFrame(recs)
M = R.groupby(["model", "seed"])[["auroc", "zdr", "tpr"]].mean().reset_index()
out = []
for m in MODELS:
    g = M[M.model == m]
    r = {"Detector": m}
    for _, x in g.iterrows():
        r[f"AUROC, {x.seed}"] = round(x.auroc, 3); r[f"ZDR, {x.seed}"] = round(x.zdr, 3)
    r["AUROC mean"] = round(g.auroc.mean(), 3); r["AUROC range"] = round(g.auroc.max() - g.auroc.min(), 3)
    r["ZDR mean"] = round(g.zdr.mean(), 3); r["ZDR range"] = round(g.zdr.max() - g.zdr.min(), 3)
    out.append(r)
O = pd.DataFrame(out)
O.to_csv("tables/s_table_e21_detector_seeds.csv", index=False)
pd.set_option("display.width", 250)
print(O.to_string(index=False))
# per-seed rank of the hybrid and the DNN among the six (by mean AUROC)
rk = M.assign(rank=M.groupby("seed").auroc.rank(ascending=False))
print(rk.pivot(index="model", columns="seed", values="rank"))
json.dump(dict(table=O.to_dict("records"), ranks=rk.pivot(index="model", columns="seed", values="rank").to_dict()),
          open("results/e21_summary.json", "w"), indent=1, default=float)
