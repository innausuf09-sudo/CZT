"""Protocol P6 (E23) summary: means over three seeds. Writes tables/s_table_e23_site.csv, s_table_e23_perclass.csv."""
import json
import numpy as np
import pandas as pd
R = json.load(open("results/e23_site_disjoint.json"))["results"]
rows, pc = [], []
for r in R:
    for m, d in r["det"].items():
        rows.append(dict(direction=f"{r['source']} to {r['target']}", kind=r["kind"], variant=r["variant"], seed=r["seed"], model=m,
                         auroc=d["auroc"], tpr_fpr1=d["tpr_fpr1"], detection=d["detection"], test_fpr=d["test_fpr"], removed=len(r["removed"])))
        if r["kind"] == "cross" and r["variant"] == "full":
            for c, v in d["per_class"].items():
                pc.append(dict(direction=f"{r['source']} to {r['target']}", model=m, cls=c, det=v))
D = pd.DataFrame(rows)
S = D.groupby(["kind", "direction", "variant", "model"])[["auroc", "tpr_fpr1", "detection", "test_fpr", "removed"]].mean().round(3).reset_index()
S.columns = ["Protocol", "Direction", "Features", "Detector", "AUROC", "TPR at 1% FPR", "Detection at source threshold", "Target FPR", "Removed features"]
S.to_csv("tables/s_table_e23_site.csv", index=False)
P = pd.DataFrame(pc).groupby(["direction", "cls", "model"]).det.mean().unstack().round(3).reset_index()
P.to_csv("tables/s_table_e23_perclass.csv", index=False)
pd.set_option("display.width", 220); pd.set_option("display.max_rows", 200)
print(S.to_string(index=False))
print({(r["source"], r["variant"]): r["removed"] for r in R if r["variant"] == "filtered" and r["seed"] == 0})
