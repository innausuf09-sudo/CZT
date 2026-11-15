"""Data details for the supplement (revision R1.5, R1.9).

Writes
- tables/s_table_features_unsw60.csv, s_table_features_kdd68.csv, s_table_features_harm18.csv
  (feature name, source column, transformation, categorical vocabulary)
- tables/s_table_cleaning_per_family.csv (raw, after exact de-duplication, after conflict removal)
- tables/s_table_missing_invalid.csv (missing values and negative inputs to log(1+x) per column)
- results/e15_masks.npz (boolean masks over the merged UNSW-NB15 release: kept after step 1, kept after step 2)
Run from code/:  python -m analysis.data_details
"""
import os

import numpy as np
import pandas as pd

from zdt.data import (DATA_DIR, HARM_FEATURES, KDD_COLS, KDD_TOPK, UNSW_CAT, UNSW_TOPK, kdd_matrix, load_kdd,
                      unsw_matrix)


def raw_unsw():
    a = pd.read_csv(os.path.join(DATA_DIR, "UNSW_NB15_training-set.csv"))
    b = pd.read_csv(os.path.join(DATA_DIR, "UNSW_NB15_testing-set.csv"))
    for d in (a, b):
        d.columns = [c.strip().lstrip("﻿") for c in d.columns]
    df = pd.concat([a, b], ignore_index=True).drop(columns=["id"])
    df["attack_cat"] = df["attack_cat"].str.strip()
    return df


df = raw_unsw()
feats = [c for c in df.columns if c not in ("attack_cat", "label")]
keep1 = ~df.duplicated(subset=feats + ["attack_cat"])
d1 = df[keep1]
g = d1.groupby(feats, dropna=False)["attack_cat"].transform("nunique")
keep2 = pd.Series(False, index=df.index)
keep2[d1.index[g.values == 1]] = True
np.savez_compressed("results/e15_masks.npz", kept_after_exact_dedup=keep1.values, kept_final=keep2.values,
                    note=np.array("row order = training-set file followed by testing-set file, id column dropped"))
rows = []
for fam in ["Normal", "Generic", "Exploits", "Fuzzers", "DoS", "Reconnaissance", "Analysis", "Backdoor", "Shellcode", "Worms"]:
    m = df["attack_cat"] == fam
    rows.append(dict(family=fam, raw=int(m.sum()), after_exact_dedup=int((m & keep1).sum()),
                     final=int((m & keep2).sum()),
                     removed_exact=int((m & ~keep1).sum()), removed_conflict=int((m & keep1 & ~keep2).sum())))
C = pd.DataFrame(rows)
C.loc[len(C)] = dict(family="Total", **{k: int(C[k].sum()) for k in C.columns if k != "family"})
C["removed_share"] = 1 - C["final"] / C["raw"]
C.to_csv("tables/s_table_cleaning_per_family.csv", index=False)
print(C.to_string(index=False))

# invalid inputs: missing values and negative values that are clipped before log(1+x)
num = [c for c in feats if c not in UNSW_CAT]
mi = pd.DataFrame(dict(column=num, missing=[int(df[c].isna().sum()) for c in num],
                       negative=[int((pd.to_numeric(df[c], errors="coerce") < 0).sum()) for c in num]))
tr, te = load_kdd()
knum = [c for c in KDD_COLS if c not in ("protocol_type", "service", "flag", "attack", "difficulty", "num_outbound_cmds")]
kall = pd.concat([tr, te])
mi = pd.concat([mi.assign(dataset="UNSW-NB15"),
                pd.DataFrame(dict(column=knum, missing=[int(kall[c].isna().sum()) for c in knum],
                                  negative=[int((kall[c] < 0).sum()) for c in knum])).assign(dataset="NSL-KDD")])
mi.to_csv("tables/s_table_missing_invalid.csv", index=False)
print("missing total", int(mi.missing.sum()), "negative total", int(mi.negative.sum()))

# feature lists
_, names60 = unsw_matrix(df.head(100))
F = []
for n in names60:
    if "=" in n:
        col, v = n.split("=", 1)
        F.append(dict(feature=n, source=col, transform="one-hot",
                      vocabulary=", ".join(UNSW_TOPK[col]) + ", other (any value outside the list)"))
    else:
        F.append(dict(feature=n, source=n, transform="max(x, 0) then log(1 + x)", vocabulary=""))
pd.DataFrame(F).to_csv("tables/s_table_features_unsw60.csv", index=False)
_, names68 = kdd_matrix(tr.head(100))
F = []
for n in names68:
    if "=" in n:
        col, v = n.split("=", 1)
        F.append(dict(feature=n, source=col, transform="one-hot",
                      vocabulary=", ".join(KDD_TOPK[col]) + ", other"))
    else:
        F.append(dict(feature=n, source=n, transform="max(x, 0) then log(1 + x)", vocabulary=""))
pd.DataFrame(F).to_csv("tables/s_table_features_kdd68.csv", index=False)
H = {"log_duration": ("dur", "duration"), "log_src_bytes": ("sbytes", "src_bytes"), "log_dst_bytes": ("dbytes", "dst_bytes"),
     "log_conn_count": ("ct_dst_ltm", "count"), "log_srv_count": ("ct_srv_src", "srv_count")}
F = []
for n in HARM_FEATURES:
    if n in H:
        F.append(dict(feature=n, unsw_source=H[n][0], kdd_source=H[n][1], transform="max(x, 0) then log(1 + x)"))
    elif n.startswith("proto"):
        F.append(dict(feature=n, unsw_source="proto", kdd_source="protocol_type", transform="one-hot (tcp, udp, icmp, other)"))
    elif n.startswith("svc"):
        F.append(dict(feature=n, unsw_source="service ('-' -> other, ftp-data -> ftp)",
                      kdd_source="service (ftp_data -> ftp, domain_u/domain -> dns)", transform="one-hot"))
    else:
        F.append(dict(feature=n, unsw_source="state (FIN, CON -> established; RST, CLO, ACC -> reset; else incomplete)",
                      kdd_source="flag (SF -> established; REJ, RSTO, RSTR, RSTOS0 -> reset; else incomplete)",
                      transform="one-hot"))
pd.DataFrame(F).to_csv("tables/s_table_features_harm18.csv", index=False)
print(len(names60), len(names68), len(HARM_FEATURES))
