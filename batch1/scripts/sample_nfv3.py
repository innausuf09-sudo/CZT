"""Stratified sample of a NetFlow v3 dataset (e.g. NF-CSE-CIC-IDS2018-v3.csv) small enough to attach.

Usage (Python 3 with pandas installed):
    python sample_nfv3.py NF-CSE-CIC-IDS2018-v3.csv

Keeps every attack class (up to 60,000 rows each, all rows if fewer), 1,000,000 benign rows,
and all columns including the flow timestamps. Output: <input>_sample.csv and a zipped copy.
The sampling is seeded, so it can be reported and repeated.
"""
import sys, zipfile, hashlib, os
import numpy as np
import pandas as pd

SRC = sys.argv[1]
CAP_ATTACK, N_BENIGN, SEED, CHUNK = 60_000, 1_000_000, 20261109, 1_000_000

sha1 = hashlib.sha1()
with open(SRC, "rb") as f:
    for block in iter(lambda: f.read(1 << 24), b""):
        sha1.update(block)

# pass 1: class counts
counts = pd.Series(dtype="int64")
for ch in pd.read_csv(SRC, usecols=["Attack"], chunksize=CHUNK):
    counts = counts.add(ch["Attack"].value_counts(), fill_value=0)
counts = counts.astype(int)
prob = {a: min(1.0, (N_BENIGN if a == "Benign" else CAP_ATTACK) / n) for a, n in counts.items()}

# pass 2: Bernoulli sampling with per-class probability
rng = np.random.default_rng(SEED)
out = os.path.splitext(SRC)[0] + "_sample.csv"
first = True
for ch in pd.read_csv(SRC, chunksize=CHUNK, low_memory=False):
    keep = rng.random(len(ch)) < ch["Attack"].map(prob).to_numpy()
    ch[keep].to_csv(out, mode="w" if first else "a", header=first, index=False)
    first = False

kept = pd.read_csv(out, usecols=["Attack"])["Attack"].value_counts()
report = pd.DataFrame({"original": counts, "sampled": kept}).fillna(0).astype(int)
info = os.path.splitext(SRC)[0] + "_sample_info.txt"
with open(info, "w") as f:
    f.write(f"source file: {os.path.basename(SRC)}\nsource SHA-1: {sha1.hexdigest()}\nseed: {SEED}\n")
    f.write(f"cap per attack class: {CAP_ATTACK}, benign rows: {N_BENIGN}\n\n{report.to_string()}\n")
with zipfile.ZipFile(out + ".zip", "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
    z.write(out, os.path.basename(out)); z.write(info, os.path.basename(info))
print(report.to_string())
print("\nAttach this file:", out + ".zip", f"({os.path.getsize(out + '.zip') / 1e6:.0f} MB)")
