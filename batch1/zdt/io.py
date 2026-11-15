"""Compact storage of the per-fold detector scores of protocol P2.

The scores are kept in a compressed NumPy archive (results/e2_scores.npz)
instead of a pickle, so that the repository stays small and the file can be
loaded without executing code."""
import os
import pickle
import numpy as np

FIELDS = ("p", "pu", "pva_normal")  # calibrated, raw, benign-validation scores


def save_scores(store, path="results/e2_scores.npz"):
    arrays = {}
    for fold, st in store.items():
        arrays[f"{fold}|groups"] = np.asarray(st["groups"]).astype(str)
        arrays[f"{fold}|fam"] = np.asarray(st["fam"]).astype(str)
        arrays[f"{fold}|y"] = np.asarray(st["y"]).astype(np.int8)
        for model, sc in st["scores"].items():
            for k in FIELDS:
                v = np.asarray(sc[k])
                arrays[f"{fold}|{model}|{k}"] = v.astype(np.float32)
            arrays[f"{fold}|{model}|thr"] = np.float32(sc["thr"])
    np.savez_compressed(path, **arrays)


def load_scores(path="results/e2_scores.npz"):
    if not os.path.exists(path) and os.path.exists(path.replace(".npz", ".pkl")):
        return pickle.load(open(path.replace(".npz", ".pkl"), "rb"))
    z = np.load(path, allow_pickle=False)
    store = {}
    for key in z.files:
        parts = key.split("|")
        fold = parts[0]
        st = store.setdefault(fold, {"scores": {}})
        if len(parts) == 2:
            st[parts[1]] = z[key]
        else:
            model, field = parts[1], parts[2]
            v = z[key]
            st["scores"].setdefault(model, {})[field] = float(v) if field == "thr" else v
    order = ["Generic", "Exploits", "Fuzzers", "DoS", "Reconnaissance", "Analysis", "Backdoor", "Shellcode", "Worms"]
    return {f: store[f] for f in order if f in store}
