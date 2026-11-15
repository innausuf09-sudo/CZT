"""Data loading, leakage-averse de-duplication and preprocessing for
UNSW-NB15 and NSL-KDD, plus a harmonised cross-dataset schema."""
import os
import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

# ---------------------------------------------------------------- UNSW-NB15
UNSW_CAT = ["proto", "service", "state"]
UNSW_TOPK = {"proto": ["tcp", "udp", "unas", "arp", "ospf"],
             "service": ["-", "dns", "http", "smtp", "ftp", "ftp-data", "pop3", "ssh"],
             "state": ["FIN", "INT", "CON", "REQ", "RST"]}
UNSW_FAMILIES = ["Generic", "Exploits", "Fuzzers", "DoS", "Reconnaissance",
                 "Analysis", "Backdoor", "Shellcode", "Worms"]
# Severity proxy s_f in [0,1] (analyst-assigned, MITRE ATT&CK-inspired)
UNSW_SEVERITY = {"Normal": 0.0, "Generic": 0.55, "Exploits": 0.85, "Fuzzers": 0.45,
                 "DoS": 0.70, "Reconnaissance": 0.35, "Analysis": 0.40,
                 "Backdoor": 0.95, "Shellcode": 0.90, "Worms": 1.00}


def load_unsw(dedup=True, verbose=True):
    a = pd.read_csv(os.path.join(DATA_DIR, "UNSW_NB15_training-set.csv"))
    b = pd.read_csv(os.path.join(DATA_DIR, "UNSW_NB15_testing-set.csv"))
    a.columns = [c.strip().lstrip("﻿") for c in a.columns]
    b.columns = [c.strip().lstrip("﻿") for c in b.columns]
    df = pd.concat([a, b], ignore_index=True).drop(columns=["id"])
    df["attack_cat"] = df["attack_cat"].str.strip()
    stats = {"raw_rows": len(df)}
    feats = [c for c in df.columns if c not in ("attack_cat", "label")]
    if dedup:
        df = df.drop_duplicates(subset=feats + ["attack_cat"]).reset_index(drop=True)
        stats["after_exact_dedup"] = len(df)
        # remove feature-identical rows that carry conflicting labels
        g = df.groupby(feats, dropna=False)["attack_cat"].transform("nunique")
        stats["conflicting_rows_removed"] = int((g > 1).sum())
        df = df[g == 1].reset_index(drop=True)
        stats["final_rows"] = len(df)
    if verbose:
        print("UNSW stats", stats)
    return df, stats


def _onehot(df, col, cats, prefix):
    out = {}
    v = df[col].astype(str).str.strip()
    for c in cats:
        out[f"{prefix}={c}"] = (v == c).astype(np.float32)
    out[f"{prefix}=other"] = (~v.isin(cats)).astype(np.float32)
    return pd.DataFrame(out, index=df.index)


def unsw_matrix(df):
    num = [c for c in df.columns if c not in UNSW_CAT + ["attack_cat", "label"]]
    X = df[num].astype(np.float64).clip(lower=0)
    X = np.log1p(X)
    parts = [X] + [_onehot(df, c, UNSW_TOPK[c], c) for c in UNSW_CAT]
    M = pd.concat(parts, axis=1)
    return M.values.astype(np.float32), list(M.columns)


# ---------------------------------------------------------------- NSL-KDD
KDD_COLS = ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land",
            "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
            "num_compromised", "root_shell", "su_attempted", "num_root",
            "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
            "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
            "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
            "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
            "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
            "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
            "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
            "dst_host_srv_rerror_rate", "attack", "difficulty"]
KDD_MAP = {
    "normal": "Normal",
    # DoS
    "back": "DoS", "land": "DoS", "neptune": "DoS", "pod": "DoS", "smurf": "DoS",
    "teardrop": "DoS", "apache2": "DoS", "mailbomb": "DoS", "processtable": "DoS",
    "udpstorm": "DoS",
    # Probe
    "ipsweep": "Probe", "nmap": "Probe", "portsweep": "Probe", "satan": "Probe",
    "mscan": "Probe", "saint": "Probe",
    # R2L
    "ftp_write": "R2L", "guess_passwd": "R2L", "imap": "R2L", "multihop": "R2L",
    "phf": "R2L", "spy": "R2L", "warezclient": "R2L", "warezmaster": "R2L",
    "sendmail": "R2L", "named": "R2L", "snmpgetattack": "R2L", "snmpguess": "R2L",
    "xlock": "R2L", "xsnoop": "R2L", "worm": "R2L",
    # U2R
    "buffer_overflow": "U2R", "loadmodule": "U2R", "perl": "U2R", "rootkit": "U2R",
    "httptunnel": "U2R", "ps": "U2R", "sqlattack": "U2R", "xterm": "U2R"}
KDD_SEVERITY = {"Normal": 0.0, "DoS": 0.70, "Probe": 0.35, "R2L": 0.85, "U2R": 1.00}
KDD_TOPK = {"protocol_type": ["tcp", "udp", "icmp"],
            "service": ["http", "private", "domain_u", "smtp", "ftp_data", "ecr_i", "eco_i",
                        "other", "ftp", "telnet", "finger", "urp_i", "auth", "imap4", "pop_3"],
            "flag": ["SF", "S0", "REJ", "RSTR", "SH", "RSTO", "S1", "RSTOS0", "S3", "S2", "OTH"]}


def load_kdd():
    tr = pd.read_csv(os.path.join(DATA_DIR, "KDDTrain+.txt"), header=None, names=KDD_COLS)
    te = pd.read_csv(os.path.join(DATA_DIR, "KDDTest+.txt"), header=None, names=KDD_COLS)
    for d in (tr, te):
        d["category"] = d["attack"].map(KDD_MAP).fillna("Unknown")
    train_types = set(tr["attack"].unique())
    te["novel"] = (~te["attack"].isin(train_types)) & (te["attack"] != "normal")
    return tr, te


def kdd_matrix(df):
    cat = ["protocol_type", "service", "flag"]
    num = [c for c in KDD_COLS if c not in cat + ["attack", "difficulty", "num_outbound_cmds"]]
    X = np.log1p(df[num].astype(np.float64).clip(lower=0))
    parts = [X] + [_onehot(df, c, KDD_TOPK[c], c) for c in cat]
    M = pd.concat(parts, axis=1)
    return M.values.astype(np.float32), list(M.columns)


# ------------------------------------------------------- harmonised schema
HARM_FEATURES = ["log_duration", "log_src_bytes", "log_dst_bytes", "log_conn_count",
                 "log_srv_count", "proto=tcp", "proto=udp", "proto=icmp", "proto=other",
                 "svc=http", "svc=ftp", "svc=smtp", "svc=dns", "svc=ssh", "svc=other",
                 "state=established", "state=incomplete", "state=reset"]


def _harm(duration, sb, db, cnt, srv, proto, svc, state):
    out = pd.DataFrame({
        "log_duration": np.log1p(np.clip(duration, 0, None)),
        "log_src_bytes": np.log1p(np.clip(sb, 0, None)),
        "log_dst_bytes": np.log1p(np.clip(db, 0, None)),
        "log_conn_count": np.log1p(np.clip(cnt, 0, None)),
        "log_srv_count": np.log1p(np.clip(srv, 0, None))})
    for p in ["tcp", "udp", "icmp"]:
        out[f"proto={p}"] = (proto == p).astype(np.float32)
    out["proto=other"] = (~proto.isin(["tcp", "udp", "icmp"])).astype(np.float32)
    for s in ["http", "ftp", "smtp", "dns", "ssh"]:
        out[f"svc={s}"] = (svc == s).astype(np.float32)
    out["svc=other"] = (~svc.isin(["http", "ftp", "smtp", "dns", "ssh"])).astype(np.float32)
    for s in ["established", "incomplete", "reset"]:
        out[f"state={s}"] = (state == s).astype(np.float32)
    return out[HARM_FEATURES].values.astype(np.float32)


def unsw_harmonised(df):
    svc = df["service"].astype(str).replace({"ftp-data": "ftp", "-": "other"})
    st = df["state"].astype(str).map(lambda s: "established" if s in ("FIN", "CON") else
                                      ("reset" if s in ("RST", "CLO", "ACC") else "incomplete"))
    return _harm(df["dur"].values, df["sbytes"].values, df["dbytes"].values,
                 df["ct_dst_ltm"].values, df["ct_srv_src"].values,
                 df["proto"].astype(str), svc, st)


def kdd_harmonised(df):
    svc = df["service"].astype(str).replace({"ftp_data": "ftp", "domain_u": "dns", "domain": "dns"})
    st = df["flag"].astype(str).map(lambda s: "established" if s == "SF" else
                                    ("reset" if s in ("REJ", "RSTO", "RSTR", "RSTOS0") else "incomplete"))
    return _harm(df["duration"].values, df["src_bytes"].values, df["dst_bytes"].values,
                 df["count"].values, df["srv_count"].values,
                 df["protocol_type"].astype(str), svc, st)


def stratified_cap(labels, cap_per_class, rng):
    """Return indices with at most cap_per_class rows per label (stratified cap)."""
    labels = np.asarray(labels)
    idx = []
    for c in np.unique(labels):
        ii = np.where(labels == c)[0]
        if len(ii) > cap_per_class:
            ii = rng.choice(ii, cap_per_class, replace=False)
        idx.append(ii)
    return np.sort(np.concatenate(idx))
