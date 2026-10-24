"""Leakage-safe data handling for NSL-KDD and the SynNet-Bench protocol.

The single rule this module enforces is Chapter 8's: the train/test split happens
before anything is fitted. `load_nslkdd` returns the raw frame; `leakage_safe_split`
splits first and returns partitions that every downstream fitted step (scaler,
generator, classifier) must respect.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split

# 41 canonical NSL-KDD feature names, in column order
FEATURES = ["duration","protocol_type","service","flag","src_bytes","dst_bytes","land","wrong_fragment",
 "urgent","hot","num_failed_logins","logged_in","num_compromised","root_shell","su_attempted","num_root",
 "num_file_creations","num_shells","num_access_files","num_outbound_cmds","is_host_login","is_guest_login",
 "count","srv_count","serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
 "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count","dst_host_same_srv_rate",
 "dst_host_diff_srv_rate","dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_serror_rate",
 "dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate"]
DISCRETE = ["protocol_type", "service", "flag"]
CLASSES = ["normal", "DoS", "Probe", "R2L", "U2R"]

# full attack -> category map, including the 17 test-only attack types
CATEGORY = {"normal": "normal",
 "back": "DoS", "land": "DoS", "neptune": "DoS", "pod": "DoS", "smurf": "DoS", "teardrop": "DoS",
 "apache2": "DoS", "udpstorm": "DoS", "processtable": "DoS", "mailbomb": "DoS", "worm": "DoS",
 "satan": "Probe", "ipsweep": "Probe", "nmap": "Probe", "portsweep": "Probe", "mscan": "Probe", "saint": "Probe",
 "guess_passwd": "R2L", "ftp_write": "R2L", "imap": "R2L", "phf": "R2L", "multihop": "R2L",
 "warezmaster": "R2L", "warezclient": "R2L", "spy": "R2L", "xlock": "R2L", "xsnoop": "R2L",
 "snmpguess": "R2L", "snmpgetattack": "R2L", "httptunnel": "R2L", "sendmail": "R2L", "named": "R2L",
 "buffer_overflow": "U2R", "loadmodule": "U2R", "rootkit": "U2R", "perl": "U2R",
 "sqlattack": "U2R", "xterm": "U2R", "ps": "U2R"}


def load_nslkdd(path, drop_duplicates=True):
    """Load an NSL-KDD file (KDDTrain+.txt or KDDTest+.txt) into (X, y).

    X keeps raw feature values with named columns; categorical columns are integer-coded.
    y is the four-category + normal label. Exact duplicates are removed by default
    (Chapter 8), which must happen before any split.
    """
    d = pd.read_csv(Path(path), header=None)
    if drop_duplicates:
        d = d.drop_duplicates(subset=list(range(42))).reset_index(drop=True)
    y = d[41].map(CATEGORY).values
    if pd.isna(y).any():
        raise ValueError("unmapped attack label(s): " + ", ".join(sorted(set(d[41][pd.isna(y)]))))
    X = d.iloc[:, :41].copy(); X.columns = FEATURES
    for c in DISCRETE:
        X[c] = X[c].astype("category").cat.codes
    return X.astype(float), y


def leakage_safe_split(X, y, test_size=0.3, seed=0):
    """Split BEFORE anything is fitted (Chapter 8). Returns Xtr, Xte, ytr, yte with
    reset indices. Every fitted step downstream must use the training part only."""
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)
    return Xtr.reset_index(drop=True), Xte.reset_index(drop=True), ytr, yte


def reduce_class(Xtr, ytr, cls, n, seed=0):
    """Reduce a class to n records in the TRAINING part only (for scarcity studies)."""
    rng = np.random.RandomState(1000 + seed)
    idx = np.where(ytr == cls)[0]
    if n >= len(idx):
        return Xtr, ytr
    keep = rng.choice(idx, n, replace=False)
    mask = (ytr != cls); mask[keep] = True
    return Xtr[mask].reset_index(drop=True), ytr[mask]
