# %% [markdown]
# # Chapter 8 companion notebook, part 1: leakage in augmentation pipelines
#
# Reproduces Experiment 8.1. Data: `KDDTrain+.txt` from the NSL-KDD distribution
# (download yourself; set DATA). Requirements: numpy, pandas, scikit-learn,
# imbalanced-learn.
#   python ch08_leakage.py

# %%
from pathlib import Path
import json, time, warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import f1_score, recall_score
from imblearn.over_sampling import SMOTE, SMOTENC

warnings.filterwarnings("ignore")
DATA = Path("/home/claude/data/nslkdd/KDDTrain+.txt")
OUT = Path("out"); OUT.mkdir(exist_ok=True)
SEEDS = [0, 1, 2, 3, 4]
CLASSES = ["normal", "DoS", "Probe", "R2L", "U2R"]
CATEGORY = {"normal": "normal",
    "back": "DoS", "land": "DoS", "neptune": "DoS", "pod": "DoS", "smurf": "DoS", "teardrop": "DoS",
    "ipsweep": "Probe", "nmap": "Probe", "portsweep": "Probe", "satan": "Probe",
    "ftp_write": "R2L", "guess_passwd": "R2L", "imap": "R2L", "multihop": "R2L", "phf": "R2L", "spy": "R2L",
    "warezclient": "R2L", "warezmaster": "R2L",
    "buffer_overflow": "U2R", "loadmodule": "U2R", "perl": "U2R", "rootkit": "U2R"}

# %% [markdown]
# ## Data
# The 41 features, with protocol, service and flag integer-coded; the label is
# the attack category. The full KDDTrain+ (125,973 records) is split 70/30,
# stratified by category, with a different split for each seed.

# %%
d = pd.read_csv(DATA, header=None)
y_all = d[41].map(CATEGORY).values
X_all = d.iloc[:, :41].copy()
for c in (1, 2, 3):
    X_all[c] = X_all[c].astype("category").cat.codes
X_all = X_all.values.astype(float)
print("records", len(X_all), "class counts", pd.Series(y_all).value_counts().to_dict())
dup = pd.DataFrame(np.column_stack([X_all, y_all])).duplicated().sum()
print("exact duplicate records in KDDTrain+:", int(dup))

# %% [markdown]
# ## Five pipelines
#
# * correct     : split; oversample the minority classes (SMOTE-NC) in the training part only; train; test.
# * smote_first : oversample the whole dataset, then split. Synthetic records are
#                 interpolated between records that end up in the test part.
# * select_all  : choose the 15 features with the highest mutual information with
#                 the label using the whole dataset (test labels included), then as correct.
# * gen_on_all  : fit the "generator" (SMOTE) on the whole dataset and train the
#                 classifier on its synthetic minority records plus the real training
#                 records of the other classes; the test part is unchanged.
# * tune_on_test: as correct, but the number of boosting iterations is chosen from
#                 {50, 100, 200, 400} by the macro F1 on the test part itself.
#
# Oversampling target: R2L and U2R raised to 5,000 records each in the data that
# is oversampled. Classifier: histogram gradient boosting, categorical features
# declared, fixed seed.

# %%
TARGET = 5000
CAT_IDX = [1, 2, 3]

def clf(seed, iters=100, cat=CAT_IDX):
    return HistGradientBoostingClassifier(random_state=seed, max_iter=iters, categorical_features=cat)

def oversample(X, y, seed, cat=CAT_IDX):
    # SMOTENC: interpolates numeric features and takes the most frequent value among
    # the neighbours for categorical ones, so categorical codes stay valid
    counts = pd.Series(y).value_counts()
    strat = {c: max(TARGET, counts[c]) for c in ("R2L", "U2R")}
    sm = SMOTENC(categorical_features=cat, sampling_strategy=strat, k_neighbors=5, random_state=seed) if cat else \
         SMOTE(sampling_strategy=strat, k_neighbors=5, random_state=seed)
    return sm.fit_resample(X, y)

def score(y_true, p):
    rec = recall_score(y_true, p, labels=CLASSES, average=None, zero_division=0)
    return {"macro_f1": float(f1_score(y_true, p, labels=CLASSES, average="macro", zero_division=0)),
            **{f"rec_{c}": float(r) for c, r in zip(CLASSES, rec)}}

def run(pipeline, seed):
    Xtr, Xte, ytr, yte = train_test_split(X_all, y_all, test_size=0.3, stratify=y_all, random_state=seed)
    if pipeline == "correct":
        Xo, yo = oversample(Xtr, ytr, seed)
        return score(yte, clf(seed).fit(Xo, yo).predict(Xte))
    if pipeline == "smote_first":
        Xo, yo = oversample(X_all, y_all, seed)
        Xtr2, Xte2, ytr2, yte2 = train_test_split(Xo, yo, test_size=0.3, stratify=yo, random_state=seed)
        return score(yte2, clf(seed).fit(Xtr2, ytr2).predict(Xte2))
    if pipeline == "select_all":
        mi = mutual_info_classif(X_all, y_all, discrete_features=[i in CAT_IDX for i in range(41)], random_state=seed)
        keep = np.sort(np.argsort(mi)[::-1][:15]); cat = [i for i, k in enumerate(keep) if k in CAT_IDX]
        Xo, yo = oversample(Xtr[:, keep], ytr, seed, cat)
        return {**score(yte, clf(seed, cat=cat).fit(Xo, yo).predict(Xte[:, keep])), "features": keep.tolist()}
    if pipeline == "gen_on_all":
        Xs, ys = oversample(X_all, y_all, seed)
        syn = len(X_all)                                  # SMOTE appends synthetic records after the originals
        Xsyn, ysyn = Xs[syn:], ys[syn:]
        keep = ~np.isin(ytr, ["R2L", "U2R"])
        Xo = np.vstack([Xtr[keep], Xsyn]); yo = np.concatenate([ytr[keep], ysyn])
        return score(yte, clf(seed).fit(Xo, yo).predict(Xte))
    if pipeline == "gen_on_train":                      # the honest version of gen_on_all
        Xs, ys = oversample(Xtr, ytr, seed)
        Xsyn, ysyn = Xs[len(Xtr):], ys[len(Xtr):]
        keep = ~np.isin(ytr, ["R2L", "U2R"])
        Xo = np.vstack([Xtr[keep], Xsyn]); yo = np.concatenate([ytr[keep], ysyn])
        return score(yte, clf(seed).fit(Xo, yo).predict(Xte))
    if pipeline == "tune_on_test":
        Xo, yo = oversample(Xtr, ytr, seed)
        best = max((score(yte, clf(seed, iters=it).fit(Xo, yo).predict(Xte)) | {"iters": it} for it in (50, 100, 200, 400)),
                   key=lambda r: r["macro_f1"])
        return best
    if pipeline == "tune_on_val":                       # the honest version of tune_on_test
        Xa, Xv, ya, yv = train_test_split(Xtr, ytr, test_size=0.2, stratify=ytr, random_state=seed)
        Xo, yo = oversample(Xa, ya, seed)
        it = max((50, 100, 200, 400), key=lambda it: score(yv, clf(seed, iters=it).fit(Xo, yo).predict(Xv))["macro_f1"])
        Xo, yo = oversample(Xtr, ytr, seed)
        return score(yte, clf(seed, iters=it).fit(Xo, yo).predict(Xte)) | {"iters": it}

PIPES = ["correct", "smote_first", "select_all", "gen_on_train", "gen_on_all", "tune_on_val", "tune_on_test"]
RESULTS = {}
for pname in PIPES:
    t0 = time.time()
    rows = [run(pname, s) for s in SEEDS]
    RESULTS[pname] = rows
    f = lambda k, d=3: f"{np.mean([r[k] for r in rows]):.{d}f} ± {np.std([r[k] for r in rows], ddof=1):.{d}f}"
    print(f"{pname:13s} macro F1 {f('macro_f1')}  R2L {f('rec_R2L')}  U2R {f('rec_U2R')}  Probe {f('rec_Probe')}"
          + (f"  iters {[r['iters'] for r in rows]}" if "iters" in rows[0] else "") + f"  [{time.time()-t0:.0f}s]", flush=True)
# Test-part class counts for reference
_, _, _, yte0 = train_test_split(X_all, y_all, test_size=0.3, stratify=y_all, random_state=0)
print("test part class counts (seed 0):", pd.Series(yte0).value_counts().to_dict())
Path(OUT / "ch08_leakage_results.json").write_text(json.dumps(RESULTS, indent=1))
print("done")
