# %% [markdown]
# # Chapter 12 companion notebook: which measure detects which failure?
#
# Every number in Chapter 12 is produced by this file. It reuses the data, encoding
# and utility measurement of the Chapter 7 notebook (set CH07) and applies controlled
# failures to real records, so that the right answer for every measure is known.
#   python ch12_metrics.py
# Requirements: numpy, pandas, scipy, scikit-learn.

# %%
from pathlib import Path
import json, sys, time, warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
CH07 = Path("/home/claude/ch07/ch07_tabular.py")
sys.argv = ["ch07_tabular.py", "worker", "99", "100"]
_src = CH07.read_text()
exec(_src[:_src.rindex('if MODE in ("all", "worker")')])     # REAL, TEST, DISCRETE, CONTINUOUS, tstr, assoc, ...
OUT = Path("out"); OUT.mkdir(exist_ok=True)
SEEDS = [0, 1, 2, 3, 4]
RESULTS = {}

# %% [markdown]
# ## Set-up
#
# The 25,192 records of the NSL-KDD 20% training file are split 50/50 (stratified by
# category) into a TRAIN half, which plays the part of the generator's training data,
# and a HOLDOUT half, which plays the part of an honest generator: new records from the
# same distribution. Each "synthetic" table below is the HOLDOUT half with one
# controlled failure applied, and is compared with the TRAIN half.
#
# Failures:
#   honest     : HOLDOUT unchanged
#   copy       : the TRAIN half itself (a generator that memorised its data)
#   near_copy  : TRAIN records with continuous values multiplied by 1 + N(0, 0.01^2)
#   drop_rare  : HOLDOUT without its R2L and U2R records (rare classes lost), resampled to size
#   collapse   : HOLDOUT records of the 20 most frequent rows only, resampled to size (mode collapse)
#   shuffle    : every column of HOLDOUT permuted independently (marginals kept, structure lost)
#   noise      : HOLDOUT with N(0, (0.5 sd)^2) added to every continuous column
#   label_swap : HOLDOUT with the category column permuted (features right, labels uninformative)

# %%
def split(seed):
    a, b = train_test_split(REAL, test_size=0.5, stratify=REAL["category"], random_state=seed)
    return a.reset_index(drop=True), b.reset_index(drop=True)

def corrupt(kind, train, hold, seed):
    rng = np.random.RandomState(seed); n = len(hold)
    if kind == "honest": return hold.copy()
    if kind == "copy": return train.copy()
    if kind == "near_copy":
        x = train.copy()
        for c in CONTINUOUS: x[c] = x[c] * (1 + 0.01 * rng.randn(len(x)))
        return x
    if kind == "drop_rare":
        x = hold[~hold["category"].isin(["R2L", "U2R"])]; return x.sample(n, replace=True, random_state=seed).reset_index(drop=True)
    if kind == "collapse":
        key = hold.astype(str).agg("|".join, axis=1); top = key.value_counts().index[:20]
        x = hold[key.isin(top)]; return x.sample(n, replace=True, random_state=seed).reset_index(drop=True)
    if kind == "shuffle":
        return pd.DataFrame({c: hold[c].sample(frac=1.0, random_state=seed + i).values for i, c in enumerate(hold.columns)})
    if kind == "noise":
        x = hold.copy()
        for c in CONTINUOUS: x[c] = x[c] + 0.5 * hold[c].std() * rng.randn(n)
        return x
    if kind == "label_swap":
        x = hold.copy(); x["category"] = rng.permutation(x["category"].values); return x

FAILURES = ["honest", "copy", "near_copy", "drop_rare", "collapse", "shuffle", "noise", "label_swap"]

# %% [markdown]
# ## Measures
#
# Encoding for distance-based measures: continuous columns through a Gaussian quantile
# transform fitted on TRAIN, discrete columns one-hot (scaled by 1/sqrt 2 so that two
# different categories are at distance 1). Distance-based measures use 3,000 records of
# each table.
#
#   ks, tvd, assoc        : as in Chapter 7
#   mmd_median, mmd_fixed : MMD^2 with a Gaussian kernel; bandwidth by the median heuristic on
#                           the encoded data, or fixed at 1.0 on the raw, unscaled continuous columns
#   precision, recall     : Kynkäänniemi et al., k = 5
#   density, coverage     : Naeem et al., k = 5
#   dcr_ratio             : median DCR of synthetic records to TRAIN / median DCR of HOLDOUT to TRAIN
#   exact_copies          : share of synthetic records identical to a TRAIN record
#   tstr                  : macro F1 on KDDTest+ (restricted) of a classifier trained on the table
#   rare_share            : share of R2L + U2R records

# %%
class Enc:
    def __init__(self, ref):
        self.q = QuantileTransformer(n_quantiles=1000, output_distribution="normal", random_state=0).fit(ref[CONTINUOUS].astype(float))
        self.cats = {c: sorted(ref[c].astype(str).unique()) for c in DISCRETE}
    def __call__(self, df):
        parts = [self.q.transform(df[CONTINUOUS].astype(float))]
        for c, v in self.cats.items():
            parts.append(np.eye(len(v))[pd.Categorical(df[c].astype(str), categories=v).codes] / np.sqrt(2))
        return np.hstack(parts)

def mmd2(X, Y, bw):
    def k(A, B):
        d = (A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None] - 2 * A @ B.T
        return np.exp(-d / (2 * bw ** 2))
    m, n = len(X), len(Y)
    kxx, kyy, kxy = k(X, X), k(Y, Y), k(X, Y)
    return float((kxx.sum() - np.trace(kxx)) / (m * (m - 1)) + (kyy.sum() - np.trace(kyy)) / (n * (n - 1)) - 2 * kxy.mean())

def knn_metrics(R, F, k=5):
    rad = NearestNeighbors(n_neighbors=k + 1).fit(R).kneighbors(R)[0][:, k]
    radf = NearestNeighbors(n_neighbors=k + 1).fit(F).kneighbors(F)[0][:, k]
    dRF = np.sqrt(np.maximum(((F ** 2).sum(1)[:, None] + (R ** 2).sum(1)[None] - 2 * F @ R.T), 0))   # (M, N)
    inside = dRF <= rad[None, :]
    precision = float(inside.any(1).mean()); density = float(inside.sum(1).mean() / k); coverage = float(inside.any(0).mean())
    recall = float((dRF <= radf[:, None]).any(0).mean())
    return precision, recall, density, coverage

def evaluate(syn, train, hold, seed):
    rng = np.random.RandomState(seed); enc = Enc(train)
    s = syn.sample(3000, random_state=seed); t = train.sample(3000, random_state=seed + 1); h = hold.sample(3000, random_state=seed + 2)
    S, Tt, Hh = enc(s), enc(t), enc(h)
    med = float(np.median(np.sqrt(((Tt[:500, None] - Tt[None, :500]) ** 2).sum(-1))))
    raw = lambda d: d[CONTINUOUS].astype(float).values
    p, r, d, c = knn_metrics(Tt, S)
    nn = NearestNeighbors(n_neighbors=1).fit(enc(train))
    dcr_s = nn.kneighbors(S)[0].ravel(); dcr_h = nn.kneighbors(Hh)[0].ravel()
    trainkeys = set(train.astype(str).agg("|".join, axis=1))
    out = {"ks": float(np.mean([stats.ks_2samp(syn[c].astype(float), train[c]).statistic for c in CONTINUOUS])),
           "tvd": float(np.mean([0.5 * syn[c].astype(str).value_counts(normalize=True).subtract(train[c].astype(str).value_counts(normalize=True), fill_value=0).abs().sum() for c in DISCRETE])),
           "mmd_median": mmd2(Tt, S, med), "mmd_fixed": mmd2(raw(t), raw(s), 1.0),
           "precision": p, "recall": r, "density": d, "coverage": c,
           "dcr_ratio": float(np.median(dcr_s) / np.median(dcr_h)),
           "exact_copies": float(np.mean([k in trainkeys for k in s.astype(str).agg("|".join, axis=1)])),
           "rare_share": float(syn["category"].isin(["R2L", "U2R"]).mean())}
    out["tstr"] = tstr(syn.reset_index(drop=True), seed)["macro_f1"]
    return out

# %%
A_cache = {}
for seed in SEEDS:
    train, hold = split(seed)
    A_t = assoc(train)
    for f in FAILURES:
        t0 = time.time()
        syn = corrupt(f, train, hold, seed)
        m = evaluate(syn, train, hold, seed)
        m["assoc"] = float(np.mean(np.abs(assoc(syn) - A_t)))
        RESULTS.setdefault(f, []).append(m)
        print(f"seed {seed} {f:10s} done [{time.time()-t0:.0f}s]", flush=True)
ref_rare = float(REAL["category"].isin(["R2L", "U2R"]).mean())
print(f"\nreal share of R2L + U2R: {ref_rare:.4f}")
KEYS = ["ks", "tvd", "assoc", "mmd_median", "mmd_fixed", "precision", "recall", "density", "coverage", "dcr_ratio", "exact_copies", "rare_share", "tstr"]
for f in FAILURES:
    rows = RESULTS[f]
    print(f"{f:10s} " + "  ".join(f"{k} {np.mean([r[k] for r in rows]):.4f}±{np.std([r[k] for r in rows], ddof=1):.4f}" for k in KEYS), flush=True)
Path(OUT / "ch12_results.json").write_text(json.dumps(RESULTS, indent=1))
print("done")
