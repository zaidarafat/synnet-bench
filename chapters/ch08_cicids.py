# %% [markdown]
# # Chapter 8 application: CSE-CIC-IDS-2018 preprocessing and leakage
#
# A real-data counterpart to Experiment 8.1, on CSE-CIC-IDS-2018 flow records. The sample
# combines the two "web attack" days (02-22 and 02-23, 2018), which carry the rare
# Brute Force-Web, Brute Force-XSS and SQL Injection classes against a Benign majority — the
# CIC-IDS-2018 analogue of NSL-KDD's rare R2L/U2R classes. We document the dataset's
# data-quality defects, apply a leakage-safe cleaning and augmentation pipeline, and measure
# the inflation produced by two common leaks.
#   python ch08_cicids.py      # expects the sampled CSV at DATA
# Requirements: numpy, pandas, scikit-learn, imbalanced-learn.

# %%
from pathlib import Path
import json, warnings
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score, recall_score
from imblearn.over_sampling import SMOTENC

warnings.filterwarnings("ignore")
DATA = Path("/home/claude/ch08/cicids/sample.csv")   # stratified sample of days 02-22 and 02-23
OUT = Path("out"); OUT.mkdir(exist_ok=True)
SEEDS = [0, 1, 2, 3, 4]
TARGET = 2000           # records per minority class after augmentation
CATEGORICAL = ["Protocol"]   # Dst Port kept numeric (4,979 distinct values, too high-cardinality for a categorical split)
LEAK_NEG = ["Init Fwd Win Byts", "Init Bwd Win Byts"]   # -1 is a "no value" sentinel

# %% [markdown]
# ## Load and inventory the defects (on the whole sample, for reporting only)

# %%
raw = pd.read_csv(DATA, low_memory=False)
label = raw.columns[-1]
feat_cols = [c for c in raw.columns if c not in (label, "Timestamp")]   # drop temporal identifier
num = raw[feat_cols].apply(pd.to_numeric, errors="coerce")
DEFECTS = {
    "rows": int(len(raw)),
    "classes": raw[label].value_counts().to_dict(),
    "infinite_values": int(np.isinf(num.values).sum()),
    "infinite_cols": [c for c in num.columns if np.isinf(num[c].values).any()],
    "negative_sentinels": int((num[LEAK_NEG] < 0).values.sum()),
    "timestamp_dropped": "Timestamp" in raw.columns,
}
print("CSE-CIC-IDS-2018 sample defects:")
print(f"  rows {DEFECTS['rows']}, classes {DEFECTS['classes']}")
print(f"  infinite values {DEFECTS['infinite_values']} in {DEFECTS['infinite_cols']}")
print(f"  negative sentinels (-1) {DEFECTS['negative_sentinels']} in {LEAK_NEG}")
print(f"  Timestamp column dropped as a temporal identifier: {DEFECTS['timestamp_dropped']}", flush=True)

# %% [markdown]
# ## Cleaning (fit on a given partition only) and the three protocols
#
# clean(): coerce to numeric, map infinities and the -1 window sentinels to NaN; the
# gradient-boosting classifier consumes NaN natively, so no imputation step is fitted and the
# only fitted artefact is the classifier. The leaks are therefore in the DATA ORDER, not in a
# scaler: a correct run splits first and oversamples the training part only; the leaky runs
# oversample before the split (synthetic records enter the test set) or draw the test set from
# the same oversampled pool.

# %%
def clean(df):
    X = df[feat_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    for c in LEAK_NEG:
        X.loc[X[c] < 0, c] = np.nan
    return X

cat_idx = [feat_cols.index(c) for c in CATEGORICAL]

def smote_aug(X, y, seed):
    counts = pd.Series(y).value_counts()
    strat = {c: TARGET for c in counts.index if c != "Benign" and counts[c] < TARGET}
    k = max(1, min(5, min(counts[c] for c in strat) - 1)) if strat else 5
    Xf = X.fillna(X.median(numeric_only=True))      # SMOTE needs finite inputs
    Xo, yo = SMOTENC(categorical_features=cat_idx, sampling_strategy=strat,
                     k_neighbors=k, random_state=seed).fit_resample(Xf, y)
    return Xo, yo

def clf(seed):
    return HistGradientBoostingClassifier(random_state=seed, categorical_features=cat_idx)

def metrics(model, Xte, yte):
    p = model.predict(Xte)
    return {"macro_f1": float(f1_score(yte, p, average="macro", zero_division=0)),
            "sqli_recall": float(recall_score(yte == "SQL Injection", p == "SQL Injection")),
            "web_recall": float(recall_score(yte == "Brute Force -Web", p == "Brute Force -Web"))}

def run(protocol, seed):
    X = clean(raw); y = raw[label].values
    if protocol == "correct":
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed)
        Xo, yo = smote_aug(Xtr.reset_index(drop=True), ytr, seed)
        return metrics(clf(seed).fit(Xo, yo), Xte, yte)
    if protocol == "leak_oversample_before_split":
        Xo, yo = smote_aug(X.reset_index(drop=True), y, seed)           # oversample WHOLE set
        Xtr, Xte, ytr, yte = train_test_split(Xo, yo, test_size=0.3, stratify=yo, random_state=seed)
        return metrics(clf(seed).fit(Xtr, ytr), Xte, yte)              # synthetic rows leak into test
    if protocol == "correct_no_aug":
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed)
        return metrics(clf(seed).fit(Xtr, ytr), Xte, yte)

# %%
RUNS = OUT / "cic_runs"; RUNS.mkdir(exist_ok=True)
def cached(proto, s):
    p = RUNS / f"{proto}_{s}.json"
    if p.exists(): return json.loads(p.read_text())
    r = run(proto, s); p.write_text(json.dumps(r)); print(f"  {proto} seed {s} done", flush=True); return r

RESULTS = {"defects": DEFECTS, "runs": {}}
for proto in ["correct_no_aug", "correct", "leak_oversample_before_split"]:
    rows = [cached(proto, s) for s in SEEDS]
    RESULTS["runs"][proto] = rows
    f = lambda k: f"{np.mean([r[k] for r in rows]):.3f} ± {np.std([r[k] for r in rows], ddof=1):.3f}"
    print(f"{proto:32s} macro-F1 {f('macro_f1')}  SQLi recall {f('sqli_recall')}  Web recall {f('web_recall')}", flush=True)
Path(OUT / "ch08_cicids_results.json").write_text(json.dumps(RESULTS, indent=1))
print("done")
