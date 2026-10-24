# %% [markdown]
# # Chapter 7 companion notebook: tabular GANs on NSL-KDD
#
# Every number in Chapter 7 is produced by this file.
#
# Data (download yourself; not redistributed here): from the NSL-KDD
# distribution, `KDDTrain+_20Percent.txt` (training, 25,192 records) and
# `KDDTest+.txt` (testing). Set DATA below.
#
# External code: the `ctgan` package (version 0.12.1) for CTGAN and TVAE, and
# the official CTAB-GAN+ repository (github.com/Team-TUD/CTAB-GAN-Plus, commit
# 6a6f90188cca3dac2c533fd5e8e7f20de074365b). Set CTABGAN_PLUS below.
#
#   python ch07_tabular.py worker 0 2 ; python ch07_tabular.py worker 1 2
#   python ch07_tabular.py report

# %%
from pathlib import Path
import json, sys, time, warnings

import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score, recall_score

warnings.filterwarnings("ignore")
DATA = Path("/home/claude/data/nslkdd")
CTABGAN_PLUS = Path("/home/claude/team-tud/ctab-gan-plus")
OUT = Path("out"); OUT.mkdir(exist_ok=True)
RUNS = OUT / "runs"; RUNS.mkdir(exist_ok=True)
SEEDS = [0, 1, 2, 3, 4]
EPOCHS = 150          # the CTAB-GAN+ default; applied to CTGAN and TVAE as well
MODE = sys.argv[1] if len(sys.argv) > 1 else "all"
torch.set_num_threads(1)
RESULTS = {}

# %% [markdown]
# ## Data
#
# 41 features, the attack name and a difficulty score. Attack names are grouped
# into the four standard categories (the mapping of Chapter 1, Table 1.1). The
# difficulty score is dropped, and so are the two columns that are constant in
# the training file (num_outbound_cmds and is_host_login). The test set is
# KDDTest+ restricted to normal records and the 22 attack types that occur in
# training, so that utility is measured on classes the generators could learn.

# %%
COLS = ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land", "wrong_fragment",
        "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised", "root_shell", "su_attempted",
        "num_root", "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
        "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
        "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
        "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
        "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
        "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "attack", "difficulty"]
CATEGORY = {"normal": "normal",
    "back": "DoS", "land": "DoS", "neptune": "DoS", "pod": "DoS", "smurf": "DoS", "teardrop": "DoS",
    "ipsweep": "Probe", "nmap": "Probe", "portsweep": "Probe", "satan": "Probe",
    "ftp_write": "R2L", "guess_passwd": "R2L", "imap": "R2L", "multihop": "R2L", "phf": "R2L", "spy": "R2L",
    "warezclient": "R2L", "warezmaster": "R2L",
    "buffer_overflow": "U2R", "loadmodule": "U2R", "perl": "U2R", "rootkit": "U2R"}
CLASSES = ["normal", "DoS", "Probe", "R2L", "U2R"]

def load(name):
    d = pd.read_csv(DATA / name, header=None, names=COLS)
    return d

train_raw = load("KDDTrain+_20Percent.txt")
test_raw = load("KDDTest+.txt")
test_raw = test_raw[test_raw["attack"].isin(CATEGORY)].copy()
for d in (train_raw, test_raw):
    d["category"] = d["attack"].map(CATEGORY)
DROP = ["attack", "difficulty", "num_outbound_cmds", "is_host_login"]
REAL = train_raw.drop(columns=DROP).reset_index(drop=True)
TEST = test_raw.drop(columns=DROP).reset_index(drop=True)
CAT = ["protocol_type", "service", "flag", "category"]
SMALL_INT = [c for c in REAL.columns if c not in CAT and REAL[c].nunique() <= 10]     # flags and small counts
DISCRETE = CAT + SMALL_INT
CONTINUOUS = [c for c in REAL.columns if c not in DISCRETE]
INTEGER = [c for c in REAL.columns if c not in CAT and np.all(np.mod(REAL[c], 1) == 0)]
for c in SMALL_INT: REAL[c] = REAL[c].astype(int)

# %% [markdown]
# ## Generators
#
# * marginals : each column sampled independently from its empirical distribution (a floor)
# * copula    : Gaussian copula; every column mapped to normal scores through its
#               empirical CDF (categories ordered by frequency, ties broken at random),
#               correlation estimated, samples mapped back through empirical quantiles
# * ctgan     : CTGAN, ctgan 0.12.1 defaults, 150 epochs
# * tvae      : TVAE, ctgan 0.12.1 defaults, 150 epochs
# * ctabgan+  : official CTAB-GAN+ code, defaults (150 epochs), auxiliary classifier on
#               the category column; columns with more than half zeros are declared
#               "mixed" with a point mass at 0.

# %%
def to_codes(df):
    codes, maps = {}, {}
    for c in df.columns:
        if c in CAT:
            order = df[c].value_counts().index.tolist(); maps[c] = order
            codes[c] = df[c].map({v: i for i, v in enumerate(order)}).values.astype(float)
        else:
            codes[c] = df[c].values.astype(float)
    return pd.DataFrame(codes), maps

def fit_sample_marginals(df, n, seed):
    rng = np.random.RandomState(seed)
    return pd.DataFrame({c: df[c].values[rng.randint(0, len(df), n)] for c in df.columns})

def fit_sample_copula(df, n, seed):
    rng = np.random.RandomState(seed)
    X, maps = to_codes(df)
    U = np.column_stack([(stats.rankdata(X[c].values + 1e-9 * rng.rand(len(X)), method="ordinal") - 0.5) / len(X) for c in X.columns])
    Z = stats.norm.ppf(U)
    C = np.corrcoef(Z, rowvar=False)
    Zs = rng.multivariate_normal(np.zeros(len(C)), C, size=n)
    Us = stats.norm.cdf(Zs)
    out = {}
    for j, c in enumerate(X.columns):
        q = np.sort(X[c].values)
        v = q[np.minimum((Us[:, j] * len(q)).astype(int), len(q) - 1)]
        out[c] = [maps[c][int(i)] for i in v] if c in CAT else v
    return pd.DataFrame(out)[df.columns]

def fit_sample_ctgan(df, n, seed, kind):
    from ctgan import CTGAN, TVAE
    torch.manual_seed(seed); np.random.seed(seed)
    m = (CTGAN if kind == "ctgan" else TVAE)(epochs=EPOCHS, enable_gpu=False)
    m.set_random_state(seed)
    m.fit(df, discrete_columns=DISCRETE)
    return m.sample(n)

def fit_sample_ctabgan(df, n, seed):
    sys.path.insert(0, str(CTABGAN_PLUS))
    from model.pipeline.data_preparation import DataPrep
    from model.synthesizer.ctabgan_synthesizer import CTABGANSynthesizer
    torch.manual_seed(seed); np.random.seed(seed)
    mixed = {c: [0.0] for c in CONTINUOUS if (df[c] == 0).mean() > 0.5}
    general = []
    prep = DataPrep(df.copy(), DISCRETE, [], mixed, general, [], INTEGER, {None: None}, 0.0)
    syn = CTABGANSynthesizer(epochs=EPOCHS)
    syn.fit(train_data=prep.df, categorical=prep.column_types["categorical"], mixed=prep.column_types["mixed"],
            general=prep.column_types["general"], non_categorical=prep.column_types["non_categorical"],
            type={"Classification": "category"})
    out = prep.inverse_prep(syn.sample(n))
    return out[df.columns]

def generate(kind, seed, n):
    if kind == "marginals": return fit_sample_marginals(REAL, n, seed)
    if kind == "copula": return fit_sample_copula(REAL, n, seed)
    if kind in ("ctgan", "tvae"): return fit_sample_ctgan(REAL, n, seed, kind)
    if kind == "ctabgan+": return fit_sample_ctabgan(REAL, n, seed)

# %% [markdown]
# ## Measurements
#
# * fidelity: mean KS statistic over continuous columns; mean total variation
#   distance over discrete columns
# * association: mean absolute difference between the association matrices of
#   real and synthetic data (Spearman for two numeric columns, Cramér's V for two
#   discrete columns, correlation ratio for a mixed pair)
# * class shares in the synthetic data
# * validity: share of records whose (protocol_type, service, flag) combination
#   never occurs in the real training data; share of records with a rate column
#   outside [0, 1]
# * utility (TSTR): a histogram gradient boosting classifier trained on 25,192
#   synthetic records and tested on real KDDTest+ (restricted as above): macro F1
#   and recall per class. The same classifier trained on the real training data
#   gives the TRTR reference.

# %%
def cramers_v(a, b):
    t = pd.crosstab(a, b).values
    if t.shape[0] < 2 or t.shape[1] < 2: return 0.0
    chi2 = stats.chi2_contingency(t, correction=False)[0]
    return float(np.sqrt(chi2 / (t.sum() * (min(t.shape) - 1))))

def corr_ratio(cat, num):
    num = np.asarray(num, float); g = pd.Series(num).groupby(np.asarray(cat))
    ss_b = sum(len(v) * (v.mean() - num.mean()) ** 2 for _, v in g); ss_t = ((num - num.mean()) ** 2).sum()
    return float(np.sqrt(ss_b / ss_t)) if ss_t > 0 else 0.0

def assoc(df):
    cols = list(df.columns); k = len(cols); A = np.zeros((k, k))
    for i in range(k):
        for j in range(i + 1, k):
            a, b = cols[i], cols[j]
            if a in DISCRETE and b in DISCRETE: v = cramers_v(df[a], df[b])
            elif a in DISCRETE: v = corr_ratio(df[a], df[b])
            elif b in DISCRETE: v = corr_ratio(df[b], df[a])
            else:
                v = stats.spearmanr(df[a], df[b]).correlation
                v = 0.0 if np.isnan(v) else abs(v)
            A[i, j] = v
    return A[np.triu_indices(k, 1)]

A_REAL = None
RATE = [c for c in REAL.columns if c.endswith("_rate")]
COMBOS = set(map(tuple, REAL[["protocol_type", "service", "flag"]].values))

def encode_X(df, ref):
    X = df.drop(columns=["category"]).copy()
    for c in ["protocol_type", "service", "flag"]:
        cats = {v: i for i, v in enumerate(sorted(ref[c].unique()))}
        X[c] = X[c].map(cats).fillna(-1).astype(int)
    return X.astype(float)

def tstr(df, seed):
    clf = HistGradientBoostingClassifier(random_state=seed, categorical_features=[1, 2, 3])
    y = df["category"].astype(str)
    if y.nunique() < 2: return {"macro_f1": float("nan"), "recall": {c: float("nan") for c in CLASSES}}
    clf.fit(encode_X(df, REAL), y)
    p = clf.predict(encode_X(TEST, REAL))
    rec = recall_score(TEST["category"], p, labels=CLASSES, average=None, zero_division=0)
    return {"macro_f1": float(f1_score(TEST["category"], p, labels=CLASSES, average="macro", zero_division=0)),
            "recall": {c: float(r) for c, r in zip(CLASSES, rec)}}

def postprocess(syn):
    """Standard decoding step: clip continuous columns to the training range, round
    integer columns, and round rates to two decimals as they are recorded in NSL-KDD."""
    out = syn.copy()
    for c in CONTINUOUS:
        out[c] = np.clip(pd.to_numeric(out[c]), REAL[c].min(), REAL[c].max())
        if c in INTEGER: out[c] = np.round(out[c])
    for c in RATE: out[c] = np.round(out[c], 2)
    return out

def measure(syn, seed):
    global A_REAL
    if A_REAL is None: A_REAL = assoc(REAL)
    syn = syn[REAL.columns].copy()
    for c in CAT: syn[c] = syn[c].astype(str)
    for c in SMALL_INT: syn[c] = np.round(pd.to_numeric(syn[c], errors="coerce").fillna(0)).astype(int)
    m = {"ks": float(np.mean([stats.ks_2samp(syn[c].astype(float), REAL[c]).statistic for c in CONTINUOUS])),
         "tvd": float(np.mean([0.5 * (syn[c].astype(str).value_counts(normalize=True)
                                       .subtract(REAL[c].astype(str).value_counts(normalize=True), fill_value=0).abs().sum())
                               for c in DISCRETE])),
         "assoc_err": float(np.mean(np.abs(assoc(syn) - A_REAL))),
         "shares": {c: float((syn["category"] == c).mean()) for c in CLASSES},
         "invalid_combo": float(np.mean([tuple(r) not in COMBOS for r in syn[["protocol_type", "service", "flag"]].values])),
         "rate_out": float(np.mean(((syn[RATE] < 0) | (syn[RATE] > 1)).any(axis=1))),
         "zero_share": float(np.mean([(syn[c].astype(float) == 0).mean() for c in CONTINUOUS]))}
    m.update(tstr(syn, seed))
    return m

# %% [markdown]
# ## Jobs

# %%
KINDS = ["marginals", "copula", "ctgan", "tvae"]
JOBS = [(k, s) for k in ["ctgan", "tvae"] for s in SEEDS] + [(k, s) for k in ["marginals", "copula"] for s in SEEDS]
def job_path(j): return RUNS / f"{j[0]}_{j[1]}.json"

def do_job(j):
    p = job_path(j)
    if p.exists(): return
    t0 = time.time()
    syn = generate(j[0], j[1], len(REAL))
    fit_time = time.time() - t0
    syn.to_csv(RUNS / f"{j[0]}_{j[1]}.csv.gz", index=False)
    m = measure(syn, j[1]); m["fit_seconds"] = fit_time
    p.write_text(json.dumps(m))
    print(f"job {j[0]} seed {j[1]} done [fit {fit_time:.0f}s, total {time.time()-t0:.0f}s]", flush=True)

if MODE in ("all", "worker"):
    k, n = (int(sys.argv[2]), int(sys.argv[3])) if MODE == "worker" else (0, 1)
    for i, j in enumerate(JOBS):
        if i % n == k: do_job(j)
    if MODE == "worker": sys.exit(0)

# %% [markdown]
# ## Report

# %%
missing = [j for j in JOBS if not job_path(j).exists()]
if missing: raise SystemExit(f"{len(missing)} jobs not finished")
ms = lambda a, d=3: f"{np.nanmean(a):.{d}f} ± {np.nanstd(a, ddof=1):.{d}f}"
print(f"Training records {len(REAL)}, test records {len(TEST)}; columns {REAL.shape[1]} "
      f"({len(CONTINUOUS)} continuous, {len(DISCRETE)} discrete)")
print("Class shares, training (%):", {c: round(100 * (REAL['category'] == c).mean(), 3) for c in CLASSES},
      " counts:", REAL["category"].value_counts().to_dict())
print("Class counts, test:", TEST["category"].value_counts().to_dict())
ref = tstr(REAL, 0); RESULTS["trtr"] = ref
print(f"TRTR: macro F1 {ref['macro_f1']:.3f}  recall " + "  ".join(f"{c} {ref['recall'][c]:.3f}" for c in CLASSES))
ref2 = measure(REAL.sample(frac=1.0, replace=True, random_state=1).reset_index(drop=True), 0)
RESULTS["bootstrap_real"] = ref2
print(f"Bootstrap resample of the real data: KS {ref2['ks']:.3f}  TVD {ref2['tvd']:.3f}  assoc err {ref2['assoc_err']:.3f}")
print(f"Real data: mean share of exact zeros over continuous columns {np.mean([(REAL[c] == 0).mean() for c in CONTINUOUS]):.3f}")
for kind, version in [(k, v) for k in KINDS for v in ("raw", "post")]:
    rows = []
    for s in SEEDS:
        syn = pd.read_csv(RUNS / f"{kind}_{s}.csv.gz")
        m = measure(postprocess(syn) if version == "post" else syn, s)
        m["fit_seconds"] = json.loads(job_path((kind, s)).read_text())["fit_seconds"]
        rows.append(m)
    RESULTS[f"{kind}/{version}"] = rows
    print(f"{kind+'/'+version:15s} zeros {ms([r['zero_share'] for r in rows])}  KS {ms([r['ks'] for r in rows])}  TVD {ms([r['tvd'] for r in rows])}  assoc {ms([r['assoc_err'] for r in rows])}"
          f"  invalid combo {ms([100*r['invalid_combo'] for r in rows],2)}%  rate out {ms([100*r['rate_out'] for r in rows],2)}%"
          f"  | macro F1 {ms([r['macro_f1'] for r in rows])}  " + "  ".join(f"rec {c} {ms([r['recall'][c] for r in rows])}" for c in CLASSES)
          + "  | shares " + "  ".join(f"{c} {ms([100*r['shares'][c] for r in rows],3)}" for c in CLASSES)
          + f"  | fit {ms([r['fit_seconds'] for r in rows],0)} s", flush=True)
Path(OUT / "ch07_results.json").write_text(json.dumps(RESULTS, indent=1, default=float))
print("done")
