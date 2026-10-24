# %% [markdown]
# # Chapter 20 companion notebook: membership inference against synthetic-data generators
#
# A membership-inference attack (LOGAN-style, distance-based) tries to decide whether a
# real record was in a generator's training set, from the synthetic data alone. We measure
# the attack's AUC for several generators and relate privacy leakage to the copying measured
# in Chapter 11. Defender's experiment: it quantifies a privacy risk, releases nothing.
# Data: NSL-KDD via the Chapter 7 notebook (set CH07).
#   python ch20_privacy.py   (CPU; the deep models reuse ch11 budgets)
# Requirements: torch, numpy, pandas, scipy, scikit-learn, ctgan.

# %%
from pathlib import Path
import json, sys, time, warnings
import numpy as np, pandas as pd
from scipy import stats
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
CH07 = Path("/home/claude/ch07/ch07_tabular.py")
sys.argv = ["ch07_tabular.py", "worker", "99", "100"]
_src = CH07.read_text()
exec(_src[:_src.rindex('if MODE in ("all", "worker")')])   # REAL, TEST, DISCRETE, CONTINUOUS, SMALL_INT, fit_sample_copula...
OUT = Path("out"); OUT.mkdir(exist_ok=True)
SEEDS = [0, 1, 2]
RESULTS = {}

# %% [markdown]
# ## Set-up
#
# Split REAL in half: MEMBERS (the generator's training set) and NONMEMBERS (held out,
# same distribution). A generator is fitted on MEMBERS and produces synthetic records.
# Distance encoder: quantile-normal continuous columns + scaled one-hot discrete columns.
#
# Attack (LOGAN / distance-based): for each real record, score = -distance to the nearest
# synthetic record. If the generator memorised its training data, members score higher than
# non-members. Attack AUC = AUROC of separating members from non-members by this score;
# 0.5 means no leakage, 1.0 means perfect membership inference.

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

def smote_table(df, n, seed):
    from imblearn.over_sampling import SMOTENC
    rng = np.random.RandomState(seed); cat = [list(df.columns).index(c) for c in DISCRETE]
    y = df["category"].astype(str)
    cnt = y.value_counts(); strat = {c: cnt[c] * 2 for c in cnt.index if cnt[c] > 6}
    Xo, yo = SMOTENC(categorical_features=[list(df.drop(columns=['category']).columns).index(c) for c in DISCRETE if c != 'category'],
                     sampling_strategy=strat, k_neighbors=5, random_state=seed).fit_resample(df.drop(columns=['category']), y)
    out = Xo.copy(); out["category"] = yo
    return out.sample(min(n, len(out)), random_state=seed).reset_index(drop=True)[df.columns]

def tvae_table(df, n, seed):
    from ctgan import TVAE
    import torch; torch.manual_seed(seed); np.random.seed(seed)
    m = TVAE(epochs=120, enable_gpu=False); m.set_random_state(seed); m.fit(df, discrete_columns=DISCRETE)
    return m.sample(n)

def generate(kind, members, n, seed):
    if kind == "smote": return smote_table(members, n, seed)
    if kind == "copula":
        global REAL
        _save = REAL; REAL = members
        try: out = fit_sample_copula(members, n, seed)
        finally: REAL = _save
        return out
    if kind == "tvae": return tvae_table(members, n, seed)

def attack_auc(syn, members, nonmembers, enc):
    nn = NearestNeighbors(n_neighbors=1).fit(enc(syn))
    dm = nn.kneighbors(enc(members))[0].ravel(); dn = nn.kneighbors(enc(nonmembers))[0].ravel()
    y = np.r_[np.ones(len(dm)), np.zeros(len(dn))]; score = -np.r_[dm, dn]   # closer => more likely member
    return float(roc_auc_score(y, score))

def exact_copy_rate(syn, members):
    keys = set(members.astype(str).agg("|".join, axis=1))
    return float(np.mean([k in keys for k in syn.astype(str).agg("|".join, axis=1)]))

# %%
for kind in ["smote", "copula", "tvae"]:
    rows = []
    for seed in SEEDS:
        members, nonmembers = train_test_split(REAL, test_size=0.5, stratify=REAL["category"], random_state=seed)
        members = members.reset_index(drop=True); nonmembers = nonmembers.reset_index(drop=True)
        enc = Enc(members)
        t0 = time.time(); syn = generate(kind, members, len(members), seed)
        mem = members.sample(min(3000, len(members)), random_state=seed); non = nonmembers.sample(min(3000, len(nonmembers)), random_state=seed)
        auc = attack_auc(syn, mem, non, enc); cp = exact_copy_rate(syn, members)
        rows.append({"attack_auc": auc, "exact_copies": cp, "seconds": time.time() - t0})
    RESULTS[kind] = rows
    f = lambda k, d=3: f"{np.mean([r[k] for r in rows]):.{d}f} ± {np.std([r[k] for r in rows], ddof=1):.{d}f}"
    print(f"{kind:7s} membership-inference AUC {f('attack_auc')}  exact copies {f('exact_copies',3)}  time {np.mean([r['seconds'] for r in rows]):.0f}s", flush=True)
print("(attack AUC 0.5 = no leakage; 1.0 = perfect membership inference)")
Path(OUT / "ch20_results.json").write_text(json.dumps(RESULTS, indent=1))
print("done")
