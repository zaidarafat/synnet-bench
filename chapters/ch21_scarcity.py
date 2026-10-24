# %% [markdown]
# # Chapter 21 companion notebook: GANs under sample scarcity
#
# Two experiments answer "when does augmentation help?".
#   Experiment 21.1 (real data, NSL-KDD R2L): the augmentation gain as a function of the
#     real minority sample size, measured alongside the downstream model's variance and the
#     generator's copying, to expose the variance-reduction mechanism.
#   Experiment 21.2 (controlled, known ground truth): a synthetic Gaussian-mixture problem
#     in which the true minority distribution is known, so augmentation from the TRUE
#     distribution (an unreachable oracle) can be compared with a correctly-specified fitted
#     generator (a Gaussian mixture) and with SMOTE interpolation. This isolates the
#     information a generator cannot invent: the data-processing bound.
#
#   python ch21_scarcity.py            # runs both, caches per job, writes out/ch21_results.json
# Requirements: numpy, pandas, scipy, scikit-learn.  NSL-KDD at DATA for 21.1.

# %%
from pathlib import Path
import json, warnings
import numpy as np, pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import recall_score, balanced_accuracy_score, roc_auc_score

warnings.filterwarnings("ignore")
DATA = Path("/home/claude/data/nslkdd/KDDTrain+.txt")
OUT = Path("out"); OUT.mkdir(exist_ok=True)
RESULTS = {}
CATEGORY = {"normal": "normal",
    "back": "DoS", "land": "DoS", "neptune": "DoS", "pod": "DoS", "smurf": "DoS", "teardrop": "DoS",
    "ipsweep": "Probe", "nmap": "Probe", "portsweep": "Probe", "satan": "Probe",
    "ftp_write": "R2L", "guess_passwd": "R2L", "imap": "R2L", "multihop": "R2L", "phf": "R2L", "spy": "R2L",
    "warezclient": "R2L", "warezmaster": "R2L",
    "buffer_overflow": "U2R", "loadmodule": "U2R", "perl": "U2R", "rootkit": "U2R"}

# %% [markdown]
# ## Experiment 21.1: the gain curve, the variance and the copying on NSL-KDD
#
# For each real R2L training size n, over 5 seeds: fit a classifier with no augmentation
# (baseline) and with SMOTE-NC augmentation to 5,000 R2L records, both on a leakage-free
# split; record the R2L recall of each. We also measure, per seed, the fraction of the
# synthetic R2L records that exactly copy a real training R2L record (memorisation). We then
# report, against n: the baseline recall and its across-seed standard deviation (the
# downstream variance), the SMOTE recall gain, and the SMOTE copy rate.

# %%
d = pd.read_csv(DATA, header=None).drop_duplicates(subset=list(range(42))).reset_index(drop=True)
y_all = d[41].map(CATEGORY).values
X_all = d.iloc[:, :41].copy(); X_all.columns = [f"f{i}" for i in range(41)]
CAT = ["f1", "f2", "f3"]
for c in CAT: X_all[c] = X_all[c].astype("category").cat.codes
X_all = X_all.astype(float)
NS = [25, 50, 100, 200, 400, "all"]; SEEDS = [0, 1, 2, 3, 4]; TARGET = 5000

def split(seed, n):
    Xtr, Xte, ytr, yte = train_test_split(X_all, y_all, test_size=0.3, stratify=y_all, random_state=seed)
    Xtr, Xte = Xtr.reset_index(drop=True), Xte.reset_index(drop=True)
    if n != "all":
        rng = np.random.RandomState(1000 + seed); r = np.where(ytr == "R2L")[0]
        keep = rng.choice(r, n, replace=False); m = (ytr != "R2L"); m[keep] = True
        Xtr, ytr = Xtr[m].reset_index(drop=True), ytr[m]
    return Xtr, ytr, Xte, yte

def smote_r2l(Xtr, ytr, m, seed):
    rng = np.random.RandomState(seed); X = Xtr[ytr == "R2L"].values; n = len(X); k = min(5, n - 1)
    cat = [list(Xtr.columns).index(c) for c in CAT]; num = [i for i in range(X.shape[1]) if i not in cat]
    mu, sd = X[:, num].mean(0), X[:, num].std(0) + 1e-9
    Z = np.hstack([(X[:, num] - mu) / sd] + [np.eye(int(X_all[c].max()) + 1)[X[:, i].astype(int)] / np.sqrt(2) for i, c in zip(cat, CAT)])
    nb = NearestNeighbors(n_neighbors=k + 1).fit(Z).kneighbors(Z, return_distance=False)[:, 1:]
    a = rng.randint(0, n, m); b = nb[a, rng.randint(0, k, m)]; lam = rng.rand(m, 1)
    S = X[a] + lam * (X[b] - X[a])
    for i in cat: S[:, i] = [np.bincount(v.astype(int)).argmax() for v in X[nb[a], i]]
    return pd.DataFrame(S, columns=Xtr.columns)

def clf(seed): return HistGradientBoostingClassifier(random_state=seed, categorical_features=[1, 2, 3])

rows21 = {}
for n in NS:
    base, smote, copy = [], [], []
    for s in SEEDS:
        Xtr, ytr, Xte, yte = split(s, n); n_real = int((ytr == "R2L").sum()); m = TARGET - n_real
        base.append(recall_score(yte == "R2L", clf(s).fit(Xtr, ytr).predict(Xte) == "R2L"))
        S = smote_r2l(Xtr, ytr, m, s)
        Xa = pd.concat([Xtr, S], ignore_index=True); ya = np.r_[ytr, ["R2L"] * len(S)]
        smote.append(recall_score(yte == "R2L", clf(s).fit(Xa, ya).predict(Xte) == "R2L"))
        keys = set(Xtr[ytr == "R2L"].astype(str).agg("|".join, axis=1))
        copy.append(float(np.mean([k_ in keys for k_ in S.astype(str).agg("|".join, axis=1)])))
    base, smote, copy = map(np.array, (base, smote, copy))
    rows21[str(n)] = {"n_real": n_real, "base_recall": float(base.mean()), "base_sd": float(base.std(ddof=1)),
                      "smote_recall": float(smote.mean()), "gain": float((smote - base).mean()),
                      "gain_sd": float((smote - base).std(ddof=1)), "copy_rate": float(copy.mean())}
    r = rows21[str(n)]
    print(f"n={str(n):4s} (real {r['n_real']:3d})  baseline {r['base_recall']:.3f}±{r['base_sd']:.3f}  "
          f"SMOTE gain {r['gain']:+.3f}±{r['gain_sd']:.3f}  copy {r['copy_rate']:.3f}", flush=True)
RESULTS["exp21_1"] = rows21

# %% [markdown]
# ## Experiment 21.2: the information bound, with ground truth known
#
# A controlled two-class problem in d=8 dimensions. The majority class is a single Gaussian;
# the minority class is a two-component Gaussian mixture that overlaps it, so the Bayes error
# is non-zero and recall is not trivially perfect. Because we define the distributions, we can
# draw from the TRUE minority distribution as an (unreachable) oracle augmenter.
#
# For each real minority training size n (majority fixed at 2000), augment the minority to 1000
# records by four methods and train a logistic classifier, evaluating minority recall and
# balanced accuracy on a large test set drawn from the true distributions:
#   none  : real data only (baseline)
#   true  : draw the extra minority records from the TRUE minority distribution (oracle upper bound)
#   gmm   : fit a 2-component Gaussian mixture on the n real minority points and sample it
#           (a correctly-specified generator: it cannot know more than the n points reveal)
#   smote : interpolate between nearest neighbours of the n real minority points
# 30 seeds. The gap between 'true' and 'gmm' is the estimation error no generator can escape;
# the gap between 'gmm' and 'smote' is the cost of an interpolation bias.

# %%
D = 8; N_MAJ = 2000; N_AUG = 1000; N_TEST = 20000; NS2 = [10, 20, 40, 80, 160, 320]; SEEDS2 = list(range(30))
rng0 = np.random.RandomState(12345)
mu_maj = np.zeros(D)
mu_min = np.stack([rng0.randn(D) * 1.1 + 1.3, rng0.randn(D) * 1.1 - 1.3])   # two minority components
w_min = np.array([0.6, 0.4])

def draw_min(n, rng):
    comp = rng.choice(2, n, p=w_min); return mu_min[comp] + rng.randn(n, D) * 0.9
def draw_maj(n, rng): return mu_maj + rng.randn(n, D) * 1.0

def smote_pts(X, m, rng):
    n = len(X); k = min(5, n - 1)
    nb = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X, return_distance=False)[:, 1:]
    a = rng.randint(0, n, m); b = nb[a, rng.randint(0, k, m)]; lam = rng.rand(m, 1)
    return X[a] + lam * (X[b] - X[a])

rows22 = {}
for n in NS2:
    agg = {k: {"recall": [], "bacc": [], "auc": []} for k in ["none", "true", "gmm", "smote"]}
    for s in SEEDS2:
        rng = np.random.RandomState(s)
        Xmin = draw_min(n, rng); Xmaj = draw_maj(N_MAJ, rng)
        Xte = np.vstack([draw_min(N_TEST // 2, rng), draw_maj(N_TEST // 2, rng)])
        yte = np.r_[np.ones(N_TEST // 2), np.zeros(N_TEST // 2)]
        m = N_AUG - n
        aug = {"none": None,
               "true": draw_min(m, rng),
               "gmm": GaussianMixture(2, random_state=s).fit(Xmin).sample(m)[0] if n >= 2 else draw_min(m, rng),
               "smote": smote_pts(Xmin, m, rng)}
        for method, extra in aug.items():
            Xm = Xmin if extra is None else np.vstack([Xmin, extra])
            X = np.vstack([Xm, Xmaj]); y = np.r_[np.ones(len(Xm)), np.zeros(len(Xmaj))]
            mdl = LogisticRegression(max_iter=1000, class_weight="balanced").fit(X, y)
            p = mdl.predict(Xte); pr = mdl.predict_proba(Xte)[:, 1]
            agg[method]["recall"].append(recall_score(yte, p))
            agg[method]["bacc"].append(balanced_accuracy_score(yte, p))
            agg[method]["auc"].append(roc_auc_score(yte, pr))
    rows22[str(n)] = {k: {mm: [float(np.mean(v)), float(np.std(v, ddof=1))] for mm, v in d_.items()} for k, d_ in agg.items()}
    f = lambda k: f"{np.mean(agg[k]['bacc']):.3f}"
    print(f"n={n:4d}  balanced-acc  none {f('none')}  smote {f('smote')}  gmm {f('gmm')}  true {f('true')}", flush=True)
RESULTS["exp21_2"] = rows22
RESULTS["exp21_2_meta"] = {"D": D, "N_MAJ": N_MAJ, "N_AUG": N_AUG, "N_TEST": N_TEST, "seeds": len(SEEDS2)}

Path(OUT / "ch21_results.json").write_text(json.dumps(RESULTS, indent=1))
print("done")
