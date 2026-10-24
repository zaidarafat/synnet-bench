# %% [markdown]
# # Chapter 19 companion notebook: statistical design of experiments
#
# Part A: a real paired comparison of augmentation methods on NSL-KDD over many seeds
#         (effect size, confidence interval, paired tests, single-seed instability).
# Part B: a Monte-Carlo demonstration of the multiple-comparisons problem and the
#         Holm-Bonferroni correction (pure statistics, no classifier needed).
# Data: `KDDTrain+.txt` (set DATA).  Run: python ch19_stats.py
# Requirements: numpy, pandas, scipy, scikit-learn, imbalanced-learn.

# %%
from pathlib import Path
import json, warnings
import numpy as np, pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from imblearn.over_sampling import SMOTENC

warnings.filterwarnings("ignore")
DATA = Path("/home/claude/data/nslkdd/KDDTrain+.txt")
OUT = Path("out"); OUT.mkdir(exist_ok=True)
NSEED = 20
CLASSES = ["normal", "DoS", "Probe", "R2L", "U2R"]
CATEGORY = {"normal": "normal",
    "back": "DoS", "land": "DoS", "neptune": "DoS", "pod": "DoS", "smurf": "DoS", "teardrop": "DoS",
    "ipsweep": "Probe", "nmap": "Probe", "portsweep": "Probe", "satan": "Probe",
    "ftp_write": "R2L", "guess_passwd": "R2L", "imap": "R2L", "multihop": "R2L", "phf": "R2L", "spy": "R2L",
    "warezclient": "R2L", "warezmaster": "R2L",
    "buffer_overflow": "U2R", "loadmodule": "U2R", "perl": "U2R", "rootkit": "U2R"}
d = pd.read_csv(DATA, header=None); y_all = d[41].map(CATEGORY).values
X_all = d.iloc[:, :41].copy()
for c in (1, 2, 3): X_all[c] = X_all[c].astype("category").cat.codes
X_all = X_all.values.astype(float); CAT_IDX = [1, 2, 3]

def r2l_recall(method, seed, n_r2l=100):
    Xtr, Xte, ytr, yte = train_test_split(X_all, y_all, test_size=0.3, stratify=y_all, random_state=seed)
    rng = np.random.RandomState(1000 + seed)
    r = np.where(ytr == "R2L")[0]; keep = rng.choice(r, n_r2l, replace=False)
    m = (ytr != "R2L"); m[keep] = True; Xtr, ytr = Xtr[m], ytr[m]
    if method in ("smote", "ros"):
        cnt = pd.Series(ytr).value_counts()
        if method == "smote":
            Xo, yo = SMOTENC(categorical_features=CAT_IDX, sampling_strategy={"R2L": 5000}, k_neighbors=5, random_state=seed).fit_resample(Xtr, ytr)
        else:
            extra = Xtr[np.random.RandomState(seed).choice(np.where(ytr == "R2L")[0], 5000 - cnt["R2L"])]
            Xo = np.vstack([Xtr, extra]); yo = np.r_[ytr, ["R2L"] * len(extra)]
        clf = HistGradientBoostingClassifier(random_state=seed, categorical_features=CAT_IDX).fit(Xo, yo)
    else:
        w = None
        if method == "weights":
            w = pd.Series(ytr).map(len(ytr) / (len(CLASSES) * pd.Series(ytr).value_counts())).values
        clf = HistGradientBoostingClassifier(random_state=seed, categorical_features=CAT_IDX).fit(Xtr, ytr, sample_weight=w)
    p = clf.predict(Xte)
    return float(np.mean((p == "R2L")[yte == "R2L"]))

# %% [markdown]
# ## Part A: a real paired comparison (SMOTE vs none, weights vs none) over 20 seeds

# %%
def cohen_d_paired(diff):
    return float(np.mean(diff) / (np.std(diff, ddof=1) + 1e-12))

RESULTS = {"partA": {}, "partB": {}}
base = np.array([r2l_recall("none", s) for s in range(NSEED)])
for method in ["smote", "ros", "weights"]:
    vals = np.array([r2l_recall(method, s) for s in range(NSEED)])
    diff = vals - base
    t_p = stats.ttest_rel(vals, base).pvalue
    w_p = stats.wilcoxon(vals, base).pvalue if np.any(diff != 0) else 1.0
    ci = stats.t.interval(0.95, len(diff) - 1, loc=diff.mean(), scale=stats.sem(diff))
    RESULTS["partA"][method] = {"mean_diff": float(diff.mean()), "ci": [float(ci[0]), float(ci[1])],
                                "cohen_d": cohen_d_paired(diff), "t_p": float(t_p), "wilcoxon_p": float(w_p),
                                "n_seeds_negative": int(np.sum(diff < 0)), "best_seed_diff": float(diff.max()),
                                "worst_seed_diff": float(diff.min()), "method_mean": float(vals.mean())}
    print(f"{method:8s} mean diff {diff.mean():+.3f}  95% CI [{ci[0]:+.3f}, {ci[1]:+.3f}]  Cohen d {cohen_d_paired(diff):+.2f}"
          f"  paired-t p {t_p:.2e}  Wilcoxon p {w_p:.2e}  seeds worse than baseline {np.sum(diff<0)}/{NSEED}"
          f"  (best seed {diff.max():+.3f}, worst {diff.min():+.3f})", flush=True)
print(f"baseline R2L recall: {base.mean():.3f} ± {base.std(ddof=1):.3f}  (range {base.min():.3f}-{base.max():.3f})")

# %% [markdown]
# ## Part B: multiple comparisons and Holm-Bonferroni (Monte Carlo, null is true)
#
# We simulate m=40 comparisons in which the two methods are statistically identical
# (both samples drawn from the same normal), each with 20 paired observations, and run a
# paired t-test on each. Under the null, p-values are uniform, so about 5% are "significant"
# at 0.05 by chance. Reporting the single best (smallest p) is the selective-reporting
# mistake; Holm-Bonferroni controls the family-wise error. Repeated over 2000 trials.

# %%
def holm(pvals, alpha=0.05):
    m = len(pvals); order = np.argsort(pvals); reject = np.zeros(m, bool)
    for rank, idx in enumerate(order):
        if pvals[idx] <= alpha / (m - rank): reject[idx] = True
        else: break
    return reject

m, n, trials = 40, 20, 2000
rng = np.random.RandomState(0)
any_raw = 0; any_holm = 0; best_p = []
for _ in range(trials):
    ps = []
    for _ in range(m):
        a = rng.randn(n); b = rng.randn(n)                 # identical null distributions
        ps.append(stats.ttest_rel(a, b).pvalue)
    ps = np.array(ps)
    any_raw += int(np.any(ps < 0.05))                      # at least one "significant" uncorrected
    any_holm += int(np.any(holm(ps)))                      # at least one after Holm
    best_p.append(ps.min())
RESULTS["partB"] = {"m": m, "n": n, "trials": trials,
                    "family_false_positive_raw": any_raw / trials,
                    "family_false_positive_holm": any_holm / trials,
                    "median_best_p": float(np.median(best_p))}
print(f"\nMultiple comparisons (m={m} null tests, {trials} trials):")
print(f"  P(at least one p<0.05 uncorrected) = {any_raw/trials:.3f}   (per-test rate is 0.05)")
print(f"  P(at least one rejected after Holm) = {any_holm/trials:.3f}  (family-wise error, target <=0.05)")
print(f"  median smallest-of-{m} p (what selective reporting would quote) = {np.median(best_p):.4f}")
Path(OUT / "ch19_results.json").write_text(json.dumps(RESULTS, indent=1))
print("done")
