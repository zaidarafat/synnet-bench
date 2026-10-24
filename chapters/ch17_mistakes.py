# %% [markdown]
# # Chapter 17 companion notebook: a ledger of methodological mistakes
#
# Each mistake is reproduced deliberately on ONE NSL-KDD augmentation task and its effect
# on the headline number is measured against a correct baseline. SMOTE-NC stands in for a
# generator (fast, deterministic); the leakage mechanisms are identical for a GAN.
# Data: `KDDTrain+.txt` (set DATA).  Run: python ch17_mistakes.py
# Requirements: numpy, pandas, scipy, scikit-learn, imbalanced-learn.

# %%
from pathlib import Path
import json, warnings
import numpy as np, pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import f1_score, recall_score, accuracy_score
from imblearn.over_sampling import SMOTENC

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

d = pd.read_csv(DATA, header=None)
y_all = d[41].map(CATEGORY).values
X_all = d.iloc[:, :41].copy()
for c in (1, 2, 3): X_all[c] = X_all[c].astype("category").cat.codes
X_all = X_all.values.astype(float)
CAT_IDX = [1, 2, 3]; TARGET = 5000

def clf(seed, iters=100):
    return HistGradientBoostingClassifier(random_state=seed, max_iter=iters, categorical_features=CAT_IDX)

def oversample(X, y, seed, cat=CAT_IDX):
    cnt = pd.Series(y).value_counts(); strat = {c: max(TARGET, cnt[c]) for c in ("R2L", "U2R")}
    return SMOTENC(categorical_features=cat, sampling_strategy=strat, k_neighbors=5, random_state=seed).fit_resample(X, y)

def macro_f1(ytrue, p): return f1_score(ytrue, p, labels=CLASSES, average="macro", zero_division=0)

# %% [markdown]
# ## The correct baseline and six mistakes
#
# Correct: split; oversample TRAIN only; fit; report macro-F1 (mean over seeds).
# M1 accuracy paradox: report overall accuracy instead of macro-F1.
# M2 selective seeds: report the BEST of five seeds' macro-F1 instead of the mean.
# M3 test-set tuning: pick the number of iterations that maximises TEST macro-F1.
# M4 oversample-before-split leakage: oversample the whole dataset, then split.
# M5 generator-on-all-data: fit SMOTE on all data; train rare classes on its output.
# M6 wrong fidelity metric: declare synthetic data "indistinguishable" from an MMD with a
#    mis-scaled kernel (fixed bandwidth on raw features) that is near zero regardless.

# %%
def correct(seed):
    Xtr, Xte, ytr, yte = train_test_split(X_all, y_all, test_size=0.3, stratify=y_all, random_state=seed)
    Xo, yo = oversample(Xtr, ytr, seed)
    p = clf(seed).fit(Xo, yo).predict(Xte)
    return {"macro_f1": macro_f1(yte, p), "accuracy": accuracy_score(yte, p)}

def m3_test_tuning(seed):
    Xtr, Xte, ytr, yte = train_test_split(X_all, y_all, test_size=0.3, stratify=y_all, random_state=seed)
    Xo, yo = oversample(Xtr, ytr, seed)
    best = -1
    for it in (50, 100, 200, 400):
        p = clf(seed, iters=it).fit(Xo, yo).predict(Xte); best = max(best, macro_f1(yte, p))
    # honest counterpart: tune on a validation split of train
    Xa, Xv, ya, yv = train_test_split(Xtr, ytr, test_size=0.2, stratify=ytr, random_state=seed)
    Xo2, yo2 = oversample(Xa, ya, seed)
    it = max((50, 100, 200, 400), key=lambda it: macro_f1(yv, clf(seed, iters=it).fit(Xo2, yo2).predict(Xv)))
    Xo, yo = oversample(Xtr, ytr, seed)
    honest = macro_f1(yte, clf(seed, iters=it).fit(Xo, yo).predict(Xte))
    return best, honest

def m4_oversample_first(seed):
    Xo, yo = oversample(X_all, y_all, seed)
    Xtr, Xte, ytr, yte = train_test_split(Xo, yo, test_size=0.3, stratify=yo, random_state=seed)
    return macro_f1(yte, clf(seed).fit(Xtr, ytr).predict(Xte))

def m5_gen_on_all(seed):
    Xtr, Xte, ytr, yte = train_test_split(X_all, y_all, test_size=0.3, stratify=y_all, random_state=seed)
    Xs, ys = oversample(X_all, y_all, seed); Xsyn, ysyn = Xs[len(X_all):], ys[len(X_all):]
    keep = ~np.isin(ytr, ["R2L", "U2R"])
    Xo = np.vstack([Xtr[keep], Xsyn]); yo = np.concatenate([ytr[keep], ysyn])
    return macro_f1(yte, clf(seed).fit(Xo, yo).predict(Xte))

def m6_mmd(seed):
    """Correct vs mis-scaled MMD^2 between two genuinely different distributions (real R2L
    and real normal). The correct metric (median bandwidth on standardised features) detects
    the difference; the mis-scaled one (fixed bandwidth on raw features) reports ~0, which a
    careless study would read as 'indistinguishable'."""
    rng = np.random.RandomState(seed)
    num = [i for i in range(41) if i not in CAT_IDX]
    R = X_all[y_all == "R2L"][:, num]; N = X_all[y_all == "normal"][:, num]
    R = R[rng.choice(len(R), min(500, len(R)), replace=False)]; N = N[rng.choice(len(N), 500, replace=False)]
    def mmd2(A, Bb, bw):
        k = lambda P, Q: np.exp(-((P[:, None] - Q[None]) ** 2).sum(2) / (2 * bw ** 2))
        m, n = len(A), len(Bb)
        return (k(A, A).sum() - np.trace(k(A, A))) / (m * (m - 1)) + (k(Bb, Bb).sum() - np.trace(k(Bb, Bb))) / (n * (n - 1)) - 2 * k(A, Bb).mean()
    sd = np.vstack([R, N]).std(0) + 1e-9; Rn, Nn = R / sd, N / sd
    med = np.median(np.sqrt(((Rn[:200, None] - Rn[None, :200]) ** 2).sum(2)) + 1e-9)
    return {"mmd_correct": float(mmd2(Rn, Nn, med)), "mmd_wrong": float(mmd2(R, N, 1.0))}

# %%
RESULTS = {}
base = [correct(s) for s in SEEDS]
RESULTS["correct"] = base
mf = [b["macro_f1"] for b in base]; acc = [b["accuracy"] for b in base]
print(f"Correct baseline: macro-F1 {np.mean(mf):.3f} ± {np.std(mf, ddof=1):.3f}  accuracy {np.mean(acc):.3f} ± {np.std(acc, ddof=1):.3f}")
print(f"M1 accuracy paradox:  headline accuracy {np.mean(acc):.3f} vs true macro-F1 {np.mean(mf):.3f}  (gap {np.mean(acc)-np.mean(mf):+.3f})")
print(f"M2 selective seeds:   best-of-5 macro-F1 {max(mf):.3f} vs mean {np.mean(mf):.3f}  (inflation {max(mf)-np.mean(mf):+.3f})")
m3 = [m3_test_tuning(s) for s in SEEDS]; RESULTS["m3"] = m3
print(f"M3 test-set tuning:   {np.mean([a for a,_ in m3]):.3f} vs honest (val-tuned) {np.mean([b for _,b in m3]):.3f}  (inflation {np.mean([a-b for a,b in m3]):+.3f})")
m4 = [m4_oversample_first(s) for s in SEEDS]; RESULTS["m4"] = m4
print(f"M4 oversample-first:  {np.mean(m4):.3f} vs correct {np.mean(mf):.3f}  (inflation {np.mean(m4)-np.mean(mf):+.3f})")
m5 = [m5_gen_on_all(s) for s in SEEDS]; RESULTS["m5"] = m5
print(f"M5 generator-on-all:  {np.mean(m5):.3f} vs correct {np.mean(mf):.3f}  (inflation {np.mean(m5)-np.mean(mf):+.3f})")
m6 = [m6_mmd(s) for s in SEEDS]; RESULTS["m6"] = m6
cw = np.mean([x["mmd_correct"] for x in m6]); ww = np.mean([x["mmd_wrong"] for x in m6])
print(f"M6 wrong MMD kernel:  R2L vs normal correct MMD^2 {cw:.3f} (distinguishable) vs mis-scaled {ww:.5f} (~0, 'indistinguishable')")
Path(OUT / "ch17_results.json").write_text(json.dumps(RESULTS, indent=1, default=float))
print("done")
