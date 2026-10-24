# %% [markdown]
# # Chapter 23 companion notebook: engineering and deployment
#
# Experiment 23.1 (drift and retraining): NSL-KDD's KDDTest+ is a genuine distribution shift
#   from KDDTrain+ (harder, with 17 attack types absent from training). We measure the
#   performance a detector loses to the shift, an unsupervised drift score that fires without
#   labels, and the recovery from retraining on a small labelled sample of the new data.
# Experiment 23.2 (cost and footprint): training time, sampling time, serialized model size
#   and inference throughput for the detector and each generator, for edge-deployment budgets.
#
#   python ch23_deploy.py       # writes out/ch23_results.json
# Reuses the Chapter 13 notebook for the generators. Requirements: torch, numpy, pandas,
# scikit-learn.

# %%
from pathlib import Path
import json, sys, time, io, pickle, warnings
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score, recall_score, roc_auc_score

warnings.filterwarnings("ignore")
CH13 = Path("/home/claude/ch13/ch13_minority.py")
_src = CH13.read_text(); sys.argv = ["ch13_minority.py"]
exec(_src[:_src.rindex('if MODE in ("all", "worker")')])   # make_split, smote_r2l, cwgan_r2l, clf, CAT, X_all, CLASSES
OUT = Path("out"); OUT.mkdir(exist_ok=True)
DATA = Path("/home/claude/data/nslkdd")
SEEDS = [0, 1, 2, 3, 4]
RESULTS = {}

# Full NSL-KDD attack -> category map, including the 17 test-only attack types
CAT_FULL = {"normal": "normal",
 "back": "DoS", "land": "DoS", "neptune": "DoS", "pod": "DoS", "smurf": "DoS", "teardrop": "DoS",
 "apache2": "DoS", "udpstorm": "DoS", "processtable": "DoS", "mailbomb": "DoS", "worm": "DoS",
 "satan": "Probe", "ipsweep": "Probe", "nmap": "Probe", "portsweep": "Probe", "mscan": "Probe", "saint": "Probe",
 "guess_passwd": "R2L", "ftp_write": "R2L", "imap": "R2L", "phf": "R2L", "multihop": "R2L",
 "warezmaster": "R2L", "warezclient": "R2L", "spy": "R2L", "xlock": "R2L", "xsnoop": "R2L",
 "snmpguess": "R2L", "snmpgetattack": "R2L", "httptunnel": "R2L", "sendmail": "R2L", "named": "R2L",
 "buffer_overflow": "U2R", "loadmodule": "U2R", "rootkit": "U2R", "perl": "U2R",
 "sqlattack": "U2R", "xterm": "U2R", "ps": "U2R"}

def load(path):
    d = pd.read_csv(path, header=None); y = d[41].map(CAT_FULL).values
    X = d.iloc[:, :41].copy(); X.columns = [f"f{i}" for i in range(41)]
    for c in CAT: X[c] = X[c].astype("category").cat.codes
    return X.astype(float), y

# %% [markdown]
# ## Experiment 23.1: drift, its unsupervised signal, and retraining
#
# For each seed: split KDDTrain+ 70/30 into a training part and an in-distribution holdout.
# Train the detector on the training part (R2L augmented to 5,000 by SMOTE-NC, the book's
# baseline augmenter). Evaluate macro-F1 on the in-distribution holdout and on KDDTest+ (the
# shifted deployment data). The drop between them is the cost of distribution drift.
#
# Drift score (unsupervised, no labels): a classifier trained to tell training-part records
# from KDDTest+ records; its ROC-AUC measures how far the deployment distribution has moved.
#
# Retraining: add a fraction f of KDDTest+ (labelled) to the training data, retrain, and
# evaluate on the held-out remainder of KDDTest+. f = 0, 0.05, 0.10, 0.25.

# %%
Xtrain_full, ytrain_full = load(DATA / "KDDTrain+.txt")
Xtest_full, ytest_full = load(DATA / "KDDTest+.txt")
FRACS = [0.0, 0.05, 0.10, 0.25]

def macro(model, X, y): return f1_score(y, model.predict(X), labels=CLASSES, average="macro", zero_division=0)

rows = {"in_dist": [], "shifted": [], "drift_auc": [], "retrain": {f"{f}": [] for f in FRACS}}
for s in SEEDS:
    Xtr, Xhold, ytr, yhold = train_test_split(Xtrain_full, ytrain_full, test_size=0.3, stratify=ytrain_full, random_state=s)
    Xtr, ytr = Xtr.reset_index(drop=True), ytr
    # augment R2L with SMOTE to 5000
    m = max(5000 - int((ytr == "R2L").sum()), 0); S = smote_r2l(Xtr, ytr, m, s)
    Xa = pd.concat([Xtr, S], ignore_index=True); ya = np.r_[ytr, ["R2L"] * len(S)]
    det = clf(s).fit(Xa, ya)
    rows["in_dist"].append(macro(det, Xhold, yhold))
    rows["shifted"].append(macro(det, Xtest_full, ytest_full))
    # unsupervised drift score: train vs deployment discriminator (balanced)
    rng = np.random.RandomState(s); n = min(len(Xtr), len(Xtest_full), 8000)
    A = Xtr.iloc[rng.choice(len(Xtr), n, replace=False)]; B = Xtest_full.iloc[rng.choice(len(Xtest_full), n, replace=False)]
    Xd = pd.concat([A, B], ignore_index=True); yd = np.r_[np.zeros(n), np.ones(n)]
    idx = rng.permutation(len(Xd)); cut = int(0.7 * len(Xd))
    disc = HistGradientBoostingClassifier(random_state=s, categorical_features=[1, 2, 3]).fit(Xd.iloc[idx[:cut]], yd[idx[:cut]])
    rows["drift_auc"].append(roc_auc_score(yd[idx[cut:]], disc.predict_proba(Xd.iloc[idx[cut:]])[:, 1]))
    # retraining with a fraction of the deployment data
    Xnew, Xeval, ynew, yeval = train_test_split(Xtest_full, ytest_full, test_size=0.5, stratify=ytest_full, random_state=s)
    for f in FRACS:
        if f == 0.0:
            mdl = det
        else:
            k = int(f * len(Xnew)); sel = np.random.RandomState(s).choice(len(Xnew), k, replace=False)
            Xr = pd.concat([Xa, Xnew.iloc[sel]], ignore_index=True); yr = np.r_[ya, ynew[sel]]
            mdl = clf(s).fit(Xr, yr)
        rows["retrain"][f"{f}"].append(macro(mdl, Xeval, yeval))
    print(f"seed {s}: in-dist {rows['in_dist'][-1]:.3f}  shifted {rows['shifted'][-1]:.3f}  drift-AUC {rows['drift_auc'][-1]:.3f}", flush=True)

def ms(a): return [float(np.mean(a)), float(np.std(a, ddof=1))]
RESULTS["exp23_1"] = {"in_dist": ms(rows["in_dist"]), "shifted": ms(rows["shifted"]),
                      "drift_auc": ms(rows["drift_auc"]),
                      "retrain": {f: ms(v) for f, v in rows["retrain"].items()}}
print("\nExperiment 23.1 summary:")
print(f"  macro-F1 in-distribution holdout : {np.mean(rows['in_dist']):.3f} ± {np.std(rows['in_dist'],ddof=1):.3f}")
print(f"  macro-F1 on shifted KDDTest+     : {np.mean(rows['shifted']):.3f} ± {np.std(rows['shifted'],ddof=1):.3f}")
print(f"  unsupervised drift ROC-AUC       : {np.mean(rows['drift_auc']):.3f} ± {np.std(rows['drift_auc'],ddof=1):.3f}")
for f in FRACS:
    v = rows["retrain"][f"{f}"]; print(f"  retrain with {int(f*100):2d}% new data  : {np.mean(v):.3f} ± {np.std(v,ddof=1):.3f}")

# %% [markdown]
# ## Experiment 23.2: cost and footprint
#
# Median over 3 repeats of: detector training time; the time to generate 5,000 R2L records
# with each generator; the serialized size of each fitted model; and detector inference
# throughput (records/second). These are the quantities an edge deployment is budgeted against.

# %%
def size_mb(obj):
    b = io.BytesIO(); pickle.dump(obj, b); return len(b.getvalue()) / 1e6

Xtr, ytr, Xte, yte = make_split(0, 100)
m = 5000 - int((ytr == "R2L").sum())
def t_med(fn, r=3):
    ts = []
    for _ in range(r):
        t0 = time.time(); out = fn(); ts.append(time.time() - t0)
    return float(np.median(ts)), out

det_t, det = t_med(lambda: clf(0).fit(pd.concat([Xtr, smote_r2l(Xtr, ytr, m, 0)], ignore_index=True),
                                      np.r_[ytr, ["R2L"] * m]))
smote_t, _ = t_med(lambda: smote_r2l(Xtr, ytr, m, 0))
cwgan_t, _ = t_med(lambda: cwgan_r2l(Xtr, ytr, m, 0), r=1)   # expensive: single timing
# inference throughput
big = pd.concat([Xte] * 10, ignore_index=True)
t0 = time.time(); det.predict(big); infer_s = time.time() - t0
thru = len(big) / infer_s
RESULTS["exp23_2"] = {"detector_fit_s": det_t, "smote_generate_s": smote_t, "cwgan_generate_s": cwgan_t,
                      "detector_size_mb": size_mb(det), "infer_records_per_s": float(thru), "infer_n": int(len(big))}
print("\nExperiment 23.2 (cost, median of repeats):")
print(f"  detector training time        : {det_t:.2f} s")
print(f"  SMOTE generate 5000 records   : {smote_t:.3f} s")
print(f"  cWGAN generate 5000 records   : {cwgan_t:.1f} s (train + sample)")
print(f"  detector serialized size      : {size_mb(det):.2f} MB")
print(f"  detector inference throughput : {thru:,.0f} records/s ({len(big)} records in {infer_s:.3f}s)")
Path(OUT / "ch23_results.json").write_text(json.dumps(RESULTS, indent=1))
print("done")
