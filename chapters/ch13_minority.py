# %% [markdown]
# # Chapter 13 companion notebook: minority-class augmentation and the sample-size boundary
#
# Every number in Chapter 13 is produced by this file. Data: `KDDTrain+.txt` from the
# NSL-KDD distribution (set DATA). Requirements: torch, numpy, pandas, scikit-learn.
#   python ch13_minority.py worker 0 2 ; python ch13_minority.py worker 1 2
#   python ch13_minority.py report

# %%
from pathlib import Path
import json, sys, time, warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score, recall_score, precision_score, average_precision_score
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings("ignore")
DATA = Path("/home/claude/data/nslkdd/KDDTrain+.txt")
OUT = Path("out"); OUT.mkdir(exist_ok=True)
RUNS = OUT / "runs"; RUNS.mkdir(exist_ok=True)
SEEDS = [0, 1, 2, 3, 4]
MODE = sys.argv[1] if len(sys.argv) > 1 else "all"
torch.set_num_threads(1)
CLASSES = ["normal", "DoS", "Probe", "R2L", "U2R"]
CATEGORY = {"normal": "normal",
    "back": "DoS", "land": "DoS", "neptune": "DoS", "pod": "DoS", "smurf": "DoS", "teardrop": "DoS",
    "ipsweep": "Probe", "nmap": "Probe", "portsweep": "Probe", "satan": "Probe",
    "ftp_write": "R2L", "guess_passwd": "R2L", "imap": "R2L", "multihop": "R2L", "phf": "R2L", "spy": "R2L",
    "warezclient": "R2L", "warezmaster": "R2L",
    "buffer_overflow": "U2R", "loadmodule": "U2R", "perl": "U2R", "rootkit": "U2R"}

# %% [markdown]
# ## Data and splits
#
# Full KDDTrain+ (125,973 records), 41 features, label = category. For each seed the
# records are split 70/30, stratified by category. The test part is never touched by
# any augmentation. In the training part, the R2L class is reduced to n records chosen
# at random (n = 25, 50, 100, 200, 400, or all, 696 or 697 depending on the split);
# every other class is kept whole.

# %%
d = pd.read_csv(DATA, header=None)
d = d.drop_duplicates(subset=list(range(42))).reset_index(drop=True)   # remove exact duplicates first (Section 8.2)
y_all = d[41].map(CATEGORY).values
X_all = d.iloc[:, :41].copy(); X_all.columns = [f"f{i}" for i in range(41)]
CAT = ["f1", "f2", "f3"]
for c in CAT: X_all[c] = X_all[c].astype("category").cat.codes
NUM = [c for c in X_all.columns if c not in CAT]
X_all = X_all.astype(float)
NS = [25, 50, 100, 200, 400, "all"]
TARGET = 5000                                   # R2L training records after augmentation

def make_split(seed, n):
    Xtr, Xte, ytr, yte = train_test_split(X_all, y_all, test_size=0.3, stratify=y_all, random_state=seed)
    Xtr, Xte = Xtr.reset_index(drop=True), Xte.reset_index(drop=True)
    if n != "all":
        rng = np.random.RandomState(1000 + seed)
        r2l = np.where(ytr == "R2L")[0]; keep = rng.choice(r2l, size=n, replace=False)
        mask = (ytr != "R2L"); mask[keep] = True
        Xtr, ytr = Xtr[mask].reset_index(drop=True), ytr[mask]
    return Xtr, ytr, Xte, yte

# %% [markdown]
# ## Augmentation methods (applied to R2L only, to TARGET records)
#
# * none         : real training data only
# * weights      : real data, sample weights inversely proportional to class frequency
# * weights_thr  : as weights, then the R2L decision threshold is tuned on a validation part
#                  of the training data (20%, stratified) to maximise R2L F1 — the
#                  threshold-tuned baseline recommended in Section 12.5
# * ros          : random oversampling of the R2L records
# * smote        : SMOTE-NC on the R2L records (k = 5 or n - 1 if smaller)
# * cwgan        : conditional WGAN-GP trained on the training part (each class capped at
#                  5,000 records), labels concatenated to the inputs of both networks;
#                  continuous features through a Gaussian quantile transform, categorical
#                  features one-hot with Gumbel-softmax outputs (temperature 0.2);
#                  generator 3 x 256, critic 3 x 256, lambda = 10, 5 critic steps,
#                  Adam 1e-4 (0, 0.9), 2,000 generator updates, batch 256.

# %%
def clf(seed, weights=None):
    return HistGradientBoostingClassifier(random_state=seed, categorical_features=[1, 2, 3])

def smote_r2l(Xtr, ytr, m, seed):
    rng = np.random.RandomState(seed)
    X = Xtr[ytr == "R2L"].values; n = len(X); k = min(5, n - 1)
    num = [i for i, c in enumerate(Xtr.columns) if c not in CAT]; cat = [i for i, c in enumerate(Xtr.columns) if c in CAT]
    mu, sd = X[:, num].mean(0), X[:, num].std(0) + 1e-9
    Z = np.hstack([(X[:, num] - mu) / sd] + [np.eye(int(X_all[c].max()) + 1)[X[:, i].astype(int)] / np.sqrt(2) for i, c in zip(cat, CAT)])
    nb = NearestNeighbors(n_neighbors=k + 1).fit(Z).kneighbors(Z, return_distance=False)[:, 1:]
    a = rng.randint(0, n, m); b = nb[a, rng.randint(0, k, m)]; lam = rng.rand(m, 1)
    S = X[a] + lam * (X[b] - X[a])
    for i in cat:                                            # categorical: most frequent among the k neighbours
        vals = X[nb[a], i]; S[:, i] = [np.bincount(v.astype(int)).argmax() for v in vals]
    return pd.DataFrame(S, columns=Xtr.columns)

class Codec:
    def __init__(self, X):
        self.q = QuantileTransformer(n_quantiles=1000, output_distribution="normal", random_state=0).fit(X[NUM])
        self.levels = {c: int(X_all[c].max()) + 1 for c in CAT}
        self.dim = len(NUM) + sum(self.levels.values())
    def enc(self, X):
        return np.hstack([self.q.transform(X[NUM])] + [np.eye(self.levels[c])[X[c].astype(int)] for c in CAT]).astype(np.float32)
    def dec(self, Z):
        out = pd.DataFrame(self.q.inverse_transform(np.clip(Z[:, :len(NUM)], -5.2, 5.2)), columns=NUM); k = len(NUM)
        for c in CAT: out[c] = Z[:, k:k + self.levels[c]].argmax(1); k += self.levels[c]
        return out[X_all.columns]

def mlp(i, o, w=256, depth=3):
    L, dd = [], i
    for _ in range(depth): L += [nn.Linear(dd, w), nn.LeakyReLU(0.2)]; dd = w
    return nn.Sequential(*L, nn.Linear(dd, o))

def cwgan_r2l(Xtr, ytr, m, seed, steps=2000):
    torch.manual_seed(seed); rng = np.random.RandomState(seed)
    idx = np.concatenate([rng.choice(np.where(ytr == c)[0], size=min(5000, (ytr == c).sum()), replace=False) for c in CLASSES])
    X, y = Xtr.iloc[idx].reset_index(drop=True), ytr[idx]
    codec = Codec(X); D_ = codec.dim; K = len(CLASSES)
    Xt = torch.tensor(codec.enc(X)); Y = torch.tensor(np.eye(K)[[CLASSES.index(v) for v in y]], dtype=torch.float32)
    G, C = mlp(32 + K, D_), mlp(D_ + K, 1)
    og = torch.optim.Adam(G.parameters(), 1e-4, betas=(0.0, 0.9)); oc = torch.optim.Adam(C.parameters(), 1e-4, betas=(0.0, 0.9))
    blocks = [len(NUM)] + [codec.levels[c] for c in CAT]
    def act(h):
        out = [h[:, :blocks[0]]]; k = blocks[0]
        for b in blocks[1:]: out.append(F.gumbel_softmax(h[:, k:k + b], tau=0.2)); k += b
        return torch.cat(out, 1)
    B = 256
    for t in range(steps):
        for _ in range(5):
            i = torch.randint(0, len(Xt), (B,)); xr, yr = Xt[i], Y[i]
            with torch.no_grad(): xg = act(G(torch.cat([torch.randn(B, 32), yr], 1)))
            a = torch.rand(B, 1); xi = (a * xr + (1 - a) * xg).requires_grad_(True)
            g = torch.autograd.grad(C(torch.cat([xi, yr], 1)).sum(), xi, create_graph=True)[0]
            lc = C(torch.cat([xg, yr], 1)).mean() - C(torch.cat([xr, yr], 1)).mean() + 10 * ((g.norm(dim=1) - 1) ** 2).mean()
            oc.zero_grad(); lc.backward(); oc.step()
        i = torch.randint(0, len(Xt), (B,)); yr = Y[i]
        lg = -C(torch.cat([act(G(torch.cat([torch.randn(B, 32), yr], 1))), yr], 1)).mean()
        og.zero_grad(); lg.backward(); og.step()
    with torch.no_grad():
        yr = torch.tensor(np.tile(np.eye(K)[CLASSES.index("R2L")], (m, 1)), dtype=torch.float32)
        Z = act(G(torch.cat([torch.randn(m, 32), yr], 1))).numpy()
    return codec.dec(Z)

def evaluate(model, Xte, yte, thr=None):
    if thr is None:
        p = model.predict(Xte)
    else:
        proba = model.predict_proba(Xte); j = list(model.classes_).index("R2L")
        p = np.where(proba[:, j] >= thr, "R2L", model.classes_[np.argmax(np.delete(proba, j, 1), 1) + (np.argmax(np.delete(proba, j, 1), 1) >= j)])
        p = np.asarray(p, dtype=object)
    proba = model.predict_proba(Xte)[:, list(model.classes_).index("R2L")]
    return {"r2l_recall": float(recall_score(yte == "R2L", p == "R2L")),
            "r2l_precision": float(precision_score(yte == "R2L", p == "R2L", zero_division=0)),
            "r2l_ap": float(average_precision_score(yte == "R2L", proba)),
            "macro_f1": float(f1_score(yte, p, labels=CLASSES, average="macro", zero_division=0)),
            "u2r_recall": float(recall_score(yte == "U2R", p == "U2R"))}

def run(method, seed, n):
    Xtr, ytr, Xte, yte = make_split(seed, n); n_real = int((ytr == "R2L").sum()); m = max(TARGET - n_real, 0)
    info = {"n_real": n_real}
    if method == "none":
        return {**evaluate(clf(seed).fit(Xtr, ytr), Xte, yte), **info}
    if method in ("weights", "weights_thr"):
        w = pd.Series(ytr).map(len(ytr) / (len(CLASSES) * pd.Series(ytr).value_counts())).values
        if method == "weights":
            return {**evaluate(clf(seed).fit(Xtr, ytr, sample_weight=w), Xte, yte), **info}
        Xa, Xv, ya, yv, wa, _ = train_test_split(Xtr, ytr, w, test_size=0.2, stratify=ytr, random_state=seed)
        mdl = clf(seed).fit(Xa, ya, sample_weight=wa); pv = mdl.predict_proba(Xv)[:, list(mdl.classes_).index("R2L")]
        thr = max(np.linspace(0.05, 0.95, 19), key=lambda t: f1_score(yv == "R2L", pv >= t))
        return {**evaluate(clf(seed).fit(Xtr, ytr, sample_weight=w), Xte, yte, thr=thr), **info, "thr": float(thr)}
    if method == "ros":
        rng = np.random.RandomState(seed); r = np.where(ytr == "R2L")[0]; extra = Xtr.iloc[rng.choice(r, m)]
        Xa = pd.concat([Xtr, extra], ignore_index=True); ya = np.r_[ytr, ["R2L"] * m]
        return {**evaluate(clf(seed).fit(Xa, ya), Xte, yte), **info}
    if method == "smote":
        S = smote_r2l(Xtr, ytr, m, seed)
    elif method == "cwgan":
        t0 = time.time(); S = cwgan_r2l(Xtr, ytr, m, seed); info["gan_seconds"] = time.time() - t0
    Xa = pd.concat([Xtr, S], ignore_index=True); ya = np.r_[ytr, ["R2L"] * len(S)]
    return {**evaluate(clf(seed).fit(Xa, ya), Xte, yte), **info}

METHODS = ["none", "weights", "weights_thr", "ros", "smote", "cwgan"]
JOBS = [(m, s, n) for n in NS for s in SEEDS for m in ["cwgan"]] + [(m, s, n) for n in NS for s in SEEDS for m in METHODS if m != "cwgan"]
def job_path(j): return RUNS / f"{j[0]}_{j[1]}_{j[2]}.json"

def do_job(j):
    p = job_path(j)
    if p.exists(): return
    t0 = time.time(); res = run(*j); p.write_text(json.dumps(res))
    print(f"job {j} done [{time.time()-t0:.0f}s]", flush=True)

if MODE in ("all", "worker"):
    k, n_ = (int(sys.argv[2]), int(sys.argv[3])) if MODE == "worker" else (0, 1)
    for i, j in enumerate(JOBS):
        if i % n_ == k: do_job(j)
    if MODE == "worker": sys.exit(0)

# %% [markdown]
# ## Report

# %%
missing = [j for j in JOBS if not job_path(j).exists()]
if missing: raise SystemExit(f"{len(missing)} jobs not finished")
print("records after removing exact duplicates:", len(X_all), pd.Series(y_all).value_counts().to_dict())
_, _, _, yte0 = make_split(0, "all"); print("test part, seed 0:", pd.Series(yte0).value_counts().to_dict())
RESULTS = {}
for n in NS:
    for m in METHODS:
        rows = [json.loads(job_path((m, s, n)).read_text()) for s in SEEDS]; RESULTS[f"{m}/{n}"] = rows
        f = lambda k, d=3: f"{np.mean([r[k] for r in rows]):.{d}f} ± {np.std([r[k] for r in rows], ddof=1):.{d}f}"
        print(f"n={str(n):4s} {m:12s} R2L recall {f('r2l_recall')}  precision {f('r2l_precision')}  AP {f('r2l_ap')}  macro F1 {f('macro_f1')}  U2R recall {f('u2r_recall')}"
              + (f"  gan {f('gan_seconds',0)} s" if "gan_seconds" in rows[0] else ""), flush=True)
# paired differences against 'none' for R2L recall
print("\nPaired difference in R2L recall against 'none' (mean, sd over seeds):")
for n in NS:
    base = np.array([json.loads(job_path(("none", s, n)).read_text())["r2l_recall"] for s in SEEDS])
    line = f"n={str(n):4s} "
    for m in METHODS[1:]:
        v = np.array([json.loads(job_path((m, s, n)).read_text())["r2l_recall"] for s in SEEDS]) - base
        line += f" {m} {100*v.mean():+.1f}±{100*v.std(ddof=1):.1f}"
    print(line)
Path(OUT / "ch13_results.json").write_text(json.dumps(RESULTS, indent=1))
print("done")
