# %% [markdown]
# # Chapter 14 companion notebook: GAN-based anomaly detection on NSL-KDD
#
# Every number in Chapter 14 is produced by this file. Data: `KDDTrain+.txt` and
# `KDDTest+.txt` from the NSL-KDD distribution (set DATA).
#   python ch14_anomaly.py worker 0 2 ; python ch14_anomaly.py worker 1 2
#   python ch14_anomaly.py report
# Requirements: torch, numpy, pandas, scikit-learn.

# %%
from pathlib import Path
import json, sys, time, warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import QuantileTransformer
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, average_precision_score

warnings.filterwarnings("ignore")
DATA = Path("/home/claude/data/nslkdd")
OUT = Path("out"); OUT.mkdir(exist_ok=True)
RUNS = OUT / "runs"; RUNS.mkdir(exist_ok=True)
SEEDS = [0, 1, 2, 3, 4]
MODE = sys.argv[1] if len(sys.argv) > 1 else "all"
torch.set_num_threads(1)
CATEGORY = {"normal": "normal",
    "back": "DoS", "land": "DoS", "neptune": "DoS", "pod": "DoS", "smurf": "DoS", "teardrop": "DoS",
    "ipsweep": "Probe", "nmap": "Probe", "portsweep": "Probe", "satan": "Probe",
    "ftp_write": "R2L", "guess_passwd": "R2L", "imap": "R2L", "multihop": "R2L", "phf": "R2L", "spy": "R2L",
    "warezclient": "R2L", "warezmaster": "R2L",
    "buffer_overflow": "U2R", "loadmodule": "U2R", "perl": "U2R", "rootkit": "U2R"}
ATTACKS = ["DoS", "Probe", "R2L", "U2R"]

# %% [markdown]
# ## Data
#
# Training: the normal records of KDDTrain+ only (67,343), as an anomaly detector
# is trained without attacks. A random 20,000 of them are used per seed. Testing:
# KDDTest+ restricted to normal records and the 22 attack types seen in training
# (as in Chapter 7). Encoding: Gaussian quantile transform of the numeric columns
# (fitted on the training normal records) and one-hot categorical columns.

# %%
cols = list(range(41))
tr = pd.read_csv(DATA / "KDDTrain+.txt", header=None); te = pd.read_csv(DATA / "KDDTest+.txt", header=None)
te = te[te[41].isin(CATEGORY)].reset_index(drop=True)
tr["cat"] = tr[41].map(CATEGORY); te["cat"] = te[41].map(CATEGORY)
NUMC = [c for c in cols if c not in (1, 2, 3) and tr[c].nunique() > 1]
CATC = [1, 2, 3]
LEVELS = {c: sorted(set(tr[c]) | set(te[c])) for c in CATC}
NORMAL = tr[tr["cat"] == "normal"].reset_index(drop=True)

def encoder_for(train_rows):
    q = QuantileTransformer(n_quantiles=1000, output_distribution="normal", random_state=0).fit(train_rows[NUMC].astype(float))
    def enc(df):
        parts = [q.transform(df[NUMC].astype(float))]
        for c in CATC: parts.append(np.eye(len(LEVELS[c]))[pd.Categorical(df[c], categories=LEVELS[c]).codes])
        return np.hstack(parts).astype(np.float32)
    return enc

# %% [markdown]
# ## Detectors (anomaly score: higher = more anomalous)
#
# * iforest   : isolation forest, 200 trees
# * ae        : autoencoder (encoder 3 x 128 to a 16-d code, mirrored decoder); score = reconstruction error
# * dscore    : the discriminator of a GAN trained on normal records (NS + R1); score = -logit
# * anogan    : AnoGAN-style: the same GAN; each test record is mapped to the latent space by
#               100 Adam steps on (1 - lambda) * residual + lambda * feature-matching loss,
#               lambda = 0.1; score = the final loss
# * bigan     : BiGAN / EGBAD-style: generator, encoder and joint discriminator D(x, z);
#               score = (1 - lambda) * ||x - G(E(x))||_1 + lambda * feature-matching, lambda = 0.1
# * ganomaly  : GANomaly-style: encoder-decoder-encoder generator with an adversarial
#               feature-matching loss; score = ||E1(x) - E2(G(x))||^2
# All neural detectors: MLPs with LeakyReLU, latent dimension 16, Adam 1e-4 (0.5, 0.999),
# batch 256, 3,000 updates.

# %%
H, ZD, B, STEPS = 128, 16, 256, 3000

def mlp(i, o, depth=3):
    L, d = [], i
    for _ in range(depth): L += [nn.Linear(d, H), nn.LeakyReLU(0.2)]; d = H
    return nn.Sequential(*L, nn.Linear(d, o))

class Disc(nn.Module):
    def __init__(self, d_in):
        super().__init__(); self.f = mlp(d_in, H, 2); self.out = nn.Linear(H, 1)
    def forward(self, x): h = torch.nn.functional.leaky_relu(self.f(x), 0.2); return self.out(h), h

def batches(X):
    while True: yield X[torch.randint(0, len(X), (B,))]

def train_gan(X):
    D_ = X.shape[1]; G, D = mlp(ZD, D_), Disc(D_)
    og = torch.optim.Adam(G.parameters(), 1e-4, betas=(0.5, 0.999)); od = torch.optim.Adam(D.parameters(), 1e-4, betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss(); it = batches(X)
    for _ in range(STEPS):
        xr = next(it).clone().requires_grad_(True)
        with torch.no_grad(): xg = G(torch.randn(B, ZD))
        sr, _ = D(xr); sg, _ = D(xg)
        g = torch.autograd.grad(sr.sum(), xr, create_graph=True)[0]
        ld = bce(sr, torch.ones_like(sr)) + bce(sg, torch.zeros_like(sg)) + 5.0 * g.pow(2).sum(1).mean()
        od.zero_grad(); ld.backward(); od.step()
        s, _ = D(G(torch.randn(B, ZD))); lg = bce(s, torch.ones_like(s))
        og.zero_grad(); lg.backward(); og.step()
    return G, D

def anogan_scores(G, D, X, lam=0.1, steps=100):
    for p in list(G.parameters()) + list(D.parameters()): p.requires_grad_(False)
    out = []
    for i in range(0, len(X), 2048):
        x = X[i:i + 2048]; z = torch.zeros(len(x), ZD, requires_grad=True); oz = torch.optim.Adam([z], 1e-2)
        _, fx = D(x)
        for _ in range(steps):
            gz = G(z); _, fg = D(gz)
            loss = (1 - lam) * (x - gz).abs().sum(1) + lam * (fx - fg).abs().sum(1)
            oz.zero_grad(); loss.sum().backward(); oz.step()
        out.append(loss.detach().numpy())
    return np.concatenate(out)

def train_bigan(X):
    D_ = X.shape[1]; G, E = mlp(ZD, D_), mlp(D_, ZD); D = Disc(D_ + ZD)
    og = torch.optim.Adam(list(G.parameters()) + list(E.parameters()), 1e-4, betas=(0.5, 0.999)); od = torch.optim.Adam(D.parameters(), 1e-4, betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss(); it = batches(X)
    for _ in range(STEPS):
        x = next(it); z = torch.randn(B, ZD)
        sr, _ = D(torch.cat([x, E(x)], 1)); sg, _ = D(torch.cat([G(z), z], 1))
        ld = bce(sr, torch.ones_like(sr)) + bce(sg, torch.zeros_like(sg))
        od.zero_grad(); ld.backward(retain_graph=False); od.step()
        x = next(it); z = torch.randn(B, ZD)
        sr, _ = D(torch.cat([x, E(x)], 1)); sg, _ = D(torch.cat([G(z), z], 1))
        lg = bce(sr, torch.zeros_like(sr)) + bce(sg, torch.ones_like(sg))
        og.zero_grad(); lg.backward(); og.step()
    def score(Xt, lam=0.1):
        with torch.no_grad():
            z = E(Xt); xr = G(z); _, f1 = D(torch.cat([Xt, z], 1)); _, f2 = D(torch.cat([xr, z], 1))
            return ((1 - lam) * (Xt - xr).abs().sum(1) + lam * (f1 - f2).abs().sum(1)).numpy()
    return score

def train_ganomaly(X):
    D_ = X.shape[1]; E1, Dec, E2 = mlp(D_, ZD), mlp(ZD, D_), mlp(D_, ZD); D = Disc(D_)
    og = torch.optim.Adam(list(E1.parameters()) + list(Dec.parameters()) + list(E2.parameters()), 1e-4, betas=(0.5, 0.999))
    od = torch.optim.Adam(D.parameters(), 1e-4, betas=(0.5, 0.999)); bce = nn.BCEWithLogitsLoss(); it = batches(X)
    for _ in range(STEPS):
        x = next(it); z = E1(x); xh = Dec(z); zh = E2(xh)
        _, fr = D(x); _, ff = D(xh)
        lg = 1.0 * (fr.detach() - ff).pow(2).mean() + 50.0 * (x - xh).abs().mean() + 1.0 * (z - zh).pow(2).mean()
        og.zero_grad(); lg.backward(); og.step()
        sr, _ = D(x); sf, _ = D(xh.detach())
        ld = bce(sr, torch.ones_like(sr)) + bce(sf, torch.zeros_like(sf))
        od.zero_grad(); ld.backward(); od.step()
    def score(Xt):
        with torch.no_grad(): return (E1(Xt) - E2(Dec(E1(Xt)))).pow(2).sum(1).numpy()
    return score

def train_ae(X):
    D_ = X.shape[1]; E, Dc = mlp(D_, ZD), mlp(ZD, D_)
    o = torch.optim.Adam(list(E.parameters()) + list(Dc.parameters()), 1e-3); it = batches(X)
    for _ in range(STEPS):
        x = next(it); l = (Dc(E(x)) - x).pow(2).mean(); o.zero_grad(); l.backward(); o.step()
    def score(Xt):
        with torch.no_grad(): return (Dc(E(Xt)) - Xt).pow(2).sum(1).numpy()
    return score

# %% [markdown]
# ## Measures, per attack category against normal test records
#
#   AUROC, average precision (AUPRC), and detection rate at a 1% false-positive rate
#   (threshold = 99th percentile of the scores of 5,000 held-out normal training records)

# %%
def measures(score_test, score_val):
    thr = np.quantile(score_val, 0.99); yN = (te["cat"] == "normal").values; res = {}
    for a in ATTACKS + ["all"]:
        m = yN | (te["cat"] == a).values if a != "all" else np.ones(len(te), bool)
        y = (~yN)[m]; s = score_test[m]
        res[a] = {"auroc": float(roc_auc_score(y, s)), "auprc": float(average_precision_score(y, s)),
                  "det_at_1fpr": float(np.mean(s[y] > thr))}
    res["fpr_test"] = float(np.mean(score_test[yN] > thr))
    return res

def run(kind, seed):
    torch.manual_seed(seed); rng = np.random.RandomState(seed)
    idx = rng.permutation(len(NORMAL)); trn = NORMAL.iloc[idx[:20000]]; val = NORMAL.iloc[idx[20000:25000]]
    enc = encoder_for(trn); Xtr, Xval, Xte = map(lambda d: torch.tensor(enc(d)), (trn, val, te))
    t0 = time.time()
    if kind == "iforest":
        f = IsolationForest(n_estimators=200, random_state=seed).fit(Xtr.numpy())
        sc = lambda X: -f.score_samples(X.numpy())
    elif kind == "ae": sc = train_ae(Xtr)
    elif kind in ("dscore", "anogan"):
        G, D = train_gan(Xtr)
        if kind == "dscore":
            sc = lambda X: -D(X)[0].detach().numpy().ravel()
        else:
            sc = lambda X: anogan_scores(G, D, X)
    elif kind == "bigan": sc = train_bigan(Xtr)
    elif kind == "ganomaly": sc = train_ganomaly(Xtr)
    s_te, s_val = sc(Xte), sc(Xval)
    out = measures(s_te, s_val); out["seconds"] = time.time() - t0
    return out

KINDS = ["iforest", "ae", "dscore", "anogan", "bigan", "ganomaly"]
JOBS = [(k, s) for k in ["anogan", "bigan", "ganomaly", "dscore", "ae", "iforest"] for s in SEEDS]
def job_path(j): return RUNS / f"{j[0]}_{j[1]}.json"

def do_job(j):
    p = job_path(j)
    if p.exists(): return
    t0 = time.time(); r = run(*j); p.write_text(json.dumps(r)); print(f"job {j} done [{time.time()-t0:.0f}s]", flush=True)

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
print("test records:", te["cat"].value_counts().to_dict())
RESULTS = {}
for k in KINDS:
    rows = [json.loads(job_path((k, s)).read_text()) for s in SEEDS]; RESULTS[k] = rows
    f = lambda a, m: f"{np.mean([r[a][m] for r in rows]):.3f}±{np.std([r[a][m] for r in rows], ddof=1):.3f}"
    print(f"{k:9s} " + "  ".join(f"{a}: AUROC {f(a,'auroc')} AP {f(a,'auprc')} det@1% {f(a,'det_at_1fpr')}" for a in ATTACKS + ["all"])
          + f"  test FPR {np.mean([r['fpr_test'] for r in rows]):.3f}  time {np.mean([r['seconds'] for r in rows]):.0f}s", flush=True)
Path(OUT / "ch14_results.json").write_text(json.dumps(RESULTS, indent=1))
print("done")
