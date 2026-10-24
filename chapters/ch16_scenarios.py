# %% [markdown]
# # Chapter 16 companion notebook: diversity regularization and unknown-attack detection
#
# Defender's experiment. Attack SUBTYPES are held out of training to play the role of
# unknown ("zero-day") attacks. A conditional generator augments the KNOWN attack
# categories, with and without a diversity (mode-seeking) regularizer, and we measure
# whether the extra diversity helps a detector flag the held-out unknown subtypes.
# The notebook generates synthetic records of KNOWN attack categories only, to train a
# better detector; it saves no traffic and builds no evasion tool.
# Data: `KDDTrain+.txt` from the NSL-KDD distribution (set DATA).
#   python ch16_scenarios.py worker 0 2 ; python ch16_scenarios.py worker 1 2
#   python ch16_scenarios.py report
# Requirements: torch, numpy, pandas, scikit-learn.

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

warnings.filterwarnings("ignore")
DATA = Path("/home/claude/data/nslkdd/KDDTrain+.txt")
OUT = Path("out"); OUT.mkdir(exist_ok=True)
RUNS = OUT / "runs"; RUNS.mkdir(exist_ok=True)
SEEDS = [0, 1, 2]
MODE = sys.argv[1] if len(sys.argv) > 1 else "all"
torch.set_num_threads(1)
CATEGORY = {"normal": "normal",
    "back": "DoS", "land": "DoS", "neptune": "DoS", "pod": "DoS", "smurf": "DoS", "teardrop": "DoS",
    "ipsweep": "Probe", "nmap": "Probe", "portsweep": "Probe", "satan": "Probe",
    "ftp_write": "R2L", "guess_passwd": "R2L", "imap": "R2L", "multihop": "R2L", "phf": "R2L", "spy": "R2L",
    "warezclient": "R2L", "warezmaster": "R2L",
    "buffer_overflow": "U2R", "loadmodule": "U2R", "perl": "U2R", "rootkit": "U2R"}
# Subtypes held out as "unknown" (present only in the test set):
UNKNOWN = ["back", "portsweep"]           # a DoS subtype and a Probe subtype
AUG_CATS = ["DoS", "Probe"]               # categories the generator augments (known subtypes only)

# %% [markdown]
# ## Data
#
# Binary detector task: attack (1) vs normal (0). The training set contains normal
# records and the KNOWN attack subtypes; the two UNKNOWN subtypes are removed from
# training and kept only for testing. 70/30 split, stratified by subtype.

# %%
d = pd.read_csv(DATA, header=None)
d["sub"] = d[41]; d["cat"] = d[41].map(CATEGORY)
CAT = [1, 2, 3]
for c in CAT: d[c] = d[c].astype("category").cat.codes
NUM = [c for c in range(41) if c not in CAT]
INTEGER = [c for c in NUM if np.all(np.mod(d[c], 1) == 0)]
LEVELS = {c: int(d[c].max()) + 1 for c in CAT}
FEATS = NUM + CAT

tr_idx, te_idx = train_test_split(np.arange(len(d)), test_size=0.3, stratify=d["sub"], random_state=12345)
train_all = d.iloc[tr_idx].reset_index(drop=True); test_all = d.iloc[te_idx].reset_index(drop=True)
# remove unknown subtypes from training
train = train_all[~train_all["sub"].isin(UNKNOWN)].reset_index(drop=True)

class Codec:
    def __init__(self, X):
        self.q = QuantileTransformer(n_quantiles=1000, output_distribution="normal", random_state=0).fit(X[NUM].astype(float))
        self.dim = len(NUM) + sum(LEVELS[c] for c in CAT)
    def enc(self, X):
        return np.hstack([self.q.transform(X[NUM].astype(float))] + [np.eye(LEVELS[c])[X[c].astype(int)] for c in CAT]).astype(np.float32)
    def dec(self, Z):
        out = pd.DataFrame(self.q.inverse_transform(np.clip(Z[:, :len(NUM)], -5.2, 5.2)), columns=NUM); k = len(NUM)
        for c in CAT: out[c] = Z[:, k:k + LEVELS[c]].argmax(1); k += LEVELS[c]
        for c in INTEGER: out[c] = np.round(out[c])
        return out[FEATS]

def mlp(i, o, w=256, depth=3):
    L, dd = [], i
    for _ in range(depth): L += [nn.Linear(dd, w), nn.LeakyReLU(0.2)]; dd = w
    return nn.Sequential(*L, nn.Linear(dd, o))

# %% [markdown]
# ## Conditional WGAN-GP over the known attack categories, with optional diversity term
#
# Condition = one-hot over AUG_CATS. Generator maps (noise, label) to an encoded record.
# Diversity (mode-seeking, Mao et al.): maximize the ratio of output distance to latent
# distance between the two halves of the batch, added as L_div = 1 / (mean ratio + eps).

# %%
K = len(AUG_CATS); ZD = 32; B = 256

def gumbel_act(h, codec):
    blocks = [len(NUM)] + [LEVELS[c] for c in CAT]; out = [h[:, :blocks[0]]]; k = blocks[0]
    for b in blocks[1:]: out.append(F.gumbel_softmax(h[:, k:k + b], tau=0.2)); k += b
    return torch.cat(out, 1)

def train_cgan(seed, diversity, steps=2000):
    torch.manual_seed(seed); rng = np.random.RandomState(seed)
    sub = train[train["cat"].isin(AUG_CATS)]
    # cap each category at 5000 for balance
    idx = np.concatenate([rng.choice(sub.index[sub["cat"] == c], size=min(5000, (sub["cat"] == c).sum()), replace=False) for c in AUG_CATS])
    S = sub.loc[idx]; codec = Codec(train)
    X = torch.tensor(codec.enc(S)); Y = torch.tensor(np.eye(K)[[AUG_CATS.index(c) for c in S["cat"]]], dtype=torch.float32)
    G, C = mlp(ZD + K, codec.dim), mlp(codec.dim + K, 1)
    og = torch.optim.Adam(G.parameters(), 1e-4, betas=(0.0, 0.9)); oc = torch.optim.Adam(C.parameters(), 1e-4, betas=(0.0, 0.9))
    for t in range(steps):
        for _ in range(5):
            i = torch.randint(0, len(X), (B,)); xr, yr = X[i], Y[i]
            z = torch.randn(B, ZD)
            with torch.no_grad(): xg = gumbel_act(G(torch.cat([z, yr], 1)), codec)
            a = torch.rand(B, 1); xi = (a * xr + (1 - a) * xg).requires_grad_(True)
            g = torch.autograd.grad(C(torch.cat([xi, yr], 1)).sum(), xi, create_graph=True)[0]
            lc = C(torch.cat([xg, yr], 1)).mean() - C(torch.cat([xr, yr], 1)).mean() + 10 * ((g.norm(dim=1) - 1) ** 2).mean()
            oc.zero_grad(); lc.backward(); oc.step()
        i = torch.randint(0, len(X), (B,)); yr = Y[i]; z = torch.randn(B, ZD)
        xg = gumbel_act(G(torch.cat([z, yr], 1)), codec)
        lg = -C(torch.cat([xg, yr], 1)).mean()
        if diversity:
            h = B // 2
            dout = (xg[:h] - xg[h:2 * h]).abs().mean(1); dz = (z[:h] - z[h:2 * h]).abs().mean(1) + 1e-6
            lg = lg + 1.0 / ((dout / dz).mean() + 1e-5)      # mode-seeking: encourage large output spread per latent step
        og.zero_grad(); lg.backward(); og.step()
    def sample(m):
        yr = torch.tensor(np.eye(K)[rng.randint(0, K, m)], dtype=torch.float32)
        with torch.no_grad(): Z = gumbel_act(G(torch.cat([torch.randn(m, ZD), yr], 1)), codec).numpy()
        out = codec.dec(Z); out["cat"] = [AUG_CATS[i] for i in yr.argmax(1).numpy()]
        return out, codec, Z
    return sample

# %% [markdown]
# ## Detector and evaluation
#
# Detector: histogram gradient boosting, attack(1) vs normal(0). Trained on the known
# training set, optionally plus synthetic records of the known attack categories.
# Threshold set for 1% FPR on normal test records. We report recall on each UNKNOWN
# subtype (the zero-day test), on the known attack subtypes, and the overall FPR; plus
# the diversity of the generated DoS records (mean pairwise L1 distance in encoded space).

# %%
def detector_scores(train_df, extra=None):
    X = train_df[FEATS].astype(float); y = (train_df["cat"] != "normal").astype(int)
    if extra is not None:
        X = pd.concat([X, extra[FEATS].astype(float)], ignore_index=True); y = np.r_[y, np.ones(len(extra), int)]
    clf = HistGradientBoostingClassifier(random_state=0, categorical_features=[NUM.index(c) if c in NUM else len(NUM) + CAT.index(c) for c in CAT])
    clf.fit(X, y)
    p = clf.predict_proba(test_all[FEATS].astype(float))[:, 1]
    return p

def evaluate(p):
    yN = (test_all["cat"] == "normal").values
    thr = np.quantile(p[yN], 0.99)
    res = {"fpr": float(np.mean(p[yN] > thr))}
    for s in UNKNOWN:
        m = (test_all["sub"] == s).values; res[f"unknown_{s}"] = float(np.mean(p[m] > thr))
    known_att = (~yN) & (~test_all["sub"].isin(UNKNOWN)).values
    res["known_attacks"] = float(np.mean(p[known_att] > thr))
    return res

def diversity_of(Z_enc):
    sub = Z_enc[np.random.RandomState(0).choice(len(Z_enc), min(1000, len(Z_enc)), replace=False)]
    D = np.abs(sub[:, None, :] - sub[None, :, :]).mean(2)
    return float(D[np.triu_indices(len(sub), 1)].mean())

# %% [markdown]
# ## Jobs

# %%
CONDS = ["none", "cgan", "cgan_div"]
JOBS = [(c, s) for c in ["cgan_div", "cgan"] for s in SEEDS] + [("none", s) for s in SEEDS]
def job_path(j): return RUNS / f"{j[0]}_{j[1]}.json"

def do_job(j):
    p = job_path(j)
    if p.exists(): return
    cond, seed = j; t0 = time.time()
    if cond == "none":
        res = evaluate(detector_scores(train)); res["diversity"] = None
    else:
        sample = train_cgan(seed, diversity=(cond == "cgan_div"))
        syn, codec, _ = sample(10000)
        # diversity of generated DoS records (encoded)
        dos = syn[syn["cat"] == "DoS"]; Zd = codec.enc(dos) if len(dos) else np.zeros((1, codec.dim))
        res = evaluate(detector_scores(train, extra=syn)); res["diversity"] = diversity_of(Zd)
    res["seconds"] = time.time() - t0
    p.write_text(json.dumps(res)); print(f"job {j} done [{time.time()-t0:.0f}s]", flush=True)

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
print("UNKNOWN subtypes held out of training:", UNKNOWN)
print("test counts:", {s: int((test_all['sub'] == s).sum()) for s in UNKNOWN},
      "known attacks (test):", int(((test_all['cat'] != 'normal') & (~test_all['sub'].isin(UNKNOWN))).sum()),
      "normal (test):", int((test_all['cat'] == 'normal').sum()))
ms = lambda v, d=3: f"{np.mean(v):.{d}f} ± {np.std(v, ddof=1):.{d}f}"
RESULTS = {}
for c in CONDS:
    rows = [json.loads(job_path((c, s)).read_text()) for s in SEEDS]; RESULTS[c] = rows
    div = [r["diversity"] for r in rows if r["diversity"] is not None]
    print(f"{c:9s} unknown back {ms([r['unknown_back'] for r in rows])}  unknown portsweep {ms([r['unknown_portsweep'] for r in rows])}"
          f"  known attacks {ms([r['known_attacks'] for r in rows])}  FPR {ms([r['fpr'] for r in rows])}"
          + (f"  gen-diversity {ms(div)}" if div else "") + f"  time {np.mean([r['seconds'] for r in rows]):.0f}s", flush=True)
Path(OUT / "ch16_results.json").write_text(json.dumps(RESULTS, indent=1))
print("done")
