# %% [markdown]
# # Chapter 11 companion notebook: GANs against other generative families
#
# Every number in Chapter 11 is produced by this file. It reuses the data loading,
# decoding and measurements of the Chapter 7 notebook (ch07_tabular.py), so that
# the two chapters are measured identically. Set CH07 below.
#   python ch11_families.py worker 0 2 ; python ch11_families.py worker 1 2
#   python ch11_families.py report

# %%
from pathlib import Path
import json, sys, time, warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import QuantileTransformer

warnings.filterwarnings("ignore")
CH07 = Path("/home/claude/ch07/ch07_tabular.py")
_mode = sys.argv[1] if len(sys.argv) > 1 else "all"
_argv = list(sys.argv)
sys.argv = ["ch07_tabular.py", "worker", "99", "100"]
_src = CH07.read_text()
exec(_src[:_src.index('if MODE in ("all", "worker")')])        # REAL, TEST, measure, postprocess, fit_sample_copula, ...
sys.argv = _argv
MODE = _mode
OUT = Path("out"); OUT.mkdir(exist_ok=True)
RUNS = OUT / "runs"; RUNS.mkdir(exist_ok=True)
BUDGET = 600          # seconds of training on one CPU thread for every trainable model
torch.set_num_threads(1)
RESULTS = {}

# %% [markdown]
# ## Families
#
# * smote    : SMOTE-NC applied class by class to produce a whole table: a record of the
#              class is drawn, one of its 5 nearest neighbours of the same class (numeric
#              columns standardised, categorical mismatch counted as distance 1) is drawn,
#              numeric columns are interpolated at a uniform random point and each
#              discrete column takes the value of the record or of the neighbour at random.
# * copula   : the Gaussian copula of Chapter 7 (no training).
# * tvae     : TVAE (ctgan 0.12.1), epochs set so that training takes about BUDGET seconds.
# * ctgan    : CTGAN (ctgan 0.12.1), same budget.
# * ddpm     : Gaussian denoising diffusion on an encoded table — continuous columns by a
#              Gaussian quantile transform (as TabDDPM), discrete columns one-hot — with a
#              3 x 256 MLP denoiser, 1,000 noise levels with a linear schedule, trained for
#              BUDGET seconds; discrete columns decoded by argmax. A simplification of TabDDPM,
#              which uses multinomial diffusion for discrete columns.
# * flow     : RealNVP with 8 affine coupling layers (MLP 2 x 128) on the same encoding with
#              uniform dequantisation of the one-hot blocks, trained for BUDGET seconds.

# %%
def smote_table(df, n, seed):
    rng = np.random.RandomState(seed)
    num = [c for c in df.columns if c not in DISCRETE]
    out = []
    for cls, part in df.groupby("category"):
        m = int(round(n * len(part) / len(df)))
        if m == 0: continue
        Xn = part[num].values.astype(float); mu, sd = Xn.mean(0), Xn.std(0) + 1e-9
        Z = np.hstack([(Xn - mu) / sd, pd.get_dummies(part[[c for c in DISCRETE if c != "category"]].astype(str)).values.astype(float) / np.sqrt(2)])
        k = min(5, len(part) - 1)
        if k < 1:
            out.append(part.sample(m, replace=True, random_state=seed)); continue
        nb = NearestNeighbors(n_neighbors=k + 1).fit(Z).kneighbors(Z, return_distance=False)[:, 1:]
        a = rng.randint(0, len(part), m); b = nb[a, rng.randint(0, k, m)]
        lam = rng.rand(m, 1)
        rec = {}
        for j, c in enumerate(num): rec[c] = Xn[a, j] + lam[:, 0] * (Xn[b, j] - Xn[a, j])
        pick = rng.rand(m) < 0.5
        for c in DISCRETE:
            v = part[c].values; rec[c] = np.where(pick, v[a], v[b])
        out.append(pd.DataFrame(rec)[df.columns])
    return pd.concat(out, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)

class TableCodec:
    """Continuous columns: Gaussian quantile transform; discrete columns: one-hot."""
    def __init__(self, df, seed):
        self.q = QuantileTransformer(n_quantiles=1000, output_distribution="normal", random_state=seed)
        self.q.fit(df[CONTINUOUS].values.astype(float))
        self.cats = {c: sorted(df[c].astype(str).unique()) for c in DISCRETE}
        self.dim = len(CONTINUOUS) + sum(len(v) for v in self.cats.values())
    def encode(self, df):
        parts = [self.q.transform(df[CONTINUOUS].values.astype(float))]
        for c, vals in self.cats.items():
            idx = pd.Categorical(df[c].astype(str), categories=vals).codes
            parts.append(np.eye(len(vals))[idx])
        return np.hstack(parts).astype(np.float32)
    def decode(self, X):
        k = len(CONTINUOUS); out = pd.DataFrame(self.q.inverse_transform(np.clip(X[:, :k], -5.2, 5.2)), columns=CONTINUOUS)
        for c, vals in self.cats.items():
            out[c] = np.array(vals)[X[:, k:k + len(vals)].argmax(1)]; k += len(vals)
        for c in SMALL_INT: out[c] = out[c].astype(int)
        return out[REAL.columns]

def mlp(d_in, d_out, w, depth):
    layers, d = [], d_in
    for _ in range(depth): layers += [nn.Linear(d, w), nn.SiLU()]; d = w
    return nn.Sequential(*layers, nn.Linear(d, d_out))

def fit_sample_ddpm(df, n, seed):
    torch.manual_seed(seed); codec = TableCodec(df, seed)
    X = torch.tensor(codec.encode(df)); D = X.shape[1]; Tn = 1000
    beta = torch.linspace(1e-4, 0.02, Tn); alpha = 1 - beta; abar = torch.cumprod(alpha, 0)
    net = mlp(D + 1, D, 256, 3); opt = torch.optim.Adam(net.parameters(), 1e-3)
    t0, steps = time.time(), 0
    while time.time() - t0 < BUDGET:
        x0 = X[torch.randint(0, len(X), (256,))]; t = torch.randint(0, Tn, (256,))
        e = torch.randn_like(x0); ab = abar[t][:, None]
        xt = ab.sqrt() * x0 + (1 - ab).sqrt() * e
        loss = ((net(torch.cat([xt, (t[:, None] / Tn)], 1)) - e) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step(); steps += 1
    with torch.no_grad():
        x = torch.randn(n, D)
        for k in reversed(range(Tn)):
            tt = torch.full((n, 1), k / Tn)
            eps = net(torch.cat([x, tt], 1))
            x = (x - beta[k] / (1 - abar[k]).sqrt() * eps) / alpha[k].sqrt()
            if k > 0: x = x + beta[k].sqrt() * torch.randn_like(x)
    return codec.decode(x.numpy()), {"train_steps": steps}

class Coupling(nn.Module):
    def __init__(self, D, mask):
        super().__init__(); self.register_buffer("m", mask); self.net = mlp(D, 2 * D, 128, 2)
    def forward(self, x):
        h = self.net(x * self.m); s, t = h.chunk(2, 1); s = torch.tanh(s) * (1 - self.m); t = t * (1 - self.m)
        return x * self.m + (1 - self.m) * (x * torch.exp(s) + t), s.sum(1)
    def inverse(self, y):
        h = self.net(y * self.m); s, t = h.chunk(2, 1); s = torch.tanh(s) * (1 - self.m); t = t * (1 - self.m)
        return y * self.m + (1 - self.m) * ((y - t) * torch.exp(-s))

def fit_sample_flow(df, n, seed):
    torch.manual_seed(seed); rng = np.random.RandomState(seed); codec = TableCodec(df, seed)
    X = codec.encode(df); k = len(CONTINUOUS)
    X = torch.tensor(X); D = X.shape[1]
    masks = [torch.tensor((np.arange(D) + i) % 2, dtype=torch.float32) if i % 2 == 0 else
             torch.tensor(rng.permutation(np.r_[np.ones(D // 2), np.zeros(D - D // 2)]), dtype=torch.float32) for i in range(8)]
    layers = nn.ModuleList(Coupling(D, m) for m in masks); opt = torch.optim.Adam(layers.parameters(), 1e-3)
    t0, steps = time.time(), 0
    while time.time() - t0 < BUDGET:
        x = X[torch.randint(0, len(X), (256,))].clone()
        x[:, k:] = x[:, k:] + 0.5 * torch.rand_like(x[:, k:])          # dequantise the one-hot blocks
        ld = 0
        for L in layers: x, s = L(x); ld = ld + s
        loss = (0.5 * (x ** 2).sum(1) - ld).mean()
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(layers.parameters(), 10.0); opt.step(); steps += 1
    with torch.no_grad():
        z = torch.randn(n, D)
        for L in reversed(layers): z = L.inverse(z)
    return codec.decode(z.numpy()), {"train_steps": steps}

def fit_sample_ctgan_budget(df, n, seed, kind):
    from ctgan import CTGAN, TVAE
    torch.manual_seed(seed); np.random.seed(seed)
    epochs = {"ctgan": 90, "tvae": 230}[kind]      # set from a timing run so that fitting takes about BUDGET s
    m = (CTGAN if kind == "ctgan" else TVAE)(epochs=epochs, enable_gpu=False); m.set_random_state(seed)
    m.fit(df, discrete_columns=DISCRETE)
    return m.sample(n), {"epochs": epochs}

def generate11(kind, seed):
    n = len(REAL)
    if kind == "smote": return smote_table(REAL, n, seed), {}
    if kind == "copula": return fit_sample_copula(REAL, n, seed), {}
    if kind in ("ctgan", "tvae"): return fit_sample_ctgan_budget(REAL, n, seed, kind)
    if kind == "ddpm": return fit_sample_ddpm(REAL, n, seed)
    if kind == "flow": return fit_sample_flow(REAL, n, seed)

# %% [markdown]
# ## Jobs

# %%
FAMILIES = ["smote", "copula", "tvae", "ctgan", "ddpm", "flow"]
JOBS11 = [(k, s) for k in ["ctgan", "ddpm", "flow", "tvae", "smote", "copula"] for s in SEEDS]
def job_path11(j): return RUNS / f"{j[0]}_{j[1]}.json"

def do_job11(j):
    p = job_path11(j)
    if p.exists(): return
    t0 = time.time()
    syn, info = generate11(*j); fit = time.time() - t0
    syn.to_csv(RUNS / f"{j[0]}_{j[1]}.csv.gz", index=False)
    m = measure(postprocess(syn), j[1]); m.update(info); m["fit_seconds"] = fit
    p.write_text(json.dumps(m))
    print(f"job {j[0]} seed {j[1]} done [fit {fit:.0f}s, total {time.time()-t0:.0f}s]", flush=True)

if MODE in ("all", "worker"):
    k, n = (int(sys.argv[2]), int(sys.argv[3])) if MODE == "worker" else (0, 1)
    for i, j in enumerate(JOBS11):
        if i % n == k: do_job11(j)
    if MODE == "worker": sys.exit(0)

# %% [markdown]
# ## Report

# %%
missing = [j for j in JOBS11 if not job_path11(j).exists()]
if missing: raise SystemExit(f"{len(missing)} jobs not finished")
ms = lambda a, d=3: f"{np.nanmean(a):.{d}f} ± {np.nanstd(a, ddof=1):.{d}f}"
# Distance to closest record (DCR): for 2,000 synthetic records, the distance to the nearest
# training record in the encoded space (quantile-normal continuous columns, one-hot discrete
# columns). Reference: the same distance for 2,000 real records not used for training (KDDTest+).
_codec = TableCodec(REAL, 0); _nn = NearestNeighbors(n_neighbors=1).fit(_codec.encode(REAL))
def dcr(df, seed):
    sub = df.sample(min(2000, len(df)), random_state=seed)
    return _nn.kneighbors(_codec.encode(sub), return_distance=True)[0].ravel()
_ref = dcr(TEST, 0); RESULTS["dcr_holdout"] = {"median": float(np.median(_ref)), "share_exact": float(np.mean(_ref < 1e-6))}
print(f"DCR of held-out real records: median {np.median(_ref):.3f}, share at distance 0 {np.mean(_ref < 1e-6):.3f}")
for kind in FAMILIES:
    rows = [json.loads(job_path11((kind, s)).read_text()) for s in SEEDS]
    for s, r in zip(SEEDS, rows):
        d = dcr(postprocess(pd.read_csv(RUNS / f"{kind}_{s}.csv.gz")), s)
        r["dcr_median"] = float(np.median(d)); r["dcr_exact"] = float(np.mean(d < 1e-6))
    RESULTS[kind] = rows
    extra = ""
    if "train_steps" in rows[0]: extra = f"  steps {ms([r['train_steps'] for r in rows],0)}"
    if "epochs" in rows[0]: extra = f"  epochs {rows[0]['epochs']}"
    print(f"{kind:7s} zeros {ms([r['zero_share'] for r in rows])}  KS {ms([r['ks'] for r in rows])}  TVD {ms([r['tvd'] for r in rows])}  assoc {ms([r['assoc_err'] for r in rows])}"
          f"  invalid combo {ms([100*r['invalid_combo'] for r in rows],2)}%"
          f"  | macro F1 {ms([r['macro_f1'] for r in rows])}  " + "  ".join(f"rec {c} {ms([r['recall'][c] for r in rows])}" for c in CLASSES)
          + "  | shares " + "  ".join(f"{c} {ms([100*r['shares'][c] for r in rows],3)}" for c in ("R2L", "U2R"))
          + f"  | DCR median {ms([r['dcr_median'] for r in rows])}  exact copies {ms([100*r['dcr_exact'] for r in rows],2)}%"
          + f"  | fit {ms([r['fit_seconds'] for r in rows],0)} s{extra}", flush=True)
Path(OUT / "ch11_results.json").write_text(json.dumps(RESULTS, indent=1, default=float))
print("done")
