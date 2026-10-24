# %% [markdown]
# # Chapter 9 companion notebook: GANs for time series
#
# Every number and figure in Chapter 9 is produced by this file. No external data:
# the per-host traffic series are simulated (Section 9.5 describes the simulation).
#   python ch09_timeseries.py worker 0 2 ; python ch09_timeseries.py worker 1 2
#   python ch09_timeseries.py report
# Requirements: torch, numpy, scipy, matplotlib.

# %%
from pathlib import Path
import json, sys, time, math

import numpy as np
import torch
import torch.nn as nn
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("out"); OUT.mkdir(exist_ok=True)
RUNS = OUT / "runs"; RUNS.mkdir(exist_ok=True)
SEEDS = [0, 1, 2, 3, 4]
MODE = sys.argv[1] if len(sys.argv) > 1 else "all"
torch.set_num_threads(1)
plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 150})
RESULTS = {}

# %% [markdown]
# ## Simulated per-host traffic
#
# A sequence is 48 consecutive five-minute bins (four hours) of one host, with
# three channels, all on the scale log(1 + count):
#   flows : Poisson count with log-mean log(20 * (1 + 0.6 sin(2 pi t/48 + phase))) + e_t,
#           e_t an AR(1) process with coefficient 0.7 and innovation sd 0.3
#   bytes : flows channel + log(500) + N(0, 0.2^2), on the scale log(1 + bytes)
#   ports : distinct destination ports, Poisson(2 + 0.1 * flow count); for a
#           scanning host (probability 0.2) a burst of 6 to 12 consecutive bins in
#           which Poisson(200) further ports are added.

# %%
T, C = 48, 3
CH = ["flows", "bytes", "ports"]

def simulate(n, rng):
    t = np.arange(T)
    phase = rng.uniform(0, 2 * np.pi, size=(n, 1))
    base = 1 + 0.6 * np.sin(2 * np.pi * t / T + phase)
    e = np.zeros((n, T)); eps = rng.normal(0, 0.3, size=(n, T))
    for k in range(T):
        e[:, k] = (0.7 * e[:, k - 1] if k else 0) + eps[:, k]
    flows = rng.poisson(np.exp(np.log(20 * base) + e))
    x1 = np.log1p(flows)
    x2 = np.log1p(flows * 500 * np.exp(rng.normal(0, 0.2, size=(n, T))))
    ports = rng.poisson(2 + 0.1 * flows).astype(float)
    scanner = rng.rand(n) < 0.2
    for i in np.where(scanner)[0]:
        L = rng.randint(6, 13); s = rng.randint(0, T - L + 1)
        ports[i, s:s + L] += rng.poisson(200, size=L)
    x3 = np.log1p(ports)
    return np.stack([x1, x2, x3], axis=2).astype(np.float32), scanner

TRAIN, SCAN_TR = simulate(3000, np.random.RandomState(100))
TEST, SCAN_TE = simulate(1000, np.random.RandomState(200))
LO, HI = TRAIN.min(axis=(0, 1)), TRAIN.max(axis=(0, 1))
scale = lambda x: (x - LO) / (HI - LO)            # all models work on [0, 1] per channel
unscale = lambda x: x * (HI - LO) + LO

# %% [markdown]
# ## Generators
#
# Budget: 2,000 generator updates, batch 128, Adam 1e-3 (betas 0.5, 0.999) for
# the recurrent models and 1e-4 for the transformer, hidden size 32, noise
# dimension 4 per time step. The GANs use the non-saturating loss without a
# gradient penalty: the R1 penalty needs second derivatives, which are slow
# through recurrent layers and not implemented for attention on CPU in PyTorch.
#
# * rgan     : LSTM generator, LSTM discriminator scoring every step (Esteban et al.)
# * timegan  : embedder, recovery, generator, supervisor and discriminator (GRUs),
#              trained in three phases as in Yoon et al.: 1,000 autoencoder steps,
#              1,000 supervised steps, 2,000 joint steps (each with two generator updates)
# * hybrid   : LSTM generator; discriminator = two 1-D convolutions (kernel 5) then an LSTM
# * transf   : transformer encoder generator and discriminator (2 layers, 4 heads, width 32)
# * iid      : baseline — each time step drawn independently from the real records at that step
# * ar       : baseline — Gaussian VAR(1) fitted to the training series (per-step means)

# %%
H, ZD, B = 32, 4, 128

class RNNGen(nn.Module):
    def __init__(self, cell=nn.LSTM):
        super().__init__(); self.rnn = cell(ZD, H, num_layers=2, batch_first=True); self.out = nn.Linear(H, C)
    def forward(self, z): return torch.sigmoid(self.out(self.rnn(z)[0]))

class RNNDis(nn.Module):
    def __init__(self):
        super().__init__(); self.rnn = nn.LSTM(C, H, num_layers=2, batch_first=True); self.out = nn.Linear(H, 1)
    def forward(self, x): return self.out(self.rnn(x)[0]).mean(1)

class HybridDis(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(C, H, 5, padding=2), nn.LeakyReLU(0.2), nn.Conv1d(H, H, 5, padding=2), nn.LeakyReLU(0.2))
        self.rnn = nn.LSTM(H, H, batch_first=True); self.out = nn.Linear(H, 1)
    def forward(self, x):
        h = self.conv(x.transpose(1, 2)).transpose(1, 2)
        return self.out(self.rnn(h)[0][:, -1])

class TransBlock(nn.Module):
    def __init__(self, d_in, d_out, final=None):
        super().__init__()
        self.inp = nn.Linear(d_in, H); self.pos = nn.Parameter(torch.randn(1, T, H) * 0.02)
        layer = nn.TransformerEncoderLayer(H, 4, dim_feedforward=64, dropout=0.0, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, 2); self.out = nn.Linear(H, d_out); self.final = final
    def forward(self, x):
        h = self.out(self.enc(self.inp(x) + self.pos))
        return torch.sigmoid(h) if self.final == "sigmoid" else h.mean(1)

def r1(D, x):
    x = x.clone().requires_grad_(True)
    g = torch.autograd.grad(D(x).sum(), x, create_graph=True)[0]
    return g.pow(2).sum((1, 2)).mean()

def train_gan(G, D, data, steps, lr):
    og = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999)); od = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss(); ones, zeros = torch.ones(B, 1), torch.zeros(B, 1)
    for _ in range(steps):
        xr = data[torch.randint(0, len(data), (B,))]
        with torch.no_grad(): xg = G(torch.randn(B, T, ZD))
        ld = bce(D(xr), ones) + bce(D(xg), zeros)
        od.zero_grad(); ld.backward(); od.step()
        lg = bce(D(G(torch.randn(B, T, ZD))), ones)
        og.zero_grad(); lg.backward(); og.step()
    return lambda n: G(torch.randn(n, T, ZD))

def train_timegan(data, n1=1000, n2=1000, n3=2000, lr=1e-3):
    gru = lambda i, o: nn.GRU(i, o, num_layers=2, batch_first=True)
    class Net(nn.Module):
        def __init__(self, i, o, act=True):
            super().__init__(); self.r = gru(i, H); self.l = nn.Linear(H, o); self.act = act
        def forward(self, x):
            h = self.l(self.r(x)[0]); return torch.sigmoid(h) if self.act else h
    E, Rc, Gn, S, D = Net(C, H), Net(H, C), Net(ZD, H), Net(H, H), Net(H, 1, act=False)
    opt = lambda *m: torch.optim.Adam([p for x in m for p in x.parameters()], lr=lr)
    oer, os_, og, oe, od = opt(E, Rc), opt(S), opt(Gn, S), opt(E, Rc), opt(D)
    mse = nn.MSELoss(); bce = nn.BCEWithLogitsLoss()
    batch = lambda: data[torch.randint(0, len(data), (B,))]
    for _ in range(n1):                                        # phase 1: autoencoder
        x = batch(); l = mse(Rc(E(x)), x); oer.zero_grad(); l.backward(); oer.step()
    for _ in range(n2):                                        # phase 2: supervisor on real embeddings
        h = E(batch()).detach(); l = mse(S(h)[:, :-1], h[:, 1:]); os_.zero_grad(); l.backward(); os_.step()
    for _ in range(n3):                                        # phase 3: joint
        for _ in range(2):                                     # generator and supervisor, twice per step
            x = batch(); z = torch.randn(B, T, ZD); h = E(x)
            e_hat = Gn(z); h_hat = S(e_hat); x_hat = Rc(h_hat)
            yf, yfe = D(h_hat), D(e_hat)
            l_u = bce(yf, torch.ones_like(yf)) + bce(yfe, torch.ones_like(yfe))
            l_s = mse(S(h)[:, :-1], h[:, 1:])
            l_m = (torch.abs(x_hat.std(0) - x.std(0)).mean() + torch.abs(x_hat.mean(0) - x.mean(0)).mean())
            lg = l_u + 100 * torch.sqrt(l_s) + 100 * l_m
            og.zero_grad(); lg.backward(); og.step()
            h = E(x); l_r = mse(Rc(h), x); l_s = mse(S(h)[:, :-1], h[:, 1:])
            le = 10 * torch.sqrt(l_r) + 0.1 * l_s
            oe.zero_grad(); le.backward(); oe.step()
        x = batch(); z = torch.randn(B, T, ZD)
        with torch.no_grad(): h = E(x); e_hat = Gn(z); h_hat = S(e_hat)
        yr, yf, yfe = D(h), D(h_hat), D(e_hat)
        ld = bce(yr, torch.ones_like(yr)) + bce(yf, torch.zeros_like(yf)) + bce(yfe, torch.zeros_like(yfe))
        if ld.item() > 0.15:                                   # as in the reference implementation
            od.zero_grad(); ld.backward(); od.step()
    return lambda n: Rc(S(Gn(torch.randn(n, T, ZD))))

def sample_iid(n, rng):
    idx = rng.randint(0, len(TRAIN), size=(n, T))
    return TRAIN[idx, np.arange(T)[None, :], :]

def sample_ar(n, rng):
    mu = TRAIN.mean(0)                                         # per-step mean (T, C)
    r = TRAIN - mu
    X0, X1 = r[:, :-1].reshape(-1, C), r[:, 1:].reshape(-1, C)
    A = np.linalg.lstsq(X0, X1, rcond=None)[0]
    res = X1 - X0 @ A; Sig = np.cov(res, rowvar=False)
    out = np.zeros((n, T, C)); out[:, 0] = rng.multivariate_normal(np.zeros(C), np.cov(r[:, 0], rowvar=False), size=n)
    for k in range(1, T):
        out[:, k] = out[:, k - 1] @ A + rng.multivariate_normal(np.zeros(C), Sig, size=n)
    return (out + mu).astype(np.float32)

def generate(kind, seed, n=1000):
    torch.manual_seed(seed); rng = np.random.RandomState(seed)
    if kind == "iid": return sample_iid(n, rng)
    if kind == "ar": return sample_ar(n, rng)
    data = torch.tensor(scale(TRAIN))
    if kind == "rgan": f = train_gan(RNNGen(), RNNDis(), data, 2000, 1e-3)
    elif kind == "hybrid": f = train_gan(RNNGen(), HybridDis(), data, 2000, 1e-3)
    elif kind == "transf": f = train_gan(TransBlock(ZD, C, "sigmoid"), TransBlock(C, 1), data, 2000, 1e-4)
    elif kind == "timegan": f = train_timegan(data)
    with torch.no_grad(): return unscale(f(n).numpy())

# %% [markdown]
# ## Measurements (synthetic set of 1,000 sequences against the 1,000 test sequences)
#
# * KS: mean Kolmogorov–Smirnov statistic over the three channels, values pooled over time
# * ACF error: mean absolute difference of the autocorrelation at lags 1 to 12, averaged over
#   channels (autocorrelation computed per sequence and averaged)
# * cross-correlation error: mean absolute difference of the lag-0 correlations between channels
# * burst share: share of sequences whose ports channel exceeds log(1 + 50) for at least three
#   consecutive bins (the scanning behaviour; 20% of real hosts)
# * discriminative score: |accuracy - 0.5| of a 2-layer GRU classifier trained to separate real
#   (training) from synthetic sequences, evaluated on held-out halves (Yoon et al.)
# * predictive score: mean absolute error of a 2-layer GRU trained on the synthetic sequences to
#   predict the next step of all channels, evaluated on the real test sequences (Yoon et al.)

# %%
def acf(x, lags=12):
    x = x - x.mean(1, keepdims=True); v = (x ** 2).mean(1) + 1e-9
    return np.stack([(x[:, :-k] * x[:, k:]).mean(1) / v for k in range(1, lags + 1)], 1).mean(0)

def bursts(x):
    hi = x[:, :, 2] > np.log1p(50)
    run = np.zeros(len(x), int); best = np.zeros(len(x), int)
    for k in range(T):
        run = np.where(hi[:, k], run + 1, 0); best = np.maximum(best, run)
    return float(np.mean(best >= 3))

class GRUNet(nn.Module):
    def __init__(self, out):
        super().__init__(); self.r = nn.GRU(C, 16, num_layers=2, batch_first=True); self.l = nn.Linear(16, out)
    def forward(self, x): return self.l(self.r(x)[0])

def disc_score(real, syn, seed):
    torch.manual_seed(seed); rng = np.random.RandomState(seed)
    n = min(len(real), len(syn)); r = scale(real[rng.permutation(len(real))[:n]]); s = scale(syn[:n])
    X = torch.tensor(np.concatenate([r, s])); y = torch.tensor(np.r_[np.ones(n), np.zeros(n)], dtype=torch.float32)
    idx = rng.permutation(2 * n); tr, te = idx[: n], idx[n:]
    m = GRUNet(1); o = torch.optim.Adam(m.parameters(), 1e-3); bce = nn.BCEWithLogitsLoss()
    for _ in range(1500):
        b = tr[rng.randint(0, len(tr), 128)]
        l = bce(m(X[b])[:, -1, 0], y[b]); o.zero_grad(); l.backward(); o.step()
    with torch.no_grad(): acc = ((m(X[te])[:, -1, 0] > 0).float() == y[te]).float().mean().item()
    return abs(acc - 0.5)

def pred_score(syn, seed):
    torch.manual_seed(seed); rng = np.random.RandomState(seed)
    X = torch.tensor(scale(syn)); Xt = torch.tensor(scale(TEST))
    m = GRUNet(C); o = torch.optim.Adam(m.parameters(), 1e-3)
    for _ in range(1500):
        b = X[rng.randint(0, len(X), 128)]
        l = (m(b[:, :-1]) - b[:, 1:]).abs().mean(); o.zero_grad(); l.backward(); o.step()
    with torch.no_grad(): return float((m(Xt[:, :-1]) - Xt[:, 1:]).abs().mean())

def measure(syn, seed):
    a_r = np.stack([acf(TEST[:, :, c]) for c in range(C)]); a_s = np.stack([acf(syn[:, :, c]) for c in range(C)])
    cr = lambda x: np.corrcoef(x.reshape(-1, C), rowvar=False)[np.triu_indices(C, 1)]
    return {"ks": float(np.mean([stats.ks_2samp(syn[:, :, c].ravel(), TEST[:, :, c].ravel()).statistic for c in range(C)])),
            "acf_err": float(np.abs(a_r - a_s).mean()),
            "acf_err_by_channel": np.abs(a_r - a_s).mean(1).tolist(),
            "xcorr_err": float(np.abs(cr(TEST) - cr(syn)).mean()),
            "burst": bursts(syn),
            "disc": disc_score(TRAIN, syn, seed),
            "pred": pred_score(syn, seed)}

# %% [markdown]
# ## Jobs

# %%
KINDS = ["iid", "ar", "rgan", "hybrid", "transf", "timegan"]
JOBS = [(k, s) for k in ["timegan", "transf", "rgan", "hybrid", "ar", "iid"] for s in SEEDS]
def job_path(j): return RUNS / f"{j[0]}_{j[1]}.json"

def do_job(j):
    p = job_path(j)
    if p.exists(): return
    t0 = time.time()
    syn = generate(*j); fit = time.time() - t0
    if not np.all(np.isfinite(syn)): raise RuntimeError(f"non-finite output for {j}")
    np.save(RUNS / f"{j[0]}_{j[1]}.npy", syn)
    m = measure(syn, j[1]); m["fit_seconds"] = fit
    p.write_text(json.dumps(m))
    print(f"job {j[0]} seed {j[1]} done [fit {fit:.0f}s, total {time.time()-t0:.0f}s]", flush=True)

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
ms = lambda a, d=3: f"{np.mean(a):.{d}f} ± {np.std(a, ddof=1):.{d}f}"
print(f"Real data: burst share train {SCAN_TR.mean():.3f} (detector on train {bursts(TRAIN):.3f}, on test {bursts(TEST):.3f})")
ref = {"ks": float(np.mean([stats.ks_2samp(TRAIN[:1000, :, c].ravel(), TEST[:, :, c].ravel()).statistic for c in range(C)]))}
a_r = np.stack([acf(TEST[:, :, c]) for c in range(C)]); a_t = np.stack([acf(TRAIN[:, :, c]) for c in range(C)])
ref["acf_err"] = float(np.abs(a_r - a_t).mean())
ref["disc"] = disc_score(TRAIN[:1500], TRAIN[1500:2500], 0)
ref["pred"] = pred_score(TRAIN, 0)
RESULTS["reference_real"] = ref
print(f"Reference (real vs real): KS {ref['ks']:.3f}  ACF err {ref['acf_err']:.3f}  disc {ref['disc']:.3f}  pred (train on real) {ref['pred']:.4f}")
for kind in KINDS:
    rows = [json.loads(job_path((kind, s)).read_text()) for s in SEEDS]
    RESULTS[kind] = rows
    print(f"{kind:8s} KS {ms([r['ks'] for r in rows])}  ACF {ms([r['acf_err'] for r in rows])} "
          f"(by channel {[round(float(np.mean([r['acf_err_by_channel'][c] for r in rows])),3) for c in range(C)]})  "
          f"xcorr {ms([r['xcorr_err'] for r in rows])}  burst {ms([r['burst'] for r in rows])}  "
          f"disc {ms([r['disc'] for r in rows])}  pred {ms([r['pred'] for r in rows],4)}  fit {ms([r['fit_seconds'] for r in rows],0)} s", flush=True)

# Figure 9.1: example sequences (ports channel) and mean ACF of the flows channel
fig, axes = plt.subplots(2, len(KINDS) + 1, figsize=(14, 3.8))
rng = np.random.RandomState(3)
for j, (name, x) in enumerate([("real", TEST)] + [(k, np.load(RUNS / f"{k}_0.npy")) for k in KINDS]):
    for i in rng.choice(len(x), 6, replace=False): axes[0, j].plot(x[i, :, 2], lw=0.8)
    axes[0, j].set_title(name, fontsize=9); axes[0, j].set_ylim(0, 6.5); axes[0, j].set_xticks([])
    axes[1, j].plot(range(1, 13), acf(TEST[:, :, 0]), color="#999999", lw=2)
    axes[1, j].plot(range(1, 13), acf(x[:, :, 0]), color="#1f6fb4", lw=1.2); axes[1, j].set_ylim(-0.2, 1)
axes[0, 0].set_ylabel("ports\n(log scale)", fontsize=8); axes[1, 0].set_ylabel("ACF of flows", fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "fig9_1_series.png"); plt.close(fig)

Path(OUT / "ch09_results.json").write_text(json.dumps(RESULTS, indent=1, default=float))
print("done")
