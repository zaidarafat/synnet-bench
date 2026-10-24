# %% [markdown]
# # Chapter 4 companion notebook: stabilising GAN training
#
# Every number and figure in Chapter 4 is produced by this file. No external data.
#   python ch04_stabilisation.py
# Requirements: torch, numpy, scipy, matplotlib.

# %%
from pathlib import Path
import json, time

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("out"); OUT.mkdir(exist_ok=True)
SEEDS = [0, 1, 2, 3, 4]
import sys
MODE = sys.argv[1] if len(sys.argv) > 1 else "all"   # "all", "worker K N", or "report"
torch.set_num_threads(1)  # one thread: identical results in every mode
RUNS = OUT / "runs"; RUNS.mkdir(exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 150})
RESULTS = {}

def set_seed(s):
    torch.manual_seed(s); np.random.seed(s)

def mlp(d_in, d_out, width=128, depth=3, sn=False):
    wrap = spectral_norm if sn else (lambda m: m)
    layers, d = [], d_in
    for _ in range(depth):
        layers += [wrap(nn.Linear(d, width)), nn.LeakyReLU(0.2)]
        d = width
    layers.append(wrap(nn.Linear(d, d_out)))
    return nn.Sequential(*layers)

# %% [markdown]
# ## Ring of eight Gaussians (as in Chapter 3)

# %%
K, R, STD = 8, 2.0, 0.05
CENTRES = np.array([[R * np.cos(2 * np.pi * k / K), R * np.sin(2 * np.pi * k / K)] for k in range(K)])
W_RARE = np.array([0.01] + [0.99 / 7] * 7)

def sample_ring(n, weights, rng):
    w = np.full(K, 1 / K) if weights is None else weights
    idx = rng.choice(K, size=n, p=w)
    return torch.tensor(CENTRES[idx] + STD * rng.randn(n, 2), dtype=torch.float32)

def mode_stats(x):
    d = np.linalg.norm(x[:, None, :] - CENTRES[None], axis=2)
    near = d.min(axis=1) < 3 * STD
    nearest = d.argmin(axis=1)
    frac = np.array([np.mean(near & (nearest == k)) for k in range(K)])
    return float(near.mean()), int((frac >= 0.01).sum()), frac

# %% [markdown]
# ## Six training variants
#
# All variants: generator and critic/discriminator are MLPs (3 x 128), batch 256,
# 10,000 generator updates. n_critic = discriminator updates per generator update.
#
# | variant    | loss                     | constraint / regulariser          | optimiser            | n_critic |
# |------------|--------------------------|-----------------------------------|----------------------|----------|
# | ns         | non-saturating BCE       | none                              | Adam 1e-4 (0.5, .999) | 1 |
# | ns_r1      | non-saturating BCE       | R1 penalty, gamma = 10            | Adam 1e-4 (0.5, .999) | 1 |
# | lsgan      | least squares (0/1 targets) | none                           | Adam 1e-4 (0.5, .999) | 1 |
# | hinge_sn   | hinge                    | spectral normalisation in D       | Adam 1e-4 (0.0, .9)   | 1 |
# | wgan_clip  | Wasserstein              | weight clipping at 0.01           | RMSprop 5e-5          | 5 |
# | wgan_gp    | Wasserstein              | gradient penalty, lambda = 10     | Adam 1e-4 (0.0, .9)   | 5 |

# %%
VARIANTS = ["ns", "ns_r1", "lsgan", "hinge_sn", "wgan_clip", "wgan_gp"]
G_STEPS = 10000

def grad_penalty(D, xr, xg):
    a = torch.rand(xr.size(0), 1)
    xi = (a * xr + (1 - a) * xg).requires_grad_(True)
    g = torch.autograd.grad(D(xi).sum(), xi, create_graph=True)[0]
    return ((g.norm(dim=1) - 1) ** 2).mean()

def r1_penalty(D, xr):
    xr = xr.clone().requires_grad_(True)
    g = torch.autograd.grad(D(xr).sum(), xr, create_graph=True)[0]
    return (g.pow(2).sum(dim=1)).mean()

def train(variant, seed, weights):
    set_seed(seed); rng = np.random.RandomState(seed)
    G = mlp(2, 2)
    D = mlp(2, 1, sn=(variant == "hinge_sn"))
    if variant == "wgan_clip":
        og = torch.optim.RMSprop(G.parameters(), lr=5e-5); od = torch.optim.RMSprop(D.parameters(), lr=5e-5)
    elif variant in ("hinge_sn", "wgan_gp"):
        og = torch.optim.Adam(G.parameters(), lr=1e-4, betas=(0.0, 0.9)); od = torch.optim.Adam(D.parameters(), lr=1e-4, betas=(0.0, 0.9))
    else:
        og = torch.optim.Adam(G.parameters(), lr=1e-4, betas=(0.5, 0.999)); od = torch.optim.Adam(D.parameters(), lr=1e-4, betas=(0.5, 0.999))
    n_critic = 5 if variant.startswith("wgan") else 1
    bce = nn.BCEWithLogitsLoss()
    B = 256
    ones, zeros = torch.ones(B, 1), torch.zeros(B, 1)
    for t in range(G_STEPS):
        for _ in range(n_critic):
            xr = sample_ring(B, weights, rng)
            with torch.no_grad():
                xg = G(torch.randn(B, 2))
            dr, dg = D(xr), D(xg)
            if variant in ("ns", "ns_r1"):
                ld = bce(dr, ones) + bce(dg, zeros)
                if variant == "ns_r1": ld = ld + 0.5 * 10.0 * r1_penalty(D, xr)
            elif variant == "lsgan":
                ld = 0.5 * ((dr - 1) ** 2).mean() + 0.5 * (dg ** 2).mean()
            elif variant == "hinge_sn":
                ld = torch.relu(1 - dr).mean() + torch.relu(1 + dg).mean()
            else:
                ld = dg.mean() - dr.mean()
                if variant == "wgan_gp": ld = ld + 10.0 * grad_penalty(D, xr, xg)
            od.zero_grad(); ld.backward(); od.step()
            if variant == "wgan_clip":
                with torch.no_grad():
                    for p in D.parameters(): p.clamp_(-0.01, 0.01)
        out = D(G(torch.randn(B, 2)))
        if variant in ("ns", "ns_r1"): lg = bce(out, ones)
        elif variant == "lsgan": lg = 0.5 * ((out - 1) ** 2).mean()
        else: lg = -out.mean()
        og.zero_grad(); lg.backward(); og.step()
    with torch.no_grad():
        return G(torch.randn(5000, 2)).numpy()

# %% [markdown]
# ## Experiment 4.2 — heavy-tailed features and numerical failure (Section 4.6)
#
# One feature drawn from a log-normal distribution, exp(N(8, 3^2)), which spans
# many orders of magnitude, like byte or packet rates in flow records.
# WGAN-GP is trained on three encodings of it:
#   raw      : values as they are
#   minmax   : (x - min) / (max - min) on the training sample
#   log_std  : z-score of log1p(x)
# We record whether any loss became non-finite, and the Kolmogorov–Smirnov
# statistic between generated and real data, computed after mapping
# generated values back to the original scale.

# %%
def heavy(n, rng): return np.exp(rng.normal(8.0, 3.0, size=n))

def run_heavy(enc, seed, loss, steps=3000):
    set_seed(seed); rng = np.random.RandomState(seed)
    train_x = heavy(50000, rng)
    if enc == "raw":
        fwd = lambda v: v; inv = lambda v: v
    elif enc == "minmax":
        lo, hi = train_x.min(), train_x.max()
        fwd = lambda v: (v - lo) / (hi - lo); inv = lambda v: v * (hi - lo) + lo
    else:
        mu, sd = np.log1p(train_x).mean(), np.log1p(train_x).std()
        fwd = lambda v: (np.log1p(v) - mu) / sd; inv = lambda v: np.expm1(v * sd + mu)
    data = torch.tensor(fwd(train_x), dtype=torch.float32).unsqueeze(1)
    G, D = mlp(1, 1, 64, 2), mlp(1, 1, 64, 2)
    betas = (0.0, 0.9) if loss == "wgan_gp" else (0.5, 0.999)
    og = torch.optim.Adam(G.parameters(), lr=1e-4, betas=betas); od = torch.optim.Adam(D.parameters(), lr=1e-4, betas=betas)
    bce = nn.BCEWithLogitsLoss(); ones, zeros = torch.ones(256, 1), torch.zeros(256, 1)
    nonfinite_at, trace = None, []
    for t in range(steps):
        for _ in range(5 if loss == "wgan_gp" else 1):
            xr = data[torch.randint(0, len(data), (256,))]
            with torch.no_grad(): xg = G(torch.randn(256, 1))
            if loss == "wgan_gp":
                ld = D(xg).mean() - D(xr).mean() + 10.0 * grad_penalty(D, xr, xg)
            else:
                ld = bce(D(xr), ones) + bce(D(xg), zeros) + 0.5 * 10.0 * r1_penalty(D, xr)
            od.zero_grad(); ld.backward(); od.step()
        out = D(G(torch.randn(256, 1)))
        lg = -out.mean() if loss == "wgan_gp" else bce(out, ones)
        og.zero_grad(); lg.backward(); og.step()
        if nonfinite_at is None and not (torch.isfinite(ld) and torch.isfinite(lg)):
            nonfinite_at = t; break
        if t % 100 == 0:
            with torch.no_grad():
                g = G(torch.randn(2000, 1)).numpy().ravel()
            trace.append([t, float(np.median(g)), float(np.median(data.numpy()))])
    with torch.no_grad():
        gen = G(torch.randn(20000, 1)).numpy().ravel()
    if nonfinite_at is not None or not np.all(np.isfinite(gen)):
        return {"nonfinite_at": nonfinite_at, "ks": None, "trace": trace}
    gen_orig = inv(gen.astype(np.float64))
    test = heavy(20000, np.random.RandomState(1000 + seed))
    ks = stats.ks_2samp(np.nan_to_num(gen_orig, nan=-1.0), test).statistic
    return {"nonfinite_at": nonfinite_at, "ks": float(ks), "trace": trace}

# %% [markdown]
# ## Job list
#
# Each (dataset, variant, seed) run and each (encoding, seed) run of
# Experiment 4.2 is one job. Results are cached in out/runs/, so the script
# can be interrupted and resumed, or split across processes:
#   python ch04_stabilisation.py worker 0 2   and   python ch04_stabilisation.py worker 1 2
#   python ch04_stabilisation.py report
# Running without arguments does everything in one process.

# %%
DATASETS = {"balanced": None, "rare": W_RARE}
ENCODINGS = ["raw", "minmax", "log_std"]
HEAVY_LOSSES = ["ns_r1", "wgan_gp"]
JOBS = [("ring", d, v, s) for d in DATASETS for v in VARIANTS for s in SEEDS] + \
       [("heavy", e, l, s) for l in HEAVY_LOSSES for e in ENCODINGS for s in SEEDS]

def job_path(j):
    kind, a, b, s = j
    return RUNS / (f"{a}_{b}_{s}.npz" if kind == "ring" else f"heavy_{a}_{b}_{s}.json")

def do_job(j):
    p = job_path(j)
    if p.exists(): return
    kind, a, b, s = j
    t0 = time.time()
    if kind == "ring":
        x = train(b, s, DATASETS[a])
        np.savez(p, x=x)
    else:
        p.write_text(json.dumps(run_heavy(a, s, b)))
    print(f"job {kind} {a} {b} seed {s} done [{time.time()-t0:.0f}s]", flush=True)

if MODE in ("all", "worker"):
    k, n = (int(sys.argv[2]), int(sys.argv[3])) if MODE == "worker" else (0, 1)
    for i, j in enumerate(JOBS):
        if i % n == k: do_job(j)
    if MODE == "worker": sys.exit(0)

# %% [markdown]
# ## Report: Table 4.2, Figure 4.1 and Experiment 4.2 numbers

# %%
missing = [j for j in JOBS if not job_path(j).exists()]
if missing: raise SystemExit(f"{len(missing)} jobs not finished")
ring = {}
for dataset in DATASETS:
    for v in VARIANTS:
        rows = []
        for s in SEEDS:
            x = np.load(job_path(("ring", dataset, v, s)))["x"]
            q, cov, frac = mode_stats(x)
            rows.append({"quality": q, "modes": cov, "rare_share": float(frac[0]),
                         "common_modes": int((frac[1:] >= 0.01).sum())})
            if s == 0: ring[(dataset, v)] = x
        RESULTS[f"{dataset}/{v}"] = rows
        m = np.array([r["modes"] for r in rows]); q = np.array([r["quality"] for r in rows]); rs = np.array([r["rare_share"] for r in rows])
        print(f"{dataset:8s} {v:10s} modes {m.mean():.1f} ± {m.std(ddof=1):.1f} (min {m.min()})  "
              f"quality {100*q.mean():.1f} ± {100*q.std(ddof=1):.1f}%  rare share {100*rs.mean():.2f} ± {100*rs.std(ddof=1):.2f}%  "
              f"per seed modes {m.tolist()}", flush=True)

fig, axes = plt.subplots(2, len(VARIANTS), figsize=(12, 4.3))
for i, ds in enumerate(["balanced", "rare"]):
    for j, v in enumerate(VARIANTS):
        ax = axes[i, j]
        ax.scatter(*CENTRES.T, s=40, facecolors="none", edgecolors="#999999", lw=0.8)
        ax.scatter(*ring[(ds, v)].T, s=0.6, color="#1f6fb4")
        if ds == "rare": ax.scatter(*CENTRES[0], s=90, facecolors="none", edgecolors="#d1495b", lw=1.4)
        ax.set_xlim(-2.8, 2.8); ax.set_ylim(-2.8, 2.8); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        if i == 0: ax.set_title({"ns": "NS", "ns_r1": "NS + R1", "lsgan": "LSGAN", "hinge_sn": "Hinge + SN", "wgan_clip": "WGAN", "wgan_gp": "WGAN-GP"}[v], fontsize=9)
    axes[i, 0].set_ylabel(ds, fontsize=9)
fig.tight_layout(); fig.savefig(OUT / "fig4_1_variants_ring.png"); plt.close(fig)

rng0 = np.random.RandomState(0); sample = heavy(50000, rng0)
print(f"\nExperiment 4.2 data: min {sample.min():.3g}, median {np.median(sample):.3g}, max {sample.max():.3g}; "
      f"after min-max, share of values below 0.001: {np.mean((sample - sample.min())/(sample.max()-sample.min()) < 1e-3):.4f}")
RESULTS["heavy_data"] = {"min": float(sample.min()), "median": float(np.median(sample)), "max": float(sample.max()),
                         "minmax_share_below_1e-3": float(np.mean((sample - sample.min())/(sample.max()-sample.min()) < 1e-3))}
for loss in HEAVY_LOSSES:
    for enc in ENCODINGS:
        rows = [json.loads(job_path(("heavy", enc, loss, s)).read_text()) for s in SEEDS]
        RESULTS[f"heavy/{loss}/{enc}"] = rows
        ks = [r["ks"] for r in rows if r["ks"] is not None]
        nf = sum(r["nonfinite_at"] is not None for r in rows)
        print(f"  {loss:8s} {enc:8s} non-finite in {nf}/5 runs; KS = "
              + (f"{np.mean(ks):.3f} ± {np.std(ks, ddof=1):.3f} (n={len(ks)})" if len(ks) > 1 else str(ks)), flush=True)

# Figure 4.2: median of generated values (encoded scale) during training, log_std encoding, seed 0
fig, ax = plt.subplots(figsize=(6.4, 2.8))
for loss, col, lab in [("ns_r1", "#1f6fb4", "NS + R1"), ("wgan_gp", "#d1495b", "WGAN-GP")]:
    tr = np.array(RESULTS[f"heavy/{loss}/log_std"][0]["trace"])
    ax.plot(tr[:, 0], tr[:, 1], color=col, lw=1.6, label=lab)
ax.axhline(tr[0, 2], color="#555555", lw=0.9, ls="--", label="median of data")
ax.set_xlabel("generator updates"); ax.set_ylabel("median of generated values\n(standardised log scale)")
ax.legend(frameon=False, fontsize=8); fig.tight_layout(); fig.savefig(OUT / "fig4_2_heavy_trace.png"); plt.close(fig)
for loss in HEAVY_LOSSES:
    tr = np.array(RESULTS[f"heavy/{loss}/log_std"][0]["trace"])
    print(f"  trace {loss} log_std seed 0: median range over updates 1000-3000: [{tr[10:,1].min():.2f}, {tr[10:,1].max():.2f}]")

Path(OUT / "ch04_results.json").write_text(json.dumps(RESULTS, indent=1, default=float))
print("done")
