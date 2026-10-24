# %% [markdown]
# # Chapter 3 companion notebook: the original GAN and its pathologies
#
# Every number and figure in Chapter 3 is produced by this file. No external data.
#   python ch03_pathologies.py            (CPU, a few minutes)
# Requirements: torch, numpy, scipy, matplotlib.

# %%
from pathlib import Path
import json

import numpy as np
import torch
import torch.nn as nn
from scipy import integrate, stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("out"); OUT.mkdir(exist_ok=True)
SEEDS = [0, 1, 2, 3, 4]
torch.set_num_threads(2)
plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 150})
C1, C2, C3 = "#1f6fb4", "#d1495b", "#2a9d8f"
RESULTS = {}


def mlp(d_in, d_out, width=128, depth=3):
    layers, d = [], d_in
    for _ in range(depth):
        layers += [nn.Linear(d, width), nn.LeakyReLU(0.2)]
        d = width
    layers.append(nn.Linear(d, d_out))
    return nn.Sequential(*layers)


def set_seed(s):
    torch.manual_seed(s); np.random.seed(s)

# %% [markdown]
# ## Experiment 3.1 — the optimal discriminator (Section 3.1)
#
# P = N(0, 1), P_g = N(1, 1), both fixed. A discriminator is trained with the
# GAN loss; its output is compared with D*(x) = p(x) / (p(x) + p_g(x)), and the
# value V(D, G) with -log 4 + 2 JS(P || P_g).

# %%
def js_shift(theta):
    def f(x):
        a, b = stats.norm.pdf(x, 0, 1), stats.norm.pdf(x, theta, 1)
        m = 0.5 * (a + b)
        return 0.5 * a * np.log(a / m) + 0.5 * b * np.log(b / m)
    return integrate.quad(f, -12, 12 + theta, limit=400)[0]

JS1 = js_shift(1.0)
V_STAR = -np.log(4) + 2 * JS1
xs = torch.linspace(-3, 4, 701).unsqueeze(1)
d_star = (stats.norm.pdf(xs.numpy(), 0, 1) / (stats.norm.pdf(xs.numpy(), 0, 1) + stats.norm.pdf(xs.numpy(), 1, 1))).ravel()

errs, vals, curves = [], [], []
for s in SEEDS:
    set_seed(s)
    D = mlp(1, 1, width=64, depth=2)
    opt = torch.optim.Adam(D.parameters(), lr=1e-3)
    bce = nn.BCEWithLogitsLoss()
    for _ in range(4000):
        xr, xg = torch.randn(512, 1), torch.randn(512, 1) + 1.0
        loss = bce(D(xr), torch.ones(512, 1)) + bce(D(xg), torch.zeros(512, 1))
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        d = torch.sigmoid(D(xs)).numpy().ravel()
        xr, xg = torch.randn(200000, 1), torch.randn(200000, 1) + 1.0
        v = (torch.log(torch.sigmoid(D(xr))).mean() + torch.log(1 - torch.sigmoid(D(xg))).mean()).item()
    mask = (xs.numpy().ravel() >= -2) & (xs.numpy().ravel() <= 3)
    errs.append(np.abs(d - d_star)[mask].max()); vals.append(v); curves.append(d)

RESULTS["exp3_1"] = {"JS": JS1, "V_star": V_STAR, "V_mean": float(np.mean(vals)), "V_sd": float(np.std(vals, ddof=1)),
                     "maxerr_mean": float(np.mean(errs)), "maxerr_sd": float(np.std(errs, ddof=1))}
print("Experiment 3.1:", json.dumps(RESULTS["exp3_1"], indent=1))

fig, ax = plt.subplots(figsize=(5.6, 2.6))
ax.plot(xs.numpy().ravel(), d_star, color="#222222", lw=2.2, label="D*(x) = p / (p + p_g)")
for i, d in enumerate(curves):
    ax.plot(xs.numpy().ravel(), d, color=C1, lw=1, alpha=0.7, ls="--", label="trained D, 5 seeds" if i == 0 else None)
ax.set_xlabel("x"); ax.set_ylabel("D(x)"); ax.legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "fig3_1_optimal_discriminator.png"); plt.close(fig)

# %% [markdown]
# ## Experiment 3.2 — saturating and non-saturating generator losses (Section 3.2)
#
# Data P = N(0, 1). Generator G(z) = z + b with b = 4 held fixed, so the
# generated distribution sits far from the data. The discriminator is trained
# for an increasing number of steps; after each block we measure the gradient
# of each generator loss with respect to b.

# %%
CHECKS = [0, 10, 50, 100, 250, 500, 1000, 2000]
sat, nonsat, acc = {c: [] for c in CHECKS}, {c: [] for c in CHECKS}, {c: [] for c in CHECKS}
for s in SEEDS:
    set_seed(s)
    D = mlp(1, 1, width=64, depth=2)
    opt = torch.optim.Adam(D.parameters(), lr=1e-3)
    bce = nn.BCEWithLogitsLoss()
    b = torch.tensor(4.0, requires_grad=True)
    step = 0
    for c in CHECKS:
        while step < c:
            xr, xg = torch.randn(256, 1), torch.randn(256, 1) + b.detach()
            loss = bce(D(xr), torch.ones(256, 1)) + bce(D(xg), torch.zeros(256, 1))
            opt.zero_grad(); loss.backward(); opt.step(); step += 1
        z = torch.randn(20000, 1)
        logits = D(z + b)
        g_sat = torch.autograd.grad(torch.log(1 - torch.sigmoid(logits)).mean(), b, retain_graph=True)[0].abs().item()
        g_ns = torch.autograd.grad(-torch.log(torch.sigmoid(logits)).mean(), b)[0].abs().item()
        with torch.no_grad():
            a = 0.5 * ((torch.sigmoid(D(torch.randn(20000, 1))) > 0.5).float().mean()
                       + (torch.sigmoid(D(torch.randn(20000, 1) + 4.0)) < 0.5).float().mean()).item()
        sat[c].append(g_sat); nonsat[c].append(g_ns); acc[c].append(a)

RESULTS["exp3_2"] = {c: {"acc": float(np.mean(acc[c])), "sat_mean": float(np.mean(sat[c])),
                         "nonsat_mean": float(np.mean(nonsat[c]))} for c in CHECKS}
print("Experiment 3.2")
for c in CHECKS:
    print(f"  D steps {c:5d}: accuracy {np.mean(acc[c]):.3f}  |grad| saturating {np.mean(sat[c]):.2e}"
          f"  non-saturating {np.mean(nonsat[c]):.2e}")

fig, ax = plt.subplots(figsize=(5.6, 2.8))
cs = np.array(CHECKS[1:])
for data, col, lab in [(sat, C2, "saturating: log(1 − D(G(z)))"), (nonsat, C1, "non-saturating: −log D(G(z))")]:
    m = np.array([np.mean(data[c]) for c in CHECKS[1:]])
    lo = np.array([np.min(data[c]) for c in CHECKS[1:]]); hi = np.array([np.max(data[c]) for c in CHECKS[1:]])
    ax.plot(cs, m, color=col, lw=2, marker="o", ms=3, label=lab)
    ax.fill_between(cs, lo, hi, color=col, alpha=0.15)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("discriminator training steps"); ax.set_ylabel("|∂ loss / ∂b|")
ax.legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "fig3_2_saturation.png"); plt.close(fig)

# %% [markdown]
# ## Experiment 3.3 — mode collapse on a ring of eight Gaussians (Section 3.3)
#
# Data: 8 Gaussians, std 0.05, centres on a circle of radius 2.
# A sample is "high quality" if it lies within 3 std of a centre; a mode is
# "covered" if at least 1% of 5000 samples fall within 3 std of it.
# Non-saturating GAN loss, Adam (lr 1e-4, betas 0.5/0.999), batch 256.

# %%
K, R, STD = 8, 2.0, 0.05
CENTRES = np.array([[R * np.cos(2 * np.pi * k / K), R * np.sin(2 * np.pi * k / K)] for k in range(K)])

def sample_ring(n, weights=None, rng=None):
    rng = rng or np.random
    w = np.full(K, 1 / K) if weights is None else np.asarray(weights)
    idx = rng.choice(K, size=n, p=w)
    return torch.tensor(CENTRES[idx] + STD * rng.randn(n, 2), dtype=torch.float32)

def mode_stats(x):
    d = np.linalg.norm(x[:, None, :] - CENTRES[None], axis=2)
    near = d.min(axis=1) < 3 * STD
    nearest = d.argmin(axis=1)
    frac = np.array([np.mean(near & (nearest == k)) for k in range(K)])
    return near.mean(), int((frac >= 0.01).sum()), frac

def train_gan(seed, steps, weights=None, snapshots=()):
    set_seed(seed); rng = np.random.RandomState(seed)
    G, D = mlp(2, 2), mlp(2, 1)
    og = torch.optim.Adam(G.parameters(), lr=1e-4, betas=(0.5, 0.999))
    od = torch.optim.Adam(D.parameters(), lr=1e-4, betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss()
    snaps, trace = {}, []
    for t in range(1, steps + 1):
        xr = sample_ring(256, weights, rng)
        xg = G(torch.randn(256, 2))
        ld = bce(D(xr), torch.ones(256, 1)) + bce(D(xg.detach()), torch.zeros(256, 1))
        od.zero_grad(); ld.backward(); od.step()
        lg = bce(D(G(torch.randn(256, 2))), torch.ones(256, 1))
        og.zero_grad(); lg.backward(); og.step()
        if t in snapshots or t % 500 == 0:
            with torch.no_grad():
                x = G(torch.randn(5000, 2)).numpy()
            if t in snapshots: snaps[t] = x
            if t % 500 == 0: trace.append((t, *mode_stats(x)[:2], mode_stats(x)[2].argmax()))
    with torch.no_grad():
        x = G(torch.randn(5000, 2)).numpy()
    return x, snaps, trace

STEPS = 20000
SNAPS = (2000, 5000, 10000, 20000)
final = {}
for s in SEEDS:
    x, snaps, trace = train_gan(s, STEPS, snapshots=SNAPS if s == 0 else ())
    q, cov, frac = mode_stats(x)
    final[s] = {"quality": q, "modes": cov, "frac": frac.round(4).tolist(), "trace": trace}
    if s == 0:
        snaps0, trace0 = snaps, trace
    print(f"Experiment 3.3 seed {s}: modes covered {cov}/8, high-quality {100*q:.1f}%")

RESULTS["exp3_3"] = {str(s): {k: v for k, v in final[s].items() if k != "trace"} for s in SEEDS}
RESULTS["exp3_3"]["seed0_trace"] = [list(map(float, r)) for r in trace0]

fig, axes = plt.subplots(1, len(SNAPS) + 1, figsize=(10.5, 2.3))
axes[0].scatter(*sample_ring(2000).numpy().T, s=1, color="#888888"); axes[0].set_title("data", fontsize=9)
for ax, t in zip(axes[1:], SNAPS):
    ax.scatter(*snaps0[t].T, s=1, color=C1); ax.set_title(f"step {t}", fontsize=9)
for ax in axes:
    ax.set_xlim(-2.8, 2.8); ax.set_ylim(-2.8, 2.8); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
fig.tight_layout(); fig.savefig(OUT / "fig3_3_ring.png"); plt.close(fig)

# %% [markdown]
# ## Experiment 3.4 — a rare mode (Section 3.4)
#
# Same ring, but one mode carries 1% of the data and the other seven share 99%.
# This mimics a rare attack class. We record the share of generated samples
# that land on the rare mode.

# %%
W_RARE = np.array([0.01] + [0.99 / 7] * 7)
rare = []
for s in SEEDS:
    x, _, _ = train_gan(s, STEPS, weights=W_RARE)
    q, cov, frac = mode_stats(x)
    rare.append({"quality": q, "modes": cov, "rare_share": float(frac[0])})
    print(f"Experiment 3.4 seed {s}: modes covered {cov}/8, rare-mode share {100*frac[0]:.2f}% (data: 1.00%)")
RESULTS["exp3_4"] = rare

Path(OUT / "ch03_results.json").write_text(json.dumps(RESULTS, indent=1, default=float))
print("done")
