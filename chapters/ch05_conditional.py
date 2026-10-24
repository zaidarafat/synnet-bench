# %% [markdown]
# # Chapter 5 companion notebook: conditional GANs
#
# Every number and figure in Chapter 5 is produced by this file. No external data.
#   python ch05_conditional.py                 # everything, one process
#   python ch05_conditional.py worker 0 2      # half of the training jobs
#   python ch05_conditional.py worker 1 2      # the other half
#   python ch05_conditional.py report          # tables and figures from cached jobs
# Requirements: torch, numpy, scipy, matplotlib.

# %%
from pathlib import Path
import json, sys, time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("out"); OUT.mkdir(exist_ok=True)
RUNS = OUT / "runs"; RUNS.mkdir(exist_ok=True)
SEEDS = [0, 1, 2, 3, 4]
MODE = sys.argv[1] if len(sys.argv) > 1 else "all"
torch.set_num_threads(1)  # one thread: identical results in every mode
plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 150})
RESULTS = {}

def set_seed(s):
    torch.manual_seed(s); np.random.seed(s)

def layers(d_in, width, depth, sn=False):
    wrap = spectral_norm if sn else (lambda m: m)
    out, d = [], d_in
    for _ in range(depth):
        out += [wrap(nn.Linear(d, width)), nn.LeakyReLU(0.2)]
        d = width
    return nn.Sequential(*out)

# %% [markdown]
# ## Generators and discriminators
#
# The generator is always conditioned the same way: the one-hot label is
# concatenated to the noise vector (Mirza and Osindero). The unconditional
# generator receives no label. Discriminators differ:
#
# | name   | discriminator input   | output                                   | loss                     |
# |--------|-----------------------|------------------------------------------|--------------------------|
# | uncond | x                     | one logit                                | non-saturating + R1      |
# | cgan   | [x, one-hot y]        | one logit                                | non-saturating + R1      |
# | acgan  | x                     | one source logit + K class logits        | non-saturating + R1 + CE |
# | proj   | x                     | psi(phi(x)) + embed(y) . phi(x)          | hinge, spectral norm     |

# %%
class G(nn.Module):
    def __init__(self, d_x, k, cond, width, depth, d_z):
        super().__init__()
        self.k, self.cond, self.d_z = k, cond, d_z
        self.body = layers(d_z + (k if cond else 0), width, depth)
        self.out = nn.Linear(width, d_x)
    def forward(self, z, y):
        h = torch.cat([z, F.one_hot(y, self.k).float()], 1) if self.cond else z
        return self.out(self.body(h))

class D(nn.Module):
    def __init__(self, d_x, k, kind, width, depth):
        super().__init__()
        self.k, self.kind = k, kind
        sn = kind == "proj"
        wrap = spectral_norm if sn else (lambda m: m)
        self.phi = layers(d_x + (k if kind == "cgan" else 0), width, depth, sn)
        self.psi = wrap(nn.Linear(width, 1))
        if kind == "acgan": self.cls = nn.Linear(width, k)
        if kind == "proj": self.embed = wrap(nn.Embedding(k, width))
    def forward(self, x, y):
        if self.kind == "cgan": x = torch.cat([x, F.one_hot(y, self.k).float()], 1)
        h = self.phi(x)
        s = self.psi(h)
        if self.kind == "proj": s = s + (self.embed(y) * h).sum(1, keepdim=True)
        return s, (self.cls(h) if self.kind == "acgan" else None)

def r1(Dn, x, y):
    x = x.clone().requires_grad_(True)
    s, _ = Dn(x, y)
    g = torch.autograd.grad(s.sum(), x, create_graph=True)[0]
    return g.pow(2).sum(1).mean()

def train(kind, seed, sample, k, d_x, steps, width, depth, d_z):
    """sample(n, rng) -> (x tensor, y tensor). Returns the trained generator."""
    set_seed(seed); rng = np.random.RandomState(seed)
    Gn = G(d_x, k, kind != "uncond", width, depth, d_z); Dn = D(d_x, k, kind, width, depth)
    betas = (0.0, 0.9) if kind == "proj" else (0.5, 0.999)
    og = torch.optim.Adam(Gn.parameters(), lr=1e-4, betas=betas); od = torch.optim.Adam(Dn.parameters(), lr=1e-4, betas=betas)
    B = 256; ones, zeros = torch.ones(B, 1), torch.zeros(B, 1)
    bce = nn.BCEWithLogitsLoss()
    for t in range(steps):
        xr, yr = sample(B, rng)
        yg = yr[torch.randperm(B)]           # labels for generated records: same class frequencies as the data
        with torch.no_grad(): xg = Gn(torch.randn(B, d_z), yg)
        sr, cr = Dn(xr, yr); sg, cg = Dn(xg, yg)
        if kind == "proj":
            ld = torch.relu(1 - sr).mean() + torch.relu(1 + sg).mean()
        else:
            ld = bce(sr, ones) + bce(sg, zeros) + 0.5 * 10.0 * r1(Dn, xr, yr)
            if kind == "acgan": ld = ld + F.cross_entropy(cr, yr) + F.cross_entropy(cg, yg)
        od.zero_grad(); ld.backward(); od.step()
        yg = yr[torch.randperm(B)]
        sg, cg = Dn(Gn(torch.randn(B, d_z), yg), yg)
        lg = -sg.mean() if kind == "proj" else bce(sg, ones)
        if kind == "acgan": lg = lg + F.cross_entropy(cg, yg)
        og.zero_grad(); lg.backward(); og.step()
    return Gn

# %% [markdown]
# ## Experiment 5.1 — conditioning and the rare mode (Section 5.6)
#
# The imbalanced ring of Experiments 3.4 and 4.1: eight Gaussians (std 0.05) on a
# circle of radius 2; mode 0 carries 1% of the data, the other seven 99/7% each.
# The label of a record is the index of its mode.

# %%
K, R, STD = 8, 2.0, 0.05
CENTRES = np.array([[R * np.cos(2 * np.pi * k / K), R * np.sin(2 * np.pi * k / K)] for k in range(K)])
W_RARE = np.array([0.01] + [0.99 / 7] * 7)

def sample_ring(n, rng):
    y = rng.choice(K, size=n, p=W_RARE)
    x = CENTRES[y] + STD * rng.randn(n, 2)
    return torch.tensor(x, dtype=torch.float32), torch.tensor(y)

def near_mode(x):
    d = np.linalg.norm(x[:, None, :] - CENTRES[None], axis=2)
    return d.argmin(1), d.min(1) < 3 * STD

def ring_eval(Gn, cond, seed):
    torch.manual_seed(10_000 + seed); rng = np.random.RandomState(10_000 + seed)
    res = {}
    with torch.no_grad():
        # (a) records drawn as the data are: labels with the data frequencies
        y = torch.tensor(rng.choice(K, size=5000, p=W_RARE))
        x = Gn(torch.randn(5000, 2), y).numpy()
        nearest, ok = near_mode(x)
        frac = np.array([np.mean(ok & (nearest == k)) for k in range(K)])
        res["quality"] = float(ok.mean()); res["rare_share"] = float(frac[0])
        res["common_modes"] = int((frac[1:] >= 0.01).sum())
        if cond:
            # (b) 1,000 records requested for each class
            acc, spread = [], []
            for k in range(K):
                xk = Gn(torch.randn(1000, 2), torch.full((1000,), k)).numpy()
                nk, okk = near_mode(xk)
                hit = okk & (nk == k)
                acc.append(float(hit.mean()))
                spread.append(float(np.sqrt(((xk[hit] - CENTRES[k]) ** 2).mean(0)).mean()) if hit.sum() > 10 else float("nan"))
            res["class_acc"] = acc; res["class_spread"] = spread
    return res, x

# %% [markdown]
# ## Experiment 5.2 — overlapping classes (Section 5.7)
#
# One feature, two classes of equal weight: class 0 ~ N(-1, 1), class 1 ~ N(1, 1).
# The classes overlap: 15.9% of class 0 lies above 0.

# %%
def sample_overlap(n, rng):
    y = rng.randint(0, 2, size=n)
    x = rng.randn(n) + np.where(y == 0, -1.0, 1.0)
    return torch.tensor(x, dtype=torch.float32).unsqueeze(1), torch.tensor(y)

def overlap_eval(Gn, seed, kind):
    torch.manual_seed(20_000 + seed)
    res = {}
    with torch.no_grad():
        for c, mu in [(0, -1.0), (1, 1.0)]:
            xc = Gn(torch.randn(20000, 4), torch.full((20000,), c)).numpy().ravel()
            res[f"ks{c}"] = float(stats.kstest(xc, stats.norm(loc=mu, scale=1.0).cdf).statistic)
            res[f"sd{c}"] = float(xc.std())
            res[f"mean{c}"] = float(xc.mean())
            res[f"cross{c}"] = float(np.mean(xc > 0) if c == 0 else np.mean(xc < 0))
            if seed == 0: np.save(RUNS / f"overlap_samples_{kind}_{c}.npy", xc)
    return res

# %% [markdown]
# ## Jobs

# %%
RING_KINDS = ["uncond", "cgan", "acgan", "proj"]
OVERLAP_KINDS = ["cgan", "acgan", "proj"]
JOBS = [("ring", k, s) for k in RING_KINDS for s in SEEDS] + [("overlap", k, s) for k in OVERLAP_KINDS for s in SEEDS]

def job_path(j): return RUNS / f"{j[0]}_{j[1]}_{j[2]}.json"

def do_job(j):
    p = job_path(j)
    if p.exists(): return
    exp, kind, s = j; t0 = time.time()
    if exp == "ring":
        Gn = train(kind, s, sample_ring, K, 2, 10000, 128, 3, 2)
        res, x = ring_eval(Gn, kind != "uncond", s)
        if s == 0: np.save(RUNS / f"ring_samples_{kind}.npy", x)
    else:
        Gn = train(kind, s, sample_overlap, 2, 1, 5000, 64, 2, 4)
        res = overlap_eval(Gn, s, kind)
    p.write_text(json.dumps(res))
    print(f"job {exp} {kind} seed {s} done [{time.time()-t0:.0f}s]", flush=True)

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
ms = lambda a, d=1: f"{np.mean(a):.{d}f} ± {np.std(a, ddof=1):.{d}f}"
NAMES = {"uncond": "Unconditional", "cgan": "cGAN (concatenation)", "acgan": "AC-GAN", "proj": "Projection"}

print("Experiment 5.1 (labels drawn with the data frequencies; then 1,000 records requested per class)")
for kind in RING_KINDS:
    rows = [json.loads(job_path(("ring", kind, s)).read_text()) for s in SEEDS]
    RESULTS[f"ring/{kind}"] = rows
    line = (f"  {NAMES[kind]:22s} common modes {ms([r['common_modes'] for r in rows])} (min {min(r['common_modes'] for r in rows)})  "
            f"quality {ms([100*r['quality'] for r in rows])}%  rare share {ms([100*r['rare_share'] for r in rows], 2)}%")
    if kind != "uncond":
        ra = [100 * r["class_acc"][0] for r in rows]
        ca = [100 * np.mean(r["class_acc"][1:]) for r in rows]
        rs = [r["class_spread"][0] / STD for r in rows]
        cs = [np.nanmean(r["class_spread"][1:]) / STD for r in rows]
        line += (f"\n  {'':22s} requested class 0: on-mode {ms(ra)}% (min {min(ra):.1f})  spread/true {ms(rs, 2)};"
                 f"  classes 1-7: on-mode {ms(ca)}%  spread/true {ms(cs, 2)}")
    print(line, flush=True)

print("\nExperiment 5.2 (two overlapping classes, N(-1,1) and N(1,1); true crossing share 15.87%)")
for kind in OVERLAP_KINDS:
    rows = [json.loads(job_path(("overlap", kind, s)).read_text()) for s in SEEDS]
    RESULTS[f"overlap/{kind}"] = rows
    ks = [(r["ks0"] + r["ks1"]) / 2 for r in rows]; cr = [100 * (r["cross0"] + r["cross1"]) / 2 for r in rows]
    sd = [(r["sd0"] + r["sd1"]) / 2 for r in rows]; gap = [r["mean1"] - r["mean0"] for r in rows]
    print(f"  {NAMES[kind]:22s} KS {ms(ks, 3)}  share across 0 {ms(cr)}%  class std {ms(sd, 2)}  gap between class means {ms(gap, 2)} (true 2.00)", flush=True)
print(f"  true share of each class on the other side of 0: {100*stats.norm.cdf(-1):.2f}%")

# Figure 5.1: ring samples, seed 0
fig, axes = plt.subplots(1, 4, figsize=(10, 2.8))
for ax, kind in zip(axes, RING_KINDS):
    x = np.load(RUNS / f"ring_samples_{kind}.npy")
    ax.scatter(*CENTRES.T, s=40, facecolors="none", edgecolors="#999999", lw=0.8)
    ax.scatter(*x.T, s=0.6, color="#1f6fb4")
    ax.scatter(*CENTRES[0], s=90, facecolors="none", edgecolors="#d1495b", lw=1.4)
    ax.set_xlim(-2.8, 2.8); ax.set_ylim(-2.8, 2.8); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(NAMES[kind], fontsize=9)
fig.tight_layout(); fig.savefig(OUT / "fig5_1_conditional_ring.png"); plt.close(fig)

# Figure 5.2: class-conditional densities, overlapping classes, seed 0
fig, axes = plt.subplots(1, 3, figsize=(10, 2.6))
grid = np.linspace(-5, 5, 400)
for ax, kind in zip(axes, OVERLAP_KINDS):
    for c, mu, col in [(0, -1.0, "#1f6fb4"), (1, 1.0, "#d1495b")]:
        xc = np.load(RUNS / f"overlap_samples_{kind}_{c}.npy")
        ax.hist(xc, bins=80, range=(-5, 5), density=True, color=col, alpha=0.35)
        ax.plot(grid, stats.norm.pdf(grid, mu, 1), color=col, lw=1.2)
    ax.axvline(0, color="#555555", lw=0.7, ls="--")
    ax.set_title(NAMES[kind], fontsize=9); ax.set_xlabel("x")
axes[0].set_ylabel("density")
fig.tight_layout(); fig.savefig(OUT / "fig5_2_overlap.png"); plt.close(fig)

Path(OUT / "ch05_results.json").write_text(json.dumps(RESULTS, indent=1, default=float))
print("done")
