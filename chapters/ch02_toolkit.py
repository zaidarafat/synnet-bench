# %% [markdown]
# # Chapter 2 companion notebook: the mathematical toolkit
#
# Every number and figure in Chapter 2 is produced by this file.
# No external data is needed. Run top to bottom; outputs go to ./out/.
#
#   python ch02_toolkit.py
#
# Requirements: numpy, scipy, matplotlib.

# %%
from pathlib import Path

import numpy as np
from scipy import integrate, optimize, stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("out")
OUT.mkdir(exist_ok=True)
SEEDS = [0, 1, 2, 3, 4]
LOG2 = np.log(2.0)

plt.rcParams.update({
    "font.family": "serif", "font.size": 10, "axes.spines.top": False,
    "axes.spines.right": False, "figure.dpi": 150,
})
INK, C1, C2, C3, C4 = "#222222", "#1f6fb4", "#d1495b", "#2a9d8f", "#e9a03b"

# %% [markdown]
# ## Example 2.1 — forward and reverse KL (Section 2.2)
#
# Target p: equal mixture of N(-3, 1) and N(3, 1).
# Model q: a single Gaussian N(mu, sigma^2).

# %%
def p_mix(x):
    return 0.5 * stats.norm.pdf(x, -3, 1) + 0.5 * stats.norm.pdf(x, 3, 1)

GRID = np.linspace(-15, 15, 30001)
DX = GRID[1] - GRID[0]

def kl(pdf_a, pdf_b, grid=GRID):
    a, b = pdf_a(grid), pdf_b(grid)
    mask = a > 1e-300
    return np.sum(a[mask] * (np.log(a[mask]) - np.log(np.maximum(b[mask], 1e-300)))) * DX

def q_pdf(params):
    mu, log_s = params
    return lambda x: stats.norm.pdf(x, mu, np.exp(log_s))

fwd = optimize.minimize(lambda t: kl(p_mix, q_pdf(t)), x0=[0.5, 0.0], method="Nelder-Mead",
                        options={"xatol": 1e-6, "fatol": 1e-9, "maxiter": 4000})
rev = optimize.minimize(lambda t: kl(q_pdf(t), p_mix), x0=[2.0, 0.0], method="Nelder-Mead",
                        options={"xatol": 1e-6, "fatol": 1e-9, "maxiter": 4000})
fwd_mu, fwd_s = fwd.x[0], np.exp(fwd.x[1])
rev_mu, rev_s = rev.x[0], np.exp(rev.x[1])
print("Example 2.1")
print(f"  forward KL(p||q) minimiser: mu = {fwd_mu:.3f}, sigma = {fwd_s:.3f}, KL = {fwd.fun:.3f}")
print(f"  reverse KL(q||p) minimiser: mu = {rev_mu:.3f}, sigma = {rev_s:.3f}, KL = {rev.fun:.3f}")
print(f"  moment-matching check: sigma = sqrt(1 + 9) = {np.sqrt(10):.3f}")

fig, ax = plt.subplots(figsize=(6.4, 2.6))
xs = np.linspace(-8, 8, 800)
ax.fill_between(xs, p_mix(xs), color="#cccccc", label="target p (two modes)")
ax.plot(xs, q_pdf(fwd.x)(xs), color=C1, lw=2, label=f"argmin KL(p‖q): μ={abs(fwd_mu):.2f}, σ={fwd_s:.2f}")
ax.plot(xs, q_pdf(rev.x)(xs), color=C2, lw=2, ls="--", label=f"argmin KL(q‖p): μ={rev_mu:.2f}, σ={rev_s:.2f}")
ax.set_xlabel("x"); ax.set_ylabel("density"); ax.legend(frameon=False, fontsize=8, loc="upper left")
fig.tight_layout(); fig.savefig(OUT / "fig2_1_forward_reverse_kl.png"); plt.close(fig)

# %% [markdown]
# ## Example 2.2 — divergences as one distribution moves (Sections 2.3, 2.5, 2.6)
#
# (a) P = U[0, w], Q_theta = U[theta, theta + w], w = 0.1. Closed forms:
#     JS = log 2 * min(|theta|/w, 1);  TV = min(|theta|/w, 1);  W1 = |theta|;
#     KL(P||Q) = infinity for every theta != 0.
# (b) P = N(0, 1), Q_theta = N(theta, 1). Closed forms:
#     KL = theta^2 / 2;  W1 = |theta|;
#     MMD^2 with Gaussian kernel of bandwidth s: 2 s / sqrt(s^2 + 2) * (1 - exp(-theta^2 / (2 (s^2 + 2))));
#     JS computed by numerical integration.

# %%
W = 0.1
thetas_a = np.linspace(-0.5, 0.5, 1001)
js_a = LOG2 * np.minimum(np.abs(thetas_a) / W, 1.0)
w1_a = np.abs(thetas_a)

def js_gauss(theta):
    p = lambda x: stats.norm.pdf(x, 0, 1)
    q = lambda x: stats.norm.pdf(x, theta, 1)
    def integrand(x):
        a, b = p(x), q(x); m = 0.5 * (a + b)
        t = 0.0
        if a > 0: t += 0.5 * a * np.log(a / m)
        if b > 0: t += 0.5 * b * np.log(b / m)
        return t
    return integrate.quad(integrand, -12 + min(0, theta), 12 + max(0, theta), limit=400)[0]

S_KERNEL = 1.0
def mmd2_gauss(theta, s=S_KERNEL):
    return 2 * s / np.sqrt(s**2 + 2) * (1 - np.exp(-theta**2 / (2 * (s**2 + 2))))

thetas_b = np.linspace(0, 6, 61)
js_b = np.array([js_gauss(t) for t in thetas_b])
kl_b = thetas_b**2 / 2
w1_b = thetas_b
mmd_b = np.sqrt(mmd2_gauss(thetas_b))

print("\nExample 2.2(b): Gaussian pair N(0,1) vs N(theta,1)")
for t in [1, 2, 4, 6]:
    i = int(np.argmin(np.abs(thetas_b - t)))
    print(f"  theta={t}: KL={kl_b[i]:.3f}  JS={js_b[i]:.4f} ({100*js_b[i]/LOG2:.1f}% of log 2)  "
          f"W1={w1_b[i]:.3f}  MMD={mmd_b[i]:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
ax = axes[0]
ax.plot(thetas_a, js_a, color=C2, lw=2, label="JS")
ax.plot(thetas_a, w1_a, color=C1, lw=2, label="W₁")
ax.axhline(LOG2, color="#999999", lw=0.8, ls=":")
ax.text(0.5, LOG2 + 0.02, "log 2", ha="right", va="bottom", fontsize=8, color="#666666")
ax.set_title("(a) U[0, 0.1] vs U[θ, θ+0.1]", fontsize=9)
ax.set_xlabel("θ"); ax.legend(frameon=False, fontsize=8)
ax = axes[1]
ax.plot(thetas_b, js_b, color=C2, lw=2, label="JS")
ax.plot(thetas_b, w1_b, color=C1, lw=2, label="W₁")
ax.plot(thetas_b, mmd_b, color=C3, lw=2, ls="--", label="MMD (Gaussian kernel, s=1)")
ax.plot(thetas_b, kl_b, color=C4, lw=1.5, ls="-.", label="KL")
ax.axhline(LOG2, color="#999999", lw=0.8, ls=":")
ax.set_ylim(0, 6.2); ax.set_title("(b) N(0, 1) vs N(θ, 1)", fontsize=9)
ax.set_xlabel("θ"); ax.legend(frameon=False, fontsize=8, loc="upper left")
fig.tight_layout(); fig.savefig(OUT / "fig2_2_divergences_shift.png"); plt.close(fig)

# %% [markdown]
# ## Example 2.3 — estimating MMD from samples (Section 2.5)
#
# Unbiased quadratic-time estimator of MMD^2, Gaussian kernel s = 1,
# n = m = 2000 samples per distribution, five seeds.

# %%
def mmd2_unbiased(x, y, s=S_KERNEL):
    x = x.reshape(-1, 1); y = y.reshape(-1, 1)
    k = lambda a, b: np.exp(-(a - b.T) ** 2 / (2 * s**2))
    kxx, kyy, kxy = k(x, x), k(y, y), k(x, y)
    n, m = len(x), len(y)
    np.fill_diagonal(kxx, 0.0); np.fill_diagonal(kyy, 0.0)
    return kxx.sum() / (n * (n - 1)) + kyy.sum() / (m * (m - 1)) - 2 * kxy.mean()

N = 2000
print("\nExample 2.3: unbiased MMD^2 estimate vs closed form (n = m = 2000, 5 seeds)")
rows = []
for theta in [0.0, 0.25, 0.5, 1.0]:
    est = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        x = rng.normal(0, 1, N); y = rng.normal(theta, 1, N)
        est.append(mmd2_unbiased(x, y))
    est = np.array(est)
    rows.append((theta, mmd2_gauss(theta), est.mean(), est.std(ddof=1)))
    print(f"  theta={theta:<4}  exact={mmd2_gauss(theta):.5f}  estimate={est.mean():.5f} ± {est.std(ddof=1):.5f}")

# %% [markdown]
# ## Example 2.4 — the bilinear game min_x max_y x*y (Section 2.7)
#
# Unique equilibrium (0, 0). Step size eta = 0.1, start (1, 1), 200 steps.
# Simultaneous gradient descent-ascent: update matrix [[1, -eta], [eta, 1]],
# eigenvalues 1 ± i*eta, modulus sqrt(1 + eta^2) > 1.

# %%
ETA, STEPS = 0.1, 200
def simultaneous(x, y):
    return x - ETA * y, y + ETA * x
def alternating(x, y):
    x_new = x - ETA * y
    return x_new, y + ETA * x_new

paths = {}
for name, step in [("simultaneous", simultaneous), ("alternating", alternating)]:
    pts = [(1.0, 1.0)]
    for _ in range(STEPS):
        pts.append(step(*pts[-1]))
    paths[name] = np.array(pts)

r0 = np.hypot(1, 1)
r_sim = np.hypot(*paths["simultaneous"][-1])
print("\nExample 2.4: bilinear game, eta = 0.1, 200 steps from (1, 1)")
print(f"  simultaneous: distance from equilibrium {r0:.3f} -> {r_sim:.3f} "
      f"(predicted {r0 * (1 + ETA**2) ** (STEPS / 2):.3f})")
r_alt = np.hypot(paths["alternating"][:, 0], paths["alternating"][:, 1])
print(f"  alternating: distance stays in [{r_alt.min():.3f}, {r_alt.max():.3f}]")

fig, ax = plt.subplots(figsize=(5.2, 3.4))
ax.plot(*paths["simultaneous"].T, color=C2, lw=1.4, label="simultaneous")
ax.plot(*paths["alternating"].T, color=C1, lw=1.4, ls="--", label="alternating")
ax.plot([0], [0], marker="*", color=INK, ms=10, ls="none", label="equilibrium")
ax.plot([1], [1], marker="o", color=INK, ms=4, ls="none")
ax.set_xlabel("x (minimising player)"); ax.set_ylabel("y (maximising player)")
ax.set_aspect("equal"); ax.legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(1.02, 0.5))
fig.tight_layout(); fig.savefig(OUT / "fig2_3_bilinear_game.png"); plt.close(fig)

print("\nFigures written to", OUT.resolve())
