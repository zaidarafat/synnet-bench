# %% [markdown]
# # Chapter 6 companion notebook: encoding structured data for a GAN
#
# Every number and figure in Chapter 6 is produced by this file. No external data:
# the flow table is simulated (Section 6.5 describes the simulation).
#   python ch06_encodings.py                 # everything, one process
#   python ch06_encodings.py worker 0 2      # half of the training jobs
#   python ch06_encodings.py worker 1 2      # the other half
#   python ch06_encodings.py report          # tables and figures from cached jobs
# Requirements: torch, numpy, scipy, scikit-learn, matplotlib.

# %%
from pathlib import Path
import json, sys, time, warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats
from sklearn.mixture import BayesianGaussianMixture
from sklearn.preprocessing import QuantileTransformer

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
OUT = Path("out"); OUT.mkdir(exist_ok=True)
RUNS = OUT / "runs"; RUNS.mkdir(exist_ok=True)
SEEDS = [0, 1, 2, 3, 4]
MODE = sys.argv[1] if len(sys.argv) > 1 else "all"
torch.set_num_threads(1)
plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 150})
RESULTS = {}

# %% [markdown]
# ## The simulated flow table
#
# Four columns, built to have the properties of Section 6.1 in one table:
#
# * protocol  — categorical: TCP 80%, UDP 18%, ICMP 2%.
# * packets   — integer >= 1, heavy-tailed. A flow is a single packet with
#               probability 0.2 (TCP), 0.6 (UDP) or 0.9 (ICMP); otherwise
#               packets = 2 + floor(LogNormal(1.5, 1.2)).
# * duration  — seconds; exactly 0 for single-packet flows (a point mass),
#               otherwise LogNormal(-1 + 0.5 log(packets), 1.5).
# * bytes     — packets x size per packet, where the size per packet is bimodal:
#               N(70, 10) clipped to [40, 120] with probability 0.55, otherwise
#               N(1200, 150) clipped to [600, 1500]; rounded to an integer.
#
# Logical constraints that every real record satisfies:
#   C1  20 * packets <= bytes <= 1500 * packets
#   C2  duration == 0  if and only if  packets == 1
#   C3  packets is a positive integer, bytes a non-negative integer, duration >= 0

# %%
PROTOS = ["TCP", "UDP", "ICMP"]
P_PROTO = np.array([0.80, 0.18, 0.02])
P_SINGLE = np.array([0.2, 0.6, 0.9])

def simulate(n, rng):
    proto = rng.choice(3, size=n, p=P_PROTO)
    single = rng.rand(n) < P_SINGLE[proto]
    packets = np.where(single, 1, 2 + np.floor(rng.lognormal(1.5, 1.2, size=n))).astype(np.int64)
    duration = np.where(single, 0.0, rng.lognormal(-1 + 0.5 * np.log(packets), 1.5))
    small = rng.rand(n) < 0.55
    size = np.where(small, np.clip(rng.normal(70, 10, n), 40, 120), np.clip(rng.normal(1200, 150, n), 600, 1500))
    bytes_ = np.round(packets * size).astype(np.int64)
    return {"protocol": proto, "packets": packets.astype(float), "duration": duration, "bytes": bytes_.astype(float)}

CONT = ["packets", "duration", "bytes"]

# %% [markdown]
# ## Six encodings of the continuous columns
#
# The protocol is one-hot encoded in all six. The continuous columns are encoded as:
#
# | name     | encoding                                                              | generator output |
# |----------|-----------------------------------------------------------------------|------------------|
# | minmax   | (x - min)/(max - min) * 2 - 1                                         | tanh             |
# | standard | (x - mean)/sd                                                         | linear           |
# | log      | standardised log(1 + x)                                               | linear           |
# | quantile | quantile transform to a standard normal (1,000 quantiles)             | linear           |
# | vgm      | mode-specific normalisation (CTGAN): Bayesian Gaussian mixture of up  | tanh + softmax   |
# |          | to 10 components; value -> (one-hot mode, (x - mu_k)/(4 sd_k))        |                  |
# | mixed    | as vgm, applied to log(1 + x), and a zero indicator for duration      | tanh + softmax   |
# |          | (the "mixed" variable of CTAB-GAN): zero values get their own category|                  |
#
# Decoding inverts the encoding, clips to the training range, rounds packets
# and bytes to integers and sets packets >= 1.

# %%
class Encoder:
    def __init__(self, kind, data, seed):
        self.kind, self.blocks, self.p = kind, [], {}
        self.blocks.append(("cat", 3))
        for c in CONT:
            x = data[c]
            self.p[c] = {"lo": float(x.min()), "hi": float(x.max())}
            if kind == "minmax":
                self.blocks.append(("tanh", 1))
            elif kind in ("standard", "log"):
                v = np.log1p(x) if kind == "log" else x
                self.p[c].update(mu=float(v.mean()), sd=float(v.std()))
                self.blocks.append(("lin", 1))
            elif kind == "quantile":
                q = QuantileTransformer(n_quantiles=1000, output_distribution="normal", random_state=seed, subsample=10**9)
                q.fit(x.reshape(-1, 1)); self.p[c]["q"] = q
                self.blocks.append(("lin", 1))
            else:
                v = np.log1p(x) if kind == "mixed" else x
                zero = (kind == "mixed" and c == "duration")
                fit_v = v[x > 0] if zero else v
                rng = np.random.RandomState(seed)
                sub = fit_v[rng.choice(len(fit_v), size=min(10000, len(fit_v)), replace=False)]
                g = BayesianGaussianMixture(n_components=10, weight_concentration_prior_type="dirichlet_process",
                                            weight_concentration_prior=0.001, max_iter=200, random_state=seed)
                g.fit(sub.reshape(-1, 1))
                keep = g.weights_ > 0.005
                self.p[c].update(g=g, keep=keep, mu=g.means_.ravel()[keep], sd=np.sqrt(g.covariances_.ravel()[keep]), zero=zero)
                self.blocks.append(("tanh", 1)); self.blocks.append(("cat", int(keep.sum()) + (1 if zero else 0)))
        self.dim = sum(b[1] for b in self.blocks)

    def encode(self, data, rng):
        cols = [np.eye(3)[data["protocol"]]]
        for c in CONT:
            x, p = data[c], self.p[c]
            if self.kind == "minmax":
                cols.append(((x - p["lo"]) / (p["hi"] - p["lo"]) * 2 - 1)[:, None])
            elif self.kind in ("standard", "log"):
                v = np.log1p(x) if self.kind == "log" else x
                cols.append(((v - p["mu"]) / p["sd"])[:, None])
            elif self.kind == "quantile":
                cols.append(p["q"].transform(x.reshape(-1, 1)))
            else:
                v = np.log1p(x) if self.kind == "mixed" else x
                post = p["g"].predict_proba(v.reshape(-1, 1))[:, p["keep"]]
                tot = post.sum(1, keepdims=True)
                # a value far in the tail can have zero posterior under every kept component:
                # assign it to the kept component with the nearest mean
                near = np.abs(v[:, None] - p["mu"][None, :]).argmin(1)
                post = np.where(tot > 0, post / np.where(tot > 0, tot, 1), np.eye(len(p["mu"]))[near])
                k = np.array([rng.choice(len(r), p=r) for r in post])       # sampled mode, as in CTGAN
                alpha = np.clip((v - p["mu"][k]) / (4 * p["sd"][k]), -0.99, 0.99)
                onehot = np.eye(len(p["mu"]))[k]
                if p["zero"]:
                    z = (x == 0)
                    alpha = np.where(z, 0.0, alpha)
                    onehot = np.hstack([np.where(z[:, None], 0.0, onehot), z[:, None].astype(float)])
                cols += [alpha[:, None], onehot]
        return np.hstack(cols).astype(np.float32)

    def decode(self, e):
        i = 0; out = {}
        out["protocol"] = e[:, i:i + 3].argmax(1); i += 3
        for c in CONT:
            p = self.p[c]
            if self.kind == "minmax":
                x = (e[:, i] + 1) / 2 * (p["hi"] - p["lo"]) + p["lo"]; i += 1
            elif self.kind in ("standard", "log"):
                v = e[:, i] * p["sd"] + p["mu"]; i += 1
                x = np.expm1(v) if self.kind == "log" else v
            elif self.kind == "quantile":
                x = p["q"].inverse_transform(e[:, i:i + 1]).ravel(); i += 1
            else:
                alpha = e[:, i]; m = len(p["mu"]) + (1 if p["zero"] else 0)
                k = e[:, i + 1:i + 1 + m].argmax(1); i += 1 + m
                kk = np.minimum(k, len(p["mu"]) - 1)
                v = alpha * 4 * p["sd"][kk] + p["mu"][kk]
                x = np.expm1(v) if self.kind == "mixed" else v
                if p["zero"]: x = np.where(k == len(p["mu"]), 0.0, x)
            x = np.clip(x, p["lo"], p["hi"])
            if c in ("packets", "bytes"): x = np.round(x)
            if c == "packets": x = np.maximum(x, 1)
            out[c] = x
        return out

# %% [markdown]
# ## The GAN (fixed across encodings)
#
# Unconditional NS + R1 (gamma = 10), the configuration of Table 4.1. Generator
# and discriminator: three hidden layers of 128 units, leaky ReLU. Noise
# dimension 16. Categorical blocks of the generator use the Gumbel-softmax with
# temperature 0.2, as CTGAN does. 10,000 generator updates, batch 500, Adam 1e-4.

# %%
def mlp(d_in, d_out, width=128, depth=3):
    layers, d = [], d_in
    for _ in range(depth):
        layers += [nn.Linear(d, width), nn.LeakyReLU(0.2)]; d = width
    layers.append(nn.Linear(d, d_out))
    return nn.Sequential(*layers)

def activate(h, blocks):
    out, i = [], 0
    for kind, n in blocks:
        s = h[:, i:i + n]; i += n
        out.append(torch.tanh(s) if kind == "tanh" else F.gumbel_softmax(s, tau=0.2) if kind == "cat" else s)
    return torch.cat(out, 1)

def train_and_sample(kind, seed, n_out=20000, steps=10000):
    torch.manual_seed(seed); rng = np.random.RandomState(seed)
    data = simulate(50000, rng)
    enc = Encoder(kind, data, seed)
    X = torch.tensor(enc.encode(data, rng))
    G, D = mlp(16, enc.dim), mlp(enc.dim, 1)
    og = torch.optim.Adam(G.parameters(), lr=1e-4, betas=(0.5, 0.999)); od = torch.optim.Adam(D.parameters(), lr=1e-4, betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss(); B = 500; ones, zeros = torch.ones(B, 1), torch.zeros(B, 1)
    for t in range(steps):
        xr = X[torch.randint(0, len(X), (B,))]
        with torch.no_grad(): xg = activate(G(torch.randn(B, 16)), enc.blocks)
        xr_ = xr.clone().requires_grad_(True); dr = D(xr_)
        g = torch.autograd.grad(dr.sum(), xr_, create_graph=True)[0]
        ld = bce(dr, ones) + bce(D(xg), zeros) + 0.5 * 10.0 * g.pow(2).sum(1).mean()
        od.zero_grad(); ld.backward(); od.step()
        lg = bce(D(activate(G(torch.randn(B, 16)), enc.blocks)), ones)
        og.zero_grad(); lg.backward(); og.step()
    with torch.no_grad():
        e = activate(G(torch.randn(n_out, 16)), enc.blocks).numpy()
    return enc.decode(e)

# %% [markdown]
# ## Measurements
#
# Against 20,000 fresh records from the simulator:
# * KS statistic for packets, duration and bytes;
# * share of zero durations (true 30.1%, see report) and total variation distance of protocol frequencies;
# * share of single-packet flows within each protocol;
# * mean absolute difference between the Spearman correlation matrices of the three continuous columns;
# * violation rates of constraints C1 and C2 (C3 is enforced by decoding).

# %%
def measure(gen, test):
    m = {f"ks_{c}": float(stats.ks_2samp(gen[c], test[c]).statistic) for c in CONT}
    m["zero_dur"] = float(np.mean(gen["duration"] == 0))
    fg = np.bincount(gen["protocol"], minlength=3) / len(gen["protocol"]); ft = np.bincount(test["protocol"], minlength=3) / len(test["protocol"])
    m["tv_proto"] = float(0.5 * np.abs(fg - ft).sum())
    m["single_by_proto"] = [float(np.mean(gen["packets"][gen["protocol"] == k] == 1)) if np.any(gen["protocol"] == k) else float("nan") for k in range(3)]
    sg = stats.spearmanr(np.column_stack([gen[c] for c in CONT])).correlation
    st = stats.spearmanr(np.column_stack([test[c] for c in CONT])).correlation
    m["corr_err"] = float(np.abs(sg - st)[np.triu_indices(3, 1)].mean())
    m["viol_c1"] = float(np.mean((gen["bytes"] < 20 * gen["packets"]) | (gen["bytes"] > 1500 * gen["packets"])))
    m["viol_c2"] = float(np.mean((gen["duration"] == 0) != (gen["packets"] == 1)))
    return m

# %% [markdown]
# ## Jobs

# %%
KINDS = ["minmax", "standard", "log", "quantile", "vgm", "mixed"]
JOBS = [(k, s) for k in KINDS for s in SEEDS]
def job_path(j): return RUNS / f"{j[0]}_{j[1]}.json"

TEST = simulate(20000, np.random.RandomState(999))

def do_job(j):
    p = job_path(j)
    if p.exists(): return
    t0 = time.time()
    gen = train_and_sample(*j)
    res = measure(gen, TEST)
    if j[1] == 0: np.savez(RUNS / f"samples_{j[0]}.npz", **gen)
    p.write_text(json.dumps(res))
    print(f"job {j[0]} seed {j[1]} done [{time.time()-t0:.0f}s]", flush=True)

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
ms = lambda a, d=3: f"{np.nanmean(a):.{d}f} ± {np.nanstd(a, ddof=1):.{d}f}"

print("Simulated table, 20,000 test records:")
for c in CONT:
    v = TEST[c]
    print(f"  {c:9s} min {v.min():.4g}  median {np.median(v):.4g}  99th pct {np.percentile(v, 99):.4g}  max {v.max():.4g}")
print(f"  share of zero durations {100*np.mean(TEST['duration']==0):.1f}%;  protocol shares {np.round(100*np.bincount(TEST['protocol'])/len(TEST['protocol']),1)}")
print(f"  single-packet share by protocol (TCP, UDP, ICMP): {[round(100*np.mean(TEST['packets'][TEST['protocol']==k]==1),1) for k in range(3)]}")
st = stats.spearmanr(np.column_stack([TEST[c] for c in CONT])).correlation
print(f"  Spearman correlations: packets-duration {st[0,1]:.3f}, packets-bytes {st[0,2]:.3f}, duration-bytes {st[1,2]:.3f}")
real_m = measure(simulate(20000, np.random.RandomState(1234)), TEST)
RESULTS["reference_fresh_sample"] = real_m
print(f"  reference (a second real sample of 20,000): KS {[round(real_m['ks_'+c],3) for c in CONT]}  corr err {real_m['corr_err']:.3f}  tv {real_m['tv_proto']:.3f}")

print("\nRound trip: encode then decode the 20,000 test records (encoder fitted on the seed-0 training table)")
for kind in KINDS:
    rng = np.random.RandomState(0); tr = simulate(50000, rng)
    enc = Encoder(kind, tr, 0); back = enc.decode(enc.encode(TEST, np.random.RandomState(5)))
    bad = {c: float(np.mean(np.abs(back[c] - TEST[c]) > 0.01 * np.maximum(TEST[c], 1e-9) + 1e-9)) for c in CONT}
    RESULTS.setdefault("roundtrip", {})[kind] = bad
    print(f"  {kind:9s} share of records changed by more than 1%: " + "  ".join(f"{c} {100*v:.2f}%" for c, v in bad.items()))
print("\nExperiment 6.1")
for kind in KINDS:
    rows = [json.loads(job_path((kind, s)).read_text()) for s in SEEDS]
    RESULTS[kind] = rows
    print(f"  {kind:9s} KS packets {ms([r['ks_packets'] for r in rows])}  duration {ms([r['ks_duration'] for r in rows])}  bytes {ms([r['ks_bytes'] for r in rows])}"
          f"  | zero dur {ms([100*r['zero_dur'] for r in rows],1)}%  TV proto {ms([r['tv_proto'] for r in rows])}"
          f"  | corr err {ms([r['corr_err'] for r in rows])}  | C1 viol {ms([100*r['viol_c1'] for r in rows],1)}%  C2 viol {ms([100*r['viol_c2'] for r in rows],1)}%"
          f"  | single by proto {[ms([100*r['single_by_proto'][k] for r in rows],1) for k in range(3)]}", flush=True)

# Figure 6.1: distribution of bytes per packet (log scale) and duration, seed 0
fig, axes = plt.subplots(2, len(KINDS), figsize=(13, 4.4), sharex="row")
bpp_t = TEST["bytes"] / TEST["packets"]
for j, kind in enumerate(KINDS):
    g = dict(np.load(RUNS / f"samples_{kind}.npz"))
    bpp = g["bytes"] / g["packets"]
    bins = np.logspace(np.log10(10), np.log10(3000), 60)
    axes[0, j].hist(bpp_t, bins=bins, density=True, color="#999999", alpha=0.6)
    axes[0, j].hist(bpp, bins=bins, density=True, histtype="step", color="#1f6fb4", lw=1.2)
    axes[0, j].set_xscale("log"); axes[0, j].set_title(kind, fontsize=9)
    dbins = np.logspace(-4, 3, 60)
    axes[1, j].hist(TEST["duration"][TEST["duration"] > 0], bins=dbins, density=True, color="#999999", alpha=0.6)
    axes[1, j].hist(g["duration"][g["duration"] > 0], bins=dbins, density=True, histtype="step", color="#1f6fb4", lw=1.2)
    axes[1, j].set_xscale("log")
    axes[1, j].text(0.03, 0.92, f"zeros {100*np.mean(g['duration']==0):.0f}%", transform=axes[1, j].transAxes, fontsize=7)
    for ax in axes[:, j]: ax.set_yticks([])
axes[0, 0].set_ylabel("bytes per packet", fontsize=9); axes[1, 0].set_ylabel("duration > 0 (s)", fontsize=9)
fig.tight_layout(); fig.savefig(OUT / "fig6_1_encodings.png"); plt.close(fig)

Path(OUT / "ch06_results.json").write_text(json.dumps(RESULTS, indent=1, default=float))
print("done")
