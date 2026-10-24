# %% [markdown]
# # Chapter 10 companion notebook: protocol consistency of generated packet sequences
#
# Every number and figure in Chapter 10 is produced by this file. No external data:
# the TCP connections are simulated (Section 10.5 describes the simulation).
#   python ch10_protocol.py worker 0 2 ; python ch10_protocol.py worker 1 2
#   python ch10_protocol.py report
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
torch.set_num_threads(1)
plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 150})
RESULTS = {}

# %% [markdown]
# ## Simulated TCP connections
#
# A connection is a sequence of 16 packet tokens (padded with END) and a payload
# size per packet. Tokens combine direction (C = client, S = server) and TCP flags:
#   C_SYN, S_SYNACK, C_ACK, S_ACK, C_DATA, S_DATA, C_FIN, S_FIN, S_RST, C_RST, END
# Four kinds of connection:
#   web (60%)          : handshake; 1-2 rounds of [C_DATA, S_ACK, S_DATA x 1-2, C_ACK]; C_FIN, S_FIN, C_ACK
#   scan, closed (25%) : C_SYN, S_RST
#   scan, open (10%)   : C_SYN, S_SYNACK, C_RST
#   failed (5%)        : C_SYN repeated 2-3 times (no answer)
# Payload: 0 bytes for control packets; client data LogNormal(log 300, 0.5),
# server data LogNormal(log 900, 0.6), both rounded and clipped to [1, 1460].

# %%
TOK = ["C_SYN", "S_SYNACK", "C_ACK", "S_ACK", "C_DATA", "S_DATA", "C_FIN", "S_FIN", "S_RST", "C_RST", "END"]
K, T = len(TOK), 16
I = {t: i for i, t in enumerate(TOK)}
DATA = {I["C_DATA"], I["S_DATA"]}
KINDS_C = ["web", "scan_closed", "scan_open", "failed"]

def simulate(n, rng):
    toks = np.full((n, T), I["END"]); size = np.zeros((n, T)); kind = rng.choice(4, size=n, p=[0.6, 0.25, 0.1, 0.05])
    for i in range(n):
        if kind[i] == 0:
            s = ["C_SYN", "S_SYNACK", "C_ACK"]
            for _ in range(rng.randint(1, 3)):
                s += ["C_DATA", "S_ACK"] + ["S_DATA"] * rng.randint(1, 3) + ["C_ACK"]
            s += ["C_FIN", "S_FIN", "C_ACK"]
        elif kind[i] == 1: s = ["C_SYN", "S_RST"]
        elif kind[i] == 2: s = ["C_SYN", "S_SYNACK", "C_RST"]
        else: s = ["C_SYN"] * rng.randint(2, 4)
        for k, t in enumerate(s):
            toks[i, k] = I[t]
            if t == "C_DATA": size[i, k] = np.clip(np.round(rng.lognormal(np.log(300), 0.5)), 1, 1460)
            if t == "S_DATA": size[i, k] = np.clip(np.round(rng.lognormal(np.log(900), 0.6)), 1, 1460)
    return toks, size, kind

TRAIN = simulate(4000, np.random.RandomState(100))
TEST = simulate(2000, np.random.RandomState(200))

# %% [markdown]
# ## Protocol rules (a small TCP state machine)
#
#   R1  the first packet is C_SYN
#   R2  S_SYNACK appears only immediately after C_SYN
#   R3  data packets appear only after the handshake (C_SYN, S_SYNACK, C_ACK) is complete
#   R4  after END, S_RST or C_RST, every following token is END
#   R5  control packets carry 0 bytes; data packets carry 1 to 1460 bytes
# A connection is valid if it satisfies all five.

# %%
def check(toks, size):
    n = len(toks); v = {f"R{k}": np.zeros(n, bool) for k in range(1, 6)}
    for i in range(n):
        t = toks[i]
        v["R1"][i] = t[0] != I["C_SYN"]
        hs, ended = 0, False
        for k in range(T):
            x = t[k]
            if ended and x != I["END"]: v["R4"][i] = True
            if x in (I["END"], I["S_RST"], I["C_RST"]): ended = True
            if x == I["S_SYNACK"] and (k == 0 or t[k - 1] != I["C_SYN"]): v["R2"][i] = True
            if x == I["C_SYN"] and hs == 0: hs = 1
            elif x == I["S_SYNACK"] and hs == 1: hs = 2
            elif x == I["C_ACK"] and hs == 2: hs = 3
            if x in DATA and hs < 3: v["R3"][i] = True
            if x in DATA: bad = not (1 <= size[i, k] <= 1460)
            else: bad = size[i, k] != 0
            if bad: v["R5"][i] = True
    out = {r: float(m.mean()) for r, m in v.items()}
    out["valid"] = float(1 - np.mean(np.any(np.stack(list(v.values())), 0)))
    return out

def kind_of(toks):
    k = []
    for t in toks:
        s = [TOK[x] for x in t if x != I["END"]]
        if s[:2] == ["C_SYN", "S_RST"] and len(s) == 2: k.append(1)
        elif s[:3] == ["C_SYN", "S_SYNACK", "C_RST"] and len(s) == 3: k.append(2)
        elif len(s) in (2, 3) and all(x == "C_SYN" for x in s): k.append(3)
        elif s[:3] == ["C_SYN", "S_SYNACK", "C_ACK"] and s[-3:] == ["C_FIN", "S_FIN", "C_ACK"]: k.append(0)
        else: k.append(4)                                   # none of the four kinds
    return np.array(k)

def fidelity(toks, size):
    ref_t, ref_s, _ = TEST
    uni = lambda t: np.bincount(t.ravel(), minlength=K) / t.size
    def bi(t):
        m = np.zeros((K, K))
        np.add.at(m, (t[:, :-1].ravel(), t[:, 1:].ravel()), 1); return m / m.sum()
    length = lambda t: (t != I["END"]).sum(1)
    kd, kr = np.bincount(kind_of(toks), minlength=5) / len(toks), np.bincount(kind_of(ref_t), minlength=5) / len(ref_t)
    sbytes = lambda t, s: (s * (t == I["S_DATA"])).sum(1)
    w = kind_of(ref_t) == 0; wg = kind_of(toks) == 0
    return {"tv_unigram": float(0.5 * np.abs(uni(toks) - uni(ref_t)).sum()),
            "tv_bigram": float(0.5 * np.abs(bi(toks) - bi(ref_t)).sum()),
            "ks_length": float(stats.ks_2samp(length(toks), length(ref_t)).statistic),
            "kind_share": kd.tolist(), "tv_kind": float(0.5 * np.abs(kd - kr).sum()),
            "ks_server_bytes_web": float(stats.ks_2samp(sbytes(toks, size)[wg], sbytes(ref_t, ref_s)[w]).statistic) if wg.sum() > 10 else float("nan")}

# %% [markdown]
# ## Generators
#
# * iid     : token and size at each position drawn independently from the training connections at that position
# * markov  : first-order Markov chain over tokens, fitted by counting; size drawn from the training sizes of the token
# * gan     : LSTM generator (noise of dimension 4 per step, hidden 64) emitting token logits and a size per step;
#             LSTM discriminator with spectral normalisation on every weight matrix; hinge loss;
#             Adam 2e-4, betas (0, 0.9); 3,000 generator updates, batch 128.
#             Tokens pass through a Gumbel-softmax (temperature 0.5) during training and the
#             most probable token is taken at sampling. Sizes are encoded as log(1 + bytes)/log(1461).
# * gan+cd  : the same trained generator with constrained decoding: at each step the most probable
#             token among those the state machine allows is taken, and the size is set to 0 for
#             control packets and clipped to [1, 1460] for data packets.

# %%
def gen_iid(n, rng):
    t, s, _ = TRAIN; idx = rng.randint(0, len(t), size=(n, T))
    return t[idx, np.arange(T)], s[idx, np.arange(T)]

def gen_markov(n, rng):
    t, s, _ = TRAIN
    P = np.ones((K, K)) * 1e-9; np.add.at(P, (t[:, :-1].ravel(), t[:, 1:].ravel()), 1); P /= P.sum(1, keepdims=True)
    p0 = np.bincount(t[:, 0], minlength=K) / len(t)
    pools = {k: s[t == k] for k in range(K)}
    out = np.zeros((n, T), int); sz = np.zeros((n, T))
    out[:, 0] = rng.choice(K, size=n, p=p0)
    for k in range(1, T):
        for i in range(n): out[i, k] = rng.choice(K, p=P[out[i, k - 1]])
    for k in range(K):
        m = out == k
        if m.any(): sz[m] = rng.choice(pools[k], size=m.sum()) if len(pools[k]) else 0
    return out, sz

H, ZD, B = 64, 4, 128
LOGMAX = np.log(1461.0)

class Gen(nn.Module):
    def __init__(self):
        super().__init__(); self.rnn = nn.LSTM(ZD, H, num_layers=2, batch_first=True); self.tok = nn.Linear(H, K); self.size = nn.Linear(H, 1)
    def forward(self, z):
        h = self.rnn(z)[0]; return self.tok(h), torch.sigmoid(self.size(h))

class Dis(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.LSTM(K + 1, H, batch_first=True)
        for name in ("weight_ih_l0", "weight_hh_l0"): spectral_norm(self.rnn, name=name)
        self.out = spectral_norm(nn.Linear(H, 1))
    def forward(self, x): return self.out(self.rnn(x)[0]).mean(1)

def encode(t, s):
    return torch.tensor(np.concatenate([np.eye(K)[t], (np.log1p(s) / LOGMAX)[..., None]], 2), dtype=torch.float32)

def train_gan(seed, steps=3000):
    torch.manual_seed(seed)
    X = encode(TRAIN[0], TRAIN[1])
    G, D = Gen(), Dis()
    og = torch.optim.Adam(G.parameters(), 2e-4, betas=(0.0, 0.9)); od = torch.optim.Adam(D.parameters(), 2e-4, betas=(0.0, 0.9))
    def fake(n):
        lg, sz = G(torch.randn(n, T, ZD)); return torch.cat([F.gumbel_softmax(lg, tau=0.5), sz], 2)
    for _ in range(steps):
        xr = X[torch.randint(0, len(X), (B,))]
        with torch.no_grad(): xg = fake(B)
        ld = torch.relu(1 - D(xr)).mean() + torch.relu(1 + D(xg)).mean()
        od.zero_grad(); ld.backward(); od.step()
        lg = -D(fake(B)).mean(); og.zero_grad(); lg.backward(); og.step()
    return G

ALLOWED_AFTER_END = {I["END"]}

def decode(G, n, seed, constrained):
    torch.manual_seed(10_000 + seed)
    with torch.no_grad():
        lg, sz = G(torch.randn(n, T, ZD))
    lg = lg.numpy(); sz = np.expm1(sz.numpy()[..., 0] * LOGMAX)
    if not constrained:
        t = lg.argmax(2); return t, np.round(sz)
    t = np.full((n, T), I["END"]); out_sz = np.zeros((n, T))
    for i in range(n):
        hs, ended, prev = 0, False, None
        for k in range(T):
            allowed = set(range(K))
            if k == 0: allowed = {I["C_SYN"]}
            elif ended: allowed = {I["END"]}
            else:
                if prev != I["C_SYN"]: allowed.discard(I["S_SYNACK"])
                if hs < 3: allowed -= DATA
            x = max(allowed, key=lambda a: lg[i, k, a])
            t[i, k] = x
            if x in (I["END"], I["S_RST"], I["C_RST"]): ended = True
            if x == I["C_SYN"] and hs == 0: hs = 1
            elif x == I["S_SYNACK"] and hs == 1: hs = 2
            elif x == I["C_ACK"] and hs == 2: hs = 3
            out_sz[i, k] = np.clip(np.round(sz[i, k]), 1, 1460) if x in DATA else 0
            prev = x
    return t, out_sz

# %% [markdown]
# ## Jobs

# %%
GENS = ["iid", "markov", "gan", "gan+cd"]
JOBS = [("gan", s) for s in SEEDS] + [(k, s) for k in ("iid", "markov") for s in SEEDS]
def job_path(j): return RUNS / f"{j[0]}_{j[1]}.json"

def evaluate(t, s):
    return {**check(t, s), **fidelity(t, s)}

def do_job(j):
    p = job_path(j)
    if p.exists(): return
    t0 = time.time(); kind, seed = j; rng = np.random.RandomState(seed)
    if kind == "gan":
        G = train_gan(seed); fit = time.time() - t0
        res = {}
        for name, c in (("gan", False), ("gan+cd", True)):
            t, s = decode(G, 2000, seed, c); res[name] = evaluate(t, s)
            if seed == 0: np.savez(RUNS / f"{name}_0.npz", t=t, s=s)
        res["fit_seconds"] = fit
    else:
        t, s = (gen_iid if kind == "iid" else gen_markov)(2000, rng)
        res = {kind: evaluate(t, s)}
        if seed == 0: np.savez(RUNS / f"{kind}_0.npz", t=t, s=s)
    p.write_text(json.dumps(res))
    print(f"job {kind} seed {seed} done [{time.time()-t0:.0f}s]", flush=True)

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
ref = evaluate(TEST[0][:2000], TEST[1][:2000]); RESULTS["test_vs_test"] = ref
tr = evaluate(TRAIN[0][:2000], TRAIN[1][:2000]); RESULTS["train_vs_test"] = tr
print("Real (train vs test):", {k: (round(v, 3) if isinstance(v, float) else [round(x, 3) for x in v]) for k, v in tr.items()})
rows = {g: [] for g in GENS}
for s in SEEDS:
    for kind in ("iid", "markov", "gan"):
        r = json.loads(job_path((kind, s)).read_text())
        for g in GENS:
            if g in r: rows[g].append(r[g] | ({"fit_seconds": r["fit_seconds"]} if "fit_seconds" in r else {}))
for g in GENS:
    R = rows[g]; RESULTS[g] = R
    print(f"{g:7s} valid {ms([100*r['valid'] for r in R],1)}%  " + "  ".join(f"{k} {ms([100*r[k] for r in R],1)}%" for k in ("R1","R2","R3","R4","R5"))
          + f"  | TV uni {ms([r['tv_unigram'] for r in R])}  TV bi {ms([r['tv_bigram'] for r in R])}  KS len {ms([r['ks_length'] for r in R])}"
          + f"  TV kind {ms([r['tv_kind'] for r in R])}  KS server bytes {ms([r['ks_server_bytes_web'] for r in R])}"
          + "  | kinds " + " ".join(f"{n} {ms([100*r['kind_share'][i] for r in R],1)}" for i, n in enumerate(KINDS_C + ["other"]))
          + (f"  fit {ms([r['fit_seconds'] for r in R],0)} s" if "fit_seconds" in R[0] else ""), flush=True)

# Figure 10.1: share of each connection kind, real vs generators (seed 0)
fig, ax = plt.subplots(figsize=(7.5, 2.8))
labels = ["web", "scan (closed)", "scan (open)", "failed", "none of these"]
series = [("real", kind_of(TEST[0]))] + [(g, kind_of(np.load(RUNS / f"{g}_0.npz")["t"])) for g in GENS]
w = 0.16
for j, (name, k) in enumerate(series):
    ax.bar(np.arange(5) + (j - 2) * w, np.bincount(k, minlength=5) / len(k), w, label=name)
ax.set_xticks(range(5)); ax.set_xticklabels(labels, fontsize=8); ax.set_ylabel("share of connections"); ax.legend(frameon=False, fontsize=7, ncol=5)
fig.tight_layout(); fig.savefig(OUT / "fig10_1_kinds.png"); plt.close(fig)

Path(OUT / "ch10_results.json").write_text(json.dumps(RESULTS, indent=1, default=float))
print("done")
