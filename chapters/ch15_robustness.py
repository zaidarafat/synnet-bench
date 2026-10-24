# %% [markdown]
# # Chapter 15 companion notebook: robustness of intrusion detectors (a defender's view)
#
# This notebook is written from the defender's side. It measures how much an intrusion
# detector's recall on attacks drops when attack feature-vectors are perturbed within a
# small budget, and how much standard adversarial training (a defence) restores it. It is
# a robustness *evaluation*: the perturbation is the stress test, adversarial training is
# the remedy, and the quantity reported is detector recall. The notebook trains no
# generator of attack traffic, saves no perturbed records, and produces nothing that could
# be replayed on a network; it works on feature vectors of a public benchmark in memory.
#
# Data: `KDDTrain+.txt` from the NSL-KDD distribution (set DATA).
#   python ch15_robustness.py
# Requirements: torch, numpy, pandas, scikit-learn.

# %%
from pathlib import Path
import json, time, warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
DATA = Path("/home/claude/data/nslkdd/KDDTrain+.txt")
OUT = Path("out"); OUT.mkdir(exist_ok=True)
SEEDS = [0, 1, 2, 3, 4]
torch.set_num_threads(2)
CATEGORY = {"normal": "normal",
    "back": "DoS", "land": "DoS", "neptune": "DoS", "pod": "DoS", "smurf": "DoS", "teardrop": "DoS",
    "ipsweep": "Probe", "nmap": "Probe", "portsweep": "Probe", "satan": "Probe",
    "ftp_write": "R2L", "guess_passwd": "R2L", "imap": "R2L", "multihop": "R2L", "phf": "R2L", "spy": "R2L",
    "warezclient": "R2L", "warezmaster": "R2L",
    "buffer_overflow": "U2R", "loadmodule": "U2R", "perl": "U2R", "rootkit": "U2R"}

# %% [markdown]
# ## Data and feature groups
#
# Binary task: attack (1) vs normal (0). Numeric features min-max scaled to [0,1] on the
# training part; protocol, service and flag one-hot (held fixed throughout). NSL-KDD
# feature groups (column indices): intrinsic 0-8, content 9-21, time-based traffic 22-30,
# host-based traffic 31-40. Following the feature taxonomy used by Lin et al. (IDSGAN), the
# groups that carry an attack's function are treated as fixed, so the robustness test only
# varies the non-functional groups of each category. This makes the stress test a
# conservative one: it is the subset of perturbations that could leave an attack's function
# intact, which is the realistic threat a detector faces.

# %%
d = pd.read_csv(DATA, header=None)
cat_all = d[41].map(CATEGORY).values
y_all = (cat_all != "normal").astype(np.float32)
NUM = [c for c in range(41) if c not in (1, 2, 3)]
GROUP = {c: ("intrinsic" if c <= 8 else "content" if c <= 21 else "time" if c <= 30 else "host") for c in NUM}
FUNCTIONAL = {"DoS": {"intrinsic", "time"}, "Probe": {"intrinsic", "time", "host"},
              "R2L": {"intrinsic", "content"}, "U2R": {"intrinsic", "content"}, "normal": set()}
onehots = [pd.get_dummies(d[c].astype(str)).values.astype(np.float32) for c in (1, 2, 3)]
NNUM = len(NUM)
RAW = d[NUM].values.astype(np.float64)

def encode(idx_tr):
    lo, hi = RAW[idx_tr].min(0), RAW[idx_tr].max(0); span = np.where(hi > lo, hi - lo, 1.0)
    Xn = np.clip((RAW - lo) / span, 0, 1).astype(np.float32)
    return np.hstack([Xn] + onehots).astype(np.float32)

def nonfunc_mask(cats):
    """1 where a numeric feature is OUTSIDE the functional groups of its record's category."""
    M = np.zeros((len(cats), NNUM), np.float32)
    for k, c in enumerate(NUM):
        for cat, fun in FUNCTIONAL.items():
            M[cats == cat, k] = 0.0 if GROUP[c] in fun else 1.0
    return M

# %% [markdown]
# ## Detector and the robustness stress test
#
# Detector: a small MLP (2 x 128, ReLU) classifying attack vs normal from the encoded
# vector. The stress test is a bounded L-inf perturbation of the numeric block only,
# found by projected gradient ascent on the detector's loss (the standard PGD evaluation
# of Madry et al.), confined to the non-functional features of each attack record and
# clipped back into [0,1]. One-hot blocks are never changed. The reported quantity is the
# detector's recall on attacks before and after the perturbation; lower after-recall means
# a less robust detector. "constrained" confines the perturbation to non-functional
# features; "unconstrained" allows all numeric features, as an upper bound on the drop.

# %%
D_IN = None
class Net(nn.Module):
    def __init__(self, d_in):
        super().__init__(); self.net = nn.Sequential(nn.Linear(d_in, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 1))
    def forward(self, x): return self.net(x).squeeze(1)

def pgd(model, X, y, mask_cols, eps, steps=20, alpha=None):
    """Projected gradient perturbation of the numeric block, restricted to mask_cols (per-row)."""
    alpha = alpha or eps / 4
    X0 = X.clone(); delta = torch.zeros_like(X)
    m = torch.zeros_like(X); m[:, :NNUM] = torch.tensor(mask_cols)
    bce = nn.BCEWithLogitsLoss()
    for _ in range(steps):
        delta.requires_grad_(True)
        loss = bce(model(X0 + delta * m), y)
        g, = torch.autograd.grad(loss, delta)
        delta = (delta + alpha * g.sign() * m).clamp(-eps, eps).detach()
        delta = (X0 + delta * m).clamp(0, 1) - X0            # keep inside [0,1]
    return (X0 + delta * m).detach()

def train_detector(Xtr, ytr, cats_tr, adv_eps=0.0, epochs=12, seed=0):
    torch.manual_seed(seed)
    model = Net(Xtr.shape[1]); opt = torch.optim.Adam(model.parameters(), 1e-3); bce = nn.BCEWithLogitsLoss()
    X = torch.tensor(Xtr); y = torch.tensor(ytr); B = 512
    mask = nonfunc_mask(cats_tr)
    for ep in range(epochs):
        perm = torch.randperm(len(X))
        for i in range(0, len(X), B):
            idx = perm[i:i + B]; xb, yb = X[idx], y[idx]
            if adv_eps > 0:                                   # adversarial training: fit on perturbed attacks
                att = yb > 0.5
                if att.any():
                    xb = xb.clone(); xb[att] = pgd(model, xb[att], yb[att], mask[idx.numpy()][att.numpy()], adv_eps, steps=7)
            opt.zero_grad(); bce(model(xb), yb).backward(); opt.step()
    return model

def recall_at(model, X, y, thr):
    with torch.no_grad(): p = torch.sigmoid(model(torch.tensor(X))).numpy()
    att = y > 0.5
    return float(np.mean(p[att] > thr))

def run(seed):
    idx = np.arange(len(d))
    tr, te = train_test_split(idx, test_size=0.3, stratify=cat_all, random_state=seed)
    X = encode(tr); cats = cat_all
    Xtr, ytr, ctr = X[tr], y_all[tr], cats[tr]
    Xte, yte, cte = X[te], y_all[te], cats[te]
    out = {}
    for name, eps_tr in [("standard", 0.0), ("adv_trained", 0.1)]:
        model = train_detector(Xtr, ytr, ctr, adv_eps=eps_tr, seed=seed)
        # threshold for 1% false positive rate on normal test records
        with torch.no_grad(): pN = torch.sigmoid(model(torch.tensor(Xte[yte < 0.5]))).numpy()
        thr = float(np.quantile(pN, 0.99))
        rec = {"clean": {}, "constrained": {}, "unconstrained": {}}
        for cat in ["DoS", "Probe", "R2L", "U2R"]:
            m = (cte == cat); Xc, yc = Xte[m], yte[m]
            rec["clean"][cat] = recall_at(model, Xc, yc, thr)
            for mode, mk in [("constrained", nonfunc_mask(cte[m])), ("unconstrained", np.ones((m.sum(), NNUM), np.float32))]:
                Xadv = pgd(model, torch.tensor(Xc), torch.tensor(yc), mk, eps=0.1, steps=20).numpy()
                rec[mode][cat] = recall_at(model, Xadv, yc, thr)
        out[name] = rec
    return out

RESULTS = {}
for seed in SEEDS:
    t0 = time.time(); RESULTS[str(seed)] = run(seed)
    print(f"seed {seed} done [{time.time()-t0:.0f}s]", flush=True)

CATS = ["DoS", "Probe", "R2L", "U2R"]
ms = lambda vals: f"{np.mean(vals):.3f} ± {np.std(vals, ddof=1):.3f}"
for name in ["standard", "adv_trained"]:
    print(f"\n{name}")
    for mode in ["clean", "constrained", "unconstrained"]:
        print(f"  {mode:14s} " + "  ".join(f"{c} {ms([RESULTS[str(s)][name][mode][c] for s in SEEDS])}" for c in CATS), flush=True)
Path(OUT / "ch15_results.json").write_text(json.dumps(RESULTS, indent=1))
print("done")
