# %% [markdown]
# # Chapter 4 companion (Experiment 4.3): the mechanism of rare-mode suppression
#
# Section 4.6 found that the discriminator regularisers which make common-mode coverage
# reliable also suppress the rare mode most, and offered a hypothesis: a smoother
# discriminator responds less to a mode that contributes only two or three records to a
# batch of 256. This notebook tests that hypothesis two ways.
#   Part A  sweeps the regularisation STRENGTH (R1 gamma, gradient-penalty lambda) on the
#           rare-mode ring and measures, per strength, the rare-mode share AND a direct probe
#           of the discriminator's response: the mean input-gradient norm of D at the rare
#           mode and at the common modes. If suppression is caused by a flatter discriminator,
#           rare-mode share and the discriminator's response at the rare mode should both fall
#           as regularisation strengthens.
#   Part B  fixes WGAN-GP (lambda = 10) and raises the rare mode's TRUE share, so it
#           contributes more records per batch. If the mechanism is the per-batch signal,
#           raising the share should recover the mode.
#   python ch04_ablation.py worker 0 2 ; python ch04_ablation.py worker 1 2 ; python ch04_ablation.py report
# Requirements: torch, numpy.

# %%
from pathlib import Path
import json, sys, warnings
import numpy as np, torch, torch.nn as nn
warnings.filterwarnings("ignore")
OUT = Path("out"); OUT.mkdir(exist_ok=True); RUNS = OUT / "abl_runs"; RUNS.mkdir(exist_ok=True)
torch.set_num_threads(1)
SEEDS = [0, 1, 2, 3, 4]
MODE = sys.argv[1] if len(sys.argv) > 1 else "all"
K, R, STD = 8, 2.0, 0.05
CENTRES = np.array([[R*np.cos(2*np.pi*k/K), R*np.sin(2*np.pi*k/K)] for k in range(K)])
STEPS = 6000; B = 256

def set_seed(s): torch.manual_seed(s); np.random.seed(s)
def mlp(di, do):
    L, d = [], di
    for _ in range(3): L += [nn.Linear(d, 128), nn.LeakyReLU(0.2)]; d = 128
    return nn.Sequential(*L, nn.Linear(d, do))
def sample_ring(n, w, rng):
    idx = rng.choice(K, size=n, p=w)
    return torch.tensor(CENTRES[idx] + STD*rng.randn(n, 2), dtype=torch.float32)
def rare_share(x):
    d = np.linalg.norm(x[:, None, :] - CENTRES[None], axis=2)
    near = d.min(1) < 3*STD; nearest = d.argmin(1)
    return float(np.mean(near & (nearest == 0)))        # mode 0 is the rare mode
def grad_penalty(D, xr, xg):
    a = torch.rand(len(xr), 1); xi = (a*xr + (1-a)*xg).requires_grad_(True)
    g = torch.autograd.grad(D(xi).sum(), xi, create_graph=True)[0]
    return ((g.norm(dim=1) - 1)**2).mean()
def r1_penalty(D, xr):
    xr = xr.requires_grad_(True)
    g = torch.autograd.grad(D(xr).sum(), xr, create_graph=True)[0]
    return (g.norm(dim=1)**2).mean()

def d_response(D, rng):
    """Mean input-gradient norm of D at real rare-mode points and at common-mode points."""
    def gnorm(points):
        x = torch.tensor(points, dtype=torch.float32, requires_grad=True)
        g = torch.autograd.grad(D(x).sum(), x)[0]
        return float(g.norm(dim=1).mean())
    rare = CENTRES[0] + STD*rng.randn(500, 2)
    common = CENTRES[rng.choice(range(1, K), 500)] + STD*rng.randn(500, 2)
    return gnorm(rare), gnorm(common)

def train(reg, strength, p_rare, seed):
    set_seed(seed); rng = np.random.RandomState(seed)
    w = np.array([p_rare] + [(1-p_rare)/7]*7)
    G, D = mlp(2, 2), mlp(2, 1)
    wgan = (reg == "gp"); betas = (0.0, 0.9) if wgan else (0.5, 0.999)
    og = torch.optim.Adam(G.parameters(), 1e-4, betas=betas)
    od = torch.optim.Adam(D.parameters(), 1e-4, betas=betas)
    bce = nn.BCEWithLogitsLoss(); n_critic = 5 if wgan else 1
    for _ in range(STEPS):
        for _ in range(n_critic):
            xr = sample_ring(B, w, rng)
            with torch.no_grad(): xg = G(torch.randn(B, 2))
            if wgan:
                ld = D(xg).mean() - D(xr).mean() + strength*grad_penalty(D, xr, xg)
            else:
                ld = bce(D(xr), torch.ones(B,1)) + bce(D(xg), torch.zeros(B,1))
                if strength > 0: ld = ld + 0.5*strength*r1_penalty(D, xr)
            od.zero_grad(); ld.backward(); od.step()
        xg = G(torch.randn(B, 2)); out = D(xg)
        lg = -out.mean() if wgan else bce(out, torch.ones(B,1))
        og.zero_grad(); lg.backward(); og.step()
    with torch.no_grad():
        gen = G(torch.randn(5000, 2)).numpy()
    gr, gc = d_response(D, rng)
    return {"rare_share": rare_share(gen), "dgrad_rare": gr, "dgrad_common": gc}

# %%
JOBS = [("r1", g, 0.01) for g in [0.0, 1.0, 10.0, 100.0]] + \
       [("gp", l, 0.01) for l in [1.0, 10.0, 100.0]] + \
       [("gp", 10.0, p) for p in [0.05, 0.20]]
def jp(j, s): return RUNS / f"{j[0]}_{j[1]}_{j[2]}_{s}.json"
def do(j, s):
    p = jp(j, s)
    if p.exists(): return
    r = train(j[0], j[1], j[2], s); p.write_text(json.dumps(r))
    print(f"job {j} seed {s} done  rare {r['rare_share']*100:.2f}%", flush=True)

if MODE in ("all", "worker"):
    k, n = (int(sys.argv[2]), int(sys.argv[3])) if MODE == "worker" else (0, 1)
    flat = [(j, s) for j in JOBS for s in SEEDS]
    for i, (j, s) in enumerate(flat):
        if i % n == k: do(j, s)
    if MODE == "worker": sys.exit(0)

# %% report
missing = [(j, s) for j in JOBS for s in SEEDS if not jp(j, s).exists()]
if missing: raise SystemExit(f"{len(missing)} jobs unfinished")
def agg(j):
    rs = [json.loads(jp(j, s).read_text()) for s in SEEDS]
    m = lambda k: (float(np.mean([r[k] for r in rs])), float(np.std([r[k] for r in rs], ddof=1)))
    return {"rare_share": m("rare_share"), "dgrad_rare": m("dgrad_rare"), "dgrad_common": m("dgrad_common")}
RES = {f"{j[0]}|{j[1]}|{j[2]}": agg(j) for j in JOBS}
print("\n=== Part A: regularisation-strength sweep (rare true share 1%) ===")
for j in [j for j in JOBS if j[2] == 0.01]:
    a = agg(j); nm = f"{j[0]} strength {j[1]:g}"
    print(f"  {nm:18s} rare-share {100*a['rare_share'][0]:5.2f}±{100*a['rare_share'][1]:.2f}%   "
          f"|grad D| rare {a['dgrad_rare'][0]:.3f}  common {a['dgrad_common'][0]:.3f}", flush=True)
print("\n=== Part B: WGAN-GP (lambda 10), raising the rare mode's true share ===")
for p in [0.01, 0.05, 0.20]:
    a = agg(("gp", 10.0, p))
    print(f"  true share {int(p*100):2d}%   recovered rare-share {100*a['rare_share'][0]:5.2f}±{100*a['rare_share'][1]:.2f}%", flush=True)
Path(OUT / "ch04_ablation_results.json").write_text(json.dumps(RES, indent=1))
print("done")
