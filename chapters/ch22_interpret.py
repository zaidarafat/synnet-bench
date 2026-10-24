# %% [markdown]
# # Chapter 22 companion notebook: interpretability of synthetic-data pipelines
#
# Experiment 22.1: does augmentation change what the detector relies on? SHAP global
#   importances of an R2L detector trained on real-only, SMOTE-augmented and GAN-augmented
#   data are compared (Spearman rank correlation and top-5 feature overlap).
# Experiment 22.2: artifact detection. A real-vs-synthetic discriminator is trained per
#   generator; its ROC-AUC measures distinguishability (Chapter 12), and SHAP localises the
#   artifacts — the features that most betray the synthetic records.
#
#   python ch22_interpret.py       # caches per job, writes out/ch22_results.json
# Reuses the Chapter 13 notebook for data, SMOTE-NC and the conditional WGAN-GP.
# Requirements: torch, numpy, pandas, scikit-learn, shap.

# %%
from pathlib import Path
import json, sys, warnings
import numpy as np, pandas as pd
import shap
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
CH13 = Path("/home/claude/ch13/ch13_minority.py")
_src = CH13.read_text()
sys.argv = ["ch13_minority.py"]                       # MODE defaults to "all"; we only want the defs
exec(_src[:_src.rindex('if MODE in ("all", "worker")')])   # make_split, smote_r2l, cwgan_r2l, clf, CAT, X_all, CLASSES
OUT = Path("out"); OUT.mkdir(exist_ok=True); RUNS = OUT / "runs"; RUNS.mkdir(exist_ok=True)
SEEDS = [0, 1, 2, 3, 4]; N_R2L = 100; TARGET = 5000

# Canonical NSL-KDD feature names, in column order f0..f40
NAMES = ["duration","protocol_type","service","flag","src_bytes","dst_bytes","land","wrong_fragment",
 "urgent","hot","num_failed_logins","logged_in","num_compromised","root_shell","su_attempted","num_root",
 "num_file_creations","num_shells","num_access_files","num_outbound_cmds","is_host_login","is_guest_login",
 "count","srv_count","serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
 "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count","dst_host_same_srv_rate",
 "dst_host_diff_srv_rate","dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_serror_rate",
 "dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate"]
NMAP = {f"f{i}": NAMES[i] for i in range(41)}

def r2l_shap(model, Xeval):
    """Mean |SHAP| per feature for the R2L class, using the fast tree explainer."""
    expl = shap.TreeExplainer(model)
    sv = expl.shap_values(Xeval)
    sv = np.asarray(sv)
    # locate the R2L class axis
    j = list(model.classes_).index("R2L")
    if sv.ndim == 3:                      # (n, features, classes) or (classes, n, features)
        arr = sv[:, :, j] if sv.shape[-1] == len(model.classes_) else sv[j]
    else:
        arr = sv
    return np.abs(arr).mean(0)

# %% [markdown]
# ## Experiment 22.1: feature reliance under augmentation
#
# R2L reduced to 100 training records (leakage-free split). Three detectors: real-only,
# SMOTE-augmented to 5,000, GAN-augmented to 5,000. For each, SHAP mean-|value| importances
# for the R2L class are computed on the same 1,500-record test sample, and the three rankings
# are compared by Spearman correlation and top-5 overlap.

# %%
def exp1_seed(seed):
    p = RUNS / f"e1_{seed}.json"
    if p.exists(): return json.loads(p.read_text())
    Xtr, ytr, Xte, yte = make_split(seed, N_R2L); m = TARGET - int((ytr == "R2L").sum())
    rng = np.random.RandomState(seed); samp = rng.choice(len(Xte), min(1500, len(Xte)), replace=False)
    Xev = Xte.iloc[samp].reset_index(drop=True)
    imp = {}
    # real only
    imp["real"] = r2l_shap(clf(seed).fit(Xtr, ytr), Xev)
    # SMOTE augmented
    S = smote_r2l(Xtr, ytr, m, seed); Xa = pd.concat([Xtr, S], ignore_index=True); ya = np.r_[ytr, ["R2L"] * len(S)]
    imp["smote"] = r2l_shap(clf(seed).fit(Xa, ya), Xev)
    # GAN augmented
    G = cwgan_r2l(Xtr, ytr, m, seed); Xa = pd.concat([Xtr, G], ignore_index=True); ya = np.r_[ytr, ["R2L"] * len(G)]
    imp["gan"] = r2l_shap(clf(seed).fit(Xa, ya), Xev)
    res = {k: v.tolist() for k, v in imp.items()}
    p.write_text(json.dumps(res)); print(f"  exp1 seed {seed} done", flush=True)
    return res

def topk(imp, k=5): return [list(X_all.columns)[i] for i in np.argsort(imp)[::-1][:k]]

E1 = [exp1_seed(s) for s in SEEDS]
cols = list(X_all.columns)
mean_imp = {k: np.mean([np.array(e[k]) for e in E1], 0) for k in ["real", "smote", "gan"]}
sp = {"smote": [], "gan": []}
ov = {"smote": [], "gan": []}
for e in E1:
    for k in ["smote", "gan"]:
        sp[k].append(stats.spearmanr(e["real"], e[k]).correlation)
        ov[k].append(len(set(topk(np.array(e["real"]))) & set(topk(np.array(e[k])))))
print("Experiment 22.1 — feature reliance vs real-only detector (mean over seeds):")
for k in ["smote", "gan"]:
    print(f"  {k:5s}: Spearman {np.mean(sp[k]):.3f} ± {np.std(sp[k], ddof=1):.3f}   top-5 overlap {np.mean(ov[k]):.1f}/5", flush=True)
print("  real-only top-5 :", [NMAP[c] for c in topk(mean_imp['real'])])
print("  smote-aug top-5 :", [NMAP[c] for c in topk(mean_imp['smote'])])
print("  gan-aug   top-5 :", [NMAP[c] for c in topk(mean_imp['gan'])])

# %% [markdown]
# ## Experiment 22.2: artifact detection via a real-vs-synthetic discriminator
#
# For each generator (SMOTE, GAN), a classifier is trained to tell real R2L records from the
# generator's synthetic R2L records (balanced, stratified 70/30). Its ROC-AUC on held-out
# records is the discriminative score of Chapter 12: 0.5 means indistinguishable, 1.0 means
# trivially separable. SHAP on the discriminator ranks the features that most betray the
# synthetic data — the artifacts the generator leaves.

# %%
def exp2_seed(seed):
    p = RUNS / f"e2_{seed}.json"
    if p.exists(): return json.loads(p.read_text())
    Xtr, ytr, _, _ = make_split(seed, N_R2L); real = Xtr[ytr == "R2L"].reset_index(drop=True)
    m = 2000
    gens = {"smote": smote_r2l(Xtr, ytr, m, seed), "gan": cwgan_r2l(Xtr, ytr, m, seed)}
    out = {}
    for name, syn in gens.items():
        n = min(len(real), len(syn))
        rng = np.random.RandomState(seed)
        R = real.iloc[rng.choice(len(real), n, replace=len(real) < n)].reset_index(drop=True)
        Sg = syn.iloc[rng.choice(len(syn), n, replace=False)].reset_index(drop=True)
        X = pd.concat([R, Sg], ignore_index=True); y = np.r_[np.zeros(n), np.ones(n)]
        idx = rng.permutation(len(X)); cut = int(0.7 * len(X)); tr, te = idx[:cut], idx[cut:]
        disc = HistGradientBoostingClassifier(random_state=seed, categorical_features=[1, 2, 3]).fit(X.iloc[tr], y[tr])
        auc = roc_auc_score(y[te], disc.predict_proba(X.iloc[te])[:, 1])
        expl = shap.TreeExplainer(disc); sv = np.asarray(expl.shap_values(X.iloc[te]))
        arr = sv[:, :, 1] if sv.ndim == 3 and sv.shape[-1] == 2 else (sv[1] if sv.ndim == 3 else sv)
        out[name] = {"auc": float(auc), "imp": np.abs(arr).mean(0).tolist()}
    p.write_text(json.dumps(out)); print(f"  exp2 seed {seed} done", flush=True)
    return out

E2 = [exp2_seed(s) for s in SEEDS]
print("\nExperiment 22.2 — real-vs-synthetic discriminator:")
res2 = {}
for name in ["smote", "gan"]:
    aucs = [e[name]["auc"] for e in E2]
    mimp = np.mean([np.array(e[name]["imp"]) for e in E2], 0)
    tops = [NMAP[c] for c in topk(mimp)]
    res2[name] = {"auc_mean": float(np.mean(aucs)), "auc_sd": float(np.std(aucs, ddof=1)), "top_artifacts": tops}
    print(f"  {name:5s}: discriminative ROC-AUC {np.mean(aucs):.3f} ± {np.std(aucs, ddof=1):.3f}   top artifact features {tops}", flush=True)

RESULTS = {"exp22_1": {"spearman": {k: [float(np.mean(sp[k])), float(np.std(sp[k], ddof=1))] for k in sp},
                        "top5_overlap": {k: float(np.mean(ov[k])) for k in ov},
                        "top5": {k: [NMAP[c] for c in topk(mean_imp[k])] for k in mean_imp}},
           "exp22_2": res2}
Path(OUT / "ch22_results.json").write_text(json.dumps(RESULTS, indent=1))
print("done")
