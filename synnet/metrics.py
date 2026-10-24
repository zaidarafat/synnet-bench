"""Reference metrics for the SynNet-Bench unified report: fidelity, utility and privacy.

All metrics operate on encoded numeric arrays (use synnet.reference.TabularCodec to
encode), except the label-aware utility metric which takes frames and a classifier.
Each is the reference implementation the book's chapters use; see the cited chapter.
"""
from __future__ import annotations
import numpy as np
from scipy import stats
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, recall_score, f1_score


# ---------------------------------------------------------------- fidelity (Chapter 12)
def mmd2(X, Y, bandwidth="median"):
    """Squared maximum mean discrepancy with a Gaussian kernel (median-heuristic
    bandwidth by default). 0 means the samples are indistinguishable in the RKHS."""
    X, Y = np.asarray(X, float), np.asarray(Y, float)
    Z = np.vstack([X, Y])
    if bandwidth == "median":
        s = NearestNeighbors(n_neighbors=2).fit(Z)  # cheap scale estimate
        d = s.kneighbors(Z[: min(1000, len(Z))])[0][:, 1]
        sigma = np.median(d[d > 0]) if np.any(d > 0) else 1.0
    else:
        sigma = float(bandwidth)
    g = 1.0 / (2 * sigma ** 2)
    def k(A, B):
        aa = (A * A).sum(1)[:, None]; bb = (B * B).sum(1)[None, :]
        return np.exp(-g * (aa + bb - 2 * A @ B.T))
    m, n = len(X), len(Y)
    kxx = (k(X, X).sum() - np.trace(k(X, X))) / (m * (m - 1))
    kyy = (k(Y, Y).sum() - np.trace(k(Y, Y))) / (n * (n - 1))
    kxy = k(X, Y).mean()
    return float(kxx + kyy - 2 * kxy)


def ks_mean(Xreal, Xsyn):
    """Mean Kolmogorov-Smirnov statistic over columns (0 identical, 1 disjoint)."""
    Xreal, Xsyn = np.asarray(Xreal, float), np.asarray(Xsyn, float)
    return float(np.mean([stats.ks_2samp(Xreal[:, j], Xsyn[:, j]).statistic for j in range(Xreal.shape[1])]))


def discriminative_auc(real, synth, seed=0):
    """ROC-AUC of a classifier separating real from synthetic (Chapter 12). 0.5 is
    indistinguishable; 1.0 is trivially separable."""
    real, synth = np.asarray(real, float), np.asarray(synth, float)
    n = min(len(real), len(synth)); rng = np.random.RandomState(seed)
    R = real[rng.choice(len(real), n, replace=False)]; S = synth[rng.choice(len(synth), n, replace=False)]
    X = np.vstack([R, S]); y = np.r_[np.zeros(n), np.ones(n)]
    idx = rng.permutation(len(X)); cut = int(0.7 * len(X))
    clf = HistGradientBoostingClassifier(random_state=seed).fit(X[idx[:cut]], y[idx[:cut]])
    return float(roc_auc_score(y[idx[cut:]], clf.predict_proba(X[idx[cut:]])[:, 1]))


# ---------------------------------------------------------------- privacy (Chapters 11, 20)
def exact_copy_rate(synth_df, train_df):
    """Fraction of synthetic records that exactly copy a training record."""
    keys = set(train_df.astype(str).agg("|".join, axis=1))
    return float(np.mean([k in keys for k in synth_df.astype(str).agg("|".join, axis=1)]))


def dcr(synth, train):
    """Median distance to the closest real record (Chapter 11). Small means copying."""
    nn = NearestNeighbors(n_neighbors=1).fit(np.asarray(train, float))
    d = nn.kneighbors(np.asarray(synth, float))[0].ravel()
    return float(np.median(d))


def membership_inference_auc(synth, members, nonmembers):
    """Distance-based membership-inference attack AUC (Chapter 20). A record is scored by
    -distance to the nearest synthetic record; AUC separates members from non-members.
    0.5 means no leakage; 1.0 means perfect membership inference."""
    nn = NearestNeighbors(n_neighbors=1).fit(np.asarray(synth, float))
    dm = nn.kneighbors(np.asarray(members, float))[0].ravel()
    dn = nn.kneighbors(np.asarray(nonmembers, float))[0].ravel()
    y = np.r_[np.ones(len(dm)), np.zeros(len(dn))]
    return float(roc_auc_score(y, -np.r_[dm, dn]))


# ---------------------------------------------------------------- utility (Chapter 12)
def tstr_recall(train_df, test_df, label_column, positive, discrete_idx=None, seed=0):
    """Train-on-synthetic, test-on-real: train a classifier on `train_df` (which may be
    synthetic or augmented) and report recall of `positive` on real `test_df`."""
    Xtr = train_df.drop(columns=[label_column]); ytr = train_df[label_column].astype(str)
    Xte = test_df.drop(columns=[label_column]); yte = test_df[label_column].astype(str)
    clf = HistGradientBoostingClassifier(random_state=seed,
            categorical_features=discrete_idx).fit(Xtr, ytr)
    p = clf.predict(Xte)
    return float(recall_score(yte == str(positive), p == str(positive)))


def macro_f1(clf, X, y, labels):
    return float(f1_score(y, clf.predict(X), labels=labels, average="macro", zero_division=0))
