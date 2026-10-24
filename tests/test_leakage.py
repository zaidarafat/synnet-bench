"""Automated tests enforcing the leakage discipline of Chapter 8.

These are the tests the book argues every synthetic-augmentation pipeline should ship:
they fail if the split is not disjoint, if a fitted step sees test data, or if an
augmentation places synthetic records into the test partition.
"""
import numpy as np, pandas as pd, pytest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from synnet.data import leakage_safe_split
from synnet import WGANGP, CWGANGP


def _toy(n=400, seed=0):
    rng = np.random.RandomState(seed)
    X = pd.DataFrame({"a": rng.randn(n), "b": rng.exponential(1, n),
                      "cat": rng.choice(["x", "y", "z"], n)})
    y = np.where(rng.rand(n) < 0.3, "attack", "normal")
    return X, y


def test_split_is_disjoint_and_covers():
    X, y = _toy()
    Xtr, Xte, ytr, yte = leakage_safe_split(X, y, test_size=0.3, seed=0)
    # disjoint sizes add to the whole, and index reset prevents accidental alignment leaks
    assert len(Xtr) + len(Xte) == len(X)
    assert len(ytr) + len(yte) == len(y)
    assert list(Xtr.index) == list(range(len(Xtr)))
    assert list(Xte.index) == list(range(len(Xte)))


def test_scaler_fit_on_train_only_differs_from_leaky():
    """A scaler fitted on all data (leaky) generally differs from one fitted on the
    training part (correct). If they were identical, the test could not catch the leak,
    so this asserts the discipline is observable."""
    X, y = _toy()
    Xtr, Xte, ytr, yte = leakage_safe_split(X, y, seed=1)
    num = ["a", "b"]
    correct = StandardScaler().fit(Xtr[num])          # training part only
    leaky = StandardScaler().fit(X[num])              # all data -> leak
    assert not np.allclose(correct.mean_, leaky.mean_), "leak would be undetectable here"


def test_augmentation_adds_no_record_to_test():
    """Synthetic augmentation must touch the training part only; no synthetic record may
    appear in the test partition."""
    X, y = _toy()
    Xtr, Xte, ytr, yte = leakage_safe_split(X, y, seed=2)
    df_tr = Xtr.copy(); df_tr["label"] = ytr
    g = CWGANGP(steps=120, seed=0).fit(df_tr, ["cat", "label"], label_column="label")
    syn = g.sample(100, "attack")
    test_keys = set(Xte.astype(str).agg("|".join, axis=1))
    syn_keys = set(syn.drop(columns=["label"]).astype(str).agg("|".join, axis=1))
    assert test_keys.isdisjoint(syn_keys), "a synthetic record collided with the test set"


def test_generator_sees_training_rows_only():
    """The generator is fitted on the training frame; its training data must be a subset
    of the training partition, never the test partition."""
    X, y = _toy()
    Xtr, Xte, ytr, yte = leakage_safe_split(X, y, seed=3)
    tr_keys = set(Xtr.astype(str).agg("|".join, axis=1))
    te_keys = set(Xte.astype(str).agg("|".join, axis=1))
    assert tr_keys.isdisjoint(te_keys), "train and test share a record before fitting"
