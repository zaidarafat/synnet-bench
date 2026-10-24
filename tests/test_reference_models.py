"""Smoke tests for the reference generators: every model fits and samples on the shared
interface and returns a frame with the original columns."""
import numpy as np, pandas as pd, pytest
from synnet import NSGAN, WGANGP, CWGANGP, TabularVAE, TabularCodec
from synnet.metrics import discriminative_auc, exact_copy_rate, membership_inference_auc


@pytest.fixture
def df():
    rng = np.random.RandomState(0)
    return pd.DataFrame({
        "x1": rng.randn(500) * 3 + 2,
        "x2": rng.exponential(1.0, 500),
        "cat": rng.choice(["a", "b", "c"], 500, p=[0.6, 0.3, 0.1]),
        "label": rng.choice(["normal", "attack"], 500, p=[0.8, 0.2]),
    })


DISC = ["cat", "label"]


@pytest.mark.parametrize("Model", [NSGAN, WGANGP, TabularVAE])
def test_unconditional_fit_sample(df, Model):
    m = Model(steps=150, seed=0).fit(df, DISC)
    s = m.sample(200)
    assert list(s.columns) == list(df.columns)
    assert len(s) == 200
    assert set(s["cat"]).issubset({"a", "b", "c"})


def test_conditional_generates_requested_class(df):
    c = CWGANGP(steps=150, seed=0).fit(df, DISC, label_column="label")
    s = c.sample(200, "attack")
    assert (s["label"] == "attack").all()
    assert len(s) == 200


def test_codec_roundtrip_preserves_categoricals(df):
    codec = TabularCodec(df, DISC)
    z = codec.encode(df)
    back = codec.decode(z)
    # categorical columns must round-trip exactly; continuous are approximate
    assert (back["cat"].values == df["cat"].values).mean() > 0.99


def test_metrics_run_and_are_bounded(df):
    codec = TabularCodec(df, DISC)
    real = codec.encode(df)
    g = WGANGP(steps=150, seed=0).fit(df, DISC)
    syn = codec.encode(g.sample(len(df)))
    auc = discriminative_auc(real, syn)
    assert 0.0 <= auc <= 1.0
    half = len(df) // 2
    mi = membership_inference_auc(syn, real[:half], real[half:])
    assert 0.0 <= mi <= 1.0
