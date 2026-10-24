"""SynNet-Bench: a unified library for the book
"Generative Adversarial Networks for Network Intrusion Detection".

Every generator shares one interface:
    model.fit(df, discrete_columns); model.sample(n)
and every benchmark run follows one protocol: a leakage-safe split (synnet.data),
training on the training part only, and a unified fidelity/utility/privacy report
(synnet.metrics), over at least five fixed seeds.
"""
from .reference import TabularCodec, NSGAN, WGANGP, CWGANGP, TabularVAE, mlp
from . import data, metrics

__version__ = "1.0.0"
__all__ = ["TabularCodec", "NSGAN", "WGANGP", "CWGANGP", "TabularVAE", "mlp", "data", "metrics"]
