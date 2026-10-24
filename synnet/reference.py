"""
Appendix B — reference PyTorch implementations for the models of this book.

A single, minimal, unified interface. Every generator exposes:
    model.fit(df, discrete_columns)      # df is a pandas DataFrame of mixed types
    model.sample(n)                      # returns a DataFrame with the same columns

The code favours clarity over speed; it is the reference the book's experiments
follow, not an optimised library. CPU by default. Requirements: torch, numpy,
pandas, scikit-learn.

Models:
    TabularCodec   - mixed-type encoding (quantile-normal continuous + one-hot categorical)
    NSGAN          - non-saturating GAN (Chapter 3)
    WGANGP         - Wasserstein GAN with gradient penalty (Chapter 4)
    CWGANGP        - class-conditional WGAN-GP (Chapter 13)
    TabularVAE     - a small variational autoencoder (Chapter 11)

This module releases no trained model and no attack generator; it is code only.
"""
from __future__ import annotations
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.preprocessing import QuantileTransformer

torch.set_num_threads(1)


# ---------------------------------------------------------------- encoding
class TabularCodec:
    """Encode mixed tabular data to a real vector and back.

    Continuous columns -> Gaussian quantile transform (robust to heavy tails, Chapter 6).
    Categorical columns -> one-hot blocks.
    """
    def __init__(self, df: pd.DataFrame, discrete_columns):
        self.columns = list(df.columns)
        self.discrete = list(discrete_columns)
        self.continuous = [c for c in self.columns if c not in self.discrete]
        self.q = QuantileTransformer(n_quantiles=min(1000, len(df)),
                                     output_distribution="normal", random_state=0)
        if self.continuous:
            self.q.fit(df[self.continuous].astype(float))
        self.levels = {c: sorted(df[c].astype(str).unique()) for c in self.discrete}
        self.blocks = [len(self.continuous)] + [len(self.levels[c]) for c in self.discrete]
        self.dim = sum(self.blocks)

    def encode(self, df: pd.DataFrame) -> np.ndarray:
        parts = []
        if self.continuous:
            parts.append(self.q.transform(df[self.continuous].astype(float)))
        for c in self.discrete:
            codes = pd.Categorical(df[c].astype(str), categories=self.levels[c]).codes
            parts.append(np.eye(len(self.levels[c]))[codes])
        return np.hstack(parts).astype(np.float32) if parts else np.zeros((len(df), 0), np.float32)

    def decode(self, z: np.ndarray) -> pd.DataFrame:
        out, k = {}, 0
        if self.continuous:
            cont = self.q.inverse_transform(np.clip(z[:, :len(self.continuous)], -5.2, 5.2))
            for i, c in enumerate(self.continuous):
                out[c] = cont[:, i]
            k = len(self.continuous)
        for c in self.discrete:
            b = len(self.levels[c])
            idx = z[:, k:k + b].argmax(1); k += b
            out[c] = [self.levels[c][j] for j in idx]
        return pd.DataFrame(out)[self.columns]

    def activate(self, h: torch.Tensor, tau: float = 0.2) -> torch.Tensor:
        """Output activation for a generator: identity on continuous, Gumbel-softmax per
        categorical block (differentiable discrete sampling, Chapter 6)."""
        out, k = [], 0
        if self.continuous:
            out.append(h[:, :self.blocks[0]]); k = self.blocks[0]
        for b in self.blocks[1:]:
            out.append(F.gumbel_softmax(h[:, k:k + b], tau=tau)); k += b
        return torch.cat(out, 1) if out else h


def mlp(i, o, width=256, depth=3):
    layers, d = [], i
    for _ in range(depth):
        layers += [nn.Linear(d, width), nn.LeakyReLU(0.2)]; d = width
    return nn.Sequential(*layers, nn.Linear(d, o))


# ---------------------------------------------------------------- NS-GAN
class NSGAN:
    """Non-saturating GAN (Chapter 3). Generator minimises -log D(G(z))."""
    def __init__(self, latent=32, steps=2000, batch=256, lr=2e-4, seed=0):
        self.latent, self.steps, self.batch, self.lr, self.seed = latent, steps, batch, lr, seed

    def fit(self, df, discrete_columns):
        torch.manual_seed(self.seed); np.random.seed(self.seed)
        self.codec = TabularCodec(df, discrete_columns)
        X = torch.tensor(self.codec.encode(df))
        self.G = mlp(self.latent, self.codec.dim); self.D = mlp(self.codec.dim, 1)
        og = torch.optim.Adam(self.G.parameters(), self.lr, betas=(0.5, 0.9))
        od = torch.optim.Adam(self.D.parameters(), self.lr, betas=(0.5, 0.9))
        bce = nn.BCEWithLogitsLoss()
        for _ in range(self.steps):
            i = torch.randint(0, len(X), (self.batch,)); xr = X[i]
            z = torch.randn(self.batch, self.latent)
            with torch.no_grad(): xg = self.codec.activate(self.G(z))
            ld = bce(self.D(xr), torch.ones(self.batch, 1)) + bce(self.D(xg), torch.zeros(self.batch, 1))
            od.zero_grad(); ld.backward(); od.step()
            z = torch.randn(self.batch, self.latent)
            xg = self.codec.activate(self.G(z))
            lg = bce(self.D(xg), torch.ones(self.batch, 1))   # non-saturating
            og.zero_grad(); lg.backward(); og.step()
        return self

    def sample(self, n):
        with torch.no_grad():
            z = torch.randn(n, self.latent)
            return self.codec.decode(self.codec.activate(self.G(z)).numpy())


# ---------------------------------------------------------------- WGAN-GP
class WGANGP:
    """Wasserstein GAN with gradient penalty (Chapter 4). lambda=10, 5 critic steps."""
    def __init__(self, latent=32, steps=2000, batch=256, lr=1e-4, n_critic=5, gp=10.0, seed=0):
        self.latent, self.steps, self.batch, self.lr = latent, steps, batch, lr
        self.n_critic, self.gp, self.seed = n_critic, gp, seed

    def _penalty(self, xr, xg):
        a = torch.rand(len(xr), 1)
        xi = (a * xr + (1 - a) * xg).requires_grad_(True)
        g = torch.autograd.grad(self.C(xi).sum(), xi, create_graph=True)[0]
        return self.gp * ((g.norm(dim=1) - 1) ** 2).mean()

    def fit(self, df, discrete_columns):
        torch.manual_seed(self.seed); np.random.seed(self.seed)
        self.codec = TabularCodec(df, discrete_columns)
        X = torch.tensor(self.codec.encode(df))
        self.G = mlp(self.latent, self.codec.dim); self.C = mlp(self.codec.dim, 1)
        og = torch.optim.Adam(self.G.parameters(), self.lr, betas=(0.0, 0.9))
        oc = torch.optim.Adam(self.C.parameters(), self.lr, betas=(0.0, 0.9))
        for _ in range(self.steps):
            for _ in range(self.n_critic):
                i = torch.randint(0, len(X), (self.batch,)); xr = X[i]
                with torch.no_grad(): xg = self.codec.activate(self.G(torch.randn(self.batch, self.latent)))
                lc = self.C(xg).mean() - self.C(xr).mean() + self._penalty(xr, xg)
                oc.zero_grad(); lc.backward(); oc.step()
            xg = self.codec.activate(self.G(torch.randn(self.batch, self.latent)))
            lg = -self.C(xg).mean()
            og.zero_grad(); lg.backward(); og.step()
        return self

    def sample(self, n):
        with torch.no_grad():
            return self.codec.decode(self.codec.activate(self.G(torch.randn(n, self.latent))).numpy())


# ---------------------------------------------------------------- conditional WGAN-GP
class CWGANGP:
    """Class-conditional WGAN-GP (Chapter 13). Labels concatenated to both networks."""
    def __init__(self, latent=32, steps=2000, batch=256, lr=1e-4, n_critic=5, gp=10.0, seed=0):
        self.latent, self.steps, self.batch, self.lr = latent, steps, batch, lr
        self.n_critic, self.gp, self.seed = n_critic, gp, seed

    def _penalty(self, xr, xg, y):
        a = torch.rand(len(xr), 1)
        xi = (a * xr + (1 - a) * xg).requires_grad_(True)
        g = torch.autograd.grad(self.C(torch.cat([xi, y], 1)).sum(), xi, create_graph=True)[0]
        return self.gp * ((g.norm(dim=1) - 1) ** 2).mean()

    def fit(self, df, discrete_columns, label_column):
        torch.manual_seed(self.seed); np.random.seed(self.seed)
        self.label = label_column
        self.classes = sorted(df[label_column].astype(str).unique()); K = len(self.classes)
        feats = [c for c in df.columns if c != label_column]
        self.codec = TabularCodec(df[feats], [c for c in discrete_columns if c != label_column])
        X = torch.tensor(self.codec.encode(df[feats]))
        Y = torch.tensor(np.eye(K)[[self.classes.index(v) for v in df[label_column].astype(str)]], dtype=torch.float32)
        self.G = mlp(self.latent + K, self.codec.dim); self.C = mlp(self.codec.dim + K, 1); self.K = K
        og = torch.optim.Adam(self.G.parameters(), self.lr, betas=(0.0, 0.9))
        oc = torch.optim.Adam(self.C.parameters(), self.lr, betas=(0.0, 0.9))
        for _ in range(self.steps):
            for _ in range(self.n_critic):
                i = torch.randint(0, len(X), (self.batch,)); xr, yr = X[i], Y[i]
                with torch.no_grad():
                    xg = self.codec.activate(self.G(torch.cat([torch.randn(self.batch, self.latent), yr], 1)))
                lc = self.C(torch.cat([xg, yr], 1)).mean() - self.C(torch.cat([xr, yr], 1)).mean() + self._penalty(xr, xg, yr)
                oc.zero_grad(); lc.backward(); oc.step()
            i = torch.randint(0, len(X), (self.batch,)); yr = Y[i]
            xg = self.codec.activate(self.G(torch.cat([torch.randn(self.batch, self.latent), yr], 1)))
            lg = -self.C(torch.cat([xg, yr], 1)).mean()
            og.zero_grad(); lg.backward(); og.step()
        return self

    def sample(self, n, label):
        y = torch.tensor(np.tile(np.eye(self.K)[self.classes.index(str(label))], (n, 1)), dtype=torch.float32)
        with torch.no_grad():
            df = self.codec.decode(self.codec.activate(self.G(torch.cat([torch.randn(n, self.latent), y], 1))).numpy())
        df[self.label] = label
        return df


# ---------------------------------------------------------------- Tabular VAE
class TabularVAE:
    """A small variational autoencoder for mixed tabular data (Chapter 11)."""
    def __init__(self, latent=16, steps=2000, batch=256, lr=1e-3, beta=1.0, seed=0):
        self.latent, self.steps, self.batch, self.lr, self.beta, self.seed = latent, steps, batch, lr, beta, seed

    def fit(self, df, discrete_columns):
        torch.manual_seed(self.seed); np.random.seed(self.seed)
        self.codec = TabularCodec(df, discrete_columns)
        X = torch.tensor(self.codec.encode(df)); d = self.codec.dim
        self.enc = mlp(d, 2 * self.latent); self.dec = mlp(self.latent, d)
        opt = torch.optim.Adam(list(self.enc.parameters()) + list(self.dec.parameters()), self.lr)
        for _ in range(self.steps):
            i = torch.randint(0, len(X), (self.batch,)); x = X[i]
            h = self.enc(x); mu, logvar = h[:, :self.latent], h[:, self.latent:]
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
            xr = self.codec.activate(self.dec(z))
            recon = F.mse_loss(xr, x)
            kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon + self.beta * kld
            opt.zero_grad(); loss.backward(); opt.step()
        return self

    def sample(self, n):
        with torch.no_grad():
            return self.codec.decode(self.codec.activate(self.dec(torch.randn(n, self.latent))).numpy())


# ---------------------------------------------------------------- smoke test
if __name__ == "__main__":
    rng = np.random.RandomState(0)
    df = pd.DataFrame({
        "x1": rng.randn(600) * 3 + 2,
        "x2": rng.exponential(1.0, 600),
        "cat": rng.choice(["a", "b", "c"], 600, p=[0.6, 0.3, 0.1]),
        "label": rng.choice(["normal", "attack"], 600, p=[0.8, 0.2]),
    })
    disc = ["cat", "label"]
    for name, mk in [("NSGAN", lambda: NSGAN(steps=300).fit(df, disc)),
                     ("WGANGP", lambda: WGANGP(steps=300).fit(df, disc)),
                     ("TabularVAE", lambda: TabularVAE(steps=300).fit(df, disc))]:
        s = mk().sample(200)
        assert list(s.columns) == list(df.columns) and len(s) == 200
        print(f"{name:11s} OK  sample shape {s.shape}  cat levels {sorted(s['cat'].unique())}")
    c = CWGANGP(steps=300).fit(df, disc, label_column="label")
    s = c.sample(200, "attack")
    assert (s["label"] == "attack").all() and len(s) == 200
    print(f"CWGANGP     OK  sampled 200 'attack' rows")
    print("smoke test passed")
