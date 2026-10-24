# SynNet-Bench

The unified code and benchmark for the book **_Generative Adversarial Networks for Network Intrusion Detection: Synthetic Data, Rigorous Evaluation, and Trust_** (Z. Arafat, Z. A. Abdulazeez).

One library, one interface, one evaluation protocol. Every generator in the book is built on the same `fit`/`sample` interface, and every benchmark run follows the same leakage-safe protocol so that results are comparable and reproducible. The companion scripts reproduce every number in the book.

## Why this repository exists

The book's argument is that the hard part of generative modelling for intrusion detection is not building the generator but measuring, honestly, whether it helps. This repository is that argument in code:

- a **single interface** so generators compare under an equal budget (Chapter 11);
- a **leakage-safe data layer** that splits before anything is fitted (Chapter 8);
- a **unified metrics module** reporting fidelity, utility and privacy together (Chapters 12, 20);
- **automated tests that fail on leakage** — the discipline of Part IV, enforced.

## Install

```bash
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
# or, as a package:
pip install -e .
```

Versions are pinned (see `requirements.txt`); a version-pinned `Dockerfile` is provided so the examples still run years from now:

```bash
docker build -t synnet-bench .
docker run --rm synnet-bench          # runs the test suite
```

## Layout

```
synnet/              the library
  reference.py       TabularCodec, NSGAN, WGANGP, CWGANGP, TabularVAE  (Appendix B)
  data.py            NSL-KDD loader, leakage_safe_split, reduce_class   (Chapter 8)
  metrics.py         mmd2, ks_mean, discriminative_auc, exact_copy_rate,
                     dcr, membership_inference_auc, tstr_recall         (Chapters 12, 20)
chapters/            one companion script per chapter (ch01..ch23), reproducing its numbers
tests/               test_leakage.py (leakage guards), test_reference_models.py (smoke tests)
tools/               refs.py, build.py — the manuscript's bibliography and build helper
data/                where NSL-KDD goes (not redistributed; see data/README.md)
```

## The interface

```python
import pandas as pd
from synnet import WGANGP, CWGANGP
from synnet.data import load_nslkdd, leakage_safe_split
from synnet.metrics import discriminative_auc, membership_inference_auc

X, y = load_nslkdd("data/nslkdd/KDDTrain+.txt")      # exact duplicates removed
Xtr, Xte, ytr, yte = leakage_safe_split(X, y, seed=0) # split BEFORE any fit

df = Xtr.copy(); df["category"] = ytr
g = CWGANGP(steps=2000, seed=0).fit(df, ["protocol_type","service","flag","category"],
                                    label_column="category")
r2l = g.sample(5000, label="R2L")                     # augment the minority class
```

Any new generator becomes comparable with everything here by implementing the same two methods, `fit(df, discrete_columns)` and `sample(n)`.

## The benchmark protocol (SynNet-Bench)

A result in the book is reported only when it follows this protocol:

1. a documented dataset and version, with known defects handled (Chapter 18, Appendix D);
2. a leakage-safe split, with every fitted step — scaler, generator, classifier — trained on the training part only (Chapter 8);
3. at least five fixed seeds, with results reported as a distribution and compared with paired tests, effect sizes and multiple-comparison correction (Chapter 19);
4. a unified report of fidelity, utility, diversity and privacy, with correctly scaled metrics (Chapters 12, 20);
5. an equal-compute comparison against a non-generative baseline, with negative results reported (Chapters 11, 13).

## Tests

```bash
PYTHONPATH=. python -m pytest tests/ -q
```

`tests/test_leakage.py` fails if a split is not disjoint, if a fitted step could see test data, or if augmentation places a synthetic record into the test set. `tests/test_reference_models.py` checks that every generator fits, samples and round-trips the schema.

## Responsible use

This repository releases code only — no trained models and no attack generators. The security material is for defence and evaluation (Chapters 15, 24). Please use it accordingly.

## License

MIT (code). Datasets are not redistributed and retain their own licenses and citation requirements.

## Citation

> Z. Arafat and Z. A. Abdulazeez, *Generative Adversarial Networks for Network Intrusion Detection: Synthetic Data, Rigorous Evaluation, and Trust*, 2026.
