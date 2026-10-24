# %% [markdown]
# # Chapter 1 companion notebook: the data problem
#
# Reproduces Table 1.1 and supports Exercises 1, 2 and 6.
#
# Data (download yourself; not redistributed here):
# - NSL-KDD: `KDDTrain+.txt` from https://www.unb.ca/cic/datasets/nsl.html
# - CIC-IDS2017: the eight `MachineLearningCSV` files from
#   https://www.unb.ca/cic/datasets/ids-2017.html
#
# Set the two paths below and run the cells in order.
# This file uses `# %%` cell markers: it opens as a notebook in VS Code or
# Jupyter (with jupytext), and also runs as a plain script.

# %%
from pathlib import Path
import re

import numpy as np
import pandas as pd

NSL_KDD_TRAIN = Path("data/NSL-KDD/KDDTrain+.txt")
CICIDS2017_DIR = Path("data/CIC-IDS2017/MachineLearningCSV")
SEED = 0

# %% [markdown]
# ## 1. NSL-KDD: class counts in KDDTrain+
#
# KDDTrain+.txt has no header: 41 features, the attack name, and a difficulty
# score. Attack names are grouped into the four standard categories.
# Only the 22 attack types that occur in KDDTrain+ are needed here.

# %%
NSL_CATEGORY = {
    "normal": "normal",
    # DoS
    "back": "DoS", "land": "DoS", "neptune": "DoS", "pod": "DoS",
    "smurf": "DoS", "teardrop": "DoS",
    # Probe
    "ipsweep": "Probe", "nmap": "Probe", "portsweep": "Probe", "satan": "Probe",
    # R2L
    "ftp_write": "R2L", "guess_passwd": "R2L", "imap": "R2L", "multihop": "R2L",
    "phf": "R2L", "spy": "R2L", "warezclient": "R2L", "warezmaster": "R2L",
    # U2R
    "buffer_overflow": "U2R", "loadmodule": "U2R", "perl": "U2R", "rootkit": "U2R",
}


def load_nsl_kdd(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, header=None)
    if df.shape[1] != 43:
        raise ValueError(f"expected 43 columns, found {df.shape[1]}")
    df = df.rename(columns={41: "attack", 42: "difficulty"})
    unknown = set(df["attack"]) - set(NSL_CATEGORY)
    if unknown:
        raise ValueError(f"attack names without a category: {sorted(unknown)}")
    df["category"] = df["attack"].map(NSL_CATEGORY)
    return df


def class_table(labels: pd.Series) -> pd.DataFrame:
    counts = labels.value_counts()
    table = pd.DataFrame({"records": counts, "share_%": 100 * counts / counts.sum()})
    return table.sort_values("records", ascending=False)


nsl = load_nsl_kdd(NSL_KDD_TRAIN)
print("KDDTrain+ records:", len(nsl))  # Table 1.1 states 125,973
print(class_table(nsl["category"]).to_string(float_format=lambda v: f"{v:.4f}"))
# Table 1.1 states R2L = 995 (0.79%) and U2R = 52 (0.041%).

# %% [markdown]
# ## 2. CIC-IDS2017: class counts over all eight CSV files
#
# The published CSVs have leading spaces in column names and a non-UTF-8 dash
# in the web-attack labels. Both are normalised before counting.

# %%
def normalise_label(label: str) -> str:
    label = label.strip()
    # "Web Attack \x96 Brute Force" and similar variants -> "Web Attack - Brute Force"
    return re.sub(r"Web Attack\s*\W+\s*", "Web Attack - ", label)


def load_cicids2017(directory: Path) -> pd.DataFrame:
    files = sorted(directory.glob("*.csv"))
    if len(files) != 8:
        raise ValueError(f"expected 8 CSV files in {directory}, found {len(files)}")
    frames = []
    for f in files:
        part = pd.read_csv(f, encoding="latin-1", low_memory=False)
        part.columns = part.columns.str.strip()
        part["source_file"] = f.name
        frames.append(part)
    df = pd.concat(frames, ignore_index=True)
    df["Label"] = df["Label"].astype(str).map(normalise_label)
    return df


cic = load_cicids2017(CICIDS2017_DIR)
print("CIC-IDS2017 records:", len(cic))  # Table 1.1 states 2,830,743
print(class_table(cic["Label"]).to_string(float_format=lambda v: f"{v:.5f}"))
# Table 1.1 states Infiltration = 36, SQL Injection = 21, Heartbleed = 11.

# %% [markdown]
# ## 3. Exercise 1: accuracy of the majority-class predictor

# %%
for name, labels in [("NSL-KDD KDDTrain+ (category)", nsl["category"]),
                     ("CIC-IDS2017 (label)", cic["Label"])]:
    majority_share = labels.value_counts(normalize=True).iloc[0]
    print(f"{name}: majority-class accuracy = {100 * majority_share:.2f}%")

# %% [markdown]
# ## 4. Exercise 2: accuracy versus per-class recall
#
# The split is made before any preprocessing, so nothing learned from the test
# set reaches the model. Infinite values are replaced inside the training
# pipeline, and rows with missing values are dropped before the split.

# %%
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

features = cic.drop(columns=["Label", "source_file"]).select_dtypes(include=[np.number])
features = features.replace([np.inf, -np.inf], np.nan)
keep = features.notna().all(axis=1)
X, y = features[keep], cic.loc[keep, "Label"]
print(f"rows dropped for missing or infinite values: {(~keep).sum()}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=SEED
)
model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=SEED)
model.fit(X_train, y_train)
pred = model.predict(X_test)

print(f"accuracy = {100 * accuracy_score(y_test, pred):.3f}%")
report = classification_report(y_test, pred, output_dict=True, zero_division=0)
recall = pd.Series({c: report[c]["recall"] for c in sorted(y.unique())})
print("classes with recall below 0.5:")
print(recall[recall < 0.5].to_string())
weak_share = y_test.isin(recall[recall < 0.5].index).mean()
print(f"test records in those classes: {100 * weak_share:.4f}% of the test set")

# %% [markdown]
# ## 5. Exercise 6: constraint checks on CIC-IDS2017
#
# Three constraints every valid flow record must satisfy. Count violations.

# %%
checks = {
    "Flow Duration >= 0": cic["Flow Duration"] < 0,
    "Total Fwd Packets >= 0": cic["Total Fwd Packets"] < 0,
    "Flow Bytes/s is finite": ~np.isfinite(pd.to_numeric(cic["Flow Bytes/s"], errors="coerce")),
}
for rule, violated in checks.items():
    print(f"{rule}: {int(violated.sum())} violating rows")
