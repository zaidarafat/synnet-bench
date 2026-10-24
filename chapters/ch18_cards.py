# %% [markdown]
# # Chapter 18 companion notebook: dataset-card statistics
#
# Reproduces the computable numbers in the NSL-KDD card. Other datasets' figures are quoted
# from their source papers (see the chapter's references) and are not recomputed here.
#   python ch18_cards.py
# %%
from pathlib import Path
import json
import pandas as pd
DATA = Path("/home/claude/data/nslkdd")
CATEGORY = {"normal": "normal",
    "back": "DoS", "land": "DoS", "neptune": "DoS", "pod": "DoS", "smurf": "DoS", "teardrop": "DoS",
    "ipsweep": "Probe", "nmap": "Probe", "portsweep": "Probe", "satan": "Probe",
    "ftp_write": "R2L", "guess_passwd": "R2L", "imap": "R2L", "multihop": "R2L", "phf": "R2L", "spy": "R2L",
    "warezclient": "R2L", "warezmaster": "R2L",
    "buffer_overflow": "U2R", "loadmodule": "U2R", "perl": "U2R", "rootkit": "U2R"}
R = {}
for name in ["KDDTrain+.txt", "KDDTest+.txt"]:
    d = pd.read_csv(DATA / name, header=None); cat = d[41].map(CATEGORY)
    dup = int(d.duplicated(subset=list(range(42))).sum())
    R[name] = {"records": len(d), "by_category": {k: int(v) for k, v in cat.value_counts().to_dict().items()},
               "exact_duplicates": dup, "largest_share": round(float(cat.value_counts(normalize=True).iloc[0]), 4),
               "attack_types": int(d[d[41] != "normal"][41].nunique())}
    print(f"{name}: {len(d)} records, {R[name]['attack_types']} attack types, "
          f"largest class {R[name]['largest_share']*100:.1f}%, exact duplicates {dup}")
    print("   by category:", R[name]["by_category"])
tr = pd.read_csv(DATA / "KDDTrain+.txt", header=None); te = pd.read_csv(DATA / "KDDTest+.txt", header=None)
unseen = sorted(set(te[41]) - set(tr[41]))
R["test_only_attack_types"] = unseen
print(f"attack types in test but not train ({len(unseen)}):", unseen)
Path("out/ch18_results.json").write_text(json.dumps(R, indent=1))
print("done")
