"""Compute Derm7pt descriptive statistics used on the 'Model & Method' page.

Uses the same diagnosis mapping and binary checklist rules as the training notebook.
    python scripts/build_dataset_stats.py
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
META = ROOT / "data/derm7pt/meta.csv"
CLASSES = ["BCC", "Nev", "Mel", "SK", "Misc"]


def map_diagnosis(raw):
    s = str(raw).strip().lower()
    if "basal cell" in s:
        return 0
    if "nevus" in s or "naevus" in s:
        return 1
    if "melanoma" in s:
        return 2
    if "seborrheic keratosis" in s or "seborrhoeic keratosis" in s:
        return 3
    return 4


POSITIVE = {
    "PN": ("pigment_network", lambda s: s == "atypical"),
    "BWV": ("blue_whitish_veil", lambda s: s == "present"),
    "VS": ("vascular_structures", lambda s: s not in {"absent", "nan", "none", ""}),
    "PIG": ("pigmentation", lambda s: s in {"diffuse irregular", "localized irregular"}),
    "STR": ("streaks", lambda s: s == "irregular"),
    "DaG": ("dots_and_globules", lambda s: s == "irregular"),
    "RS": ("regression_structures", lambda s: s not in {"absent", "nan", "none", ""}),
}


def main():
    df = pd.read_csv(META)
    df["cls"] = df["diagnosis"].map(map_diagnosis)
    splits = {}
    for name, fname in (("train", "train_indexes.csv"), ("val", "valid_indexes.csv"), ("test", "test_indexes.csv")):
        p = ROOT / "data/derm7pt" / fname
        if p.is_file():
            splits[name] = pd.read_csv(p)["indexes"].astype(int).tolist()
    out = {"total_cases": int(len(df)),
           "class_counts_all": {c: int((df["cls"] == i).sum()) for i, c in enumerate(CLASSES)}}
    if "train" in splits:
        tr = df.loc[splits["train"]]
        out["train_cases"] = int(len(tr))
        out["class_counts_train"] = {c: int((tr["cls"] == i).sum()) for i, c in enumerate(CLASSES)}
        prev = {}
        for a, (col, rule) in POSITIVE.items():
            prev[a] = round(float(tr[col].astype(str).str.strip().str.lower().map(rule).mean()), 4)
        out["attribute_positive_rate_train"] = prev
    out["split_sizes"] = {k: len(v) for k, v in splits.items()}
    (ROOT / "model" / "dataset_stats.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
