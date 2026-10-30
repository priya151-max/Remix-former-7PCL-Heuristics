"""
Rebuild model/metadata_vocab.json from the Derm7pt metadata.

Reproduces the training-time fit from the notebook (cell 6):

    df[c] = df[c].astype("string").fillna("UNKNOWN").replace({"": "UNKNOWN"})
    OneHotEncoder(handle_unknown="ignore").fit(df.loc[train_idx, METADATA_COLUMNS])

and stores only the resulting category vocabulary (sorted, exactly as
scikit-learn orders it). At runtime the web app uses this JSON with a tiny
pure-NumPy encoder, so no CSV path, scikit-learn or pickle is needed.

Usage:
    python scripts/build_metadata_vocab.py \
        --meta data/derm7pt/meta.csv --train data/derm7pt/train_indexes.csv
"""
import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
COLUMNS = ["seven_point_score", "level_of_diagnostic_difficulty", "elevation",
           "location", "sex", "management"]
META_DIM = 30


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", default=str(ROOT / "data/derm7pt/meta.csv"))
    ap.add_argument("--train", default=str(ROOT / "data/derm7pt/train_indexes.csv"))
    ap.add_argument("--out", default=str(ROOT / "model/metadata_vocab.json"))
    a = ap.parse_args()

    df = pd.read_csv(a.meta)
    train_idx = pd.read_csv(a.train)["indexes"].astype(int).tolist()
    cats = {}
    for c in COLUMNS:
        s = df[c].astype("string").fillna("UNKNOWN").replace({"": "UNKNOWN"})
        cats[c] = sorted(set(s.loc[train_idx].astype(str)))
    dim = sum(len(v) for v in cats.values())
    if dim != META_DIM:
        raise SystemExit(f"Vocabulary has {dim} dimensions, the trained model expects {META_DIM}.")
    Path(a.out).write_text(json.dumps(
        {"columns": COLUMNS, "categories": cats, "meta_dim": dim,
         "source": "Derm7pt training split (official train_indexes.csv)"}, indent=2))
    print(f"Wrote {a.out}  ({dim} dims)")
    for c, v in cats.items():
        print(f"  {c:34s} {v}")


if __name__ == "__main__":
    main()
