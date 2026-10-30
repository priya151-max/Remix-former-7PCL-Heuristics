import json
from pathlib import Path

import numpy as np
import pytest

from src.config import META_DIM
from src.metadata_encoder import load_default
from src.metadata_schema import FIELDS, clean_values, readable, validate_against_vocab

ROOT = Path(__file__).resolve().parent.parent


def test_vocab_dim_and_form_options_are_valid():
    enc = load_default()
    assert enc.dim == META_DIM == 30
    validate_against_vocab(enc)


def test_seven_point_score_is_never_asked():
    assert "seven_point_score" not in {f["key"] for f in FIELDS}


def test_unknown_and_blank_values_give_zero_blocks():
    enc = load_default()
    v = enc.transform({"sex": "male", "location": "not a site", "elevation": ""})[0]
    assert v.sum() == 1.0                       # only 'sex' is set
    assert v[enc.offset("sex") + 1] == 1.0      # male


def test_clean_values_drops_invalid_input():
    v = clean_values({"sex": "male", "location": "<script>", "management": "excision", "evil": "x"})
    assert v["sex"] == "male" and v["location"] == "" and v["management"] == "excision"
    assert set(v) == {f["key"] for f in FIELDS}
    assert readable(v)[1]["answer"] == "Not provided"


def test_matches_sklearn_one_hot_encoder():
    sk = pytest.importorskip("sklearn.preprocessing")
    pd = pytest.importorskip("pandas")
    meta = ROOT / "data/derm7pt/meta.csv"
    if not meta.is_file():
        pytest.skip("Derm7pt metadata not present")
    df = pd.read_csv(meta)
    idx = pd.read_csv(ROOT / "data/derm7pt/train_indexes.csv")["indexes"].astype(int).tolist()
    cols = load_default().columns
    for c in cols:
        df[c] = df[c].astype("string").fillna("UNKNOWN").replace({"": "UNKNOWN"})
    ref = sk.OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(df.loc[idx, cols].astype(str))
    enc = load_default()
    for i in list(range(0, 1011, 37)):
        row = df.loc[i, cols].astype(str).to_dict()
        assert np.allclose(enc.transform(row)[0], ref.transform(df.loc[[i], cols].astype(str))[0])
