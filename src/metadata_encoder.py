"""
Metadata one-hot encoder (pure NumPy, no scikit-learn, no CSV at runtime).

The trained metadata branch takes a 30-dimensional one-hot vector built from
six Derm7pt columns. The category vocabulary is stored in
``model/metadata_vocab.json`` (generated once from the Derm7pt training split by
``scripts/build_metadata_vocab.py`` and verified to be identical to the
scikit-learn ``OneHotEncoder`` used during training).

Behaviour is identical to ``OneHotEncoder(handle_unknown="ignore")``:
a value that is missing or outside the vocabulary produces an all-zero block for
that field. The web app uses this for every field the user does not provide
(for example ``seven_point_score``, which is not asked from the user).
"""
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from .config import META_DIM, METADATA_VOCAB_PATH


class MetadataEncoderError(RuntimeError):
    pass


class MetadataEncoder:
    def __init__(self, columns: List[str], categories: Dict[str, List[str]]):
        self.columns = list(columns)
        self.categories = {c: [str(v) for v in categories[c]] for c in self.columns}
        self._offsets: Dict[str, int] = {}
        pos = 0
        for c in self.columns:
            self._offsets[c] = pos
            pos += len(self.categories[c])
        self._dim = pos

    @property
    def dim(self) -> int:
        return self._dim

    def offset(self, column: str) -> int:
        return self._offsets[column]

    def transform(self, values: Dict[str, Optional[str]]) -> np.ndarray:
        """values: {column: raw string or None}. -> [1, dim] float32."""
        out = np.zeros((1, self._dim), dtype=np.float32)
        for c in self.columns:
            v = values.get(c)
            if v is None or str(v).strip() == "":
                continue
            cats = self.categories[c]
            v = str(v).strip()
            if v in cats:
                out[0, self._offsets[c] + cats.index(v)] = 1.0
        return out

    def transform_tensor(self, values: Dict[str, Optional[str]]) -> torch.Tensor:
        return torch.from_numpy(self.transform(values))

    def describe(self, values: Dict[str, Optional[str]]) -> List[dict]:
        """Per-field slice table used by the 'Intermediate results' page."""
        vec = self.transform(values)[0]
        rows = []
        for c in self.columns:
            n = len(self.categories[c])
            s = self._offsets[c]
            active = [self.categories[c][i] for i in range(n) if vec[s + i] > 0]
            rows.append({
                "field": c, "slice": [s, s + n], "width": n,
                "options": self.categories[c],
                "active": active[0] if active else None,
                "one_hot": [int(x) for x in vec[s:s + n]],
            })
        return rows


def load_default(path=METADATA_VOCAB_PATH) -> MetadataEncoder:
    path = Path(path)
    if not path.is_file():
        raise MetadataEncoderError(
            f"Metadata vocabulary not found: {path.name}. "
            "Run `python scripts/build_metadata_vocab.py` to create it.")
    blob = json.loads(path.read_text(encoding="utf-8"))
    enc = MetadataEncoder(blob["columns"], blob["categories"])
    if enc.dim != META_DIM:
        raise MetadataEncoderError(
            f"Metadata vocabulary has {enc.dim} dimensions, expected {META_DIM}.")
    return enc
