"""
Retrieval-Augmented Generation (RAG) knowledge retrieval.

The knowledge base (``src/knowledge/skin_knowledge.json``) holds short,
plain-language passages about the five lesion groups, the seven dermoscopic
features, the metadata questions, how to read the model output, and safety
advice. For every analysed case the retriever selects the passages that are
relevant to *this* case, and only those are handed to the language model as
grounding context (or used directly by the offline fallback writer).

Retrieval is hybrid and fully deterministic, with no external service and no
embedding model to download:

  1. Tag rules   - guaranteed passages for the predicted class, the runner-up
                   class, each detected dermoscopic feature, the metadata that
                   was supplied, the checklist, the model-output reading guide
                   and the safety notes.
  2. BM25        - lexical ranking of the remaining passages against a query
                   built from the case, to fill the remaining slots.

Each retrieved passage records *why* it was retrieved, so the UI can show it.
"""
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

KB_PATH = Path(__file__).resolve().parent / "knowledge" / "skin_knowledge.json"

_STOP = set("""a an and are as at be but by for from has have in into is it its of on or that the
their this to was were which with not no than then there these those can may also more most
very such when while who whom how what why where do does did if so we you your our""".split())

_TOKEN = re.compile(r"[a-z0-9]+")


def _tok(text: str) -> List[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP and len(t) > 1]


class KnowledgeBase:
    def __init__(self, path: Path = KB_PATH):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        self.meta: Dict[str, Any] = data.get("meta", {})
        self.references: List[Dict[str, str]] = data.get("references", [])
        self.chunks: List[Dict[str, Any]] = data["chunks"]
        self.by_id = {c["id"]: c for c in self.chunks}
        if len(self.by_id) != len(self.chunks):
            raise ValueError("Duplicate chunk ids in the knowledge base.")

        # BM25 statistics
        self._docs = [_tok((c["title"] + " ") * 2 + c["text"] + " " + " ".join(
            t.split(":")[-1] for t in c.get("tags", []))) for c in self.chunks]
        self._tf = [Counter(d) for d in self._docs]
        self._len = [len(d) for d in self._docs]
        self._avg = sum(self._len) / max(len(self._len), 1)
        df: Counter = Counter()
        for d in self._docs:
            df.update(set(d))
        n = len(self._docs)
        self._idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    # ---------------------------------------------------------------- BM25 --
    def bm25(self, query: str, k1: float = 1.5, b: float = 0.75) -> List[float]:
        q = _tok(query)
        scores = []
        for tf, ln in zip(self._tf, self._len):
            s = 0.0
            for t in q:
                if t not in tf:
                    continue
                f = tf[t]
                s += self._idf.get(t, 0.0) * f * (k1 + 1) / (
                    f + k1 * (1 - b + b * ln / max(self._avg, 1e-9)))
            scores.append(s)
        return scores

    def glossary(self) -> List[Dict[str, str]]:
        return [{"term": c["term"], "text": c["text"]}
                for c in self.chunks if c["type"] == "glossary"]

    def chunk(self, cid: str) -> Optional[Dict[str, Any]]:
        return self.by_id.get(cid)


_KB: Optional[KnowledgeBase] = None


def get_kb() -> KnowledgeBase:
    global _KB
    if _KB is None:
        _KB = KnowledgeBase()
    return _KB


def _tag_index(kb: KnowledgeBase, tag: str, types: Optional[set] = None) -> List[Dict[str, Any]]:
    return [c for c in kb.chunks if tag in c.get("tags", []) and (types is None or c["type"] in types)]


def retrieve_for_case(case: Dict[str, Any], max_chunks: int = 16,
                      kb: Optional[KnowledgeBase] = None) -> List[Dict[str, Any]]:
    """
    case keys used:
      top_class, runner_up (or None), runner_up_prob,
      positive_attributes [names], metadata {field: readable value}, meta_missing (bool),
      query_terms (optional str)
    Returns [{id, type, title, text, score, reason}] in reading order.
    """
    kb = kb or get_kb()
    picked: Dict[str, Dict[str, Any]] = {}

    def add(cid: str, reason: str, score: float):
        c = kb.chunk(cid)
        if c is None or cid in picked:
            return
        picked[cid] = {"id": cid, "type": c["type"], "title": c["title"],
                       "text": c["text"], "reason": reason, "score": round(score, 3)}

    top = case["top_class"]
    for c in _tag_index(kb, f"class:{top}", {"class"}):
        add(c["id"], f"Predicted group: {top}", 1.0)

    ru = case.get("runner_up")
    if ru and case.get("runner_up_prob", 0) >= 0.15:
        c0 = kb.chunk(f"class_{ru.lower()}_overview")
        if c0:
            add(c0["id"], f"Second most likely group: {ru}", 0.9)

    for a in case.get("positive_attributes", [])[:4]:
        for c in _tag_index(kb, f"attr:{a}", {"attribute"}):
            add(c["id"], f"Feature detected: {a}", 0.85)

    add("checklist_score", "Explains how the detected features are scored", 0.8)

    meta = case.get("metadata", {})
    for f in ("location", "elevation", "sex"):
        if meta.get(f):
            add(f"meta_{f}", f"You provided: {f}", 0.6)
    if case.get("meta_missing"):
        add("meta_missing", "Some information was not provided", 0.55)

    add("model_probability", "How to read the percentages", 0.75)
    add("model_limits", "Limits of the system", 0.7)
    add("model_graph", "How the knowledge graph works", 0.72)
    add("model_heuristic", "How the rule-based adjustment works", 0.72)
    add("input_quality", "Image quality affects reliability", 0.5)
    add("safety_see_doctor", "Safety advice", 0.92)
    add("safety_next_steps", "What happens next", 0.68)
    add("safety_disclaimer", "Not a diagnosis", 0.91)
    if top == "Mel" or (ru == "Mel" and case.get("runner_up_prob", 0) >= 0.25):
        add("general_abcde", "Melanoma is a leading or close possibility", 0.8)

    # -- BM25 fills remaining slots ---------------------------------------
    if len(picked) < max_chunks:
        q = case.get("query_terms") or ""
        if q.strip():
            scores = kb.bm25(q)
            order = sorted(range(len(kb.chunks)), key=lambda i: -scores[i])
            mx = max(scores) or 1.0
            for i in order:
                c = kb.chunks[i]
                if scores[i] <= 0 or len(picked) >= max_chunks:
                    break
                if c["type"] == "glossary" or c["id"] in picked:
                    continue
                add(c["id"], "Matched the case description (BM25)", 0.3 * scores[i] / mx)

    # reading order: class -> attributes -> checklist -> metadata -> model -> safety
    rank = {"class": 0, "attribute": 1, "method": 2, "metadata": 3, "model": 4, "safety": 5}
    keep = sorted(picked.values(), key=lambda c: -c["score"])[:max_chunks]
    return sorted(keep, key=lambda c: (rank.get(c["type"], 9), -c["score"]))
