import json

import pytest

from src import explain
from src.rag import KnowledgeBase, get_kb, retrieve_for_case

CASE = {
    "classes": [], "top_class": "Mel", "top_name": "Melanoma", "top_probability": 0.62,
    "runner_up": "Nev", "runner_up_name": "Melanocytic nevus (mole)", "runner_up_prob": 0.25,
    "margin": 0.37, "neural_top_class": "Mel", "heuristic_changed_top_class": False,
    "largest_heuristic_shift": {"class": "Mel", "change": 0.02}, "heuristic_lambda": 0.1,
    "attributes": [{"code": c, "name": n, "probability": p, "detected": p >= .5, "checklist_points": 1,
                    "checklist_kind": "minor", "positive_meaning": "", "top_state_from_image_model": "x",
                    "top_state_probability": .5}
                   for c, n, p in [("PN", "Pigment network", .8), ("BWV", "Blue-whitish veil", .3),
                                   ("VS", "Vascular structures", .2), ("PIG", "Irregular pigmentation", .6),
                                   ("STR", "Streaks", .1), ("DaG", "Dots and globules", .7),
                                   ("RS", "Regression structures", .2)]],
    "positive_attributes": ["PN", "DaG", "PIG"],
    "supporting_features_for_top_class": [{"code": "DaG", "name": "Dots and globules", "probability": .7,
                                           "share_of_training_cases_in_class": .9}],
    "features_unusual_for_top_class": [], "metadata": {"location": "back", "elevation": "flat"},
    "meta_missing": False, "missing_basic_information": [], "graph_node_attention": {},
    "query_terms": "Melanoma pigment network back flat",
}


def test_knowledge_base_is_consistent():
    kb = get_kb()
    assert len(kb.chunks) >= 30 and len(kb.by_id) == len(kb.chunks)
    for c in ("BCC", "Nev", "Mel", "SK", "Misc"):
        assert kb.chunk(f"class_{c.lower()}_overview") and kb.chunk(f"class_{c.lower()}_features")
    for a in ("PN", "BWV", "VS", "PIG", "STR", "DaG", "RS"):
        assert kb.chunk(f"attr_{a.lower()}")


def test_retrieval_is_case_specific_and_always_includes_safety():
    ids = [c["id"] for c in retrieve_for_case(CASE)]
    assert "class_mel_overview" in ids and "attr_pn" in ids and "attr_dag" in ids
    assert "safety_see_doctor" in ids and "safety_disclaimer" in ids and "general_abcde" in ids
    assert "attr_str" not in ids              # not detected -> not retrieved
    ids2 = [c["id"] for c in retrieve_for_case({**CASE, "top_class": "SK", "runner_up": "Nev",
                                                 "positive_attributes": []})]
    assert "class_sk_overview" in ids2 and "class_mel_overview" not in ids2


def test_bm25_ranks_relevant_passage_first():
    kb = get_kb()
    s = kb.bm25("blue whitish veil ground glass haze")
    assert kb.chunks[max(range(len(s)), key=s.__getitem__)]["id"] == "attr_bwv"


def test_template_explanation_has_all_sections_and_valid_citations():
    out = explain.generate_explanation(CASE, use_llm=False)
    titles = [s["title"] for s in out["sections"]]
    assert titles == explain.SECTION_TITLES
    assert explain._valid_sections(out["sections"])[0]
    n = len(out["sources"])
    import re
    text = json.dumps(out["sections"])
    for m in re.findall(r"\[S(\d+)\]", text):
        assert 1 <= int(m) <= n
    assert out["mode"] == "template"


GOOD = "\n\n".join(f"## {t}\n\n" + ("This is a plain sentence about the result. " * 9 + "[S1]") for t in explain.SECTION_TITLES)


def _with_llm(monkeypatch, text, ok=True):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(explain, "_call_llm", lambda content: (ok, text, {"seconds": 0.1}))


def test_llm_answer_is_used_when_valid(monkeypatch):
    _with_llm(monkeypatch, GOOD)
    out = explain.generate_explanation(CASE)
    assert out["mode"] == "llm" and out["notice"] is None


def test_unsafe_llm_answer_is_replaced(monkeypatch):
    _with_llm(monkeypatch, GOOD.replace("plain sentence", "sentence: you have a melanoma, definitely", 1))
    out = explain.generate_explanation(CASE)
    assert out["mode"] == "template" and "safety" in out["notice"]


def test_llm_failure_falls_back(monkeypatch):
    _with_llm(monkeypatch, "the AI writer returned an error (HTTP 503)", ok=False)
    out = explain.generate_explanation(CASE)
    assert out["mode"] == "template" and "could not be reached" in out["notice"]


def test_insufficient_credit_gives_clean_message(monkeypatch):
    """A 402 from the provider must not leak raw HTTP/JSON body text to the person."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    class Resp:
        status_code = 402
        text = ('{"error":{"message":"This request requires more credits, or fewer max_tokens. '
                'You requested up to 1800 tokens, but can only afford 1456."}}')

        def json(self):
            raise ValueError("not used")

    monkeypatch.setattr(explain.requests, "post", lambda *a, **k: Resp())
    ok, msg, diag = explain._call_llm("hello")
    assert not ok and msg == "the AI writer's account has run out of credit"
    assert "credits" not in msg and "tokens" not in msg and "1456" not in msg
    out = explain.generate_explanation(CASE)
    assert out["mode"] == "template" and "run out of credit" in out["notice"]


def test_prompt_never_contains_images_and_can_hide_metadata():
    from src.rag import retrieve_for_case as r
    payload = explain._payload_for_llm(CASE, r(CASE), share_meta=False)
    assert "base64" not in payload and '"metadata": {}' in payload
