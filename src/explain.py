"""
Plain-language explanation engine (RAG + LLM, with an offline fallback).

Flow
----
1. ``build_case``            summarise the finished model outputs (numbers only).
2. ``rag.retrieve_for_case`` pick the knowledge-base passages relevant to the case.
3. ``generate_explanation``  ask the LLM (OpenRouter) to write a simple-words
                             explanation grounded ONLY in (1) and (2), citing
                             passages as [S1], [S2] ...
                             If no API key is configured, or the request fails,
                             or the answer breaks the safety rules, the same
                             passages are assembled into an equivalent
                             explanation by a deterministic writer, so the site
                             and the PDF report always work.

The LLM never receives images and never changes a prediction.
"""
import datetime as _dt
import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests
from dotenv import load_dotenv

from .display import ATTRIBUTE_INFO, CLASS_INFO, STATE_LABELS
from .rag import KnowledgeBase, get_kb, retrieve_for_case

load_dotenv()

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "anthropic/claude-sonnet-4.6"
DEFAULT_MAX_TOKENS = 1800
TIMEOUT_SECONDS = 75

SECTION_TITLES = [
    "In short",
    "What the system found",
    "How sure is the system?",
    "What it noticed in the lesion",
    "How the knowledge graph and clinical rules helped",
    "What the system cannot tell you",
    "What to do next",
    "Words explained",
]


# ------------------------------------------------------------------ config --
def llm_config() -> Dict[str, Any]:
    key = os.getenv("OPENROUTER_API_KEY", "").strip()
    try:
        max_tokens = max(600, int(os.getenv("OPENROUTER_MAX_TOKENS", DEFAULT_MAX_TOKENS)))
    except ValueError:
        max_tokens = DEFAULT_MAX_TOKENS
    return {
        "available": bool(key) and key != "your_openrouter_api_key_here",
        "model": os.getenv("OPENROUTER_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL,
        "base_url": os.getenv("OPENROUTER_BASE_URL", DEFAULT_BASE_URL).strip() or DEFAULT_BASE_URL,
        "referer": os.getenv("OPENROUTER_HTTP_REFERER", "http://localhost:8000"),
        "app_name": os.getenv("OPENROUTER_APP_NAME", "Skin Tumor Differential Diagnosis"),
        "max_tokens": max_tokens,
        "share_metadata": os.getenv("LLM_SHARE_METADATA", "true").strip().lower() != "false",
    }


# -------------------------------------------------------------------- case --
def strength_phrase(p: float) -> str:
    if p >= 0.75:
        return "strongly favours"
    if p >= 0.55:
        return "favours"
    if p >= 0.40:
        return "leans towards"
    return "does not clearly favour"


def build_case(model, R: Dict[str, Any], meta_readable: List[dict]) -> Dict[str, Any]:
    """Numbers-only summary of a finished inference, used by RAG, the LLM and the fallback writer."""
    cn, an = model.class_names, model.attribute_names
    final = np.asarray(R["heuristic_prob"][0], dtype=float)
    neural = np.asarray(R["diagnosis_prob"][0], dtype=float)
    order = list(np.argsort(-final))
    ti, ri = int(order[0]), int(order[1])
    terms = R["heuristic_terms"]
    fused = np.asarray(terms["fused_attr"][0], dtype=float)

    attrs = []
    for j, a in enumerate(an):
        p_up = np.asarray(R["upstream_checklist_prob"][a][0], dtype=float)
        states = model.checklist_specs[a]["classes"]
        attrs.append({
            "code": a,
            "name": ATTRIBUTE_INFO[a]["name"],
            "positive_meaning": ATTRIBUTE_INFO[a]["positive"],
            "probability": round(float(fused[j]), 4),
            "detected": bool(fused[j] >= 0.5),
            "top_state_from_image_model": states[int(p_up.argmax())],
            "top_state_probability": round(float(p_up.max()), 4),
            "checklist_points": ATTRIBUTE_INFO[a]["points"],
            "checklist_kind": ATTRIBUTE_INFO[a]["kind"],
        })
    positive = [a["code"] for a in sorted(attrs, key=lambda x: -x["probability"]) if a["detected"]]

    prior = np.asarray(model.class_attr_prior, dtype=float)
    supporting, contradicting = [], []
    for a in attrs:
        if not a["detected"]:
            continue
        j = an.index(a["code"])
        pr = float(prior[ti, j])
        row = {"code": a["code"], "name": a["name"], "probability": a["probability"],
               "share_of_training_cases_in_class": round(pr, 3)}
        if model.contradiction_mask[ti, j] > 0:
            contradicting.append(row)
        else:
            supporting.append(row)
    supporting.sort(key=lambda r: -r["share_of_training_cases_in_class"])
    contradicting.sort(key=lambda r: r["share_of_training_cases_in_class"])

    delta = final - neural
    ch = int(np.argmax(np.abs(delta)))

    provided = {r["key"]: r["answer"] for r in meta_readable if r["provided"]}
    missing = [r["question"] for r in meta_readable if not r["provided"] and r["group"] == "basic"]
    changed = int(np.argmax(neural)) != ti

    case = {
        "classes": [{
            "code": c, "name": CLASS_INFO[c]["name"],
            "final_probability": round(float(final[i]), 4),
            "neural_probability": round(float(neural[i]), 4),
            "rank": int(order.index(i)) + 1,
        } for i, c in enumerate(cn)],
        "top_class": cn[ti], "top_name": CLASS_INFO[cn[ti]]["name"],
        "top_probability": round(float(final[ti]), 4),
        "runner_up": cn[ri], "runner_up_name": CLASS_INFO[cn[ri]]["name"],
        "runner_up_prob": round(float(final[ri]), 4),
        "margin": round(float(final[ti] - final[ri]), 4),
        "neural_top_class": cn[int(np.argmax(neural))],
        "heuristic_changed_top_class": changed,
        "largest_heuristic_shift": {"class": cn[ch], "change": round(float(delta[ch]), 4)},
        "heuristic_lambda": round(float(model.heuristic_lambda), 4),
        "attributes": attrs,
        "positive_attributes": positive,
        "supporting_features_for_top_class": supporting,
        "features_unusual_for_top_class": contradicting,
        "metadata": {k: v for k, v in provided.items()
                     if k in ("sex", "location", "elevation")},
        "meta_missing": bool(missing),
        "missing_basic_information": missing,
        "graph_node_attention": {n: round(float(v), 4) for n, v in
                                 zip(model.graph_nodes, R["node_attention"][0])},
    }
    names = " ".join([case["top_name"], case["runner_up_name"]]
                     + [ATTRIBUTE_INFO[a]["name"] for a in positive]
                     + [str(v) for v in case["metadata"].values()])
    case["query_terms"] = names
    return case


# ------------------------------------------------------------------ prompts -
SYSTEM_PROMPT = """You write the explanation page of a research web application that analyses skin-lesion images with an AI system. The reader is an ordinary adult with no medical training.

WRITE IN SIMPLE WORDS
- Short sentences, everyday words, reading level of a 13-year-old. Use "you" and "the system".
- The first time you use a medical word, explain it in plain words in the same sentence.
- No jargon about neural networks or mathematics. Describe the knowledge graph and rules in one plain sentence each.

GROUNDING (very important)
- You receive CASE DATA (numbers produced by the system for this lesion) and KNOWLEDGE passages labelled [S1], [S2], ...
- Every statement about THIS lesion must come from CASE DATA. Every general medical statement must come from a KNOWLEDGE passage, and you must cite it like [S3] straight after the sentence.
- If something is in neither, do not say it. Never invent findings, image observations, patient details, statistics or advice.

SAFETY RULES
- The output is an experimental estimate, NOT a diagnosis. Never say the lesion is, or is not, cancer or benign. Never say "you have" a condition. Never say it is safe or nothing to worry about.
- Do not give treatment instructions. Do not claim clinical validation.
- Give percentages as the system's scores, not as the chance of disease. Round percentages to whole numbers.
- If the top score is below 60 percent, or the top two groups are within 15 percentage points, say plainly that the system is unsure.
- If basic information was not provided, say the result may be less precise.
- Always explain that a dermatologist's examination is needed and mention the doctor advice from the knowledge.
- If the heuristic changed the top group, explain that in one or two plain sentences.

FORMAT
Write Markdown with EXACTLY these eight second-level headings, in this order, and nothing before the first heading:
## In short
## What the system found
## How sure is the system?
## What it noticed in the lesion
## How the knowledge graph and clinical rules helped
## What the system cannot tell you
## What to do next
## Words explained

- "In short": 2 or 3 sentences.
- "What it noticed in the lesion": a bullet list ("- ") of the features detected, each with its score and one plain sentence on what it means, then one sentence on the features not detected.
- "Words explained": 3 to 5 bullets in the form "- **Term**: simple meaning".
- Other sections: short paragraphs (bullets allowed for "What to do next").
- Total length 450 to 700 words. No tables, no HTML, no emojis.
"""

_FORBIDDEN = [
    r"\byou (have|do not have|don't have|are suffering)\b",
    r"\b(definitely|certainly|for sure|without doubt|guaranteed)\b",
    r"\b(nothing to worry|no need to worry|safe to ignore|is safe)\b",
    r"\bthis (is|isn't|is not) (a |an )?(cancer|melanoma|carcinoma|benign|malignant)\b",
]


def _payload_for_llm(case: Dict[str, Any], sources: List[Dict[str, Any]], share_meta: bool) -> str:
    c = {k: v for k, v in case.items() if k != "query_terms"}
    if not share_meta:
        c["metadata"] = {}
    knowledge = "\n\n".join(f"[S{i + 1}] {s['title']}: {s['text']}" for i, s in enumerate(sources))
    return ("CASE DATA (JSON):\n" + json.dumps(c, indent=1)
            + "\n\nKNOWLEDGE:\n" + knowledge
            + "\n\nWrite the explanation now, following the format exactly.")


# ------------------------------------------------------------------ parsing -
def parse_markdown_sections(md: str) -> List[Dict[str, Any]]:
    sections: List[Dict[str, Any]] = []
    cur: Optional[Dict[str, Any]] = None
    para: List[str] = []
    bullets: List[str] = []

    def flush_para():
        nonlocal para
        if para and cur is not None:
            cur["blocks"].append({"type": "p", "text": " ".join(para).strip()})
        para = []

    def flush_bullets():
        nonlocal bullets
        if bullets and cur is not None:
            cur["blocks"].append({"type": "ul", "items": bullets})
        bullets = []

    for raw in md.replace("\r", "").split("\n"):
        line = raw.rstrip()
        m = re.match(r"^#{1,3}\s+(.*)$", line)
        if m:
            flush_para(); flush_bullets()
            cur = {"title": m.group(1).strip().strip("#").strip(), "blocks": []}
            sections.append(cur)
            continue
        if cur is None:
            continue
        b = re.match(r"^\s*[-*•]\s+(.*)$", line)
        if b:
            flush_para()
            bullets.append(b.group(1).strip())
        elif not line.strip():
            flush_para(); flush_bullets()
        else:
            flush_bullets()
            para.append(line.strip())
    flush_para(); flush_bullets()
    return [s for s in sections if s["blocks"]]


def _valid_sections(sections: List[Dict[str, Any]]) -> Tuple[bool, str]:
    titles = [re.sub(r"[^a-z ]", "", s["title"].lower()).strip() for s in sections]
    need = [re.sub(r"[^a-z ]", "", t.lower()).strip() for t in SECTION_TITLES]
    missing = [t for t in need if t not in titles]
    if missing:
        return False, f"missing sections: {missing[:3]}"
    text = " ".join(_section_text(s) for s in sections)
    for pat in _FORBIDDEN:
        m = re.search(pat, text, flags=re.I)
        if m:
            return False, f"safety rule triggered by: '{m.group(0)}'"
    if len(text.split()) < 200:
        return False, "answer too short"
    return True, ""


def _section_text(s: Dict[str, Any]) -> str:
    out = []
    for b in s["blocks"]:
        out.append(b["text"] if b["type"] == "p" else " ".join(b["items"]))
    return " ".join(out)


# ---------------------------------------------------------------- LLM call --
def _call_llm(user_content: str) -> Tuple[bool, str, Dict[str, Any]]:
    conf = llm_config()
    key = os.getenv("OPENROUTER_API_KEY", "").strip()
    url = conf["base_url"].rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {key}", "Content-Type": "application/json",
        "HTTP-Referer": conf["referer"], "X-Title": conf["app_name"],
    }
    body = {
        "model": conf["model"],
        "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                     {"role": "user", "content": user_content}],
        "temperature": 0.3, "max_tokens": conf["max_tokens"],
    }
    diag: Dict[str, Any] = {"model": conf["model"], "prompt_chars": len(user_content)}
    last_err = "unknown error"
    for attempt in range(2):
        try:
            t0 = time.perf_counter()
            resp = requests.post(url, headers=headers, json=body, timeout=TIMEOUT_SECONDS)
            diag["seconds"] = round(time.perf_counter() - t0, 2)
        except requests.exceptions.Timeout:
            last_err = f"request timed out after {TIMEOUT_SECONDS}s"
            continue
        except requests.exceptions.RequestException as exc:
            last_err = f"network error ({type(exc).__name__})"
            continue
        diag["status_code"] = resp.status_code
        if resp.status_code in (429, 500, 502, 503, 504) and attempt == 0:
            last_err = f"HTTP {resp.status_code}"
            time.sleep(1.5)
            continue
        if resp.status_code == 401:
            return False, "the configured API key was rejected", diag
        if resp.status_code == 402:
            return False, "the AI writer's account has run out of credit", diag
        if resp.status_code == 429:
            return False, "the AI writer is temporarily rate-limited", diag
        if resp.status_code >= 400:
            return False, f"the AI writer returned an error (HTTP {resp.status_code})", diag
        try:
            data = resp.json()
            text = data["choices"][0]["message"]["content"]
        except Exception:  # noqa: BLE001
            return False, "malformed response", diag
        if not isinstance(text, str) or not text.strip():
            return False, "empty response", diag
        diag["usage"] = data.get("usage", {})
        return True, text, diag
    return False, last_err, diag


# ------------------------------------------------------- offline (fallback) -
def _pct(p: float) -> str:
    return f"{round(100 * p)}%"


def _np(name: str, p: float) -> str:
    """'Melanoma (43%)' but avoids double brackets: 'Melanocytic nevus (mole) at 43%'."""
    return f"{name} at {_pct(p)}" if "(" in name else f"{name} ({_pct(p)})"


def _cite(idx: Dict[str, int], *ids: str) -> str:
    marks = [f"[S{idx[i]}]" for i in ids if i in idx]
    return (" " + "".join(marks)) if marks else ""


def template_explanation(case: Dict[str, Any], sources: List[Dict[str, Any]],
                         kb: KnowledgeBase) -> List[Dict[str, Any]]:
    """Deterministic writer producing the same eight sections from case data + retrieved passages."""
    idx = {s["id"]: i + 1 for i, s in enumerate(sources)}
    top, ru = case["top_class"], case["runner_up"]
    tn, rn = case["top_name"], case["runner_up_name"]
    tp, rp = case["top_probability"], case["runner_up_prob"]
    unsure = tp < 0.60 or (tp - rp) < 0.15
    S: List[Dict[str, Any]] = []

    def sec(title, *blocks):
        S.append({"title": title, "blocks": list(blocks)})

    P = lambda t: {"type": "p", "text": t}          # noqa: E731
    UL = lambda items: {"type": "ul", "items": items}  # noqa: E731

    # 1 In short
    short = (f"The system's highest score for this lesion is for {_np(tn, tp)}. "
             f"The next highest is {_np(rn, rp)}. ")
    short += ("The scores are close, so the system is not sure which group fits best. " if unsure
              else "The system leans clearly towards its top group, but that is still only an estimate. ")
    short += "This is an experimental research result, not a diagnosis, and only a doctor can tell you what the lesion is."
    sec("In short", P(short + _cite(idx, "safety_disclaimer")))

    # 2 What it found
    kb_over = kb.chunk(f"class_{top.lower()}_overview")
    found = [P(f"After looking at your two images and the basic answers you gave, the system {strength_phrase(tp)} "
               f"the group called {tn}, with a score of {_pct(tp)}. ")]
    if kb_over:
        found.append(P(kb_over["text"] + _cite(idx, kb_over["id"])))
    if case["heuristic_changed_top_class"]:
        found.append(P(f"The image network on its own had put {CLASS_INFO[case['neural_top_class']]['name']} first. "
                       f"After the rule-based check described below, {tn} moved to the top."))
    if rp >= 0.15:
        ru_over = kb.chunk(f"class_{ru.lower()}_overview")
        if ru_over:
            found.append(P(f"The second-highest group was {_np(rn, rp)}. " + ru_over["text"]
                           + _cite(idx, ru_over["id"])))
    sec("What the system found", *found)

    # 3 How sure
    others = [c for c in case["classes"] if c["code"] not in (top, ru)]
    others_txt = ", ".join(f"{c['name']} {_pct(c['final_probability'])}" for c in
                           sorted(others, key=lambda c: -c["final_probability"]))
    sure = [P(f"The five scores add up to 100%: {tn} {_pct(tp)}, {rn} {_pct(rp)}, and {others_txt}. "
              "These are the system's own scores, not the chance that the lesion has a particular condition."
              + _cite(idx, "model_probability"))]
    if unsure:
        sure.append(P("Because the top scores are close or the top score is not very high, treat this result as "
                      "especially uncertain."))
    sec("How sure is the system?", *sure)

    # 4 Noticed
    det = [a for a in case["attributes"] if a["detected"]]
    not_det = [a for a in case["attributes"] if not a["detected"]]
    items = []
    for a in sorted(det, key=lambda x: -x["probability"]):
        info = ATTRIBUTE_INFO[a["code"]]
        items.append(f"**{a['name']} ({_pct(a['probability'])})**: {info['plain']}" + _cite(idx, f"attr_{a['code'].lower()}"))
    noticed: List[Dict[str, Any]] = []
    if items:
        noticed.append(P("The system estimates that these features are present in the lesion:"))
        noticed.append(UL(items))
    else:
        noticed.append(P("The system did not find any of the seven checklist features with a score of 50% or more."))
    if not_det:
        noticed.append(P("It did not find enough evidence for: " + ", ".join(a["name"].lower() for a in not_det) + "."))
    noticed.append(P("These feature scores are estimates made by the AI from the images. A doctor looking at the lesion "
                     "through a dermatoscope may judge them differently." + _cite(idx, "checklist_score")))
    sec("What it noticed in the lesion", *noticed)

    # 5 Graph + rules
    g = [P("The system contains a map, called a knowledge graph, that links the seven features to the five lesion groups. "
           "Each link has a number that shows how often the two appeared together in the training examples."
           + _cite(idx, "model_graph"))]
    sup, con = case["supporting_features_for_top_class"], case["features_unusual_for_top_class"]
    if sup:
        s0 = sup[:3]
        g.append(P(f"For {tn}, the features found here that were common in the training examples of that group: "
                   + "; ".join(f"{r['name'].lower()} (seen in about {_pct(r['share_of_training_cases_in_class'])} of them)" for r in s0)
                   + "."))
    if con:
        g.append(P("Some detected features were uncommon in that group: "
                   + "; ".join(f"{r['name'].lower()} (about {_pct(r['share_of_training_cases_in_class'])})" for r in con[:3])
                   + ". Such features lower that group's score a little."))
    g.append(P("After the image network gave its scores, a small rule-based step adjusted them using this knowledge. "
               "It is deliberately gentle." + _cite(idx, "model_heuristic")))
    ls = case["largest_heuristic_shift"]
    if case["heuristic_changed_top_class"]:
        g.append(P("In this case the adjustment changed which group came first."))
    else:
        g.append(P(f"In this case it did not change which group came first; the biggest shift was for "
                   f"{CLASS_INFO[ls['class']]['name']} ({ls['change'] * 100:+.1f} percentage points)."))
    sec("How the knowledge graph and clinical rules helped", *g)

    # 6 Limits
    lim = [P("The system only sees the images and a few answers. It cannot feel the lesion, know how long it has been there "
             "or how it has changed, or take your medical history into account." + _cite(idx, "model_limits")),
           P("It can also be wrong when the photos are blurry, badly lit or not taken through a dermatoscope."
             + _cite(idx, "input_quality"))]
    if case["meta_missing"]:
        lim.append(P("Some of the basic information was not provided, so the result may be less precise."
                     + _cite(idx, "meta_missing")))
    else:
        lim.append(P("Some clinician-only details are not asked from users and are left blank, so the result may be a little "
                     "less precise than with a full clinical record." + _cite(idx, "meta_missing")))
    sec("What the system cannot tell you", *lim)

    # 7 Next
    nxt = [P("Please show the lesion to a doctor or dermatologist, and take this report with you."
             + _cite(idx, "safety_next_steps")),
           UL([kb.chunk("safety_see_doctor")["text"] + _cite(idx, "safety_see_doctor")])]
    if top == "Mel" or (ru == "Mel" and rp >= 0.25):
        nxt.append(P("Because melanoma is among the leading possibilities here, do not wait: arrange a skin check soon."
                     + _cite(idx, "general_abcde")))
    sec("What to do next", *nxt)

    # 8 Words
    terms = []
    for gid, tmpl in (("gloss_dermoscopy", None), ("gloss_biopsy", None), ("gloss_lesion", None)):
        c = kb.chunk(gid)
        if c:
            terms.append(f"**{c['term']}**: {c['text']}")
    c = kb.chunk("gloss_benign_malignant")
    if c:
        terms.append(f"**{c['term']}**: {c['text']}")
    sec("Words explained", UL(terms[:4]))
    return S


# ------------------------------------------------------------------- public -
def generate_explanation(case: Dict[str, Any], kb: Optional[KnowledgeBase] = None,
                         use_llm: bool = True) -> Dict[str, Any]:
    kb = kb or get_kb()
    chunks = retrieve_for_case(case, kb=kb)
    sources = [{"n": i + 1, **{k: c[k] for k in ("id", "type", "title", "reason", "score", "text")}}
               for i, c in enumerate(chunks)]
    conf = llm_config()
    result: Dict[str, Any] = {
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "sources": sources, "mode": "template", "model": None, "notice": None,
        "retrieval": {"method": "tag rules + BM25", "passages": len(sources),
                      "knowledge_base_passages": len(kb.chunks)},
    }

    sections = None
    if use_llm and conf["available"]:
        ok, text, diag = _call_llm(_payload_for_llm(case, sources, conf["share_metadata"]))
        result["llm_diagnostics"] = diag
        if ok:
            parsed = parse_markdown_sections(text)
            valid, why = _valid_sections(parsed)
            if valid:
                sections = parsed
                result.update(mode="llm", model=conf["model"])
            else:
                result["notice"] = ("The AI writer's answer did not pass the safety and format checks "
                                    f"({why}), so an automatic explanation is shown instead.")
        else:
            result["notice"] = (f"The AI writer could not be reached ({text}), so an automatic "
                                "explanation is shown instead.")
    elif use_llm:
        result["notice"] = ("The AI writer is not configured on this server, so an automatic explanation "
                            "assembled from the same knowledge passages is shown.")

    if sections is None:
        sections = template_explanation(case, sources, kb)
    result["sections"] = sections
    result["glossary"] = kb.glossary()
    result["references"] = kb.references
    return result
