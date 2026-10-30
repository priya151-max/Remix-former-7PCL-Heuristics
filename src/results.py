"""
Turn a raw inference result into the JSON document consumed by the web UI and
the PDF report. Everything here is a re-formatting of real model outputs;
the only computed additions are image overlays that visualise genuine attention
tensors (LSM score and region-transformer attention).
"""
import base64
import datetime as _dt
import io
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image, ImageDraw

from . import config as cfg
from .checkpoint_loader import LoadedModel
from .display import ATTRIBUTE_INFO, CLASS_INFO, REMIX_TOKEN_INFO, STATE_LABELS
from .metadata_encoder import MetadataEncoder
from .rag import get_kb

THUMB_MAX = 720


# ------------------------------------------------------------- image utils --
def _b64_jpeg(img: Image.Image, quality: int = 86) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=quality, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def thumbnail(img: Image.Image, max_side: int = THUMB_MAX) -> Image.Image:
    im = img.copy()
    im.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return im


def _cmap_lut() -> np.ndarray:
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import colormaps
    return (colormaps["inferno"](np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)


def heat_overlay(base: Image.Image, grid: np.ndarray, alpha: float = 0.55,
                 boxes: Optional[List[dict]] = None, draw_grid: bool = False) -> Image.Image:
    """Blend a coarse attention grid over `base`. Grid is min-max scaled for display only."""
    g = np.asarray(grid, dtype=np.float32)
    lo, hi = float(g.min()), float(g.max())
    norm = (g - lo) / (hi - lo) if hi - lo > 1e-12 else np.zeros_like(g)
    small = Image.fromarray((norm * 255).astype(np.uint8), "L")
    heat = small.resize(base.size, Image.Resampling.BICUBIC)
    lut = _cmap_lut()
    rgb = Image.fromarray(lut[np.asarray(heat)], "RGB")
    a = (np.asarray(heat, dtype=np.float32) / 255.0) * alpha + 0.12
    out = Image.composite(rgb, base.convert("RGB"),
                          Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8), "L"))
    d = ImageDraw.Draw(out)
    W, H = out.size
    if draw_grid:
        n = g.shape[0]
        for i in range(1, n):
            d.line([(i * W / n, 0), (i * W / n, H)], fill=(255, 255, 255), width=1)
            d.line([(0, i * H / n), (W, i * H / n)], fill=(255, 255, 255), width=1)
    for b in boxes or []:
        d.rectangle(b["xyxy"], outline=(56, 189, 248), width=max(2, W // 200))
        tx, ty = b["xyxy"][0] + 4, b["xyxy"][1] + 2
        d.rectangle([tx - 2, ty - 1, tx + 16, ty + 14], fill=(56, 189, 248))
        d.text((tx + 2, ty), str(b["n"]), fill=(8, 47, 73))
    return out


def crop_boxes(centers_xy: np.ndarray, size_wh) -> List[dict]:
    """LSM crop centres are in 896-px space; map to the display image (aspect preserved)."""
    W, H = size_wh
    s = cfg.CLINICAL_SIZE
    half = cfg.LOCAL_PATCH_SIZE / 2.0
    out = []
    for i, (cx, cy) in enumerate(centers_xy):
        x0 = max(0.0, cx - half) / s * W
        y0 = max(0.0, cy - half) / s * H
        x1 = min(float(s), cx + half) / s * W
        y1 = min(float(s), cy + half) / s * H
        out.append({"n": i + 1, "xyxy": [float(x0), float(y0), float(x1), float(y1)],
                    "center_px": [float(cx), float(cy)]})
    return out


# ------------------------------------------------------------------ helpers -
def _r(a, nd=4):
    return np.round(np.asarray(a, dtype=float), nd).tolist()


def _stat_row(name: str, t, extra: Dict[str, Any]) -> Dict[str, Any]:
    a = t.detach().cpu().numpy() if hasattr(t, "detach") else np.asarray(t)
    return {"input": name, "shape": list(a.shape), "dtype": str(a.dtype).replace("torch.", ""),
            "min": round(float(a.min()), 4), "max": round(float(a.max()), 4),
            "mean": round(float(a.mean()), 4), **extra}


# ------------------------------------------------------------- static info --
def model_info(model: LoadedModel, encoder: MetadataEncoder) -> Dict[str, Any]:
    """Public, non-sensitive description of the model and the stored knowledge graph. No file paths."""
    stats_path = cfg.PROJECT_ROOT / "model" / "dataset_stats.json"
    dataset = json.loads(stats_path.read_text()) if stats_path.is_file() else None
    cn, an, gn = model.class_names, model.attribute_names, model.graph_nodes
    return {
        "classes": [{"code": c, **CLASS_INFO[c]} for c in cn],
        "attributes": [{"code": a, **ATTRIBUTE_INFO[a],
                        "states": [{"value": s, "label": STATE_LABELS.get(s, s)}
                                   for s in model.checklist_specs[a]["classes"]]} for a in an],
        "graph": {
            "nodes": gn, "num_attributes": len(an),
            "A_full": _r(model.A_full, 4), "A_sparse": _r(model.A_sparse, 4),
            "A_internal": _r(model.A_internal, 4),
            "A_attr_to_class": _r(model.A_attr_to_class, 4),
            "A_class_to_attr": _r(model.A_class_to_attr, 4),
            "proximity": {str(k): _r(v.cpu().numpy(), 4) for k, v in sorted(model.proximity.items())},
            "semantic_norms": _r(np.linalg.norm(model.semantic_matrix, axis=1), 4),
            "semantic_matrix": _r(model.semantic_matrix, 3),
            "order": model.graph_order, "hidden": model.graph_hidden,
        },
        "heuristic": {
            "lambda": float(model.heuristic_lambda),
            "components": model.heuristic_components,
            "graph_weight": cfg.HEURISTIC_GRAPH_WEIGHT,
            "contradiction_weight": cfg.HEURISTIC_CONTRADICTION_WEIGHT,
            "interaction_weight": cfg.HEURISTIC_INTERACTION_WEIGHT,
            "class_attr_prior": _r(model.class_attr_prior, 4),
            "reliability": _r(model.reliability, 4),
            "contradiction_mask": _r(model.contradiction_mask, 0),
            "attr_pair_prior_mean": _r(model.attr_pair_prior.mean(axis=0), 4),
        },
        "architecture": {
            "clinical_size": cfg.CLINICAL_SIZE, "dermoscopy_size": cfg.DERMOSCOPY_SIZE,
            "global_clinical_size": cfg.GLOBAL_CLINICAL_SIZE, "local_patch_size": cfg.LOCAL_PATCH_SIZE,
            "lsm_k": cfg.LSM_K, "derm_windows": cfg.NUM_DERM_WINDOWS,
            "derm_window_size": cfg.DERM_WINDOW_SIZE,
            "token_dim": model.token_dim, "num_tokens": model.num_remix_tokens,
            "token_names": cfg.REMIX_TOKEN_NAMES[:model.num_remix_tokens],
            "token_labels": [REMIX_TOKEN_INFO[t] for t in cfg.REMIX_TOKEN_NAMES[:model.num_remix_tokens]],
            "semantic_dim": model.semantic_dim, "graph_hidden": model.graph_hidden,
            "graph_order": model.graph_order, "meta_dim": model.meta_dim,
            "parameters": {
                "RemixFormer++ (image + metadata encoder)": sum(p.numel() for p in model.remix.parameters()),
                "Multi-order graph encoder": sum(p.numel() for p in model.graph_encoder.parameters()),
                "7PCL hybrid head": sum(p.numel() for p in model.hybrid.parameters()),
            },
            "state_dict_loads": [{"module": r["module"], "tensors": r["tensors_loaded"],
                                  "strict": r["strict"]} for r in model.load_report["state_dict_loads"]],
            "metadata_columns": encoder.columns,
        },
        "dataset": dataset,
        "knowledge": {"references": get_kb().references, "glossary": get_kb().glossary(),
                      "passages": len(get_kb().chunks)},
    }


# ------------------------------------------------------------- case result --
def build_result(model: LoadedModel, encoder: MetadataEncoder, R: Dict[str, Any],
                 clinical_img: Image.Image, dermoscopy_img: Image.Image,
                 clinical_t, dermoscopy_t, metadata_t, meta_values: Dict[str, str],
                 meta_readable: List[dict], case: Dict[str, Any],
                 explanation: Dict[str, Any], case_id: str,
                 extra_timings: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    cn, an, gn = model.class_names, model.attribute_names, model.graph_nodes
    terms = R["heuristic_terms"]
    final = R["heuristic_prob"][0]
    neural = R["diagnosis_prob"][0]
    fused = terms["fused_attr"][0]
    order = list(np.argsort(-final))
    att = R["attention"]

    # ---- images & overlays ---------------------------------------------
    c_thumb = thumbnail(clinical_img)
    d_thumb = thumbnail(dermoscopy_img)
    overlays: Dict[str, Any] = {}
    if R["lsm_score"] is not None and R["lsm_crop_centers"] is not None:
        s = R["lsm_score"][0]
        side = int(round(np.sqrt(s.size)))
        grid = s.reshape(side, side)
        boxes = crop_boxes(R["lsm_crop_centers"][0], c_thumb.size)
        overlays["lsm"] = {
            "image": _b64_jpeg(heat_overlay(c_thumb, grid, boxes=boxes)),
            "grid": _r(grid, 5), "side": side,
            "crops": [_b64_jpeg(Image.fromarray(
                (np.clip(cr.transpose(1, 2, 0), 0, 1) * 255).astype(np.uint8)), 88)
                for cr in R["lsm_crops"][0]],
            "centers_px": [b["center_px"] for b in boxes],
        }
    if "dermoscopy_region_cls_last" in att:
        g = att["dermoscopy_region_cls_last"].reshape(4, 4)
        overlays["dermoscopy_region"] = {
            "image": _b64_jpeg(heat_overlay(d_thumb, g, draw_grid=True)),
            "grid": _r(g, 5),
            "uniform_baseline": round(1.0 / 16, 5),
            "by_layer": _r(att["dermoscopy_region_cls_by_layer"], 5),
        }
    if "clinical_to_dermoscopy_windows" in att:
        g = att["clinical_to_dermoscopy_windows"].reshape(4, 4)
        overlays["cross_modal"] = {
            "clinical_to_dermoscopy_grid": _r(g, 5),
            "metadata_to_dermoscopy_grid": _r(att["metadata_to_dermoscopy_windows"].reshape(4, 4), 5)
            if "metadata_to_dermoscopy_windows" in att else None,
            "dermoscopy_to_clinical_crops": _r(att.get("dermoscopy_to_clinical_crops", []), 5),
            "metadata_to_clinical_crops": _r(att.get("metadata_to_clinical_crops", []), 5),
            "image": _b64_jpeg(heat_overlay(d_thumb, g, draw_grid=True)),
        }

    # ---- predictions --------------------------------------------------------
    classes = []
    for i, c in enumerate(cn):
        classes.append({
            "code": c, "name": CLASS_INFO[c]["name"], "group": CLASS_INFO[c]["group"],
            "color": CLASS_INFO[c]["color"],
            "final_prob": round(float(final[i]), 5), "neural_prob": round(float(neural[i]), 5),
            "logit": round(float(R["diagnosis_logits"][0][i]), 5),
            "upstream_prob": round(float(R["upstream_diagnosis_prob"][0][i]), 5),
            "rank": int(order.index(i)) + 1,
            "attribute_evidence": round(float(terms["attribute_evidence"][0][i]), 5),
            "graph_support": round(float(terms["graph_support"][0][i]), 5),
            "contradiction_penalty": round(float(terms["contradiction_penalty"][0][i]), 5),
            "interaction_bonus": round(float(terms["interaction_bonus"][0][i]), 5),
            "combined_score": round(float(terms["combined_score"][0][i]), 5),
        })
    attributes = []
    for j, a in enumerate(an):
        p_up = R["upstream_checklist_prob"][a][0]
        states = model.checklist_specs[a]["classes"]
        attributes.append({
            "code": a, "name": ATTRIBUTE_INFO[a]["name"], "plain": ATTRIBUTE_INFO[a]["plain"],
            "positive_meaning": ATTRIBUTE_INFO[a]["positive"],
            "points": ATTRIBUTE_INFO[a]["points"], "kind": ATTRIBUTE_INFO[a]["kind"],
            "aux_prob": round(float(R["attribute_prob"][0][j]), 5),
            "graph_prob": round(float(R["graph_attribute_prob"][0][j]), 5),
            "fused_prob": round(float(fused[j]), 5),
            "aux_logit": round(float(R["attribute_logits"][0][j]), 5),
            "graph_logit": round(float(R["graph_attribute_logits"][0][j]), 5),
            "detected": bool(fused[j] >= 0.5),
            "reliability": round(float(model.reliability[j]), 5),
            "attribute_weight": round(float(R["attribute_weights"][j]), 5),
            "upstream_states": [{"value": s, "label": STATE_LABELS.get(s, s), "prob": round(float(p), 5)}
                                for s, p in zip(states, p_up)],
            "top_state": STATE_LABELS.get(states[int(np.argmax(p_up))], states[int(np.argmax(p_up))]),
        })

    tokens = R["remix_tokens"][0]
    feats = R["case_node_features"][0]
    emb = R["graph_node_embeddings"][0]
    tnames = R["remix_token_names"]

    rt = dict(R["timings_seconds"])
    if extra_timings:
        rt.update(extra_timings)

    result = {
        "case": {
            "id": case_id,
            "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
            "images": {"clinical": _b64_jpeg(c_thumb), "dermoscopy": _b64_jpeg(d_thumb)},
            "original_sizes": {"clinical": list(clinical_img.size), "dermoscopy": list(dermoscopy_img.size)},
            "metadata": meta_readable,
        },
        "prediction": {
            "classes": classes,
            "top_class": cn[int(order[0])], "top_name": CLASS_INFO[cn[int(order[0])]]["name"],
            "top_prob": round(float(final[order[0]]), 5),
            "runner_up": cn[int(order[1])], "runner_up_name": CLASS_INFO[cn[int(order[1])]]["name"],
            "runner_up_prob": round(float(final[order[1]]), 5),
            "neural_top_class": R["neural_predicted_class"],
            "heuristic_changed_top": R["neural_predicted_class"] != R["final_predicted_class"],
            "uncertain": bool(final[order[0]] < 0.60 or final[order[0]] - final[order[1]] < 0.15),
        },
        "attributes": attributes,
        "weights": {
            "tokens": [{"name": n, "label": REMIX_TOKEN_INFO.get(n, n),
                        "weight": round(float(w), 5)} for n, w in zip(tnames, R["token_weights"])],
            "attributes": [{"code": a, "weight": round(float(w), 5)}
                           for a, w in zip(an, R["attribute_weights"])],
        },
        "heuristic": {
            "lambda": float(model.heuristic_lambda),
            "weights": terms["weights"],
            "graph_embedding_term_used": bool(terms["graph_embedding_term_used"]),
            "fused_attr": _r(fused, 5),
            "reliability": _r(model.reliability, 5),
            "contradictions": [
                {"class": c, "attributes": [
                    {"code": an[j], "fused": round(float(fused[j]), 4)}
                    for j in range(len(an)) if model.contradiction_mask[i, j] > 0]}
                for i, c in enumerate(cn)],
        },
        "graph": {
            "node_attention": _r(R["node_attention"][0], 5),
            "embedding_norms": _r(np.linalg.norm(emb, axis=1), 4),
        },
        "overlays": overlays,
        "internals": {
            "preprocessing": [
                _stat_row("Clinical image", clinical_t,
                          {"original": list(clinical_img.size), "resize": f"bicubic to {cfg.CLINICAL_SIZE}x{cfg.CLINICAL_SIZE}"}),
                _stat_row("Dermoscopy image", dermoscopy_t,
                          {"original": list(dermoscopy_img.size), "resize": f"bicubic to {cfg.DERMOSCOPY_SIZE}x{cfg.DERMOSCOPY_SIZE}"}),
                _stat_row("Metadata one-hot", metadata_t, {"original": None, "resize": "one-hot, 30 dimensions"}),
            ],
            "metadata_encoding": encoder.describe(meta_values),
            "metadata_vector": [int(x) for x in metadata_t.numpy().ravel()],
            "remix": {
                "feature_shape": list(R["remix_feature"].shape),
                "token_shape": list(R["remix_tokens"].shape),
                "token_names": tnames,
                "token_stats": [{"token": n, "label": REMIX_TOKEN_INFO.get(n, n),
                                 "l2": round(float(np.linalg.norm(tokens[i])), 4),
                                 "mean": round(float(tokens[i].mean()), 5),
                                 "std": round(float(tokens[i].std()), 5),
                                 "min": round(float(tokens[i].min()), 4),
                                 "max": round(float(tokens[i].max()), 4)} for i, n in enumerate(tnames)],
                "tokens": _r(tokens, 3),
                "upstream_diagnosis": [{"code": c, "prob": round(float(p), 5)}
                                       for c, p in zip(cn, R["upstream_diagnosis_prob"][0])],
                "msta_template_usage": _r(att["msta_template_usage"], 4) if "msta_template_usage" in att else None,
            },
            "graph_stage": {
                "node_features_shape": list(R["case_node_features"].shape),
                "node_features_heat": _r(feats, 3),
                "node_feature_norms": _r(np.linalg.norm(feats, axis=1), 4),
                "embedding_shape": list(R["graph_node_embeddings"].shape),
                "embeddings": _r(emb, 3),
                "embedding_stats": [{"node": n, "l2": round(float(np.linalg.norm(emb[i])), 4),
                                     "mean": round(float(emb[i].mean()), 5),
                                     "std": round(float(emb[i].std()), 5),
                                     "attention": round(float(R["node_attention"][0][i]), 5)}
                                    for i, n in enumerate(gn)],
            },
            "hybrid": {
                "fused_feature_shape": list(R["fused_feature"].shape),
                "diagnosis_logits": _r(R["diagnosis_logits"][0], 5),
                "attribute_logits": _r(R["attribute_logits"][0], 5),
                "graph_attribute_logits": _r(R["graph_attribute_logits"][0], 5),
            },
            "tensor_shapes": [{"tensor": k, "shape": list(np.asarray(R[k]).shape)} for k in
                              ("remix_feature", "remix_tokens", "case_node_features",
                               "graph_node_embeddings", "diagnosis_logits", "attribute_logits",
                               "graph_attribute_logits", "heuristic_prob")],
            "timings_seconds": {k: round(float(v), 3) for k, v in rt.items()},
        },
        "explanation": explanation,
        "case_summary": {k: v for k, v in case.items() if k != "query_terms"},
        "warnings": R["warnings"],
    }
    return result


def compact_json(obj: Any) -> str:
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)
