"""
End-to-end inference pipeline.

Stage order follows the notebook exactly:

  images + metadata
    -> RemixFormerPP.forward           (feature 3840 -> reshape [B,5,768])
    -> upstream checklist softmax probabilities (per attribute, multi-class)
    -> make_case_node_features         -> [B,12,768]
    -> MultiOrderGraphEncoder          -> [B,12,256]
    -> CorrectedHybrid7PCL             -> 5-class logits + 7 attribute logits
    -> graph_integrated_heuristic_probs

Every number returned here comes from the trained weights or the stored
checkpoint matrices. Nothing is simulated.
"""
import time
from typing import Any, Callable, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

from .checkpoint_loader import LoadedModel
from .config import REMIX_TOKEN_NAMES
from .graph_reasoning import make_case_node_features
from .heuristic_reasoning import graph_integrated_heuristic_probs


@torch.inference_mode()
def run_inference(model: LoadedModel,
                  clinical: torch.Tensor,
                  dermoscopy: torch.Tensor,
                  metadata: torch.Tensor,
                  on_stage: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
    dev = model.device
    t0 = time.perf_counter()
    warnings = []

    def stage(name: str) -> None:
        if on_stage is not None:
            on_stage(name)

    clinical = clinical.to(dev)
    dermoscopy = dermoscopy.to(dev)
    metadata = metadata.to(dev)

    # ---- Stage 1: RemixFormer++ ------------------------------------------
    stage("remix")
    out = model.remix(clinical, dermoscopy, metadata, capture=True)
    feature = out["feature"].float()
    expected_width = model.num_remix_tokens * model.token_dim
    if feature.shape[-1] != expected_width:
        raise RuntimeError(
            f"RemixFormer++ produced a {tuple(feature.shape)} feature; "
            f"expected last dimension {expected_width}.")
    tokens = feature.reshape(feature.shape[0], model.num_remix_tokens, model.token_dim)
    t_remix = time.perf_counter()

    upstream_diag_prob = F.softmax(out["diagnosis"], dim=-1).float().cpu().numpy()
    check_prob = {name: F.softmax(logits, dim=-1).float().cpu().numpy()
                  for name, logits in out["checklist"].items()}

    # ---- Stage 2: graph node features ------------------------------------
    stage("graph")
    node_features = make_case_node_features(
        check_prob=check_prob,
        diag_prob=upstream_diag_prob,
        attr_names=model.attribute_names,
        class_names=model.class_names,
        checklist_specs=model.checklist_specs,
        semantic_matrix=model.semantic_matrix,
        sem_dim=model.semantic_dim,
    )
    node_tensor = torch.as_tensor(node_features, dtype=torch.float32, device=dev)

    # ---- Stage 3: graph encoder ------------------------------------------
    graph_nodes = model.graph_encoder(node_tensor, model.proximity)
    t_graph = time.perf_counter()

    # ---- Stage 4: 7PCL hybrid --------------------------------------------
    hy = model.hybrid(tokens, graph_nodes)
    diag_logits = hy["diagnosis_logits"].float().cpu().numpy()
    diag_prob = hy["diagnosis_prob"].float().cpu().numpy()
    attr_logits = hy["attribute_logits"].float().cpu().numpy()
    attr_prob = hy["attribute_prob"].float().cpu().numpy()
    graph_attr_logits = hy["graph_attribute_logits"].float().cpu().numpy()
    graph_attr_prob = hy["graph_attribute_prob"].float().cpu().numpy()
    node_attention = hy["node_attention"].float().cpu().numpy()
    token_weights = hy["token_weights"].float().cpu().numpy()
    attribute_weights = hy["attribute_weights"].float().cpu().numpy()
    graph_nodes_np = graph_nodes.float().cpu().numpy()
    t_hybrid = time.perf_counter()

    # ---- Stage 5: graph-integrated heuristic ------------------------------
    stage("heuristic")
    heur_prob, heur_terms = graph_integrated_heuristic_probs(
        diag_prob, attr_prob, graph_attr_prob, graph_nodes_np, node_attention,
        lam=model.heuristic_lambda,
        class_attr_prior=model.class_attr_prior,
        attr_pair_prior=model.attr_pair_prior,
        reliability=model.reliability,
        contradiction_mask=model.contradiction_mask,
        attr_class=model.attr_class,
        num_classes=model.num_classes,
        num_attr=model.num_attributes,
    )
    t_end = time.perf_counter()

    attention = _collect_attention(model, out)

    neural_idx = int(diag_prob[0].argmax())
    heur_idx = int(heur_prob[0].argmax())
    if neural_idx != heur_idx:
        warnings.append(
            f"The heuristic changed the top class from "
            f"'{model.class_names[neural_idx]}' to '{model.class_names[heur_idx]}'.")

    return {
        # stage 1
        "remix_feature": feature.float().cpu().numpy(),
        "remix_tokens": tokens.float().cpu().numpy(),
        "remix_token_names": REMIX_TOKEN_NAMES[:model.num_remix_tokens],
        "upstream_diagnosis_logits": out["diagnosis"].float().cpu().numpy(),
        "upstream_diagnosis_prob": upstream_diag_prob,
        "upstream_checklist_prob": check_prob,
        "lsm_score": (model.remix.clinical.last_lsm_score.float().cpu().numpy()
                      if getattr(model.remix.clinical, "last_lsm_score", None) is not None else None),
        "lsm_crops": (model.remix.clinical.last_crops.float().cpu().numpy()
                      if getattr(model.remix.clinical, "last_crops", None) is not None else None),
        "lsm_crop_centers": (model.remix.clinical.last_crop_centers.float().cpu().numpy()
                             if getattr(model.remix.clinical, "last_crop_centers", None) is not None else None),
        "attention": attention,
        # stage 2/3
        "case_node_features": node_features,
        "graph_node_embeddings": graph_nodes_np,
        # stage 4
        "diagnosis_logits": diag_logits,
        "diagnosis_prob": diag_prob,
        "attribute_logits": attr_logits,
        "attribute_prob": attr_prob,
        "graph_attribute_logits": graph_attr_logits,
        "graph_attribute_prob": graph_attr_prob,
        "node_attention": node_attention,
        "token_weights": token_weights,
        "attribute_weights": attribute_weights,
        "fused_feature": hy["feature"].float().cpu().numpy(),
        # stage 5
        "heuristic_prob": heur_prob,
        "heuristic_terms": heur_terms,
        # summary
        "neural_predicted_index": neural_idx,
        "neural_predicted_class": model.class_names[neural_idx],
        "final_predicted_index": heur_idx,
        "final_predicted_class": model.class_names[heur_idx],
        "timings_seconds": {
            "remixformer": t_remix - t0,
            "graph_encoder": t_graph - t_remix,
            "hybrid": t_hybrid - t_graph,
            "heuristic": t_end - t_hybrid,
            "total": t_end - t0,
        },
        "warnings": warnings,
        "device": str(dev),
    }


def _collect_attention(model: LoadedModel, out: Dict[str, Any]) -> Dict[str, Any]:
    """
    Genuine attention tensors captured by RemixFormerPP.forward(capture=True).
    Each entry is reduced only by averaging over heads / summing over tokens that
    belong to the same crop; nothing is invented.
    """
    att: Dict[str, Any] = {}

    last = getattr(model.remix.dermoscopy, "last", None) or {}
    layers = last.get("region_attention")
    if layers:
        # each: [B, heads, 17, 17]; row 0 = region CLS token, columns 1..16 = windows
        per_layer = np.stack(
            [l[0].float().mean(dim=0)[0, 1:].cpu().numpy() for l in layers], axis=0)  # [L,16]
        att["dermoscopy_region_cls_by_layer"] = per_layer
        att["dermoscopy_region_cls_last"] = per_layer[-1]
    msta = last.get("msta_weights")
    if msta is not None:
        att["msta_template_usage"] = msta[0].float().mean(dim=1).cpu().numpy()  # [16, 32]

    cmf = (out.get("fused") or {}).get("attention") or {}

    def _mean_heads(t):
        return None if t is None else t[0].float().mean(dim=0)[0].cpu().numpy()  # [N]

    a = _mean_heads(cmf.get("clinical_from_dermoscopy"))
    if a is not None:
        att["clinical_to_dermoscopy_windows"] = a                      # [16]
    a = _mean_heads(cmf.get("metadata_from_dermoscopy"))
    if a is not None:
        att["metadata_to_dermoscopy_windows"] = a                      # [16]
    for key, name in (("dermoscopy_from_clinical", "dermoscopy_to_clinical_crops"),
                      ("metadata_from_clinical", "metadata_to_clinical_crops")):
        a = _mean_heads(cmf.get(key))
        if a is not None:
            k = int(model.remix.clinical.last_crops.shape[1])   # number of LSM crops
            if a.size % k == 0:
                att[name] = a.reshape(k, -1).sum(axis=1)                 # [K]
    return att
