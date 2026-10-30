"""
Graph-integrated reliability-gated clinical heuristic.

EXTRACTED VERBATIM from notebook `7pcl-heuristic3 (1).ipynb` cell 54
(`graph_integrated_heuristic_probs`). Notebook globals (NUM_CLASSES, NUM_ATTR,
class_attr_prior, attr_pair_prior, reliability, contradiction_mask,
_attr_class) are passed in explicitly instead of read from module scope. The
arithmetic is unchanged.

`_attr_class` is not stored directly in the checkpoint; it is recomputed here
using exactly the notebook expression, from the checkpoint's A_attr_to_class.
"""
import numpy as np

from .config import (
    EPS,
    HEURISTIC_GRAPH_WEIGHT,
    HEURISTIC_CONTRADICTION_WEIGHT,
    HEURISTIC_INTERACTION_WEIGHT,
)


def _safe_softmax(x):
    """Verbatim from cell 54."""
    x = x - np.max(x, axis=1, keepdims=True)
    e = np.exp(np.clip(x, -50, 50))
    return e / np.maximum(e.sum(axis=1, keepdims=True), EPS)


def build_attr_class_matrix(A_attr_to_class, num_attr, num_classes):
    """Verbatim from cell 54 (the `_attr_class` block)."""
    _attr_class = np.asarray(A_attr_to_class, dtype=np.float64)
    if _attr_class.shape != (num_attr, num_classes):
        _attr_class = np.ones((num_attr, num_classes), dtype=np.float64)
    _attr_class = np.maximum(_attr_class, 0.0)
    _attr_class = _attr_class / np.maximum(_attr_class.sum(axis=0, keepdims=True), EPS)
    return _attr_class


def graph_integrated_heuristic_probs(
    diag_prob, attr_prob, graph_attr_prob=None, graph_nodes=None, node_attention=None,
    lam=1.0,
    graph_weight=HEURISTIC_GRAPH_WEIGHT,
    contradiction_weight=HEURISTIC_CONTRADICTION_WEIGHT,
    interaction_weight=HEURISTIC_INTERACTION_WEIGHT,
    *,
    class_attr_prior, attr_pair_prior, reliability, contradiction_mask, attr_class,
    num_classes, num_attr,
):
    """
    Graph-integrated heuristic.

    Terms:
      1) reliability-gated attribute likelihood evidence
      2) directed CKTG attribute-to-class graph support
      3) graph embedding support using attribute/class node similarity
      4) contradiction penalty
      5) nonlinear pairwise attribute interaction bonus
    """
    p = np.clip(np.asarray(diag_prob, dtype=np.float64), EPS, 1.0 - EPS)
    a = np.clip(np.asarray(attr_prob, dtype=np.float64), EPS, 1.0 - EPS)
    ga = a if graph_attr_prob is None else np.clip(
        np.asarray(graph_attr_prob, dtype=np.float64), EPS, 1.0 - EPS)
    B = len(p)

    # Fuse the auxiliary attribute head and the graph attribute head.
    fused_attr = 0.5 * (a + ga)
    logp = np.log(p)
    evidence = np.zeros((B, num_classes), dtype=np.float64)
    graph_support = np.zeros_like(evidence)
    contradiction = np.zeros_like(evidence)
    interaction = np.zeros_like(evidence)

    for c in range(num_classes):
        q = np.clip(class_attr_prior[c], EPS, 1.0 - EPS)
        attr_ll = fused_attr * np.log(q) + (1.0 - fused_attr) * np.log(1.0 - q)
        evidence[:, c] = (attr_ll * reliability[None, :]).sum(axis=1)

        # Directed CKTG support: attribute evidence routed through graph edges to class c.
        edge_w = attr_class[:, c][None, :]
        graph_support[:, c] = (fused_attr * edge_w * reliability[None, :]).sum(axis=1)

        # Penalize attributes that are strongly present but unlikely for class c.
        contradiction[:, c] = ((fused_attr * contradiction_mask[c][None, :]) * reliability[None, :]).sum(axis=1)

        # Pairwise nonlinear bonus for co-occurring attributes.
        for i in range(num_attr):
            for j in range(i + 1, num_attr):
                expected = attr_pair_prior[c, i, j]
                interaction[:, c] += (fused_attr[:, i] * fused_attr[:, j]) * expected * reliability[i] * reliability[j]

    # Graph embedding support: compare attribute-node and class-node representations.
    graph_embedding_used = False
    if graph_nodes is not None:
        gn = np.asarray(graph_nodes, dtype=np.float64)
        if gn.ndim == 3 and gn.shape[1] >= num_attr + num_classes:
            graph_embedding_used = True
            attr_g = gn[:, :num_attr, :]
            class_g = gn[:, num_attr:num_attr + num_classes, :]
            attr_g = attr_g / np.maximum(np.linalg.norm(attr_g, axis=-1, keepdims=True), EPS)
            class_g = class_g / np.maximum(np.linalg.norm(class_g, axis=-1, keepdims=True), EPS)
            sim = np.einsum('bah,bch->bac', attr_g, class_g)
            if node_attention is not None:
                na = np.asarray(node_attention, dtype=np.float64)[:, :num_attr]
                na = np.exp(na - na.max(axis=1, keepdims=True))
                na = na / np.maximum(na.sum(axis=1, keepdims=True), EPS)
                sim = sim * na[:, :, None]
            graph_support += 0.5 * np.sum(sim * fused_attr[:, :, None], axis=1)

    logits = logp + float(lam) * (evidence + graph_weight * graph_support
                                  - contradiction_weight * contradiction
                                  + interaction_weight * interaction)
    terms = {
        "attribute_evidence": evidence,
        "graph_support": graph_support,
        "contradiction_penalty": contradiction,
        "interaction_bonus": interaction,
        "combined_score": evidence + graph_weight * graph_support
                          - contradiction_weight * contradiction
                          + interaction_weight * interaction,
        "fused_attr": fused_attr,
        "neural_log_prob": logp,
        "graph_embedding_term_used": graph_embedding_used,
        "lambda": float(lam),
        "weights": {
            "graph_weight": float(graph_weight),
            "contradiction_weight": float(contradiction_weight),
            "interaction_weight": float(interaction_weight),
        },
    }
    return _safe_softmax(logits), terms
