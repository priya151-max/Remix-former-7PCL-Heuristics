"""
Downstream graph components.

MultiOrderGraphEncoder and CorrectedHybrid7PCL are EXTRACTED VERBATIM from
notebook `7pcl-heuristic3 (1).ipynb` cells 46 and 47. Only the notebook-global
default arguments (SEM_DIM, GRAPH_HIDDEN, GRAPH_K) were turned into explicit
constructor arguments; every layer and the forward pass are unchanged.

make_case_node_features / soft_state_embedding / deterministic_embedding are
extracted verbatim from cells 43 and 44 and reduced to a single-case path.
"""
import hashlib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------
# Cell 46 -- differentiable multi-order graph encoder (verbatim)
# --------------------------------------------------------------------------
class MultiOrderGraphEncoder(nn.Module):
    def __init__(self, in_dim, hidden, orders):
        super().__init__()
        self.orders = orders
        self.proj = nn.Linear(in_dim, hidden)
        self.order_layers = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(orders + 1)])
        self.norm = nn.LayerNorm(hidden)
        self.out = nn.Linear(hidden, hidden)

    def forward(self, x, proximity):
        # x [B,N,D], P[k] [N,N]. P(0) is identity.
        h0 = F.gelu(self.proj(x))
        pieces = [self.order_layers[0](h0)]
        for k in range(1, self.orders + 1):
            pk = proximity[k].to(device=x.device, dtype=x.dtype)
            hk = torch.einsum("ij,bjd->bid", pk, h0)
            pieces.append(self.order_layers[k](hk))
        h = torch.stack(pieces, dim=0).mean(dim=0)
        return self.out(F.gelu(self.norm(h)))


# --------------------------------------------------------------------------
# Cell 47 -- corrected five-class GD-DDW fusion (verbatim)
# --------------------------------------------------------------------------
class CorrectedHybrid7PCL(nn.Module):
    def __init__(self, token_dim=768, graph_dim=256, hidden=256,
                 num_tokens=5, num_classes=5, num_attr=7):
        super().__init__()
        self.num_tokens = num_tokens
        self.num_attr = num_attr
        self.token_adapters = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(token_dim), nn.Linear(token_dim, hidden), nn.GELU())
            for _ in range(num_tokens)
        ])
        self.graph_adapter = nn.Sequential(nn.LayerNorm(graph_dim), nn.Linear(graph_dim, hidden), nn.GELU())
        self.token_graph_fuse = nn.Sequential(
            nn.LayerNorm(hidden * 2), nn.Linear(hidden * 2, hidden), nn.GELU(), nn.Dropout(0.10))
        self.classifier = nn.Linear(hidden, num_classes)
        self.attribute_classifier = nn.Linear(hidden, num_attr)
        self.attr_node_classifier = nn.Linear(hidden, 1)
        self.raw_token_weights = nn.Parameter(torch.zeros(num_tokens))
        self.raw_attribute_weights = nn.Parameter(torch.zeros(num_attr))
        self.node_attention = nn.Linear(hidden, 1)

    def forward(self, remix_tokens, graph_nodes):
        # remix_tokens [B,5,768], graph_nodes [B,12,H]
        token_w = F.softmax(self.raw_token_weights, dim=0)
        attr_w = F.softmax(self.raw_attribute_weights, dim=0)
        token_features = torch.stack(
            [self.token_adapters[i](remix_tokens[:, i]) for i in range(self.num_tokens)], dim=1)
        # Graph pooling is attention-based over the full 12-node graph.
        node_scores = self.node_attention(graph_nodes).squeeze(-1)
        node_alpha = F.softmax(node_scores, dim=1)
        graph_global = torch.sum(node_alpha.unsqueeze(-1) * graph_nodes, dim=1)
        g = self.graph_adapter(graph_global)
        # Each Remix token is explicitly conditioned on the graph.
        conditioned = []
        for i in range(self.num_tokens):
            conditioned.append(self.token_graph_fuse(torch.cat([token_features[:, i], g], dim=-1)))
        conditioned = torch.stack(conditioned, dim=1)
        z = torch.sum(conditioned * token_w.view(1, -1, 1), dim=1)
        # Attribute evidence is predicted from the seven attribute node embeddings.
        attr_nodes = graph_nodes[:, :self.num_attr]
        attr_logits = self.attr_node_classifier(attr_nodes).squeeze(-1)
        # The learned attribute weights are used to create a graph evidence vector.
        weighted_attr = torch.sum(torch.sigmoid(attr_logits) * attr_w.view(1, -1), dim=1, keepdim=True)
        z_aug = z + weighted_attr * torch.tanh(z)
        diag_logits = self.classifier(z_aug)
        # Auxiliary attribute head from the fused representation.
        aux_attr_logits = self.attribute_classifier(z_aug)
        return {
            "feature": z_aug,
            "diagnosis_logits": diag_logits,
            "attribute_logits": aux_attr_logits,
            "graph_attribute_logits": attr_logits,
            "diagnosis_prob": F.softmax(diag_logits, dim=-1),
            "attribute_prob": torch.sigmoid(aux_attr_logits),
            "graph_attribute_prob": torch.sigmoid(attr_logits),
            "token_weights": token_w,
            "attribute_weights": attr_w,
            "node_attention": node_alpha,
        }


# --------------------------------------------------------------------------
# Cell 43 / 44 -- case-specific graph node features
# --------------------------------------------------------------------------
def deterministic_embedding(text, dim=768):
    """Verbatim from cell 43."""
    seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:4], "little")
    rng = np.random.default_rng(seed)
    v = rng.normal(0, 1, size=dim).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-8)


def soft_state_embedding(prob, names, base_name, dim=768):
    """Verbatim from cell 44. prob: [B, S] -> [B, D]."""
    vecs = [torch.tensor(deterministic_embedding(f"{base_name}: {state_name}", dim),
                         dtype=torch.float32) for state_name in names]
    E = torch.stack(vecs, dim=0)
    return torch.as_tensor(prob, dtype=torch.float32) @ E


def make_case_node_features(check_prob, diag_prob, attr_names, class_names,
                            checklist_specs, semantic_matrix, sem_dim=768):
    """
    Verbatim logic from cell 44, single-batch path.

    Attribute nodes  : probability-weighted state semantics from
                       deterministic_embedding("<attr>: <state>").
    Diagnosis nodes  : per-class semantic vector scaled by that class's
                       probability. The class semantic vectors come from the
                       checkpoint's `semantic_matrix` (rows 7..11), so if the
                       notebook used BERT embeddings the deployed values match.

    Returns X of shape [B, 12, sem_dim].
    """
    num_attr = len(attr_names)
    num_classes = len(class_names)
    n = int(np.asarray(diag_prob).shape[0])
    X = np.zeros((n, num_attr + num_classes, sem_dim), dtype=np.float32)

    for j, name in enumerate(attr_names):
        spec = checklist_specs[name]
        X[:, j] = soft_state_embedding(check_prob[name], spec["classes"], name, sem_dim).numpy()

    diag_prob = np.asarray(diag_prob, dtype=np.float32)
    if diag_prob.ndim != 2 or diag_prob.shape[1] != num_classes:
        raise ValueError(f"diag_prob must have shape [N, {num_classes}], but received {diag_prob.shape}")

    diag_E = np.asarray(semantic_matrix, dtype=np.float32)[num_attr:num_attr + num_classes]
    if diag_E.shape != (num_classes, sem_dim):
        raise ValueError(
            f"Diagnosis semantic matrix must have shape [{num_classes}, {sem_dim}], "
            f"but received {diag_E.shape}")

    for k in range(num_classes):
        X[:, num_attr + k, :] = diag_prob[:, k:k + 1] * diag_E[k:k + 1, :]
    return X
