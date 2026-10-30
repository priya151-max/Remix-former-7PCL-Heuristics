"""
Checkpoint loader for the end-to-end RemixFormer++ / 7PCL package.

Reads all runtime configuration FROM THE CHECKPOINT. Nothing about class
names, attribute names, node order, dimensions, graph matrices, priors or
the heuristic lambda is hardcoded here.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from . import config as cfg
from .graph_reasoning import CorrectedHybrid7PCL, MultiOrderGraphEncoder
from .heuristic_reasoning import build_attr_class_matrix
from .model_architecture import RemixFormerPP

EXPECTED_FORMAT_VERSION = "remixformerpp_7pcl_GDDW_5class_graph_integrated_heuristic_v3"

REQUIRED_KEYS = [
    "format_version", "class_names", "attribute_names", "graph_nodes",
    "token_dim", "num_remix_tokens", "semantic_dim", "graph_hidden", "graph_order",
    "A_internal", "A_attr_to_class", "A_class_to_attr", "A_full", "A_sparse",
    "P", "semantic_matrix", "class_attr_prior", "attr_pair_prior", "reliability",
    "contradiction_mask", "heuristic_lambda", "heuristic_components",
    "hybrid_state_dict", "graph_encoder_state_dict", "upstream_remix_state_dict",
]


class CheckpointError(RuntimeError):
    pass


@dataclass
class LoadedModel:
    device: torch.device
    checkpoint_path: Path
    format_version: str

    class_names: List[str]
    attribute_names: List[str]
    graph_nodes: List[str]
    token_dim: int
    num_remix_tokens: int
    semantic_dim: int
    graph_hidden: int
    graph_order: int
    meta_dim: int
    checklist_specs: Dict[str, Any]

    A_internal: np.ndarray
    A_attr_to_class: np.ndarray
    A_class_to_attr: np.ndarray
    A_full: np.ndarray
    A_sparse: np.ndarray
    semantic_matrix: np.ndarray
    class_attr_prior: np.ndarray
    attr_pair_prior: np.ndarray
    reliability: np.ndarray
    contradiction_mask: np.ndarray
    attr_class: np.ndarray
    proximity: Dict[int, torch.Tensor]

    heuristic_lambda: float
    heuristic_components: List[str]

    remix: RemixFormerPP
    graph_encoder: MultiOrderGraphEncoder
    hybrid: CorrectedHybrid7PCL

    upstream_checkpoint_info: Dict[str, Any] = field(default_factory=dict)
    load_report: Dict[str, Any] = field(default_factory=dict)

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    @property
    def num_attributes(self) -> int:
        return len(self.attribute_names)


def _report_state_dict_load(module: torch.nn.Module, state_dict: dict, name: str) -> Dict[str, Any]:
    """
    Strict load with an explicit, non-silent report.

    A shape mismatch or missing key raises. Nothing is loaded with
    strict=False without the differences being surfaced.
    """
    own = module.state_dict()
    missing = sorted(set(own) - set(state_dict))
    unexpected = sorted(set(state_dict) - set(own))
    mismatched = [
        (k, tuple(own[k].shape), tuple(state_dict[k].shape))
        for k in set(own) & set(state_dict)
        if tuple(own[k].shape) != tuple(state_dict[k].shape)
    ]
    if missing or unexpected or mismatched:
        raise CheckpointError(
            f"Strict load failed for {name}.\n"
            f"Missing keys: {missing}\n"
            f"Unexpected keys: {unexpected}\n"
            f"Shape mismatches: {mismatched}\n"
            f"Reason for non-strict loading: not attempted -- the architecture "
            f"reconstructed from the notebook does not match the stored weights."
        )
    result = module.load_state_dict(state_dict, strict=True)
    module.eval()
    return {
        "module": name,
        "tensors_loaded": len(state_dict),
        "missing_keys": list(getattr(result, "missing_keys", [])),
        "unexpected_keys": list(getattr(result, "unexpected_keys", [])),
        "shape_mismatches": [],
        "strict": True,
    }


def _verify_upstream_config(upstream_sd: dict) -> Dict[str, Any]:
    """
    Independently verify META_DIM and the checklist head widths against the
    stored upstream weights, so the values recovered from notebook output are
    never taken on faith.
    """
    key = "metadata.net.0.weight"
    if key not in upstream_sd:
        raise CheckpointError(f"upstream_remix_state_dict is missing '{key}'.")
    meta_dim = int(upstream_sd[key].shape[1])
    if meta_dim != cfg.META_DIM:
        raise CheckpointError(
            f"Metadata dimension mismatch: checkpoint weights imply meta_dim={meta_dim}, "
            f"but src/config.py declares META_DIM={cfg.META_DIM}."
        )
    for name, spec in cfg.CHECKLIST_SPECS.items():
        wk = f"checklist_heads.{name}.weight"
        if wk not in upstream_sd:
            raise CheckpointError(f"upstream_remix_state_dict is missing '{wk}'.")
        n = int(upstream_sd[wk].shape[0])
        if n != spec["num_classes"]:
            raise CheckpointError(
                f"Checklist head '{name}' has {n} outputs in the checkpoint but "
                f"{spec['num_classes']} state labels are configured "
                f"({spec['classes']}). The state label list cannot be trusted; "
                f"recover it from the original upstream checkpoint "
                f"(RemixFormerPP_FULL_best_model.pt -> checklist_specs)."
            )
    return {"meta_dim": meta_dim, "verified_checklist_heads": list(cfg.CHECKLIST_SPECS)}


def load_full_model(checkpoint_path=None, device=None) -> LoadedModel:
    checkpoint_path = Path(checkpoint_path or cfg.CHECKPOINT_PATH)
    if not checkpoint_path.is_file():
        raise CheckpointError(f"Checkpoint not found: {checkpoint_path}")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    try:
        pkg = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise CheckpointError(f"torch.load failed for {checkpoint_path}: {exc}") from exc

    if not isinstance(pkg, dict):
        raise CheckpointError(f"Checkpoint must be a dict, got {type(pkg).__name__}.")

    missing = [k for k in REQUIRED_KEYS if k not in pkg]
    if missing:
        raise CheckpointError(f"Checkpoint is missing required keys: {missing}")

    fmt = str(pkg["format_version"])
    format_warning = None
    if fmt != EXPECTED_FORMAT_VERSION:
        format_warning = (f"format_version is '{fmt}', expected "
                          f"'{EXPECTED_FORMAT_VERSION}'. Proceeding, but the "
                          f"architecture may not match.")

    class_names = list(pkg["class_names"])
    attribute_names = list(pkg["attribute_names"])
    graph_nodes = list(pkg["graph_nodes"])
    token_dim = int(pkg["token_dim"])
    num_remix_tokens = int(pkg["num_remix_tokens"])
    semantic_dim = int(pkg["semantic_dim"])
    graph_hidden = int(pkg["graph_hidden"])
    graph_order = int(pkg["graph_order"])
    num_attr, num_cls = len(attribute_names), len(class_names)

    # --- structural validation of the stored graph -------------------------
    if graph_nodes != attribute_names + class_names:
        raise CheckpointError(
            f"Graph node order does not match attributes+classes.\n"
            f"graph_nodes = {graph_nodes}\n"
            f"expected    = {attribute_names + class_names}")
    for key, shape in [
        ("A_internal", (num_attr, num_attr)),
        ("A_attr_to_class", (num_attr, num_cls)),
        ("A_class_to_attr", (num_cls, num_attr)),
        ("A_full", (num_attr + num_cls,) * 2),
        ("A_sparse", (num_attr + num_cls,) * 2),
        ("semantic_matrix", (num_attr + num_cls, semantic_dim)),
        ("class_attr_prior", (num_cls, num_attr)),
        ("attr_pair_prior", (num_cls, num_attr, num_attr)),
        ("reliability", (num_attr,)),
        ("contradiction_mask", (num_cls, num_attr)),
    ]:
        actual = tuple(np.asarray(pkg[key]).shape)
        if actual != shape:
            raise CheckpointError(f"'{key}' has shape {actual}, expected {shape}.")

    proximity_raw = pkg["P"]
    proximity = {int(k): torch.tensor(np.asarray(v), dtype=torch.float32, device=device)
                 for k, v in proximity_raw.items()}
    for k in range(graph_order + 1):
        if k not in proximity:
            raise CheckpointError(f"Proximity matrix P[{k}] is missing (graph_order={graph_order}).")
        if tuple(proximity[k].shape) != (len(graph_nodes), len(graph_nodes)):
            raise CheckpointError(f"P[{k}] has shape {tuple(proximity[k].shape)}.")

    if attribute_names != list(cfg.CHECKLIST_SPECS):
        raise CheckpointError(
            f"Checkpoint attribute order {attribute_names} does not match the "
            f"configured checklist specs {list(cfg.CHECKLIST_SPECS)}.")

    # --- reconstruct and load ---------------------------------------------
    upstream_sd = pkg["upstream_remix_state_dict"]
    if not upstream_sd:
        raise CheckpointError(
            "upstream_remix_state_dict is empty. The upstream RemixFormer++ "
            "weights are required; without them no Remix tokens can be produced "
            "from images. Recover them from RemixFormerPP_FULL_best_model.pt.")
    upstream_info = _verify_upstream_config(upstream_sd)
    meta_dim = upstream_info["meta_dim"]

    remix = RemixFormerPP(meta_dim=meta_dim, checklist_specs=cfg.CHECKLIST_SPECS).to(device)
    graph_encoder = MultiOrderGraphEncoder(
        in_dim=semantic_dim, hidden=graph_hidden, orders=graph_order).to(device)
    hybrid = CorrectedHybrid7PCL(
        token_dim=token_dim, graph_dim=graph_hidden, num_tokens=num_remix_tokens,
        num_classes=num_cls, num_attr=num_attr).to(device)

    reports = [
        _report_state_dict_load(remix, upstream_sd, "RemixFormerPP"),
        _report_state_dict_load(graph_encoder, pkg["graph_encoder_state_dict"], "MultiOrderGraphEncoder"),
        _report_state_dict_load(hybrid, pkg["hybrid_state_dict"], "CorrectedHybrid7PCL"),
    ]
    for m in (remix, graph_encoder, hybrid):
        m.eval()

    A_attr_to_class = np.asarray(pkg["A_attr_to_class"], dtype=np.float64)

    return LoadedModel(
        device=device,
        checkpoint_path=checkpoint_path,
        format_version=fmt,
        class_names=class_names,
        attribute_names=attribute_names,
        graph_nodes=graph_nodes,
        token_dim=token_dim,
        num_remix_tokens=num_remix_tokens,
        semantic_dim=semantic_dim,
        graph_hidden=graph_hidden,
        graph_order=graph_order,
        meta_dim=meta_dim,
        checklist_specs=cfg.CHECKLIST_SPECS,
        A_internal=np.asarray(pkg["A_internal"], dtype=np.float64),
        A_attr_to_class=A_attr_to_class,
        A_class_to_attr=np.asarray(pkg["A_class_to_attr"], dtype=np.float64),
        A_full=np.asarray(pkg["A_full"], dtype=np.float64),
        A_sparse=np.asarray(pkg["A_sparse"], dtype=np.float64),
        semantic_matrix=np.asarray(pkg["semantic_matrix"], dtype=np.float32),
        class_attr_prior=np.asarray(pkg["class_attr_prior"], dtype=np.float64),
        attr_pair_prior=np.asarray(pkg["attr_pair_prior"], dtype=np.float64),
        reliability=np.asarray(pkg["reliability"], dtype=np.float64),
        contradiction_mask=np.asarray(pkg["contradiction_mask"], dtype=np.float64),
        attr_class=build_attr_class_matrix(A_attr_to_class, num_attr, num_cls),
        proximity=proximity,
        heuristic_lambda=float(pkg["heuristic_lambda"]),
        heuristic_components=list(pkg["heuristic_components"]),
        remix=remix,
        graph_encoder=graph_encoder,
        hybrid=hybrid,
        upstream_checkpoint_info=dict(pkg.get("upstream_checkpoint_info") or {}),
        load_report={
            "checkpoint_keys": sorted(pkg.keys()),
            "state_dict_loads": reports,
            "format_warning": format_warning,
            "device": str(device),
            "upstream_verification": upstream_info,
        },
    )
