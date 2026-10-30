"""Test 4: graph construction, node order, and encoder dimensions."""
import numpy as np
import pytest
import torch

from src import config as cfg
from src.checkpoint_loader import load_full_model
from src.graph_reasoning import deterministic_embedding, make_case_node_features


@pytest.fixture(scope="module")
def model():
    if not cfg.CHECKPOINT_PATH.is_file():
        pytest.skip("Checkpoint not present")
    return load_full_model(device="cpu")


def test_twelve_nodes_in_order(model):
    assert len(model.graph_nodes) == 12
    assert model.graph_nodes == model.attribute_names + model.class_names


def test_matrix_dims(model):
    assert model.A_full.shape == (12, 12)
    assert model.A_sparse.shape == (12, 12)
    assert model.semantic_matrix.shape == (12, model.semantic_dim)
    assert model.attr_class.shape == (7, 5)


def test_deterministic_embedding_is_stable():
    a = deterministic_embedding("PN: typical")
    b = deterministic_embedding("PN: typical")
    assert np.allclose(a, b)
    assert not np.allclose(a, deterministic_embedding("PN: atypical"))
    assert abs(float(np.linalg.norm(a)) - 1.0) < 1e-5


def test_case_node_feature_shape(model):
    check_prob = {a: np.full((1, model.checklist_specs[a]["num_classes"]),
                             1.0 / model.checklist_specs[a]["num_classes"], dtype=np.float32)
                  for a in model.attribute_names}
    diag = np.full((1, 5), 0.2, dtype=np.float32)
    X = make_case_node_features(check_prob, diag, model.attribute_names,
                                model.class_names, model.checklist_specs,
                                model.semantic_matrix, model.semantic_dim)
    assert X.shape == (1, 12, 768)


def test_graph_encoder_output_shape(model):
    x = torch.randn(1, 12, model.semantic_dim)
    with torch.inference_mode():
        out = model.graph_encoder(x, model.proximity)
    assert tuple(out.shape) == (1, 12, model.graph_hidden)
