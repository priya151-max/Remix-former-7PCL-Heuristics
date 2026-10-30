"""Test 2: architecture reconstruction and strict weight loading."""
import pytest
import torch

from src import config as cfg
from src.checkpoint_loader import load_full_model


@pytest.fixture(scope="module")
def model():
    if not cfg.CHECKPOINT_PATH.is_file():
        pytest.skip("Checkpoint not present")
    return load_full_model(device="cpu")


def test_all_modules_built(model):
    assert model.remix is not None
    assert model.graph_encoder is not None
    assert model.hybrid is not None


def test_all_loads_were_strict(model):
    for rep in model.load_report["state_dict_loads"]:
        assert rep["strict"] is True
        assert rep["missing_keys"] == []
        assert rep["unexpected_keys"] == []
        assert rep["shape_mismatches"] == []


def test_eval_mode(model):
    for m in (model.remix, model.graph_encoder, model.hybrid):
        assert not m.training


def test_head_shapes(model):
    assert model.hybrid.classifier.out_features == 5
    assert model.hybrid.attribute_classifier.out_features == 7
    assert model.hybrid.attr_node_classifier.out_features == 1
    assert model.hybrid.node_attention.out_features == 1
    assert model.remix.classifier[-1].out_features == 5
    assert set(model.remix.checklist_heads) == set(model.attribute_names)
    assert model.remix.metadata.net[0].in_features == 30


def test_graph_encoder_shapes(model):
    assert model.graph_encoder.proj.in_features == 768
    assert model.graph_encoder.out.out_features == 256
    assert len(model.graph_encoder.order_layers) == model.graph_order + 1
