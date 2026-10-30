"""Test 1: checkpoint file, keys, and stored configuration."""
import numpy as np
import pytest
import torch

from src import config as cfg
from src.checkpoint_loader import EXPECTED_FORMAT_VERSION, REQUIRED_KEYS


@pytest.fixture(scope="module")
def pkg():
    if not cfg.CHECKPOINT_PATH.is_file():
        pytest.skip(f"Checkpoint not present at {cfg.CHECKPOINT_PATH}")
    return torch.load(cfg.CHECKPOINT_PATH, map_location="cpu", weights_only=False)


def test_file_exists():
    assert cfg.CHECKPOINT_PATH.is_file(), f"missing {cfg.CHECKPOINT_PATH}"


def test_required_keys(pkg):
    assert not [k for k in REQUIRED_KEYS if k not in pkg]


def test_format_version(pkg):
    assert pkg["format_version"] == EXPECTED_FORMAT_VERSION


def test_names_and_dims(pkg):
    assert pkg["class_names"] == ["BCC", "Nev", "Mel", "SK", "Misc"]
    assert pkg["attribute_names"] == ["PN", "BWV", "VS", "PIG", "STR", "DaG", "RS"]
    assert pkg["graph_nodes"] == pkg["attribute_names"] + pkg["class_names"]
    assert len(pkg["graph_nodes"]) == 12
    assert pkg["token_dim"] == 768 and pkg["num_remix_tokens"] == 5
    assert pkg["semantic_dim"] == 768 and pkg["graph_hidden"] == 256
    assert pkg["graph_order"] == 3


def test_state_dicts_nonempty(pkg):
    for k in ["hybrid_state_dict", "graph_encoder_state_dict", "upstream_remix_state_dict"]:
        assert pkg[k], f"{k} is empty"
    assert len(pkg["upstream_remix_state_dict"]) == 733


def test_heuristic_config(pkg):
    lam = float(pkg["heuristic_lambda"])
    assert lam > 0.0
    assert pkg["heuristic_components"] == [
        "attribute_evidence", "graph_support", "contradiction_penalty", "interaction_bonus"]


def test_matrix_shapes(pkg):
    assert np.asarray(pkg["A_full"]).shape == (12, 12)
    assert np.asarray(pkg["A_attr_to_class"]).shape == (7, 5)
    assert np.asarray(pkg["semantic_matrix"]).shape == (12, 768)
    assert np.asarray(pkg["attr_pair_prior"]).shape == (5, 7, 7)
    assert set(map(int, pkg["P"].keys())) >= {0, 1, 2, 3}
