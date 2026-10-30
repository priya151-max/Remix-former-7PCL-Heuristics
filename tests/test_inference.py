"""Full forward pass on the real checkpoint."""
import numpy as np
import pytest
import torch
from PIL import Image

from src import config as cfg
from src.checkpoint_loader import load_full_model
from src.prediction import run_inference
from src.preprocessing import preprocess_clinical_image, preprocess_dermoscopy_image


@pytest.fixture(scope="module")
def model():
    if not cfg.CHECKPOINT_PATH.is_file():
        pytest.skip("Checkpoint not present")
    return load_full_model(device="cpu")


@pytest.fixture(scope="module")
def result(model):
    img = Image.fromarray(np.random.RandomState(0).randint(0, 255, (512, 512, 3), dtype=np.uint8), "RGB")
    m = torch.zeros(1, model.meta_dim)
    m[0, 0] = 1.0
    return run_inference(model, preprocess_clinical_image(img), preprocess_dermoscopy_image(img), m)


def test_probabilities_are_valid(result):
    for k in ("diagnosis_prob", "heuristic_prob"):
        p = result[k][0]
        assert p.shape == (5,) and np.isclose(p.sum(), 1.0, atol=1e-4) and (p >= 0).all()


def test_attention_and_crops_captured(result):
    a = result["attention"]
    assert a["dermoscopy_region_cls_last"].shape == (16,)
    assert a["dermoscopy_region_cls_by_layer"].shape == (12, 16)
    assert result["lsm_crop_centers"].shape == (1, 4, 2)


def test_stage_callback():
    # covered through the pipeline test; here we only check the callback signature is optional
    import inspect
    assert "on_stage" in inspect.signature(run_inference).parameters
