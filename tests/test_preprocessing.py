"""Test 3: preprocessing shapes and dtypes."""
import numpy as np
import torch
from PIL import Image

from src.config import CLINICAL_SIZE, DERMOSCOPY_SIZE
from src.preprocessing import (preprocess_clinical_image, preprocess_dermoscopy_image,
                               validate_input_shapes)


def _img(w=640, h=480):
    return Image.fromarray(np.random.randint(0, 255, (h, w, 3), dtype=np.uint8), "RGB")


def test_clinical_shape():
    t = preprocess_clinical_image(_img())
    assert tuple(t.shape) == (1, 3, CLINICAL_SIZE, CLINICAL_SIZE)
    assert t.dtype == torch.float32


def test_dermoscopy_shape():
    t = preprocess_dermoscopy_image(_img(800, 600))
    assert tuple(t.shape) == (1, 3, DERMOSCOPY_SIZE, DERMOSCOPY_SIZE)


def test_no_imagenet_normalisation():
    """Training used to_tensor only, so values must stay within [0, 1]."""
    t = preprocess_clinical_image(_img())
    assert float(t.min()) >= 0.0 and float(t.max()) <= 1.0


def test_validate_input_shapes():
    c = preprocess_clinical_image(_img())
    d = preprocess_dermoscopy_image(_img())
    m = torch.zeros(1, 30, dtype=torch.float32)
    ok, msg = validate_input_shapes(c, d, m, 30)
    assert ok, msg
    bad, msg = validate_input_shapes(c, d, torch.zeros(1, 29), 30)
    assert not bad and "metadata" in msg
