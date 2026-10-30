import io
import sys
from pathlib import Path

import pytest
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config as cfg  # noqa: E402


def synth_image(size=900, w=None):
    w = w or size
    img = Image.new("RGB", (w, size), (222, 178, 158))
    d = ImageDraw.Draw(img)
    d.ellipse([w * .3, size * .28, w * .7, size * .68], fill=(95, 55, 38))
    d.ellipse([w * .4, size * .4, w * .55, size * .55], fill=(45, 28, 26))
    return img.filter(ImageFilter.GaussianBlur(2))


def jpeg_bytes(img):
    b = io.BytesIO()
    img.save(b, "JPEG", quality=88)
    return b.getvalue()


@pytest.fixture(scope="session")
def service():
    if not cfg.CHECKPOINT_PATH.is_file():
        pytest.skip("Checkpoint not present")
    from src.pipeline import AnalysisService
    return AnalysisService.load(device="cpu")


@pytest.fixture(scope="session")
def analysis(service):
    """A full analysis result produced by the real model (LLM disabled -> deterministic explanation)."""
    return service.analyze(jpeg_bytes(synth_image(900, 1200)), jpeg_bytes(synth_image(1000)),
                           {"sex": "female", "location": "back", "elevation": "flat"}, use_llm=False)
