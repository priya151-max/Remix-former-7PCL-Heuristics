"""
Inference preprocessing.

Mirrors `PaperEvalTransform` from notebook `7pcl-heuristic3 (1).ipynb` cell 7,
which is what validation and test used:

    img = img.resize((size, size), Image.Resampling.BICUBIC)
    return transforms.functional.to_tensor(img)

There is NO ImageNet mean/std normalisation in the training pipeline, so none
is applied here. Pixels stay in [0, 1].

Clinical size   = 896  (CLINICAL_SIZE)
Dermoscopy size = 1024 (DERMOSCOPY_SIZE)
"""
from typing import Tuple

import torch
from PIL import Image, ImageOps
from torchvision import transforms

from .config import CLINICAL_SIZE, DERMOSCOPY_SIZE


class PreprocessingError(ValueError):
    pass


def _eval_transform(img: Image.Image, size: int) -> torch.Tensor:
    img = img.resize((size, size), Image.Resampling.BICUBIC)
    return transforms.functional.to_tensor(img)


MAX_PIXELS = 100_000_000  # guard against decompression bombs


def load_image(file_like, label: str) -> Image.Image:
    if file_like is None:
        raise PreprocessingError(f"{label} image is missing.")
    try:
        Image.MAX_IMAGE_PIXELS = MAX_PIXELS
        img = Image.open(file_like)
        if img.format not in ("JPEG", "PNG", "MPO", "WEBP", "BMP", "TIFF"):
            raise ValueError(f"unsupported format {img.format}")
        img.load()
    except Exception as exc:
        raise PreprocessingError(
            f"The {label.lower()} image could not be read. Please upload a JPG or PNG photo ({exc}).")
    img = ImageOps.exif_transpose(img)  # honour phone-camera orientation
    if img.mode != "RGB":
        img = img.convert("RGB")
    if min(img.size) < 32:
        raise PreprocessingError(f"The {label.lower()} image is too small ({img.size[0]}x{img.size[1]} px). Please upload a larger image.")
    return img


def preprocess_clinical_image(img: Image.Image) -> torch.Tensor:
    """-> [1, 3, 896, 896], float32 in [0, 1]."""
    return _eval_transform(img, CLINICAL_SIZE).unsqueeze(0)


def preprocess_dermoscopy_image(img: Image.Image) -> torch.Tensor:
    """-> [1, 3, 1024, 1024], float32 in [0, 1]."""
    return _eval_transform(img, DERMOSCOPY_SIZE).unsqueeze(0)


def validate_input_shapes(clinical: torch.Tensor, dermoscopy: torch.Tensor,
                          metadata: torch.Tensor, meta_dim: int) -> Tuple[bool, str]:
    checks = [
        (tuple(clinical.shape) == (1, 3, CLINICAL_SIZE, CLINICAL_SIZE),
         f"clinical tensor is {tuple(clinical.shape)}, expected (1, 3, {CLINICAL_SIZE}, {CLINICAL_SIZE})"),
        (tuple(dermoscopy.shape) == (1, 3, DERMOSCOPY_SIZE, DERMOSCOPY_SIZE),
         f"dermoscopy tensor is {tuple(dermoscopy.shape)}, expected (1, 3, {DERMOSCOPY_SIZE}, {DERMOSCOPY_SIZE})"),
        (tuple(metadata.shape) == (1, meta_dim),
         f"metadata tensor is {tuple(metadata.shape)}, expected (1, {meta_dim})"),
        (clinical.dtype == torch.float32 and dermoscopy.dtype == torch.float32
         and metadata.dtype == torch.float32, "all input tensors must be float32"),
    ]
    failures = [msg for ok, msg in checks if not ok]
    return (not failures), "; ".join(failures)
