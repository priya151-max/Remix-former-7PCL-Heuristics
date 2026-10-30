"""
Analysis service: the single entry point used by the web layer.

    service = AnalysisService.load()
    result  = service.analyze(clinical_bytes, dermoscopy_bytes, raw_metadata, progress)

`progress(stage_key, message)` is called as the pipeline advances so the UI can
show live status.
"""
import io
import time
import uuid
from typing import Any, Callable, Dict, Optional

import torch

from . import config as cfg
from .checkpoint_loader import LoadedModel, load_full_model
from .explain import build_case, generate_explanation
from .metadata_encoder import MetadataEncoder, load_default
from .metadata_schema import clean_values, readable, validate_against_vocab
from .prediction import run_inference
from .preprocessing import (
    PreprocessingError, load_image, preprocess_clinical_image,
    preprocess_dermoscopy_image, validate_input_shapes,
)
from .rag import get_kb
from .results import build_result, model_info

Progress = Callable[[str, str], None]

STAGES = [
    ("check", "Checking your images"),
    ("prepare", "Preparing the inputs"),
    ("remix", "Analysing the images with the multimodal transformer"),
    ("graph", "Reasoning over the clinical knowledge graph"),
    ("heuristic", "Applying the clinical heuristic"),
    ("retrieve", "Retrieving relevant medical knowledge (RAG)"),
    ("explain", "Writing the plain-language explanation"),
    ("finish", "Preparing your results"),
]


class AnalysisService:
    def __init__(self, model: LoadedModel, encoder: MetadataEncoder):
        self.model = model
        self.encoder = encoder
        validate_against_vocab(encoder)
        if encoder.dim != model.meta_dim:
            raise RuntimeError(
                f"Metadata encoder has {encoder.dim} dimensions but the model expects {model.meta_dim}.")
        self.info = model_info(model, encoder)
        self.kb = get_kb()

    @classmethod
    def load(cls, checkpoint_path=None, device: Optional[str] = None) -> "AnalysisService":
        model = load_full_model(checkpoint_path or cfg.CHECKPOINT_PATH, device)
        return cls(model, load_default())

    # ------------------------------------------------------------------
    def analyze(self, clinical_bytes: bytes, dermoscopy_bytes: bytes,
                raw_metadata: Dict[str, str], progress: Optional[Progress] = None,
                use_llm: bool = True) -> Dict[str, Any]:
        def p(key: str):
            if progress:
                progress(key, dict(STAGES)[key])

        t0 = time.perf_counter()
        p("check")
        clin = load_image(io.BytesIO(clinical_bytes), "Clinical")
        derm = load_image(io.BytesIO(dermoscopy_bytes), "Dermoscopy")

        p("prepare")
        meta_values = clean_values(raw_metadata)
        meta_rows = readable(meta_values)
        ct = preprocess_clinical_image(clin)
        dt = preprocess_dermoscopy_image(derm)
        mt = self.encoder.transform_tensor(meta_values)
        ok, msg = validate_input_shapes(ct, dt, mt, self.model.meta_dim)
        if not ok:
            raise PreprocessingError(msg)

        R = run_inference(self.model, ct, dt, mt, on_stage=lambda k: p(k))

        p("retrieve")
        case = build_case(self.model, R, meta_rows)
        t_rag = time.perf_counter()
        p("explain")
        explanation = generate_explanation(case, self.kb, use_llm=use_llm)
        t_exp = time.perf_counter()

        p("finish")
        result = build_result(
            self.model, self.encoder, R, clin, derm, ct, dt, mt, meta_values, meta_rows,
            case, explanation, uuid.uuid4().hex[:12],
            extra_timings={"explanation": t_exp - t_rag, "end_to_end": time.perf_counter() - t0})
        return result
