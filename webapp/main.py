"""
FastAPI application: JSON API + static single-page website.

Run:   python run.py            (or)   uvicorn webapp.main:app --host 0.0.0.0 --port 8000

Deployment notes
- Use ONE worker process: the model (~0.5 GB) lives in memory and inference is serialised.
- Images and results are held in memory only (never written to disk) and purged after
  JOB_TTL_MINUTES.
"""
import logging
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from src import config as cfg
from src.explain import llm_config
from src.metadata_schema import FIELDS
from src.pipeline import STAGES, AnalysisService
from src.preprocessing import PreprocessingError
from src.report import build_pdf

log = logging.getLogger("skin_dx")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

STATIC = Path(__file__).parent / "static"
MAX_UPLOAD_MB = float(os.getenv("MAX_UPLOAD_MB", "15"))
JOB_TTL_SECONDS = 60 * float(os.getenv("JOB_TTL_MINUTES", "60"))
MAX_QUEUE = int(os.getenv("MAX_QUEUE", "8"))
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp", "image/tiff", "image/jpg", ""}

STAGE_KEYS = [k for k, _ in STAGES]


class State:
    service: Optional[AnalysisService] = None
    load_error: Optional[str] = None
    jobs: Dict[str, Dict[str, Any]] = {}
    lock = threading.Lock()
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="inference")
    pdf_cache: Dict[str, bytes] = {}


S = State()


def _load_model() -> None:
    try:
        t = time.perf_counter()
        S.service = AnalysisService.load(device=os.getenv("DEVICE") or None)
        log.info("Model ready in %.1fs", time.perf_counter() - t)
    except Exception as exc:  # noqa: BLE001
        log.exception("Model failed to load")
        S.load_error = "The analysis model could not be loaded on the server."
        S.service = None


def _janitor() -> None:
    while True:
        time.sleep(60)
        cutoff = time.time() - JOB_TTL_SECONDS
        with S.lock:
            for jid in [j for j, v in S.jobs.items() if v["created"] < cutoff and v["status"] in ("done", "error")]:
                S.jobs.pop(jid, None)
                S.pdf_cache.pop(jid, None)


@asynccontextmanager
async def lifespan(app: FastAPI):
    threading.Thread(target=_load_model, daemon=True).start()
    threading.Thread(target=_janitor, daemon=True).start()
    yield


app = FastAPI(title="Skin Tumor Differential Diagnosis", docs_url=None, redoc_url=None, lifespan=lifespan)
app.add_middleware(GZipMiddleware, minimum_size=1024)


@app.middleware("http")
async def security_headers(request: Request, call_next):
    resp = await call_next(request)
    resp.headers.setdefault("X-Content-Type-Options", "nosniff")
    resp.headers.setdefault("X-Frame-Options", "DENY")
    resp.headers.setdefault("Referrer-Policy", "no-referrer")
    resp.headers.setdefault("Content-Security-Policy",
                            "default-src 'self'; img-src 'self' data: blob:; style-src 'self' 'unsafe-inline'; "
                            "script-src 'self' 'unsafe-eval'; connect-src 'self'; frame-ancestors 'none'")
    if request.url.path.startswith("/api/"):
        resp.headers["Cache-Control"] = "no-store"
    return resp


# ---------------------------------------------------------------- API ------
@app.get("/api/health")
def health():
    ready = S.service is not None
    return {"status": "ready" if ready else ("error" if S.load_error else "loading"),
            "ready": ready, "message": S.load_error,
            "explanation_writer": "ai" if llm_config()["available"] else "automatic"}


@app.get("/api/meta-schema")
def meta_schema():
    return {"fields": FIELDS}


@app.get("/api/model-info")
def model_info_route():
    if S.service is None:
        raise HTTPException(503, "The system is still starting up.")
    return S.service.info


def _run_job(jid: str, clinical: bytes, dermoscopy: bytes, meta: Dict[str, str]) -> None:
    job = S.jobs[jid]

    def progress(key: str, message: str) -> None:
        idx = STAGE_KEYS.index(key) if key in STAGE_KEYS else 0
        job.update(stage=key, message=message, step=idx + 1)

    try:
        job["status"] = "running"
        result = S.service.analyze(clinical, dermoscopy, meta, progress=progress)
        result["case"]["id"] = jid
        job.update(status="done", result=result, step=len(STAGE_KEYS), message="Done")
    except PreprocessingError as exc:
        job.update(status="error", error=str(exc))
    except Exception:  # noqa: BLE001
        log.exception("Analysis failed")
        job.update(status="error", error="Something went wrong while analysing the images. Please try different images.")


async def _read_limited(f: UploadFile, label: str) -> bytes:
    data = await f.read(int(MAX_UPLOAD_MB * 1024 * 1024) + 1)
    if not data:
        raise HTTPException(400, f"The {label} image is empty.")
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"The {label} image is larger than {MAX_UPLOAD_MB:.0f} MB.")
    if (f.content_type or "").lower() not in ALLOWED_TYPES:
        raise HTTPException(415, f"The {label} file must be an image (JPG or PNG).")
    return data


@app.post("/api/analyze")
async def analyze(clinical: UploadFile = File(...), dermoscopy: UploadFile = File(...),
                  sex: str = Form(""), location: str = Form(""), elevation: str = Form(""),
                  level_of_diagnostic_difficulty: str = Form(""), management: str = Form("")):
    if S.service is None:
        raise HTTPException(503, S.load_error or "The system is still starting up. Please try again in a moment.")
    active = sum(1 for j in S.jobs.values() if j["status"] in ("queued", "running"))
    if active >= MAX_QUEUE:
        raise HTTPException(429, "The system is busy. Please try again in a minute.")
    c = await _read_limited(clinical, "clinical")
    d = await _read_limited(dermoscopy, "dermoscopy")
    meta = {"sex": sex, "location": location, "elevation": elevation,
            "level_of_diagnostic_difficulty": level_of_diagnostic_difficulty, "management": management}
    jid = uuid.uuid4().hex[:12]
    S.jobs[jid] = {"status": "queued", "stage": "check", "step": 0, "message": "Waiting in queue",
                   "created": time.time(), "result": None, "error": None, "total": len(STAGE_KEYS)}
    S.pool.submit(_run_job, jid, c, d, meta)
    return {"job_id": jid}


@app.get("/api/jobs/{jid}")
def job_status(jid: str):
    job = S.jobs.get(jid)
    if job is None:
        raise HTTPException(404, "This analysis has expired. Please run it again.")
    body = {k: job[k] for k in ("status", "stage", "step", "message", "total", "error")}
    body["stages"] = [{"key": k, "label": l} for k, l in STAGES]
    if job["status"] == "done":
        body["result"] = job["result"]
    return body


def _done_job(jid: str) -> Dict[str, Any]:
    job = S.jobs.get(jid)
    if job is None or job["status"] != "done":
        raise HTTPException(404, "This analysis has expired or is not finished. Please run it again.")
    return job


@app.get("/api/report/{jid}.pdf")
def report(jid: str):
    job = _done_job(jid)
    if jid not in S.pdf_cache:
        S.pdf_cache[jid] = build_pdf(job["result"], S.service.info)
    return Response(S.pdf_cache[jid], media_type="application/pdf",
                    headers={"Content-Disposition": f'attachment; filename="skin-lesion-report-{jid}.pdf"'})


@app.get("/api/results/{jid}.json")
def results_json(jid: str):
    job = _done_job(jid)
    slim = {k: v for k, v in job["result"].items() if k != "case"}
    slim["case"] = {k: v for k, v in job["result"]["case"].items() if k != "images"}
    return JSONResponse(slim, headers={"Content-Disposition": f'attachment; filename="analysis-{jid}.json"'})


# ------------------------------------------------------------- website -----
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-cache"})
