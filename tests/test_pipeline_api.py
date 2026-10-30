import time

import pytest

from src.report import build_pdf
from tests.conftest import jpeg_bytes, synth_image


def test_pipeline_result_structure(analysis, service):
    R = analysis
    assert R["prediction"]["top_class"] in ("BCC", "Nev", "Mel", "SK", "Misc")
    assert abs(sum(c["final_prob"] for c in R["prediction"]["classes"]) - 1) < 1e-3
    assert len(R["attributes"]) == 7 and len(R["explanation"]["sections"]) == 8
    assert R["case"]["images"]["clinical"].startswith("data:image/jpeg")
    assert "lsm" in R["overlays"] and len(R["overlays"]["lsm"]["crops"]) == 4
    meta = {m["key"]: m for m in R["case"]["metadata"]}
    assert meta["sex"]["provided"] and not meta["management"]["provided"]
    enc = {e["field"]: e for e in R["internals"]["metadata_encoding"]}
    assert enc["seven_point_score"]["active"] is None       # never asked -> zero block


def test_pdf_report_is_generated(analysis, service):
    pdf = build_pdf(analysis, service.info)
    assert pdf[:5] == b"%PDF-" and len(pdf) > 100_000


def test_api_flow(service):
    from fastapi.testclient import TestClient
    from webapp.main import S, app
    with TestClient(app) as c:
        for _ in range(120):
            if c.get("/api/health").json()["ready"]:
                break
            time.sleep(0.5)
        assert c.get("/api/meta-schema").json()["fields"]
        assert c.get("/api/model-info").json()["graph"]["nodes"]
        files = {"clinical": ("a.jpg", jpeg_bytes(synth_image(800)), "image/jpeg"),
                 "dermoscopy": ("b.jpg", jpeg_bytes(synth_image(800)), "image/jpeg")}
        jid = c.post("/api/analyze", files=files, data={"sex": "male", "location": "chest", "elevation": "flat"}).json()["job_id"]
        for _ in range(120):
            j = c.get(f"/api/jobs/{jid}").json()
            if j["status"] in ("done", "error"):
                break
            time.sleep(0.5)
        assert j["status"] == "done", j.get("error")
        assert c.get(f"/api/report/{jid}.pdf").headers["content-type"] == "application/pdf"
        js = c.get(f"/api/results/{jid}.json").json()
        assert "images" not in js["case"]                     # raw images are not exported
        # bad inputs
        bad = {"clinical": ("a.jpg", b"not an image", "image/jpeg"), "dermoscopy": files["dermoscopy"]}
        j2 = c.post("/api/analyze", files=bad).json()["job_id"]
        time.sleep(1.5)
        assert "could not be read" in c.get(f"/api/jobs/{j2}").json()["error"]
        pdf = {"clinical": ("a.pdf", b"%PDF-1.4", "application/pdf"), "dermoscopy": files["dermoscopy"]}
        assert c.post("/api/analyze", files=pdf).status_code == 415
        assert c.get("/api/jobs/doesnotexist").status_code == 404
        assert c.get("/").status_code == 200
