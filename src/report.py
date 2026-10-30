"""
PDF report generation (ReportLab + Matplotlib).

Builds a self-contained, printable report from a finished analysis result:
input images, attention overlays, the five class scores, the seven feature
scores, the plain-language explanation, class information, the knowledge graph
with edge weights, the heuristic adjustment, technical details, and the
knowledge sources that were retrieved (RAG).
"""
import base64
import io
import re
from typing import Any, Dict, List
from xml.sax.saxutils import escape

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.patches import FancyArrowPatch  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.enums import TA_CENTER  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle  # noqa: E402
from reportlab.lib.units import mm  # noqa: E402
from reportlab.lib.utils import ImageReader  # noqa: E402
from reportlab.pdfbase import pdfmetrics  # noqa: E402
from reportlab.pdfbase.ttfonts import TTFont  # noqa: E402
from reportlab.platypus import (  # noqa: E402
    BaseDocTemplate, Frame, Image, KeepTogether, PageBreak, PageTemplate, Paragraph,
    Spacer, Table, TableStyle,
)

from .rag import get_kb

PROJECT_TITLE = ("A Heuristic Informed Clinical Reasoning for Skin Tumor Differential Diagnosis "
                 "using Multimodal Transformers and Knowledge Graphs")
NAVY, TEAL, SLATE = "#0f2a4a", "#0e7490", "#475569"
LIGHT, LINE = "#f1f5f9", "#cbd5e1"
DISCLAIMER = ("Research prototype. This report contains experimental machine-learning estimates. It is NOT a "
              "medical diagnosis and must not replace examination by a qualified clinician.")

# ------------------------------------------------------------------ fonts ---
_FONTS_READY = False


def _register_fonts():
    global _FONTS_READY
    if _FONTS_READY:
        return
    reg = font_manager.findfont("DejaVu Sans")
    bold = font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans", weight="bold"))
    ital = font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans", style="italic"))
    pdfmetrics.registerFont(TTFont("DV", reg))
    pdfmetrics.registerFont(TTFont("DV-B", bold))
    pdfmetrics.registerFont(TTFont("DV-I", ital))
    pdfmetrics.registerFontFamily("DV", normal="DV", bold="DV-B", italic="DV-I", boldItalic="DV-B")
    _FONTS_READY = True


def _styles() -> Dict[str, ParagraphStyle]:
    _register_fonts()
    base = dict(fontName="DV", fontSize=9, leading=13.2, textColor=colors.HexColor("#1e293b"))
    return {
        "body": ParagraphStyle("body", **base),
        "small": ParagraphStyle("small", **{**base, "fontSize": 7.6, "leading": 10.4,
                                            "textColor": colors.HexColor(SLATE)}),
        "cap": ParagraphStyle("cap", **{**base, "fontSize": 7.6, "leading": 10, "fontName": "DV-I",
                                        "textColor": colors.HexColor(SLATE), "alignment": TA_CENTER}),
        "h1": ParagraphStyle("h1", **{**base, "fontName": "DV-B", "fontSize": 12.5, "leading": 16,
                                      "textColor": colors.HexColor(NAVY), "spaceBefore": 10, "spaceAfter": 5}),
        "h2": ParagraphStyle("h2", **{**base, "fontName": "DV-B", "fontSize": 10, "leading": 13,
                                      "textColor": colors.HexColor(TEAL), "spaceBefore": 7, "spaceAfter": 3}),
        "title": ParagraphStyle("title", **{**base, "fontName": "DV-B", "fontSize": 15.5, "leading": 20,
                                            "textColor": colors.HexColor(NAVY)}),
        "sub": ParagraphStyle("sub", **{**base, "fontSize": 9.4, "textColor": colors.HexColor(SLATE)}),
        "bullet": ParagraphStyle("bullet", **{**base, "leftIndent": 11, "bulletIndent": 1, "spaceAfter": 2}),
        "cell": ParagraphStyle("cell", **{**base, "fontSize": 8.2, "leading": 11}),
        "cellh": ParagraphStyle("cellh", **{**base, "fontSize": 8.2, "leading": 11, "fontName": "DV-B",
                                            "textColor": colors.white}),
        "cellb": ParagraphStyle("cellb", **{**base, "fontSize": 8.2, "leading": 11, "fontName": "DV-B"}),
        "hero": ParagraphStyle("hero", **{**base, "fontName": "DV-B", "fontSize": 17, "leading": 21,
                                          "textColor": colors.HexColor(NAVY)}),
        "warn": ParagraphStyle("warn", **{**base, "fontSize": 8.4, "leading": 11.6,
                                          "textColor": colors.HexColor("#78350f")}),
    }


# ------------------------------------------------------------ text helpers --
def _md(text: str) -> str:
    """Escape, then convert **bold** to ReportLab markup. Citation markers [S#] are stripped."""
    t = escape(text)
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"\s*\[S\d+\]", "", t)
    return t


def _img_from_dataurl(url: str, width_mm: float, max_height_mm: float = 200) -> Image:
    raw = base64.b64decode(url.split(",", 1)[1])
    ir = ImageReader(io.BytesIO(raw))
    w, h = ir.getSize()
    wpt = width_mm * mm
    hpt = wpt * h / w
    if hpt > max_height_mm * mm:
        hpt = max_height_mm * mm
        wpt = hpt * w / h
    return Image(io.BytesIO(raw), width=wpt, height=hpt)


def _fig_image(fig, width_mm: float) -> Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    ir = ImageReader(buf)
    w, h = ir.getSize()
    buf.seek(0)
    return Image(buf, width=width_mm * mm, height=width_mm * mm * h / w)


# ------------------------------------------------------------------ charts --
def _chart_classes(res: Dict[str, Any]):
    cl = sorted(res["prediction"]["classes"], key=lambda c: -c["final_prob"])
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    y = np.arange(len(cl))
    ax.barh(y + 0.19, [c["neural_prob"] * 100 for c in cl], height=0.36, color="#94a3b8",
            label="Image network only")
    ax.barh(y - 0.19, [c["final_prob"] * 100 for c in cl], height=0.36,
            color=[c["color"] for c in cl], label="Final (after rule-based step)")
    for yi, c in zip(y, cl):
        ax.text(c["final_prob"] * 100 + 1, yi - 0.19, f"{c['final_prob'] * 100:.0f}%", va="center",
                fontsize=8, color="#0f172a", fontweight="bold")
        ax.text(c["neural_prob"] * 100 + 1, yi + 0.19, f"{c['neural_prob'] * 100:.0f}%", va="center",
                fontsize=6.5, color="#64748b")
    ax.set_yticks(y)
    ax.set_yticklabels([c["name"] for c in cl], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 105)
    ax.set_xlabel("System score (%)", fontsize=8)
    ax.tick_params(axis="x", labelsize=7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    fig.tight_layout()
    return fig


def _chart_attributes(res: Dict[str, Any]):
    at = res["attributes"]
    fig, ax = plt.subplots(figsize=(7.2, 2.5))
    x = np.arange(len(at))
    w = 0.27
    ax.bar(x - w, [a["aux_prob"] * 100 for a in at], w, color="#94a3b8", label="Auxiliary head")
    ax.bar(x, [a["graph_prob"] * 100 for a in at], w, color="#38bdf8", label="Graph head")
    ax.bar(x + w, [a["fused_prob"] * 100 for a in at], w,
           color=[TEAL if a["detected"] else "#cbd5e1" for a in at], label="Combined (used)")
    ax.axhline(50, color="#dc2626", lw=0.8, ls="--")
    ax.text(len(at) - 0.45, 52, "50% threshold", fontsize=6.5, color="#dc2626", ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels([a["name"].replace(" ", "\n", 1) for a in at], fontsize=7)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Score (%)", fontsize=8)
    ax.tick_params(axis="y", labelsize=7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=6.8, frameon=False, ncol=3, loc="upper left")
    fig.tight_layout()
    return fig


def _chart_graph(res: Dict[str, Any], info: Dict[str, Any]):
    g = info["graph"]
    nodes = g["nodes"]
    n_attr = g["num_attributes"]
    A = np.asarray(g["A_sparse"])
    att = np.asarray(res["graph"]["node_attention"])
    fused = {a["code"]: a["fused_prob"] for a in res["attributes"]}
    finalp = {c["code"]: c["final_prob"] for c in res["prediction"]["classes"]}

    pos = {}
    ang = np.linspace(0, 2 * np.pi, n_attr, endpoint=False) + np.pi / 2
    for i in range(n_attr):
        pos[nodes[i]] = np.array([np.cos(ang[i]), np.sin(ang[i])])
    inner = nodes[n_attr:]
    ang2 = np.linspace(0, 2 * np.pi, len(inner), endpoint=False) + np.pi / 2
    for i, n in enumerate(inner):
        pos[n] = 0.42 * np.array([np.cos(ang2[i]), np.sin(ang2[i])])

    fig, ax = plt.subplots(figsize=(7.4, 6.6))
    ax.set_aspect("equal")
    ax.axis("off")
    rad = 0.16
    for i, s in enumerate(nodes):
        for j, t in enumerate(nodes):
            w = float(A[i, j])
            if w <= 0 or i == j:
                continue
            active = s in fused and fused[s] >= 0.5
            p0, p1 = pos[s], pos[t]
            ax.add_patch(FancyArrowPatch(
                p0, p1, connectionstyle=f"arc3,rad={rad}", arrowstyle="-|>", mutation_scale=7,
                lw=0.6 + 3.0 * w, color=("#0369a1" if active else "#94a3b8"),
                alpha=0.85 if active else 0.45, shrinkA=13, shrinkB=13, zorder=1))
            mid = (p0 + p1) / 2
            d = p1 - p0
            lp = mid + 0.5 * rad * np.array([d[1], -d[0]])
            ax.text(lp[0], lp[1], f"{w:.2f}", fontsize=6.3, ha="center", va="center", zorder=3,
                    color="#0f172a", bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85))
    amax = max(att.max(), 1e-9)
    for k, n in enumerate(nodes):
        is_attr = k < n_attr
        val = fused.get(n) if is_attr else finalp.get(n)
        col = plt.cm.Blues(0.25 + 0.6 * (val or 0)) if is_attr else plt.cm.Oranges(0.25 + 0.6 * (val or 0))
        size = 420 + 900 * (att[k] / amax)
        ax.scatter(*pos[n], s=size, color=col, edgecolor="#1e3a5f", lw=1.4, zorder=4)
        ax.text(pos[n][0], pos[n][1], n, ha="center", va="center", fontsize=8, fontweight="bold",
                color="#0f172a", zorder=5)
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.25, 1.25)
    ax.set_title("Knowledge graph with edge weights (blue arrows start at features detected in this lesion)",
                 fontsize=8.5, color=NAVY)
    fig.tight_layout()
    return fig


def _chart_heuristic(res: Dict[str, Any]):
    cl = res["prediction"]["classes"]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.7), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    x = np.arange(len(cl))
    ax.bar(x - 0.2, [c["neural_prob"] * 100 for c in cl], 0.38, color="#94a3b8", label="Before")
    ax.bar(x + 0.2, [c["final_prob"] * 100 for c in cl], 0.38, color=TEAL, label="After")
    ax.set_xticks(x)
    ax.set_xticklabels([c["code"] for c in cl], fontsize=8)
    ax.set_ylabel("Score (%)", fontsize=8)
    ax.tick_params(axis="y", labelsize=7)
    ax.set_title("Score before and after the rule-based step", fontsize=8, color=NAVY)
    ax.legend(fontsize=7, frameon=False)
    ax2 = axes[1]
    comps = [("attribute_evidence", "Feature evidence", "#0e7490"),
             ("graph_support", "Graph support", "#7c3aed"),
             ("contradiction_penalty", "Contradiction", "#dc2626"),
             ("interaction_bonus", "Interaction", "#f59e0b")]
    for k, (key, lab, col) in enumerate(comps):
        vals = [c[key] for c in cl]
        sign = -1 if key == "contradiction_penalty" else 1
        ax2.bar(x + (k - 1.5) * 0.19, [sign * v for v in vals], 0.18, color=col, label=lab)
    ax2.axhline(0, color="#64748b", lw=0.6)
    ax2.set_xticks(x)
    ax2.set_xticklabels([c["code"] for c in cl], fontsize=8)
    ax2.tick_params(axis="y", labelsize=7)
    ax2.set_title("Heuristic component scores (before weighting)", fontsize=8, color=NAVY)
    ax2.legend(fontsize=6.2, frameon=False, ncol=2)
    for a in axes:
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    fig.tight_layout()
    return fig


# -------------------------------------------------------------- page frame --
class _Doc(BaseDocTemplate):
    def __init__(self, buf, report_id, created):
        super().__init__(buf, pagesize=A4, leftMargin=17 * mm, rightMargin=17 * mm,
                         topMargin=21 * mm, bottomMargin=19 * mm,
                         title="Skin lesion analysis report", author="Skin Tumor Differential Diagnosis")
        self.report_id, self.created = report_id, created
        frame = Frame(self.leftMargin, self.bottomMargin, self.width, self.height, id="f",
                      leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
        self.addPageTemplates([PageTemplate(id="p", frames=[frame], onPage=self._decor)])

    def _decor(self, c, doc):
        _register_fonts()
        w, h = A4
        c.saveState()
        c.setFillColor(colors.HexColor(NAVY))
        c.rect(0, h - 12 * mm, w, 12 * mm, stroke=0, fill=1)
        c.setFillColor(colors.white)
        c.setFont("DV-B", 8.5)
        c.drawString(17 * mm, h - 7.6 * mm, "Skin Tumor Differential Diagnosis  |  Analysis report")
        c.setFont("DV", 7.5)
        c.drawRightString(w - 17 * mm, h - 7.6 * mm, f"Report {self.report_id}")
        c.setStrokeColor(colors.HexColor(LINE))
        c.line(17 * mm, 14 * mm, w - 17 * mm, 14 * mm)
        c.setFillColor(colors.HexColor(SLATE))
        c.setFont("DV", 6.8)
        c.drawString(17 * mm, 10 * mm, "Research prototype. Not a medical diagnosis. Consult a qualified clinician.")
        c.drawRightString(w - 17 * mm, 10 * mm, f"Generated {self.created}   |   Page {doc.page}")
        c.restoreState()


def _table(data, col_widths, header=True, zebra=True, extra=None):
    t = Table(data, colWidths=col_widths, repeatRows=1 if header else 0)
    st = [
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LINEBELOW", (0, 0), (-1, -1), 0.4, colors.HexColor(LINE)),
        ("TOPPADDING", (0, 0), (-1, -1), 3.2), ("BOTTOMPADDING", (0, 0), (-1, -1), 3.2),
        ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]
    if header:
        st += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(NAVY)),
               ("TEXTCOLOR", (0, 0), (-1, 0), colors.white)]
    if zebra:
        for r in range(1 if header else 0, len(data)):
            if r % 2 == 0:
                st.append(("BACKGROUND", (0, r), (-1, r), colors.HexColor("#f8fafc")))
    t.setStyle(TableStyle(st + (extra or [])))
    return t


# --------------------------------------------------------------- the report -
def build_pdf(res: Dict[str, Any], info: Dict[str, Any]) -> bytes:
    S = _styles()
    kb = get_kb()
    pred = res["prediction"]
    case = res["case"]
    expl = res["explanation"]
    cs = res["case_summary"]
    created = case["created_utc"].replace("T", " ").replace("+00:00", " UTC")
    W = 176  # usable width in mm

    buf = io.BytesIO()
    doc = _Doc(buf, case["id"].upper(), created)
    E: List[Any] = []

    def P(t, s="body"):
        return Paragraph(_md(t), S[s])

    # ---- title ---------------------------------------------------------
    E += [Paragraph(escape(PROJECT_TITLE), S["title"]), Spacer(1, 3),
          Paragraph("AI-assisted analysis of a skin lesion: images, metadata, knowledge graph, and plain-language explanation. "
                    f"Report ID <b>{case['id'].upper()}</b> &nbsp;|&nbsp; {created}", S["sub"]), Spacer(1, 6)]
    warn = Table([[Paragraph("<b>Important.</b> " + escape(DISCLAIMER), S["warn"])]], colWidths=[W * mm])
    warn.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fef3c7")),
                              ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#f59e0b")),
                              ("LEFTPADDING", (0, 0), (-1, -1), 7), ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                              ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
    E += [warn, Spacer(1, 8)]

    # ---- 1 summary ------------------------------------------------------
    top = next(c for c in pred["classes"] if c["code"] == pred["top_class"])
    E.append(Paragraph("1. Summary of the result", S["h1"]))
    hero = Table([[Paragraph(f"{escape(pred['top_name'])}", S["hero"]),
                   Paragraph(f"<b>{pred['top_prob'] * 100:.0f}%</b><br/><font size=7>system score</font>", S["hero"])],
                  [Paragraph(escape(top["group"]) + ". " + escape(
                      next(v["short"] for k, v in __import__("src.display", fromlist=["CLASS_INFO"]).CLASS_INFO.items()
                           if k == pred["top_class"])), S["cell"]), ""]],
                 colWidths=[(W - 38) * mm, 38 * mm])
    hero.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(LIGHT)),
                              ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor(LINE)),
                              ("LINEBEFORE", (0, 0), (0, -1), 3.2, colors.HexColor(top["color"])),
                              ("SPAN", (1, 0), (1, 1)), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                              ("ALIGN", (1, 0), (1, 1), "CENTER"),
                              ("LEFTPADDING", (0, 0), (-1, -1), 9), ("TOPPADDING", (0, 0), (-1, -1), 6),
                              ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
    E += [hero, Spacer(1, 4)]
    line = (f"This is the group the system scored highest. The runner-up is <b>{escape(pred['runner_up_name'])}</b> at "
            f"{pred['runner_up_prob'] * 100:.0f}%.")
    if pred["uncertain"]:
        line += " The top scores are close or not high, so the system is <b>uncertain</b>."
    if pred["heuristic_changed_top"]:
        line += " The rule-based step changed the top group from the image network's first choice."
    E += [Paragraph(line, S["body"]), Spacer(1, 4)]
    E.append(_fig_image(_chart_classes(res), 168))

    # ---- 2 images -------------------------------------------------------
    E.append(Paragraph("2. Images analysed", S["h1"]))
    ci = _img_from_dataurl(case["images"]["clinical"], 82, 70)
    di = _img_from_dataurl(case["images"]["dermoscopy"], 82, 70)
    t = Table([[ci, di], [Paragraph("Clinical photograph", S["cap"]), Paragraph("Dermoscopy image", S["cap"])]],
              colWidths=[88 * mm, 88 * mm])
    t.setStyle(TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER"), ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
    E.append(t)
    ov = res.get("overlays", {})
    if "lsm" in ov or "dermoscopy_region" in ov:
        E.append(Spacer(1, 4))
        cells, caps = [], []
        if "lsm" in ov:
            cells.append(_img_from_dataurl(ov["lsm"]["image"], 82, 70))
            caps.append(Paragraph("AI attention: clinical photo", S["cap"]))
        if "dermoscopy_region" in ov:
            cells.append(_img_from_dataurl(ov["dermoscopy_region"]["image"], 82, 70))
            caps.append(Paragraph("AI attention: dermoscopy regions", S["cap"]))
        while len(cells) < 2:
            cells.append(""); caps.append("")
        t2 = Table([cells, caps], colWidths=[88 * mm, 88 * mm])
        t2.setStyle(TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER"), ("VALIGN", (0, 0), (-1, -1), "TOP")]))
        E.append(t2)

    # ---- 3 information provided ----------------------------------------
    E.append(Paragraph("3. Information provided", S["h1"]))
    rows = [[Paragraph("Question", S["cellh"]), Paragraph("Answer", S["cellh"])]]
    for r in case["metadata"]:
        rows.append([Paragraph(escape(r["question"]), S["cell"]),
                     Paragraph(escape(r["answer"]) if r["provided"] else "<i>Not provided</i>", S["cell"])])
    tb = _table(rows, [90 * mm, 86 * mm])
    tb.setStyle(TableStyle([("TEXTCOLOR", (0, 0), (-1, 0), colors.white)]))
    E += [tb, Spacer(1, 2),
          Paragraph("Details that are not provided are left blank and contribute nothing to the analysis; "
                    "the seven-point checklist score is never asked from users.", S["small"])]

    # ---- 4 about the leading class --------------------------------------
    E.append(Paragraph("4. About the leading group and the runner-up", S["h1"]))
    for code, label in ((pred["top_class"], pred["top_name"]), (pred["runner_up"], pred["runner_up_name"])):
        E.append(Paragraph(escape(label), S["h2"]))
        for suffix in ("overview", "features"):
            ch = kb.chunk(f"class_{code.lower()}_{suffix}")
            if ch:
                E.append(Paragraph(escape(ch["text"]), S["body"]))
                E.append(Spacer(1, 2))
        if code == pred["runner_up"]:
            break

    # ---- 5 features -----------------------------------------------------
    E.append(PageBreak())
    E.append(Paragraph("5. What the system noticed in the lesion", S["h1"]))
    E.append(Paragraph("The seven-point checklist features are estimated by the AI from the images. A feature is "
                       "counted as detected when its combined score is 50% or more.", S["body"]))
    E.append(Spacer(1, 3))
    E.append(_fig_image(_chart_attributes(res), 168))
    rows = [[Paragraph(h, S["cellh"]) for h in ("Feature", "What it means", "Score", "Detected", "Checklist")]]
    for a in res["attributes"]:
        rows.append([Paragraph(f"<b>{escape(a['name'])}</b>", S["cell"]),
                     Paragraph(escape(a["plain"]), S["cell"]),
                     Paragraph(f"{a['fused_prob'] * 100:.0f}%", S["cell"]),
                     Paragraph("<b>Yes</b>" if a["detected"] else "No", S["cell"]),
                     Paragraph(f"{a['points']} pt ({a['kind']})", S["cell"])])
    tb = _table(rows, [30 * mm, 78 * mm, 14 * mm, 21 * mm, 33 * mm])
    for r in range(len(rows)):
        pass
    tb.setStyle(TableStyle([("TEXTCOLOR", (0, 0), (-1, 0), colors.white)]))
    E += [Spacer(1, 4), tb]

    # ---- 6 explanation ---------------------------------------------------
    E.append(PageBreak())
    E.append(Paragraph("6. Plain-language explanation", S["h1"]))
    E.append(Spacer(1, 3))
    for sec in expl["sections"]:
        block = [Paragraph(escape(sec["title"]), S["h2"])]
        for b in sec["blocks"]:
            if b["type"] == "p":
                block.append(Paragraph(_md(b["text"]), S["body"]))
                block.append(Spacer(1, 2))
            else:
                for it in b["items"]:
                    block.append(Paragraph(_md(it), S["bullet"], bulletText="•"))
        E.append(KeepTogether(block[:3]))
        E += block[3:]

    # ---- 7 how it was produced ------------------------------------------
    E.append(PageBreak())
    E.append(Paragraph("7. How the result was produced", S["h1"]))
    E.append(Paragraph(
        "<b>Step 1 - Image and metadata encoder (RemixFormer++).</b> A multimodal transformer studies the clinical photo "
        "(global view plus four close-up regions chosen by a lesion-selection module), the dermoscopy image (16 regions) "
        "and the basic patient answers, and fuses them into five 768-number summaries (tokens).<br/>"
        "<b>Step 2 - Knowledge graph.</b> The estimated features and lesion-group scores become the nodes of a 12-node graph "
        "(7 features, 5 groups). Edge weights are association strengths learned from the training data.<br/>"
        "<b>Step 3 - Rule-based (heuristic) adjustment.</b> A small fixed rule combines feature evidence, graph support, "
        "contradictions and feature pairs, and adjusts the scores (strength lambda = "
        f"{res['heuristic']['lambda']:.2f}).", S["body"]))
    E.append(Spacer(1, 4))
    E.append(_fig_image(_chart_graph(res, info), 150))
    E.append(Spacer(1, 6))
    E.append(_fig_image(_chart_heuristic(res), 168))
    rows = [[Paragraph(h, S["cellh"]) for h in ("Group", "Before", "After", "Feature evidence", "Graph support",
                                                "Contradiction", "Interaction")]]
    for c in sorted(res["prediction"]["classes"], key=lambda c: -c["final_prob"]):
        rows.append([Paragraph(escape(c["code"]), S["cell"])] +
                    [Paragraph(f"{v:.3f}", S["cell"]) for v in (
                        c["neural_prob"], c["final_prob"], c["attribute_evidence"], c["graph_support"],
                        c["contradiction_penalty"], c["interaction_bonus"])])
    tb = _table(rows, [22 * mm, 22 * mm, 22 * mm, 27 * mm, 27 * mm, 28 * mm, 28 * mm])
    tb.setStyle(TableStyle([("TEXTCOLOR", (0, 0), (-1, 0), colors.white)]))
    E += [Spacer(1, 4), tb]
    w = res["heuristic"]["weights"]
    E.append(Paragraph(
        f"Final score = log(neural score) + lambda x (feature evidence + {w['graph_weight']} x graph support - "
        f"{w['contradiction_weight']} x contradiction + {w['interaction_weight']} x interaction), then re-normalised "
        f"(lambda = {res['heuristic']['lambda']:.2f}).", S["small"]))

    # ---- 8 technical ------------------------------------------------------
    E.append(Paragraph("8. Technical details", S["h1"]))
    it = res["internals"]
    rows = [[Paragraph(h, S["cellh"]) for h in ("Input", "Tensor shape", "Range", "Preparation")]]
    for r in it["preprocessing"]:
        rows.append([Paragraph(escape(r["input"]), S["cell"]), Paragraph(str(tuple(r["shape"])), S["cell"]),
                     Paragraph(f"{r['min']:.2f} to {r['max']:.2f}", S["cell"]), Paragraph(escape(r["resize"]), S["cell"])])
    tb = _table(rows, [36 * mm, 42 * mm, 30 * mm, 68 * mm])
    tb.setStyle(TableStyle([("TEXTCOLOR", (0, 0), (-1, 0), colors.white)]))
    E += [tb, Spacer(1, 5)]
    wt = res["weights"]
    rows = [[Paragraph(h, S["cellh"]) for h in ("Learned token weight", "Weight", "Learned feature weight", "Weight")]]
    for i in range(max(len(wt["tokens"]), len(wt["attributes"]))):
        a = wt["tokens"][i] if i < len(wt["tokens"]) else None
        b = wt["attributes"][i] if i < len(wt["attributes"]) else None
        rows.append([Paragraph(escape(a["label"]) if a else "", S["cell"]), Paragraph(f"{a['weight']:.3f}" if a else "", S["cell"]),
                     Paragraph(escape(next(x["name"] for x in res["attributes"] if x["code"] == b["code"])) if b else "", S["cell"]),
                     Paragraph(f"{b['weight']:.3f}" if b else "", S["cell"])])
    tb = _table(rows, [56 * mm, 22 * mm, 56 * mm, 22 * mm])
    tb.setStyle(TableStyle([("TEXTCOLOR", (0, 0), (-1, 0), colors.white)]))
    E += [tb, Spacer(1, 4)]
    tm = it["timings_seconds"]
    E.append(Paragraph("Processing time: " + ", ".join(f"{k.replace('_', ' ')} {v:.2f}s" for k, v in tm.items()
                                                       if k in ("remixformer", "graph_encoder", "hybrid", "heuristic", "total")) + ".",
                       S["small"]))

    E += [Spacer(1, 8), Table([[Paragraph("<b>Research disclaimer.</b> " + escape(DISCLAIMER) +
                                          " The probabilities are uncalibrated and the system was trained on a small research "
                                          "dataset (Derm7pt). If you are worried about a skin lesion, see a doctor.", S["warn"])]],
                              colWidths=[W * mm], style=TableStyle([
                                  ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fef3c7")),
                                  ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#f59e0b")),
                                  ("LEFTPADDING", (0, 0), (-1, -1), 7), ("TOPPADDING", (0, 0), (-1, -1), 5),
                                  ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))]

    doc.build(E)
    return buf.getvalue()
