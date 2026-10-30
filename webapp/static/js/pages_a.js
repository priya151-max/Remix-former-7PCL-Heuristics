/* Pages: overview, new analysis, diagnosis result, AI explanation */
(() => {
  const { $, $$, esc, pct, f, rich, state, icon, api, getJSON, plot, register, go, needResult, legal } = App;

  const CLASS_FALLBACK = [
    { code: "BCC", name: "Basal cell carcinoma", short: "A common, usually slow-growing type of skin cancer.", group: "Malignant (cancer)", color: "#c2410c" },
    { code: "Nev", name: "Melanocytic nevus (mole)", short: "A common mole: a harmless cluster of pigment cells.", group: "Usually benign", color: "#0f766e" },
    { code: "Mel", name: "Melanoma", short: "A serious skin cancer of the pigment cells; early detection matters.", group: "Malignant (most serious)", color: "#b91c1c" },
    { code: "SK", name: "Seborrheic keratosis", short: "A very common, harmless, wart-like or 'stuck-on' growth.", group: "Benign", color: "#4f46e5" },
    { code: "Misc", name: "Other lesion types", short: "Less common lesions such as lentigo, vascular lesions or dermatofibroma.", group: "Mixed group", color: "#64748b" },
  ];
  const tone = (g) => (/most serious/.test(g) ? "danger" : /Malignant/.test(g) ? "warn" : /benign|Benign/.test(g) ? "ok" : "neutral");

  /* ================= OVERVIEW ================= */
  register("overview", (el) => {
    const classes = state.info ? state.info.classes : CLASS_FALLBACK;
    el.innerHTML = `
    <div class="hero-card">
      <div class="eyebrow">What this website does</div>
      <h2>A second look at a skin lesion, explained in plain words</h2>
      <p style="max-width:860px;font-size:1.03rem">You give the system two pictures of the same skin lesion and a few simple answers. It estimates which of five lesion groups the lesion most resembles, shows <b>why</b> it thinks so, and writes a clear explanation you can take to your doctor. It is a research tool: it never gives a diagnosis.</p>
      <div class="steps">
        <div class="step"><div class="num">1</div><b>Upload two images</b><span>A normal photo and a dermoscopy (magnified) image of the same lesion.</span></div>
        <div class="step"><div class="num">2</div><b>Answer 3 quick questions</b><span>Sex, body site and how the lesion feels to the touch.</span></div>
        <div class="step"><div class="num">3</div><b>The AI analyses</b><span>A multimodal transformer, a knowledge graph and clinical rules work together.</span></div>
        <div class="step"><div class="num">4</div><b>Read and download</b><span>Get an easy explanation and a PDF report with all the details.</span></div>
      </div>
      <div class="row" style="margin-top:16px"><a class="btn primary lg" href="#/analyze">${icon("upload")} Start an analysis</a><a class="btn secondary lg" href="#/method">How it works</a></div>
    </div>

    <div class="card">
      <div class="card-title"><h3>How the system reaches a result</h3></div>
      <div class="flow">
        <div class="fb"><b>1. Images and answers</b>Clinical photo, dermoscopy image, basic patient information</div><div class="arrow">\u2192</div>
        <div class="fb"><b>2. Multimodal transformer</b>RemixFormer++ studies all three and builds a summary of each</div><div class="arrow">\u2192</div>
        <div class="fb"><b>3. Knowledge graph</b>Links 7 dermoscopic features to 5 lesion groups, with weighted edges</div><div class="arrow">\u2192</div>
        <div class="fb"><b>4. Clinical heuristic</b>A gentle rule-based check refines the scores</div><div class="arrow">\u2192</div>
        <div class="fb"><b>5. Retrieval + explanation</b>Relevant medical knowledge is retrieved (RAG) and turned into a simple explanation</div>
      </div>
    </div>

    <div class="card">
      <div class="card-title"><h3>The five lesion groups the system distinguishes</h3></div>
      <div class="grid g3" style="grid-template-columns:repeat(auto-fit,minmax(230px,1fr))">
        ${classes.map((c) => `<div class="class-card" style="--c:${c.color}"><b>${esc(c.name)}</b><div style="margin:4px 0 6px"><span class="pill ${tone(c.group)}">${esc(c.group)}</span></div><span class="muted" style="font-size:.88rem">${esc(c.short)}</span></div>`).join("")}
      </div>
    </div>

    <div class="grid g2">
      <div class="callout ok" style="margin:0">${icon("check")}<div><b>What it is good for</b><br>Understanding what a computer sees in your images, learning the medical terms, and preparing questions for a doctor.</div></div>
      <div class="callout warn" style="margin:0">${icon("alert")}<div><b>What it is not</b><br>A diagnosis, a screening test or a reason to delay seeing a doctor. If a spot is new, changing, bleeding or worrying you, see a doctor.</div></div>
    </div>${legal}`;
  });

  /* ================= NEW ANALYSIS ================= */
  const MAXMB = 15;
  function zone(k, title, hint) {
    const file = state.files[k];
    if (file) {
      const url = state.files[k + "_url"] || (state.files[k + "_url"] = URL.createObjectURL(file));
      return `<label class="dz" data-k="${k}"><button class="rm" type="button" data-rm="${k}">Remove</button><img src="${url}" alt="${title} preview"><small>${esc(file.name)} \u00b7 ${file.size < 1048576 ? Math.max(1, Math.round(file.size / 1024)) + " KB" : (file.size / 1048576).toFixed(1) + " MB"}</small></label>`;
    }
    return `<label class="dz" data-k="${k}" tabindex="0"><input type="file" accept="image/jpeg,image/png,image/webp" data-in="${k}">${icon("upload")}<b>${title}</b><small>${hint}</small><small class="muted">Click to choose or drag a file here \u00b7 JPG / PNG \u00b7 up to ${MAXMB} MB</small></label>`;
  }
  function question(fld) {
    const sel = state.form[fld.key];
    const isOpt = fld.options.some((o) => o.hint);
    const opts = fld.options.map((o) =>
      `<button type="button" class="opt ${sel === o.value ? "sel" : ""}" data-q="${fld.key}" data-v="${esc(o.value)}">${esc(o.label)}${o.hint ? `<small>${esc(o.hint)}</small>` : ""}</button>`).join("");
    const na = `<button type="button" class="opt na ${sel === "" ? "sel" : ""}" data-q="${fld.key}" data-v="">I\u2019m not sure</button>`;
    return `<div class="q"><label>${esc(fld.label)}</label><div class="help">${esc(fld.help)}</div><div class="chips">${opts}${na}</div></div>`;
  }
  function clinicianFields() {
    const flds = state.schema.fields.filter((x) => x.group === "clinician");
    return flds.map((fld) => `<div class="q"><label>${esc(fld.label)}</label><div class="help">${esc(fld.help)}</div>
      <select data-sel="${fld.key}"><option value="">Not provided</option>${fld.options.map((o) => `<option value="${esc(o.value)}" ${state.form[fld.key] === o.value ? "selected" : ""}>${esc(o.label)}</option>`).join("")}</select></div>`).join("");
  }
  function valid() {
    const s = state.form;
    return state.files.clinical && state.files.dermoscopy && s.sex !== undefined && s.location !== undefined && s.elevation !== undefined && state.consent && state.health && state.health.ready;
  }

  register("analyze", (el) => {
    if (state.running) return renderProgress(el);
    if (!state.schema) { el.innerHTML = `<div class="card empty"><p class="muted">Loading the form\u2026</p></div>`; setTimeout(() => location.hash === "#/analyze" && App.route(), 800); return; }
    const basic = state.schema.fields.filter((x) => x.group === "basic");
    const ai = state.health && state.health.explanation_writer === "ai";
    el.innerHTML = `
    <div class="page-head"><div class="eyebrow">Step 1 of 2</div><h2>New analysis</h2><p>Provide two images of the same lesion and answer three quick questions. It takes about a minute.</p></div>
    ${state.error ? `<div class="callout danger">${icon("alert")}<div>${esc(state.error)}</div></div>` : ""}
    ${state.health && !state.health.ready ? `<div class="callout info">${icon("info")}<div><b>${state.health.status === "error" ? "The analysis model is unavailable." : "The system is warming up."}</b> ${state.health.status === "error" ? "Please try again later." : "You can prepare your images now; the Run button activates when it is ready."}</div></div>` : ""}
    <div class="card">
      <div class="card-title"><span class="num">1</span><h3>Upload the images</h3></div>
      <div class="grid g2">
        <div><h4>Clinical photograph</h4>${zone("clinical", "Clinical photo", "An ordinary close-up photo of the lesion, in focus and evenly lit.")}</div>
        <div><h4>Dermoscopy image</h4>${zone("dermoscopy", "Dermoscopy image", "A magnified image taken through a dermatoscope. A normal phone photo is not a dermoscopy image.")}</div>
      </div>
      <div class="callout info" style="margin:14px 0 0">${icon("info")}<div>Tips: keep the lesion centred, avoid filters, shadows, rulers or hair covering it, and use the same lesion in both images.</div></div>
    </div>
    <div class="card">
      <div class="card-title"><span class="num">2</span><h3>About the lesion</h3></div>
      ${basic.map(question).join("")}
      <details class="acc" ${state.form.management || state.form.level_of_diagnostic_difficulty ? "open" : ""}><summary>Clinician details (optional)</summary><div>
        <p class="muted" style="font-size:.86rem">These two fields come from the research dataset the system was trained on and are normally recorded by a doctor. Leave them blank if you are unsure. The seven-point checklist score is <b>not</b> needed: the system estimates the checklist features itself from the images.</p>${clinicianFields()}</div></details>
    </div>
    <div class="card">
      <label class="row" style="align-items:flex-start;gap:10px;cursor:pointer"><input type="checkbox" id="consent" ${state.consent ? "checked" : ""} style="margin-top:5px;width:18px;height:18px">
        <span>I understand this is a research tool that gives an <b>estimate, not a diagnosis</b>, and that it does not replace a doctor.</span></label>
      <p class="muted" style="font-size:.82rem;margin:10px 0 0 28px">Privacy: images are processed in memory and deleted automatically within about an hour; they are not saved to disk.${ai ? " To write the explanation, only the system\u2019s scores and your three answers (never the images) are sent to an AI writing service." : ""}</p>
      <div class="row" style="margin-top:16px"><button class="btn primary lg" id="run" ${valid() ? "" : "disabled"}>${icon("target")} Run analysis</button><span class="muted" id="hint" style="font-size:.86rem"></span></div>
    </div>${legal}`;
    wireForm(el);
    hint(el);
  });

  function hint(el) {
    const h = $("#hint", el); if (!h) return;
    const miss = [];
    if (!state.files.clinical) miss.push("clinical photo");
    if (!state.files.dermoscopy) miss.push("dermoscopy image");
    if (["sex", "location", "elevation"].some((k) => state.form[k] === undefined)) miss.push("the three questions");
    if (!state.consent) miss.push("the confirmation box");
    h.textContent = miss.length ? "Still needed: " + miss.join(", ") + "." : "";
    const b = $("#run", el); if (b) b.disabled = !valid();
  }
  function setFile(k, file, el) {
    if (!file) return;
    if (!/^image\/(jpeg|png|webp)$/.test(file.type)) { alert("Please choose a JPG or PNG image."); return; }
    if (file.size > MAXMB * 1048576) { alert("This image is larger than " + MAXMB + " MB. Please choose a smaller one."); return; }
    if (state.files[k + "_url"]) URL.revokeObjectURL(state.files[k + "_url"]);
    state.files[k] = file; state.files[k + "_url"] = null; state.error = null;
    App.route();
  }
  function wireForm(el) {
    $$(".dz", el).forEach((z) => {
      const k = z.dataset.k;
      z.addEventListener("dragover", (e) => { e.preventDefault(); z.classList.add("drag"); });
      z.addEventListener("dragleave", () => z.classList.remove("drag"));
      z.addEventListener("drop", (e) => { e.preventDefault(); z.classList.remove("drag"); if (!state.files[k]) setFile(k, e.dataTransfer.files[0], el); });
      z.addEventListener("keydown", (e) => { if ((e.key === "Enter" || e.key === " ") && !state.files[k]) $("input", z).click(); });
    });
    $$("input[data-in]", el).forEach((i) => i.addEventListener("change", () => setFile(i.dataset.in, i.files[0], el)));
    $$("[data-rm]", el).forEach((b) => b.addEventListener("click", (e) => {
      e.preventDefault(); e.stopPropagation(); const k = b.dataset.rm;
      if (state.files[k + "_url"]) URL.revokeObjectURL(state.files[k + "_url"]);
      state.files[k] = null; state.files[k + "_url"] = null; App.route();
    }));
    $$("[data-q]", el).forEach((b) => b.addEventListener("click", () => {
      state.form[b.dataset.q] = b.dataset.v;
      $$(`[data-q="${b.dataset.q}"]`, el).forEach((x) => x.classList.toggle("sel", x === b));
      hint(el);
    }));
    $$("[data-sel]", el).forEach((s) => s.addEventListener("change", () => { state.form[s.dataset.sel] = s.value; }));
    $("#consent", el).addEventListener("change", (e) => { state.consent = e.target.checked; hint(el); });
    $("#run", el).addEventListener("click", startRun);
  }

  async function startRun() {
    const fd = new FormData();
    fd.append("clinical", state.files.clinical); fd.append("dermoscopy", state.files.dermoscopy);
    ["sex", "location", "elevation", "level_of_diagnostic_difficulty", "management"].forEach((k) => fd.append(k, state.form[k] || ""));
    state.error = null; state.running = true; state.job = { step: 0, total: 8, stages: [] };
    App.route();
    try {
      const { job_id } = await (await api("/api/analyze", { method: "POST", body: fd })).json();
      await poll(job_id);
    } catch (e) {
      state.running = false; state.error = e.message; App.route();
    }
  }
  async function poll(jid) {
    for (;;) {
      const j = await getJSON("/api/jobs/" + jid);
      state.job = j;
      if (j.status === "done") {
        state.result = j.result; state.jobId = jid; state.running = false;
        sessionStorage.setItem("jobId", jid); go("result"); return;
      }
      if (j.status === "error") { state.running = false; state.error = j.error || "The analysis failed."; App.route(); return; }
      if (location.hash === "#/analyze") updateProgress();
      await new Promise((r) => setTimeout(r, 600));
    }
  }
  function renderProgress(el) {
    el.innerHTML = `<div class="page-head"><div class="eyebrow">Working</div><h2>Analysing your images\u2026</h2><p>This usually takes a few seconds. Please keep this page open.</p></div>
      <div class="card"><div class="progress"><i id="pbar" style="width:4%"></i></div><ul class="stagelist" id="stages"></ul></div>`;
    updateProgress();
  }
  function updateProgress() {
    const j = state.job || {}; const st = (j.stages && j.stages.length) ? j.stages : [
      "Checking your images", "Preparing the inputs", "Analysing the images with the multimodal transformer", "Reasoning over the clinical knowledge graph",
      "Applying the clinical heuristic", "Retrieving relevant medical knowledge (RAG)", "Writing the plain-language explanation", "Preparing your results"].map((l) => ({ label: l }));
    const cur = Math.max(0, (j.step || 1) - 1);
    const bar = $("#pbar"); if (bar) bar.style.width = Math.max(4, Math.round(100 * (j.step || 0) / st.length)) + "%";
    const ul = $("#stages"); if (!ul) return;
    ul.innerHTML = st.map((s, i) => `<li class="${i < cur ? "done" : i === cur ? "now" : ""}"><span class="st">${i < cur ? "\u2713" : ""}</span>${esc(s.label)}</li>`).join("");
  }

  /* ================= RESULT ================= */
  function ring(p, color) {
    const r = 54, c = 2 * Math.PI * r;
    return `<svg class="ring" viewBox="0 0 132 132"><circle cx="66" cy="66" r="${r}" fill="none" stroke="#e2e8f0" stroke-width="12"/><circle cx="66" cy="66" r="${r}" fill="none" stroke="${color}" stroke-width="12" stroke-linecap="round" stroke-dasharray="${(c * p).toFixed(1)} ${c.toFixed(1)}" transform="rotate(-90 66 66)"/><text x="66" y="64" text-anchor="middle" font-size="26" font-weight="800" fill="#0f172a">${Math.round(p * 100)}%</text><text x="66" y="84" text-anchor="middle" font-size="11" fill="#64748b">system score</text></svg>`;
  }
  const dl = () => `<a class="btn primary" href="/api/report/${state.jobId}.pdf" download>${icon("download")} Download PDF report</a>`;

  register("result", (el) => {
    if (needResult(el, "the diagnosis result")) return;
    const R = state.result, P = R.prediction, top = P.classes.find((c) => c.code === P.top_class);
    const srcs = R.explanation.sources;
    const about = srcs.filter((s) => s.type === "class" && s.id.startsWith("class_" + P.top_class.toLowerCase()));
    const ru = srcs.find((s) => s.id === `class_${P.runner_up.toLowerCase()}_overview`);
    const det = R.attributes.filter((a) => a.detected).length;
    const ov = R.overlays || {};
    const views = [["clinical", "Clinical photo", R.case.images.clinical], ["dermoscopy", "Dermoscopy", R.case.images.dermoscopy]];
    if (ov.lsm) views.push(["lsm", "AI focus: clinical", ov.lsm.image]);
    if (ov.dermoscopy_region) views.push(["dreg", "AI focus: dermoscopy", ov.dermoscopy_region.image]);
    el.innerHTML = `
    <div class="page-head row spread"><div><div class="eyebrow">Step 2 of 2</div><h2>Diagnosis result</h2><p>The system\u2019s estimate for this lesion. Remember: this is not a diagnosis.</p></div>
      <div class="row">${dl()}<a class="btn secondary" href="/api/results/${state.jobId}.json" download>${icon("download")} Data (JSON)</a></div></div>
    ${R.warnings && R.warnings.length ? `<div class="callout warn">${icon("alert")}<div>${R.warnings.map(esc).join("<br>")}</div></div>` : ""}
    <div class="card" style="border-left:6px solid ${top.color}">
      <div class="result-hero">${ring(top.final_prob, top.color)}
        <div style="flex:1;min-width:260px">
          <div class="muted" style="font-size:.8rem;font-weight:700;letter-spacing:.06em;text-transform:uppercase">Highest-scoring group</div>
          <h2 style="font-size:1.75rem;margin:2px 0 6px">${esc(top.name)}</h2>
          <span class="pill ${tone(top.group)}">${esc(top.group)}</span>
          <p style="margin:10px 0 0">${esc((state.info && state.info.classes.find((c) => c.code === top.code) || {}).short || "")} The runner-up is <b>${esc(P.runner_up_name)}</b> at ${pct(P.runner_up_prob)}.</p>
        </div></div>
      ${P.uncertain ? `<div class="callout warn" style="margin:16px 0 0">${icon("alert")}<div><b>The system is not sure.</b> The top scores are close, or the top score is not high. Treat this result with extra caution.</div></div>` : ""}
      ${P.heuristic_changed_top ? `<div class="callout info" style="margin:12px 0 0">${icon("info")}<div>The image network on its own preferred <b>${esc(P.classes.find((c) => c.code === P.neural_top_class).name)}</b>; the rule-based clinical check moved <b>${esc(top.name)}</b> to the top.</div></div>` : ""}
    </div>

    <div class="grid g2">
      <div class="card"><div class="card-title"><h3>Scores for all five groups</h3></div>
        <div class="tabs"><button class="tab active" data-m="final">Final scores</button><button class="tab" data-m="both">Before vs after rules</button></div>
        <div id="pbars" class="plot"></div></div>
      <div class="card"><div class="card-title"><h3>Your images</h3></div>
        <div class="tabs" id="vtabs">${views.map((v, i) => `<button class="tab ${i ? "" : "active"}" data-v="${v[0]}">${v[1]}</button>`).join("")}</div>
        <div class="imgbox"><img id="vimg" src="${views[0][2]}" alt="analysed image"></div></div>
    </div>

    <div class="card"><div class="card-title"><h3>What the system noticed in the lesion</h3><span class="pill info">${det} of 7 detected</span></div>
      <p class="muted" style="font-size:.9rem">Doctors look for seven patterns with a dermatoscope (the \u201c7-point checklist\u201d). The AI estimates each one from your images. A pattern counts as detected at 50% or more.</p>
      ${R.attributes.map((a) => `<div class="feat"><div class="nm"><b>${esc(a.name)}</b><small class="muted">${a.kind === "major" ? "Major" : "Minor"} checklist feature</small></div>
        <div><div class="muted" style="font-size:.85rem;margin-bottom:5px">${esc(a.plain)}</div><div class="bars"><div class="track" style="position:relative"><i style="width:${a.fused_prob * 100}%;background:${a.detected ? "#0e7490" : "#94a3b8"}"></i><span style="position:absolute;left:50%;top:-2px;bottom:-2px;width:2px;background:#ef4444;opacity:.55"></span></div></div></div>
        <div style="text-align:right"><b style="font-size:1.05rem">${pct(a.fused_prob)}</b> <span class="pill ${a.detected ? "ok" : "neutral"}">${a.detected ? "Detected" : "Not detected"}</span></div></div>`).join("")}</div>

    <div class="grid g2">
      <div class="card"><div class="card-title"><h3>About ${esc(top.name)}</h3></div>${about.map((s) => `<p>${esc(s.text)}</p>`).join("") || "<p class='muted'>No description available.</p>"}
        ${ru ? `<h4 style="margin-top:14px">Runner-up: ${esc(P.runner_up_name)}</h4><p>${esc(ru.text)}</p>` : ""}
        <a href="#/explanation">Read the full plain-language explanation \u2192</a></div>
      <div class="card"><div class="card-title"><h3>Information you provided</h3></div>
        <div class="tbl-wrap"><table><tbody>${R.case.metadata.map((m) => `<tr><td>${esc(m.question)}</td><td><b>${m.provided ? esc(m.answer) : '<span class="muted">Not provided</span>'}</b></td></tr>`).join("")}</tbody></table></div>
        <p class="muted" style="font-size:.82rem;margin-top:8px">Anything not provided is simply left blank; it adds no information to the analysis.</p></div>
    </div>${legal}`;

    const drawBars = (m) => {
      const cs = [...P.classes].sort((a, b) => b.final_prob - a.final_prob);
      const tr = [];
      if (m === "both") tr.push({ type: "bar", orientation: "h", name: "Image network only", y: cs.map((c) => c.name), x: cs.map((c) => c.neural_prob * 100), marker: { color: "#cbd5e1" }, text: cs.map((c) => (c.neural_prob * 100).toFixed(0) + "%"), textposition: "outside" });
      tr.push({ type: "bar", orientation: "h", name: "Final", y: cs.map((c) => c.name), x: cs.map((c) => c.final_prob * 100), marker: { color: cs.map((c) => c.color) }, text: cs.map((c) => (c.final_prob * 100).toFixed(1) + "%"), textposition: "outside", hovertemplate: "%{y}: %{x:.1f}%<extra></extra>" });
      plot("pbars", tr, { barmode: "group", yaxis: { autorange: "reversed", automargin: true }, xaxis: { range: [0, 112], title: "System score (%)" }, showlegend: m === "both", margin: { l: 10, r: 30, t: 6, b: 40 } }, 300);
    };
    drawBars("final");
    $$(".tab[data-m]", el).forEach((t) => t.onclick = () => { $$(".tab[data-m]", el).forEach((x) => x.classList.toggle("active", x === t)); drawBars(t.dataset.m); });
    const setV = (k) => { const v = views.find((x) => x[0] === k); $("#vimg").src = v[2]; $$("#vtabs .tab").forEach((x) => x.classList.toggle("active", x.dataset.v === k)); };
    $$("#vtabs .tab").forEach((t) => t.onclick = () => setV(t.dataset.v)); setV("clinical");
  });

  /* ================= EXPLANATION ================= */
  register("explanation", (el) => {
    if (needResult(el, "the plain-language explanation")) return;
    const E = state.result.explanation;
    el.innerHTML = `
    <div class="page-head row spread"><div><div class="eyebrow">In simple words</div><h2>AI explanation</h2><p>What the result means, why the system reached it, and what to do next.</p></div><div>${dl()}</div></div>
    ${E.notice ? `<div class="callout info">${icon("info")}<div>${esc(E.notice)}</div></div>` : ""}
    ${E.sections.map((s, i) => `<div class="card"><div class="expl-sec"><div class="num">${i + 1}</div><div><h3>${esc(s.title)}</h3>${s.blocks.map((b) => b.type === "p" ? `<p>${rich(b.text)}</p>` : `<ul>${b.items.map((x) => `<li>${rich(x)}</li>`).join("")}</ul>`).join("")}</div></div></div>`).join("")}

    <div class="card"><div class="card-title"><h3>Full glossary</h3></div>${(E.glossary || []).map((g) => `<details class="acc"><summary>${esc(g.term)}</summary><div>${esc(g.text)}</div></details>`).join("")}</div>${legal}`;
  });
})();
