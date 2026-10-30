/* Pages: knowledge graph, heuristic reasoning, intermediate results, model & method */
(() => {
  const { $, $$, esc, pct, f, rich, state, icon, api, plot, heat, register, needResult, legal } = App;

  const loading = (el) => { el.innerHTML = `<div class="card empty"><p class="muted">Loading model information\u2026</p></div>`; setTimeout(() => location.hash && App.route(), 1200); };
  const tbl = (head, rows, cls = "") => `<div class="tbl-wrap"><table class="${cls}"><thead><tr>${head.map((h) => `<th>${h}</th>`).join("")}</tr></thead><tbody>${rows.map((r) => `<tr>${r.map((c) => `<td>${c}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`;
  const kpi = (v, l) => `<div class="kpi"><div class="v">${v}</div><div class="l">${l}</div></div>`;
  const note = (t) => `<div class="callout info">${icon("info")}<div>${t}</div></div>`;
  const shape = (s) => "(" + s.join(" \u00d7 ") + ")";

  /* ================= KNOWLEDGE GRAPH ================= */
  const NODE_NAMES = (info) => {
    const m = {};
    info.attributes.forEach((a) => (m[a.code] = a.name));
    info.classes.forEach((c) => (m[c.code] = c.name));
    return m;
  };
  function layout(nodes, nAttr) {
    const pos = {};
    for (let i = 0; i < nAttr; i++) { const a = Math.PI / 2 + (2 * Math.PI * i) / nAttr; pos[nodes[i]] = [Math.cos(a), Math.sin(a)]; }
    const inner = nodes.slice(nAttr);
    inner.forEach((n, i) => { const a = Math.PI / 2 + (2 * Math.PI * i) / inner.length; pos[n] = [0.42 * Math.cos(a), 0.42 * Math.sin(a)]; });
    return pos;
  }
  function drawGraph(opt) {
    const info = state.info, g = info.graph, nodes = g.nodes, nAttr = g.num_attributes, R = state.result;
    const A = opt.which === "full" ? g.A_full : g.A_sparse, pos = layout(nodes, nAttr), names = NODE_NAMES(info);
    const fused = {}, finalp = {};
    if (R) { R.attributes.forEach((a) => (fused[a.code] = a.fused_prob)); R.prediction.classes.forEach((c) => (finalp[c.code] = c.final_prob)); }
    const rad = 0.16, traces = [], anns = [], lx = [], ly = [], lt = [], lh = [];
    let shown = 0;
    const anyDet = R && R.attributes.some((a) => a.detected);
    nodes.forEach((s, i) => nodes.forEach((t, j) => {
      const w = A[i][j]; if (i === j || w <= 0 || w < opt.minW) return;
      if (opt.focus && s !== opt.focus && t !== opt.focus) return;
      shown++;
      const p0 = pos[s], p1 = pos[t], mx = (p0[0] + p1[0]) / 2, my = (p0[1] + p1[1]) / 2, dx = p1[0] - p0[0], dy = p1[1] - p0[1];
      const c = [mx + rad * dy, my - rad * dx];
      const B = (u) => [(1 - u) ** 2 * p0[0] + 2 * (1 - u) * u * c[0] + u * u * p1[0], (1 - u) ** 2 * p0[1] + 2 * (1 - u) * u * c[1] + u * u * p1[1]];
      const active = opt.highlight && R && anyDet && fused[s] !== undefined && fused[s] >= 0.5;
      const dim = opt.highlight && R && anyDet && !active;
      const col = active ? "#0369a1" : dim ? "#cbd5e1" : "#475569";
      const xs = [], ys = [];
      for (let k = 0; k <= 16; k++) { const q = B(0.12 + 0.76 * (k / 16)); xs.push(q[0]); ys.push(q[1]); }
      traces.push({ type: "scatter", mode: "lines", x: xs, y: ys, line: { color: col, width: 0.8 + 5 * w }, opacity: active ? 0.95 : dim ? 0.55 : 0.7, hoverinfo: "skip", showlegend: false });
      const a1 = B(0.88), a0 = B(0.8);
      anns.push({ x: a1[0], y: a1[1], ax: a0[0], ay: a0[1], xref: "x", yref: "y", axref: "x", ayref: "y", showarrow: true, arrowhead: 2, arrowsize: 1.3, arrowwidth: 1.2, arrowcolor: col, opacity: 0.9 });
      const m = B(0.5); lx.push(m[0]); ly.push(m[1]); lt.push(w.toFixed(2)); lh.push(`<b>${s} \u2192 ${t}</b><br>${esc(names[s])} \u2192 ${esc(names[t])}<br>edge weight ${w.toFixed(4)}`);
    }));
    if (opt.labels) traces.push({ type: "scatter", mode: "markers+text", x: lx, y: ly, text: lt, textfont: { size: 9, color: "#0f172a" }, textposition: "middle center", marker: { size: 24, color: "rgba(255,255,255,.92)", line: { color: "#cbd5e1", width: 1 } }, hovertext: lh, hoverinfo: "text", showlegend: false });
    const att = R ? R.graph.node_attention : null, amax = att ? Math.max(...att) : 1;
    traces.push({
      type: "scatter", mode: "markers+text", x: nodes.map((n) => pos[n][0]), y: nodes.map((n) => pos[n][1]), text: nodes, textfont: { size: 11, color: "#0f172a", family: App.FONT.family },
      marker: {
        size: nodes.map((n, k) => (att && opt.highlight ? 40 + 22 * (att[k] / amax) : 46)), line: { color: "#1e3a5f", width: 2 },
        color: nodes.map((n, k) => (k < nAttr ? (opt.highlight && R ? `rgba(14,116,144,${0.25 + 0.7 * fused[n]})` : "#bae6fd") : (opt.highlight && R ? `rgba(234,88,12,${0.25 + 0.7 * finalp[n]})` : "#fed7aa"))),
      },
      hovertext: nodes.map((n, k) => `<b>${n}</b> ${esc(names[n])}<br>${k < nAttr ? "dermoscopic feature" : "lesion group"}` + (R && opt.highlight ? `<br>this case: ${k < nAttr ? "feature score " + pct(fused[n]) : "final score " + pct(finalp[n])}<br>node attention ${f(att[k], 3)}` : "")),
      hoverinfo: "text", showlegend: false,
    });
    plot("graphplot", traces, { xaxis: { visible: false, range: [-1.32, 1.32], fixedrange: true }, yaxis: { visible: false, range: [-1.22, 1.22], scaleanchor: "x", fixedrange: true }, annotations: anns, margin: { l: 0, r: 0, t: 0, b: 0 }, hovermode: "closest" }, 660);
    const c = $("#gcount"); if (c) c.textContent = shown + " edges shown";
  }
  function edgeType(i, j, nAttr) { const a = i < nAttr, b = j < nAttr; return a && b ? ["Feature \u2192 feature", "How often the two features appear together in the same lesion."] : a ? ["Feature \u2192 group", "How often lesions showing this feature belonged to that group."] : b ? ["Group \u2192 feature", "How often lesions of this group showed that feature."] : ["Group \u2192 group", "Relationship between two lesion groups."]; }

  register("graph", (el) => {
    if (!state.info) return loading(el);
    const info = state.info, g = info.graph, nodes = g.nodes, nAttr = g.num_attributes, R = state.result, names = NODE_NAMES(info);
    const A = g.A_sparse; let nnz = 0; A.forEach((r) => r.forEach((v) => v > 0 && nnz++));
    el.innerHTML = `
    <div class="page-head"><div class="eyebrow">Reasoning</div><h2>Clinical knowledge graph</h2>
      <p>A map that connects the seven dermoscopic features (outer ring) to the five lesion groups (inner ring). The <b>number on each arrow is its edge weight</b>: a stronger link means the two things occurred together more often in the training data.</p></div>
    <div class="grid g4" style="margin-bottom:20px">${kpi(nodes.length, "nodes (7 features + 5 groups)")}${kpi(nnz, "edges used by the model")}${kpi(g.order, "hop orders combined (multi-order encoder)")}${kpi(g.hidden, "node embedding size")}</div>
    <div class="card">
      <div class="row spread" style="margin-bottom:10px"><div class="row">
        <label>Edges: <select id="gwhich" style="min-width:180px"><option value="sparse">Used by the model (strongest)</option><option value="full">All connections</option></select></label>
        <label>Hide weights below <input id="gmin" class="range" type="range" min="0" max="0.9" step="0.05" value="0"> <b id="gminv">0.00</b></label></div>
        <div class="row"><label>Focus on: <select id="gfocus" style="min-width:150px"><option value="">All nodes</option>${nodes.map((n) => `<option value="${n}">${n} \u00b7 ${esc(names[n])}</option>`).join("")}</select></label><label><input type="checkbox" id="glab" checked> Edge weights</label>${R ? `<label><input type="checkbox" id="ghi" checked> Highlight this case</label>` : ""}<span class="pill neutral" id="gcount"></span></div></div>
      <div id="graphplot" class="plot"></div>
      <div class="legend"><span><i style="background:#bae6fd;border:2px solid #1e3a5f"></i>Feature</span><span><i style="background:#fed7aa;border:2px solid #1e3a5f"></i>Lesion group</span>
        <span>Arrow thickness \u221d weight</span>${R ? `<span><i style="background:#0369a1"></i>Blue arrows start at features detected in your lesion</span>` : ""}</div>
    </div>
    <div class="grid g-2-1">
      <div class="card"><div class="card-title"><h3>Strongest connections</h3></div><div id="edgetbl"></div></div>
      <div class="card"><div class="card-title"><h3>${R ? "Node attention for this case" : "Node meaning"}</h3></div>${R ? `<div id="nattn" class="plot"></div>` : tbl(["Node", "Meaning"], nodes.map((n) => [`<b>${n}</b>`, esc(names[n])]))}</div>
    </div>
    <div class="card"><div class="card-title"><h3>Adjacency matrices</h3></div>
      <div class="tabs" id="mtabs">${[["sparse", "Used by model"], ["full", "All connections"], ["a2c", "Feature \u2192 group"], ["c2a", "Group \u2192 feature"], ["int", "Feature \u2194 feature"], ["p1", "Proximity 1-hop"], ["p2", "Proximity 2-hop"], ["p3", "Proximity 3-hop"]].map((t, i) => `<button class="tab ${i ? "" : "active"}" data-t="${t[0]}">${t[1]}</button>`).join("")}</div>
      <div id="mplot" class="plot"></div></div>${legal}`;

    const cfg = { which: "sparse", minW: 0, labels: true, highlight: !!R };
    const redraw = () => drawGraph(cfg);
    $("#gwhich").onchange = (e) => { cfg.which = e.target.value; redraw(); edgeTable(); };
    $("#gmin").oninput = (e) => { cfg.minW = +e.target.value; $("#gminv").textContent = cfg.minW.toFixed(2); redraw(); };
    $("#glab").onchange = (e) => { cfg.labels = e.target.checked; redraw(); };
    if (R) $("#ghi").onchange = (e) => { cfg.highlight = e.target.checked; redraw(); };
    $("#gfocus").onchange = (e) => { cfg.focus = e.target.value; redraw(); };
    redraw();
    document.getElementById("graphplot").on("plotly_click", (ev) => {
      const pt = ev.points && ev.points.find((p) => p.text && nodes.includes(p.text));
      if (pt) { cfg.focus = cfg.focus === pt.text ? "" : pt.text; $("#gfocus").value = cfg.focus; redraw(); }
    });

    function edgeTable() {
      const M = cfg.which === "full" ? g.A_full : g.A_sparse, rows = [];
      nodes.forEach((s, i) => nodes.forEach((t, j) => { if (i !== j && M[i][j] > 0) rows.push([s, t, M[i][j], i, j]); }));
      rows.sort((a, b) => b[2] - a[2]);
      $("#edgetbl").innerHTML = tbl(["From", "To", "Weight", "Type"], rows.map((r) => [`<b>${r[0]}</b> <span class="muted">${esc(names[r[0]])}</span>`, `<b>${r[1]}</b> <span class="muted">${esc(names[r[1]])}</span>`, `<b>${r[2].toFixed(3)}</b>`, `<span class="pill neutral" title="${esc(edgeType(r[3], r[4], nAttr)[1])}">${edgeType(r[3], r[4], nAttr)[0]}</span>`]));
    }
    edgeTable();
    if (R) plot("nattn", [{ type: "bar", x: nodes, y: R.graph.node_attention, marker: { color: nodes.map((_, k) => (k < nAttr ? "#0e7490" : "#ea580c")) }, hovertemplate: "%{x}: %{y:.4f}<extra></extra>" }], { yaxis: { title: "attention" }, margin: { l: 50, r: 10, t: 10, b: 30 } }, 280);

    const attrs = nodes.slice(0, nAttr), cls = nodes.slice(nAttr);
    const drawM = (k) => {
      const C = { sparse: [g.A_sparse, nodes, nodes, "Edges kept by the sparse graph (strongest links per node). Row = source, column = target.", "Target", "Source"],
        full: [g.A_full, nodes, nodes, "Every learned connection before sparsification.", "Target", "Source"],
        a2c: [g.A_attr_to_class, cls, attrs, "P(group | feature): among training lesions with this feature, the share in each group.", "Lesion group", "Feature"],
        c2a: [g.A_class_to_attr, attrs, cls, "P(feature | group): the share of lesions in each group that show the feature (smoothed training prior).", "Feature", "Lesion group"],
        int: [g.A_internal, attrs, attrs, "How strongly two features tend to occur together.", "Feature", "Feature"] };
      ["1", "2", "3"].forEach((o) => (C["p" + o] = [g.proximity[o], nodes, nodes, `Multi-hop proximity: ${o}-hop path strength between nodes (used by the multi-order graph encoder).`, "Target", "Source"]));
      const [z, x, y, , xt, yt] = C[k]; heat("mplot", z, x, y, { xtitle: xt, ytitle: yt, zmin: 0, h: 460 });
      $$("#mtabs .tab").forEach((t) => t.classList.toggle("active", t.dataset.t === k));
    };
    $$("#mtabs .tab").forEach((t) => (t.onclick = () => drawM(t.dataset.t))); drawM("sparse");
  });

  /* ================= HEURISTIC ================= */
  register("heuristic", (el) => {
    if (!state.info) return loading(el);
    const info = state.info, H = info.heuristic, R = state.result, an = info.attributes, cl = info.classes;
    const lam = R ? R.heuristic.lambda : H.lambda, w = R ? R.heuristic.weights : { graph_weight: H.graph_weight, contradiction_weight: H.contradiction_weight, interaction_weight: H.interaction_weight };
    el.innerHTML = `
    <div class="page-head"><div class="eyebrow">Reasoning</div><h2>Heuristic reasoning</h2>
      <p>After the neural network gives its scores, a small rule-based step checks them against clinical knowledge learned from the training data. It rewards groups whose typical features were found and penalises groups that clash with what was seen. It is deliberately gentle (strength \u03bb = ${lam.toFixed(2)}), so it refines the network\u2019s answer rather than replacing it.</p></div>
    <div class="grid g4" style="margin-bottom:20px">
      <div class="card flat"><h4>1. Feature evidence</h4><p class="muted" style="font-size:.86rem;margin:0">How well the detected and missing features match what is typical for each group, weighted by how reliably each feature is detected.</p></div>
      <div class="card flat"><h4>2. Graph support</h4><p class="muted" style="font-size:.86rem;margin:0">Support routed along the knowledge-graph edges from the features to each group (weight ${w.graph_weight}).</p></div>
      <div class="card flat"><h4>3. Contradiction penalty</h4><p class="muted" style="font-size:.86rem;margin:0">A penalty when a strongly detected feature is unusual for a group (below 25% of that group\u2019s training cases; weight ${w.contradiction_weight}).</p></div>
      <div class="card flat"><h4>4. Interaction bonus</h4><p class="muted" style="font-size:.86rem;margin:0">A small bonus when pairs of features that often appear together in a group are both present (weight ${w.interaction_weight}).</p></div>
    </div>
    <div class="card"><div class="card-title"><h3>The rule</h3></div>
      <div class="formula">score<sub>g</sub> = <b>log</b>(neural score<sub>g</sub>) + <b>\u03bb</b> \u00b7 ( evidence<sub>g</sub> + ${w.graph_weight}\u00b7graph support<sub>g</sub> \u2212 ${w.contradiction_weight}\u00b7contradiction<sub>g</sub> + ${w.interaction_weight}\u00b7interaction<sub>g</sub> )<br>final scores = <b>softmax</b>(score) &nbsp;&nbsp; with \u03bb = ${lam.toFixed(2)}<br>feature score = average of the auxiliary head and the graph head</div></div>
    ${R ? `
    <div class="grid g2">
      <div class="card"><div class="card-title"><h3>Scores before and after</h3></div><div id="hba" class="plot"></div></div>
      <div class="card"><div class="card-title"><h3>Component scores by group</h3></div><div id="hcomp" class="plot"></div></div>
    </div>
    <div class="card"><div class="card-title"><h3>Numbers for this case</h3></div>
      ${tbl(["Group", "Neural score", "Feature evidence", "Graph support", "Contradiction", "Interaction", "Combined", "Final score", "Change"], [...R.prediction.classes].sort((a, b) => b.final_prob - a.final_prob).map((c) => [`<b>${esc(c.name)}</b>`, f(c.neural_prob, 4), f(c.attribute_evidence, 3), f(c.graph_support, 3), f(c.contradiction_penalty, 3), f(c.interaction_bonus, 3), f(c.combined_score, 3), `<b>${f(c.final_prob, 4)}</b>`, `<span style="color:${c.final_prob >= c.neural_prob ? "#0f766e" : "#b91c1c"}">${(c.final_prob - c.neural_prob >= 0 ? "+" : "") + ((c.final_prob - c.neural_prob) * 100).toFixed(2)} pp</span>`]))}</div>
    <div class="card"><div class="card-title"><h3>Contradiction check</h3></div>
      ${(() => { const rows = R.heuristic.contradictions.filter((c) => c.attributes.length); return rows.length ? tbl(["Group", "Features unusual for this group (with this case\u2019s score)"], rows.map((c) => [`<b>${esc(cl.find((x) => x.code === c.class).name)}</b>`, c.attributes.map((a) => `${esc(an.find((x) => x.code === a.code).name)} <span class="pill ${a.fused >= 0.5 ? "warn" : "neutral"}">${pct(a.fused)}</span>`).join(" ")])) : `<p class="muted">No feature is flagged as unusual for any group.</p>`; })()}</div>` : `<div class="callout info">${icon("info")}<div>Run an analysis to see how the rule changes the scores for a real case. The knowledge used by the rule is shown below.</div></div>`}
    <div class="grid g2">
      <div class="card"><div class="card-title"><h3>Feature reliability weights</h3></div><div id="hrel" class="plot"></div></div>
      <div class="card"><div class="card-title"><h3>Training prior: share of each group showing each feature</h3></div><div id="hprior" class="plot"></div></div>
    </div>${legal}`;
    if (R) {
      const cs = R.prediction.classes;
      plot("hba", [{ type: "bar", name: "Before (neural)", x: cs.map((c) => c.code), y: cs.map((c) => c.neural_prob * 100), marker: { color: "#94a3b8" } }, { type: "bar", name: "After (final)", x: cs.map((c) => c.code), y: cs.map((c) => c.final_prob * 100), marker: { color: "#0e7490" } }], { barmode: "group", yaxis: { title: "score (%)" } }, 300);
      const comp = [["attribute_evidence", "Feature evidence", "#0e7490"], ["graph_support", "Graph support", "#7c3aed"], ["contradiction_penalty", "Contradiction", "#dc2626"], ["interaction_bonus", "Interaction", "#f59e0b"]];
      plot("hcomp", comp.map(([k, n, c]) => ({ type: "bar", name: n, x: cs.map((x) => x.code), y: cs.map((x) => x[k]), marker: { color: c } })), { barmode: "group", yaxis: { title: "component value" } }, 300);
    }
    plot("hrel", [{ type: "bar", x: an.map((a) => a.name), y: H.reliability, marker: { color: "#0e7490" }, text: H.reliability.map((v) => v.toFixed(2)), textposition: "outside", hovertemplate: "%{x}: %{y:.3f}<extra></extra>" }], { yaxis: { range: [0, Math.max(...H.reliability) * 1.2], title: "reliability" }, xaxis: { tickangle: -25, automargin: true } }, 300);
    heat("hprior", H.class_attr_prior, an.map((a) => a.code), cl.map((c) => c.name), { zmin: 0, zmax: 1, scale: "Blues", xtitle: "Feature", ml: 130, h: 300 });
  });

  /* ================= INTERMEDIATE RESULTS ================= */
  register("intermediate", (el) => {
    if (needResult(el, "every intermediate output and plot")) return;
    const R = state.result, X = R.internals, ov = R.overlays || {}, info = state.info, an = R.attributes;
    const stages = [["s0", "0 Inputs"], ["s1", "1 Metadata"], ["s2", "2 Clinical branch"], ["s3", "3 Dermoscopy branch"], ["s4", "4 Fusion & tokens"], ["s5", "5 Graph"], ["s6", "6 Prediction heads"], ["s7", "7 Heuristic"], ["s8", "8 Explanation (RAG)"], ["s9", "9 Timing & shapes"]];
    const stage = (id, n, t, d, body) => `<div class="card stage" id="${id}"><div class="card-title"><span class="num">${n}</span><h3>${t}</h3></div><p class="muted" style="margin-top:-4px">${d}</p>${body}</div>`;
    const cross = ov.cross_modal || {};
    const bars = (obj) => obj.map((s) => `<div class="b" style="display:grid;grid-template-columns:150px 1fr 52px;gap:8px;align-items:center;font-size:.83rem;margin:4px 0"><span>${esc(s.label)}</span><div class="track" style="height:8px;background:#e9eef5;border-radius:9px;overflow:hidden"><i style="display:block;height:100%;width:${s.prob * 100}%;background:#0e7490"></i></div><span>${pct(s.prob, 1)}</span></div>`).join("");
    el.innerHTML = `
    <div class="page-head row spread"><div><div class="eyebrow">Technical</div><h2>Intermediate results</h2><p>Every stage of the pipeline for this case, in order, with the actual outputs and plots: from the prepared inputs to the final scores. Case <code>${esc(R.case.id)}</code>.</p></div>
      <div class="row"><a class="btn secondary" href="/api/results/${state.jobId}.json" download>${icon("download")} All data (JSON)</a><a class="btn primary" href="/api/report/${state.jobId}.pdf" download>${icon("download")} PDF report</a></div></div>
    <div class="stage-nav">${stages.map((s) => `<a href="#/intermediate" data-jump="${s[0]}">${s[1]}</a>`).join("")}</div>

    ${stage("s0", 0, "Inputs and preprocessing", "Both images are resized and normalised; the answers are turned into numbers.", `
      <div class="grid g2"><div><div class="imgbox"><img src="${R.case.images.clinical}" alt="clinical"></div><p class="muted" style="text-align:center;font-size:.8rem">Clinical photo (original ${R.case.original_sizes.clinical.join("\u00d7")} px)</p></div>
      <div><div class="imgbox"><img src="${R.case.images.dermoscopy}" alt="dermoscopy"></div><p class="muted" style="text-align:center;font-size:.8rem">Dermoscopy (original ${R.case.original_sizes.dermoscopy.join("\u00d7")} px)</p></div></div>
      <div style="margin-top:14px">${tbl(["Input", "Original", "Preparation", "Tensor shape", "dtype", "min", "max", "mean"], X.preprocessing.map((r) => [`<b>${esc(r.input)}</b>`, r.original ? r.original.join("\u00d7") : "-", esc(r.resize), `<span class="mono">${shape(r.shape)}</span>`, r.dtype, f(r.min, 3), f(r.max, 3), f(r.mean, 3)]))}</div>`)}

    ${stage("s1", 1, "Metadata encoding", "Each answer becomes a block of 0/1 values (one-hot). Blocks for information that was not provided stay all zero. The vector has 30 numbers.", `
      ${tbl(["Field", "Slice", "Answer", "One-hot block"], X.metadata_encoding.map((m) => [`<span class="mono">${esc(m.field)}</span>`, `[${m.slice[0]}:${m.slice[1]}]`, m.active ? `<b>${esc(m.active)}</b>` : '<span class="muted">not provided (all zeros)</span>', `<div class="onehot">${m.one_hot.map((v) => `<i class="${v ? "on" : ""}" title="${v}"></i>`).join("")}</div>`]))}
      <p class="muted" style="font-size:.84rem;margin-top:8px">Full vector: <span class="mono">[${X.metadata_vector.join(", ")}]</span></p>`)}

    ${stage("s2", 2, "Clinical branch: lesion selection module (LSM)", "A 12 \u00d7 12 score map over the clinical photo picks four close-up regions (crops) that the network studies at higher detail.", ov.lsm ? `
      <div class="grid g3"><div><div class="imgbox"><img src="${ov.lsm.image}" alt="lsm overlay"></div><p class="muted" style="font-size:.8rem;text-align:center">Score map overlay + chosen crops</p></div>
        <div><div id="lsmheat" class="plot" style="min-height:340px"></div></div>
        <div><h4>Selected crops (224 \u00d7 224 px)</h4><div class="cropgrid" style="grid-template-columns:repeat(2,1fr)">${ov.lsm.crops.map((c, i) => `<div><img src="${c}" alt="crop ${i + 1}"><small class="muted">Crop ${i + 1} \u00b7 centre (${ov.lsm.centers_px[i].map((v) => Math.round(v)).join(", ")})</small></div>`).join("")}</div></div></div>` : `<p class="muted">Not available.</p>`)}

    ${stage("s3", 3, "Dermoscopy branch: regions and attention", "The dermoscopy image is split into 16 regions (4 \u00d7 4). A region transformer attends across them; the last layer\u2019s attention from its summary token is shown, plus attention by layer and the texture-template usage (MSTA).", ov.dermoscopy_region ? `
      <div class="grid g3"><div><div class="imgbox"><img src="${ov.dermoscopy_region.image}" alt="dermoscopy attention"></div><p class="muted" style="font-size:.8rem;text-align:center">Last-layer attention over 16 regions</p></div>
        <div><div id="dbylayer" class="plot"></div></div><div><div id="dmsta" class="plot"></div></div></div>` : `<p class="muted">Not available.</p>`)}

    ${stage("s4", 4, "Cross-modal fusion and the five tokens", "The clinical, dermoscopy and metadata views are fused by cross-modal attention into five 768-number tokens. A learned weight decides how much each token counts.", `
      <div class="grid g2"><div><h4>Metadata token attention over dermoscopy regions</h4><div id="cmeta" class="plot"></div></div><div><h4>Clinical global token attention over dermoscopy regions</h4><div id="cclin" class="plot"></div></div></div>
      <div class="grid g2"><div><h4>Dermoscopy token attention over clinical crops</h4><div id="ccrop" class="plot"></div></div><div><h4>Learned token weights</h4><div id="tokw" class="plot"></div></div></div>
      <h4 style="margin-top:8px">Token values (5 \u00d7 768)</h4><div id="tokheat" class="plot"></div>
      ${tbl(["Token", "Meaning", "L2 norm", "mean", "std", "min", "max"], X.remix.token_stats.map((t) => [`<span class="mono">${esc(t.token)}</span>`, esc(t.label), f(t.l2, 2), f(t.mean, 4), f(t.std, 3), f(t.min, 2), f(t.max, 2)]))}
      <h4 style="margin-top:14px">Image-model diagnosis probabilities (before the graph)</h4><div id="updiag" class="plot"></div>`)}

    ${stage("s5", 5, "Knowledge graph stage", "The tokens and the learned semantic features become 12 node features; a multi-order graph encoder then mixes information along the graph edges (see the Knowledge graph page).", `
      <div class="grid g2"><div><h4>Node feature norms</h4><div id="nfn" class="plot"></div></div><div><h4>Node attention</h4><div id="nat" class="plot"></div></div></div>
      <h4>Node features (12 \u00d7 768)</h4><div id="nfheat" class="plot"></div><h4>Graph node embeddings (12 \u00d7 256)</h4><div id="embheat" class="plot"></div>
      ${tbl(["Node", "L2 norm", "mean", "std", "attention"], X.graph_stage.embedding_stats.map((r) => [`<b>${esc(r.node)}</b>`, f(r.l2, 3), f(r.mean, 4), f(r.std, 3), f(r.attention, 4)]))}`)}

    ${stage("s6", 6, "Prediction heads", "Three heads read the fused features: a diagnosis head (5 groups), an auxiliary feature head and a graph feature head (7 features each). The image model also predicts each checklist feature\u2019s detailed state.", `
      <div class="grid g2"><div><h4>Diagnosis logits and neural probabilities</h4><div id="dlog" class="plot"></div></div><div><h4>Feature scores by head</h4><div id="alog" class="plot"></div></div></div>
      <h4 style="margin-top:8px">Detailed feature states predicted by the image model</h4>
      <div class="grid g3">${an.map((a) => `<div class="card flat" style="padding:14px;margin:0"><b>${esc(a.name)}</b><div class="muted" style="font-size:.78rem;margin-bottom:6px">Most likely: ${esc(a.top_state)}</div>${bars(a.upstream_states)}</div>`).join("")}</div>
      <h4 style="margin-top:14px">Learned feature weights</h4><div id="attw" class="plot"></div>`)}

    ${stage("s7", 7, "Heuristic adjustment", "Gentle rule-based refinement of the neural scores. Full detail on the Heuristic reasoning page.", `
      <div class="grid g2"><div id="h7a" class="plot"></div><div>${tbl(["Group", "Neural", "Final", "Evidence", "Graph", "Contra.", "Interact."], R.prediction.classes.map((c) => [`<b>${c.code}</b>`, f(c.neural_prob, 3), `<b>${f(c.final_prob, 3)}</b>`, f(c.attribute_evidence, 2), f(c.graph_support, 2), f(c.contradiction_penalty, 2), f(c.interaction_bonus, 2)]))}
      <p class="muted" style="font-size:.84rem;margin-top:8px">\u03bb = ${R.heuristic.lambda.toFixed(2)} \u00b7 graph-embedding term ${R.heuristic.graph_embedding_term_used ? "used" : "not used"} \u00b7 weights ${esc(JSON.stringify(R.heuristic.weights))}</p></div></div>`)}

    ${stage("s8", 8, "Explanation", "The case is summarised in numbers and turned into the plain-language explanation.", `
      <details class="acc"><summary>Case summary passed to the explanation writer (numbers only, never images)</summary><div><pre class="mono" style="white-space:pre-wrap;font-size:.76rem;max-height:340px;overflow:auto">${esc(JSON.stringify(R.case_summary, null, 1))}</pre></div></details>`)}

    ${stage("s9", 9, "Tensor shapes and timing", "Sizes of the main tensors and how long each stage took on the server.", `
      <div class="grid g2"><div>${tbl(["Tensor", "Shape"], X.tensor_shapes.map((t) => [`<span class="mono">${t.tensor}</span>`, `<span class="mono">${shape(t.shape)}</span>`]))}</div><div><div id="timing" class="plot"></div></div></div>`)}
    ${legal}`;

    $$("[data-jump]", el).forEach((a) => (a.onclick = (e) => { e.preventDefault(); document.getElementById(a.dataset.jump).scrollIntoView({ behavior: "smooth" }); }));
    const cls = R.prediction.classes, nodes = info ? info.graph.nodes : [];
    if (ov.lsm) heat("lsmheat", ov.lsm.grid, [...Array(ov.lsm.side).keys()].map((i) => i + 1), [...Array(ov.lsm.side).keys()].map((i) => i + 1), { text: false, gap: 0, tf: ".1e", scale: "Inferno", h: 340, xtitle: "column", ytitle: "row", hover: "row %{y}, col %{x}: %{z:.4f}", ml: 40 });
    if (ov.dermoscopy_region) {
      heat("dbylayer", ov.dermoscopy_region.by_layer, [...Array(16).keys()].map((i) => "R" + (i + 1)), [...Array(12).keys()].map((i) => "L" + (i + 1)), { text: false, gap: 0, scale: "Viridis", h: 340, xtitle: "region (row-major)", ytitle: "layer", hover: "layer %{y}, region %{x}: %{z:.4f}", ml: 40 });
      if (X.remix.msta_template_usage) heat("dmsta", X.remix.msta_template_usage, [...Array(32).keys()].map((i) => "T" + (i + 1)), [...Array(16).keys()].map((i) => "R" + (i + 1)), { text: false, gap: 0, scale: "Magma", h: 340, xtitle: "texture template", ytitle: "region", hover: "region %{y}, template %{x}: %{z:.4f}", ml: 40 });
    }
    const g44 = (grid, id) => grid ? heat(id, grid, ["c1", "c2", "c3", "c4"], ["r1", "r2", "r3", "r4"], { text: "%{z:.3f}", scale: "Inferno", h: 280, ml: 40 }) : ($("#" + id).innerHTML = '<p class="muted">Not available.</p>');
    g44(cross.metadata_to_dermoscopy_grid, "cmeta"); g44(cross.clinical_to_dermoscopy_grid, "cclin");
    const cr = cross.dermoscopy_to_clinical_crops || [], mc = cross.metadata_to_clinical_crops || [];
    plot("ccrop", [{ type: "bar", name: "Dermoscopy token", x: cr.map((_, i) => "Crop " + (i + 1)), y: cr, marker: { color: "#0e7490" } }, { type: "bar", name: "Metadata token", x: mc.map((_, i) => "Crop " + (i + 1)), y: mc, marker: { color: "#f59e0b" } }], { barmode: "group", yaxis: { title: "attention share" } }, 280);
    plot("tokw", [{ type: "bar", orientation: "h", y: R.weights.tokens.map((t) => t.label), x: R.weights.tokens.map((t) => t.weight), marker: { color: "#7c3aed" }, text: R.weights.tokens.map((t) => t.weight.toFixed(3)), textposition: "outside" }], { yaxis: { autorange: "reversed", automargin: true }, xaxis: { title: "weight", range: [0, Math.max(...R.weights.tokens.map((t) => t.weight)) * 1.25] }, margin: { l: 10, r: 40, t: 10, b: 40 } }, 280);
    heat("tokheat", X.remix.tokens, [...Array(X.remix.tokens[0].length).keys()], X.remix.token_stats.map((t) => t.label), { text: false, gap: 0, scale: "RdBu", rev: true, h: 260, ml: 190, xtitle: "embedding dimension", hover: "%{y}, dim %{x}: %{z:.3f}", layout: {} });
    plot("updiag", [{ type: "bar", x: X.remix.upstream_diagnosis.map((d) => d.code), y: X.remix.upstream_diagnosis.map((d) => d.prob * 100), marker: { color: cls.map((c) => c.color) }, text: X.remix.upstream_diagnosis.map((d) => (d.prob * 100).toFixed(1) + "%"), textposition: "outside" }], { yaxis: { title: "score (%)", range: [0, 110] } }, 260);
    plot("nfn", [{ type: "bar", x: nodes, y: X.graph_stage.node_feature_norms, marker: { color: "#0e7490" } }], { yaxis: { title: "L2 norm" } }, 260);
    plot("nat", [{ type: "bar", x: nodes, y: R.graph.node_attention, marker: { color: "#ea580c" } }], { yaxis: { title: "attention" } }, 260);
    heat("nfheat", X.graph_stage.node_features_heat, [...Array(X.graph_stage.node_features_heat[0].length).keys()], nodes, { text: false, gap: 0, scale: "RdBu", rev: true, h: 340, xtitle: "feature dimension", hover: "%{y}, dim %{x}: %{z:.3f}" });
    heat("embheat", X.graph_stage.embeddings, [...Array(X.graph_stage.embeddings[0].length).keys()], nodes, { text: false, gap: 0, scale: "RdBu", rev: true, h: 340, xtitle: "embedding dimension", hover: "%{y}, dim %{x}: %{z:.3f}" });
    plot("dlog", [{ type: "bar", name: "logit", x: cls.map((c) => c.code), y: cls.map((c) => c.logit), marker: { color: "#64748b" } }, { type: "bar", name: "probability", x: cls.map((c) => c.code), y: cls.map((c) => c.neural_prob), yaxis: "y2", marker: { color: "#0e7490" } }], { barmode: "group", yaxis: { title: "logit" }, yaxis2: { overlaying: "y", side: "right", title: "probability", range: [0, 1] } }, 280);
    plot("alog", [{ type: "bar", name: "Auxiliary head", x: an.map((a) => a.code), y: an.map((a) => a.aux_prob), marker: { color: "#94a3b8" } }, { type: "bar", name: "Graph head", x: an.map((a) => a.code), y: an.map((a) => a.graph_prob), marker: { color: "#38bdf8" } }, { type: "bar", name: "Combined", x: an.map((a) => a.code), y: an.map((a) => a.fused_prob), marker: { color: "#0e7490" } }], { barmode: "group", yaxis: { title: "probability", range: [0, 1] }, shapes: [{ type: "line", x0: -0.5, x1: 6.5, y0: 0.5, y1: 0.5, line: { color: "#ef4444", dash: "dot" } }] }, 280);
    plot("attw", [{ type: "bar", x: R.weights.attributes.map((a) => a.code), y: R.weights.attributes.map((a) => a.weight), marker: { color: "#7c3aed" }, text: R.weights.attributes.map((a) => a.weight.toFixed(3)), textposition: "outside" }], { yaxis: { title: "weight" } }, 250);
    plot("h7a", [{ type: "bar", name: "Neural", x: cls.map((c) => c.code), y: cls.map((c) => c.neural_prob * 100), marker: { color: "#94a3b8" } }, { type: "bar", name: "Final", x: cls.map((c) => c.code), y: cls.map((c) => c.final_prob * 100), marker: { color: "#0e7490" } }], { barmode: "group", yaxis: { title: "score (%)" } }, 280);
    const tm = Object.entries(X.timings_seconds);
    plot("timing", [{ type: "bar", orientation: "h", y: tm.map((t) => t[0].replace(/_/g, " ")), x: tm.map((t) => t[1]), marker: { color: "#0e7490" }, text: tm.map((t) => t[1].toFixed(2) + " s"), textposition: "outside" }], { yaxis: { autorange: "reversed", automargin: true }, xaxis: { title: "seconds" }, margin: { l: 10, r: 50, t: 10, b: 40 } }, 300);
  });

  /* ================= MODEL & METHOD ================= */
  register("method", (el) => {
    if (!state.info) return loading(el);
    const I = state.info, A = I.architecture, D = I.dataset;
    const params = Object.entries(A.parameters), total = params.reduce((s, p) => s + p[1], 0);
    el.innerHTML = `
    <div class="page-head"><div class="eyebrow">Technical</div><h2>Model & method</h2><p>What is inside the system, what it was trained on, and where its limits are.</p></div>
    <div class="card"><div class="card-title"><h3>Architecture</h3></div>
      <div class="flow">
        <div class="fb"><b>Clinical branch</b>Global ${A.global_clinical_size}\u00b2 view + ${A.lsm_k} lesion-selected ${A.local_patch_size}\u00b2 crops (Swin transformer)</div><div class="arrow">+</div>
        <div class="fb"><b>Dermoscopy branch</b>${A.dermoscopy_size}\u00b2 image \u2192 ${A.derm_windows} windows (ViT-S/16) \u2192 MSTA texture templates \u2192 region transformer</div><div class="arrow">+</div>
        <div class="fb"><b>Metadata</b>${A.meta_dim}-d one-hot from Derm7pt fields \u2192 embedding</div><div class="arrow">\u2192</div>
        <div class="fb"><b>Cross-modal fusion</b>${A.num_tokens} tokens \u00d7 ${A.token_dim} dims, learned token weights</div></div>
      <div class="flow" style="margin-top:10px"><div class="arrow">\u21b3</div>
        <div class="fb"><b>Node features</b>Tokens + ${A.semantic_dim}-d semantic features \u2192 12 graph nodes</div><div class="arrow">\u2192</div>
        <div class="fb"><b>Multi-order graph encoder</b>${A.graph_order} hop orders, hidden size ${A.graph_hidden}</div><div class="arrow">\u2192</div>
        <div class="fb"><b>7PCL hybrid heads</b>5-group diagnosis + 7 checklist features (auxiliary and graph)</div><div class="arrow">\u2192</div>
        <div class="fb"><b>Heuristic</b>\u03bb = ${I.heuristic.lambda.toFixed(2)} rule-based refinement</div><div class="arrow">\u2192</div>
        <div class="fb"><b>RAG explanation</b>Retrieval + plain-language writer</div></div></div>
    <div class="grid g2">
      <div class="card"><div class="card-title"><h3>Parameters</h3><span class="pill info">${(total / 1e6).toFixed(1)} M total</span></div><div id="ppar" class="plot"></div>
        ${tbl(["Module", "Tensors loaded", "Strict match"], A.state_dict_loads.map((r) => [esc(r.module), r.tensors, r.strict ? "\u2713 yes" : "no"]))}</div>
      <div class="card"><div class="card-title"><h3>Key settings</h3></div>${tbl(["Setting", "Value"], [["Clinical image size", A.clinical_size + " px"], ["Dermoscopy image size", A.dermoscopy_size + " px"], ["Local crops (LSM)", A.lsm_k + " \u00d7 " + A.local_patch_size + " px"], ["Dermoscopy windows", A.derm_windows + " (" + A.derm_window_size + " px)"], ["Fused tokens", A.num_tokens + " \u00d7 " + A.token_dim], ["Token names", A.token_names.join(", ")], ["Metadata columns (model input)", A.metadata_columns.join(", ") + " \u2014 the checklist score is never asked; it is left blank"], ["Graph nodes", I.graph.nodes.join(", ")]])}</div>
    </div>
    ${D ? `<div class="grid g2">
      <div class="card"><div class="card-title"><h3>Training data: Derm7pt</h3></div><p>${D.total_cases} cases, each with a clinical photo, a dermoscopy image and seven-point checklist annotations. ${D.train_cases ? `${D.train_cases} cases are in the training split used here.` : ""} The five groups are very unbalanced, so scores for the smaller groups are less reliable.</p><div id="pcls" class="plot"></div></div>
      <div class="card"><div class="card-title"><h3>How common each feature is (training)</h3></div><div id="pattr" class="plot"></div></div></div>` : ""}
    ${legal}`;
    plot("ppar", [{ type: "bar", orientation: "h", y: params.map((p) => p[0]), x: params.map((p) => p[1] / 1e6), marker: { color: "#0e7490" }, text: params.map((p) => (p[1] / 1e6).toFixed(2) + " M"), textposition: "outside", cliponaxis: false }], { yaxis: { autorange: "reversed", automargin: true }, xaxis: { title: "million parameters", range: [0, Math.max(...params.map((p) => p[1] / 1e6)) * 1.3] }, margin: { l: 10, r: 60, t: 10, b: 40 } }, 210);
    if (D) {
      const cs = I.classes;
      plot("pcls", [{ type: "bar", name: "All cases", x: cs.map((c) => c.code), y: cs.map((c) => D.class_counts_all[c.code]), marker: { color: "#cbd5e1" } }, D.class_counts_train ? { type: "bar", name: "Training split", x: cs.map((c) => c.code), y: cs.map((c) => D.class_counts_train[c.code]), marker: { color: "#0e7490" } } : null].filter(Boolean), { barmode: "group", yaxis: { title: "cases" } }, 280);
      if (D.attribute_positive_rate_train) plot("pattr", [{ type: "bar", x: I.attributes.map((a) => a.name), y: I.attributes.map((a) => D.attribute_positive_rate_train[a.code] * 100), marker: { color: "#0e7490" }, text: I.attributes.map((a) => (D.attribute_positive_rate_train[a.code] * 100).toFixed(0) + "%"), textposition: "outside" }], { yaxis: { title: "% of cases", range: [0, 60] }, xaxis: { tickangle: -25, automargin: true } }, 280);
    }
  });
})();
