/* Core: state, helpers, API, router, navigation. Plain JS, no build step. */
const App = (() => {
  const $ = (s, el = document) => el.querySelector(s);
  const $$ = (s, el = document) => [...el.querySelectorAll(s)];
  const esc = (t) => String(t ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const pct = (x, d = 0) => (x * 100).toFixed(d) + "%";
  const f = (x, d = 3) => (x == null ? "-" : Number(x).toFixed(d));
  /* **bold** -> HTML; citation markers [S#] are stripped (sources are not shown to the user) */
  const rich = (t) => esc(t)
    .replace(/\*\*(.+?)\*\*/g, "<b>$1</b>")
    .replace(/\s*\[S\d+\]/g, "");

  const state = {
    info: null, schema: null, health: null, result: null, jobId: null,
    files: { clinical: null, dermoscopy: null },
    form: { sex: undefined, location: undefined, elevation: undefined, level_of_diagnostic_difficulty: "", management: "" },
    consent: false, running: false, job: null, error: null,
  };

  const I = { // icons (feather-style paths)
    home: '<path d="M3 11l9-8 9 8v9a1 1 0 0 1-1 1h-5v-6H9v6H4a1 1 0 0 1-1-1z"/>',
    upload: '<path d="M12 16V4m0 0l-4 4m4-4l4 4M4 16v3a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-3"/>',
    target: '<circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="4"/><circle cx="12" cy="12" r=".6"/>',
    chat: '<path d="M21 12a8 8 0 0 1-11.6 7.1L4 20l1-4.6A8 8 0 1 1 21 12z"/>',
    graph: '<circle cx="5" cy="6" r="2.4"/><circle cx="19" cy="6" r="2.4"/><circle cx="12" cy="19" r="2.4"/><path d="M7 7l4 10M17 7l-4 10M7.4 6h9.2"/>',
    rules: '<path d="M4 6h16M4 12h10M4 18h6"/><path d="M17 15l2 2 3-4"/>',
    layers: '<path d="M12 3l9 5-9 5-9-5z"/><path d="M3 13l9 5 9-5M3 17.5l9 5 9-5" opacity=".6"/>',
    book: '<path d="M4 4h11a3 3 0 0 1 3 3v13H7a3 3 0 0 1-3-3z"/><path d="M4 17a3 3 0 0 1 3-3h11"/>',
    download: '<path d="M12 4v11m0 0l-4-4m4 4l4-4M4 19h16"/>',
    info: '<circle cx="12" cy="12" r="9"/><path d="M12 11v5M12 8v.1"/>',
    alert: '<path d="M12 3l10 18H2z"/><path d="M12 10v5M12 18v.1"/>',
    check: '<path d="M5 13l4 4L19 7"/>',
    plus: '<path d="M12 5v14M5 12h14"/>',
  };
  const icon = (n, cls = "") => `<svg viewBox="0 0 24 24" class="${cls}" aria-hidden="true">${I[n] || ""}</svg>`;

  /* ---- API ---- */
  async function api(path, opts) {
    const r = await fetch(path, opts);
    if (!r.ok) {
      let msg = "Request failed (" + r.status + ")";
      try { const j = await r.json(); if (j.detail) msg = typeof j.detail === "string" ? j.detail : msg; } catch (e) {}
      const err = new Error(msg); err.status = r.status; throw err;
    }
    return r;
  }
  const getJSON = async (p) => (await api(p)).json();

  async function loadStatic() {
    try { state.schema = await getJSON("/api/meta-schema"); } catch (e) {}
    for (let i = 0; i < 90 && !state.info; i++) {
      try { state.health = await getJSON("/api/health"); } catch (e) {}
      if (state.health && state.health.status === "error") break;
      try { state.info = await getJSON("/api/model-info"); }
      catch (e) { await new Promise((r) => setTimeout(r, 1500)); }
      updateStatus();
    }
    try { state.health = await getJSON("/api/health"); } catch (e) {}
    updateStatus();
    if (state.info) route(); // re-render pages that depend on model info
  }

  /* ---- plotting ---- */
  const FONT = { family: "-apple-system,BlinkMacSystemFont,Segoe UI,Inter,Roboto,Arial,sans-serif", size: 12, color: "#334155" };
  function plot(id, data, layout = {}, height) {
    const el = typeof id === "string" ? document.getElementById(id) : id;
    if (!el || !window.Plotly) return;
    const lay = Object.assign({
      font: FONT, paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
      margin: { l: 50, r: 16, t: 28, b: 42 }, height: height || undefined,
      legend: { orientation: "h", y: -0.2, font: { size: 11 } },
      hoverlabel: { bgcolor: "#0f172a", bordercolor: "#0f172a", font: { color: "#fff", size: 12, family: FONT.family } },
    }, layout);
    Plotly.newPlot(el, data, lay, { displayModeBar: false, responsive: true });
  }
  function heat(id, z, x, y, o = {}) {
    plot(id, [{
      type: "heatmap", z, x, y, colorscale: o.scale || "YlGnBu", reversescale: o.rev ?? false,
      zmin: o.zmin, zmax: o.zmax, xgap: o.gap ?? 1, ygap: o.gap ?? 1,
      texttemplate: o.text === false ? undefined : (o.text || "%{z:.2f}"), textfont: { size: o.fs || 11 },
      hovertemplate: (o.hover || "%{y} \u2192 %{x}: %{z:.4f}") + "<extra></extra>",
      showscale: o.scalebar ?? true, colorbar: { thickness: 10, len: 0.9, tickformat: o.tf || ".2g" },
    }], Object.assign({
      margin: { l: o.ml || 70, r: 20, t: o.mt ?? 10, b: o.mb || 50 },
      yaxis: { autorange: "reversed", automargin: true, title: o.ytitle }, xaxis: { automargin: true, title: o.xtitle, side: o.top ? "top" : "bottom" },
    }, o.layout || {}), o.h);
  }

  /* ---- navigation ---- */
  const NAV = [
    { group: "Start", items: [{ id: "overview", label: "About the system", icon: "home" }] },
    { group: "Analysis", items: [
      { id: "analyze", label: "New analysis", icon: "upload" },
      { id: "result", label: "Diagnosis result", icon: "target", needs: true },
      { id: "explanation", label: "AI explanation", icon: "chat", needs: true }] },
    { group: "Reasoning", items: [
      { id: "graph", label: "Knowledge graph", icon: "graph" },
      { id: "heuristic", label: "Heuristic reasoning", icon: "rules" }] },
    { group: "Technical", items: [
      { id: "intermediate", label: "Intermediate results", icon: "layers", needs: true },
      { id: "method", label: "Model & method", icon: "book" }] },
  ];
  function renderNav() {
    const cur = currentRoute();
    $("#sidebar").innerHTML =
      `<div class="status" id="status"><i></i><span>Starting up\u2026</span></div>` +
      NAV.map((g) => `<div class="nav-group">${g.group}</div>` + g.items.map((it) =>
        `<a class="nav-item ${cur === it.id ? "active" : ""} ${it.needs && !state.result ? "locked" : ""}" href="#/${it.id}">${icon(it.icon)}<span>${it.label}</span>${it.needs && state.result ? '<span class="dot" title="Results ready"></span>' : ""}</a>`).join("")).join("") +
      `<div class="side-foot"><a class="btn dark" style="width:100%" href="#/analyze" id="newBtn">${icon("plus")} Start new analysis</a>
       <div class="side-note" style="margin-top:12px"><b>Not a diagnosis.</b> Research prototype for education. Always consult a qualified clinician about skin changes.</div></div>`;
    updateStatus();
  }
  function updateStatus() {
    const s = $("#status"); if (!s) return;
    const h = state.health;
    s.className = "status " + (h && h.ready ? "ready" : h && h.status === "error" ? "error" : "");
    s.querySelector("span").textContent = h && h.ready ? "System ready" : h && h.status === "error" ? "Temporarily unavailable" : "Starting up\u2026";
  }
  const currentRoute = () => (location.hash.replace(/^#\//, "") || "overview").split("?")[0];

  const pages = {};
  function register(id, fn) { pages[id] = fn; }
  function route() {
    const id = currentRoute();
    const fn = pages[id] || pages.overview;
    renderNav();
    const v = $("#view");
    v.innerHTML = '<div class="page" id="page"></div>';
    fn($("#page"));
    window.scrollTo({ top: 0 });
    $("#sidebar").classList.remove("open"); $("#scrim").classList.remove("show");
  }
  function go(id) { location.hash = "#/" + id; }

  function needResult(el, what) {
    if (state.result) return false;
    el.innerHTML = `<div class="card empty">${icon("target")}<h3>No analysis yet</h3><p class="muted">Run an analysis first, then ${what} will appear here.</p><a class="btn primary" href="#/analyze">${icon("upload")} Start an analysis</a></div>`;
    return true;
  }

  function boot() {
    renderNav();
    $("#menuBtn").onclick = () => { $("#sidebar").classList.toggle("open"); $("#scrim").classList.toggle("show"); };
    $("#scrim").onclick = () => { $("#sidebar").classList.remove("open"); $("#scrim").classList.remove("show"); };
    window.addEventListener("hashchange", route);
    document.addEventListener("click", (e) => {
      const a = e.target.closest("a[data-cite]");
      if (a) { state.citeFocus = a.dataset.cite; }
    });
    route();
    const saved = sessionStorage.getItem("jobId");
    if (saved) getJSON("/api/jobs/" + saved).then((j) => {
      if (j.status === "done") { state.result = j.result; state.jobId = saved; route(); }
    }).catch(() => sessionStorage.removeItem("jobId"));
    loadStatic();
  }

  const legal = `<footer class="legal">Research prototype for education and research only. Not a medical device and not a diagnosis. Scores are uncalibrated estimates from a model trained on a small research dataset (Derm7pt).</footer>`;
  return { $, $$, esc, pct, f, rich, state, icon, api, getJSON, plot, heat, register, route, go, needResult, boot, legal, FONT };
})();
