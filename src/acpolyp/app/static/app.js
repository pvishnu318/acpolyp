"use strict";
const $ = (s, r = document) => r.querySelector(s);
const pct = (v, d = 1) => (v * 100).toFixed(d) + "%";
const fmt = (v, d = 3) => (v === null || v === undefined || Number.isNaN(v) ? "–" : Number(v).toFixed(d));
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const NICE = {
  "dyed-lifted-polyps": "Dyed lifted polyps", "dyed-resection-margins": "Dyed resection margins", esophagitis: "Esophagitis",
  "normal-cecum": "Normal cecum", "normal-pylorus": "Normal pylorus", "normal-z-line": "Normal z-line", polyps: "Polyps", "ulcerative-colitis": "Ulcerative colitis",
};

/* ---------- theme ---------- */
(() => {
  try { const t = localStorage.getItem("theme"); if (t) document.documentElement.dataset.theme = t; } catch (e) { /* storage blocked */ }
  $("#theme").onclick = () => {
    const cur = document.documentElement.dataset.theme || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
    const next = cur === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem("theme", next); } catch (e) { /* ignore */ }
  };
})();

/* ---------- demo ---------- */
let images = {}, view = "canonical";
const drop = $("#drop"), file = $("#file");
$("#pick").onclick = () => file.click();
file.onchange = () => file.files[0] && run(file.files[0]);
["dragenter", "dragover"].forEach((e) => drop.addEventListener(e, (ev) => { ev.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((e) => drop.addEventListener(e, (ev) => { ev.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", (ev) => ev.dataTransfer.files[0] && run(ev.dataTransfer.files[0]));
document.querySelectorAll(".tabs button").forEach((b) => (b.onclick = () => { view = b.dataset.view; setTabs(); showView(); }));
const setTabs = () => document.querySelectorAll(".tabs button").forEach((b) => b.classList.toggle("on", b.dataset.view === view));
const showView = () => { if (images[view]) $("#viewImg").src = images[view]; };

async function run(f) {
  $("#preview").src = URL.createObjectURL(f); $("#preview").hidden = false; $("#dropInner").hidden = true;
  $("#out").hidden = false; $("#spin").hidden = false;
  const fd = new FormData(); fd.append("file", f);
  try {
    const r = await fetch("/api/predict", { method: "POST", body: fd });
    if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
    render(await r.json());
  } catch (e) {
    $("#verdict").innerHTML = `<b style="color:var(--bad)">${esc(e.message)}</b>`;
  } finally { $("#spin").hidden = true; }
}

function render(d) {
  images = d.images; view = "canonical"; setTabs(); showView();
  $("#verdict").innerHTML = `<b>${esc(NICE[d.top] || d.top)}</b><span class="muted">${pct(d.confidence)} calibrated</span>` +
    `<span class="pill ${d.polyp_flag ? "pos" : ""}">polyp P = ${pct(d.polyp_prob)} ${d.polyp_flag ? "· above threshold" : "· below threshold"} (τ=${fmt(d.polyp_threshold, 2)})</span>`;
  const entries = Object.entries(d.probs).sort((a, b) => b[1] - a[1]);
  $("#bars").innerHTML = entries.map(([k, v], i) =>
    `<div class="bar ${i === 0 ? "top" : ""}"><span>${esc(NICE[k] || k)}</span><div class="track"><i style="width:${(v * 100).toFixed(1)}%"></i></div><span>${pct(v)}</span></div>`).join("");
  $("#conformal").innerHTML = `<span class="muted">${Math.round((1 - d.conformal_alpha) * 100)}% conformal set:</span>` +
    d.conformal_set.map((c) => `<span class="chip">${esc(NICE[c] || c)}</span>`).join("") +
    (d.conformal_set.length > 1 ? `<span class="muted">· ambiguous, defer to a clinician</span>` : "");
  $("#meta").textContent = `Inference ${d.latency_ms} ms · contour covers ${pct(d.ac_area_frac, 0)} of the field of view · model mode: ${d.mode}`;
}

(async () => {
  try {
    const s = await (await fetch("/api/samples")).json();
    if (!s.available) return;
    const box = $("#samples"); box.hidden = false;
    box.innerHTML = s.samples.slice(0, 10).map((x) => `<img src="/api/sample/${x.name}" title="${esc(NICE[x.class])}" data-n="${x.name}" loading="lazy">`).join("");
    box.onclick = async (e) => {
      const n = e.target.dataset && e.target.dataset.n; if (!n) return;
      const blob = await (await fetch(`/api/sample/${n}`)).blob(); run(new File([blob], n.split("/")[1], { type: "image/jpeg" }));
    };
  } catch (e) { /* no local samples */ }
})();

/* ---------- SVG helpers ---------- */
const COL = ["var(--c1)", "var(--c2)", "var(--c3)", "var(--c4)", "var(--muted)"];
function axes(W, H, m, xt, yt, xl, yl) {
  let s = `<line class="axis" x1="${m.l}" y1="${H - m.b}" x2="${W - m.r}" y2="${H - m.b}"/><line class="axis" x1="${m.l}" y1="${m.t}" x2="${m.l}" y2="${H - m.b}"/>`;
  xt.forEach(([v, p]) => (s += `<text x="${p}" y="${H - m.b + 15}" text-anchor="middle">${v}</text>`));
  yt.forEach(([v, p]) => (s += `<text x="${m.l - 6}" y="${p + 4}" text-anchor="end">${v}</text><line class="axis" x1="${m.l}" y1="${p}" x2="${W - m.r}" y2="${p}" opacity=".35"/>`));
  s += `<text x="${(m.l + W - m.r) / 2}" y="${H - 4}" text-anchor="middle">${xl}</text>`;
  s += `<text transform="translate(12 ${(m.t + H - m.b) / 2}) rotate(-90)" text-anchor="middle">${yl}</text>`;
  return s;
}
function lineChart(title, curves, xl, yl, opts = {}) {
  const W = 460, H = 330, m = { l: 48, r: 14, t: 28, b: 40 };
  const X = (v) => m.l + v * (W - m.l - m.r), Y = (v) => H - m.b - ((v - (opts.ymin || 0)) / (1 - (opts.ymin || 0))) * (H - m.t - m.b);
  const ticks = [0, 0.25, 0.5, 0.75, 1];
  let s = `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(title)}"><text class="title" x="${m.l}" y="16">${esc(title)}</text>`;
  s += axes(W, H, m, ticks.map((t) => [t, X(t)]), ticks.map((t) => [t, Y((opts.ymin || 0) + t * (1 - (opts.ymin || 0)))]).map(([t, y]) => [((opts.ymin || 0) + t * (1 - (opts.ymin || 0))).toFixed(2), y]), xl, yl);
  if (opts.diag) s += `<line x1="${X(0)}" y1="${Y(0)}" x2="${X(1)}" y2="${Y(1)}" stroke="var(--line)" stroke-dasharray="4 4"/>`;
  curves.forEach((c, i) => {
    const pts = c.x.map((x, k) => [X(x), Y(c.y[k])]).filter((p, k) => c.y[k] !== null && c.x[k] !== null && !Number.isNaN(p[1]));
    s += `<polyline fill="none" stroke="${COL[i % 5]}" stroke-width="2" points="${pts.map((p) => p.join(",")).join(" ")}"/>`;
  });
  curves.forEach((c, i) => (s += `<text x="${W - m.r - 4}" y="${m.t + 14 + i * 15}" text-anchor="end" style="fill:${COL[i % 5]}">${esc(c.name)}</text>`));
  return s + "</svg>";
}
function barChart(title, rows, xl) {
  const W = 920, rowH = 30, m = { l: 230, r: 80, t: 30, b: 34 }, H = m.t + m.b + rows.length * rowH;
  const lo = Math.min(...rows.map((r) => r.lo !== undefined ? r.lo : r.v - (r.sd || 0))) , hi = Math.max(...rows.map((r) => r.hi !== undefined ? r.hi : r.v + (r.sd || 0)));
  const a = lo - (hi - lo) * 0.15, b = hi + (hi - lo) * 0.15, X = (v) => m.l + ((v - a) / (b - a)) * (W - m.l - m.r);
  let s = `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(title)}"><text class="title" x="8" y="16">${esc(title)}</text>`;
  for (let i = 0; i <= 4; i++) { const v = a + ((b - a) * i) / 4; s += `<line class="axis" x1="${X(v)}" x2="${X(v)}" y1="${m.t}" y2="${H - m.b}" opacity=".35"/><text x="${X(v)}" y="${H - m.b + 15}" text-anchor="middle">${v.toFixed(3)}</text>`; }
  rows.forEach((r, i) => {
    const y = m.t + i * rowH + rowH / 2, l = r.lo !== undefined ? r.lo : r.v - (r.sd || 0), h = r.hi !== undefined ? r.hi : r.v + (r.sd || 0);
    s += `<text x="${m.l - 8}" y="${y + 4}" text-anchor="end" style="fill:var(--ink)">${esc(r.label)}</text><circle cx="${X(r.v)}" cy="${y}" r="5" fill="${r.color || "var(--c1)"}"/>`;
    s += `<line x1="${X(l)}" x2="${X(h)}" y1="${y}" y2="${y}" stroke="${r.color || "var(--c1)"}" stroke-width="2"/><text x="${X(h) + 6}" y="${y + 4}">${r.v.toFixed(3)}</text>`;
  });
  return s + `<text x="${(m.l + W - m.r) / 2}" y="${H - 2}" text-anchor="middle">${xl}</text></svg>`;
}
function heatmap(title, labels, M) {
  const n = labels.length, cell = 46, m = { l: 150, t: 110 }, W = m.l + n * cell + 10, H = m.t + n * cell + 24;
  const rows = M.map((r) => { const t = r.reduce((a, b) => a + b, 0) || 1; return r.map((v) => v / t); });
  let s = `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(title)}"><text class="title" x="8" y="16">${esc(title)}</text>`;
  rows.forEach((r, i) => {
    s += `<text x="${m.l - 6}" y="${m.t + i * cell + cell / 2 + 4}" text-anchor="end">${esc(NICE[labels[i]] || labels[i])}</text>`;
    r.forEach((v, j) => {
      s += `<rect x="${m.l + j * cell}" y="${m.t + i * cell}" width="${cell - 2}" height="${cell - 2}" rx="4" fill="var(--c1)" opacity="${(0.08 + 0.92 * v).toFixed(2)}"/>`;
      s += `<text x="${m.l + j * cell + cell / 2 - 1}" y="${m.t + i * cell + cell / 2 + 4}" text-anchor="middle" style="fill:var(--ink);font-size:10px">${M[i][j]}</text>`;
    });
  });
  labels.forEach((l, j) => (s += `<text transform="translate(${m.l + j * cell + cell / 2} ${m.t - 6}) rotate(-45)" text-anchor="start">${esc(NICE[l] || l)}</text>`));
  return s + `<text x="${m.l}" y="${H - 4}">true class (rows) × predicted class (columns); cell = count, shade = row fraction</text></svg>`;
}
function table(cols, rows, bestKeys = {}) {
  const best = {};
  Object.entries(bestKeys).forEach(([k, dir]) => { const v = rows.map((r) => r[k]?.[0] ?? r[k]).filter((x) => typeof x === "number"); best[k] = dir > 0 ? Math.max(...v) : Math.min(...v); });
  const cell = (r, c) => {
    const v = r[c.key]; let txt;
    if (Array.isArray(v)) txt = `${fmt(v[0], c.d ?? 3)} ± ${fmt(v[1], c.d ?? 3)}`; else if (typeof v === "number") txt = fmt(v, c.d ?? 3); else txt = esc(v ?? "–");
    const val = Array.isArray(v) ? v[0] : v;
    return `<td class="${c.key in best && val === best[c.key] ? "best" : ""}">${txt}</td>`;
  };
  return `<div class="scroll"><table><thead><tr>${cols.map((c) => `<th>${c.label}</th>`).join("")}</tr></thead><tbody>${rows.map((r) => `<tr>${cols.map((c) => cell(r, c)).join("")}</tr>`).join("")}</tbody></table></div>`;
}

/* ---------- results ---------- */
(async () => {
  let R;
  try { const r = await fetch("/api/results"); if (!r.ok) throw 0; R = await r.json(); } catch (e) {
    $("#resultsBody").innerHTML = `<p class="muted">Results are not exported yet. Run <code>python scripts/export_results.py</code>.</p>`; return;
  }
  $("#heroStats").innerHTML = (R.hero || []).map((h) => `<div class="stat"><b>${esc(h.value)}</b><span>${esc(h.label)}</span></div>`).join("");
  const g = (R.ablation || []).find((a) => a.label.startsWith("ACG-ConvNeXt"));
  if (g && g.ci) {
    const f = $("#finding"); f.hidden = false;
    f.innerHTML = `<b>Headline finding.</b> Contour guidance changed macro-F1 by ${g.delta >= 0 ? "+" : "−"}${Math.abs(g.delta).toFixed(4)} (95% CI ${g.ci[0].toFixed(4)} to ${g.ci[1].toFixed(4)}; Holm p = ${g.p_adj.toFixed(2)}): <b>not a statistically supported improvement</b>, and controls using the wrong image's contour or a central ellipse perform the same. What carries over is the audit, calibration and error analysis below.`;
  }
  let h = "";
  if (R.leaderboard) h += `<div class="card"><h3>Test-set leaderboard</h3>${table([
    { key: "name", label: "Model" }, { key: "macro_f1", label: "Macro-F1" }, { key: "acc", label: "Accuracy" }, { key: "mcc", label: "MCC" },
    { key: "polyp_auroc", label: "Polyp AUROC" }, { key: "ece", label: "ECE ↓" }, { key: "params_m", label: "Params (M)", d: 1 }, { key: "latency_ms", label: "ms/img (bs1)", d: 1 }],
    R.leaderboard, { macro_f1: 1, acc: 1, mcc: 1, polyp_auroc: 1, ece: -1 })}<p class="note">Best value per column in teal. Deep models: mean ± SD over seeds; classical baselines in the lower rows. ECE after temperature scaling.</p></div>`;
  if (R.ablation) h += `<div class="card">${barChart("Component ablation (ConvNeXt-T) — macro-F1, mean over seeds with 95% paired-bootstrap interval of the difference to the plain baseline shown in the table below", R.ablation.map((a, i) => ({ label: a.label, v: a.macro_f1[0], sd: a.macro_f1[1], color: COL[i % 4] })), "macro-F1 (error bar = SD over seeds)")}` +
    table([{ key: "label", label: "Variant" }, { key: "macro_f1", label: "Macro-F1" }, { key: "delta", label: "Δ vs plain", d: 4 }, { key: "ci", label: "95% CI of Δ" }, { key: "p_adj", label: "p (Holm)" }],
      R.ablation.map((a) => ({ ...a, ci: a.ci ? `[${fmt(a.ci[0], 4)}, ${fmt(a.ci[1], 4)}]` : "–" }))) + `<p class="note">${esc(R.ablation_note || "")}</p></div>`;
  h += `<div class="two">`;
  if (R.confusion) h += `<div class="card">${heatmap("Confusion matrix — " + R.confusion.model, R.confusion.labels, R.confusion.matrix)}</div>`;
  if (R.roc) h += `<div class="card">${lineChart("Polyp vs rest — ROC", R.roc.curves.map((c) => ({ name: `${c.name} (AUC ${fmt(c.auc, 4)})`, x: c.fpr, y: c.tpr })), "false-positive rate", "sensitivity", { diag: true })}</div>`;
  h += `</div><div class="two">`;
  if (R.reliability) h += `<div class="card">${lineChart("Reliability diagram", [{ name: `raw ECE ${fmt(R.reliability.ece_raw, 3)}`, x: R.reliability.raw.conf, y: R.reliability.raw.acc }, { name: `calibrated ECE ${fmt(R.reliability.ece_cal, 3)}`, x: R.reliability.cal.conf, y: R.reliability.cal.acc }], "confidence", "accuracy", { diag: true })}</div>`;
  if (R.risk_coverage) h += `<div class="card">${lineChart("Selective classification", R.risk_coverage.curves.map((c) => ({ name: c.name, x: c.coverage, y: c.risk.map((r) => Math.min(1, r * R.risk_coverage.scale)) })), "coverage (fraction answered)", `error ×${R.risk_coverage.scale}`)}</div>`;
  h += `</div>`;
  if (R.ac) h += `<div class="card"><h3>How good are the contours?</h3><div class="kv">${R.ac.map((k) => `<div><b>${esc(k.value)}</b><span class="muted">${esc(k.label)}</span></div>`).join("")}</div><p class="note">${esc(R.ac_note || "")}</p></div>`;
  $("#resultsBody").innerHTML = h;
  if (R.audit) $("#auditBody").innerHTML = `<div class="kv">${R.audit.items.map((k) => `<div><b>${esc(k.value)}</b><span class="muted">${esc(k.title)}</span><p class="note">${esc(k.detail || "")}</p></div>`).join("")}</div>` +
    (R.audit.shortcut ? `<div class="card">${barChart("How well can the label be guessed from artefacts alone? (macro-F1)", R.audit.shortcut.map((s, i) => ({ label: s.name, v: s.f1, lo: s.lo ?? s.f1, hi: s.hi ?? s.f1, color: COL[i % 4] })), "macro-F1 on held-out test split")}<p class="note">${esc(R.audit.shortcut_note || "")}</p></div>` : "");
})();
