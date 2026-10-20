/* CareWatch front end — no build step. Talks to the FastAPI backend in backend/server.py */
"use strict";

const $ = (s, el = document) => el.querySelector(s);
const ls = (k, d) => { try { return localStorage.getItem(k) ?? d; } catch { return d; } };
const lsSet = (k, v) => { try { localStorage.setItem(k, v); } catch { /* private mode */ } };
const esc = (t) => String(t ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

const S = {
  page: "monitor",                      // "monitor" | "fl" (federated-learning demo)
  ready: false, loadError: "",
  mode: ls("mode", "icu"),             // "icu" | "home"
  pace: +ls("pace", 10),               // seconds between readings
  cam: false,                          // live webcam (always starts off — never auto-activated)
  view: "picker", tab: "live",
  patients: [], patient: null,         // chosen patient {id,title,summary}
  res: null, history: [], report: null, reportFor: null, model: null,
  paused: false, busy: false, nextAt: 0, camStatus: null, noteDraft: "", animate: true,
};

const LABELS = {
  HR: ["Heart rate", "bpm"], RR: ["Respiratory rate", "br/min"], SpO2: ["Oxygen (SpO₂)", "%"],
  Temp: ["Temperature", "°C"], SBP: ["Systolic BP", "mmHg"], DBP: ["Diastolic BP", "mmHg"],
  MAP: ["Mean arterial pressure", "mmHg"],
};
const GCS_TEXT = { 4: "Spontaneous", 3: "To voice", 2: "To pain", 1: "No response" };
const TIER_TXT = { stable: "Low risk", watch: "Moderate risk", alert: "High-risk alert" };

/* ── api ─────────────────────────────────────────────────────────────────── */
async function api(path, body) {
  const r = await fetch(path, body === undefined ? {} : {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  });
  if (!r.ok) {
    let msg = r.statusText;
    try { msg = (await r.json()).detail || msg; } catch { /* keep statusText */ }
    throw new Error(msg);
  }
  return r.json();
}
function toast(msg) {
  const t = $("#toast"); t.textContent = msg; t.hidden = false;
  clearTimeout(toast._t); toast._t = setTimeout(() => (t.hidden = true), 4000);
}

/* ── svg charts ──────────────────────────────────────────────────────────── */
function lineChart(vals, o = {}) {
  const w = o.w || 320, h = o.h || 110, px = o.labels ? 30 : 4, py = 8;
  if (!vals || !vals.length) return "";
  let lo = Math.min(...vals), hi = Math.max(...vals);
  if (o.lo !== undefined) { lo = Math.min(lo, o.lo); hi = Math.max(hi, o.hi); }
  if (o.min !== undefined) lo = Math.min(lo, o.min);
  if (o.max !== undefined) hi = Math.max(hi, o.max);
  if (hi === lo) { hi += 1; lo -= 1; }
  const pad = (hi - lo) * 0.08; lo -= pad; hi += pad;
  const X = (i) => px + (vals.length === 1 ? (w - px - 4) / 2 : (i * (w - px - 4)) / (vals.length - 1));
  const Y = (v) => py + (1 - (v - lo) / (hi - lo)) * (h - 2 * py);
  let g = "";
  (o.bands || (o.lo !== undefined ? [{ from: o.lo, to: o.hi, c: "#E5F3EA" }] : [])).forEach((b) => {
    const y1 = Y(Math.min(b.to, hi)), y2 = Y(Math.max(b.from, lo));
    g += `<rect x="${px}" y="${y1}" width="${w - px - 4}" height="${Math.max(0, y2 - y1)}" fill="${b.c}"/>`;
  });
  (o.lines || []).forEach((l) => {
    if (l.v > lo && l.v < hi) g += `<line x1="${px}" x2="${w - 4}" y1="${Y(l.v)}" y2="${Y(l.v)}" stroke="#C9C3B3" stroke-dasharray="3 3"/><text x="${px - 4}" y="${Y(l.v) + 3}" font-size="9" fill="#A9ADA3" text-anchor="end">${l.v}</text>`;
  });
  const pts = vals.map((v, i) => `${X(i).toFixed(1)},${Y(v).toFixed(1)}`).join(" ");
  const c = o.color || "#1F6B58";
  g += `<polyline points="${pts}" fill="none" stroke="${c}" stroke-width="${o.sw || 2}" stroke-linejoin="round" stroke-linecap="round"/>`;
  g += `<circle cx="${X(vals.length - 1)}" cy="${Y(vals[vals.length - 1])}" r="${o.sw ? 3 : 3.5}" fill="${c}"/>`;
  if (o.labels) g += `<text x="${px - 4}" y="${py + 8}" font-size="9" fill="#A9ADA3" text-anchor="end">${(hi - pad).toFixed(o.dec ?? 0)}</text><text x="${px - 4}" y="${h - py}" font-size="9" fill="#A9ADA3" text-anchor="end">${(lo + pad).toFixed(o.dec ?? 0)}</text>`;
  return `<svg viewBox="0 0 ${w} ${h}" ${o.labels ? "" : 'preserveAspectRatio="none"'} role="img">${g}</svg>`;
}

/* ── top bar + controls ──────────────────────────────────────────────────── */
function renderTop() {
  $("#top").innerHTML = `<div class="top-in">
    <div class="brand"><i></i>CareWatch</div>
    <div class="seg" role="tablist" aria-label="Section">
      <button data-page="monitor" class="${S.page === "monitor" ? "on" : ""}">Monitor</button>
      <button data-page="fl" class="${S.page === "fl" ? "on" : ""}">Federated learning</button>
    </div>
    ${S.page === "monitor" ? `<div class="seg" role="tablist" aria-label="Care setting">
      <button data-mode="icu" class="${S.mode === "icu" ? "on" : ""}">ICU</button>
      <button data-mode="home" class="${S.mode === "home" ? "on" : ""}">Home care</button>
    </div>` : `<span class="small mute">Independent of the patient monitor</span>`}</div>`;
}

function renderCtrl() {
  const el = $("#ctrl");
  if (S.page === "fl" || S.view !== "monitor" || !S.patient) { el.innerHTML = ""; return; }
  el.innerHTML = `<div class="ctrl">
    <button class="back" id="back">← All patients</button>
    <div class="ctrl-row">
      <div class="who"><h1>Patient ${S.patient.id}</h1><span class="mute">${esc(S.patient.title)}</span></div>
      <div class="tools">
        <span class="pill ${S.paused ? "paused" : ""}" id="pill"><i class="dot"></i>${S.paused ? "Paused" : "Live"}</span>
        <div class="seg" aria-label="Reading pace">
          <button data-pace="10" class="${S.pace === 10 ? "on" : ""}" title="A new reading every 10 seconds">Demo 10s</button>
          <button data-pace="30" class="${S.pace === 30 ? "on" : ""}" title="A new reading every 30 seconds">Standard 30s</button>
        </div>
        <label class="switch"><input type="checkbox" id="cam" ${S.cam ? "checked" : ""}><span></span>Live camera</label>
        <span class="camstat" id="camstat"></span>
        <button class="btn" id="pause">${S.paused ? "Resume" : "Pause"}</button>
      </div>
    </div>
    <div class="timer"><div id="timerbar"></div></div>
  </div>`;
  updateCamUI();
}

/* ── picker ──────────────────────────────────────────────────────────────── */
function renderPicker() {
  const home = S.mode === "home";
  const cards = S.patients.map((p) => `
    <button class="pcard" data-pid="${p.id}">
      <div class="n">${p.id}</div>
      <h3>${esc(p.title)}</h3>
      <div class="mute small">${esc(p.summary)}</div>
      <div class="tiny mute" style="margin-top:10px">${p.readings} readings · one every 10 min (replayed)</div>
      <div class="go">Start monitoring →</div>
    </button>`).join("");
  return `<section class="hero ${S.animate ? "fade" : ""}">
      <h1>${home ? "Remote monitoring for patients at home" : "Continuous monitoring for ICU patients"}</h1>
      <p>${home
        ? "Wearable vitals, a home camera and caregiver notes are fused into one risk score, with a clear next step: stay, watch, or escalate to hospital."
        : "Vitals, a bedside camera and nursing notes are fused into one organ-failure (SOFA) score that updates with every reading."}</p>
    </section>
    <div class="patients">${cards}</div>`;
}

/* ── monitor: tabs ───────────────────────────────────────────────────────── */
const TABS = [["live", "Live"], ["trends", "Trends"], ["why", "Why this score"], ["report", "AI report"], ["model", "Model"]];

function renderMonitor() {
  const r = S.res;
  if (!r) return `<div class="center"><div class="spin"></div><div class="mute">Taking the first reading…</div></div>`;
  const tabs = `<nav class="tabs" role="tablist">${TABS.map(([k, t]) => `<button data-tab="${k}" class="${S.tab === k ? "on" : ""}">${t}</button>`).join("")}</nav>`;
  const body = { live: tabLive, trends: tabTrends, why: tabWhy, report: tabReport, model: tabModel }[S.tab](r);
  const anim = S.animate; S.animate = false;
  return tabs + `<div class="${anim ? "fade" : ""}" id="tabbody">${body}</div>`;
}

function tabLive(r) {
  const tier = r.tier;
  const interval = `${r.interval[0]} – ${r.interval[1]}`;
  const sofa = `<div class="card sofa">
      <div class="eyebrow">Predicted SOFA score</div>
      <div class="sofa-top">
        <div class="sofa-n num">${r.sofa.toFixed(1)}<small>/ 24</small></div>
        <span class="chip bg-${tier} t-${tier}">${TIER_TXT[tier]}</span>
      </div>
      <div><div class="scale"><b style="left:${Math.min(100, (r.sofa / 24) * 100)}%"></b></div>
        <div class="scale-l"><span style="left:0">0</span><span style="left:20.8%">5</span><span style="left:33.3%">8 · alert</span><span style="right:0">24</span></div></div>
      <div class="meta"><span>90% interval ${interval}</span><span>Window ${r.window_n}/20</span><span>Checks ${r.validation_passed}/${r.validation.length} passed</span></div>
    </div>`;
  const e = r.escalations[S.mode] || r.escalation;
  const action = `<div class="card action ${tier}">
      <div class="eyebrow">${S.mode === "home" ? "Home-care action" : "Recommended attention"}</div>
      <h2>${esc(e.title)}</h2>
      <ul>${e.actions.map((a) => `<li>${esc(a)}</li>`).join("")}</ul>
      <div class="why">${r.alert_note ? esc(r.alert_note) + ". " : ""}${S.mode === "home" ? "Demo plan — real thresholds and actions must be set by the supervising clinician." : "Decision support only."}</div>
    </div>`;

  const tiles = r.vitals.map((v) => {
    const [name, unit] = LABELS[v.key];
    const fmt = v.key === "Temp" || v.key === "SpO2" ? v.value.toFixed(1) : Math.round(v.value);
    const st = v.status === "ok" ? "Normal" : v.status === "high" ? "High" : "Low";
    return `<div class="tile ${v.status}"><div class="lab"><span>${name}</span><span class="st">${st}</span></div>
      <div class="val num">${fmt}<small>${unit}</small></div>
      ${lineChart(v.series, { w: 160, h: 34, lo: v.lo, hi: v.hi, sw: 1.6 })}</div>`;
  }).join("");
  const gcsOk = r.gcs.value === 4, stressOk = r.stress.value <= 4;
  const cvTiles = `
    <div class="tile ${gcsOk ? "" : "low"}"><div class="lab"><span>Eye opening (GCS)</span><span class="st">${gcsOk ? "Normal" : "Reduced"}</span></div>
      <div class="val num">${r.gcs.value}<small>/ 4 · ${GCS_TEXT[r.gcs.value] || ""}</small></div>
      ${lineChart(r.gcs.series, { w: 160, h: 34, lo: 4, hi: 4.01, min: 1, max: 4, sw: 1.6 })}</div>
    <div class="tile ${stressOk ? "" : "high"}"><div class="lab"><span>Stress</span><span class="st">${stressOk ? "Normal" : "Elevated"}</span></div>
      <div class="val num">${r.stress.value}<small>/ 10</small></div>
      ${lineChart(r.stress.series, { w: 160, h: 34, lo: 0, hi: 4, min: 0, max: 10, sw: 1.6 })}</div>`;

  const home = S.mode === "home";
  const camSrc = r.cv_source === "live" ? `<span class="src live">● Live webcam</span>`
    : r.cv_source === "no_feed" ? `<span class="src warn">No camera feed — using replay</span>`
    : `<span class="src replay">Replayed recording</span>`;
  const noteHtml = esc(r.replay_note).replace(/\[(\d\d:\d\d)\]/g, '<span class="stamp">$1</span>')
    + (r.live_note ? ` <span class="livetag">Live note</span> ${esc(r.live_note)}` : "");
  const inputs = `<div class="grid g3">
      <div class="card"><div class="eyebrow">${home ? "Wearable sensors" : "Bedside monitor"}</div>
        <h3 style="margin-top:6px">Vitals</h3><div class="tiny mute">HR · RR · SpO₂ · Temp · BP</div>
        <div class="src replay" style="margin-top:10px">Replayed recording</div></div>
      <div class="card"><div class="eyebrow">${home ? "Home camera" : "Bedside camera"}</div>
        <h3 style="margin-top:6px">Stress &amp; eye opening</h3><div class="tiny mute">Read from the face — no contact</div>
        <div style="margin-top:10px" id="camsrc">${camSrc}</div>
        <div class="tiny mute" id="camchip" style="margin-top:4px">${esc(camChipText())}</div></div>
      <div class="card"><div class="eyebrow">${home ? "Caregiver notes" : "Nursing notes"}</div>
        <h3 style="margin-top:6px">Clinical note</h3><div class="tiny mute">Read by the model every reading</div>
        <div class="src ${r.live_note ? "live" : "replay"}" style="margin-top:10px">${r.live_note ? "● Live note active" : "Replayed note"}</div></div>
    </div>`;
  const note = `<div class="card"><div class="eyebrow">Note used for reading ${r.reading_no} of ${r.total_rows}</div>
      <div class="notebox" style="margin-top:10px">${noteHtml}</div>
      <form class="noteform" id="noteform"><input id="notein" maxlength="600" autocomplete="off" placeholder="${home ? "Type a caregiver observation, e.g. “feels dizzy and confused”" : "Type a nursing observation, e.g. “patient now unresponsive”"}" value="${esc(S.noteDraft)}">
        <button class="btn primary" type="submit">Add note</button>${r.live_note ? `<button class="btn" type="button" id="clearnote">Clear</button>` : ""}</form>
      <div class="tiny mute" style="margin-top:8px">A typed note is added to every following reading, so you can watch the score react.</div></div>`;

  return `<div class="stack">
    <div class="grid g2">${sofa}${action}</div>
    <div class="grid g-auto">${tiles}${cvTiles}</div>
    ${inputs}${note}</div>`;
}

function camState() {
  const c = S.camStatus;
  if (!S.cam) return { cls: "off", txt: "", chip: "Camera off — turn on “Live camera” above" };
  if (c && c.live) return { cls: "live", txt: `● Camera live · stress ${c.stress.toFixed(1)} · eye ${c.gcs}/4`, chip: `Camera on · stress ${c.stress.toFixed(1)} · eye ${c.gcs}/4` };
  if (c && c.no_face) return { cls: "wait", txt: "Camera on — no face detected", chip: "Camera is on but no face is in view — look at the camera" };
  if (c && c.waited > 20) return { cls: "wait", txt: "Camera not responding yet…", chip: "The camera program is running but sent no frames — check the camera window" };
  return { cls: "wait", txt: "Starting camera…", chip: "Starting camera — a window opens; keep your face in view for ~3 s" };
}
function camChipText() { return camState().chip; }
function updateCamUI() {
  const st = camState();
  const el = $("#camstat"); if (el) { el.textContent = st.txt; el.className = "camstat " + st.cls; }
  const chip = $("#camchip"); if (chip) chip.textContent = st.chip;
}

function tabTrends(r) {
  const sofaChart = lineChart(r.sofa_series, {
    w: 760, h: 170, labels: true, dec: 0, min: 0, max: 10, sw: 2.4,
    bands: [{ from: 0, to: 5, c: "#E5F3EA" }, { from: 5, to: 8, c: "#FBF0D6" }, { from: 8, to: 40, c: "#FBE5E0" }],
    lines: [{ v: 5 }, { v: 8 }],
    color: r.tier === "alert" ? "#C23B28" : r.tier === "watch" ? "#B97A06" : "#2E8B5A",
  });
  const charts = r.vitals.map((v) => {
    const [name, unit] = LABELS[v.key];
    const cur = v.key === "Temp" || v.key === "SpO2" ? v.value.toFixed(1) : Math.round(v.value);
    return `<div class="card chart-card"><h3>${name}<span class="now num">${cur} ${unit}</span></h3>
      ${lineChart(v.series, { w: 320, h: 110, lo: v.lo, hi: v.hi, labels: true, dec: v.key === "Temp" ? 1 : 0 })}
      <div class="tiny mute">Green band = normal range</div></div>`;
  }).join("");
  const rows = S.history.slice(0, 12).map((h) => `<tr><td>${esc(h.Reading)}</td><td>${esc(String(h.Timestamp).slice(11))}</td>
      <td class="num"><b>${Number(h.SOFA).toFixed(1)}</b></td><td>${esc(h.Risk)}</td><td class="num">${h.HR}</td><td class="num">${h.RR}</td>
      <td class="num">${h["SpO₂"]}</td><td class="num">${h.SBP}</td><td class="num">${h["GCS Eye"]}</td><td class="num">${h.Stress}</td></tr>`).join("");
  return `<div class="stack">
    <div class="card"><h3>SOFA score over this session</h3><div class="tiny mute">Bands: low &lt; 5 · moderate 5–8 · alert ≥ 8</div>${sofaChart}</div>
    <div class="grid g-auto" style="grid-template-columns:repeat(auto-fill,minmax(300px,1fr))">${charts}</div>
    <div class="card"><h3>Reading history</h3><div class="tbl-wrap"><table>
      <thead><tr><th>Reading</th><th>Time</th><th>SOFA</th><th>Risk</th><th>HR</th><th>RR</th><th>SpO₂</th><th>SBP</th><th>GCS</th><th>Stress</th></tr></thead>
      <tbody>${rows}</tbody></table></div></div></div>`;
}

function tabWhy(r) {
  const ex = r.explain;
  const max = Math.max(...ex.drivers.map((d) => d.impact), 0.01);
  const drivers = ex.available
    ? (ex.drivers.length ? ex.drivers.map((d) => `<div class="drv">
        <div><b>${esc(d.kind === "note" ? "“" + d.label + "”" : d.label)}</b><div class="v">${esc(d.value)}</div></div>
        <div class="bar"><i style="width:${(d.impact / max) * 100}%"></i></div><div class="n num">+${d.impact.toFixed(2)}</div></div>`).join("")
        : `<div class="mute">No single factor is pushing the score up right now.</div>`)
    : `<div class="mute">SHAP explanations are not available on this machine.</div>`;
  const checks = r.validation.map((c) => `<div class="check"><div class="ic ${c.ok ? "ok" : "no"}">${c.ok ? "✓" : "!"}</div>
      <div><b>${esc(c.title)}</b><span class="small mute">${esc(c.detail)}</span></div></div>`).join("");
  return `<div class="stack">
    <div class="card"><h3>What is pushing the score up</h3>
      <div class="small mute" style="margin-bottom:6px">Longer bar = stronger influence on this prediction${ex.lowered ? ` · ${ex.lowered} other top factor(s) lowered it and are not listed` : ""}.</div>${drivers}</div>
    <div class="grid" style="grid-template-columns:repeat(auto-fit,minmax(300px,1fr))">
      <div class="card"><h3>In plain words</h3>${ex.interpretations.length ? `<ul class="plain">${ex.interpretations.map((t) => `<li>${esc(t)}</li>`).join("")}</ul>` : `<div class="mute small" style="margin-top:6px">Nothing notable.</div>`}
        ${ex.key_risks.length ? `<h3 style="margin-top:18px">Key risk factors</h3><div class="tags" style="margin-top:8px">${ex.key_risks.map((k) => `<span class="tag">${esc(k)}</span>`).join("")}</div>` : ""}</div>
      <div class="card"><h3>Is the prediction trustworthy?</h3><div class="small mute">Four runtime checks on every reading</div>${checks}</div>
    </div></div>`;
}

function tabReport() {
  const rep = S.report, forR = S.reportFor;
  const hdr = (extra = "") => `<div class="eyebrow">AI clinical assessment${forR ? ` · reading ${esc(forR.split("-")[1])}` : ""}</div>${extra}`;
  if (!rep || rep.status === "pending") {
    return `<div class="card">${hdr()}<div style="margin-top:14px"><div class="skel" style="width:40%"></div><div class="skel"></div><div class="skel" style="width:90%"></div><div class="skel" style="width:70%"></div></div>
      <div class="small mute">Writing the assessment and checking it against the vitals…</div></div>`;
  }
  if (rep.status !== "ok") {
    return `<div class="notice">AI clinical report unavailable — ${esc(rep.reason || "unknown reason")}. The SOFA prediction, validation checks and explanations are computed locally and are not affected.</div>`;
  }
  const rel = rep.reliability, col = rel >= 0.8 ? "#2E8B5A" : rel >= 0.6 ? "#B97A06" : "#C23B28";
  const lbl = rel >= 0.8 ? "High reliability" : rel >= 0.6 ? "Moderate reliability" : "Low reliability";
  const C = 2 * Math.PI * 36;
  const ring = `<div class="ring"><svg width="84" height="84"><circle cx="42" cy="42" r="36" fill="none" stroke="#EFEBE0" stroke-width="8"/>
      <circle cx="42" cy="42" r="36" fill="none" stroke="${col}" stroke-width="8" stroke-linecap="round" stroke-dasharray="${(rel * C).toFixed(1)} ${C.toFixed(1)}"/></svg><div class="num">${Math.round(rel * 100)}</div></div>`;
  const brk = Object.entries(rep.breakdown).map(([k, v]) => `<div class="brk"><span>${esc(k)}</span><div class="bar"><i style="width:${v * 100}%;background:${v >= 0.8 ? "#2E8B5A" : v >= 0.5 ? "#E0A528" : "#C23B28"}"></i></div><span class="num">${v.toFixed(2)}</span></div>`).join("");
  const secs = Object.keys(rep.sections || {}).length
    ? Object.entries(rep.sections).map(([n, c]) => `<div class="card sec"><h3>${esc(n)}</h3>${renderSection(c)}</div>`).join("")
    : `<div class="card sec">${renderSection(rep.text)}</div>`;
  const viol = rep.violations.length ? `<div class="notice" style="margin-top:12px"><b>Flagged by the safety checks</b><ul style="margin:6px 0 0;padding-left:18px">${rep.violations.map((v) => `<li>${esc(v)}</li>`).join("")}</ul></div>` : "";
  return `<div class="stack">
    <div class="card"><div class="rel">${ring}<div><div class="eyebrow">Report reliability</div><h2 style="font-size:22px;color:${col}">${lbl}</h2>
      <div class="small mute">Nine checks: facts vs vitals, clinical rules, SHAP agreement, structure, severity, safety, numbers, and an AI judge.${rep.judge_ok ? "" : " (AI judge unavailable — neutral score used.)"}</div></div></div>${viol}</div>
    ${secs}
    <div class="card"><h3>How the reliability score is built</h3><div style="margin-top:8px">${brk}</div></div></div>`;
}

function renderSection(content) {
  const lines = String(content).split("\n").map((l) => l.trim()).filter(Boolean);
  const inline = (t) => esc(t).replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");
  return lines.map((line) => {
    const bullet = /^[•·-]/.test(line);
    if (!bullet && line.endsWith(":") && line.length < 55 && /^[A-Z][A-Za-z ,&/]+:$/.test(line)) return `<div class="dom">${esc(line.slice(0, -1))}</div>`;
    if (line.includes("⚡")) return `<div class="prio">${inline(line.replace(/^[•·\-\s]*⚡\s*/, ""))}</div>`;
    return `<div class="li"><span>${inline(line.replace(/^[•·-]\s*/, ""))}</span></div>`;
  }).join("");
}

function tabModel() {
  const m = S.model;
  if (!m) return `<div class="center"><div class="spin"></div></div>`;
  const t = m.meta;
  const stat = (k, v) => `<div class="card stat"><div class="k">${k}</div><div class="v num">${v}</div></div>`;
  const hosp = (t.hospital_names || []).map((n, i) => `<div class="hosp"><div class="n">${i}</div><div><b>${esc(n)}</b><div class="tiny mute">~${Math.round((t.train_samples || 0) / (t.hospitals || 1)).toLocaleString()} private training samples</div></div></div>`).join("");
  const rb = (t.risk_classification_metrics || {}).risk_bands, ha = (t.risk_classification_metrics || {}).high_risk_alert;
  const bandRows = rb ? Object.entries(rb.per_class).map(([k, v]) => `<tr><td>${esc(k)}</td><td class="num">${v.precision.toFixed(2)}</td><td class="num">${v.recall.toFixed(2)}</td><td class="num">${v.f1.toFixed(2)}</td></tr>`).join("") : "";
  return `<div class="stack">
    <div class="card"><h3>Trained without sharing patient data</h3>
      <div class="small mute" style="margin:2px 0 14px">${t.hospitals} hospitals each trained on their own private records. Only model weights were shared and averaged — ${t.num_rounds} rounds of ${esc(t.aggregation || "")}.</div>
      <div class="grid g3">${hosp}</div></div>
    <div class="grid g-auto">
      ${stat("Mean absolute error", (t.final_mae ?? 0).toFixed(2) + " SOFA")}${stat("R²", (t.final_r2 ?? 0).toFixed(2))}
      ${stat("Training samples", (t.train_samples || 0).toLocaleString())}${stat("Held-out samples", (t.test_samples || 0).toLocaleString())}
      ${stat("Best round", t.best_round ?? "–")}${stat("Input features", m.n_features)}${ha ? stat("High-risk AUROC", ha.auroc.toFixed(2)) : ""}${stat("Alert threshold", "SOFA ≥ " + m.alert_threshold)}</div>
    ${rb ? `<div class="card"><h3>How well it sorts patients into risk bands</h3><div class="tbl-wrap"><table><thead><tr><th>Band</th><th>Precision</th><th>Recall</th><th>F1</th></tr></thead><tbody>${bandRows}</tbody></table></div>
      ${ha ? `<div class="small mute" style="margin-top:10px">Alert at predicted SOFA ≥ ${ha.predicted_sofa_threshold}: catches ${(ha.recall * 100).toFixed(0)}% of true high-risk cases (precision ${(ha.precision * 100).toFixed(0)}%). The model tends to under-predict the most severe patients.</div>` : ""}</div>` : ""}
    <div class="card"><h3>Privacy &amp; limits</h3><ul class="plain small">
      <li>Differential privacy is ${t.differential_privacy ? "enabled" : "supported but switched off in this build"}.</li>
      <li>Architecture: ${esc(t.model_architecture || "")}.</li>
      <li>${esc(t.evaluation_limitation || "Treat these as initial evaluation results, not clinical validation.")}</li>
      <li>${m.conformal_calibrated ? "Prediction interval calibrated on the federated training data." : "Prediction interval uses a default width because the federated calibration data is not loaded."}</li>
      <li>${m.llm_enabled ? "AI report: enabled." : "AI report: needs a real GROQ_API_KEY in .env."}</li></ul></div></div>`;
}

/* ── render orchestration ────────────────────────────────────────────────── */
function renderView() {
  const view = $("#view");
  if (S.page === "fl") { view.innerHTML = renderFL(); updateFL(); return; }
  const inp = $("#notein"), hadFocus = inp && document.activeElement === inp;
  const caret = hadFocus ? inp.selectionStart : 0;
  if (S.loadError) view.innerHTML = `<div class="center"><h2>The model could not load</h2><p class="mute">${esc(S.loadError)}</p></div>`;
  else if (!S.ready) view.innerHTML = `<div class="center"><div class="spin"></div><h2>Warming up the model…</h2><p class="mute">The first start loads the neural network and the explainer — about 30 seconds.</p></div>`;
  else if (S.view === "picker") { view.innerHTML = renderPicker(); S.animate = false; }
  else view.innerHTML = renderMonitor();
  const again = $("#notein");
  if (again && hadFocus) { again.focus(); again.setSelectionRange(caret, caret); }
}
function renderAll() { renderTop(); renderCtrl(); renderView(); }

/* ── monitoring loop ─────────────────────────────────────────────────────── */
async function openPatient(id) {
  S.patient = S.patients.find((p) => p.id === id);
  S.animate = true; S.view = "monitor"; S.tab = "live"; S.res = null; S.report = null; S.reportFor = null;
  S.history = []; S.paused = false; S.noteDraft = "";
  renderAll();
  try {
    await api("/api/start", { patient: id });
    await tick();
  } catch (e) { toast("Could not start: " + e.message); }
}

async function tick() {
  if (S.busy || S.page === "fl" || S.view !== "monitor" || !S.patient) return;
  S.busy = true;
  const pid = S.patient.id;
  try {
    const res = await api("/api/tick", { patient: pid, mode: S.mode, use_cam: S.cam });
    if (S.patient && S.patient.id !== pid) return;          // user switched patient mid-request
    S.res = res;
    S.history = await api("/api/history/" + pid);
    S.reportFor = res.report.id; S.report = { status: "pending" };
    pollReport(res.report.id);
  } catch (e) { toast(e.message); }
  finally {
    S.busy = false; S.nextAt = Date.now() + S.pace * 1000;
    if (S.view === "monitor") renderView();
  }
}

async function pollReport(id) {
  for (let i = 0; i < 80; i++) {
    await new Promise((r) => setTimeout(r, 1500));
    if (S.reportFor !== id || S.view !== "monitor") return;     // a newer reading took over
    try {
      const rep = await api("/api/report/" + id);
      if (rep.status !== "pending") { S.report = rep; if (S.tab === "report" && S.page !== "fl") renderView(); return; }
    } catch { return; }
  }
}

function everyFrame() {
  const bar = $("#timerbar");
  if (S.page === "fl" || S.view !== "monitor" || !bar) return;
  if (S.paused) return;
  const total = S.pace * 1000, left = S.nextAt - Date.now();
  bar.style.width = S.busy || !S.res ? "100%" : Math.max(0, Math.min(100, ((total - left) / total) * 100)) + "%";
  if (!S.busy && S.res && left <= 0) tick();
}

async function pollCamera() {
  if (S.page === "fl" || S.view !== "monitor" || !S.cam) return;
  try {
    S.camStatus = await api("/api/camera");
    if (S.camStatus.error) { await stopCamera(S.camStatus.error); return; }
    updateCamUI();
  } catch { /* ignore */ }
}

async function startCamera() {
  S.cam = true; S.camStatus = null; updateCamUI();
  try { S.camStatus = await api("/api/camera/start", {}); }
  catch (e) { await stopCamera(e.message); return; }
  toast("Camera starting — a window will open. Keep your face in view for about 3 seconds.");
  updateCamUI();
}

async function stopCamera(reason) {
  S.cam = false; S.camStatus = null;
  const cb = $("#cam"); if (cb) cb.checked = false;
  try { await api("/api/camera/stop", {}); } catch { /* already stopped */ }
  updateCamUI();
  if (reason) toast("Camera stopped: " + reason);
}

/* ── events ──────────────────────────────────────────────────────────────── */
document.addEventListener("click", async (e) => {
  const t = e.target.closest("button"); if (!t) return;
  if (t.dataset.page) { setPage(t.dataset.page); return; }
  if (t.dataset.fla) { flaAction(t.dataset.fla); return; }
  if (t.dataset.mode) {
    S.mode = t.dataset.mode; lsSet("mode", S.mode);
    S.patients = await api("/api/patients?mode=" + S.mode).catch(() => []);
    if (S.view === "monitor" && !S.patients.find((p) => p.id === S.patient.id)) { S.view = "picker"; S.res = null; S.patient = null; }   // e.g. ICU-only patient
    else if (S.view === "monitor") S.patient = S.patients.find((p) => p.id === S.patient.id);
    renderAll();
  } else if (t.dataset.pid) openPatient(+t.dataset.pid);
  else if (t.dataset.tab) {
    S.tab = t.dataset.tab; S.animate = true; renderView();
    if (S.tab === "model" && !S.model) { S.model = await api("/api/model").catch(() => null); renderView(); }
  } else if (t.dataset.pace) { S.pace = +t.dataset.pace; lsSet("pace", S.pace); S.nextAt = Date.now() + S.pace * 1000; renderCtrl(); }
  else if (t.id === "pause") { S.paused = !S.paused; if (!S.paused) S.nextAt = Date.now() + S.pace * 1000; renderCtrl(); }
  else if (t.id === "back") { if (S.cam) stopCamera(); S.animate = true; S.view = "picker"; S.res = null; S.patient = null; renderAll(); }
  else if (t.id === "clearnote") { await api("/api/note", { patient: S.patient.id, text: "" }); S.noteDraft = ""; toast("Live note cleared — applies from the next reading"); }
});

document.addEventListener("change", (e) => {
  if (e.target.id === "cam") { e.target.checked ? startCamera() : stopCamera(); }
});
document.addEventListener("input", (e) => { if (e.target.id === "notein") S.noteDraft = e.target.value; });
document.addEventListener("submit", async (e) => {
  if (e.target.id !== "noteform") return;
  e.preventDefault();
  const text = S.noteDraft.trim(); if (!text) return;
  try {
    await api("/api/note", { patient: S.patient.id, text });
    toast("Note added — it is used from the next reading");
    S.nextAt = Math.min(S.nextAt, Date.now() + 1500);          // reading follows shortly so the effect is visible
  } catch (err) { toast(err.message); }
});

/* ── boot ────────────────────────────────────────────────────────────────── */
/* ── federated learning: one simple animation ─────────────────────────────── */
const FLA = { i: 0, playing: false, timer: null, steps: [], keys: false };
(function () {
  const s = [{ p: "intro", r: 0 }];
  for (let r = 1; r <= 3; r++) ["send", "train", "return", "average"].forEach((p) => s.push({ p, r }));
  s.push({ p: "done", r: 3 });
  FLA.steps = s;
})();
const FLA_TEXT = {
  intro: ["Three organisations, three private sets of patients", "Each one keeps its patient records locked inside its own walls. Together they want to build one smarter AI, without sharing a single record."],
  send: ["The server sends a model to everyone", "Every organisation receives the same starting model."],
  train: ["Each organisation trains it on its own patients", "The learning happens inside each organisation. The records never move."],
  return: ["Only the lessons are sent back", "Small packets of learned changes (“weights”) travel to the server. No patient data."],
  average: ["The server combines the lessons", "It averages them into one improved shared model."],
  done: ["One model that learned from everyone", "After a few rounds the shared model has learned from all three organisations, and no patient record ever left its owner."],
};
const FLA_DUR = { intro: 4200, send: 3300, train: 3800, return: 3300, average: 3400, done: 0 };
const FLA_SITES = ["Home-care provider", "Regional hospital", "Tertiary ICU"];
const FLA_X = [16, 50, 84], FLA_SY = 76, FLA_SRV = [50, 21];
const FLA_METER = [8, 40, 68, 92];

const ico = {
  db: (c) => `<svg viewBox="0 0 32 36" width="34" height="38"><ellipse cx="16" cy="7" rx="12" ry="5" fill="${c}"/><path d="M4 7v20c0 2.8 5.4 5 12 5s12-2.2 12-5V7c0 2.8-5.4 5-12 5S4 9.8 4 7z" fill="${c}" opacity=".78"/></svg>`,
  lock: `<svg viewBox="0 0 16 18" width="13" height="15"><rect x="2" y="8" width="12" height="9" rx="2.2" fill="#1F6B58"/><path d="M5 8V5.5a3 3 0 016 0V8" fill="none" stroke="#1F6B58" stroke-width="1.8"/></svg>`,
  net: (c) => `<svg viewBox="0 0 34 26" width="30" height="23"><g stroke="${c}" stroke-width="1.1" opacity=".55"><path d="M6 7L17 5M6 7l11 9M6 19l11-14M6 19l11-3M17 5l11 8M17 16l11-3M17 5l11 8"/></g><g fill="${c}"><circle cx="6" cy="7" r="2.6"/><circle cx="6" cy="19" r="2.6"/><circle cx="17" cy="5" r="2.6"/><circle cx="17" cy="16" r="2.6"/><circle cx="28" cy="13" r="2.8"/></g></svg>`,
  nums: `<svg viewBox="0 0 18 14" width="16" height="13"><g fill="#fff"><rect x="0" y="1" width="18" height="3" rx="1.5"/><rect x="0" y="6" width="12" height="3" rx="1.5"/><rect x="0" y="11" width="15" height="3" rx="1.5"/></g></svg>`,
  server: `<svg viewBox="0 0 40 34" width="38" height="32"><rect x="3" y="2" width="34" height="9" rx="3" fill="#1F6B58"/><rect x="3" y="13" width="34" height="9" rx="3" fill="#1F6B58" opacity=".8"/><rect x="3" y="24" width="34" height="8" rx="3" fill="#1F6B58" opacity=".6"/><g fill="#fff"><circle cx="9" cy="6.5" r="1.4"/><circle cx="9" cy="17.5" r="1.4"/><circle cx="9" cy="28" r="1.4"/></g></svg>`,
};

function renderFL() {
  const site = (k) => `<div class="fnode fsite" id="fsite${k}" style="left:${FLA_X[k]}%;top:${FLA_SY}%">
      <div class="fname">${FLA_SITES[k]}</div>
      <div class="frow"><div class="fdb">${ico.db("#9AA59C")}<span class="flockbadge">${ico.lock}</span><span class="fghost">✕</span></div>
        <div class="fmini">${ico.net("#1F6B58")}<div class="fbars"><i></i><i></i><i></i><i></i></div></div></div>
      <div class="fcap">Private patient records</div></div>`;
  const pk = (cls, label, k) => `<div class="fpk ${cls}" id="${cls}${k}">${cls === "fmodel" ? ico.net("#fff") : ico.nums}<span>${label}</span></div>`;
  return `<div class="fla">
    <div class="fcaption" id="fcaption"></div>
    <div class="fscene" id="fscene" data-phase="intro">
      <svg class="flines" viewBox="0 0 160 90" preserveAspectRatio="none" aria-hidden="true">
        ${FLA_X.map((x) => `<line x1="80" y1="29" x2="${(x * 1.6).toFixed(1)}" y2="57" />`).join("")}</svg>
      <div class="fnode fserver" id="fserver" style="left:${FLA_SRV[0]}%;top:${FLA_SRV[1]}%">
        <div class="frow">${ico.server}<div><div class="fname">Central server</div><div class="fcap" style="margin:0">Never sees a patient record</div></div></div>
        <div class="fmeter"><span>Shared model</span><div class="ftrack"><div class="ffill" id="ffill"></div></div></div></div>
      ${[0, 1, 2].map(site).join("")}
      ${[0, 1, 2].map((k) => pk("fmodel", "model", k)).join("")}
      ${[0, 1, 2].map((k) => pk("fweights", "weights", k)).join("")}
      <div class="fround" id="fround"></div>
    </div>
    <div class="fcontrols">
      <button class="btn" data-fla="restart" title="Start again">↺ Restart</button>
      <button class="btn" data-fla="back">← Back</button>
      <button class="btn primary" id="fplay" data-fla="play">▶ Play</button>
      <button class="btn" data-fla="next">Next →</button>
      <div class="fsteps" id="fsteps">${["Send the model", "Train locally", "Send lessons back", "Combine"].map((t, n) => `<span data-n="${n}"><b>${n + 1}</b>${t}</span>`).join("")}</div>
    </div>
    <div class="flegend"><span><i class="m"></i>Model</span><span><i class="w"></i>Learned changes (weights)</span><span><i class="r"></i>Patient records, which never move</span></div>
  </div>`;
}

function flaPut(el, x, y, o, sc, instant) {
  if (!el) return;
  if (instant) el.style.transition = "none";
  el.style.left = x + "%"; el.style.top = y + "%"; el.style.opacity = o;
  el.style.transform = `translate(-50%,-50%) scale(${sc})`;
  if (instant) { void el.offsetWidth; el.style.transition = ""; }
}

function flaGo(i) {
  i = Math.max(0, Math.min(FLA.steps.length - 1, i));
  FLA.i = i;
  const { p, r } = FLA.steps[i];
  const scene = $("#fscene"); if (!scene) return;
  scene.dataset.phase = p;
  for (let k = 0; k < 3; k++) {
    const m = $("#fmodel" + k), w = $("#fweights" + k), x = FLA_X[k];
    if (p === "intro" || p === "done") { flaPut(m, FLA_SRV[0], 30, 0, 0.6, true); flaPut(w, x, 62, 0, 0.6, true); }
    if (p === "send") { flaPut(m, FLA_SRV[0], 31, 1, 0.9, true); flaPut(w, x, 62, 0, 0.6, true); setTimeout(() => flaPut(m, x, 58, 1, 1, false), 60); }
    if (p === "train") { flaPut(m, x, 58, 0, 0.5, false); flaPut(w, x, 62, 0, 0.6, true); }
    if (p === "return") { flaPut(m, x, 58, 0, 0.5, true); flaPut(w, x, 60, 1, 0.9, true); setTimeout(() => flaPut(w, FLA_SRV[0], 36, 1, 0.9, false), 60); }
    if (p === "average") { flaPut(m, x, 58, 0, 0.5, true); flaPut(w, FLA_SRV[0], 36, 1, 0.9, true); setTimeout(() => flaPut(w, FLA_SRV[0], 24, 0, 0.3, false), 60); }
  }
  const level = p === "intro" ? 0 : p === "done" ? 3 : p === "average" ? r : r - 1;
  $("#ffill").style.width = FLA_METER[Math.min(3, level)] + "%";
  const [title, text] = FLA_TEXT[p];
  $("#fcaption").innerHTML = `${r && p !== "done" ? `<div class="fr">Round ${r} of 3</div>` : ""}<h2>${title}</h2><p>${text}</p>`;
  document.querySelectorAll("#fsteps span").forEach((s) => s.classList.toggle("on", ["send", "train", "return", "average"][+s.dataset.n] === p));
  $("#fround").textContent = r && p !== "done" ? `Round ${r} / 3` : "";
  const pb = $("#fplay"); if (pb) pb.textContent = FLA.playing ? "❚❚ Pause" : (p === "done" ? "▶ Play again" : "▶ Play");
}

function flaSchedule() {
  clearTimeout(FLA.timer);
  if (!FLA.playing) return;
  const d = FLA_DUR[FLA.steps[FLA.i].p];
  if (!d) { FLA.playing = false; flaGo(FLA.i); return; }
  FLA.timer = setTimeout(() => { flaGo(FLA.i + 1); flaSchedule(); }, d);
}
function flaPause() { FLA.playing = false; clearTimeout(FLA.timer); }
function flaAction(a) {
  if (a === "play") {
    if (FLA.playing) flaPause();
    else { if (FLA.steps[FLA.i].p === "done") flaGo(0); FLA.playing = true; if (FLA.i === 0) flaGo(1); flaSchedule(); }
    flaGo(FLA.i);
  } else if (a === "next") { flaPause(); flaGo(FLA.i + 1); }
  else if (a === "back") { flaPause(); flaGo(FLA.i - 1); }
  else if (a === "restart") { flaPause(); flaGo(0); }
}
function updateFL() {
  flaGo(FLA.i);
  if (!FLA.keys) {
    FLA.keys = true;
    document.addEventListener("keydown", (e) => {
      if (S.page !== "fl" || /INPUT|TEXTAREA|SELECT/.test(e.target.tagName)) return;
      if (e.key === " ") { e.preventDefault(); flaAction("play"); }
      else if (e.key === "ArrowRight") flaAction("next");
      else if (e.key === "ArrowLeft") flaAction("back");
    });
  }
}

function setPage(p) {
  if (S.page === p) return;
  S.page = p;
  if (p !== "fl") flaPause();
  if (p === "monitor") S.nextAt = Date.now() + S.pace * 1000;    // do not fire a reading the instant we return
  S.animate = true;
  renderAll();
}


(async function boot() {
  renderAll();
  for (;;) {
    try {
      const h = await api("/api/health");
      if (h.error) { S.loadError = h.error; renderView(); return; }
      if (h.ready) break;
    } catch { /* server still starting */ }
    await new Promise((r) => setTimeout(r, 1500));
  }
  S.ready = true;
  S.patients = await api("/api/patients?mode=" + S.mode);
  renderAll();
  setInterval(everyFrame, 200);
  setInterval(pollCamera, 1000);
})();
