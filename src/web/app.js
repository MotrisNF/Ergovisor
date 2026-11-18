// The posture analysis interface. Plain JavaScript, no build step.
"use strict";

const LEVELS = ["Inapreciable", "Bajo", "Medio", "Alto", "Muy alto"];
const KIND = { camera: "Cámara del equipo", address: "Cámara IP", video: "Vídeo" };
const STATUS = {
  loading: "Cargando el modelo…", running: "Analizando", saving: "Guardando el informe…",
  done: "Terminado", error: "Error",
};
// Short names of the corpus documents, by file name.
const DOCS = [
  [/rd_487_1997/, "Real Decreto 487/1997 (manipulación manual de cargas)"],
  [/ley_31_1995/, "Ley 31/1995 de Prevención de Riesgos Laborales"],
  [/guia_tecnica_mmc/, "Guía técnica de manipulación manual de cargas (INSST, 2024)"],
  [/ntp_601/, "NTP 601: método REBA (INSST)"],
  [/ntp_477/, "NTP 477: ecuación de NIOSH (INSST)"],
];
const POLL_MS = 1000;
// How serious each screen-work warning is shown (REBA's colours).
const OFFICE_LEVEL = {
  too_close: 3, neck_flexed: 2, neck_extended: 2, neck_turned: 2,
  no_visual_rest: 2, no_break: 2,
};

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];

function clock(seconds) {
  const s = Math.max(0, seconds || 0);
  return `${String(Math.floor(s / 60)).padStart(2, "0")}:${(s % 60).toFixed(1).padStart(4, "0").replace(".", ",")}`;
}

// A number with a decimal comma: 1,25.
function num(value, digits = 1) {
  return value == null ? "—" : Number(value).toLocaleString("es-ES", {
    minimumFractionDigits: digits, maximumFractionDigits: digits,
  });
}

function duration(seconds) {
  const s = Math.round(seconds || 0);
  if (s < 60) return `${s} s`;
  const m = Math.floor(s / 60);
  return m < 60 ? `${m} min ${s % 60} s` : `${Math.floor(m / 60)} h ${m % 60} min`;
}

function date(iso) {
  if (!iso) return "—";
  const d = new Date(iso);
  return isNaN(d) ? iso : d.toLocaleString("es-ES", {
    day: "2-digit", month: "2-digit", year: "numeric", hour: "2-digit", minute: "2-digit",
  });
}

function badge(level, text) {
  const lv = Math.max(0, Math.min(4, level ?? 0));
  return `<span class="badge lv${lv}">${esc(text ?? LEVELS[lv])}</span>`;
}

function docName(path) {
  const name = (path || "").split("/").pop();
  const found = DOCS.find(([pattern]) => pattern.test(name));
  return found ? found[1] : name;
}

// The article or section of a passage, from its heading.
function section(heading) {
  const found = String(heading || "").match(/(Artículo|Disposición|ANEXO|Anexo|CAPÍTULO|TÍTULO|Factores de riesgo).*$/);
  return found ? ` · ${found[0]}` : "";
}

function niceSource(source) {
  const text = String(source || "").replace(/^camera /, "");
  // A file: only its name, not the folder it was read from.
  return /^[a-z]+:\/\//i.test(text) ? text : text.split("/").pop();
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: options.body && typeof options.body === "string"
      ? { "Content-Type": "application/json" } : {},
    ...options,
  });
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const data = await response.json();
      detail = typeof data.detail === "string" ? data.detail
        : (data.detail || []).map(d => d.msg).join("; ") || detail;
    } catch { /* not JSON */ }
    throw new Error(detail);
  }
  return response.json();
}

function toast(html, timeout = 6000) {
  const el = document.createElement("div");
  el.className = "toast";
  el.innerHTML = html;
  $("#toasts").append(el);
  setTimeout(() => el.remove(), timeout);
}

/* ---------- Routing ---------- */

function route() {
  const hash = location.hash || "#/directo";
  const [, view, name] = hash.match(/^#\/([^/]+)\/?(.*)$/) || [, "directo", ""];
  const current = ["directo", "informes", "informe", "resumen"].includes(view) ? view : "directo";
  $$(".view").forEach(v => { v.hidden = v.id !== `view-${current}`; });
  $$(".tabs a").forEach(a => {
    const on = a.dataset.tab === current || (["informe", "resumen"].includes(current) && a.dataset.tab === "informes");
    if (on) a.setAttribute("aria-current", "page"); else a.removeAttribute("aria-current");
  });
  document.title = { directo: "En directo", informes: "Informes", informe: "Informe", resumen: "Resumen" }[current];
  if (current === "informes") loadReports();
  if (current === "informe") showReport(decodeURIComponent(name));
  if (current === "resumen") showSummary(decodeURIComponent(name));
  window.scrollTo(0, 0);
}

/* ---------- Live sessions ---------- */

const tiles = new Map();
const announced = new Set();

function tileHtml(s) {
  return `
    <div class="tile-header">
      <span class="dot"></span>
      <div class="tile-title">
        <strong title="${esc(s.label)}">${esc(s.label)}</strong>
        <small class="status"></small>
      </div>
      <span class="tag view-tag" hidden>Puesto con pantalla</span>
      ${s.recording ? `<span class="tag rec">● Grabando</span>` : ""}
      <span class="tag">${esc(KIND[s.kind])}</span>
      <span class="tag">${esc(s.model)}</span>
      <button class="icon-button close" type="button" title="Cerrar y guardar el informe" aria-label="Cerrar">
        <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M6 6l12 12M18 6L6 18"/></svg>
      </button>
    </div>
    <div class="video">
      <img alt="Imagen en directo de ${esc(s.label)} con el esqueleto" hidden>
      <div class="state"><div><div class="spinner"></div><span class="state-text">Cargando el modelo…</span></div></div>
    </div>
    <div class="progress" hidden><span></span></div>
    <div class="tile-body">
      <div class="stats">
        <span>Tiempo <b class="time">00:00.0</b></span>
        <span>Momentos de riesgo <b class="n-events">0</b></span>
        <span>Levantamientos <b class="n-lifts">0</b></span>
        <span>Empujes y arrastres <b class="n-pushes">0</b></span>
        <span>Riesgo máx. <b class="max">—</b></span>
        <span class="people-stat" hidden>Personas <b class="n-people">0</b></span>
      </div>
      <div class="stats office-stats" hidden></div>
      <p class="warning-line" hidden></p>
      <ul class="events"></ul>
      <details class="findings" hidden>
        <summary>Normativa encontrada (<span class="n-findings">0</span>)</summary>
        <ul></ul>
      </details>
    </div>`;
}

function updateTile(tile, s) {
  const el = tile.el;
  $(".dot", el).className = `dot ${s.status}`;
  let status = STATUS[s.status] || s.status;
  if (s.status === "running" && s.progress != null) status += ` · ${Math.round(s.progress * 100)} %`;
  $(".status", el).textContent = status;
  const img = $("img", el);
  const state = $(".state", el);
  if (s.status === "running" && !tile.streaming) {
    img.src = `/api/sessions/${s.id}/stream`;
    img.hidden = false;
    tile.streaming = true;
    img.addEventListener("load", () => { state.hidden = true; }, { once: true });
  }
  if (s.status === "error") {
    state.hidden = false;
    state.innerHTML = `<div><strong>No se pudo analizar</strong><br>${esc(s.error)}</div>`;
    img.hidden = true;
  } else if (s.status === "saving") {
    $(".state-text", el) && ($(".state-text", el).textContent = "Guardando el informe…");
  }
  const progress = $(".progress", el);
  progress.hidden = s.progress == null;
  if (s.progress != null) $("span", progress).style.width = `${s.progress * 100}%`;
  $(".time", el).textContent = clock(s.second);
  $(".n-events", el).textContent = s.events.length + s.office_events.length;
  $(".n-lifts", el).textContent = s.lifts.length;
  $(".n-pushes", el).textContent = (s.pushes || []).length;
  const worst = s.events.reduce((m, e) => (e.level > (m?.level ?? -1) ? e : m), null);
  $(".max", el).innerHTML = worst ? badge(worst.level, `${LEVELS[worst.level]} · REBA ${worst.score}`) : "—";
  // People in the picture now; shown once there has been more than one.
  const persons = new Set([...s.events, ...s.lifts, ...(s.pushes || []), ...s.office_events]
    .map(e => e.person || 1));
  tile.several = tile.several || s.people > 1 || persons.size > 1;
  $(".people-stat", el).hidden = !tile.several;
  $(".n-people", el).textContent = s.people || 0;
  const who = e => tile.several ? `<span class="tag">P${e.person || 1}</span>` : "";
  const warning = $(".warning-line", el);
  warning.hidden = !s.warnings.length;
  warning.textContent = s.warnings.at(-1) || "";
  const office = s.office;
  $(".view-tag", el).hidden = !office;
  const officeStats = $(".office-stats", el);
  officeStats.hidden = !office;
  if (office) {
    officeStats.innerHTML = `
      ${tile.several ? `<span>Pantalla <b>Persona ${office.person || 1}</b></span>` : ""}
      <span>Distancia <b>${office.distance_cm != null ? `${num(office.distance_cm, 0)} cm` : "—"}</b></span>
      <span>Cuello <b>${office.pitch != null ? `${num(office.pitch, 0)}°` : "—"}</b></span>
      <span>Parpadeo <b>${office.blinks_per_minute != null ? `${num(office.blinks_per_minute, 0)}/min` : "—"}</b></span>
      <span>Sin descanso visual <b>${duration(office.since_rest)}</b></span>
      <span>Sin pausa <b>${duration(office.since_break)}</b></span>`;
  }
  const total = s.events.length + s.office_events.length;
  if (tile.events !== total) {
    tile.events = total;
    const items = [
      ...s.events.map(e => ({ start: e.start, html: `${who(e)}${badge(e.level, `REBA ${e.score}`)}
        <span class="causes">${esc(e.causes.join(", ") || "postura")}</span>` })),
      ...s.office_events.map(e => ({ start: e.start, html: `${who(e)}${badge(OFFICE_LEVEL[e.kind] ?? 2, e.title)}
        <span class="causes">${esc(e.detail)}</span>` })),
    ].sort((a, b) => a.start - b.start);
    $(".events", el).innerHTML = items.slice(-6).reverse()
      .map(i => `<li><time>${clock(i.start)}</time>${i.html}</li>`).join("");
  }
  const findings = $(".findings", el);
  findings.hidden = !s.findings.length;
  if (tile.findings !== s.findings.length) {
    tile.findings = s.findings.length;
    $(".n-findings", el).textContent = s.findings.length;
    $("ul", findings).innerHTML = s.findings.map(f => `<li><strong>${esc(f.title)}</strong>: ${
      f.passages.map(p => esc(docName(p.file_path))).join("; ") || "sin pasajes"}</li>`).join("");
  }
  $(".close", el).title = s.status === "error" ? "Quitar" : "Cerrar y guardar el informe";
}

async function closeSession(id) {
  try {
    await api(`/api/sessions/${id}`, { method: "DELETE" });
    pollSessions();
  } catch (error) {
    toast(esc(error.message));
  }
}

async function pollSessions() {
  let sessions;
  try {
    sessions = await api("/api/sessions");
  } catch {
    return;
  }
  const seen = new Set();
  for (const s of sessions) {
    seen.add(s.id);
    if (s.status === "done") {
      // Saved: it goes to the history and frees its place.
      if (!announced.has(s.id)) {
        announced.add(s.id);
        toast(`Informe guardado: ${esc(s.label)}${s.recording ? ` · vídeo en ${esc(s.recording)}` : ""} <a href="#/informe/${encodeURIComponent(s.report)}">Ver</a>`, 12000);
        api(`/api/sessions/${s.id}`, { method: "DELETE" }).catch(() => {});
        refreshReportCount();
      }
      continue;
    }
    let tile = tiles.get(s.id);
    if (!tile) {
      const el = document.createElement("article");
      el.className = "tile";
      el.innerHTML = tileHtml(s);
      $(".close", el).addEventListener("click", () => closeSession(s.id));
      // A click on the picture shows it full screen; another click (or
      // Esc) goes back.
      const video = $(".video", el);
      video.title = "Clic para verla a pantalla completa";
      video.addEventListener("click", () => {
        if (document.fullscreenElement) document.exitFullscreen();
        else if (!$("img", video).hidden) video.requestFullscreen?.();
      });
      $("#sessions").append(el);
      tile = { el, streaming: false, events: -1, findings: -1 };
      tiles.set(s.id, tile);
    }
    updateTile(tile, s);
  }
  for (const [id, tile] of tiles) {
    if (!seen.has(id) || sessions.find(s => s.id === id)?.status === "done") {
      $("img", tile.el).src = "";
      tile.el.remove();
      tiles.delete(id);
    }
  }
  const active = sessions.filter(s => s.status !== "done").length;
  $("#live-count").textContent = active;
  $("#sessions-empty").hidden = active > 0;
}

/* ---------- Add a source ---------- */

let kind = "camera";
let busyCameras = new Set();

function selectKind(next) {
  kind = next;
  // Recording is only offered when the server allows it, for cameras.
  $("#record-option").hidden = !llm.record || kind === "video";
  $$(".segmented button").forEach(b => b.setAttribute("aria-selected", b.dataset.kind === kind));
  $$(".pane").forEach(p => { p.hidden = p.dataset.pane !== kind; });
  $("#add-error").hidden = true;
}

async function openAdd() {
  $("#add-error").hidden = true;
  $("#record").checked = false;
  $("#record-option").hidden = !llm.record || kind === "video";
  $("#upload").hidden = true;
  $("#add-submit").disabled = false;
  const dialog = $("#add-dialog");
  dialog.showModal();
  const list = $("#devices");
  list.innerHTML = `<p class="muted">Buscando cámaras…</p>`;
  try {
    const [devices, sessions] = await Promise.all([api("/api/devices"), api("/api/sessions")]);
    busyCameras = new Set(sessions.filter(s => s.camera != null && ["loading", "running", "saving"].includes(s.status))
      .map(s => String(s.camera)));
    list.innerHTML = devices.length ? devices.map((d, i) => {
      const busy = busyCameras.has(String(d.number));
      return `<label class="choice"><input type="radio" name="device" value="${d.number}"
        ${busy ? "disabled" : ""} ${i === 0 && !busy ? "checked" : ""}>
        <span>${esc(d.name)}</span><small>${busy ? "En uso" : `/dev/video${d.number}`}</small></label>`;
    }).join("") : `<p class="muted">No se ha encontrado ninguna cámara en este equipo.</p>`;
  } catch (error) {
    list.innerHTML = `<p class="error">${esc(error.message)}</p>`;
  }
}

function showAddError(message) {
  const el = $("#add-error");
  el.textContent = message;
  el.hidden = false;
}

function uploadVideo(file, model) {
  return new Promise((resolve, reject) => {
    const request = new XMLHttpRequest();
    request.open("POST", `/api/sessions/video?filename=${encodeURIComponent(file.name)}&model=${model}`);
    request.upload.addEventListener("progress", e => {
      if (!e.lengthComputable) return;
      $("#upload-bar").style.width = `${(e.loaded / e.total) * 100}%`;
      $("#upload-text").textContent = `Subiendo… ${Math.round((e.loaded / e.total) * 100)} %`;
    });
    request.addEventListener("load", () => {
      let data = {};
      try { data = JSON.parse(request.responseText); } catch { /* empty */ }
      if (request.status >= 200 && request.status < 300) resolve(data);
      else reject(new Error(typeof data.detail === "string" ? data.detail : `Error ${request.status}`));
    });
    request.addEventListener("error", () => reject(new Error("No se pudo subir el vídeo.")));
    request.send(file);
  });
}

function recordChosen() {
  return Boolean(llm.record && $("#record").checked);
}

async function submitAdd() {
  const model = $("#model").value;
  const button = $("#add-submit");
  button.disabled = true;
  try {
    if (kind === "camera") {
      const chosen = $("input[name=device]:checked");
      if (!chosen) throw new Error("Elige una cámara.");
      await api("/api/sessions", { method: "POST", body: JSON.stringify({ kind, source: chosen.value, model, record: recordChosen() }) });
    } else if (kind === "address") {
      const address = $("#address").value.trim();
      if (!address) throw new Error("Escribe la dirección de la cámara.");
      await api("/api/sessions", { method: "POST", body: JSON.stringify({ kind, source: address, model, record: recordChosen() }) });
    } else {
      const file = $("#file").files[0];
      if (!file) throw new Error("Elige un vídeo.");
      $("#upload").hidden = false;
      await uploadVideo(file, model);
    }
    $("#add-dialog").close();
    $("#address").value = "";
    $("#file").value = "";
    $("#file-name").textContent = "Arrastra un vídeo aquí o haz clic para elegirlo";
    location.hash = "#/directo";
    pollSessions();
  } catch (error) {
    showAddError(error.message);
  } finally {
    button.disabled = false;
  }
}

function setupAdd() {
  $("#add").addEventListener("click", openAdd);
  document.addEventListener("click", e => {
    if (e.target.closest("[data-action=add]")) openAdd();
  });
  $$(".segmented button").forEach(b => b.addEventListener("click", () => selectKind(b.dataset.kind)));
  $("#add-submit").addEventListener("click", submitAdd);
  const drop = $("#drop");
  const file = $("#file");
  file.addEventListener("change", () => {
    $("#file-name").textContent = file.files[0]?.name || "Arrastra un vídeo aquí o haz clic para elegirlo";
  });
  ["dragenter", "dragover"].forEach(t => drop.addEventListener(t, e => {
    e.preventDefault(); drop.classList.add("over");
  }));
  ["dragleave", "drop"].forEach(t => drop.addEventListener(t, e => {
    e.preventDefault(); drop.classList.remove("over");
  }));
  drop.addEventListener("drop", e => {
    if (e.dataTransfer.files.length) {
      file.files = e.dataTransfer.files;
      file.dispatchEvent(new Event("change"));
    }
  });
}

/* ---------- Reports ---------- */

let reports = [];

async function refreshReportCount() {
  try {
    reports = await api("/api/reports");
    $("#report-count").textContent = reports.length;
    if (!$("#view-informes").hidden) renderReports();
  } catch { /* keep the old count */ }
}

async function loadReports() {
  await refreshReportCount();
  renderReports();
  loadSummaries();
}

function renderReports() {
  const filter = $("#report-filter").value.trim().toLowerCase();
  const shown = reports.filter(r => !filter
    || `${r.source} ${r.name} ${date(r.created)}`.toLowerCase().includes(filter));
  $("#reports-empty").hidden = reports.length > 0;
  $("#reports").parentElement.hidden = reports.length === 0;
  $("#reports tbody").innerHTML = shown.map(r => `
    <tr class="link" data-name="${esc(r.name)}">
      <td>${esc(date(r.created))}</td>
      <td class="source" title="${esc(niceSource(r.source))}">${esc(niceSource(r.source))}${
        r.people > 1 ? ` <span class="tag">${r.people} personas</span>` : ""}</td>
      <td class="num">${esc(duration(r.seconds))}</td>
      <td>${r.max_score ? badge(r.max_level, `${LEVELS[r.max_level]} · REBA ${r.max_score}`)
        : r.screen_warnings != null ? badge(r.screen_warnings ? 2 : 1, `Pantalla · ${r.screen_warnings} avisos`)
        : badge(0, "Sin datos")}</td>
      <td class="num">${r.events}</td>
      <td class="num">${r.lifts}</td>
      <td class="num">${maxIndex(r.max_index, r.beyond_limits)}</td>
      <td class="num">${r.topics != null ? `${r.topics} temas` : "—"}</td>
      <td class="actions">
        <button class="icon-button delete" type="button" title="Borrar el informe" aria-label="Borrar el informe">
          <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 7h16M10 11v6M14 11v6M6 7l1 13h10l1-13M9 7V4h6v3"/></svg>
        </button>
      </td>
    </tr>`).join("");
}

function confirmAction(title, text, ok = "Borrar", danger = true) {
  return new Promise(resolve => {
    const dialog = $("#confirm-dialog");
    $("#confirm-title").textContent = title;
    $("#confirm-text").textContent = text;
    $("#confirm-ok").textContent = ok;
    $("#confirm-ok").className = `button ${danger ? "danger" : "primary"}`;
    dialog.returnValue = "";
    dialog.showModal();
    dialog.addEventListener("close", () => resolve(dialog.returnValue === "ok"), { once: true });
  });
}

function confirmDelete(name, label) {
  return confirmAction("¿Borrar el informe?",
    `Se borrarán el análisis, la normativa y el informe redactado de «${label}». No se puede deshacer.`);
}

async function deleteReport(name, label) {
  if (!(await confirmDelete(name, label))) return;
  try {
    await api(`/api/reports/${encodeURIComponent(name)}`, { method: "DELETE" });
    toast("Informe borrado.");
    if (location.hash.startsWith("#/informe/")) location.hash = "#/informes";
    else loadReports();
    refreshReportCount();
  } catch (error) {
    toast(esc(error.message));
  }
}

function setupReports() {
  $("#report-filter").addEventListener("input", renderReports);
  $("#reports tbody").addEventListener("click", e => {
    const row = e.target.closest("tr[data-name]");
    if (!row) return;
    const report = reports.find(r => r.name === row.dataset.name);
    if (e.target.closest(".delete")) {
      deleteReport(row.dataset.name, niceSource(report?.source) || row.dataset.name);
    } else {
      location.hash = `#/informe/${encodeURIComponent(row.dataset.name)}`;
    }
  });
}

/* ---------- One report ---------- */

function levelStack(summary) {
  const byLevel = summary.seconds_by_level || {};
  const total = Object.values(byLevel).reduce((a, b) => a + b, 0) || 1;
  const parts = [0, 1, 2, 3, 4].map(l => ({ l, s: byLevel[l] || 0 })).filter(p => p.s > 0);
  return `
    <div class="stack" role="img" aria-label="Tiempo en cada nivel de riesgo">
      ${parts.map(p => `<span class="lv${p.l}" style="width:${(p.s / total) * 100}%" title="${LEVELS[p.l]}: ${duration(p.s)}"></span>`).join("")}
    </div>
    <div class="legend">
      ${[0, 1, 2, 3, 4].map(l => `<span class="lv${l}"><i></i>${LEVELS[l]}: <b>${Math.round(((byLevel[l] || 0) / total) * 100)} %</b> (${duration(byLevel[l] || 0)})</span>`).join("")}
    </div>`;
}

function timeline(analysis) {
  const total = Math.max(analysis.summary.seconds || 0,
    ...analysis.events.map(e => e.end), ...(analysis.lifting?.lifts || []).map(l => l.end.second),
    ...(analysis.pushing?.pushes || []).map(p => p.end), 1);
  const width = 1000, left = 120, right = 16, plot = width - left - right;
  const x = s => left + (s / total) * plot;
  const step = [5, 10, 15, 30, 60, 120, 300, 600, 900, 1800].find(v => total / v <= 10) || 3600;
  const ticks = [];
  for (let t = 0; t <= total + 0.001; t += step) ticks.push(t);
  const lifts = analysis.lifting?.lifts || [];
  const color = l => `var(--lv${l})`;
  const liftLevel = l => l.beyond_limits ? 4 : l.index == null ? 0 : l.index > 3 ? 4 : l.index > 1 ? 3 : 1;
  const pushes = analysis.pushing?.pushes || [];
  // A third track only when there were pushes or pulls.
  const bottom = pushes.length ? 132 : 92;
  return `
    <svg viewBox="0 0 ${width} ${bottom + 28}" role="img" aria-label="Línea de tiempo de los momentos de riesgo, los levantamientos y los empujes">
      ${ticks.map(t => `<line class="grid-line" x1="${x(t)}" x2="${x(t)}" y1="10" y2="${bottom}"/>
        <text class="axis" x="${x(t)}" y="${bottom + 18}" text-anchor="middle">${clock(t).replace(/,\d$/, "")}</text>`).join("")}
      ${pushes.length ? `<text class="label" x="0" y="115">Empujes</text>
      <rect class="track" x="${left}" y="98" width="${plot}" height="26" rx="3"/>
      ${pushes.map(p => `<rect x="${x(p.start)}" y="98" width="${Math.max(3, x(p.end) - x(p.start))}" height="26"
        rx="2" style="fill:${color(p.notes.length ? 2 : 1)}"><title>${clock(p.start)}–${clock(p.end)} · ${esc(p.kind)} de ${num(p.metres)} m${p.notes.length ? ": " + esc(p.notes.join("; ")) : ""}</title></rect>`).join("")}` : ""}
      <text class="label" x="0" y="35">Posturas (REBA)</text>
      <rect class="track" x="${left}" y="18" width="${plot}" height="26" rx="3"/>
      ${analysis.events.map(e => `<rect x="${x(e.start)}" y="18" width="${Math.max(2, x(e.end) - x(e.start))}" height="26"
        rx="2" style="fill:${color(e.level)}"><title>${clock(e.start)}–${clock(e.end)} · REBA ${e.score} (${LEVELS[e.level]}): ${esc(e.causes.join(", "))}</title></rect>`).join("")}
      <text class="label" x="0" y="75">Levantamientos</text>
      <rect class="track" x="${left}" y="58" width="${plot}" height="26" rx="3"/>
      ${lifts.map(l => `<rect x="${x(l.start.second)}" y="58" width="${Math.max(3, x(l.end.second) - x(l.start.second))}" height="26"
        rx="2" style="fill:${color(liftLevel(l))}"><title>${clock(l.start.second)}–${clock(l.end.second)} · ${esc(l.kind)} · ${l.beyond_limits ? "LPR 0 kg" : `IL ${num(l.index, 2)}`} (${esc(l.risk || "sin carga indicada")})</title></rect>`).join("")}
    </svg>`;
}

// The highest lifting index; lifts with a limit of 0 kg have none and
// are worse than any.
function maxIndex(index, beyond) {
  return beyond ? "LPR 0 kg" : num(index, 2);
}

function liftLevelBadge(lift) {
  if (lift.beyond_limits) return badge(4, "LPR 0 kg · no aceptable con ninguna carga");
  if (lift.index == null) return badge(0, "Sin carga indicada");
  const level = lift.index > 3 ? 4 : lift.index > 1 ? 3 : 1;
  return badge(level, `IL ${num(lift.index, 2)} · ${lift.risk}`);
}

function regulationHtml(regulation) {
  if (!regulation) {
    return `<p class="muted">Este análisis se hizo sin buscar la normativa.</p>`;
  }
  if (!regulation.findings.length) return `<p class="muted">No hubo riesgos que buscar en la normativa.</p>`;
  return regulation.findings.map(f => `
    <div class="finding">
      <h3>${esc(f.title)} <span class="tag">${f.seen.length} ${f.seen.length === 1 ? "vez" : "veces"}</span></h3>
      <p class="seen">${esc(f.seen.slice(0, 3).join(" · "))}${f.seen.length > 3 ? " …" : ""}</p>
      ${f.passages.map(p => `
        <blockquote class="passage">
          <cite>${esc(docName(p.file_path))}
            <small>${esc(p.section ? ` · ${p.section}` : section(p.heading))}</small></cite>
          <p>${esc(p.excerpt)}${p.excerpt.length >= 300 ? "…" : ""}</p>
        </blockquote>`).join("")}
    </div>`).join("");
}

// The pushes and pulls of a report.
function pushingHtml(p) {
  const [initial, sustained] = p.force_limits_n;
  const forces = p.force_initial_n || p.force_sustained_n
    ? `Fuerza medida: inicial ${p.force_initial_n ? `${num(p.force_initial_n, 0)} N` : "—"}, mantenida ${p.force_sustained_n ? `${num(p.force_sustained_n, 0)} N` : "—"} (límites ${num(initial, 0)} N y ${num(sustained, 0)} N).`
    : `Sin fuerza medida: no se puede comparar con los límites de la guía (${num(initial, 0)} N para iniciar, ${num(sustained, 0)} N para mantener).`;
  return `<section class="card">
      <h2>Empuje y tracción de cargas</h2>
      <p class="muted">${esc(p.method)}. ${esc(forces)}</p>
      <div class="table-wrap"><table class="table">
        <thead><tr><th>Intervalo</th><th>Tipo</th><th class="num">Distancia</th><th class="num">Altura de las manos</th><th class="num">Tronco</th><th>Avisos</th></tr></thead>
        <tbody>${p.pushes.map(x => `<tr>
          <td><code>${clock(x.start)}–${clock(x.end)}</code></td><td>${esc(x.kind)}</td>
          <td class="num">${num(x.metres)} m</td>
          <td class="num">${num(x.hands_low_cm, 0)}–${num(x.hands_high_cm, 0)} cm</td>
          <td class="num">${num(x.trunk, 0)}°</td><td>${esc(x.notes.join("; ") || "—")}</td></tr>`).join("")}
        </tbody></table></div>
      ${p.notes.length ? `<ul class="notes">${p.notes.map(n => `<li>${esc(n)}</li>`).join("")}</ul>` : ""}
    </section>`;
}

// The screen work of a report: measures against the guide and warnings.
function officeHtml(o) {
  const face = o.seconds_with_face || 0;
  const share = seconds => face ? `${Math.round((seconds / face) * 100)} %` : "—";
  const distanceBadge = o.distance_cm == null ? badge(0, "Sin medir")
    : o.distance_cm < 30 ? badge(4, "Menos de 30 cm")
    : o.distance_cm < 40 || o.distance_cm > 75 ? badge(2, "Fuera de 40-75 cm") : badge(1, "Adecuada");
  const restBadge = o.longest_without_rest >= 20 * 60 ? badge(2, "Más de 20 min") : badge(1, "Cada 20 min o menos");
  const breakBadge = o.longest_without_break >= 30 * 60 ? badge(2, "Más de 30 min") : badge(1, "Cada 30 min o menos");
  return `
    <section class="card">
      <h2>Trabajo con pantalla</h2>
      <p class="muted">Medido con la cámara de la pantalla según la guía técnica del INSST (RD 488/1997) y los criterios del cuello de REBA.</p>
      <div class="kpis">
        <div class="kpi"><small>Distancia habitual a la pantalla</small>
          <div>${o.distance_cm != null ? `${num(o.distance_cm, 0)} cm` : "—"}</div>${distanceBadge}</div>
        <div class="kpi"><small>Cuello flexionado &gt; 20°</small><div>${share(o.seconds_neck_flexed)}</div>
          <small>${esc(duration(o.seconds_neck_flexed))} del tiempo con la cara visible</small></div>
        <div class="kpi"><small>Cuello en extensión, girado o inclinado</small>
          <div>${share(o.seconds_neck_extended + o.seconds_neck_turned)}</div>
          <small>Extensión ${esc(duration(o.seconds_neck_extended))} · giro ${esc(duration(o.seconds_neck_turned))}</small></div>
        <div class="kpi"><small>Mayor tiempo sin descanso visual</small><div>${esc(duration(o.longest_without_rest))}</div>${restBadge}
          <small>${o.visual_rests} descansos de 20 s o más</small></div>
        <div class="kpi"><small>Mayor tiempo sin pausa</small><div>${esc(duration(o.longest_without_break))}</div>${breakBadge}
          <small>${o.breaks} pausas de 30 s o más</small></div>
        <div class="kpi"><small>Parpadeo (orientativo)</small>
          <div>${o.blinks_per_minute != null ? `${num(o.blinks_per_minute, 0)}/min` : "—"}</div>
          <small>${o.blinks} parpadeos; la guía no fija un mínimo</small></div>
      </div>
      <h3 style="margin-top:18px">Avisos</h3>
      ${o.events.length ? `<div class="table-wrap" style="margin-top:8px"><table class="table">
        <thead><tr><th>Intervalo</th><th>Aviso</th><th>Detalle</th></tr></thead>
        <tbody>${o.events.map(e => `<tr>
          <td><code>${clock(e.start)}–${clock(e.end)}</code></td>
          <td>${badge(OFFICE_LEVEL[e.kind] ?? 2, e.title)}</td><td>${esc(e.detail)}</td></tr>`).join("")}
        </tbody></table></div>` : `<p class="muted">Ningún aviso: la distancia, el cuello, los descansos y las pausas estuvieron dentro de los criterios.</p>`}
    </section>`;
}

// Whether a person's body was scored (not only at a screen).
function hasBody(p) {
  return p.summary.frames_with_person > 0 || !p.office;
}

// One row per person, to compare them at a glance.
function peopleTable(people) {
  return `<section class="card">
      <h2>Resumen por persona</h2>
      <p class="muted">Cada persona se analiza por separado; el detalle de cada una está más abajo.</p>
      <div class="table-wrap"><table class="table">
        <thead><tr><th>Persona</th><th class="num">Tiempo</th><th>Riesgo postural máx.</th><th class="num">Momentos de riesgo</th>
          <th class="num">Levantamientos</th><th class="num">IL máx.</th><th class="num">Empujes</th><th class="num">Avisos de pantalla</th></tr></thead>
        <tbody>${people.map(p => {
          const s = p.summary;
          return `<tr>
          <td><a href="#persona-${p.person}">Persona ${p.person}</a></td>
          <td class="num">${esc(duration(Math.max(s.seconds || 0, p.office?.seconds_at_screen || 0)))}</td>
          <td>${s.max_score ? badge(s.max_level, `${LEVELS[s.max_level]} · REBA ${s.max_score}`) : "—"}</td>
          <td class="num">${p.events.length}</td>
          <td class="num">${p.lifting?.lifts?.length || 0}</td>
          <td class="num">${maxIndex(p.lifting?.max_index, p.lifting?.beyond_limits)}</td>
          <td class="num">${p.pushing?.pushes?.length || 0}</td>
          <td class="num">${p.office ? p.office.events.length : "—"}</td></tr>`;
        }).join("")}
        </tbody></table></div>
    </section>`;
}

// What one person shows: before the detailed report (screen work, key
// figures, levels and timeline) and after it (the tables).
function personHtml(p) {
  const s = p.summary;
  const lifting = p.lifting;
  const lifts = lifting?.lifts || [];
  const worst = s.max_score ? badge(s.max_level, `${LEVELS[s.max_level]}`) : badge(0, "Sin datos");
  const body = hasBody(p);
  const top = `
    ${p.office ? officeHtml(p.office) : ""}

    ${body ? `<div class="kpis">
      <div class="kpi"><small>Riesgo postural máximo</small><div>REBA ${s.max_score || "—"}</div>${worst}</div>
      <div class="kpi"><small>Momentos de riesgo</small><div>${p.events.length}</div></div>
      <div class="kpi"><small>Levantamientos y descensos</small><div>${lifts.length}</div>
        ${lifting ? `<small>${num(lifting.lifts_per_minute)} por minuto</small>` : ""}</div>
      <div class="kpi"><small>Índice de levantamiento máx.</small><div>${maxIndex(lifting?.max_index, lifting?.beyond_limits)}</div>
        ${lifting?.beyond_limits ? `<small>${lifting.beyond_limits} con límite de 0 kg (fuera de la ecuación)</small>` : ""}
        ${lifting?.load_kg ? `<small>Carga ${lifting.load_kg} kg</small>` : `<small>Sin carga indicada</small>`}</div>
    </div>

    <section class="card">
      <h2>Tiempo en cada nivel de riesgo</h2>
      <p class="muted">Nivel de acción REBA de cada instante con la persona visible de cuerpo entero (NTP 601).</p>
      ${levelStack(s)}
    </section>

    <section class="card timeline">
      <h2>Línea de tiempo</h2>
      <p class="muted">Momentos de riesgo postural y levantamientos a lo largo del análisis. Pasa el ratón por encima para ver el detalle.</p>
      ${timeline(p)}
    </section>` : ""}`;
  const bottom = `
    ${body ? `<section class="card">
      <h2>Momentos de riesgo postural</h2>
      <p class="muted">Intervalos con nivel de acción REBA ${p.events.length ? "igual o superior al de alerta" : ""}.</p>
      ${p.events.length ? `<div class="table-wrap"><table class="table">
        <thead><tr><th>Intervalo</th><th>Nivel</th><th class="num">REBA</th><th>Actuación</th><th>Causas</th></tr></thead>
        <tbody>${p.events.map(e => `<tr>
          <td><code>${clock(e.start)}–${clock(e.end)}</code></td><td>${badge(e.level)}</td>
          <td class="num">${e.score}</td><td>${esc(e.action)}</td><td>${esc(e.causes.join(", ") || "—")}</td></tr>`).join("")}
        </tbody></table></div>` : `<p class="muted">No hubo momentos de riesgo.</p>`}
    </section>` : ""}

    ${body && p.pushing ? pushingHtml(p.pushing) : ""}

    ${body && lifting ? `<section class="card">
      <h2>Levantamientos (ecuación de NIOSH)</h2>
      <p class="muted">${esc(lifting.method)}.</p>
      ${lifts.length ? `<div class="table-wrap"><table class="table">
        <thead><tr><th>Intervalo</th><th>Tipo</th><th class="num">Recorrido</th><th class="num">LPR</th><th>Índice</th><th>Avisos</th></tr></thead>
        <tbody>${lifts.map(l => `<tr>
          <td><code>${clock(l.start.second)}–${clock(l.end.second)}</code></td><td>${esc(l.kind)}</td>
          <td class="num">${num(l.d, 0)} cm</td><td class="num">${num(l.limit)} kg</td>
          <td>${liftLevelBadge(l)}</td><td>${esc(l.notes.join("; ") || "—")}</td></tr>`).join("")}
        </tbody></table></div>` : `<p class="muted">No se detectaron levantamientos.</p>`}
      ${lifting.notes.length ? `<ul class="notes">${lifting.notes.map(n => `<li>${esc(n)}</li>`).join("")}</ul>` : ""}
    </section>` : ""}`;
  return [top, bottom];
}

async function showReport(name) {
  const view = $("#view-informe");
  view.innerHTML = `<p class="muted">Cargando el informe…</p>`;
  let data;
  try {
    data = await api(`/api/reports/${encodeURIComponent(name)}`);
  } catch (error) {
    view.innerHTML = `<div class="crumbs"><a href="#/informes">← Informes</a></div>
      <div class="empty"><h2>No se pudo abrir el informe</h2><p class="muted">${esc(error.message)}</p></div>`;
    return;
  }
  const a = data.analysis;
  const people = a.people || [];
  const several = people.length > 1;
  const source = niceSource(a.source);
  const seconds = Math.max(0, ...people.map(p => Math.max(p.summary.seconds || 0, p.office?.seconds_at_screen || 0)));
  const atScreen = people.filter(p => p.office);
  const anyBody = people.some(hasBody);
  const lifting = people.find(p => p.lifting)?.lifting;
  const pushing = people.find(p => p.pushing)?.pushing;
  const office = atScreen[0]?.office;
  let body;
  if (!people.length) {
    body = `<div class="empty"><h2>No se detectó a ninguna persona</h2>
      <p class="muted">La cámara no vio a nadie el tiempo suficiente para analizarlo.</p></div>
      <details class="card fold" id="written"></details>`;
  } else if (!several) {
    const [top, bottom] = personHtml(people[0]);
    body = `${top}<details class="card fold" id="written"></details>${bottom}`;
  } else {
    body = `${peopleTable(people)}<details class="card fold" id="written"></details>
      ${people.map(p => {
        const [top, bottom] = personHtml(p);
        return `<section class="person" id="persona-${p.person}">
          <h2 class="person-title">Persona ${p.person}</h2>${top}${bottom}</section>`;
      }).join("")}`;
  }
  const one = people.length === 1 ? people[0] : null;
  const s = one?.summary;
  view.innerHTML = `
    <div class="crumbs">
      <a href="#/informes"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M15 6l-6 6 6 6"/></svg>Informes</a>
      <div>
        <button class="button ghost" type="button" id="print">Imprimir</button>
        <button class="button ghost" type="button" id="delete-report">Borrar</button>
      </div>
    </div>
    <div class="report-head">
      <div>
        <h1>${esc(source)}</h1>
        <div class="meta">
          <span>${esc(date(a.created))}</span>
          <span>Duración analizada: ${esc(duration(seconds))}</span>
          ${several ? `<span>${people.length} personas, analizadas por separado</span>` : ""}
          ${one && hasBody(one) ? `<span>Cuerpo entero visible en el ${s.frames ? Math.round((s.frames_with_person / s.frames) * 100) : 0} % de los fotogramas</span>` : ""}
          ${one?.office ? `<span>Frente a la pantalla: ${esc(duration(one.office.seconds_at_screen))}</span>` : ""}
        </div>
      </div>
    </div>

    ${body}

    <section class="card">
      <h2>Normativa aplicable</h2>
      <p class="muted">Pasajes de la normativa y de las guías técnicas encontrados para cada tipo de riesgo.</p>
      ${regulationHtml(data.regulation)}
    </section>

    <section class="card">
      <h2>Método, supuestos y limitaciones</h2>
      <ul class="notes">
        ${several ? `<li>Cada persona se analiza por separado y se numera por su posición en la imagen, sin reconocerla: si una persona sale del plano y entra otra, puede heredar su número.</li>` : ""}
        ${anyBody ? `<li>${esc(a.method)}.</li>
        <li>Aspectos que el esqueleto no permite medir y no se puntúan: ${esc(a.unmeasured.join(", "))}.</li>
        ${(lifting?.assumptions || []).map(n => `<li>${esc(n)}</li>`).join("")}
        ${(pushing?.assumptions || []).map(n => `<li>${esc(n)}</li>`).join("")}` : ""}
        ${office ? `<li>${esc(office.method)}.</li>
        ${office.assumptions.map(n => `<li>${esc(n)}</li>`).join("")}` : ""}
      </ul>
    </section>`;
  $("#print").addEventListener("click", () => printDocument("report", name));
  $("#delete-report").addEventListener("click", () => deleteReport(name, source));
  renderWritten(name, data.written, data.writing);
}

/* ---------- Written reports (Gemini) ---------- */

let llm = { available: false, model: "" };

function writtenMeta(w) {
  return `Generado el ${esc(date(w.created))}`
    + (w.uncited?.length ? ` · se quitaron ${w.uncited.length} citas a pasajes inexistentes` : "");
}

// Whether the detailed report is unfolded; it starts folded.
let writtenOpen = false;
let writtenPoll = null;

function renderWritten(name, written, writing) {
  const box = $("#written");
  if (!box) return;
  clearTimeout(writtenPoll);
  const failed = writing && writing !== "writing";
  let state, actions = "", body = "";
  if (written) {
    state = writtenMeta(written);
    actions = `<a class="button" href="/api/reports/${encodeURIComponent(name)}/docx">Descargar .docx</a>
      <button class="button ghost" type="button" id="write-again" ${llm.available ? "" : "disabled"}>Volver a redactar</button>`;
    body = `<article class="prose">${markdown(written.markdown)}</article>`;
  } else if (writing === "writing") {
    state = `<span class="writing-inline"><span class="spinner"></span>Generando informe…</span>`;
    body = `<p class="muted">El informe aparecerá aquí en cuanto esté listo.</p>`;
    // Ask again until it is written.
    writtenPoll = setTimeout(async () => {
      if (location.hash !== `#/informe/${encodeURIComponent(name)}`) return;
      try {
        const data = await api(`/api/reports/${encodeURIComponent(name)}`);
        renderWritten(name, data.written, data.writing);
      } catch { /* try on the next view */ }
    }, 3000);
  } else {
    state = failed ? `<span class="error-inline">No se pudo generar: ${esc(writing)}</span>`
      : llm.available ? "Sin generar" : "La generación no está activada en este servidor";
    actions = `<button class="button primary" type="button" id="write" ${llm.available ? "" : "disabled"}>
      ${failed ? "Reintentar" : "Generar"}</button>`;
    body = `<p class="muted">Informe formal (objeto, metodología, resultados, valoración, medidas preventivas y limitaciones) redactado por el modelo de lenguaje a partir de este análisis, citando la normativa encontrada.</p>`;
  }
  box.open = writtenOpen;
  box.innerHTML = `
    <summary>
      <svg class="chevron" viewBox="0 0 24 24" aria-hidden="true"><path d="M9 6l6 6-6 6"/></svg>
      <h2>Informe detallado</h2>
      <span class="fold-state">${state}</span>
      <span class="written-actions">${actions}</span>
    </summary>
    <div class="fold-body">${body}</div>`;
  box.ontoggle = () => { writtenOpen = box.open; };
  // The buttons work without folding or unfolding the section.
  $$(".written-actions a, .written-actions button", box).forEach(el =>
    el.addEventListener("click", e => e.stopPropagation()));
  $("#write", box)?.addEventListener("click", e => { e.preventDefault(); writeReport(name, false); });
  $("#write-again", box)?.addEventListener("click", async e => {
    e.preventDefault();
    if (await confirmAction("¿Volver a redactar el informe detallado?",
      `Se hará otra llamada a ${llm.model} y se sustituirá el informe actual.`, "Volver a redactar", false)) {
      writeReport(name, true);
    }
  });
}

async function writeReport(name, again) {
  renderWritten(name, null, "writing");
  clearTimeout(writtenPoll);
  try {
    const written = await api(`/api/reports/${encodeURIComponent(name)}/written?again=${again}`, { method: "POST" });
    renderWritten(name, written, null);
  } catch (error) {
    renderWritten(name, null, error.message);
  }
}

// The printable document of a report or a summary, in a new tab.
async function printDocument(kind, name) {
  if (kind === "report") {
    try {
      const data = await api(`/api/reports/${encodeURIComponent(name)}`);
      if (!data.written) {
        toast("El informe detallado aún no está generado.");
        return;
      }
    } catch (error) {
      toast(esc(error.message));
      return;
    }
  }
  window.open(`/static/print.html?${kind}=${encodeURIComponent(name)}`, "_blank");
}

/* ---------- Summaries ---------- */

let summaries = [];

function localDay(iso) {
  const d = new Date(iso);
  if (isNaN(d)) return "";
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
}

async function loadSummaries() {
  try {
    summaries = await api("/api/summaries");
  } catch { summaries = []; }
  $("#summaries-block").hidden = !summaries.length;
  $("#summaries tbody").innerHTML = summaries.map(s => `
    <tr class="link" data-name="${esc(s.name)}">
      <td>${esc(date(s.created))}</td><td>${esc(s.title)}</td>
      <td class="num">${s.reports}</td><td>${esc(s.model)}</td>
      <td class="actions"><button class="icon-button delete" type="button" title="Borrar el resumen" aria-label="Borrar el resumen">
        <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 7h16M10 11v6M14 11v6M6 7l1 13h10l1-13M9 7V4h6v3"/></svg></button></td>
    </tr>`).join("");
}

function countInPeriod() {
  const start = $("#summary-start").value, end = $("#summary-end").value;
  const n = reports.filter(r => { const d = localDay(r.created); return d >= start && d <= end; }).length;
  $("#summary-count").textContent = `${n} ${n === 1 ? "informe" : "informes"} en el periodo`
    + (llm.available ? ` · una llamada a ${llm.model}` : "");
  $("#summary-submit").disabled = !n || !llm.available;
}

function openSummary() {
  const days = reports.map(r => localDay(r.created)).filter(Boolean).sort();
  $("#summary-start").value = days[0] || localDay(new Date().toISOString());
  $("#summary-end").value = days.at(-1) || localDay(new Date().toISOString());
  $("#summary-error").hidden = true;
  countInPeriod();
  $("#summary-dialog").showModal();
}

async function submitSummary() {
  const button = $("#summary-submit");
  button.disabled = true;
  button.textContent = "Generando resumen…";
  try {
    const written = await api("/api/summaries", { method: "POST",
      body: JSON.stringify({ start: $("#summary-start").value, end: $("#summary-end").value }) });
    $("#summary-dialog").close();
    location.hash = `#/resumen/${encodeURIComponent(written.name)}`;
  } catch (error) {
    $("#summary-error").textContent = error.message;
    $("#summary-error").hidden = false;
  } finally {
    button.textContent = "Redactar resumen";
    countInPeriod();
  }
}

async function showSummary(name) {
  const view = $("#view-resumen");
  view.innerHTML = `<p class="muted">Cargando el resumen…</p>`;
  let s;
  try {
    s = await api(`/api/summaries/${encodeURIComponent(name)}`);
  } catch (error) {
    view.innerHTML = `<div class="crumbs"><a href="#/informes">← Informes</a></div>
      <div class="empty"><h2>No se pudo abrir el resumen</h2><p class="muted">${esc(error.message)}</p></div>`;
    return;
  }
  view.innerHTML = `
    <div class="crumbs">
      <a href="#/informes"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M15 6l-6 6 6 6"/></svg>Informes</a>
      <div class="written-actions">
        <a class="button ghost" href="/api/summaries/${encodeURIComponent(name)}/docx">Descargar .docx</a>
        <button class="button ghost" type="button" id="print">Imprimir</button>
        <button class="button ghost" type="button" id="delete-summary">Borrar</button>
      </div>
    </div>
    <section class="card">
      <p class="note">${writtenMeta(s)} · ${s.reports.length} informes</p>
      <article class="prose">${markdown(s.markdown)}</article>
    </section>`;
  $("#print").addEventListener("click", () => printDocument("summary", name));
  $("#delete-summary").addEventListener("click", () => deleteSummary(name, s.title));
}

async function deleteSummary(name, title) {
  if (!(await confirmAction("¿Borrar el resumen?", `Se borrará «${title}». No se puede deshacer.`))) return;
  try {
    await api(`/api/summaries/${encodeURIComponent(name)}`, { method: "DELETE" });
    toast("Resumen borrado.");
    location.hash = "#/informes";
  } catch (error) {
    toast(esc(error.message));
  }
}

function setupSummaries() {
  $("#summarise").addEventListener("click", openSummary);
  $("#summary-start").addEventListener("change", countInPeriod);
  $("#summary-end").addEventListener("change", countInPeriod);
  $("#summary-submit").addEventListener("click", submitSummary);
  $("#summaries tbody").addEventListener("click", e => {
    const row = e.target.closest("tr[data-name]");
    if (!row) return;
    const summary = summaries.find(s => s.name === row.dataset.name);
    if (e.target.closest(".delete")) deleteSummary(row.dataset.name, summary?.title || row.dataset.name);
    else location.hash = `#/resumen/${encodeURIComponent(row.dataset.name)}`;
  });
}

/* ---------- Start ---------- */

setupAdd();
setupReports();
setupSummaries();
window.addEventListener("hashchange", route);
// Enter in a field of a dialog does what its main button does. Without
// this the browser presses the form's first submit button, the close
// one: pasting an address and pressing Enter closed the dialog.
for (const [form, main] of [["#add-form", "#add-submit"], ["#summary-form", "#summary-submit"]]) {
  $(form).addEventListener("keydown", event => {
    const field = event.target.closest("input, select");
    if (event.key !== "Enter" || !field || event.isComposing) return;
    event.preventDefault();
    const button = $(main);
    if (!button.disabled) button.click();
  });
}

// A folded section would not be printed: unfold it while printing.
let foldedForPrint = [];
window.addEventListener("beforeprint", () => {
  foldedForPrint = $$("details.fold:not([open])");
  foldedForPrint.forEach(d => { d.open = true; });
});
window.addEventListener("afterprint", () => {
  foldedForPrint.forEach(d => { d.open = false; });
  foldedForPrint = [];
});
// Whether the model can write is known before the first view.
api("/api/llm").then(info => { llm = info; }).catch(() => {}).finally(route);
pollSessions();
refreshReportCount();
setInterval(pollSessions, POLL_MS);
