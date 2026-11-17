// The printable document of a detailed report (?report=name) or a
// summary (?summary=name). It opens the print dialog once drawn.
"use strict";

const params = new URLSearchParams(location.search);

function day(iso) {
  const d = new Date(iso);
  return isNaN(d) ? (iso || "—") : d.toLocaleString("es-ES", {
    day: "2-digit", month: "2-digit", year: "numeric", hour: "2-digit", minute: "2-digit",
  });
}

function minutes(seconds) {
  const s = Math.round(seconds || 0);
  return s < 60 ? `${s} s` : `${Math.floor(s / 60)} min ${s % 60} s`;
}

function sourceName(source) {
  const text = String(source || "").replace(/^camera /, "");
  return /^[a-z]+:\/\//i.test(text) ? text : text.split("/").pop();
}

// The methods an analysis used, for the facts table.
function methods(a) {
  const found = [];
  if (a.summary?.frames_with_person) found.push("REBA (NTP 601, INSST)");
  if (a.lifting?.lifts?.length) found.push("Ecuación de NIOSH (NTP 477) y Guía técnica de manipulación manual de cargas (INSST, 2024)");
  if (a.office) found.push("RD 488/1997 y Guía técnica de pantallas de visualización (INSST, 2021)");
  return found.join("; ") || "—";
}

function facts(rows) {
  return `<table class="facts"><tbody>${rows.map(([k, v]) =>
    `<tr><th>${esc(k)}</th><td>${esc(v)}</td></tr>`).join("")}</tbody></table>`;
}

const NOTICE = "Documento generado automáticamente a partir de un análisis por visión artificial "
  + "de vídeo y redactado con un modelo de lenguaje, que cita los textos normativos y "
  + "técnicos recuperados del corpus documental. No sustituye a la evaluación de riesgos "
  + "realizada por personal técnico competente.";

async function load(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`No se pudo cargar el documento (${response.status}).`);
  return response.json();
}

async function draw() {
  const sheet = document.getElementById("document");
  let type, title, rows, text;
  try {
    if (params.has("summary")) {
      const s = await load(`/api/summaries/${encodeURIComponent(params.get("summary"))}`);
      type = "Informe resumen";
      title = s.title;
      text = s.markdown;
      const period = (s.title.split(":")[1] || "").trim();
      rows = [["Periodo", period || "—"], ["Análisis incluidos", String(s.reports.length)],
              ["Fecha de emisión", day(s.created)], ["Referencia", params.get("summary")]];
    } else {
      const name = params.get("report") || "";
      const data = await load(`/api/reports/${encodeURIComponent(name)}`);
      if (!data.written) throw new Error("El informe detallado aún no está generado.");
      const a = data.analysis;
      const seconds = Math.max(a.summary?.seconds || 0, a.office?.seconds_at_screen || 0);
      type = "Informe detallado";
      title = data.written.title;
      text = data.written.markdown;
      rows = [["Puesto o fuente analizada", sourceName(a.source)],
              ["Fecha del análisis", day(a.created)], ["Duración analizada", minutes(seconds)],
              ["Métodos y criterios", methods(a)], ["Fecha de emisión", day(data.written.created)],
              ["Referencia", name]];
    }
  } catch (error) {
    sheet.innerHTML = `<p class="error">${esc(error.message)}</p>`;
    document.getElementById("toolbar-title").textContent = "No se pudo cargar";
    return;
  }
  document.title = title;
  document.getElementById("toolbar-title").textContent = title;
  sheet.innerHTML = `
    <p class="doc-type">${esc(type)}</p>
    <h1>${esc(title)}</h1>
    ${facts(rows)}
    <div class="body">${markdown(text)}</div>
    <p class="notice">${esc(NOTICE)}</p>`;
  const button = document.getElementById("print");
  button.disabled = false;
  button.addEventListener("click", () => window.print());
  // ?preview only shows it (to check how it looks).
  if (!params.has("preview")) setTimeout(() => window.print(), 300);
}

document.getElementById("close").addEventListener("click", () => window.close());
draw();
