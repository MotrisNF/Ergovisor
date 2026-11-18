// Shared by the interface and the printable document: escaping and
// the small Markdown the language model is told to write.
"use strict";

function esc(value) {
  return String(value ?? "").replace(/[&<>"']/g, c => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  })[c]);
}

// The Markdown the model is told to use: titles, paragraphs, lists,
// tables, bold, italics and citations [n]. Escaped first: nothing it
// writes becomes HTML.
function inline(text) {
  return esc(text)
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[^*])\*([^*]+)\*/g, "$1<em>$2</em>")
    .replace(/\[(\d+(?:\s*,\s*\d+)*)\]/g, '<span class="cite">[$1]</span>');
}

function markdown(text) {
  const lines = String(text || "").split("\n");
  const out = [];
  let list = null;
  const closeList = () => { if (list) { out.push(`</${list}>`); list = null; } };
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trim();
    if (!line) { closeList(); continue; }
    const heading = line.match(/^(#{1,4})\s+(.*)$/);
    if (heading) {
      closeList();
      const level = heading[1].length;
      out.push(`<h${level}>${inline(heading[2])}</h${level}>`);
      continue;
    }
    if (line.startsWith("|") && /^\|?\s*:?-{2,}/.test((lines[i + 1] || "").trim())) {
      closeList();
      const cells = row => row.trim().replace(/^\||\|$/g, "").split("|").map(c => c.trim());
      const head = cells(line);
      const rows = [];
      i += 2;
      while (i < lines.length && lines[i].trim().startsWith("|")) rows.push(cells(lines[i++]));
      i--;
      out.push(`<div class="table-wrap"><table class="table"><thead><tr>${head.map(h => `<th>${inline(h)}</th>`).join("")}</tr></thead>
        <tbody>${rows.map(r => `<tr>${head.map((_, c) => `<td>${inline(r[c] ?? "")}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`);
      continue;
    }
    const bullet = line.match(/^[-*]\s+(.*)$/);
    const numbered = line.match(/^(\d+)[.)]\s+(.*)$/);
    if (bullet || numbered) {
      const kind = bullet ? "ul" : "ol";
      // A numbered list cut by bullets goes on with the model's own
      // number, not from 1 again.
      const start = numbered && numbered[1] !== "1" ? ` start="${Number(numbered[1])}"` : "";
      if (list !== kind) { closeList(); out.push(`<${kind}${start}>`); list = kind; }
      out.push(`<li>${inline(bullet ? bullet[1] : numbered[2])}</li>`);
      continue;
    }
    closeList();
    out.push(`<p>${inline(line)}</p>`);
  }
  closeList();
  return out.join("\n");
}

