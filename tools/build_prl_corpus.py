"""Build the corpus of occupational risk prevention texts for the RAG.

The laws come from the open data API of the BOE (their consolidated,
current text); the INSST's guides and technical notes come from PDFs
left in ``data/sources/``, because the INSST's site does not allow
automatic downloads. Everything is written as Markdown to
``data/raw/prl/``, ready for ``index``.

From the project root:
    uv run python tools/build_prl_corpus.py
"""
from collections import Counter
from pathlib import Path
import re
import shutil
import subprocess
import sys
import urllib.request
import xml.etree.ElementTree as ET

BOE_API = ("https://www.boe.es/datosabiertos/api/legislacion-consolidada/"
           "id/{}/texto")
# BOE id -> (file name, title).
LAWS = {
    "BOE-A-1997-8670": ("rd_487_1997_manipulacion_cargas",
                        "Real Decreto 487/1997, de 14 de abril, sobre "
                        "disposiciones mínimas de seguridad y salud relativas "
                        "a la manipulación manual de cargas"),
    "BOE-A-1995-24292": ("ley_31_1995_prevencion_riesgos",
                         "Ley 31/1995, de 8 de noviembre, de Prevención de "
                         "Riesgos Laborales"),
}
# PDF file name in data/sources -> title. Other PDFs there are used too,
# titled by their file name.
PDF_TITLES = {
    "guia_tecnica_mmc_2024": "Guía técnica para la evaluación y prevención "
                             "de los riesgos derivados de la manipulación "
                             "manual de cargas (INSST, 2024)",
    "ntp_601": "NTP 601: Evaluación de las condiciones de trabajo: carga "
               "postural. Método REBA (INSST)",
    "ntp_477": "NTP 477: Levantamiento manual de cargas: ecuación del NIOSH "
               "(INSST)",
}
# Paragraph classes of the BOE that are headings.
HEADINGS = {"articulo", "titulo", "titulo_num", "titulo_tit", "capitulo",
            "capitulo_num", "capitulo_tit", "seccion", "anexo",
            "anexo_num", "anexo_tit", "centro_redonda"}
SOURCES = Path("data/sources")
OUT = Path("data/raw/prl")
# A line on at least this share of the pages is a running header or
# footer.
RUNNING = 0.3


def _clean(text: str) -> str:
    """Single spaces inside a paragraph."""
    return re.sub(r"\s+", " ", text).strip()


def law(boe_id: str, name: str, title: str) -> Path:
    """Download a law's consolidated text and write it as Markdown."""
    request = urllib.request.Request(BOE_API.format(boe_id),
                                     headers={"Accept": "application/xml"})
    with urllib.request.urlopen(request, timeout=60) as response:
        root = ET.fromstring(response.read())
    lines = [f"# {title}", "",
             f"Fuente: BOE, texto consolidado ({boe_id}).", ""]
    for block in root.iter("bloque"):
        versions = block.findall("version")
        if not versions:
            continue
        # The last version is the one in force.
        for element in versions[-1]:
            if element.tag == "table":
                for row in element.iter("tr"):
                    cells = [_clean("".join(c.itertext())) for c in row]
                    if any(cells):
                        lines += [" | ".join(cells), ""]
                continue
            text = _clean("".join(element.itertext()))
            if not text:
                continue
            heading = element.get("class", "") in HEADINGS
            lines += [f"## {text}" if heading else text, ""]
    path = OUT / f"{name}.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def pdf(source: Path) -> Path:
    """Turn a PDF into Markdown: its text in reading order, without the
    running headers, footers and page numbers, with its paragraphs
    joined."""
    raw = subprocess.run(["pdftotext", str(source), "-"], check=True,
                         capture_output=True, text=True).stdout
    pages = raw.split("\f")
    seen = Counter(line.strip() for page in pages
                   for line in set(page.splitlines()) if line.strip())
    running = {line for line, count in seen.items()
               if count >= max(3, RUNNING * len(pages)) and len(line) < 100}
    kept = [line.rstrip() for page in pages for line in page.splitlines()
            if line.strip() not in running
            and not re.fullmatch(r"\s*\d{1,3}\s*", line)]
    text = "\n".join(kept)
    # Words cut at the end of a line: "manipula-\nción".
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    # A line that goes on in lower case continues its paragraph.
    text = re.sub(r"(?<![.:;!?])\n(?=[a-záéíóúñü(])", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    title = PDF_TITLES.get(source.stem, source.stem.replace("_", " "))
    path = OUT / f"{source.stem}.md"
    path.write_text(f"# {title}\n\nFuente: {source.name}.\n\n{text}\n",
                    encoding="utf-8")
    return path


def main() -> int:
    """Write every text of the corpus; say which PDFs are missing."""
    OUT.mkdir(parents=True, exist_ok=True)
    for boe_id, (name, title) in LAWS.items():
        path = law(boe_id, name, title)
        print(f"{path}: {path.stat().st_size:,} bytes")
    if shutil.which("pdftotext") is None:
        print("pdftotext (poppler-utils) is needed for the PDFs.",
              file=sys.stderr)
        return 1
    for source in sorted(SOURCES.glob("*.pdf")):
        path = pdf(source)
        print(f"{path}: {path.stat().st_size:,} bytes")
    missing = [f"{name}.pdf" for name in PDF_TITLES
               if not (SOURCES / f"{name}.pdf").is_file()]
    if missing:
        print(f"Missing in {SOURCES}/: {', '.join(missing)}",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
