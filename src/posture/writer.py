"""The detailed report, written by a language model from the analysis
and the regulation found (the generation step of the RAG).

The model gets only numbers and words: the measures of the analysis and
the passages of the corpus found for each risk, numbered. It must cite
those numbers (``[3]``) for every normative statement; the references at
the end are written here, from the numbers it really cited, so they
always point to real passages. Nothing is generated without a person
asking for it (a button), and what is generated is saved, so a report
costs one call to the model.

Two kinds of document:

- the detailed report of one analysis (``write_report``);
- the summary of every analysis of a period, for the company and the
  Labour Inspectorate (``write_summary``).
"""
from collections.abc import Callable, Iterable
from datetime import datetime
import json
import os
from pathlib import Path
import re
import time
from typing import Any, Literal

from pydantic import BaseModel

from ..exceptions import RagError

# The models with the most quota for this project's key: ``lite`` for
# everyday use, ``flash`` writes better (for a demonstration).
LlmChoice = Literal["lite", "flash"]
MODELS: dict[str, str] = {
    "lite": "gemini-3.5-flash-lite",
    "flash": "gemini-3.8-flash",
}
API = ("https://generativelanguage.googleapis.com/v1beta/models/"
       "{model}:generateContent")
KEY_ENV = "GEMINI_API_KEY"
TIMEOUT = 240.0
# Busy model (500, 503): wait these seconds and try again. These answers
# write nothing, so they do not spend quota.
RETRY_WAITS = (5.0, 15.0)
# Characters of corpus text given per passage, at most.
PASSAGE_CHARS = 2000
# Passages given to the model, at most (the most cited topics first).
MAX_PASSAGES = 30
# Moments of risk listed one by one, at most (the worst first).
MAX_EVENTS = 25
CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
# A call to the model: (system instructions, prompt) -> text.
Llm = Callable[[str, str], str]


class Source(BaseModel):
    """A numbered passage given to the model.

    Attributes:
        number: How the text cites it (``[number]``).
        file_path: The corpus file.
        first_character_index: Start of the passage.
        last_character_index: End of the passage.
        document: The document's name.
        section: Its article or section, if known.
    """

    number: int
    file_path: str
    first_character_index: int
    last_character_index: int
    document: str
    section: str


class Written(BaseModel):
    """A document written by the model.

    Attributes:
        kind: ``report`` (one analysis) or ``summary`` (a period).
        title: Its title.
        model: The model that wrote it.
        created: When (local time, ISO).
        markdown: The text, with the references added at the end.
        sources: The passages it cites, by number.
        uncited: Numbers it cited that were not given (removed from the
            text); should be empty.
        reports: For a summary, the reports it covers.
    """

    kind: Literal["report", "summary"]
    title: str
    model: str
    created: str
    markdown: str
    sources: list[Source]
    uncited: list[int] = []
    reports: list[str] = []


DOCUMENTS = [
    ("rd_487_1997", "Real Decreto 487/1997, sobre manipulación manual de "
                    "cargas"),
    ("rd_488_1997", "Real Decreto 488/1997, sobre equipos con pantallas de "
                    "visualización"),
    ("ley_31_1995", "Ley 31/1995, de Prevención de Riesgos Laborales"),
    ("guia_tecnica_mmc", "Guía técnica para la evaluación y prevención de "
                         "los riesgos relativos a la manipulación manual de "
                         "cargas (INSST, 2024)"),
    ("guia_tecnica_pvd", "Guía técnica para la evaluación y prevención de "
                         "los riesgos relativos a la utilización de equipos "
                         "con pantallas de visualización (INSST, 2021)"),
    ("ntp_601", "NTP 601. Evaluación de las condiciones de trabajo: carga "
                "postural. Método REBA (INSST)"),
    ("ntp_477", "NTP 477. Levantamiento manual de cargas: ecuación del "
                "NIOSH (INSST)"),
]

SYSTEM = """Eres un técnico superior en prevención de riesgos laborales \
en España, experto en ergonomía (posturas, manipulación de cargas y \
trabajo con pantallas). Los análisis son solo ergonómicos. Redactas \
informes técnicos en castellano, con estilo formal, preciso y sobrio.

Reglas obligatorias:
1. Usa SOLO los datos del análisis y los pasajes de referencia que se te \
dan. No inventes mediciones, artículos, valores límite ni documentos.
2. Toda afirmación sobre lo que dice una norma o una guía técnica lleva su \
cita con el número del pasaje entre corchetes, por ejemplo [2] o [1, 4]. \
Cita solo números de la lista de pasajes.
3. Si un dato no está en el análisis, no lo supongas: indica que no se ha \
medido.
4. Distingue los resultados medidos de las recomendaciones.
5. El análisis es automático, con visión artificial a partir de vídeo; \
recoge sus limitaciones y deja claro que no sustituye a la evaluación de \
riesgos realizada por personal técnico competente.
6. No escribas una sección de referencias ni bibliografía: se añade \
después automáticamente.
7. No saques conclusiones que los datos no permitan. En particular, si \
la carga no está indicada en kg, no digas que se supera (ni que se \
respeta) el límite de peso recomendado: di que no se puede valorar sin \
el peso de la carga. Usa la actuación de cada nivel REBA tal como se da \
en los datos.
8. Formato Markdown: títulos con ##, listas con -, tablas con |, negrita \
con **. Sin bloques de código ni HTML."""


def document_name(file_path: str) -> str:
    """The name of a corpus document, from its file name."""
    name = Path(file_path).stem
    return next((title for part, title in DOCUMENTS if part in name), name)


def _clock(second: float) -> str:
    return f"{int(second // 60):02d}:{second % 60:04.1f}"


def _percent(part: float, whole: float) -> str:
    return f"{100 * part / whole:.0f} %" if whole else "—"


def describe_analysis(analysis: dict[str, Any]) -> str:
    """The analysis in words and numbers, for the model."""
    lines = [f"Fuente: {analysis.get('source', '')}",
             f"Fecha del análisis: {analysis.get('created', '')}"]
    summary = analysis.get("summary", {})
    events = analysis.get("events", [])
    office = analysis.get("office")
    if summary.get("frames_with_person"):
        levels = summary.get("seconds_by_level", {})
        total = sum(levels.values()) or 1.0
        names = ["inapreciable (actuación no necesaria)",
                 "bajo (actuación puede ser necesaria)",
                 "medio (actuación necesaria)",
                 "alto (actuación necesaria pronto)",
                 "muy alto (actuación inmediata)"]
        lines += [
            "",
            "## Posturas (método REBA, Rapid Entire Body Assessment; "
            "NTP 601)",
            f"Duración analizada con la persona de cuerpo entero: "
            f"{summary.get('seconds', 0):.0f} s.",
            f"Puntuación REBA máxima: {summary.get('max_score', 0)} "
            f"(nivel de acción {summary.get('max_level', 0)}).",
            "Tiempo en cada nivel de riesgo: " + "; ".join(
                f"{names[int(level)]} {_percent(seconds, total)}"
                for level, seconds in sorted(levels.items())) + ".",
            f"Momentos de riesgo (nivel de acción de alerta o superior): "
            f"{len(events)}.",
        ]
        worst = sorted(events, key=lambda e: -e["score"])[:MAX_EVENTS]
        for event in sorted(worst, key=lambda e: e["start"]):
            lines.append(
                f"- {_clock(event['start'])}–{_clock(event['end'])}: REBA "
                f"{event['score']}, riesgo {event['risk'].lower()}, "
                f"actuación {event['action'].lower()}; causas: "
                f"{', '.join(event['causes']) or 'postura'}.")
        unmeasured = analysis.get("unmeasured", [])
        if unmeasured:
            lines.append("Aspectos de REBA no medibles con el esqueleto "
                         f"(no puntuados): {', '.join(unmeasured)}.")
    lifting = analysis.get("lifting")
    if lifting and lifting.get("lifts"):
        lines += [
            "",
            "## Manipulación manual de cargas (ecuación de NIOSH, NTP 477; "
            "Guía técnica del INSST)",
            f"Carga indicada: {lifting.get('load_kg') or 'no indicada'} kg; "
            f"agarre {lifting.get('grip')}; duración de la tarea "
            f"{lifting.get('duration')}; peso de referencia "
            f"{lifting.get('reference_kg')} kg.",
            f"Levantamientos y descensos: {len(lifting['lifts'])} "
            f"({lifting.get('lifts_per_minute', 0):.1f} por minuto).",
        ]
        if not lifting.get("load_kg"):
            lines.append(
                "La carga no se indicó en kg: no se puede calcular el índice "
                "de levantamiento ni saber si se supera el límite de peso "
                "recomendado; solo se conocen los límites y los avisos de "
                "posición.")
        for lift in lifting["lifts"]:
            index = (f"IL {lift['index']:.2f} ({lift['risk']})"
                     if lift.get("index") is not None
                     else "sin índice (carga no indicada)")
            lines.append(
                f"- {_clock(lift['start']['second'])}: {lift['kind']} de "
                f"{lift['d']:.0f} cm, límite de peso recomendado "
                f"{lift['limit']:.1f} kg, {index}; avisos: "
                f"{'; '.join(lift['notes']) or 'ninguno'}.")
        lines += [f"Aviso de la tarea: {note}." for note in lifting["notes"]]
        lines += [f"Supuesto: {note}" for note in lifting["assumptions"]]
    if office:
        face = office.get("seconds_with_face") or 0.0
        lines += [
            "",
            "## Trabajo con pantalla de visualización (RD 488/1997 y su "
            "guía técnica; cuello según REBA)",
            f"Tiempo frente a la pantalla: "
            f"{office['seconds_at_screen']:.0f} s.",
            f"Distancia habitual ojos-pantalla: "
            f"{office.get('distance_cm') or 'no medida'} cm; la más "
            f"cercana: {office.get('closest_cm') or 'no medida'} cm.",
            f"Cuello flexionado más de 20°: "
            f"{_percent(office['seconds_neck_flexed'], face)} del tiempo "
            f"con la cara visible; en extensión "
            f"{_percent(office['seconds_neck_extended'], face)}; girado o "
            f"inclinado {_percent(office['seconds_neck_turned'], face)}.",
            f"Hombros elevados: {office['seconds_shoulders_raised']:.0f} s.",
            f"Descansos visuales de 20 s o más: {office['visual_rests']}; "
            f"mayor tiempo sin ninguno: "
            f"{office['longest_without_rest'] / 60:.1f} min.",
            f"Pausas de 30 s o más: {office['breaks']}; mayor tiempo sin "
            f"pausa: {office['longest_without_break'] / 60:.1f} min.",
            f"Parpadeo: {office.get('blinks_per_minute') or 'no medido'} "
            "por minuto (orientativo).",
        ]
        if office["seconds_at_screen"] < 30 * 60:
            lines.append(
                "La sesión frente a la pantalla duró menos de 30 min: no "
                "permite valorar si las pausas son suficientes"
                + (" ni los descansos visuales (cada 20 min)"
                   if office["seconds_at_screen"] < 20 * 60 else "") + ".")
        for event in office.get("events", []):
            lines.append(f"- Aviso {_clock(event['start'])}–"
                         f"{_clock(event['end'])}: {event['title']}, "
                         f"{event['detail']}.")
        lines += [f"Supuesto: {note}" for note in office["assumptions"]]
    return "\n".join(lines)


def collect_sources(regulations: Iterable[dict[str, Any] | None],
                    read: Callable[[str], str]) -> list[tuple[Source, str]]:
    """The passages of one or more regulation reports, numbered once
    each (the same chunk found for two topics is given once).

    Args:
        regulations: The ``RegulationReport`` dicts.
        read: Reads a corpus file's text.

    Returns:
        ``(source, its text)`` pairs, the topics seen most first.
    """
    count: dict[tuple[str, int, int], int] = {}
    found: dict[tuple[str, int, int], dict[str, Any]] = {}
    for regulation in regulations:
        for finding in (regulation or {}).get("findings", []):
            for passage in finding.get("passages", []):
                key = (passage["file_path"],
                       passage["first_character_index"],
                       passage["last_character_index"])
                count[key] = count.get(key, 0) + len(finding.get("seen", []))
                found.setdefault(key, passage)
    ordered = sorted(found, key=lambda key: -count[key])[:MAX_PASSAGES]
    sources = []
    texts: dict[str, str] = {}
    for number, key in enumerate(ordered, start=1):
        passage = found[key]
        path, start, end = key
        if path not in texts:
            try:
                texts[path] = read(path)
            except OSError:
                texts[path] = ""
        text = " ".join(texts[path][start:end].split())[:PASSAGE_CHARS]
        sources.append((Source(
            number=number, file_path=path, first_character_index=start,
            last_character_index=end, document=document_name(path),
            section=passage.get("section", "")), text or passage.get(
                "excerpt", "")))
    return sources


def _passages_block(sources: list[tuple[Source, str]]) -> str:
    return "\n\n".join(
        f"[{source.number}] {source.document}"
        f"{' — ' + source.section if source.section else ''}\n{text}"
        for source, text in sources)


def check_citations(markdown: str, numbers: set[int]
                    ) -> tuple[str, list[int], list[int]]:
    """Keep the citations to given passages and drop the others.

    Returns:
        The text, the numbers cited (in order of first use) and the
        numbers cited that were not given.
    """
    cited: list[int] = []
    unknown: list[int] = []

    def fix(match: re.Match[str]) -> str:
        kept = []
        for part in match.group(1).split(","):
            number = int(part)
            if number in numbers:
                kept.append(number)
                if number not in cited:
                    cited.append(number)
            elif number not in unknown:
                unknown.append(number)
        return f"[{', '.join(map(str, kept))}]" if kept else ""

    return CITATION.sub(fix, markdown), cited, unknown


def _references(sources: list[tuple[Source, str]], cited: list[int]) -> str:
    """The references, one line per document with the numbers of its
    passages cited (and their sections, when known)."""
    by_document: dict[str, list[Source]] = {}
    for source, _ in sources:
        if source.number in cited:
            by_document.setdefault(source.document, []).append(source)
    lines = ["", "## Referencias", ""]
    for document, cited_sources in sorted(
            by_document.items(), key=lambda item: item[1][0].number):
        numbers = ", ".join(str(s.number) for s in cited_sources)
        sections = sorted({s.section for s in cited_sources if s.section})
        where = f" ({'; '.join(sections)})" if sections else ""
        lines.append(f"- [{numbers}] {document}{where}.")
    return "\n".join(lines)


def _finish(kind: Literal["report", "summary"], title: str, model: str,
            text: str, sources: list[tuple[Source, str]],
            reports: list[str]) -> Written:
    numbers = {source.number for source, _ in sources}
    text, cited, unknown = check_citations(text.strip(), numbers)
    markdown = f"# {title}\n\n{text}"
    if cited:
        markdown += "\n" + _references(sources, cited)
    return Written(kind=kind, title=title, model=model,
                   created=datetime.now().isoformat(timespec="seconds"),
                   markdown=markdown,
                   sources=[s for s, _ in sources if s.number in cited],
                   uncited=unknown, reports=reports)


def report_prompt(analysis: dict[str, Any],
                  sources: list[tuple[Source, str]]) -> str:
    """What the model is asked for one analysis."""
    return f"""Redacta el informe detallado de evaluación de este puesto de \
trabajo, a partir del análisis automático por vídeo que sigue.

Estructura (omite las secciones de riesgos que no se hayan analizado):
## 1. Objeto y alcance
## 2. Metodología
## 3. Resultados
## 4. Valoración del riesgo
## 5. Medidas preventivas propuestas (ordenadas por prioridad)
## 6. Limitaciones del análisis

# DATOS DEL ANÁLISIS
{describe_analysis(analysis)}

# PASAJES DE REFERENCIA (cítalos por su número)
{_passages_block(sources)}
"""


def summary_prompt(analyses: list[tuple[str, dict[str, Any]]],
                   sources: list[tuple[Source, str]], start: str,
                   end: str) -> str:
    """What the model is asked for the summary of a period."""
    blocks = "\n\n".join(
        f"### Análisis {number}: {name}\n{describe_analysis(analysis)}"
        for number, (name, analysis) in enumerate(analyses, start=1))
    return f"""Redacta un informe resumen de las evaluaciones automáticas \
de puestos de trabajo realizadas entre {start} y {end}, dirigido a la \
dirección de la empresa y útil como documentación ante la Inspección de \
Trabajo y Seguridad Social. Agrega los resultados: no repitas cada \
análisis por separado; señala los riesgos recurrentes, los más graves y \
su frecuencia.

Estructura:
## 1. Objeto
## 2. Alcance (periodo, número de análisis, tipos de puesto)
## 3. Metodología
## 4. Resultados agregados (incluye una tabla con una fila por análisis)
## 5. Riesgos principales y recurrentes
## 6. Medidas preventivas recomendadas y prioridades
## 7. Limitaciones

# ANÁLISIS DEL PERIODO ({len(analyses)})
{blocks}

# PASAJES DE REFERENCIA (cítalos por su número)
{_passages_block(sources)}
"""


def write_report(name: str, analysis: dict[str, Any],
                 regulation: dict[str, Any] | None, llm: Llm, model: str,
                 read: Callable[[str], str]) -> Written:
    """The detailed report of one analysis.

    Raises:
        RagError: The model could not write it.
    """
    sources = collect_sources([regulation], read)
    text = llm(SYSTEM, report_prompt(analysis, sources))
    source = str(analysis.get("source", name)).removeprefix("camera ")
    title = f"Informe detallado de evaluación: {Path(source).name}"
    return _finish("report", title, model, text, sources, [name])


def write_summary(analyses: list[tuple[str, dict[str, Any]]],
                  regulations: list[dict[str, Any] | None], start: str,
                  end: str, llm: Llm, model: str,
                  read: Callable[[str], str]) -> Written:
    """The summary of the analyses of a period.

    Raises:
        RagError: There is no analysis, or the model could not write it.
    """
    if not analyses:
        raise RagError("No hay informes en ese periodo.")
    sources = collect_sources(regulations, read)
    text = llm(SYSTEM, summary_prompt(analyses, sources, start, end))
    title = f"Informe resumen de evaluaciones: {start} a {end}"
    return _finish("summary", title, model, text, sources,
                   [name for name, _ in analyses])


def gemini(model: str, key: str | None = None,
           timeout: float = TIMEOUT,
           waits: tuple[float, ...] = RETRY_WAITS) -> Llm:
    """A call to a Gemini model through its REST API.

    Args:
        model: The model's name (``gemini-3.5-flash-lite``...).
        key: The API key; by default ``GEMINI_API_KEY``.
        timeout: Seconds to wait for the answer.
        waits: Seconds to wait before each new attempt when the model is
            busy (500 or 503).

    Returns:
        A function (system instructions, prompt) -> text.
    """

    def call(system: str, prompt: str) -> str:
        import httpx

        api_key = key or os.environ.get(KEY_ENV)
        if not api_key:
            raise RagError(f"Falta la clave de Gemini: define {KEY_ENV}.")
        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.2,
                                 "maxOutputTokens": 16384},
        }
        for wait in (*waits, None):
            try:
                response = httpx.post(API.format(model=model), json=body,
                                      headers={"x-goog-api-key": api_key},
                                      timeout=timeout)
            except httpx.HTTPError as error:
                raise RagError(f"No se pudo conectar con Gemini: "
                               f"{type(error).__name__}") from error
            if response.status_code not in (500, 503) or wait is None:
                break
            time.sleep(wait)
        if response.status_code == 429:
            raise RagError("Gemini rechaza la petición por cuota (429): "
                           "espera o usa otro modelo.")
        if response.status_code != 200:
            try:
                message = response.json()["error"]["message"]
            except (ValueError, KeyError, TypeError):
                message = response.text[:200]
            raise RagError(f"Gemini respondió {response.status_code}: "
                           f"{message}")
        try:
            data = response.json()
            parts = data["candidates"][0]["content"]["parts"]
            text = "".join(part.get("text", "") for part in parts
                           if not part.get("thought"))
        except (ValueError, KeyError, IndexError, TypeError) as error:
            raise RagError("Gemini no devolvió texto (respuesta "
                           "bloqueada o vacía).") from error
        if not text.strip():
            raise RagError("Gemini devolvió un texto vacío.")
        return text

    return call


def save(written: Written, path: Path) -> None:
    """Write a document as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(written.model_dump_json(indent=2), encoding="utf-8")


def load(path: Path) -> Written | None:
    """A saved document, or ``None`` if there is none."""
    if not path.is_file():
        return None
    try:
        return Written.model_validate(json.loads(path.read_text(
            encoding="utf-8")))
    except (OSError, ValueError):
        return None
