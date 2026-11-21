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
from datetime import date
import json
import os
from pathlib import Path
import re
import time
from typing import Any, Literal

from pydantic import BaseModel

from ..exceptions import RagError
from .analysis import people_of
from .assessment import (assess, assessment_table, data_table,
                         methods_table)
from .regulation import duration

# The models with the most quota for this project's key. Compared on
# 2026-11-17 with the same analysis: ``lite`` in 7 s, sound but close to
# a copy of the data; ``hydrogen`` in about 1 min, the best report (it
# interprets the data and ranks the measures); ``flash`` in about 5 min,
# between the two.
LlmChoice = Literal["lite", "flash", "hydrogen"]
MODELS: dict[str, str] = {
    "lite": "gemini-3.5-flash-lite",
    "flash": "gemini-3.8-flash",
    "hydrogen": "gemini-4-hydrogen",
}
API = ("https://generativelanguage.googleapis.com/v1beta/models/"
       "{model}:generateContent")
KEY_ENV = "GEMINI_API_KEY"
# Seconds to wait for an answer: a stronger model thinks before writing
# and can take minutes on a long report; giving up earlier loses a call
# that was already counted.
TIMEOUT = 600.0
# Busy model (500, 503): wait these seconds and try again. These answers
# write nothing, so they do not spend quota.
RETRY_WAITS = (5.0, 15.0)
# Characters of corpus text given per passage, at most.
PASSAGE_CHARS = 2000
# Passages given to the model, at most (the most cited topics first).
MAX_PASSAGES = 30
# Causes of risk listed one by one, at most (the most frequent first).
MAX_EVENTS = 25
# A decimal number written with a point: 2.5.
DECIMAL = re.compile(r"(?<![\w.])(\d+)\.(\d+)(?![\w.])")
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
        created: The day it was written (``YYYY-MM-DD``; not the time).
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
con **. Sin bloques de código ni HTML, ni fórmulas LaTeX: escribe los \
símbolos tal cual (≤, ≥, °).
9. Cita solo lo que el pasaje dice. Las advertencias propias de este \
informe (que es un análisis automático, sus limitaciones, que no \
sustituye a la evaluación por personal técnico competente) no llevan \
cita: no están en las fuentes.
10. No exageres los datos: usa sus mismos términos (si dicen "cuello en \
extensión", no escribas "hiperextensión"; si dicen "inclinado", no \
escribas "severamente flexionado").
11. La valoración de cada riesgo (nivel, prioridad y plazo) ya está \
calculada con la escala de cada método en la TABLA DE VALORACIÓN: \
cópiala tal cual y no cambies ningún nivel ni prioridad; tu trabajo es \
explicarla y justificarla con los datos y las citas.
12. Las medidas preventivas deben ser concretas y responder a las causas \
medidas (por ejemplo, tronco flexionado más de 60° al coger del suelo: \
elevar el origen de la carga a la altura de los nudillos), cada una \
ligada a un riesgo de la tabla, con la prioridad y el plazo de ese \
riesgo. Ordénalas según los principios de la acción preventiva: evitar \
el riesgo, combatirlo en su origen, adaptar el puesto a la persona, \
medidas organizativas, y formación e información. Sin medidas \
genéricas que no respondan a un dato. Cuando la prioridad sea \
"Pendiente de datos", la medida es obtener el dato que falta (pesar la \
carga, medir la fuerza con un dinamómetro, indicar la duración).
13. Los datos son anónimos: las personas son "Persona 1", "Persona 2"... \
y solo se sabe cuántas veces ocurrió cada cosa y cuánto duró en total, \
no cuándo. No intentes situarlo en el tiempo ni identificar a nadie."""


def document_name(file_path: str) -> str:
    """The name of a corpus document, from its file name."""
    name = Path(file_path).stem
    return next((title for part, title in DOCUMENTS if part in name), name)


def _percent(part: float, whole: float) -> str:
    return f"{100 * part / whole:.0f} %" if whole else "—"


def _by_level(analysis: dict[str, Any], names: list[str]) -> str:
    """`` (2 de riesgo alto; 1 de riesgo medio)``, or nothing."""
    counts = analysis.get("moments_by_level") or {}
    if not counts:
        return ""
    return " (" + "; ".join(
        f"{count} de riesgo {names[int(level)]}" for level, count in
        sorted(counts.items(), key=lambda item: -int(item[0]))) + ")"


def describe_analysis(analysis: dict[str, Any]) -> str:
    """The analysis in words and numbers, for the model: each person
    under their own heading when there are several."""
    lines = [f"Fuente: {analysis.get('source', '')}",
             f"Fecha del análisis: {analysis.get('created', '')}"]
    people = people_of(analysis)
    unmeasured = analysis.get("unmeasured", [])
    if not people:
        lines.append("No se detectó a ninguna persona en la grabación.")
    elif len(people) == 1:
        lines += _describe_person(people[0], unmeasured)
    else:
        lines += [
            "",
            f"Personas analizadas: {len(people)}, cada una por separado "
            "(numeradas por su posición en la imagen, sin identificarlas: "
            "si una persona sale del plano y entra otra, puede heredar su "
            "número).",
        ]
        for person in people:
            lines += ["", f"# Persona {person.get('person', 1)}"]
            lines += _describe_person(person, unmeasured)
    # Spanish writes decimals with a comma (the model copies them).
    return DECIMAL.sub(r"\1,\2", "\n".join(lines))


def _describe_person(analysis: dict[str, Any], unmeasured: list[str]
                     ) -> list[str]:
    """One person's analysis in words and numbers."""
    lines: list[str] = []
    summary = analysis.get("summary", {})
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
            f"Puntuación REBA máxima de un solo instante: "
            f"{summary.get('max_score', 0)} (nivel de acción "
            f"{summary.get('max_level', 0)}); un instante no es un momento "
            "de riesgo: la valoración se hace con los momentos.",
            f"Puntuación REBA media (ponderada por el tiempo): "
            f"{summary.get('mean_score', 0)} (nivel de acción "
            f"{summary.get('mean_level', 0)}, riesgo "
            f"{names[int(summary.get('mean_level', 0))]}): cómo es el "
            "trabajo habitualmente, frente al momento peor.",
            "Tiempo en cada nivel de riesgo: " + "; ".join(
                f"{names[int(level)]} {_percent(seconds, total)}"
                for level, seconds in sorted(levels.items())) + ".",
            f"Momentos de riesgo (nivel de acción de alerta o superior): "
            f"{analysis.get('moments', 0)}{_by_level(analysis, names)}.",
        ]
        risks = analysis.get("risks", [])
        if risks:
            lines.append("Causas de esos momentos (cuántas veces y cuánto "
                         "tiempo en total; no se guarda cuándo):")
        for risk in risks[:MAX_EVENTS]:
            lines.append(
                f"- {risk['cause']}: {risk['times']} "
                f"{'vez' if risk['times'] == 1 else 'veces'}, "
                f"{duration(risk['seconds'])} en total, REBA máximo "
                f"{risk['max_score']}.")
        if unmeasured:
            lines.append("Aspectos de REBA no medibles con el esqueleto "
                         f"(no puntuados): {', '.join(unmeasured)}.")
    lifting = analysis.get("lifting")
    if lifting and lifting.get("lifts"):
        lines += [
            "",
            "## Manipulación manual de cargas (ecuación de NIOSH, NTP 477; "
            "Guía técnica del INSST)",
            "Carga: " + (f"{lifting['load_kg']} kg"
                         + (" (supuesta: valor por defecto, no pesada)"
                            if lifting.get("load_assumed") else "")
                         if lifting.get("load_kg") else "no indicada")
            + f"; agarre {lifting.get('grip')}; duración de la tarea "
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
                     else "sin índice: el límite es 0 kg, ninguna carga "
                          "es aceptable en esas condiciones"
                          + (f" ({lift['risk']})" if lift.get("risk")
                             else " (sin carga indicada no se asigna un "
                                  "nivel de riesgo)")
                     if lift.get("beyond_limits")
                     else "sin índice (carga no indicada)")
            lines.append(
                f"- {lift['kind'].capitalize()} de "
                f"{lift['d']:.0f} cm, límite de peso recomendado "
                f"{lift['limit']:.1f} kg, {index}; avisos: "
                f"{'; '.join(lift['notes']) or 'ninguno'}.")
        lines += [f"Aviso de la tarea: {note}." for note in lifting["notes"]]
        lines += [f"Supuesto: {note}" for note in lifting["assumptions"]]
    pushing = analysis.get("pushing")
    if pushing and pushing.get("pushes"):
        initial, sustained = pushing["force_limits_n"]
        lines += [
            "",
            "## Empuje y tracción de cargas (Guía técnica del INSST, "
            "2024; límites de fuerza de la norma ISO/TR 12295)",
            f"Empujes y arrastres: {len(pushing['pushes'])} "
            f"({pushing.get('per_minute', 0):.1f} por minuto).",
        ]
        for push in pushing["pushes"]:
            lines.append(
                f"- {push['kind'].capitalize()} de {push['metres']:.1f} m "
                f"({push.get('seconds', 0):.0f} s); manos entre "
                f"{push['hands_low_cm']:.0f} y {push['hands_high_cm']:.0f} "
                f"cm del suelo; tronco hasta {push['trunk']:.0f}°, giro "
                f"hasta {push['twist']:.0f}°; avisos: "
                f"{'; '.join(push['notes']) or 'ninguno'}.")
        if pushing.get("force_initial_n") or pushing.get(
                "force_sustained_n"):
            lines.append(
                f"Fuerza medida: inicial "
                f"{pushing.get('force_initial_n') or 'no medida'} N, "
                f"mantenida {pushing.get('force_sustained_n') or 'no medida'}"
                f" N; límites de la guía: {initial:.0f} N y "
                f"{sustained:.0f} N.")
        else:
            lines.append(
                "La fuerza no se midió: no se puede saber si se superan los "
                f"límites de la guía ({initial:.0f} N para iniciar el "
                f"movimiento y {sustained:.0f} N para mantenerlo).")
        lines += [f"Aviso de la tarea: {note}." for note in pushing["notes"]]
        lines += [f"Supuesto: {note}" for note in pushing["assumptions"]]
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
        for warning in office.get("warnings", []):
            lines.append(f"- Aviso: {warning['title']}, "
                         f"{warning['times']} "
                         f"{'vez' if warning['times'] == 1 else 'veces'}, "
                         f"{duration(warning['seconds'])} en total.")
        lines += [f"Supuesto: {note}" for note in office["assumptions"]]
    return lines


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


# LaTeX a model may write despite being told not to, and its symbol.
LATEX = {r"\le": "≤", r"\leq": "≤", r"\ge": "≥", r"\geq": "≥",
         r"^\circ": "°", r"^{\circ}": "°", r"\times": "×",
         r"\approx": "≈", r"\%": "%"}
LATEX_MATH = re.compile(r"\$([^$\n]{1,20})\$")


def plain_symbols(text: str) -> str:
    """``$\\le$ 25 cm`` -> ``≤ 25 cm``: short LaTeX formulas as the
    symbols they stand for (a formula it does not know is left as is)."""

    def symbol(match: re.Match[str]) -> str:
        inner = match.group(1).strip()
        if "\\" not in inner:
            # Not LaTeX: two dollar signs of prices, for example.
            return match.group(0)
        for code, sign in sorted(LATEX.items(), key=lambda i: -len(i[0])):
            inner = inner.replace(code, sign)
        return match.group(0) if "\\" in inner else inner.strip()

    return LATEX_MATH.sub(symbol, text)


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
    text, cited, unknown = check_citations(plain_symbols(text.strip()),
                                           numbers)
    markdown = f"# {title}\n\n{text}"
    if cited:
        markdown += "\n" + _references(sources, cited)
    return Written(kind=kind, title=title, model=model,
                   created=date.today().isoformat(),
                   markdown=markdown,
                   sources=[s for s, _ in sources if s.number in cited],
                   uncited=unknown, reports=reports)


def report_prompt(analysis: dict[str, Any],
                  sources: list[tuple[Source, str]]) -> str:
    """What the model is asked for one analysis."""
    people = len(people_of(analysis))
    several = (f"""
En la grabación hay {people} personas, analizadas por separado: es un \
solo informe conjunto. En Resultados y en Valoración del riesgo, una \
subsección (###) por persona ("Persona 1", "Persona 2"...) y al final un \
resumen conjunto; las medidas preventivas, comunes, indicando a qué \
persona se refieren cuando sea solo a alguna.
""" if people > 1 else "")
    table = assessment_table(assess(analysis))
    return f"""Redacta el informe de evaluación ergonómica de este puesto \
de trabajo, a partir del análisis automático por vídeo que sigue. Debe \
recoger lo que exige la documentación de una evaluación de riesgos \
(puesto, riesgos, trabajadores afectados, resultado, medidas y métodos) \
y ser útil para planificar la actividad preventiva.
{several}
Estructura:
## 1. Datos de la evaluación
Copia la TABLA DE DATOS sin cambios.
## 2. Objeto y alcance
Breve: qué riesgos ergonómicos se evalúan y cuáles no.
## 3. Metodología y criterios de valoración
Copia la TABLA DE MÉTODOS sin cambiar su contenido; añade en la columna \
de referencia las citas de los pasajes que los describen.
## 4. Resultados
Por tipo de riesgo analizado (omite los no analizados), con tablas de \
los datos clave.
## 5. Valoración del riesgo y conclusiones
Copia la TABLA DE VALORACIÓN tal cual y, debajo, una conclusión breve \
por riesgo que diga por qué tiene ese nivel, con citas.
## 6. Medidas preventivas y planificación
Una tabla: n.º, medida, riesgo que corrige, tipo (técnica, \
organizativa o formación), prioridad, plazo orientativo, cómo \
comprobar su eficacia. Al menos una medida por cada causa de la TABLA \
DE VALORACIÓN (no una sola por riesgo), más la formación e información \
de las personas. Ordenadas por prioridad. Si alguna prioridad es \
"Inmediata", indica también la medida provisional. Añade que los plazos \
son orientativos y los fija la empresa al planificar.
## 7. Limitaciones del análisis
Separa lo medido por el sistema, lo indicado por el usuario y lo \
supuesto; lo que el método no puede medir con un esqueleto.
## 8. Seguimiento
Cuándo y cómo reevaluar tras aplicar las medidas.

# TABLA DE DATOS (cópiala sin cambios)
{data_table(analysis)}

# TABLA DE MÉTODOS (cópiala; añade las citas)
{methods_table(analysis) or "Ningún método aplicado."}

# TABLA DE VALORACIÓN (calculada; cópiala sin cambios)
{table or "No se encontró ningún riesgo que valorar."}

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
        f"### Análisis {number}: {name}\n{describe_analysis(analysis)}\n"
        f"Valoración calculada (no cambies sus niveles ni prioridades):\n"
        f"{assessment_table(assess(analysis)) or 'ningún riesgo'}"
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
## 6. Medidas preventivas recomendadas y prioridades (las de la \
valoración calculada de cada análisis; plazos orientativos)
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
    title = f"Informe de evaluación ergonómica: {Path(source).name}"
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
