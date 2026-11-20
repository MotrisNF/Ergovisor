"""The assessment of each risk found, and how soon to act on it: worked
out here from the methods' own scales, so the written report states the
same level every time (the language model only writes it up, and does
not grade the risks itself).

A risk assessment must record, for each job, the risks, the workers
affected, the result of the assessment and the preventive measures
(article 7 of the Reglamento de los Servicios de Prevención, RD
39/1997), and the measures are planned "estableciendo las fases y
prioridades de su desarrollo en función de la magnitud de los riesgos y
del número de trabajadores expuestos" (its article 9). The magnitude
comes from each method:

- postures: the REBA action level of the worst moment (NTP 601: 4
  "actuación inmediata", 3 "necesaria pronto", 2 "necesaria", 1 "puede
  ser necesaria");
- lifts: the lifting index (NTP 477: over 3 "inaceptable... debe ser
  modificada", 1 to 3 "deben rediseñarse"); with a limit of 0 kg and a
  load, unacceptable; without the load, it cannot be graded;
- pushes and pulls: the forces are not seen, so they cannot be graded;
  only the INSST guide's warnings on the posture;
- screen work: the warnings against the INSST guide (distance, neck,
  visual rests, breaks).

The deadlines are ours (the regulation sets none: the company plans
them): they are given as a guide and the report says so.
"""
from typing import Any, NamedTuple

from .analysis import people_of
from .regulation import duration

# Ours: how soon to act at each priority (a guide; the company decides).
DEADLINES = {
    "Inmediata": "inmediato, con medidas provisionales mientras tanto",
    "Alta": "en 1 mes como máximo",
    "Media": "en 3 meses como máximo",
    "Baja": "en la próxima revisión de la evaluación",
    "Pendiente de datos": "tras completar los datos que faltan",
}
# NTP 601: REBA action level -> (risk, priority).
REBA_LEVELS = {4: ("Muy alto", "Inmediata"), 3: ("Alto", "Alta"),
               2: ("Medio", "Media"), 1: ("Bajo", "Baja")}
# Screen-work warnings -> priority (ours: the guide's criteria are
# recommendations whose breach builds up over the day).
SCREEN = {"too_close": "Media", "neck_flexed": "Media",
          "neck_extended": "Media", "neck_turned": "Baja",
          "no_visual_rest": "Media", "no_break": "Media"}


class Row(NamedTuple):
    """One risk of one person, assessed.

    Attributes:
        person: Their number (0: the only person).
        risk: The risk, in words.
        method: The method that grades it.
        result: What was measured, in short.
        level: The level of risk the method gives (or why there is none).
        priority: How soon to act (a key of ``DEADLINES``).
    """

    person: int
    risk: str
    method: str
    result: str
    level: str
    priority: str


def _num(value: float, digits: int = 1) -> str:
    """A number with a decimal comma, as Spanish writes it."""
    return f"{value:.{digits}f}".replace(".", ",")


def _times(n: int) -> str:
    return f"{n} {'vez' if n == 1 else 'veces'}"


def _postures(number: int, person: dict[str, Any]) -> list[Row]:
    levels = {int(level): count for level, count
              in (person.get("moments_by_level") or {}).items()}
    if not levels:
        return []
    worst = max(levels)
    risk, priority = REBA_LEVELS.get(worst, ("Inapreciable", "Baja"))
    causes = "; ".join(
        f"{r['cause']} ({_times(r['times'])}, {duration(r['seconds'])})"
        for r in (person.get("risks") or [])[:3])
    # The highest score of the moments of risk (the summary's can be a
    # single frame, under a level that lasted).
    top = max((r["max_score"] for r in person.get("risks") or []),
              default=(person.get("summary") or {}).get("max_score", "—"))
    return [Row(number, "Posturas forzadas", "REBA (NTP 601)",
                f"REBA máximo en esos momentos {top}; "
                f"{person.get('moments', 0)} momentos de riesgo"
                + (f": {causes}" if causes else ""),
                f"{risk} (nivel de acción {worst})", priority)]


def _lifts(number: int, person: dict[str, Any]) -> list[Row]:
    lifting = person.get("lifting") or {}
    lifts = lifting.get("lifts") or []
    if not lifts:
        return []
    method = "Ecuación de NIOSH (NTP 477)"
    count = f"{len(lifts)} levantamientos o descensos"
    load = lifting.get("load_kg")
    indexes = [lift["index"] for lift in lifts
               if lift.get("index") is not None]
    beyond = sum(1 for lift in lifts if lift.get("beyond_limits"))
    if not load:
        lowest = min(lift["limit"] for lift in lifts)
        return [Row(number, "Manipulación manual de cargas", method,
                    f"{count}; límite de peso recomendado más bajo "
                    f"{_num(lowest)} kg"
                    + (f" ({beyond} con 0 kg)" if beyond else ""),
                    "Sin valorar: falta el peso de la carga",
                    "Pendiente de datos")]
    if beyond:
        level, priority = "Incremento acusado del riesgo", "Inmediata"
    elif indexes and max(indexes) > 3:
        level, priority = "Incremento acusado del riesgo", "Inmediata"
    elif indexes and max(indexes) > 1:
        level, priority = "Incremento moderado del riesgo", "Alta"
    else:
        level, priority = "Riesgo limitado", "Baja"
    result = f"{count} de {_num(load).removesuffix(',0')} kg"
    if indexes:
        result += f"; índice máximo {_num(max(indexes), 2)}"
    if beyond:
        result += f"; {beyond} con límite de 0 kg"
    return [Row(number, "Manipulación manual de cargas", method, result,
                level, priority)]


def _pushes(number: int, person: dict[str, Any]) -> list[Row]:
    pushing = person.get("pushing") or {}
    pushes = pushing.get("pushes") or []
    if not pushes:
        return []
    warned = sum(1 for push in pushes if push.get("notes"))
    measured = (pushing.get("force_initial_n")
                or pushing.get("force_sustained_n"))
    over = bool(pushing.get("notes"))
    level = ("Fuerza por encima del límite de la guía" if over
             else "Fuerza dentro del límite de la guía" if measured
             else "Fuerza sin valorar: falta medirla con un dinamómetro")
    if warned and not over:
        level += "; postura con avisos de la guía"
    priority = ("Alta" if over else "Media" if warned
                else "Baja" if measured else "Pendiente de datos")
    return [Row(number, "Empuje y tracción de cargas",
                "Guía técnica del INSST (2024), ISO/TR 12295",
                f"{len(pushes)} empujes o arrastres"
                + (f", {warned} con avisos de postura" if warned else ""),
                level, priority)]


def _screen(number: int, person: dict[str, Any]) -> list[Row]:
    office = person.get("office") or {}
    rows = []
    for warning in office.get("warnings") or []:
        rows.append(Row(
            number, f"Trabajo con pantalla: {warning['title'].lower()}",
            "Guía técnica del INSST (RD 488/1997)",
            f"{_times(warning['times'])}, {duration(warning['seconds'])} "
            "en total", "Fuera de los criterios de la guía",
            SCREEN.get(warning["kind"], "Media")))
    return rows


def assess(analysis: dict[str, Any]) -> list[Row]:
    """Every risk of an analysis, assessed, the most urgent first."""
    people = people_of(analysis)
    rows = []
    for person in people:
        number = person.get("person", 1) if len(people) > 1 else 0
        for part in (_postures, _lifts, _pushes, _screen):
            rows += part(number, person)
    order = list(DEADLINES)
    return sorted(rows, key=lambda row: (order.index(row.priority),
                                         row.person))


def assessment_table(rows: list[Row]) -> str:
    """The rows as a Markdown table (``""`` without any)."""
    if not rows:
        return ""
    several = any(row.person for row in rows)
    head = (["Persona"] if several else []) + [
        "Riesgo", "Método", "Resultado", "Nivel de riesgo", "Prioridad",
        "Plazo orientativo"]
    lines = ["| " + " | ".join(head) + " |",
             "|" + "---|" * len(head)]
    for row in rows:
        cells = ([str(row.person)] if several else []) + [
            row.risk, row.method, row.result, row.level, row.priority,
            DEADLINES[row.priority]]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _day(iso: str) -> str:
    """``2026-11-20`` as Spanish writes it: ``20/11/2026``."""
    parts = iso[:10].split("-")
    return "/".join(reversed(parts)) if len(parts) == 3 else iso or "—"


def data_table(analysis: dict[str, Any]) -> str:
    """The data of the assessment (who, when, how), as a Markdown table:
    fixed facts, so the model copies them instead of writing them."""
    people = people_of(analysis)
    seconds = max([max((p.get("summary") or {}).get("seconds", 0.0),
                       (p.get("office") or {}).get("seconds_at_screen", 0.0))
                   for p in people] + [0.0])
    liftings = [p["lifting"] for p in people if p.get("lifting")]
    pushings = [p["pushing"] for p in people if p.get("pushing")]
    task = []
    if liftings:
        lifting = liftings[0]
        load = lifting.get("load_kg")
        task.append(f"carga {_num(load).removesuffix(',0')} kg (indicada)"
                    if load else "carga no indicada")
        assumed = any("supuesta" in note
                      for note in lifting.get("assumptions", []))
        task.append(f"duración de la tarea {lifting.get('duration')} "
                    f"({'supuesta' if assumed else 'indicada'})")
    if pushings:
        pushing = pushings[0]
        measured = (pushing.get("force_initial_n")
                    or pushing.get("force_sustained_n"))
        task.append("fuerza de empuje medida (indicada)" if measured
                    else "fuerza de empuje no medida")
    source = str(analysis.get("source", "—")).removeprefix("camera ")
    rows = [
        ("Puesto", f"No indicado (fuente analizada: {source})"),
        ("Fecha del análisis", _day(str(analysis.get("created", "")))),
        ("Personas analizadas", f"{len(people)}, anónimas (numeradas por "
                                "su posición en la imagen)"),
        ("Tiempo analizado", duration(seconds)),
        ("Obtención de los datos", "Análisis automático de vídeo (visión "
                                   "artificial): ángulos, alturas, "
                                   "distancias y tiempos"),
        ("Datos de la tarea", "; ".join(task) or "ninguno necesario"),
    ]
    return "\n".join(["| Dato | Valor |", "|---|---|"]
                     + [f"| {name} | {value} |" for name, value in rows])


# Each risk -> (method, how it is graded, the documents that describe it).
METHODS = {
    "postures": ("Posturas forzadas", "REBA (Rapid Entire Body "
                 "Assessment)", "Puntuación de 1 a 15 y nivel de acción de "
                 "0 (inapreciable) a 4 (muy alto, actuación inmediata)",
                 "NTP 601 (INSST)"),
    "lifts": ("Manipulación manual de cargas", "Ecuación de NIOSH "
              "revisada", "Límite de peso recomendado (LPR) e índice de "
              "levantamiento (IL = carga/LPR): hasta 1 riesgo limitado, de "
              "1 a 3 incremento moderado, más de 3 incremento acusado",
              "NTP 477; Guía técnica del INSST (2024); RD 487/1997"),
    "pushes": ("Empuje y tracción de cargas", "Criterios de la Guía "
               "técnica del INSST", "Fuerza inicial y mantenida frente a "
               "los límites de la ISO/TR 12295; altura de las manos y "
               "postura del tronco", "Guía técnica del INSST (2024)"),
    "screen": ("Trabajo con pantallas", "Criterios de la Guía técnica del "
               "INSST", "Distancia ojos-pantalla, postura del cuello, "
               "descansos visuales y pausas", "Guía técnica del INSST "
               "(2021); RD 488/1997"),
}


def methods_table(analysis: dict[str, Any]) -> str:
    """The methods used for the risks analysed, as a Markdown table."""
    people = people_of(analysis)
    used = []
    if any((p.get("summary") or {}).get("frames_with_person")
           for p in people):
        used.append("postures")
    if any((p.get("lifting") or {}).get("lifts") for p in people):
        used.append("lifts")
    if any((p.get("pushing") or {}).get("pushes") for p in people):
        used.append("pushes")
    if any(p.get("office") for p in people):
        used.append("screen")
    if not used:
        return ""
    return "\n".join(
        ["| Riesgo | Método | Criterio de valoración | Referencia |",
         "|---|---|---|---|"]
        + ["| " + " | ".join(METHODS[key]) + " |" for key in used])
