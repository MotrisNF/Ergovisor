"""The regulation behind each risk an analysis found.

Every warning of the analysis (a REBA cause, a note on a lift, the
lifting index, the frequency) belongs to a ``Topic``: a fixed search in
the prevention corpus, written with the words the texts themselves use
(the RD 487/1997, the Ley 31/1995, the INSST guide and its NTP). The
search only finds the passages; it does not write or cite anything yet.

A warning is matched by how its text starts, so the texts written by
``timeline.causes`` and ``Lifting`` must keep starting as in ``CAUSES``
(the tests check it).
"""
from collections.abc import Callable
import re
from typing import NamedTuple

from pydantic import BaseModel

from ..base_retriever import Retriever
from ..tokenizer_bm import Tokenizer
from .analysis import Analysis, PersonAnalysis
from .lifting import Lift
from .office import OfficeEvent, WarningCount
from .pushing import Push
from .reba import LEVELS
from .timeline import Event

# The search returns this many chunks, of which the best one of each file
# is kept: a topic gets its legal text and its technical criterion,
# instead of three chunks of the same guide.
CANDIDATES = 10
# A passage that scores less than this share of the best one is left
# out: on this corpus that drops the third document when only one or two
# speak of the topic. At 0.3 four topics still got a passage on something
# else (0.31-0.37 of the best: the neck got the load held away from the
# trunk, the push got REBA's table A...); the passages on the topic
# score 0.40 or more (2026-11-18).
RELATIVE = 0.4
# Characters of each passage shown in the report.
EXCERPT = 300
# A Markdown heading line, as the corpus writes them.
HEADING = re.compile(r"^#{1,6} +(.+)$", re.MULTILINE)


# Documents of each kind of work, by file name: a topic of one kind
# leaves the documents of the other out, so a guide on screens does not
# take the place of the law on loads (NTP 601, REBA, serves both).
LOADS = ("rd_487", "guia_tecnica_mmc", "ntp_477")
SCREENS = ("rd_488", "guia_tecnica_pvd")


class Topic(NamedTuple):
    """A kind of risk and how to look it up.

    Attributes:
        title: What it is, in a few words, for the report.
        query: The search, in the words of the regulation.
        avoid: Parts of file names left out of its search.
    """

    title: str
    query: str
    avoid: tuple[str, ...] = SCREENS


TOPICS: dict[str, Topic] = {
    "trunk_flexion": Topic(
        "Flexión o extensión del tronco",
        "movimiento de flexión del tronco postura TRONCO flexión extensión"),
    "trunk_twist": Topic(
        "Torsión o inclinación lateral del tronco",
        "torsión o inclinación del tronco movimiento de torsión TRONCO "
        "inclinación lateral"),
    "neck": Topic(
        "Postura del cuello",
        "CUELLO flexión extensión torsión inclinación lateral"),
    "legs": Topic(
        "Rodillas flexionadas o apoyo sobre una pierna",
        "PIERNAS soporte bilateral unilateral flexión de rodillas"),
    "arm": Topic(
        "Brazo elevado",
        "BRAZOS flexión abducción hombro elevado"),
    "activity": Topic(
        "Postura mantenida, repetida o brusca",
        "actividad partes del cuerpo estáticas movimientos repetitivos "
        "cambios posturales importantes posturas inestables"),
    "reba_level": Topic(
        "Nivel de riesgo y de acción REBA",
        "puntuación final REBA niveles de riesgo y acción intervención"),
    "distance": Topic(
        "Carga separada del cuerpo",
        "carga manipulada a distancia del tronco con torsión o "
        "inclinación distancia horizontal alejada del cuerpo",
        # REBA does not speak of it: its passage on how the method was
        # made scored 0.86 of the best.
        SCREENS + ("ntp_601",)),
    "height": Topic(
        "Altura de las manos fuera de la zona recomendada",
        "altura de los hombros y de los nudillos altura segura "
        "manipulación de la carga"),
    "weight": Topic(
        "Peso por encima del de referencia",
        "carga demasiado pesada pesos de referencia máximos sexo edad"),
    "frequency": Topic(
        "Frecuencia de manipulación",
        "esfuerzos físicos demasiado frecuentes frecuencia de la "
        "manipulación elevaciones por minuto duración del trabajo "
        "factor de frecuencia FM"),
    "lifting_index": Topic(
        "Índice de levantamiento (NIOSH)",
        "índice de levantamiento mayor que 1 riesgo limitado incremento "
        "moderado acusado del riesgo"),
    "employer": Topic(
        "Obligaciones del empresario en la manipulación de cargas",
        "obligaciones generales del empresario manipulación manual de "
        "cargas evaluación medidas"),
    "push_pull": Topic(
        "Empuje y tracción de cargas",
        "en caso de que la tarea implique un empuje y tracción de cargas "
        "se recomiendan los siguientes métodos de evaluación tablas de "
        "Snook y Ciriello fuerza inicial fuerza mantenida"),
    "push_height": Topic(
        "Altura de las manos al empujar o tirar",
        "se empuja o tracciona una carga con las manos por debajo de la "
        "altura de los nudillos o por encima del nivel de los hombros "
        "punto de aplicación de las fuerzas"),
    "push_force": Topic(
        "Fuerza de empuje y tracción",
        "empuje o tracción fuerza aplicada fuerza inicial y mantenida de "
        "referencia máximos que no se deben sobrepasar ISO/TR 12295"),
    "screen_distance": Topic(
        "Distancia de visión a la pantalla",
        "distancia de visualización pantalla colocación de la pantalla "
        "en ningún caso a menos de 300 mm", LOADS),
    "screen_neck": Topic(
        "Postura del cuello y altura de la pantalla",
        "altura de la pantalla parte superior altura de los ojos línea de "
        "visión horizontal 40° bajo la horizontal cuello", LOADS),
    "visual_rest": Topic(
        "Descansos visuales y fatiga visual",
        "fatiga visual regla 20-20-20 mirar lejos de la pantalla parpadear "
        "sequedad ocular", LOADS),
    "screen_breaks": Topic(
        "Pausas en el trabajo con pantallas",
        "pausas cambios de actividad trabajo continuado con pantalla "
        "pequeñas pausas cada 30 minutos cambios de postura",
        # REBA's "cambios posturales" is not about breaks.
        LOADS + ("ntp_601",)),
    "screen_employer": Topic(
        "Obligaciones del empresario en el trabajo con pantallas",
        "obligaciones generales del empresario equipos que incluyen "
        "pantallas de visualización evaluación de los riesgos", LOADS),
}
# A screen-work warning -> its topic.
OFFICE_TOPICS: dict[str, str] = {
    "too_close": "screen_distance",
    "neck_flexed": "screen_neck",
    "neck_extended": "screen_neck",
    "neck_turned": "screen_neck",
    "no_visual_rest": "visual_rest",
    "no_break": "screen_breaks",
}

# How a warning starts -> its topic.
CAUSES: tuple[tuple[str, str], ...] = (
    ("tronco girado", "trunk_twist"),
    ("tronco", "trunk_flexion"),
    ("cuello", "neck"),
    ("rodillas", "legs"),
    ("apoyo sobre una pierna", "legs"),
    ("brazo", "arm"),
    ("actividad", "activity"),
    ("carga separada", "distance"),
    ("manos por", "height"),
    ("peso superior", "weight"),
)
# A cause like "2 punto(s) de actividad (...)" starts with its number.
ACTIVITY = "punto(s) de actividad"
FREQUENCY = "levantamientos por minuto"


def topic_of(warning: str) -> str | None:
    """The topic of a warning of the analysis, or ``None`` if unknown."""
    if ACTIVITY in warning:
        return "activity"
    if FREQUENCY in warning:
        return "frequency"
    return next((topic for start, topic in CAUSES
                 if warning.startswith(start)), None)


class Passage(BaseModel):
    """A chunk of the corpus found for a topic.

    Attributes:
        file_path: The corpus file.
        first_character_index: Start of the chunk (included).
        last_character_index: End of the chunk (excluded).
        heading: The document and the section the chunk starts in.
        section: The last heading inside the chunk before the excerpt
            (an article that starts in the middle of the chunk); empty
            when the excerpt is under ``heading``'s section.
        excerpt: Its paragraph that shares the most words with the
            search, cut to ``EXCERPT`` characters.
        score: How well it matched the search.
    """

    file_path: str
    first_character_index: int
    last_character_index: int
    heading: str
    section: str = ""
    excerpt: str
    score: float


class Finding(BaseModel):
    """One kind of risk seen in the video, with its regulation.

    Attributes:
        topic: The key of the topic in ``TOPICS``.
        title: What it is.
        seen: What was seen of it: how often, and its measures (never
            when).
        query: The search made.
        passages: The best chunk of each file, best first.
    """

    topic: str
    title: str
    seen: list[str]
    query: str
    passages: list[Passage]


class RegulationReport(BaseModel):
    """The regulation for every risk of an analysis.

    Attributes:
        source: The video or camera analysed.
        index_dir: The index searched.
        findings: One per topic, in the order of the warnings.
        unmatched: Warnings with no topic (none, unless a new warning was
            added to the analysis and not to ``CAUSES``).
    """

    source: str
    index_dir: str
    findings: list[Finding]
    unmatched: list[str]


def duration(seconds: float) -> str:
    """A time in words: ``45 s``, ``3 min 20 s``."""
    total = round(seconds)
    if total < 60:
        return f"{total} s"
    minutes, rest = divmod(total, 60)
    return f"{minutes} min {rest} s" if rest else f"{minutes} min"


def _times(n: int) -> str:
    return f"{n} {'vez' if n == 1 else 'veces'}"


Warning = tuple[str | None, str]


def risk_warnings(person: PersonAnalysis) -> list[Warning]:
    """The moments of risk of a person: how many at each REBA level, and
    how often each cause was seen (never when, see
    ``analysis.PersonAnalysis``).

    Returns:
        ``(topic, warning)`` pairs.
    """
    found: list[Warning] = []
    for level, moments in sorted(person.moments_by_level.items(),
                                 reverse=True):
        _, risk, action = next((lv, r, a) for _, lv, r, a in LEVELS
                               if lv == level)
        found.append(("reba_level", f"{moments} momento(s) de riesgo "
                      f"{risk.lower()} (nivel de acción {level}): "
                      f"actuación {action.lower()}"))
    for count in person.risks:
        found.append((topic_of(count.cause),
                      f"{count.cause}: {_times(count.times)}, "
                      f"{duration(count.seconds)} en total (REBA máximo "
                      f"{count.max_score})"))
    return found


def lift_warnings(lift: Lift) -> list[Warning]:
    """A lift: its index when above 1 (or its limit of 0 kg), and its
    notes.

    Returns:
        ``(topic, warning)`` pairs.
    """
    found: list[Warning] = []
    if lift.index is not None and lift.index > 1:
        found.append(("lifting_index", f"{lift.kind}: índice "
                      f"{lift.index:.2f} ({lift.risk.lower()})"))
    notes = lift.notes
    if lift.beyond_limits and notes:
        # Its first note says why the limit is 0 kg.
        risk = f" ({lift.risk.lower()})" if lift.risk else ""
        found.append(("lifting_index", f"{lift.kind}: {notes[0]}{risk}"))
        notes = notes[1:]
    return found + [(topic_of(note), f"{lift.kind}: {note}")
                    for note in notes]


def push_warnings(push: Push) -> list[Warning]:
    """A push or a pull: what it was, and its warnings.

    Returns:
        ``(topic, warning)`` pairs.
    """
    found: list[Warning] = [("push_pull", f"{push.kind} de "
                             f"{push.metres:.1f} m")]
    for note in push.notes:
        topic = ("push_height" if note.startswith("manos")
                 else "trunk_twist" if note.startswith("tronco girado")
                 else "trunk_flexion")
        found.append((topic, f"{push.kind}: {note}"))
    return found


def office_warnings(count: WarningCount) -> list[Warning]:
    """A warning of the screen work, how often, with its topic.

    Returns:
        ``(topic, "title: times, time in all")`` pairs.
    """
    return [(OFFICE_TOPICS[count.kind],
             f"{count.title}: {_times(count.times)}, "
             f"{duration(count.seconds)} en total")]


def warnings(analysis: Analysis) -> list[Warning]:
    """Every warning of an analysis, with its topic, person by person.

    Args:
        analysis: What ``analyse`` found.

    Returns:
        ``(topic, warning)`` pairs. The moments of risk add how many
        there were at each REBA level and how often each cause was seen;
        a lift adds its notes, and its index when above 1; a task with
        lifts adds the duties of the employer and its notes (the
        frequency); screen work adds its warnings and the duties of the
        employer for screens. With several people, each warning starts
        with whose it is.
    """
    found: list[Warning] = []
    several = len(analysis.people) > 1
    for person in analysis.people:
        found += [(topic, f"Persona {person.person}: {text}"
                   if several else text)
                  for topic, text in _person_warnings(person)]
    return found


def _person_warnings(analysis: PersonAnalysis) -> list[Warning]:
    """The warnings of one person."""
    found = risk_warnings(analysis)
    report = analysis.lifting
    if report is not None and report.lifts:
        for lift in report.lifts:
            found += lift_warnings(lift)
        found += [(topic_of(note), note) for note in report.notes]
        found.append(("employer", f"{len(report.lifts)} levantamientos o "
                      "descensos de cargas"))
    pushing = analysis.pushing
    if pushing is not None and pushing.pushes:
        for push in pushing.pushes:
            found += push_warnings(push)
        found += [("push_force", note) for note in pushing.notes]
        found.append(("push_force", f"{len(pushing.pushes)} empujes o "
                      "arrastres (límites de fuerza de la guía)"))
    office = analysis.office
    if office is not None:
        for count in office.warnings:
            found += office_warnings(count)
        if office.warnings:
            found.append(("screen_employer",
                          f"{office.seconds_at_screen / 60:.0f} min de "
                          "trabajo frente a la pantalla"))
    return found


def _best_line(text: str, query: str) -> tuple[int, str]:
    """Where the line of ``text`` sharing the most words with ``query``
    starts, and that line in one line of text (the first on a tie)."""
    tokenizer = Tokenizer()
    wanted = set(tokenizer.tokenize_query(query))
    best, best_start, best_score = "", 0, -1
    start = 0
    for line in text.split("\n"):
        flat = " ".join(line.split())
        if flat:
            score = len(wanted & set(tokenizer.tokenize(flat)))
            if score > best_score:
                best, best_start, best_score = flat, start, score
        start += len(line) + 1
    return best_start, best


def excerpt(text: str, query: str) -> str:
    """The paragraph of a chunk that shares the most words with a search.

    A chunk of 2000 characters often starts with the end of the previous
    section; its paragraph about the topic is what a reader needs to see.

    Args:
        text: The chunk.
        query: The search.

    Returns:
        That paragraph in one line, cut to ``EXCERPT`` characters; the
        first one on a tie.
    """
    return _best_line(text, query)[1][:EXCERPT]


def section_before(text: str, position: int) -> str:
    """The last Markdown heading of ``text`` before ``position``, or an
    empty string.

    Args:
        text: A chunk.
        position: Where its excerpt starts.

    Returns:
        The heading's title, without its ``#``.
    """
    found = HEADING.findall(text[:position])
    return found[-1].strip() if found else ""


def search_topic(retriever: Retriever, query: str, k: int,
                 avoid: tuple[str, ...] = ()) -> list[Passage]:
    """The best chunk of each file for a search, ``k`` files at most.

    Args:
        retriever: The index of the prevention corpus.
        query: The topic's search.
        k: How many files.
        avoid: Parts of file names whose chunks are left out.

    Returns:
        The passages, best first; none scoring below ``RELATIVE`` times
        the first one.
    """
    passages: list[Passage] = []
    files: set[str] = set()
    for hit in retriever.search(query, max(CANDIDATES, k)):
        chunk = hit.chunk
        if chunk.file_path in files or any(part in chunk.file_path
                                           for part in avoid):
            continue
        if passages and hit.score < RELATIVE * passages[0].score:
            break
        files.add(chunk.file_path)
        start, line = _best_line(chunk.text, query)
        passages.append(Passage(
            file_path=chunk.file_path,
            first_character_index=chunk.first_character_index,
            last_character_index=chunk.last_character_index,
            heading=chunk.heading,
            section=section_before(chunk.text, start),
            excerpt=line[:EXCERPT],
            score=round(hit.score, 4)))
        if len(passages) == k:
            break
    return passages


def find_regulation(analysis: Analysis, retriever: Retriever,
                    index_dir: str, k: int = 3,
                    on_finding: Callable[[Finding], None] | None = None,
                    searched: dict[str, list[Passage]] | None = None
                    ) -> RegulationReport:
    """Look up the regulation for every kind of risk an analysis found.

    Each topic is searched once, however many times it was seen.

    Args:
        analysis: What ``analyse`` found.
        retriever: The index of the prevention corpus.
        index_dir: Where that index is, for the report.
        k: Passages per topic (each from a different file).
        on_finding: Called with each finding as soon as it is searched.
        searched: Passages already found per topic (by ``LiveRegulation``
            during the analysis), not searched again.

    Returns:
        The findings, in the order their topic was first seen.
    """
    searched = searched if searched is not None else {}
    seen: dict[str, list[str]] = {}
    unmatched: list[str] = []
    for topic, text in warnings(analysis):
        if topic is None:
            unmatched.append(text)
        else:
            seen.setdefault(topic, []).append(text)
    findings = []
    for topic, texts in seen.items():
        title, query, avoid = TOPICS[topic]
        if topic not in searched:
            searched[topic] = search_topic(retriever, query, k, avoid)
        finding = Finding(topic=topic, title=title, seen=texts, query=query,
                          passages=searched[topic])
        findings.append(finding)
        if on_finding is not None:
            on_finding(finding)
    return RegulationReport(source=analysis.source, index_dir=index_dir,
                            findings=findings, unmatched=unmatched)


class LiveRegulation:
    """The regulation of each risk as soon as it is seen.

    For a live camera: each moment of risk or lift that ends gives the
    findings of the topics not shown yet, searched there and then. A
    topic is searched and shown once; what was searched is kept in
    ``searched`` for the final report.

    Attributes:
        retriever: The index of the prevention corpus.
        k: Passages per topic.
        searched: Passages found per topic so far.
    """

    def __init__(self, retriever: Retriever, k: int = 3) -> None:
        self.retriever = retriever
        self.k = k
        self.searched: dict[str, list[Passage]] = {}

    def _new(self, found: list[Warning]) -> list[Finding]:
        """Findings for the topics of ``found`` not searched yet."""
        texts: dict[str, list[str]] = {}
        for topic, text in found:
            if topic is not None and topic not in self.searched:
                texts.setdefault(topic, []).append(text)
        findings = []
        for topic, seen in texts.items():
            title, query, avoid = TOPICS[topic]
            self.searched[topic] = search_topic(self.retriever, query,
                                                self.k, avoid)
            findings.append(Finding(topic=topic, title=title, seen=seen,
                                    query=query,
                                    passages=self.searched[topic]))
        return findings

    def event(self, event: Event) -> list[Finding]:
        """The new findings of a moment of risk that just ended."""
        return self._new(
            [("reba_level", f"REBA {event.score}: riesgo "
              f"{event.risk.lower()}, actuación {event.action.lower()}")]
            + [(topic_of(kind), kind) for kind in event.kinds])

    def office(self, event: OfficeEvent) -> list[Finding]:
        """The new findings of a screen-work warning; the first one also
        brings the duties of the employer for screens."""
        return self._new([
            (OFFICE_TOPICS[event.kind], f"{event.title}: {event.detail}"),
            ("screen_employer", "trabajo frente a la pantalla")])

    def push(self, push: Push) -> list[Finding]:
        """The new findings of a push or pull that just ended; the first
        one also brings the guide's force limits."""
        return self._new(push_warnings(push) + [
            ("push_force", "límites de fuerza de la guía")])

    def lift(self, lift: Lift) -> list[Finding]:
        """The new findings of a lift that just ended; the first one also
        brings the duties of the employer."""
        return self._new(lift_warnings(lift) + [
            ("employer", "levantamientos o descensos de cargas")])


def describe_finding(finding: Finding) -> str:
    """A few lines for a finding: what, what was seen, and where to
    read."""
    lines = [finding.title] + [f"  · {seen}" for seen in finding.seen]
    for passage in finding.passages:
        name = passage.file_path.rsplit("/", 1)[-1]
        lines.append(f"  - {name} [{passage.first_character_index}:"
                     f"{passage.last_character_index}] "
                     f"{passage.excerpt[:100]}...")
    return "\n".join(lines)
