"""Tests for the regulation of the risks an analysis found."""
from typing import Any

import pytest

from src.base_retriever import Retriever
from src.models import Chunk, SearchHit
from src.posture.analysis import Analysis
from src.posture.lifting import Lift
from src.posture.regulation import (RELATIVE, TOPICS, LiveRegulation,
                                    describe_finding, excerpt,
                                    find_regulation, search_topic, topic_of,
                                    warnings)
from src.posture.timeline import Event


class FakeRetriever(Retriever):
    """Returns the same hits for any search, and records the searches."""

    def __init__(self, hits: list[tuple[str, float, str]]) -> None:
        self.hits = [
            SearchHit(chunk=Chunk(file_path=path, first_character_index=i,
                                  last_character_index=i + len(text),
                                  text=text, heading=f"Doc {path}"),
                      score=score, chunk_id=i)
            for i, (path, score, text) in enumerate(hits)]
        self.queries: list[str] = []

    def search(self, query: str, k: int) -> list[SearchHit]:
        self.queries.append(query)
        return self.hits[:k]


def make_analysis(risks: list[dict[str, Any]] | None = None,
                  lifts: list[dict[str, Any]] | None = None,
                  notes: list[str] | None = None) -> Analysis:
    """An analysis of one person with these causes of risk (each in one
    moment of level 3) and lifts."""
    person: dict[str, Any] = {
        "person": 1, "summary": {}, "risks": risks or [],
        "moments": 1 if risks else 0,
        "moments_by_level": {3: 1} if risks else {}}
    if lifts is not None:
        person["lifting"] = {
            "load_kg": 5.0, "grip": "bueno", "duration": "larga",
            "lifts_per_minute": 2.0, "reference_kg": 25.0, "lifts": lifts,
            "max_index": None, "notes": notes or [], "assumptions": []}
    return Analysis.model_validate({
        "source": "video.mp4", "created": "2026-11-17", "task": {},
        "people": [person]})


def risk(cause: str, times: int = 1) -> dict[str, Any]:
    return {"cause": cause, "times": times, "seconds": 5.0 * times,
            "max_score": 9}


def event(start: float, kinds: list[str]) -> dict[str, Any]:
    return {"start": start, "end": start + 5, "score": 9, "level": 3,
            "risk": "Alto", "action": "Necesaria pronto", "causes": kinds,
            "kinds": kinds}


def lift(start: float, index: float | None,
         notes: list[str]) -> dict[str, Any]:
    place = {"second": start, "v": 30.0, "h": 40.0, "a": 0.0}
    return {"kind": "levantamiento", "start": place,
            "end": {**place, "second": start + 3}, "d": 50.0, "limit": 8.0,
            "index": index, "risk": "Incremento moderado del riesgo",
            "factors": {}, "notes": notes}


class TestTopicOf:
    """Every warning the analysis writes has its topic."""

    @pytest.mark.parametrize(("warning", "topic"), [
        # timeline.causes
        ("tronco flexionado 77°", "trunk_flexion"),
        ("tronco en extensión 25°", "trunk_flexion"),
        ("tronco inclinado 16°", "trunk_flexion"),
        ("tronco girado o inclinado de lado", "trunk_twist"),
        ("cuello flexionado 25°", "neck"),
        ("cuello en extensión", "neck"),
        ("rodillas flexionadas 90°", "legs"),
        ("apoyo sobre una pierna", "legs"),
        ("brazo izquierdo elevado 70°", "arm"),
        ("2 punto(s) de actividad (postura mantenida, repetida o brusca)",
         "activity"),
        # timeline.kinds (the bands kept in a report)
        ("tronco flexionado más de 60°", "trunk_flexion"),
        ("rodillas flexionadas 30-60°", "legs"),
        ("brazo elevado más de 90°", "arm"),
        ("actividad: postura mantenida, repetida o brusca", "activity"),
        # Lifting notes
        ("carga separada del cuerpo (40 cm; la guía recomienda 25 cm o "
         "menos)", "distance"),
        ("manos por encima de la cabeza en el origen: nunca se deben "
         "elevar cargas ahí", "height"),
        ("manos por encima de los hombros en el destino", "height"),
        ("manos por debajo de los nudillos en el origen (30 cm del suelo)",
         "height"),
        ("peso superior al máximo de referencia de 25 kg (guía del INSST)",
         "weight"),
        ("16.0 levantamientos por minuto: más de 12, el límite de la guía "
         "del INSST para una tarea moderada", "frequency"),
    ])
    def test_known(self, warning: str, topic: str) -> None:
        assert topic_of(warning) == topic

    def test_unknown(self) -> None:
        assert topic_of("algo nuevo") is None

    def test_every_topic_has_a_query(self) -> None:
        assert all(t.title and t.query for t in TOPICS.values())


class TestWarnings:
    """The warnings of an analysis: how often, never when."""

    def test_risks_give_level_and_causes(self) -> None:
        found = warnings(make_analysis([risk("cuello en extensión", 3)]))
        assert [t for t, _ in found] == ["reba_level", "neck"]
        assert found[0][1] == ("1 momento(s) de riesgo alto (nivel de "
                               "acción 3): actuación necesaria pronto")
        assert found[1][1] == ("cuello en extensión: 3 veces, 15 s en "
                               "total (REBA máximo 9)")

    def test_lift_index_only_above_one(self) -> None:
        found = warnings(make_analysis(lifts=[lift(1, 1.0, []),
                                              lift(9, 1.4, [])]))
        topics = [t for t, _ in found]
        assert topics.count("lifting_index") == 1
        assert "employer" in topics

    def test_lift_beyond_the_equation(self) -> None:
        """A limit of 0 kg is a lifting-index warning, with its reason."""
        beyond = {**lift(4, None, ["límite de peso recomendado 0 kg por la "
                                   "carga a más de 63 cm del cuerpo: no es "
                                   "aceptable con ninguna carga"]),
                  "limit": 0.0, "beyond_limits": True,
                  "risk": "Incremento acusado del riesgo"}
        found = warnings(make_analysis(lifts=[beyond]))
        index = [text for topic, text in found if topic == "lifting_index"]
        assert len(index) == 1 and "63 cm" in index[0]
        assert "acusado" in index[0]

    def test_no_lifts_no_employer(self) -> None:
        assert warnings(make_analysis(lifts=[])) == []

    def test_order_and_no_times(self) -> None:
        """Postures first, then the lifts; no time in any warning."""
        analysis = make_analysis(
            [risk("tronco flexionado más de 60°")],
            [lift(10, 0.5, ["carga separada del cuerpo (40 cm)"])],
            ["16.0 levantamientos por minuto"])
        found = warnings(analysis)
        assert [t for t, _ in found] == ["reba_level", "trunk_flexion",
                                         "distance", "frequency",
                                         "employer"]
        assert not any("[" in text for _, text in found)


class TestSearchTopic:
    """The best chunk of each file."""

    def test_one_per_file(self) -> None:
        retriever = FakeRetriever([("a.md", 10, "x"), ("a.md", 9, "y"),
                                   ("b.md", 8, "z")])
        passages = search_topic(retriever, "q", 3)
        assert [p.file_path for p in passages] == ["a.md", "b.md"]

    def test_k_files(self) -> None:
        retriever = FakeRetriever([(f"{i}.md", 10 - i, "x")
                                   for i in range(5)])
        assert len(search_topic(retriever, "q", 2)) == 2

    def test_weak_passages_left_out(self) -> None:
        retriever = FakeRetriever([("a.md", 10, "x"),
                                   ("b.md", 10 * RELATIVE, "y"),
                                   ("c.md", 10 * RELATIVE - 0.01, "z")])
        assert [p.file_path for p in search_topic(retriever, "q", 3)] == [
            "a.md", "b.md"]

    def test_keeps_place_and_heading(self) -> None:
        retriever = FakeRetriever([("a.md", 10, "texto")])
        passage = search_topic(retriever, "q", 1)[0]
        assert (passage.first_character_index,
                passage.last_character_index) == (0, 5)
        assert passage.heading == "Doc a.md"

    def test_nothing_found(self) -> None:
        assert search_topic(FakeRetriever([]), "q", 3) == []


class TestSection:
    """The article an excerpt is in, when it starts inside the chunk."""

    def test_heading_inside_chunk(self) -> None:
        retriever = FakeRetriever([(
            "rd.md", 10, "Se autoriza al Ministro.\n\n## ANEXO\n\n"
            "## Factores de riesgo\n\nCon torsión del tronco.")])
        passage = search_topic(retriever, "torsión del tronco", 1)[0]
        assert passage.section == "Factores de riesgo"
        assert passage.excerpt == "Con torsión del tronco."

    def test_no_heading_before(self) -> None:
        retriever = FakeRetriever([("a.md", 10, "torsión\n## Luego")])
        assert search_topic(retriever, "torsión", 1)[0].section == ""


class TestExcerpt:
    """The paragraph that matches the search."""

    def test_best_paragraph(self) -> None:
        text = "Se autoriza al Ministro.\n\nCuando hay torsión del tronco."
        assert excerpt(text, "torsión del tronco") == (
            "Cuando hay torsión del tronco.")

    def test_tie_takes_first(self) -> None:
        assert excerpt("uno\ndos", "nada") == "uno"

    def test_cut_and_one_line(self) -> None:
        text = "tronco " * 100
        assert len(excerpt(text, "tronco")) <= 300
        assert "\n" not in excerpt("a\n\n", "a")

    def test_empty(self) -> None:
        assert excerpt("", "a") == ""


class TestFindRegulation:
    """One search per topic, however often it was seen."""

    def test_one_finding_per_topic(self) -> None:
        analysis = make_analysis([risk("cuello en extensión"),
                                  risk("cuello flexionado más de 20°")])
        retriever = FakeRetriever([("a.md", 10, "x")])
        report = find_regulation(analysis, retriever, "idx")
        assert [f.topic for f in report.findings] == ["reba_level", "neck"]
        assert len(report.findings[1].seen) == 2
        assert retriever.queries == [TOPICS["reba_level"].query,
                                     TOPICS["neck"].query]
        assert report.index_dir == "idx"

    def test_unmatched_kept(self) -> None:
        analysis = make_analysis([risk("algo nuevo")])
        report = find_regulation(analysis, FakeRetriever([]), "idx")
        assert report.unmatched == [
            "algo nuevo: 1 vez, 5 s en total (REBA máximo 9)"]

    def test_on_finding_called(self) -> None:
        got: list[str] = []
        find_regulation(make_analysis([risk("cuello en extensión")]),
                        FakeRetriever([]), "idx",
                        on_finding=lambda f: got.append(f.topic))
        assert got == ["reba_level", "neck"]

    def test_describe(self) -> None:
        report = find_regulation(make_analysis([risk("cuello en extensión")]),
                                 FakeRetriever([("d/a.md", 1, "texto")]),
                                 "idx")
        text = describe_finding(report.findings[0])
        assert text.startswith("Nivel de riesgo y de acción REBA\n  · 1 "
                               "momento(s) de riesgo alto")
        assert "a.md [0:5] texto" in text


class TestLiveRegulation:
    """Each topic searched and shown once, as soon as it is seen."""

    def test_new_topics_only(self) -> None:
        retriever = FakeRetriever([("a.md", 10, "x")])
        live = LiveRegulation(retriever, 2)
        first = live.event(Event(**event(1, ["cuello en extensión"])))
        assert [f.topic for f in first] == ["reba_level", "neck"]
        second = live.event(Event(**event(9, [
            "cuello flexionado más de 20°", "tronco inclinado hasta 20°"])))
        assert [f.topic for f in second] == ["trunk_flexion"]
        assert len(retriever.queries) == 3

    def test_lift_brings_employer_once(self) -> None:
        live = LiveRegulation(FakeRetriever([]))
        found = live.lift(Lift(**lift(1, 1.5, ["carga separada (40 cm)"])))
        assert [f.topic for f in found] == ["lifting_index", "distance",
                                            "employer"]
        assert live.lift(Lift(**lift(9, 0.5, []))) == []

    def test_final_report_reuses_searches(self) -> None:
        """The report at the end does not search the topics again."""
        retriever = FakeRetriever([("a.md", 10, "x")])
        live = LiveRegulation(retriever)
        live.event(Event(**event(1, [])))
        analysis = make_analysis([], [lift(5, 0.5, [])],
                                 ["16.0 levantamientos por minuto"])
        analysis.people[0].moments_by_level = {3: 1}
        report = find_regulation(analysis, retriever, "idx",
                                 searched=live.searched)
        assert [f.topic for f in report.findings] == [
            "reba_level", "frequency", "employer"]
        assert retriever.queries.count(TOPICS["reba_level"].query) == 1
        assert set(live.searched) == {"reba_level", "frequency", "employer"}
