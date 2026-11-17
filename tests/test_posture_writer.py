"""Tests for the written reports. No test calls Gemini: the model is a
fake that records the prompts and answers a fixed text."""
from io import BytesIO
import json
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient
import httpx
import pytest

from src.exceptions import RagError
from src.posture import writer
from src.posture.docx_export import to_docx
from src.posture.writer import (Written, check_citations, collect_sources,
                                describe_analysis, document_name, gemini,
                                write_report, write_summary)

ANALYSIS: dict[str, Any] = {
    "source": "camera rtsp://***@h/s", "created": "2026-11-17T10:00:00+00:00",
    "summary": {"seconds": 60.0, "frames": 600, "frames_with_person": 600,
                "seconds_by_level": {"0": 10.0, "2": 30.0, "3": 20.0},
                "max_score": 9, "max_level": 3},
    "events": [{"start": 5.0, "end": 9.0, "score": 9, "level": 3,
                "risk": "Alto", "action": "Necesaria pronto",
                "causes": ["tronco flexionado 70°"]}],
    "unmeasured": ["hombro elevado"],
    "lifting": None,
    "office": None,
}
REGULATION: dict[str, Any] = {"findings": [
    {"topic": "trunk_flexion", "seen": ["a", "b"], "passages": [
        {"file_path": "corpus/rd_487_1997_cargas.md",
         "first_character_index": 0, "last_character_index": 20,
         "section": "ANEXO", "excerpt": "x"},
        {"file_path": "corpus/ntp_601.md", "first_character_index": 5,
         "last_character_index": 15, "section": "", "excerpt": "y"}]},
    {"topic": "neck", "seen": ["c"], "passages": [
        {"file_path": "corpus/ntp_601.md", "first_character_index": 5,
         "last_character_index": 15, "section": "", "excerpt": "y"}]},
]}
TEXTS = {"corpus/rd_487_1997_cargas.md": "Torsión o flexión del tronco.  X",
         "corpus/ntp_601.md": "0123 TRONCO 20° flexión"}


class FakeLlm:
    """Answers ``answer`` and keeps what it was asked."""

    def __init__(self, answer: str = "## 1. Objeto\nTexto [1] y [9].") -> None:
        self.answer = answer
        self.calls: list[tuple[str, str]] = []

    def __call__(self, system: str, prompt: str) -> str:
        self.calls.append((system, prompt))
        return self.answer


class TestCitations:
    """Only citations to given passages stay."""

    def test_unknown_removed(self) -> None:
        text, cited, unknown = check_citations("a [1] b [7] c [2, 7]", {1, 2})
        assert text == "a [1] b  c [2]"
        assert cited == [1, 2]
        assert unknown == [7]

    def test_none(self) -> None:
        assert check_citations("sin citas", {1}) == ("sin citas", [], [])


class TestPrompt:
    """What the model is given."""

    def test_sources_numbered_once(self) -> None:
        sources = collect_sources([REGULATION], TEXTS.__getitem__)
        assert [s.number for s, _ in sources] == [1, 2]
        # The NTP passage was seen for two topics: it comes first.
        assert sources[0][0].file_path == "corpus/ntp_601.md"
        assert sources[0][1] == "TRONCO 20°"

    def test_document_names(self) -> None:
        assert "487/1997" in document_name("data/raw/prl/rd_487_1997_x.md")
        assert document_name("data/raw/prl/otro.md") == "otro"

    def test_analysis_described(self) -> None:
        text = describe_analysis(ANALYSIS)
        assert "REBA máxima: 9" in text
        assert "tronco flexionado 70°" in text
        assert "pantalla" not in text.lower()

    def test_office_described(self) -> None:
        office = {"seconds_at_screen": 120.0, "seconds_with_face": 100.0,
                  "distance_cm": 48.0, "closest_cm": 30.0,
                  "seconds_neck_flexed": 50.0, "seconds_neck_extended": 0.0,
                  "seconds_neck_turned": 0.0,
                  "seconds_shoulders_raised": 0.0, "visual_rests": 0,
                  "longest_without_rest": 120.0, "breaks": 0,
                  "longest_without_break": 120.0, "blinks_per_minute": 12.0,
                  "events": [], "assumptions": ["supuesto"]}
        text = describe_analysis({**ANALYSIS, "office": office})
        assert "48.0 cm" in text and "50 %" in text
        assert "no permite valorar si las pausas" in text


class TestWrite:
    """One call, citations checked, references added here."""

    def test_report(self) -> None:
        llm = FakeLlm()
        written = write_report("x", ANALYSIS, REGULATION, llm, "fake",
                               TEXTS.__getitem__)
        assert len(llm.calls) == 1
        assert "[1] NTP 601" in llm.calls[0][1]
        assert written.markdown.startswith("# Informe detallado")
        assert "[9]" not in written.markdown
        assert written.uncited == [9]
        assert "## Referencias" in written.markdown
        assert [s.number for s in written.sources] == [1]
        assert written.kind == "report" and written.model == "fake"

    def test_references_grouped_by_document(self) -> None:
        regulation = {"findings": [{"topic": "t", "seen": ["a"], "passages": [
            {"file_path": "corpus/ntp_601.md", "first_character_index": i,
             "last_character_index": i + 3, "section": "", "excerpt": "e"}
            for i in (0, 5)]}]}
        written = write_report("x", ANALYSIS, regulation,
                               FakeLlm("Texto [1] y [2]."), "fake",
                               TEXTS.__getitem__)
        references = written.markdown.split("## Referencias")[1]
        assert references.count("NTP 601") == 1
        assert "[1, 2] NTP 601" in references

    def test_no_load_said(self) -> None:
        lifting: dict[str, Any] = {"lifts": [], "load_kg": None}
        text = describe_analysis({**ANALYSIS, "lifting": {
            **lifting, "lifts": [{"start": {"second": 1.0},
                                  "kind": "levantamiento", "d": 40.0,
                                  "limit": 9.1, "index": None, "risk": "",
                                  "notes": []}],
            "grip": "bueno", "duration": "larga", "reference_kg": 20.0,
            "lifts_per_minute": 1.0, "notes": [], "assumptions": []}})
        assert "no se puede calcular el índice" in text

    def test_summary(self) -> None:
        llm = FakeLlm("Resumen [2].")
        written = write_summary([("a", ANALYSIS), ("b", ANALYSIS)],
                                [REGULATION, None], "01/11/2026",
                                "17/11/2026", llm, "fake",
                                TEXTS.__getitem__)
        assert len(llm.calls) == 1
        assert "Análisis 2: b" in llm.calls[0][1]
        assert "Inspección de Trabajo" in llm.calls[0][1]
        assert written.reports == ["a", "b"]

    def test_summary_needs_reports(self) -> None:
        with pytest.raises(RagError):
            write_summary([], [], "a", "b", FakeLlm(), "fake",
                          TEXTS.__getitem__)


class FakeResponse:
    def __init__(self, status: int, data: Any) -> None:
        self.status_code = status
        self._data = data
        self.text = json.dumps(data)

    def json(self) -> Any:
        return self._data


class TestGemini:
    """The REST call, without the network."""

    def test_text_without_thoughts(self,
                                   monkeypatch: pytest.MonkeyPatch) -> None:
        sent: dict[str, Any] = {}

        def post(url: str, **kwargs: Any) -> FakeResponse:
            sent.update(url=url, **kwargs)
            return FakeResponse(200, {"candidates": [{"content": {"parts": [
                {"text": "pensando", "thought": True}, {"text": "Hola"}]}}]})

        monkeypatch.setattr(httpx, "post", post)
        assert gemini("m", key="k")("s", "p") == "Hola"
        assert sent["url"].endswith("/models/m:generateContent")
        assert sent["headers"] == {"x-goog-api-key": "k"}
        assert "k" not in sent["url"]

    def test_quota(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(httpx, "post",
                            lambda url, **kw: FakeResponse(429, {}))
        with pytest.raises(RagError, match="cuota"):
            gemini("m", key="k")("s", "p")

    def test_busy_then_answers(self,
                               monkeypatch: pytest.MonkeyPatch) -> None:
        answers = [FakeResponse(503, {}), FakeResponse(200, {
            "candidates": [{"content": {"parts": [{"text": "Hola"}]}}]})]
        monkeypatch.setattr(httpx, "post",
                            lambda url, **kw: answers.pop(0))
        assert gemini("m", key="k", waits=(0.0,))("s", "p") == "Hola"

    def test_busy_gives_up(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = []

        def post(url: str, **kw: Any) -> FakeResponse:
            calls.append(url)
            return FakeResponse(503, {"error": {"message": "busy"}})

        monkeypatch.setattr(httpx, "post", post)
        with pytest.raises(RagError, match="503"):
            gemini("m", key="k", waits=(0.0, 0.0))("s", "p")
        assert len(calls) == 3

    def test_no_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(writer.KEY_ENV, raising=False)
        with pytest.raises(RagError, match="GEMINI_API_KEY"):
            gemini("m")("s", "p")

    def test_blocked(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(httpx, "post", lambda url, **kw: FakeResponse(
            200, {"candidates": []}))
        with pytest.raises(RagError, match="no devolvió"):
            gemini("m", key="k")("s", "p")


class TestDocx:
    """The Markdown subset becomes a Word document."""

    def test_structure(self) -> None:
        from docx import Document

        data = to_docx("# Título\n\n## 1. Objeto\nTexto **negrita** [1].\n\n"
                       "- uno\n- dos\n\n| A | B |\n|---|---|\n| 1 | 2 |\n")
        document = Document(BytesIO(data))
        texts = [p.text for p in document.paragraphs]
        assert "Título" in texts and "1. Objeto" in texts
        assert "Texto negrita [1]." in texts
        assert any(p.style is not None and p.style.name == "List Bullet"
                   for p in document.paragraphs)
        assert document.tables[0].cell(1, 1).text == "2"

    def test_no_colours(self) -> None:
        """Every heading is black: no colour in the document."""
        from docx import Document
        from docx.shared import RGBColor

        document = Document(BytesIO(to_docx("# T\n\n## A\n\n### B\nx")))
        for paragraph in document.paragraphs:
            style = paragraph.style
            assert style is not None
            colour = style.font.color.rgb
            assert colour in (None, RGBColor(0, 0, 0)), style.name


def write_analysis(folder: Path, name: str, created: str) -> None:
    (folder / f"{name}.json").write_text(json.dumps(
        {**ANALYSIS, "created": created}))
    (folder / f"{name}_regulation.json").write_text(json.dumps(REGULATION))


@pytest.fixture
def setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch
          ) -> tuple[TestClient, FakeLlm, Path]:
    from src.posture.sessions import SessionManager
    from src.webapp import WebApp

    monkeypatch.setattr(WebApp, "_read_corpus",
                        staticmethod(lambda path: TEXTS.get(path, "")))
    reports = tmp_path / "reports"
    reports.mkdir()
    write_analysis(reports, "a", "2026-11-10T10:00:00+00:00")
    write_analysis(reports, "b", "2026-11-17T10:00:00+00:00")
    llm = FakeLlm()
    app = WebApp(SessionManager(), str(reports), str(tmp_path / "up"), "",
                 llm=llm, llm_model="fake")
    return TestClient(app.app), llm, reports


class TestApi:
    """Writing through the interface spends one call and is kept."""

    def test_write_once(self, setup: tuple[TestClient, FakeLlm,
                                           Path]) -> None:
        client, llm, reports = setup
        first = client.post("/api/reports/a/written")
        assert first.status_code == 200
        client.post("/api/reports/a/written")
        assert len(llm.calls) == 1
        assert client.get("/api/reports/a").json()["written"]["model"] \
            == "fake"
        client.post("/api/reports/a/written?again=true")
        assert len(llm.calls) == 2
        # The written report is not a report of its own.
        names = [r["name"] for r in client.get("/api/reports").json()]
        assert sorted(names) == ["a", "b"]

    def test_docx(self, setup: tuple[TestClient, FakeLlm, Path]) -> None:
        client, _, _ = setup
        assert client.get("/api/reports/a/docx").status_code == 404
        client.post("/api/reports/a/written")
        got = client.get("/api/reports/a/docx")
        assert got.status_code == 200
        assert got.content[:2] == b"PK"
        assert "informe_a.docx" in got.headers["content-disposition"]

    def test_delete_report_deletes_written(
            self, setup: tuple[TestClient, FakeLlm, Path]) -> None:
        client, _, reports = setup
        client.post("/api/reports/a/written")
        client.delete("/api/reports/a")
        assert not (reports / "written" / "a.json").exists()

    def test_summary_by_dates(self, setup: tuple[TestClient, FakeLlm,
                                                 Path]) -> None:
        client, llm, _ = setup
        got = client.post("/api/summaries", json={"start": "2026-11-15",
                                                  "end": "2026-11-17"})
        assert got.status_code == 200
        assert got.json()["reports"] == ["b"]
        name = got.json()["name"]
        assert [s["name"] for s in client.get("/api/summaries").json()] \
            == [name]
        assert client.get(f"/api/summaries/{name}").json()["title"]
        assert client.get(f"/api/summaries/{name}/docx").status_code == 200
        assert client.delete(f"/api/summaries/{name}").status_code == 200
        assert client.get("/api/summaries").json() == []
        assert len(llm.calls) == 1

    @pytest.mark.parametrize(("start", "end"), [("2026-12-01", "2026-12-31"),
                                                ("2026-11-17", "2026-11-01")])
    def test_summary_rejected(self, setup: tuple[TestClient, FakeLlm, Path],
                              start: str, end: str) -> None:
        client, llm, _ = setup
        got = client.post("/api/summaries", json={"start": start,
                                                  "end": end})
        assert got.status_code == 422
        assert llm.calls == []

    def test_without_model(self, tmp_path: Path) -> None:
        from src.posture.sessions import SessionManager
        from src.webapp import WebApp

        write_analysis(tmp_path, "a", "2026-11-10T10:00:00+00:00")
        client = TestClient(WebApp(SessionManager(), str(tmp_path),
                                   str(tmp_path / "up"), "").app)
        assert client.get("/api/llm").json()["available"] is False
        assert client.post("/api/reports/a/written").status_code == 503

    def test_summary_not_found(self, setup: tuple[TestClient, FakeLlm,
                                                  Path]) -> None:
        client, _, _ = setup
        assert client.get("/api/summaries/..%2Fa").status_code == 404
        assert client.get("/api/summaries/nope").status_code == 404


def test_written_round_trip(tmp_path: Path) -> None:
    written = Written(kind="report", title="t", model="m", created="c",
                      markdown="x", sources=[])
    writer.save(written, tmp_path / "w" / "a.json")
    assert writer.load(tmp_path / "w" / "a.json") == written
    assert writer.load(tmp_path / "none.json") is None


class TestAutomatic:
    """Each analysis saved by the interface gets its detailed report
    written by itself, with one call."""

    def test_written_when_saved(self, tmp_path: Path,
                                monkeypatch: pytest.MonkeyPatch) -> None:
        import time

        from src.webapp import WebApp
        from test_webapp import manager

        monkeypatch.setattr(WebApp, "_read_corpus",
                            staticmethod(lambda path: TEXTS.get(path, "")))
        reports = tmp_path / "reports"
        reports.mkdir()
        # The fake session saves "report_<source>".
        write_analysis(reports, "report_0", "2026-11-17T10:00:00+00:00")
        sessions = manager([("status", "running")])
        llm = FakeLlm()
        client = TestClient(WebApp(sessions, str(reports),
                                   str(tmp_path / "up"), "", llm=llm,
                                   llm_model="fake").app)
        client.post("/api/sessions", json={"kind": "camera",
                                           "source": "0"})
        deadline = time.monotonic() + 5
        while not (reports / "written" / "report_0.json").exists():
            assert time.monotonic() < deadline, "not written"
            time.sleep(0.05)
        deadline = time.monotonic() + 2
        while client.get("/api/reports/report_0").json()["writing"]:
            assert time.monotonic() < deadline
            time.sleep(0.05)
        assert len(llm.calls) == 1

    def test_failure_kept(self, tmp_path: Path) -> None:
        import time

        from src.webapp import WebApp
        from test_webapp import manager

        reports = tmp_path / "reports"
        reports.mkdir()
        write_analysis(reports, "report_0", "2026-11-17T10:00:00+00:00")

        def busy(system: str, prompt: str) -> str:
            raise RagError("Gemini respondió 503: ocupado")

        client = TestClient(WebApp(manager([("status", "running")]),
                                   str(reports), str(tmp_path / "up"), "",
                                   llm=busy, llm_model="fake").app)
        client.post("/api/sessions", json={"kind": "camera",
                                           "source": "0"})
        deadline = time.monotonic() + 5
        while client.get("/api/reports/report_0").json()["writing"] in (
                None, "writing"):
            assert time.monotonic() < deadline
            time.sleep(0.05)
        assert "503" in client.get("/api/reports/report_0").json()["writing"]

    def test_recording_not_written(self, tmp_path: Path,
                                   monkeypatch: pytest.MonkeyPatch) -> None:
        """A recorded session never calls the model by itself."""
        import time

        from src.webapp import WebApp
        from test_webapp import manager

        monkeypatch.setattr("src.webapp.list_cameras", lambda: [])
        reports = tmp_path / "reports"
        reports.mkdir()
        write_analysis(reports, "report_0", "2026-11-17T10:00:00+00:00")
        sessions = manager([("status", "running")])
        llm = FakeLlm()
        client = TestClient(WebApp(sessions, str(reports),
                                   str(tmp_path / "up"), "", llm=llm,
                                   llm_model="fake", record=True).app)
        view = client.post("/api/sessions", json={
            "kind": "camera", "source": "0", "record": True}).json()
        deadline = time.monotonic() + 5
        while sessions.view(view["id"]).status != "done":
            assert time.monotonic() < deadline
            time.sleep(0.05)
        time.sleep(0.3)
        assert llm.calls == []
        assert client.get("/api/reports/report_0").json()["writing"] is None
        assert not (reports / "written" / "report_0.json").exists()

    def test_no_model_no_writing(self, tmp_path: Path) -> None:
        from src.webapp import WebApp
        from test_webapp import manager

        sessions = manager([])
        WebApp(sessions, str(tmp_path), str(tmp_path), "")
        assert sessions.on_done is None
