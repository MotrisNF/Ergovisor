"""Tests for the interface: sessions, cameras and the web API.

The sessions run in threads with a fake analysis instead of processes
with the pose model, so the tests are fast and need no camera.
"""
from datetime import datetime
import json
from pathlib import Path
import queue
import threading
import time
from typing import Any

from fastapi.testclient import TestClient
import pytest

from src.exceptions import RagError
from src.posture import devices
from src.posture.sessions import (Kind, SessionConfig, SessionManager,
                                  report_name)
from src import webapp
from src.webapp import WebApp

# A moment of risk as a live session sends it (with its times: they are
# only shown while it runs, never saved).
EVENT = {"start": 1.0, "end": 4.0, "score": 9, "level": 3, "risk": "Alto",
         "action": "Necesaria pronto", "causes": ["tronco flexionado 60°"]}
# How a saved report counts it.
RISK = {"cause": "tronco flexionado 20-60°", "times": 1, "seconds": 3.0,
        "max_score": 9}


def fake_launcher(script: list[tuple[str, Any]], hold: bool = False) -> Any:
    """A launcher whose session sends ``script``; with ``hold`` it then
    waits to be stopped, like a camera, before saying it is done."""

    def launch(config: SessionConfig, messages: Any, pictures: Any,
               stop: Any, go: Any) -> threading.Thread:
        def run() -> None:
            pictures.put((b"jpeg-1", 0.5))
            if config.demo_cameras:
                # Ready, and waits for the others, as a real one does.
                messages.put(("status", "ready"))
                if not go.wait(5):
                    return
            for message in script:
                messages.put(message)
            if hold:
                stop.wait(5)
            # A real session ends with "done" or "error", never both.
            if not any(kind == "error" for kind, _ in script):
                messages.put(("done", f"report_{config.source}"))
        return threading.Thread(target=run, daemon=True)
    return launch


def manager(script: list[tuple[str, Any]], hold: bool = False,
            limit: int = 4) -> SessionManager:
    return SessionManager(limit, fake_launcher(script, hold),
                          queues=lambda size: queue.Queue(size),
                          event=threading.Event)


def wait_for(check: Any, timeout: float = 3.0) -> None:
    deadline = time.monotonic() + timeout
    while not check():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.02)


def config(source: str = "0", kind: Kind = "camera") -> SessionConfig:
    return SessionConfig(kind=kind, source=source, label=f"Cámara {source}")


class TestSessionManager:
    """Starting, following and closing sessions."""

    def test_follows_messages_until_done(self) -> None:
        sessions = manager([("status", "running"), ("event", EVENT),
                            ("finding", {"topic": "neck"})])
        view = sessions.start(config())
        wait_for(lambda: sessions.view(view.id).status == "done")
        done = sessions.view(view.id)
        assert done.events == [EVENT]
        assert done.findings == [{"topic": "neck"}]
        assert done.report == "report_0"
        assert done.camera == 0

    def test_lift_scored_again_replaces(self) -> None:
        """A lift sent again (its way up went on after a pause) replaces
        the first one; another person's with the same start does not."""
        first = {"person": 1, "kind": "levantamiento", "d": 30,
                 "start": {"second": 4.0}}
        again = {**first, "d": 60}
        other = {**first, "person": 2}
        sessions = manager([("lift", first), ("lift", other),
                            ("lift", again)])
        view = sessions.start(config())
        wait_for(lambda: sessions.view(view.id).status == "done")
        assert sessions.view(view.id).lifts == [again, other]

    def test_error_kept(self) -> None:
        sessions = manager([("error", "Cannot open the camera 0")])
        view = sessions.start(config())
        wait_for(lambda: sessions.view(view.id).status == "error")
        assert sessions.view(view.id).error == "Cannot open the camera 0"

    def test_limit(self) -> None:
        sessions = manager([("status", "running")], hold=True, limit=1)
        sessions.start(config("0"))
        with pytest.raises(RagError, match="Ya hay 1"):
            sessions.start(config("1"))
        sessions.shutdown(1)

    def test_same_camera_twice(self) -> None:
        sessions = manager([("status", "running")], hold=True)
        sessions.start(config("0"))
        with pytest.raises(RagError, match="ya se está analizando"):
            sessions.start(config("0"))
        sessions.shutdown(1)

    def test_same_video_twice_allowed(self) -> None:
        sessions = manager([("status", "running")], hold=True)
        sessions.start(config("a.mp4", "video"))
        sessions.start(config("a.mp4", "video"))
        assert len(sessions.views()) == 2
        sessions.shutdown(1)

    def test_close_stops_then_removes(self) -> None:
        """Closing a running session stops it (it saves); closing it once
        ended removes it."""
        sessions = manager([("status", "running")], hold=True)
        view = sessions.start(config())
        wait_for(lambda: sessions.view(view.id).status == "running")
        sessions.close(view.id)
        wait_for(lambda: sessions.view(view.id).status == "done")
        sessions.close(view.id)
        assert sessions.views() == []

    def test_unknown(self) -> None:
        with pytest.raises(KeyError):
            manager([]).close("nope")

    def test_picture(self) -> None:
        sessions = manager([("status", "running")], hold=True)
        view = sessions.start(config())
        wait_for(lambda: sessions.view(view.id).status == "running")
        assert sessions.picture(view.id, 0, timeout=2) == (1, b"jpeg-1")
        # Nothing newer comes: it waits and gives up.
        assert sessions.picture(view.id, 1, timeout=0.1) is None
        sessions.shutdown(1)

    def test_process_gone_without_word(self) -> None:
        def launch(*args: Any) -> threading.Thread:
            return threading.Thread(target=lambda: None, daemon=True)
        sessions = SessionManager(4, launch, lambda size: queue.Queue(size),
                                  threading.Event)
        view = sessions.start(config())
        wait_for(lambda: sessions.view(view.id).status == "error")


class TestReportName:
    """Report names never overwrite each other."""

    def test_camera(self, tmp_path: Path) -> None:
        """A camera: its day and a number, never its time (RGPD)."""
        camera = SessionConfig(kind="camera", source="0", label="x",
                               save_directory=str(tmp_path))
        when = datetime(2026, 11, 17, 20, 40, 57)
        assert report_name(camera, when) == "camera_2026-11-17_1"
        (tmp_path / "camera_2026-11-17_1.json").write_text("{}")
        assert report_name(camera, when) == "camera_2026-11-17_2"

    def test_video(self, tmp_path: Path) -> None:
        video = SessionConfig(kind="video", source="up/obra.mp4", label="x",
                              save_directory=str(tmp_path))
        when = datetime(2026, 11, 17, 9, 0, 0)
        assert report_name(video, when) == "obra"
        (tmp_path / "obra.json").write_text("{}")
        assert report_name(video, when) == "obra_2"


class TestDevices:
    """Only the picture devices of each camera are listed."""

    def test_sysfs(self, tmp_path: Path) -> None:
        for name, index, label in [("video0", "0", "Webcam"),
                                   ("video1", "1", "Webcam"),
                                   ("video2", "0", "USB")]:
            (tmp_path / name).mkdir()
            (tmp_path / name / "index").write_text(index)
            (tmp_path / name / "name").write_text(label + "\n")
        found = devices.list_cameras(tmp_path)
        assert [(d.number, d.name) for d in found] == [
            (0, "Webcam"), (2, "USB")]

    def test_none(self, tmp_path: Path) -> None:
        assert devices.list_cameras(tmp_path) == []


def write_report(folder: Path, name: str, created: str,
                 regulation: bool = True) -> None:
    analysis = {
        "source": f"camera {name}", "created": created, "people": [{
            "person": 1,
            "summary": {"seconds": 12.0, "max_score": 9, "max_level": 3},
            "moments": 1, "risks": [RISK],
            "lifting": {"lifts": [{}], "max_index": 1.4}}]}
    (folder / f"{name}.json").write_text(json.dumps(analysis))
    if regulation:
        (folder / f"{name}_regulation.json").write_text(
            json.dumps({"findings": [{}, {}]}))


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setattr(webapp, "list_cameras", lambda: [
        devices.Device(number=0, name="Webcam")])
    reports = tmp_path / "reports"
    reports.mkdir()
    sessions = manager([("status", "running")], hold=True)
    app = WebApp(sessions, str(reports), str(tmp_path / "uploads"), "")
    return TestClient(app.app)


class TestApi:
    """The routes of the interface."""

    def test_page(self, client: TestClient) -> None:
        assert "Añadir fuente" in client.get("/").text
        assert client.get("/static/app.js").status_code == 200

    def test_devices(self, client: TestClient) -> None:
        assert client.get("/api/devices").json() == [
            {"number": 0, "name": "Webcam"}]

    def test_reports_newest_first(self, client: TestClient,
                                  tmp_path: Path) -> None:
        folder = tmp_path / "reports"
        write_report(folder, "old", "2026-11-01T10:00:00+00:00")
        write_report(folder, "new", "2026-11-17T10:00:00+00:00", False)
        (folder / "broken.json").write_text("{")
        got = client.get("/api/reports").json()
        assert [r["name"] for r in got] == ["new", "old"]
        assert got[1]["topics"] == 2 and got[0]["topics"] is None
        assert got[1]["max_index"] == 1.4 and got[1]["events"] == 1
        assert got[1]["people"] == 1

    def test_reports_of_several_people(self, client: TestClient,
                                       tmp_path: Path) -> None:
        """The history adds up everybody: the worst score and index, the
        moments and lifts of all, the time of the one seen longest."""
        people = [
            {"person": 1, "summary": {"seconds": 30.0, "max_score": 5,
                                      "max_level": 2},
             "moments": 1, "lifting": {"lifts": [{}], "max_index": 0.8}},
            {"person": 2, "summary": {"seconds": 12.0, "max_score": 9,
                                      "max_level": 3},
             "moments": 2,
             "lifting": {"lifts": [{}, {}], "max_index": 1.6}}]
        (tmp_path / "reports" / "two.json").write_text(json.dumps(
            {"source": "camera two", "created": "2026-11-18",
             "people": people}))
        [got] = client.get("/api/reports").json()
        assert got["people"] == 2 and got["seconds"] == 30.0
        assert (got["max_score"], got["max_level"]) == (9, 3)
        assert (got["events"], got["lifts"]) == (3, 3)
        assert got["max_index"] == 1.6

    def test_same_day_in_order(self, client: TestClient,
                               tmp_path: Path) -> None:
        """Reports keep only their day: those of one day go by their
        number, the newest first (10 after 9)."""
        folder = tmp_path / "reports"
        for n in (1, 9, 10):
            write_report(folder, f"camera_2026-11-17_{n}", "2026-11-17")
        write_report(folder, "camera_2026-11-18_1", "2026-11-18")
        got = [r["name"] for r in client.get("/api/reports").json()]
        assert got == ["camera_2026-11-18_1", "camera_2026-11-17_10",
                       "camera_2026-11-17_9", "camera_2026-11-17_1"]
        [person] = client.get("/api/reports/camera_2026-11-17_1").json()[
            "analysis"]["people"]
        assert person["risks"] == [RISK]

    def test_report_and_delete(self, client: TestClient,
                               tmp_path: Path) -> None:
        folder = tmp_path / "reports"
        write_report(folder, "a", "2026-11-17T10:00:00+00:00")
        got = client.get("/api/reports/a").json()
        assert got["analysis"]["source"] == "camera a"
        assert got["regulation"] == {"findings": [{}, {}]}
        assert client.delete("/api/reports/a").status_code == 200
        assert list(folder.iterdir()) == []

    @pytest.mark.parametrize("name", ["nope", "..%2Fsecret", "a_regulation",
                                      ".hidden"])
    def test_report_not_found(self, client: TestClient, tmp_path: Path,
                              name: str) -> None:
        write_report(tmp_path / "reports", "a", "2026-11-17")
        (tmp_path / "secret.json").write_text("{}")
        assert client.get(f"/api/reports/{name}").status_code == 404
        assert client.delete(f"/api/reports/{name}").status_code == 404
        assert (tmp_path / "secret.json").exists()

    def test_add_camera(self, client: TestClient) -> None:
        got = client.post("/api/sessions",
                          json={"kind": "camera", "source": "0"})
        assert got.status_code == 200
        assert got.json()["label"] == "Webcam (0)"
        assert got.json()["model"] == "full"

    def test_add_with_a_name(self, client: TestClient) -> None:
        """A name given to a source is its label, cleaned to one line;
        empty, the camera's own."""
        got = client.post("/api/sessions", json={
            "kind": "camera", "source": "0",
            "name": "  Almacén,\n muelle   2 "})
        assert got.json()["label"] == "Almacén, muelle 2"
        got = client.post("/api/sessions", json={
            "kind": "address", "source": "http://a/v", "name": ""})
        assert got.json()["label"] == "http://a/v"
        too_long = client.post("/api/sessions", json={
            "kind": "camera", "source": "0", "name": "x" * 61})
        assert too_long.status_code == 422

    def test_upload_with_a_name(self, client: TestClient) -> None:
        got = client.post("/api/sessions/video?filename=a.mp4&label=Obra%201",
                          content=b"video bytes")
        assert got.json()["label"] == "Obra 1"

    def test_add_address_hides_password(self, client: TestClient) -> None:
        got = client.post("/api/sessions", json={
            "kind": "address", "source": "rtsp://admin:secret@10.0.0.5/s1"})
        assert got.json()["label"] == "rtsp://***@10.0.0.5/s1"
        assert "secret" not in client.get("/api/sessions").text

    @pytest.mark.parametrize("body", [
        {"kind": "camera", "source": "x"},
        {"kind": "address", "source": "ftp://h/v"},
        {"kind": "address", "source": ""},
        {"kind": "camera", "source": "0", "model": "huge"},
    ])
    def test_add_invalid(self, client: TestClient,
                         body: dict[str, str]) -> None:
        assert client.post("/api/sessions", json=body).status_code == 422

    def test_limit_is_409(self, tmp_path: Path) -> None:
        sessions = manager([("status", "running")], hold=True, limit=1)
        api = TestClient(WebApp(sessions, str(tmp_path), str(tmp_path),
                                "").app)
        api.post("/api/sessions", json={"kind": "address",
                                        "source": "http://a/v"})
        got = api.post("/api/sessions", json={"kind": "address",
                                              "source": "http://b/v"})
        assert got.status_code == 409
        assert "Ya hay 1" in got.json()["detail"]
        sessions.shutdown(1)

    def test_upload(self, client: TestClient, tmp_path: Path) -> None:
        for _ in range(2):
            got = client.post("/api/sessions/video?filename=../mi%20obra.mp4",
                              content=b"video bytes")
            assert got.status_code == 200
            assert got.json()["kind"] == "video"
        kept = sorted(p.name for p in (tmp_path / "uploads").iterdir())
        assert kept == ["mi_obra.mp4", "mi_obra_2.mp4"]

    def test_upload_empty(self, client: TestClient) -> None:
        got = client.post("/api/sessions/video?filename=a.mp4", content=b"")
        assert got.status_code == 422

    def test_close_unknown(self, client: TestClient) -> None:
        assert client.delete("/api/sessions/nope").status_code == 404
        assert client.get("/api/sessions/nope/stream").status_code == 404


class TestRecording:
    """Recording is refused unless the server allows it."""

    def test_refused_by_default(self, client: TestClient) -> None:
        got = client.post("/api/sessions", json={
            "kind": "camera", "source": "0", "record": True})
        assert got.status_code == 403
        assert client.get("/api/sessions").json() == []
        assert client.get("/api/llm").json()["record"] is False

    def test_allowed(self, tmp_path: Path,
                     monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(webapp, "list_cameras", lambda: [])
        sessions = manager([("status", "running")], hold=True)
        api = TestClient(WebApp(sessions, str(tmp_path / "r"),
                                str(tmp_path / "up"), "", record=True).app)
        assert api.get("/api/llm").json()["record"] is True
        got = api.post("/api/sessions", json={
            "kind": "camera", "source": "0", "record": True}).json()
        recording = Path(got["recording"])
        assert recording.parent == tmp_path / "up"
        assert recording.name.startswith("grabacion_")
        assert recording.suffix == ".mp4"
        # Without asking for it, nothing is recorded.
        other = api.post("/api/sessions", json={
            "kind": "address", "source": "http://a/v"}).json()
        assert other["recording"] is None
        sessions.shutdown(1)


class TestScreenWork:
    """The work at a screen is only analysed when the app is started
    with it (``app --screen``): by default, only the body."""

    def test_off_by_default(self, tmp_path: Path) -> None:
        app = WebApp(manager([]), str(tmp_path), str(tmp_path), "")
        assert not app._config("camera", "0", "x", "full").screen

    def test_on_when_asked(self, tmp_path: Path) -> None:
        app = WebApp(manager([]), str(tmp_path), str(tmp_path), "",
                     screen=True)
        assert app._config("camera", "0", "x", "full").screen


def test_load_reaches_the_sessions(tmp_path: Path) -> None:
    """The load assumed by the app (``app --load_kg``) goes to every
    session; none by default."""
    app = WebApp(manager([]), str(tmp_path), str(tmp_path), "", load_kg=5.0)
    assert app._config("camera", "0", "x", "full").load_kg == 5.0
    plain = WebApp(manager([]), str(tmp_path), str(tmp_path), "")
    assert plain._config("camera", "0", "x", "full").load_kg is None


class TestParts:
    """A long session saves a report every few hours and goes on (here,
    every 3 s of a 10 s source)."""

    @staticmethod
    def frames(seconds: float) -> list[Any]:
        class Frame:
            def __init__(self, timestamp: float) -> None:
                self.timestamp = timestamp
        return [Frame(n / 10) for n in range(int(seconds * 10))]

    @staticmethod
    def analyse_part(frames: Any, until: float) -> list[float]:
        """Like ``analyse`` with ``seconds``: the frames up to ``until``
        (the one past it is read, and lost, as ``analyse`` does)."""
        seen = []
        for frame in frames:
            if until and frame.timestamp > until:
                break
            seen.append(frame.timestamp)
        return seen

    def test_every_few_seconds(self) -> None:
        from src.posture.sessions import in_parts

        saved: list[tuple[int, bool, float, float]] = []
        in_parts(self.frames(10), 3.0, self.analyse_part,
                 lambda part, n, final: saved.append(
                     (n, final, part[0], part[-1])),
                 lambda: False)
        assert [(n, final) for n, final, _, _ in saved] == [
            (1, False), (2, False), (3, False), (4, True)]
        # Each part covers its own seconds, one after the other.
        assert saved[0][2:] == (0.0, 3.0)
        assert saved[1][3] == pytest.approx(6.0)
        assert saved[3][3] == pytest.approx(9.9)

    def test_one_part_without_period(self) -> None:
        from src.posture.sessions import in_parts

        saved: list[bool] = []
        in_parts(self.frames(10), 0.0, self.analyse_part,
                 lambda part, n, final: saved.append(final), lambda: False)
        assert saved == [True]

    def test_stopped_is_the_last_part(self) -> None:
        """Stopped (the session closed) when a part ends: that part is
        the last one, saved as the end."""
        from src.posture.sessions import in_parts

        saved: list[tuple[int, bool]] = []
        in_parts(self.frames(10), 3.0, self.analyse_part,
                 lambda part, n, final: saved.append((n, final)),
                 lambda: True)
        assert saved == [(1, True)]

    def test_part_reports_are_kept_and_written(self) -> None:
        """Each part's report is told to the interface and written by
        itself, like a finished one."""
        sessions = manager([("status", "running"), ("report", "parte_1"),
                            ("report", "parte_2")])
        written: list[str] = []
        sessions.on_done = lambda name, config: written.append(name)
        view = sessions.start(config())
        wait_for(lambda: sessions.view(view.id).status == "done")
        assert sessions.view(view.id).reports == ["parte_1", "parte_2"]
        assert written == ["parte_1", "parte_2", "report_0"]


class TestPrivate:
    """Private mode: no picture of any source, nothing recorded."""

    @staticmethod
    def client(tmp_path: Path, **kwargs: Any) -> TestClient:
        return TestClient(WebApp(manager([("status", "running")],
                                         hold=True),
                                 str(tmp_path), str(tmp_path), "",
                                 **kwargs).app)

    def test_no_picture(self, tmp_path: Path) -> None:
        api = self.client(tmp_path, private=True)
        view = api.post("/api/sessions", json={"kind": "address",
                                               "source": "http://a/v"})
        assert view.json()["private"] is True
        got = api.get(f"/api/sessions/{view.json()['id']}/stream")
        assert got.status_code == 403
        assert api.get("/api/llm").json()["private"] is True

    def test_no_recording(self, tmp_path: Path) -> None:
        api = self.client(tmp_path, private=True, record=True)
        assert api.get("/api/llm").json()["record"] is False
        got = api.post("/api/sessions", json={
            "kind": "address", "source": "http://a/v", "record": True})
        assert got.status_code == 403

    def test_settings_reach_the_sessions(self, tmp_path: Path) -> None:
        app = WebApp(manager([]), str(tmp_path), str(tmp_path), "",
                     report_every=7200.0, private=True)
        made = app._config("camera", "0", "x", "full")
        assert (made.report_every, made.private) == (7200.0, True)
        plain = WebApp(manager([]), str(tmp_path), str(tmp_path), "")
        made = plain._config("camera", "0", "x", "full")
        assert (made.report_every, made.private) == (0.0, False)


class TestDemo:
    """Demonstration mode: the sources wait for each other and start
    together after a countdown."""

    @staticmethod
    def demo(source: str) -> SessionConfig:
        return SessionConfig(kind="camera", source=source,
                             label=f"Cámara {source}", demo_cameras=2,
                             demo_countdown=0.3, demo_seconds=30.0)

    def test_wait_for_each_other(self) -> None:
        sessions = manager([("status", "running")])
        first = sessions.start(self.demo("0"))
        wait_for(lambda: sessions.view(first.id).status == "ready")
        time.sleep(0.5)
        alone = sessions.view(first.id)
        assert alone.status == "ready" and alone.waiting == 1
        assert alone.starts_at is None
        second = sessions.start(self.demo("1"))
        wait_for(lambda: sessions.view(second.id).starts_at is not None)
        both = [sessions.view(first.id), sessions.view(second.id)]
        starts = both[0].starts_at
        assert starts is not None and starts == both[1].starts_at
        assert both[0].ends_at == pytest.approx(starts + 30.0)
        wait_for(lambda: all(sessions.view(v.id).status == "done"
                             for v in both))

    def test_time_without_pictures(self) -> None:
        """The time analysed comes on its own (private mode makes no
        pictures)."""
        sessions = manager([("status", "running"), ("second", 12.5)],
                           hold=True)
        view = sessions.start(config())
        wait_for(lambda: sessions.view(view.id).second == 12.5)
        sessions.close(view.id)

    def test_demo_reports_always_written(self, tmp_path: Path) -> None:
        """However short, a demonstration's report is written at once."""
        from src.webapp import WebApp as App

        app = App(manager([]), str(tmp_path), str(tmp_path), "",
                  demo=(2, 5.0, 30.0))
        made = app._config("camera", "0", "x", "full")
        assert (made.demo_cameras, made.demo_countdown,
                made.demo_seconds) == (2, 5.0, 30.0)
        app._queue_writing("nada", made)
        assert app._states["nada"] == "writing"
