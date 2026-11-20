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
               stop: Any) -> threading.Thread:
        def run() -> None:
            pictures.put((b"jpeg-1", 0.5))
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
