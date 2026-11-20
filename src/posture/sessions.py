"""Several cameras and videos analysed at once, for the interface.

Each source (a camera of this computer, an address, an uploaded video)
is a *session* analysed in its own process, so the sources do not slow
each other down (the pose model holds the processor while it works).
The process sends back what it finds through two queues: its messages
(moments of risk, lifts, regulation, state) and the newest picture with
the analysis drawn on it, which the interface shows live. When it ends,
by itself (a video) or when it is closed, it saves its report like
``analyze`` does, and the report goes to the history.
"""
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
import multiprocessing
from multiprocessing.synchronize import Event as StopEvent
from pathlib import Path
import queue
import threading
import time
from typing import Any, Literal
import uuid

from pydantic import BaseModel

from ..exceptions import RagError

Kind = Literal["camera", "address", "video"]
Model = Literal["lite", "full", "heavy"]
# Pictures sent to the interface, at most, per second and their width.
PICTURES_PER_SECOND = 10
PICTURE_WIDTH = 960
JPEG_QUALITY = 75
# Pictures waiting for the interface: older ones are dropped.
PICTURES_QUEUED = 2


class SessionConfig(BaseModel):
    """What a session analyses and how.

    Attributes:
        kind: A camera of this computer, an address (RTSP or HTTP) or
            a video file.
        source: The camera number, the address or the video's path.
        label: How the interface names it (without passwords).
        model: Pose model.
        every: Seconds between two analysed frames.
        regulation_dir: Index of the prevention corpus; empty: no
            regulation.
        k: Passages per kind of risk.
        save_directory: Where the reports are written.
        record: For a camera or an address, a video file where its
            original picture is kept; empty: nothing is recorded.
        screen: Also analyse the work at a screen (distance, neck,
            blinks, visual rests, breaks) when the camera is a laptop's;
            off, only the body is analysed (postures, lifts, pushes).
    """

    kind: Kind
    source: str
    label: str
    model: Model = "full"
    every: float = 0.1
    regulation_dir: str = "data/processed/prl"
    k: int = 3
    save_directory: str = "data/output/posture"
    record: str = ""
    screen: bool = False


def report_name(config: SessionConfig, started: datetime) -> str:
    """The file name (without ``.json``) of a session's report.

    A camera is named by the day it started and a number
    (``camera_2026-11-20_1``), never by its time; a video by its name,
    numbered when a report of that name already exists.
    """
    from .analysis import free_name

    folder = Path(config.save_directory)
    if config.kind != "video":
        return free_name(folder, f"camera_{started:%Y-%m-%d}", number=True)
    return free_name(folder, Path(config.source).stem)


def office_snapshot(state: Any, person: int = 1
                    ) -> dict[str, Any] | None:
    """What the interface shows of the screen work now (of ``person``);
    ``None`` when the camera sees the whole body."""
    if state is None:
        return None
    face = state.face
    return {
        "person": person,
        "distance_cm": (round(face.distance_cm, 1)
                        if face is not None and face.distance_cm is not None
                        else None),
        "pitch": round(face.pitch, 1) if face is not None else None,
        "yaw": round(face.yaw, 1) if face is not None else None,
        "roll": round(face.roll, 1) if face is not None else None,
        "blinks_per_minute": (round(state.blinks_per_minute, 1)
                              if state.blinks_per_minute is not None
                              else None),
        "since_rest": round(state.since_rest),
        "since_break": round(state.since_break),
    }


def run_session(config: SessionConfig, messages: Any, pictures: Any,
                stop: StopEvent) -> None:
    """Analyse one source until it ends or ``stop`` is set (a process).

    Sends ``(kind, data)`` messages: ``status`` (``loading``,
    ``running``, ``saving``), ``event``, ``lift``, ``push``,
    ``office_event`` (each with the ``person`` it is of), ``finding``,
    ``warning``, ``progress``, ``office`` and ``people`` (the screen work
    and the people in the picture now, every second), and finally
    ``done`` (with the report's name) or ``error``. Pictures go to
    ``pictures`` as ``(jpeg bytes, second)``, dropped when the interface
    is behind.
    """
    import cv2

    from ..bm25_index import BM25Index
    from .analysis import analyse
    from .lifting import Lifting
    from .overlay import Overlay
    from .face import FaceEstimator
    from .office import Office
    from .people import MAX_PEOPLE
    from .pushing import Pushing
    from .pose import PoseEstimator
    from .regulation import LiveRegulation, find_regulation
    from .reba import Task
    from .sources import Camera, VideoFile

    def send(kind: str, data: Any = None) -> None:
        messages.put((kind, data))

    estimator = faces = None
    try:
        send("status", "loading")
        started = datetime.now()
        frames: VideoFile | Camera
        duration = 0.0
        if config.kind == "video":
            frames = VideoFile(config.source, config.every)
            capture = cv2.VideoCapture(config.source)
            count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
            fps = capture.get(cv2.CAP_PROP_FPS)
            capture.release()
            duration = count / fps if fps > 0 else 0.0
        else:
            device: int | str = (int(config.source)
                                 if config.kind == "camera"
                                 else config.source)
            frames = Camera(device, config.every,
                            warn=lambda text: send("warning", text),
                            record=(Path(config.record) if config.record
                                    else None))
        live = None
        if config.regulation_dir:
            live = LiveRegulation(
                BM25Index.load(Path(config.regulation_dir)), config.k)
        estimator = PoseEstimator(config.model, people=MAX_PEOPLE)
        if config.screen:
            faces = FaceEstimator(people=MAX_PEOPLE)
        office = Office() if config.screen else None
        told = [0.0]
        overlay = Overlay()
        lifting = Lifting()
        last = [0.0]

        def on_result(frame: Any, people: Any) -> bool:
            if stop.is_set():
                return False
            if time.monotonic() - told[0] >= 1:
                told[0] = time.monotonic()
                desk = next((p for p in people if p.office is not None),
                            None)
                send("office", office_snapshot(desk.office, desk.person)
                     if desk is not None else None)
                send("people", len(people))
            now = time.monotonic()
            if now - last[0] < 1 / PICTURES_PER_SECOND:
                return True
            last[0] = now
            image: Any = overlay.draw(frame.image, frame.timestamp,
                                      people)
            height, width = image.shape[:2]
            if width > PICTURE_WIDTH:
                image = cv2.resize(image, (PICTURE_WIDTH,
                                           height * PICTURE_WIDTH // width))
            ok, jpeg = cv2.imencode(
                ".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            if ok:
                try:
                    pictures.put_nowait((jpeg.tobytes(), frame.timestamp))
                except queue.Full:
                    pass
            if duration:
                send("progress", min(1.0, frame.timestamp / duration))
            return True

        def on_event(person: int, event: Any) -> None:
            send("event", {**event.model_dump(), "person": person})
            if live is not None:
                for finding in live.event(event):
                    send("finding", finding.model_dump())

        def on_office(person: int, warning: Any) -> None:
            send("office_event", {**warning.model_dump(), "person": person})
            if live is not None:
                for finding in live.office(warning):
                    send("finding", finding.model_dump())

        def on_lift(person: int, lift: Any) -> None:
            send("lift", {**lift.model_dump(), "person": person})
            if live is not None:
                for finding in live.lift(lift):
                    send("finding", finding.model_dump())

        def on_push(person: int, push: Any) -> None:
            send("push", {**push.model_dump(), "person": person})
            if live is not None:
                for finding in live.push(push):
                    send("finding", finding.model_dump())

        send("status", "running")
        result = analyse(frames, estimator, config.label, Task(),
                         seconds=0.0, on_event=on_event,
                         interruptible=True, on_result=on_result,
                         lifting=lifting, on_lift=on_lift, office=office,
                         faces=faces, on_office=on_office,
                         pushing=Pushing(), on_push=on_push,
                         max_people=MAX_PEOPLE)
        send("status", "saving")
        name = report_name(config, started)
        folder = Path(config.save_directory)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"{name}.json").write_text(
            result.model_dump_json(indent=2), encoding="utf-8")
        if live is not None:
            report = find_regulation(result, live.retriever,
                                     config.regulation_dir, config.k,
                                     searched=live.searched)
            (folder / f"{name}_regulation.json").write_text(
                report.model_dump_json(indent=2), encoding="utf-8")
        if config.record and Path(config.record).is_file():
            from .overlay import _to_h264
            # Playable in any player and browser.
            # Recordings can last half an hour: the fastest encoding, so
            # the session does not take minutes to save.
            _to_h264(Path(config.record), preset="ultrafast")
        send("done", name)
    except (RagError, OSError, ValueError) as error:
        send("error", str(error))
    except Exception as error:  # noqa: BLE001 - shown in the interface
        send("error", f"{type(error).__name__}: {error}")
    finally:
        if estimator is not None:
            estimator.close()
        if faces is not None:
            faces.close()


Status = Literal["loading", "running", "saving", "done", "error"]


class SessionView(BaseModel):
    """What the interface shows of a session.

    Attributes:
        id: The session's identifier.
        kind: Camera, address or video.
        label: Its name, without passwords.
        camera: The number of a camera of this computer, else ``None``.
        recording: The video file being recorded, if any.
        model: Pose model.
        started: When it was added (local time, ISO).
        status: Loading, running, saving, done or error.
        progress: Share of a video analysed (0-1); ``None`` for a camera.
        second: Time of the newest picture.
        events: The moments of risk so far.
        lifts: The lifts so far.
        pushes: The pushes and pulls so far.
        findings: The regulation found so far.
        office: The screen work now, or ``None`` when the camera sees the
            whole body.
        office_events: The warnings of the screen work so far.
        people: The people in the picture now.
        warnings: Cuts and reconnections of the camera.
        report: The report's name once saved.
        error: What went wrong.
    """

    id: str
    kind: Kind
    label: str
    camera: int | None
    recording: str | None
    model: Model
    started: str
    status: Status
    progress: float | None
    second: float
    events: list[dict[str, Any]]
    lifts: list[dict[str, Any]]
    pushes: list[dict[str, Any]]
    findings: list[dict[str, Any]]
    office: dict[str, Any] | None
    office_events: list[dict[str, Any]]
    people: int = 0
    warnings: list[str]
    report: str | None
    error: str | None


@dataclass
class _Session:
    """A session as the manager keeps it."""

    id: str
    config: SessionConfig
    started: str
    process: Any
    messages: Any
    pictures: Any
    stop: Any
    status: Status = "loading"
    progress: float | None = None
    second: float = 0.0
    events: list[dict[str, Any]] = field(default_factory=list)
    lifts: list[dict[str, Any]] = field(default_factory=list)
    pushes: list[dict[str, Any]] = field(default_factory=list)
    findings: list[dict[str, Any]] = field(default_factory=list)
    office: dict[str, Any] | None = None
    office_events: list[dict[str, Any]] = field(default_factory=list)
    people: int = 0
    warnings: list[str] = field(default_factory=list)
    report: str | None = None
    error: str | None = None
    picture: bytes | None = None
    picture_number: int = 0

    def view(self) -> SessionView:
        return SessionView(
            id=self.id, kind=self.config.kind, label=self.config.label,
            camera=(int(self.config.source)
                    if self.config.kind == "camera" else None),
            recording=self.config.record or None,
            model=self.config.model, started=self.started,
            status=self.status, progress=self.progress, second=self.second,
            events=self.events, lifts=self.lifts, pushes=self.pushes,
            findings=self.findings,
            office=self.office, office_events=self.office_events,
            people=self.people, warnings=self.warnings, report=self.report,
            error=self.error)

    @property
    def active(self) -> bool:
        return self.status in ("loading", "running", "saving")


# Starts a session: (config, messages, pictures, stop) -> a process-like
# object with ``is_alive``, ``join`` and ``start``.
Launcher = Callable[[SessionConfig, Any, Any, Any], Any]


def process_launcher(config: SessionConfig, messages: Any, pictures: Any,
                     stop: Any) -> Any:
    """Run a session in a new process (``spawn``: a clean interpreter,
    safe with the threads of the server and of the pose model)."""
    context = multiprocessing.get_context("spawn")
    return context.Process(target=run_session,
                           args=(config, messages, pictures, stop),
                           daemon=True)


class SessionManager:
    """The sessions of the interface: start, follow and close them.

    A thread per session reads what its process sends and keeps it here,
    so the requests of the interface only read memory.

    Attributes:
        limit: Sessions analysing at the same time, at most.
    """

    def __init__(self, limit: int = 4,
                 launcher: Launcher = process_launcher,
                 queues: Callable[[int], Any] | None = None,
                 event: Callable[[], Any] | None = None) -> None:
        """Nothing runs until a session is started.

        Args:
            limit: Sessions analysing at the same time, at most.
            launcher: Makes the process of a session (a thread in the
                tests).
            queues: Makes a queue of a given size (0: unbounded).
            event: Makes the stop event.
        """
        context = multiprocessing.get_context("spawn")
        self.limit = limit
        self._launcher = launcher
        self._queue = queues or (lambda size: context.Queue(size))
        self._event = event or context.Event
        self._sessions: dict[str, _Session] = {}
        self._lock = threading.Lock()
        self._pictures = threading.Condition(self._lock)
        # Told the name of each report saved and its session's config
        # (the interface writes its detailed report then).
        self.on_done: Callable[[str, SessionConfig], None] | None = None

    def start(self, config: SessionConfig) -> SessionView:
        """Start analysing a source.

        Raises:
            RagError: ``limit`` sessions are already analysing, or that
                camera is already in use.
        """
        with self._lock:
            active = [s for s in self._sessions.values() if s.active]
            if len(active) >= self.limit:
                raise RagError(f"Ya hay {self.limit} fuentes analizándose: "
                               "cierra una antes de añadir otra.")
            if any(s.config.kind == config.kind != "video"
                   and s.config.source == config.source for s in active):
                raise RagError(f"{config.label} ya se está analizando.")
            messages, pictures = self._queue(0), self._queue(PICTURES_QUEUED)
            stop = self._event()
            process = self._launcher(config, messages, pictures, stop)
            session = _Session(
                id=uuid.uuid4().hex[:8], config=config,
                started=datetime.now().isoformat(timespec="seconds"),
                process=process, messages=messages, pictures=pictures,
                stop=stop)
            self._sessions[session.id] = session
        process.start()
        threading.Thread(target=self._follow, args=(session,),
                         daemon=True).start()
        return session.view()

    def _follow(self, session: _Session) -> None:
        """Keep what a session's process sends until it ends."""
        ended = False
        while True:
            got = False
            try:
                kind, data = session.messages.get(timeout=0.05)
                got = True
                ended = self._apply(session, kind, data) or ended
                if kind == "done" and self.on_done is not None:
                    self.on_done(data, session.config)
            except queue.Empty:
                pass
            try:
                jpeg, second = session.pictures.get_nowait()
                got = True
                with self._pictures:
                    session.picture = jpeg
                    session.picture_number += 1
                    session.second = second
                    self._pictures.notify_all()
            except queue.Empty:
                pass
            if ended and not got:
                break
            if not got and not session.process.is_alive():
                # Gone without a word (killed): nothing else will come.
                with self._lock:
                    if session.active:
                        session.status = "error"
                        session.error = "El análisis se detuvo de golpe."
                break
        session.process.join(timeout=1)
        # Free the queues' pipes and semaphores (a multiprocessing queue
        # has close(); the queues of the tests do not).
        for channel in (session.messages, session.pictures):
            if hasattr(channel, "close"):
                channel.close()
        with self._pictures:
            self._pictures.notify_all()

    def _apply(self, session: _Session, kind: str, data: Any) -> bool:
        """Keep one message; whether it was the last one."""
        with self._lock:
            if kind == "status":
                session.status = data
            elif kind == "progress":
                session.progress = data
            elif kind == "event":
                session.events.append(data)
            elif kind == "lift":
                # A lift scored again (its way up went on after a pause)
                # replaces the first one: same person, kind and start.
                same = [n for n, lift in enumerate(session.lifts)
                        if (lift.get("person"), lift.get("kind"),
                            lift.get("start", {}).get("second"))
                        == (data.get("person"), data.get("kind"),
                            data.get("start", {}).get("second"))]
                if same:
                    session.lifts[same[-1]] = data
                else:
                    session.lifts.append(data)
            elif kind == "push":
                session.pushes.append(data)
            elif kind == "finding":
                session.findings.append(data)
            elif kind == "office":
                session.office = data
            elif kind == "office_event":
                session.office_events.append(data)
            elif kind == "people":
                session.people = data
            elif kind == "warning":
                session.warnings.append(data)
            elif kind == "done":
                session.status, session.report = "done", data
                if session.progress is not None:
                    session.progress = 1.0
                return True
            elif kind == "error":
                session.status, session.error = "error", data
                return True
        return False

    def views(self) -> list[SessionView]:
        """Every session, in the order they were added."""
        with self._lock:
            return [s.view() for s in self._sessions.values()]

    def view(self, session_id: str) -> SessionView:
        """One session.

        Raises:
            KeyError: There is no such session.
        """
        with self._lock:
            return self._sessions[session_id].view()

    def close(self, session_id: str) -> SessionView:
        """Stop a session (it saves its report) or, if it has ended,
        remove it from the list.

        Raises:
            KeyError: There is no such session.
        """
        with self._lock:
            session = self._sessions[session_id]
            if not session.active:
                del self._sessions[session_id]
                return session.view()
            session.stop.set()
            return session.view()

    def picture(self, session_id: str, after: int,
                timeout: float = 5.0) -> tuple[int, bytes] | None:
        """The newest picture of a session, waiting for one newer than
        ``after``.

        Returns:
            ``(number, jpeg)``, or ``None`` when the session is gone or
            ended, or no new picture came in ``timeout`` seconds.
        """
        deadline = time.monotonic() + timeout
        with self._pictures:
            while True:
                session = self._sessions.get(session_id)
                if session is None or not session.active:
                    return None
                if (session.picture is not None
                        and session.picture_number > after):
                    return session.picture_number, session.picture
                left = deadline - time.monotonic()
                if left <= 0:
                    return None
                self._pictures.wait(left)

    def shutdown(self, timeout: float = 15.0) -> None:
        """Stop every session and wait for them to save their reports."""
        with self._lock:
            sessions = list(self._sessions.values())
        for session in sessions:
            session.stop.set()
        deadline = time.monotonic() + timeout
        for session in sessions:
            session.process.join(timeout=max(0.0,
                                             deadline - time.monotonic()))
