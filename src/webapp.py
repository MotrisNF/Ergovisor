"""The graphical interface: a local web page to analyse cameras and
videos and to read the reports.

``WebApp`` serves the page (``src/web``) and its API:

- ``GET /api/devices``: the cameras of this computer.
- ``GET /api/sessions``, ``POST /api/sessions`` (a camera or an
  address), ``POST /api/sessions/video`` (an uploaded video, as the
  request body), ``DELETE /api/sessions/{id}`` (stop it, or remove it
  once ended).
- ``GET /api/sessions/{id}/stream``: its live picture (MJPEG, for an
  ``<img>``).
- ``GET /api/reports``, ``GET /api/reports/{name}``,
  ``DELETE /api/reports/{name}``: the saved reports.
- ``POST /api/reports/{name}/written``: the language model writes the
  detailed report of an analysis (``?again=true`` to write it again);
  ``GET /api/reports/{name}/docx`` downloads it. Each analysis of the
  interface gets it written by itself as soon as it is saved.
- ``GET /api/summaries``, ``POST /api/summaries`` (a period),
  ``GET /api/summaries/{name}``, ``GET /api/summaries/{name}/docx``,
  ``DELETE /api/summaries/{name}``: summaries of every analysis of a
  period, for the company and the Labour Inspectorate.
- ``GET /api/llm``: the model that writes.

It listens on this computer only: nothing asks for a password.
"""
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from datetime import date, datetime
import json
import queue
import threading
from pathlib import Path
import re
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import (FileResponse, JSONResponse, Response,
                               StreamingResponse)
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .exceptions import RagError
from .posture.analysis import free_name, people_of
from .posture.devices import Device, list_cameras
from .posture.docx_export import to_docx
from .posture.sessions import (Model, SessionConfig, SessionManager,
                               SessionView)
from .posture.sources import safe_name
from .posture.writer import (Llm, Written, load, save, write_report,
                             write_summary)
from .reader import CorpusReader

WEB = Path(__file__).parent / "web"
REGULATION = "_regulation"
# A session analysed for less than this (s) is a test of the camera: no
# detailed report is written by itself.
AUTO_MIN_SECONDS = 60.0
BOUNDARY = "frame"
# Characters kept in an uploaded file's name.
SAFE_NAME = re.compile(r"[^\w.-]+")


# The longest name a source can be given.
NAME_LENGTH = 60


def given_name(name: str | None) -> str:
    """A name given to a source, cleaned: one line, single spaces, at
    most ``NAME_LENGTH`` characters (empty: none given)."""
    return " ".join((name or "").split())[:NAME_LENGTH]


class NewSource(BaseModel):
    """A camera or an address to analyse.

    Attributes:
        kind: ``camera`` (a camera of this computer) or ``address``
            (RTSP or HTTP).
        source: The camera number or the address.
        model: Pose model.
        record: Also keep the camera's video (only when the server was
            started with recording allowed).
        name: What to call it (several cameras are told apart by it);
            empty: the camera's own name, or its address.
    """

    kind: Literal["camera", "address"]
    source: str = Field(min_length=1)
    model: Model = "full"
    record: bool = False
    name: str = Field(default="", max_length=NAME_LENGTH)


class Period(BaseModel):
    """The days of a summary, both included.

    Attributes:
        start: First day.
        end: Last day.
    """

    start: date
    end: date


def local_date(iso: str) -> date:
    """The local day of a report (saved as its day; older ones as a time
    in UTC)."""
    try:
        moment = datetime.fromisoformat(iso)
    except ValueError:
        return date.min
    if moment.tzinfo is not None:
        moment = moment.astimezone()
    return moment.date()


def _order(name: str) -> tuple[int, str]:
    """Sessions of one day in the order they were made: by the number
    their name ends with (``camera_2026-11-20_10`` after ``_9``)."""
    head, _, tail = name.rpartition("_")
    return (int(tail), head) if tail.isdigit() and head else (0, name)


class ReportSummary(BaseModel):
    """A saved report, for the history.

    Attributes:
        name: Its file name without ``.json``.
        source: The video or camera analysed.
        created: The day it was made (``YYYY-MM-DD``; never its time).
        seconds: Time analysed (of the person seen longest).
        max_score: Highest REBA score, of anybody.
        max_level: Highest REBA action level.
        events: Moments of risk, of everybody (how many; not when).
        lifts: Lifts and lowerings, of everybody.
        max_index: Highest lifting index, if any.
        beyond_limits: Lifts with a limit of 0 kg (no index, not
            acceptable with any load).
        people: People analysed.
        topics: Kinds of risk with regulation; ``None`` without it.
        screen_warnings: Warnings of the screen work; ``None`` when the
            camera never had the screen's view.
    """

    name: str
    source: str
    created: str
    seconds: float
    max_score: int
    max_level: int
    events: int
    lifts: int
    max_index: float | None
    topics: int | None
    screen_warnings: int | None = None
    beyond_limits: int = 0
    people: int = 1


class WebApp:
    """The interface's server.

    Attributes:
        app: The FastAPI application.
        sessions: The sessions analysing.
    """

    def __init__(self, sessions: SessionManager,
                 save_directory: str = "data/output/posture",
                 upload_directory: str = "data/uploads",
                 regulation_dir: str = "data/processed/prl",
                 model: Model = "full", llm: Llm | None = None,
                 llm_model: str = "", record: bool = False,
                 screen: bool = False,
                 load_kg: float | None = None,
                 report_every: float = 0.0,
                 private: bool = False,
                 demo: tuple[int, float, float] | None = None,
                 demo_write: bool = True) -> None:
        """Set up the routes; nothing is analysed or written yet.

        Args:
            sessions: Starts and follows the sessions.
            save_directory: Where the reports are (and are saved); the
                written ones go to its ``written`` and ``summaries``
                folders.
            upload_directory: Where uploaded videos are kept.
            regulation_dir: Index of the prevention corpus; empty: no
                regulation.
            model: Pose model when a request does not choose one.
            llm: Writes the documents (Gemini); ``None``: writing is
                not available.
            llm_model: The name of that model, kept with each document.
            record: Whether cameras may be recorded (``app --record``);
                off by default, so nobody can record from the interface.
            screen: Whether the work at a screen is analysed too (``app
                --screen``); off by default: only the body.
            load_kg: The load assumed in every lift (``app --load_kg``),
                or ``None``.
            report_every: Seconds after which a session saves a report
                and goes on (``app --report_hours``); 0: one at the end.
            private: No picture of any source is made or shown, and
                nothing can be recorded (``app --private``): only the
                name of each source, that it is active, and what was
                measured.
            demo: Demonstration mode (``app --demo``): how many sources
                wait for each other, the countdown to start them together
                and how long they analyse (seconds); each one's report is
                then written at once, however short.
            demo_write: Whether a demonstration's reports are written by
                themselves (``app --demo_write=False``: not, to rehearse
                without spending calls; the button still writes one).
        """
        self.sessions = sessions
        self._record = record
        self._screen = screen
        self._load_kg = load_kg
        self._report_every = report_every
        self._private = private
        self._demo = demo
        self._demo_write = demo_write
        self._llm = llm
        self._llm_model = llm_model
        # One document written at a time: a second click does not spend
        # a second call.
        self._writing = threading.Lock()
        # Detailed reports waiting to be written by themselves, and how
        # each one is going ("writing", or the error that stopped it).
        self._pending: queue.Queue[str] = queue.Queue()
        self._states: dict[str, str] = {}
        if llm is not None:
            sessions.on_done = self._queue_writing
            threading.Thread(target=self._write_pending,
                             daemon=True).start()
        self._reports = Path(save_directory)
        self._uploads = Path(upload_directory)
        self._regulation_dir = regulation_dir
        self._model = model
        self.app = FastAPI(title="Posture analysis",
                           lifespan=self._lifespan)
        self.app.exception_handler(RagError)(self._rag_error)
        self.app.get("/")(self.index)
        self.app.get("/api/devices")(self.devices)
        self.app.get("/api/sessions")(self.list_sessions)
        self.app.post("/api/sessions")(self.add_source)
        self.app.post("/api/sessions/video")(self.add_video)
        self.app.delete("/api/sessions/{session_id}")(self.close_session)
        self.app.get("/api/sessions/{session_id}/stream")(self.stream)
        self.app.get("/api/reports")(self.list_reports)
        self.app.get("/api/reports/{name}")(self.report)
        self.app.delete("/api/reports/{name}")(self.delete_report)
        self.app.post("/api/reports/{name}/written")(self.write)
        self.app.get("/api/reports/{name}/docx")(self.report_docx)
        self.app.get("/api/summaries")(self.list_summaries)
        self.app.post("/api/summaries")(self.summarise)
        self.app.get("/api/summaries/{name}")(self.summary)
        self.app.get("/api/summaries/{name}/docx")(self.summary_docx)
        self.app.delete("/api/summaries/{name}")(self.delete_summary)
        self.app.get("/api/llm")(self.llm_info)
        self.app.mount("/static", StaticFiles(directory=WEB), name="static")

    @asynccontextmanager
    async def _lifespan(self, app: FastAPI) -> AsyncIterator[None]:
        """When the server stops, every session stops and saves."""
        yield
        self.sessions.shutdown()

    @staticmethod
    def _rag_error(request: Request, error: Exception) -> JSONResponse:
        """A ``RagError`` is a 409 with its message (the page shows it)."""
        return JSONResponse(status_code=409, content={"detail": str(error)})

    def index(self) -> FileResponse:
        """The page."""
        return FileResponse(WEB / "index.html")

    def devices(self) -> list[Device]:
        """The cameras of this computer."""
        return list_cameras()

    def list_sessions(self) -> list[SessionView]:
        """The sessions, analysing or just ended."""
        return self.sessions.views()

    def _config(self, kind: Literal["camera", "address", "video"],
                source: str, label: str, model: Model,
                record: str = "") -> SessionConfig:
        return SessionConfig(kind=kind, source=source, label=label,
                             model=model,
                             regulation_dir=self._regulation_dir,
                             save_directory=str(self._reports),
                             record=record, screen=self._screen,
                             load_kg=self._load_kg,
                             report_every=self._report_every,
                             private=self._private,
                             demo_cameras=self._demo[0] if self._demo else 0,
                             demo_countdown=(self._demo[1] if self._demo
                                             else 5.0),
                             demo_seconds=(self._demo[2] if self._demo
                                           else 30.0))

    def add_source(self, new: NewSource) -> SessionView:
        """Start analysing a camera of this computer or an address."""
        source = new.source.strip()
        if new.kind == "camera":
            if not source.isdigit():
                raise HTTPException(422, "El número de cámara no es válido.")
            label = f"Cámara {source}"
            names = {str(d.number): d.name for d in list_cameras()}
            if source in names:
                label = f"{names[source]} ({source})"
        else:
            if not re.match(r"(rtsp|rtsps|http|https)://", source):
                raise HTTPException(
                    422, "La dirección debe empezar por rtsp:// o http://.")
            label = safe_name(source).removeprefix("camera ")
        label = given_name(new.name) or label
        record = ""
        if new.record:
            if not self._record or self._private:
                raise HTTPException(403, "La grabación no está permitida "
                                         "en este servidor.")
            record = str(self._uploads / (free_name(
                self._uploads, f"grabacion_{datetime.now():%Y-%m-%d}",
                ".mp4", number=True) + ".mp4"))
        return self.sessions.start(self._config(new.kind, source, label,
                                                new.model, record))

    async def add_video(self, request: Request, filename: str,
                        model: Model | None = None,
                        label: str = "") -> SessionView:
        """Keep an uploaded video (the request body) and analyse it."""
        name = SAFE_NAME.sub("_", Path(filename).name).strip("._")
        if not name:
            raise HTTPException(422, "El nombre del vídeo no es válido.")
        self._uploads.mkdir(parents=True, exist_ok=True)
        path = self._uploads / name
        stem, suffix, number = path.stem, path.suffix, 1
        while path.exists():
            number += 1
            path = self._uploads / f"{stem}_{number}{suffix}"
        with path.open("wb") as file:
            async for piece in request.stream():
                file.write(piece)
        if path.stat().st_size == 0:
            path.unlink()
            raise HTTPException(422, "El vídeo está vacío.")
        return self.sessions.start(self._config(
            "video", str(path), given_name(label) or Path(filename).name,
            model or self._model))

    def close_session(self, session_id: str) -> SessionView:
        """Stop a session (its report is saved) or remove an ended one."""
        try:
            return self.sessions.close(session_id)
        except KeyError:
            raise HTTPException(404, "No existe esa fuente.") from None

    def stream(self, session_id: str) -> StreamingResponse:
        """The live picture of a session, as MJPEG (never in private
        mode: there is none)."""
        if self._private:
            raise HTTPException(403, "Modo privado: no se muestra la "
                                     "imagen de ninguna fuente.")
        try:
            self.sessions.view(session_id)
        except KeyError:
            raise HTTPException(404, "No existe esa fuente.") from None

        def pictures() -> Iterator[bytes]:
            number = 0
            while True:
                got = self.sessions.picture(session_id, number)
                if got is None:
                    try:
                        if self.sessions.view(session_id).status in (
                                "loading", "running", "saving"):
                            continue
                    except KeyError:
                        pass
                    return
                number, jpeg = got
                yield (f"--{BOUNDARY}\r\nContent-Type: image/jpeg\r\n"
                       f"Content-Length: {len(jpeg)}\r\n\r\n").encode()
                yield jpeg + b"\r\n"

        return StreamingResponse(
            pictures(),
            media_type=f"multipart/x-mixed-replace; boundary={BOUNDARY}")

    def _report_path(self, name: str) -> Path:
        """The file of a report, checked to be in the reports folder."""
        if (not name or Path(name).name != name or name.startswith(".")
                or name.endswith(REGULATION)):
            raise HTTPException(404, "No existe ese informe.")
        path = self._reports / f"{name}.json"
        if not path.is_file():
            raise HTTPException(404, "No existe ese informe.")
        return path

    def list_reports(self) -> list[ReportSummary]:
        """The saved reports, newest first."""
        found: list[ReportSummary] = []
        if not self._reports.is_dir():
            return found
        for path in self._reports.glob("*.json"):
            if path.stem.endswith(REGULATION):
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                people = people_of(data)
                if not people and "people" not in data:
                    continue
                summaries = [p["summary"] for p in people]
                liftings = [p.get("lifting") or {} for p in people]
                offices = [p["office"] for p in people if p.get("office")]
                indices = [lifting["max_index"] for lifting in liftings
                           if lifting.get("max_index") is not None]
                regulation = path.with_name(f"{path.stem}{REGULATION}.json")
                topics = None
                if regulation.is_file():
                    topics = len(json.loads(regulation.read_text(
                        encoding="utf-8")).get("findings", []))
                found.append(ReportSummary(
                    name=path.stem, source=data.get("source", ""),
                    created=data.get("created", ""),
                    seconds=max([s.get("seconds", 0.0) for s in summaries]
                                + [o.get("seconds_at_screen", 0.0)
                                   for o in offices] + [0.0]),
                    max_score=max([s.get("max_score", 0) for s in summaries]
                                  + [0]),
                    max_level=max([s.get("max_level", 0) for s in summaries]
                                  + [0]),
                    events=sum(p.get("moments", 0) for p in people),
                    lifts=sum(len(lifting.get("lifts", []))
                              for lifting in liftings),
                    max_index=max(indices) if indices else None,
                    beyond_limits=sum(lifting.get("beyond_limits", 0)
                                      for lifting in liftings),
                    topics=topics,
                    screen_warnings=(sum(w.get("times", 0)
                                         for o in offices
                                         for w in o.get("warnings", []))
                                     if offices else None),
                    people=len(people)))
            except (OSError, ValueError, KeyError, TypeError):
                continue
        return sorted(found, key=lambda r: (r.created, _order(r.name)),
                      reverse=True)

    def report(self, name: str) -> dict[str, Any]:
        """A report: the analysis and its regulation (``null`` if none)."""
        path = self._report_path(name)
        regulation = path.with_name(f"{name}{REGULATION}.json")
        try:
            written = load(self._written / f"{name}.json")
            analysis = json.loads(path.read_text(encoding="utf-8"))
            # Saved before there were several people: its only person.
            analysis["people"] = people_of(analysis)
            return {
                "name": name,
                "analysis": analysis,
                "regulation": (json.loads(regulation.read_text(
                    encoding="utf-8")) if regulation.is_file() else None),
                "written": written.model_dump() if written else None,
                "writing": self._states.get(name),
            }
        except (OSError, ValueError) as error:
            raise HTTPException(500, f"No se pudo leer el informe: "
                                     f"{error}") from error

    def delete_report(self, name: str) -> dict[str, str]:
        """Delete a report and its regulation."""
        path = self._report_path(name)
        path.unlink()
        path.with_name(f"{name}{REGULATION}.json").unlink(missing_ok=True)
        (self._written / f"{name}.json").unlink(missing_ok=True)
        return {"deleted": name}

    @property
    def _written(self) -> Path:
        return self._reports / "written"

    @property
    def _summaries(self) -> Path:
        return self._reports / "summaries"

    def llm_info(self) -> dict[str, Any]:
        """The model that writes, if any, and whether recording is
        allowed."""
        return {"available": self._llm is not None,
                "model": self._llm_model,
                "record": self._record and not self._private,
                "private": self._private,
                "demo": self._demo is not None}

    def _require_llm(self) -> Llm:
        if self._llm is None:
            raise HTTPException(503, "Falta la clave de Gemini: añade "
                                     "GEMINI_API_KEY en .env y reinicia.")
        return self._llm

    @staticmethod
    def _read_corpus(path: str) -> str:
        return CorpusReader.project_path(path).read_text(encoding="utf-8")

    def _queue_writing(self, name: str, config: SessionConfig) -> None:
        """An analysis was saved: write its detailed report soon.

        Not, by itself, for a recorded session (calibration videos), one
        shorter than ``AUTO_MIN_SECONDS`` (a test of the camera) or one
        that found nothing to report: those never spend a call to the
        model; the report can still be asked for with its button.
        """
        if self._llm is None:
            return
        if config.demo_cameras:
            if not self._demo_write:
                return
        elif config.record or not self._worth_writing(name):
            return
        self._states[name] = "writing"
        self._pending.put(name)

    def _worth_writing(self, name: str) -> bool:
        """Whether an analysis is long enough and found something."""
        try:
            data = json.loads((self._reports / f"{name}.json").read_text(
                encoding="utf-8"))
        except (OSError, ValueError):
            return False
        for person in people_of(data):
            office = person.get("office") or {}
            seconds = max(person.get("summary", {}).get("seconds", 0.0),
                          office.get("seconds_at_screen", 0.0))
            found = (person.get("moments")
                     or (person.get("lifting") or {}).get("lifts")
                     or (person.get("pushing") or {}).get("pushes")
                     or office.get("warnings"))
            if seconds >= AUTO_MIN_SECONDS and found:
                return True
        return False

    def _write_pending(self) -> None:
        """Write the detailed reports of the saved analyses, one by one
        (a thread for the life of the server)."""
        while True:
            name = self._pending.get()
            try:
                self._write(name, again=False, wait=True)
                self._states.pop(name, None)
            except HTTPException as error:
                self._states[name] = str(error.detail)
            except RagError as error:
                self._states[name] = str(error)

    def _write(self, name: str, again: bool, wait: bool) -> Written:
        """Write the detailed report of an analysis (one call), unless
        it is already written and not ``again``."""
        data = self.report(name)
        target = self._written / f"{name}.json"
        existing = load(target)
        if existing is not None and not again:
            return existing
        llm = self._require_llm()
        if not self._writing.acquire(blocking=wait):
            raise HTTPException(409, "Ya se está redactando otro "
                                     "documento; espera a que termine.")
        try:
            written = write_report(name, data["analysis"],
                                   data["regulation"], llm,
                                   self._llm_model, self._read_corpus)
            save(written, target)
            return written
        finally:
            self._writing.release()

    def write(self, name: str, again: bool = False) -> Written:
        """The language model writes the detailed report of an analysis.

        Spends one call to the model; a report already written is given
        back as it is, unless ``again``.
        """
        self._states[name] = "writing"
        try:
            written = self._write(name, again, wait=False)
            self._states.pop(name, None)
            return written
        except (HTTPException, RagError) as error:
            self._states[name] = str(getattr(error, "detail", error))
            raise

    @staticmethod
    def _docx(written: Written, name: str) -> Response:
        return Response(
            to_docx(written.markdown),
            media_type=("application/vnd.openxmlformats-officedocument."
                        "wordprocessingml.document"),
            headers={"Content-Disposition":
                     f'attachment; filename="{name}.docx"'})

    def report_docx(self, name: str) -> Response:
        """The written report as a Word document."""
        self._report_path(name)
        written = load(self._written / f"{name}.json")
        if written is None:
            raise HTTPException(404, "Ese informe aún no está redactado.")
        return self._docx(written, f"informe_{name}")

    def _summary_path(self, name: str) -> Path:
        if not name or Path(name).name != name or name.startswith("."):
            raise HTTPException(404, "No existe ese resumen.")
        path = self._summaries / f"{name}.json"
        if not path.is_file():
            raise HTTPException(404, "No existe ese resumen.")
        return path

    def list_summaries(self) -> list[dict[str, Any]]:
        """The summaries written, newest first."""
        found = []
        if self._summaries.is_dir():
            for path in self._summaries.glob("*.json"):
                written = load(path)
                if written is not None:
                    found.append({"name": path.stem, "title": written.title,
                                  "created": written.created,
                                  "model": written.model,
                                  "reports": len(written.reports)})
        return sorted(found, key=lambda s: s["created"], reverse=True)

    def summarise(self, period: Period) -> dict[str, Any]:
        """The language model writes the summary of every report of a
        period (one call)."""
        if period.end < period.start:
            raise HTTPException(422, "La fecha final es anterior a la "
                                     "inicial.")
        chosen = [r for r in self.list_reports()
                  if period.start <= local_date(r.created) <= period.end]
        if not chosen:
            raise HTTPException(422, "No hay informes en ese periodo.")
        llm = self._require_llm()
        if not self._writing.acquire(blocking=False):
            raise HTTPException(409, "Ya se está redactando otro "
                                     "documento; espera a que termine.")
        try:
            data = [self.report(r.name)
                    for r in sorted(chosen, key=lambda r: r.created)]
            written = write_summary(
                [(d["name"], d["analysis"]) for d in data],
                [d["regulation"] for d in data],
                period.start.strftime("%d/%m/%Y"),
                period.end.strftime("%d/%m/%Y"), llm, self._llm_model,
                self._read_corpus)
            name = free_name(self._summaries,
                             f"resumen_{period.start}_{period.end}")
            save(written, self._summaries / f"{name}.json")
            return {"name": name, **written.model_dump()}
        finally:
            self._writing.release()

    def summary(self, name: str) -> dict[str, Any]:
        """A summary."""
        written = load(self._summary_path(name))
        if written is None:
            raise HTTPException(500, "No se pudo leer el resumen.")
        return {"name": name, **written.model_dump()}

    def summary_docx(self, name: str) -> Response:
        """A summary as a Word document."""
        written = load(self._summary_path(name))
        if written is None:
            raise HTTPException(500, "No se pudo leer el resumen.")
        return self._docx(written, name)

    def delete_summary(self, name: str) -> dict[str, str]:
        """Delete a summary."""
        self._summary_path(name).unlink()
        return {"deleted": name}
