"""A person working at a screen, seen by the screen's own camera.

The pose model tells which view the camera has: the whole body (a
worker handling loads: REBA and NIOSH) or a person seated in front of
the screen, close to the camera, with the hips out of the picture. In
the second view the face model is used too, and this module follows
what the INSST's guide on display screen equipment (RD 488/1997) asks
for and a front camera can see:

- the distance from the eyes to the screen (never under 30 cm);
- the posture of the neck (REBA: more than 20° bent forward, bent
  backward, turned or tilted), held for more than a minute;
- visual rests: looking away from the screen for 20 s at least every
  20 minutes (the guide's "20-20-20" rule);
- short breaks away from the screen, for example every 30 minutes;
- the blinks per minute (the guide warns that screen work lowers them;
  it gives no figure, so it is only reported).

The camera is taken to sit on top of the screen, as in a laptop or a
webcam on the monitor: its distance is the screen's, and looking at the
screen is looking at it.
"""
from collections import deque
from dataclasses import dataclass, field
import statistics
from typing import Literal

from pydantic import BaseModel

from .face import Face
from .skeleton import Point, Skeleton

View = Literal["body", "desk"]
# Seconds of frames the view is decided from, so one bad frame does not
# switch it.
VIEW_WINDOW = 2.0
# Shoulders at least this share of the image's width apart: the person
# is close to the camera.
CLOSE_SHOULDERS = 0.28

# The guide: the screen "en ningún caso debe estar situada a menos de
# 300 mm"; usual office screens need 400 to 750 mm.
MIN_DISTANCE_CM = 30.0
USUAL_DISTANCE_CM = (40.0, 75.0)
# REBA (NTP 601): the neck scores more when bent forward more than 20°,
# bent backward, turned or tilted to a side. The head is measured from
# the camera, above the eyes: calibrated on 2026-11-18, looking at the
# screen gave 8-19° (up to 22° leaning in) and the keyboard or a phone
# 31-34°, so 25° separates them (20° of the neck plus the camera's).
NECK_FLEXION = 25.0
NECK_EXTENSION = 10.0
NECK_TURN = 20.0
NECK_TILT = 15.0
# REBA's "static" posture: held for more than a minute.
HELD = 60.0
# Too close for this long before it is a warning (not a glance).
CLOSE_HELD = 10.0
# The head turned this far, or no face with the person still there:
# the eyes are away from the screen.
# Calibrated on 2026-11-18: turning to look about 45° aside measured 34°,
# and a paper on the desk (not a rest: the eyes stay near) 28° of turn
# with 25° down. Over 25° of turn, or the head this far down or up.
AWAY_ANGLE = 25.0
# Looking this far down (the keyboard, a phone, a paper on the desk) the
# eyes stay near: it is not a visual rest, even with the head turned,
# and the neck is measured there.
NEAR_PITCH = 15.0
# The guide's 20-20-20 rule: look away at least every 20 minutes for
# at least 20 seconds.
REST_SECONDS = 20.0
REST_EVERY = 20 * 60.0
# The guide: short breaks "por ejemplo, cada 30 minutos" to change
# posture. Nobody in front of the camera for this long is a break.
BREAK_SECONDS = 30.0
BREAK_EVERY = 30 * 60.0
# Raised shoulders are not measured: tried on 2026-11-18 with the front
# camera, shrugging changed the ear-to-shoulder height and the height of
# the shoulders in the image by 4-8 %, and leaning to the screen or
# looking down changed them by 20-40 %; no reliable warning came of it.
# The distance is only measured with the head this close to facing the
# camera: looking down, the eyelids hide part of the iris and the
# distance comes out longer (checked on a recording: 50 cm read as 59).
FRONTAL = 20.0
# A blink: the eyes this closed, then open again. Calibrated on
# 2026-11-18: 10 of 10 deliberate blinks at any threshold; in a minute of
# reading, 0.5 missed 4 quick or half blinks that were checked by eye on
# the frames, 0.35 counted them.
BLINK_CLOSED = 0.35
BLINK_OPEN = 0.2

Kind = Literal["too_close", "neck_flexed", "neck_extended", "neck_turned",
               "no_visual_rest", "no_break"]
TITLES: dict[str, str] = {
    "too_close": "Pantalla demasiado cerca",
    "neck_flexed": "Cuello flexionado hacia la pantalla",
    "neck_extended": "Cuello en extensión (pantalla alta)",
    "neck_turned": "Cuello girado o inclinado",
    "no_visual_rest": "Sin descanso visual",
    "no_break": "Sin pausa",
}


def view_of(skeleton: Skeleton | None) -> View | None:
    """Which view one frame shows; ``None`` when it cannot tell.

    Seated at the screen: the face and the shoulders are seen and the
    hips are not, or the shoulders fill much of the image.
    """
    if skeleton is None or not skeleton.seen(
            Point.NOSE, Point.LEFT_SHOULDER, Point.RIGHT_SHOULDER):
        return None
    if not skeleton.seen(Point.LEFT_HIP, Point.RIGHT_HIP):
        return "desk"
    if skeleton.pixels is not None:
        width = abs(skeleton.pixels[Point.LEFT_SHOULDER][0]
                    - skeleton.pixels[Point.RIGHT_SHOULDER][0])
        if width >= CLOSE_SHOULDERS:
            return "desk"
    return "body"


class OfficeEvent(BaseModel):
    """A moment against the guide, from its start to its end.

    Attributes:
        kind: What it is (a key of ``TITLES``).
        title: What it is, in words.
        start: Second it started.
        end: Second it ended.
        detail: Its measure, in words.
    """

    kind: Kind
    title: str
    start: float
    end: float
    detail: str


class OfficeReport(BaseModel):
    """The screen work seen, for the report.

    Attributes:
        method: What was measured and against what.
        seconds_at_screen: Time with the person seated at the screen.
        seconds_with_face: Of it, time with the face measured.
        distance_cm: Usual eyes-to-screen distance (median).
        closest_cm: Shortest distance held for a few seconds.
        seconds_too_close: Time under ``MIN_DISTANCE_CM``.
        seconds_neck_flexed: Time with the neck bent forward > 20°.
        seconds_neck_extended: Time with the neck bent backward.
        seconds_neck_turned: Time with the neck turned or tilted.
        blinks: Blinks counted.
        blinks_per_minute: Blinks per minute with the face measured.
        visual_rests: Looks away from the screen of 20 s or more.
        longest_without_rest: Longest time without one, seconds.
        breaks: Times away from the screen of 30 s or more.
        longest_without_break: Longest time without one, seconds.
        events: The moments against the guide, in order.
        assumptions: What the measures take for granted.
    """

    method: str = ("Guía técnica del INSST sobre equipos con pantallas de "
                   "visualización (RD 488/1997) y criterios del cuello de "
                   "REBA (NTP 601), con la cámara de la pantalla")
    seconds_at_screen: float = 0.0
    seconds_with_face: float = 0.0
    distance_cm: float | None = None
    closest_cm: float | None = None
    seconds_too_close: float = 0.0
    seconds_neck_flexed: float = 0.0
    seconds_neck_extended: float = 0.0
    seconds_neck_turned: float = 0.0
    blinks: int = 0
    blinks_per_minute: float | None = None
    visual_rests: int = 0
    longest_without_rest: float = 0.0
    breaks: int = 0
    longest_without_break: float = 0.0
    events: list[OfficeEvent] = []
    assumptions: list[str] = [
        "La cámara está encima de la pantalla: su distancia es la de la "
        "pantalla y mirarla es mirar la pantalla.",
        "La distancia se calcula con el tamaño del iris (unos 11,7 mm) y "
        "un campo de visión típico de cámara de portátil (65°); puede "
        "variar un 15 % según la cámara.",
        "Los ángulos del cuello son los de la cabeza vista desde la "
        "cámara; si la pantalla está muy por debajo de los ojos, mirarla "
        "ya inclina la cabeza.",
        "Los hombros elevados no se miden: con la cámara frontal no se "
        "distinguen de forma fiable de acercarse a la pantalla o mirar "
        "hacia abajo.",
        "El parpadeo es orientativo: la guía no fija una frecuencia "
        "mínima.",
    ]


@dataclass
class OfficeState:
    """What the screen work shows right now, for the live drawing.

    Attributes:
        view: The view the camera has.
        face: The newest face, if any.
        blinks_per_minute: So far.
        since_rest: Seconds since the last visual rest.
        since_break: Seconds since the last break.
    """

    view: View | None
    face: Face | None
    blinks_per_minute: float | None
    since_rest: float
    since_break: float


@dataclass
class _Held:
    """A condition that has to last to be a warning."""

    kind: Kind
    minimum: float
    since: float | None = None
    worst: float = 0.0

    def add(self, second: float, on: bool,
            measure: float) -> tuple[float, float, float] | None:
        """Follow the condition; when it ends after lasting at least
        ``minimum``, its ``(start, end, worst measure)``."""
        if on:
            if self.since is None:
                self.since, self.worst = second, measure
            self.worst = max(self.worst, measure)
            return None
        ended = None
        if self.since is not None and second - self.since >= self.minimum:
            ended = (self.since, second, self.worst)
        self.since = None
        return ended


@dataclass
class Office:
    """Follows a person at a screen, frame by frame.

    ``add`` takes each frame's skeleton and, when the view is the
    screen's, its face; it returns the warnings that just ended.
    """

    _views: deque[tuple[float, View]] = field(default_factory=deque)
    view: View | None = None
    _report: OfficeReport = field(default_factory=OfficeReport)
    _last: float | None = None
    _distances: list[float] = field(default_factory=list)
    _closed: bool = False
    _away_since: float | None = None
    _absent_since: float | None = None
    _last_rest: float | None = None
    _last_break: float | None = None
    _warned_rest: bool = False
    _warned_break: bool = False
    _face: Face | None = None
    _held: dict[str, _Held] = field(default_factory=lambda: {
        "too_close": _Held("too_close", CLOSE_HELD),
        "neck_flexed": _Held("neck_flexed", HELD),
        "neck_extended": _Held("neck_extended", HELD),
        "neck_turned": _Held("neck_turned", HELD),
    })

    def see(self, second: float, skeleton: Skeleton | None) -> View | None:
        """Decide the view from the frames of the last ``VIEW_WINDOW``
        seconds (the most frequent one)."""
        found = view_of(skeleton)
        if found is not None:
            self._views.append((second, found))
        while self._views and second - self._views[0][0] > VIEW_WINDOW:
            self._views.popleft()
        if self._views:
            desk = sum(1 for _, v in self._views if v == "desk")
            self.view = "desk" if desk * 2 > len(self._views) else "body"
        return self.view

    def add(self, second: float, skeleton: Skeleton | None,
            face: Face | None) -> list[OfficeEvent]:
        """One frame; the warnings that just ended.

        Without a skeleton and a face the person is away from the
        screen: gone, or up and seen whole by the camera.
        """
        step = 0.0 if self._last is None else max(0.0, second - self._last)
        self._last = second
        report = self._report
        self._face = face
        ended: list[OfficeEvent] = []
        present = skeleton is not None or face is not None
        if self._last_rest is None:
            self._last_rest = self._last_break = second
        # Breaks: nobody in front of the screen.
        if not present:
            if self._absent_since is None:
                self._absent_since = second
            if second - self._absent_since >= BREAK_SECONDS:
                if self._last_break != self._absent_since:
                    report.breaks += 1
                self._last_break = self._absent_since
                self._last_rest = self._absent_since
                self._warned_break = self._warned_rest = False
        else:
            if (self._absent_since is not None
                    and second - self._absent_since >= BREAK_SECONDS):
                self._last_break = self._last_rest = second
            self._absent_since = None
            report.seconds_at_screen += step
        # Visual rests: the eyes away from the screen for 20 s.
        away = present and (
            face is None or face.pitch < -AWAY_ANGLE
            or (abs(face.yaw) > AWAY_ANGLE and face.pitch < NEAR_PITCH))
        if away:
            if self._away_since is None:
                self._away_since = second
            if second - self._away_since >= REST_SECONDS:
                if self._last_rest != self._away_since:
                    report.visual_rests += 1
                self._last_rest = self._away_since
                self._warned_rest = False
        elif present:
            if (self._away_since is not None
                    and second - self._away_since >= REST_SECONDS):
                self._last_rest = second
            self._away_since = None
        assert self._last_rest is not None and self._last_break is not None
        since_rest = second - self._last_rest
        since_break = second - self._last_break
        report.longest_without_rest = max(report.longest_without_rest,
                                          since_rest)
        report.longest_without_break = max(report.longest_without_break,
                                           since_break)
        if present and since_rest >= REST_EVERY and not self._warned_rest:
            self._warned_rest = True
            ended.append(self._event(
                "no_visual_rest", self._last_rest, second,
                f"{since_rest / 60:.0f} min mirando la pantalla sin "
                "apartar la vista 20 s"))
        if present and since_break >= BREAK_EVERY and not self._warned_break:
            self._warned_break = True
            ended.append(self._event(
                "no_break", self._last_break, second,
                f"{since_break / 60:.0f} min frente a la pantalla sin "
                "una pausa"))
        # The face: distance, neck and blinks.
        flexed = extended = turned = close = False
        distance = 0.0
        if face is not None and not away:
            report.seconds_with_face += step
            frontal = (abs(face.pitch) <= FRONTAL
                       and abs(face.yaw) <= FRONTAL)
            if face.distance_cm is not None and frontal:
                distance = face.distance_cm
                self._distances.append(distance)
                close = distance < MIN_DISTANCE_CM
                if close:
                    report.seconds_too_close += step
            flexed = face.pitch > NECK_FLEXION
            extended = face.pitch < -NECK_EXTENSION
            turned = (abs(face.yaw) > NECK_TURN
                      or abs(face.roll) > NECK_TILT)
            report.seconds_neck_flexed += step if flexed else 0.0
            report.seconds_neck_extended += step if extended else 0.0
            report.seconds_neck_turned += step if turned else 0.0
        if face is not None:
            if face.blink >= BLINK_CLOSED and not self._closed:
                self._closed = True
                report.blinks += 1
            elif face.blink <= BLINK_OPEN:
                self._closed = False
        checks = {
            "too_close": (close, MIN_DISTANCE_CM - distance),
            "neck_flexed": (flexed, face.pitch if face else 0.0),
            "neck_extended": (extended, -face.pitch if face else 0.0),
            "neck_turned": (turned, max(abs(face.yaw), abs(face.roll))
                            if face else 0.0),
        }
        for kind, (on, measure) in checks.items():
            held = self._held[kind]
            span = held.add(second, on, measure)
            if span is not None:
                ended.append(self._event(held.kind, span[0], span[1],
                                         self._detail(kind, span[2])))
        return ended

    @staticmethod
    def _detail(kind: str, worst: float) -> str:
        if kind == "too_close":
            return (f"a {MIN_DISTANCE_CM - worst:.0f} cm de la pantalla "
                    f"(mínimo {MIN_DISTANCE_CM:.0f} cm)")
        return f"hasta {worst:.0f}° durante más de 1 min"

    def _event(self, kind: Kind, start: float, end: float,
               detail: str) -> OfficeEvent:
        event = OfficeEvent(kind=kind, title=TITLES[kind], start=start,
                            end=end, detail=detail)
        self._report.events.append(event)
        return event

    @property
    def started(self) -> bool:
        """Whether the person has been at the screen at some point."""
        return self._last is not None

    def state(self) -> OfficeState:
        """What it shows now, for the drawing."""
        last = self._last or 0.0
        return OfficeState(
            self.view, self._face, self._rate(),
            last - (self._last_rest if self._last_rest is not None
                    else last),
            last - (self._last_break if self._last_break is not None
                    else last))

    def _rate(self) -> float | None:
        minutes = self._report.seconds_with_face / 60
        return self._report.blinks / minutes if minutes >= 0.5 else None

    def finish(self) -> list[OfficeEvent]:
        """End the warnings still going on (the end of the video)."""
        if self._last is None:
            return []
        ended = []
        for kind, held in self._held.items():
            span = held.add(self._last, False, 0.0)
            if span is not None:
                ended.append(self._event(held.kind, span[0], span[1],
                                         self._detail(kind, span[2])))
        return ended

    def report(self) -> OfficeReport | None:
        """The screen work seen, or ``None`` if the camera never had the
        screen's view."""
        report = self._report
        if report.seconds_at_screen <= 0:
            return None
        if self._distances:
            report.distance_cm = round(statistics.median(self._distances), 1)
            ordered = sorted(self._distances)
            report.closest_cm = round(ordered[len(ordered) // 20], 1)
        rate = self._rate()
        report.blinks_per_minute = round(rate, 1) if rate is not None else None
        report.seconds_at_screen = round(report.seconds_at_screen, 1)
        return report
