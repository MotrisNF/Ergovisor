"""What happens over time: held, repeated and sudden postures, the
moments of risk, and the summary of a whole recording.

Everything is updated one frame at a time and only looks back, so a live
camera gives the same results as a recording of it. A moment of risk is
given as soon as it ends, which is when a live system would warn.

NTP 601 adds one activity point for each of: a body part held more than
a minute, a movement repeated more than 4 times a minute, and large or
sudden changes of posture. How to see those in a skeleton is ours, in
the constants below.
"""
from collections import deque
from dataclasses import dataclass, field

from pydantic import BaseModel

from .angles import Angles
from .reba import SIDE_BEND, TWIST, Reba, Task, score

# NTP 601: held more than a minute; repeated more than 4 times a minute.
HOLD_SECONDS = 60.0
REPEATS_PER_MINUTE = 4
# Ours: a part is still while it stays within this many degrees.
HOLD_TOLERANCE = 10.0
# Ours: a movement repeats when the angle goes above ``high`` and back
# below ``low``: trunk bending and arm raising.
TRUNK_RISE = (20.0, 10.0)
ARM_RISE = (45.0, 30.0)
# Ours: a sudden change is the trunk moving this much in that time.
SUDDEN_DEGREES = 45.0
SUDDEN_SECONDS = 2.0
# Ours: a gap without risk shorter than this does not end a moment of
# risk, and a moment shorter than ``MIN_EVENT`` is not reported.
GAP_SECONDS = 1.0
MIN_EVENT = 1.0
# A frame stands for the time until the next one, at most this long
# (longer gaps are frames without a person).
MAX_FRAME_SECONDS = 1.0


class Event(BaseModel):
    """A moment of risk: consecutive frames at or above a level.

    Attributes:
        start: Second it starts.
        end: Second it ends.
        score: The highest REBA score in it.
        level: Its action level (0 to 4).
        risk: The risk, in words (NTP 601).
        action: The intervention needed (NTP 601).
        causes: What makes the posture risky, at its worst moment.
    """

    start: float
    end: float
    score: int
    level: int
    risk: str
    action: str
    causes: list[str]


class Summary(BaseModel):
    """A whole recording, or the camera so far.

    Attributes:
        seconds: Time covered by the frames analysed.
        frames: Frames analysed.
        frames_with_person: Frames where a skeleton was measured.
        seconds_by_level: Seconds spent at each action level, 0 to 4.
        max_score: The highest REBA score.
        max_level: Its action level.
    """

    seconds: float = 0.0
    frames: int = 0
    frames_with_person: int = 0
    seconds_by_level: dict[int, float] = {}
    max_score: int = 0
    max_level: int = 0


@dataclass
class _Still:
    """How long one angle has stayed where it is."""

    anchor: float
    since: float


@dataclass
class _Rises:
    """The times an angle went up past a threshold."""

    high: float
    low: float
    up: bool = False
    times: deque[float] = field(default_factory=deque)

    def add(self, second: float, angle: float) -> None:
        """Count a rise when the angle passes ``high`` from below ``low``."""
        if not self.up and angle > self.high:
            self.up = True
            self.times.append(second)
        elif self.up and angle < self.low:
            self.up = False
        while self.times and second - self.times[0] > 60:
            self.times.popleft()


class Timeline:
    """Scores each frame with its activity and finds the moments of risk.

    Attributes:
        task: The load and grip, which the video does not show.
        alert_level: Action level from which a moment is of risk (2 is
            "Medio", where NTP 601 says intervention is needed).
    """

    def __init__(self, task: Task = Task(), alert_level: int = 2) -> None:
        """Nothing seen yet."""
        self.task = task
        self.alert_level = alert_level
        self.summary = Summary(seconds_by_level={n: 0.0 for n in range(5)})
        self._still: dict[str, _Still] = {}
        self._rises = {"tronco": _Rises(*TRUNK_RISE),
                       "brazo izquierdo": _Rises(*ARM_RISE),
                       "brazo derecho": _Rises(*ARM_RISE)}
        self._trunk: deque[tuple[float, float]] = deque()
        self._last: tuple[float, int] | None = None
        self._open: Event | None = None
        self._last_risky = 0.0

    def _held(self, second: float, parts: dict[str, tuple[float, int]]
              ) -> bool:
        """Whether a part out of its neutral position is held still more
        than a minute. ``parts`` gives each part's angle and REBA score."""
        held = False
        for name, (angle, points) in parts.items():
            still = self._still.get(name)
            if still is None or abs(angle - still.anchor) > HOLD_TOLERANCE:
                self._still[name] = still = _Still(angle, second)
            if points > 1 and second - still.since > HOLD_SECONDS:
                held = True
        return held

    def _repeated(self, second: float, angles: Angles) -> bool:
        """Whether a movement went up more than 4 times in the last
        minute."""
        self._rises["tronco"].add(second, angles.trunk)
        for name, arm in (("brazo izquierdo", angles.left),
                          ("brazo derecho", angles.right)):
            if arm is not None:
                self._rises[name].add(second, arm.upper)
        return any(len(r.times) > REPEATS_PER_MINUTE
                   for r in self._rises.values())

    def _sudden(self, second: float, trunk: float) -> bool:
        """Whether the trunk moved more than ``SUDDEN_DEGREES`` in the
        last ``SUDDEN_SECONDS``."""
        self._trunk.append((second, trunk))
        while second - self._trunk[0][0] > SUDDEN_SECONDS:
            self._trunk.popleft()
        values = [angle for _, angle in self._trunk]
        return max(values) - min(values) > SUDDEN_DEGREES

    def add(self, second: float, angles: Angles | None
            ) -> tuple[Reba | None, Event | None]:
        """One frame: its score, and a moment of risk if one just ended.

        Args:
            second: When the frame was taken.
            angles: Its angles, or ``None`` without a person.

        Returns:
            The REBA score of the frame (``None`` without a person or an
            arm), and the moment of risk that ended before it, if any.
        """
        self.summary.frames += 1
        if self._last is not None:
            gap = second - self._last[0]
            if 0 < gap <= MAX_FRAME_SECONDS:
                self.summary.seconds += gap
                self.summary.seconds_by_level[self._last[1]] += gap
        result = None if angles is None else self._score(second, angles)
        if angles is None or result is None:
            self._last = None
            return None, self._close_if_over(second)
        self.summary.frames_with_person += 1
        self._last = (second, result.level)
        if result.score > self.summary.max_score:
            self.summary.max_score = result.score
            self.summary.max_level = result.level
        ended = None
        if result.level >= self.alert_level:
            self._extend(second, result, angles)
        else:
            ended = self._close_if_over(second)
        return result, ended

    def _score(self, second: float, angles: Angles) -> Reba | None:
        """The frame's score, with the activity points."""
        plain = score(angles, self.task)
        if plain is None:
            return None
        parts = {"tronco": (angles.trunk, plain.trunk),
                 "cuello": (angles.neck, plain.neck)}
        for name, arm in (("brazo izquierdo", angles.left),
                          ("brazo derecho", angles.right)):
            if arm is not None:
                parts[name] = (arm.upper, plain.upper_arm)
        activity = sum((self._held(second, parts),
                        self._repeated(second, angles),
                        self._sudden(second, angles.trunk)))
        return score(angles, self.task, activity)

    def _extend(self, second: float, result: Reba, angles: Angles) -> None:
        """Add a risky frame to the open moment, or open one."""
        self._last_risky = second
        worse = self._open is None or result.score > self._open.score
        if self._open is None:
            self._open = Event(start=second, end=second, score=0, level=0,
                               risk="", action="", causes=[])
        self._open.end = second
        if worse:
            self._open.score = result.score
            self._open.level = result.level
            self._open.risk = result.risk
            self._open.action = result.action
            self._open.causes = causes(result, angles)

    def _close_if_over(self, second: float) -> Event | None:
        """Close the open moment when the risk has been gone long
        enough; it is given only if it lasted long enough."""
        if (self._open is None
                or second - self._last_risky <= GAP_SECONDS):
            return None
        return self._take()

    def _take(self) -> Event | None:
        """The open moment, if long enough; it is closed either way."""
        event, self._open = self._open, None
        if event is None or event.end - event.start < MIN_EVENT:
            return None
        return event

    def finish(self) -> Event | None:
        """The moment of risk still open at the end of the recording."""
        return self._take()


def causes(result: Reba, angles: Angles) -> list[str]:
    """What makes a posture risky, in words, with its angles."""
    found = []
    if result.trunk >= 3:
        kind = "flexionado" if angles.trunk > 0 else "en extensión"
        found.append(f"tronco {kind} {abs(angles.trunk):.0f}°")
    elif result.trunk == 2 and abs(angles.trunk) >= 5:
        found.append(f"tronco inclinado {abs(angles.trunk):.0f}°")
    if angles.trunk_twist > TWIST or angles.trunk_side > SIDE_BEND:
        found.append("tronco girado o inclinado de lado")
    if result.neck >= 2:
        found.append(f"cuello flexionado {angles.neck:.0f}°"
                     if angles.neck > 0 else "cuello en extensión")
    if result.legs >= 3:
        found.append(f"rodillas flexionadas {angles.knee:.0f}°")
    if angles.one_leg:
        found.append("apoyo sobre una pierna")
    arm = angles.left if result.side == "izquierdo" else angles.right
    if arm is not None and result.upper_arm >= 3:
        found.append(f"brazo {result.side} elevado {abs(arm.upper):.0f}°")
    if result.activity:
        found.append(f"{result.activity} punto(s) de actividad "
                     "(postura mantenida, repetida o brusca)")
    return found
