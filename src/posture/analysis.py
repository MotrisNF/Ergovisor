"""The whole analysis: frames in, moments of risk and a summary out.

It reads any source of frames (a file or a camera) one frame at a time
and calls ``on_event`` as soon as a moment of risk ends: printed for a
recording, it would be a warning for a live camera.
"""
from collections import deque
from collections.abc import Callable, Iterable
import copy
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel

from .angles import Angles, measure
from .face import Face, FaceEstimator
from .lifting import BodyHeights, Lift, Lifting, LiftingReport
from .office import Office, OfficeEvent, OfficeReport, OfficeState
from .people import MAX_PEOPLE, MIN_SECONDS, Tracker, box
from .pose import PoseEstimator
from .pushing import Push, Pushing, PushingReport
from .reba import Reba, Task
from .skeleton import Point, Skeleton
from .smoothing import Smoother
from .sources import Frame
from .timeline import Event, Summary, Timeline


@dataclass(frozen=True)
class Seen:
    """What one person shows in one frame, to draw it.

    Attributes:
        person: Their number (1, 2...).
        skeleton: The smoothed skeleton.
        angles: Its angles (``None`` at a screen).
        reba: Its score (``None`` at a screen or without an arm).
        event: The moment of risk going on.
        lift: Their last lift, while it is recent.
        office: At a screen, what the screen work shows.
    """

    person: int
    skeleton: Skeleton
    angles: Angles | None = None
    reba: Reba | None = None
    event: Event | None = None
    lift: Lift | None = None
    office: OfficeState | None = None


# Called with each analysed frame and what each person in it shows (to
# draw it). Returning False stops the analysis (a window closed by the
# person).
OnResult = Callable[[Frame, list[Seen]], bool | None]
# A lift stays on the drawing this long after it ends.
LIFT_SHOWN = 5.0
# Ours: a lift starting or ending this close (s) to a push or pull of the
# same person is the hands going onto or off the thing pushed, not a lift
# (C2, C3, C4: lowerings 0.2-1.0 s before or after the push or pull,
# 2026-11-18).
NEAR_PUSH = 1.5
# Ours: a point this far out of the picture (a share of it) still
# counts as in it; the ankles, this far below it: with the feet just cut
# by the bottom edge the model places them well (the pulls of C3, ankles
# at up to 1.19 of the image's height, were all measured), while walking
# up to the camera the knees leave the picture first (2026-11-20).
INSIDE = 0.02
ANKLES_BELOW = 0.2
# A face is a person's when it is this close (share of the image) to
# their nose.
FACE_NEAR = 0.15
# Ours: a person is near enough to measure their body when, standing, they
# fill at least this share of the image's height: the tallest they were
# seen whole in the last ``TALL_SECONDS`` (bending down for a lift, they
# are smaller for a moment). Further, the pose model's error is as big
# as the movements measured: people chatting at the end of a corridor
# (0.31 of the image) gave 14 lifts and 12 moments of risk in 35 minutes;
# the people of the calibration videos filled 0.54-0.97 standing (the
# median of each, 2026-11-18).
NEAR_ENOUGH = 0.4
TALL_SECONDS = 10.0
# Ours: a trunk bent further than this (degrees from vertical) is not a
# posture but the model's error (171° in a session at a laptop, REBA 10,
# 2026-11-20): bending to the floor with straight knees, the trunk is at
# about 110-130°.
TRUNK_MAX = 135.0

METHOD = ("REBA (Rapid Entire Body Assessment), según la NTP 601 del INSST "
          "(Hignett y McAtamney)")
UNMEASURED = ["hombro elevado", "brazo apoyado", "rotación del brazo",
              "torsión o desviación de muñeca"]


class RiskCount(BaseModel):
    """How often one cause of risk was seen (no times are kept: see
    ``PersonAnalysis``).

    Attributes:
        cause: What it was, as the bands of NTP 601 ("tronco flexionado
            más de 60°").
        times: In how many moments of risk.
        seconds: Their time in all.
        max_score: The highest REBA score of those moments.
    """

    cause: str
    times: int
    seconds: float
    max_score: int


class PersonAnalysis(BaseModel):
    """What the analysis found of one person.

    Only how often and how long things happened is kept, never when
    (the minute of a session, with its date, would tell who it was):
    the reports are anonymous, as the data protection rules (RGPD) ask
    of measures of people at work.

    Attributes:
        person: Their number in the recording (see ``people``).
        seconds_seen: How long they were in the picture.
        summary: Time at each level, the highest score...
        moments: How many moments of risk (REBA at or above the alert
            level).
        moments_by_level: Of them, how many at each action level.
        risks: Each cause of those moments, most frequent first.
        lifting: The lifts and lowerings scored with the NIOSH equation,
            or ``None`` when not analysed.
        office: The work at a screen, when the camera was the screen's
            (a person seated in front of it); ``None`` otherwise.
        pushing: The pushes and pulls of loads, or ``None`` when there
            was none or they were not looked for.
    """

    person: int = 1
    seconds_seen: float = 0.0
    summary: Summary
    moments: int = 0
    moments_by_level: dict[int, int] = {}
    risks: list[RiskCount] = []
    lifting: LiftingReport | None = None
    office: OfficeReport | None = None
    pushing: PushingReport | None = None


class Analysis(BaseModel):
    """What an analysis found.

    Attributes:
        source: The video file or the camera.
        created: The day of the analysis (local date, ``YYYY-MM-DD``;
            not the time, see ``PersonAnalysis``).
        method: The method and where it is described.
        task: The load and the grip given for the task.
        unmeasured: What the method asks that a skeleton cannot show:
            those points are never added, so the score may be lower than
            an evaluator's.
        max_people: The most people looked for at once.
        people: Each person seen, by number; empty when nobody was.
    """

    source: str
    created: str
    method: str = METHOD
    task: dict[str, float | bool | int]
    unmeasured: list[str] = UNMEASURED
    max_people: int = 1
    people: list[PersonAnalysis] = []


def people_of(analysis: dict[str, Any]) -> list[dict[str, Any]]:
    """The people of a saved analysis (a JSON object)."""
    return list(analysis.get("people", []))


def count_risks(events: list[Event]) -> list[RiskCount]:
    """How often and how long each cause was seen in some moments of
    risk, most frequent first (then longest)."""
    found: dict[str, RiskCount] = {}
    for event in events:
        for cause in dict.fromkeys(event.kinds):
            count = found.setdefault(cause, RiskCount(
                cause=cause, times=0, seconds=0.0, max_score=0))
            count.times += 1
            count.seconds += event.end - event.start
            count.max_score = max(count.max_score, event.score)
    for count in found.values():
        count.seconds = round(count.seconds, 1)
    return sorted(found.values(), key=lambda c: (-c.times, -c.seconds))


def whole_body(skeleton: Skeleton | None) -> bool:
    """Whether a frame shows the whole body: the hips and the knee and
    ankle of at least one leg.

    REBA, the lifts and the pushes are only scored then. With the legs
    out of the picture (standing up from a desk, or coming close to the
    camera) the model makes up the rest of the body: standing up from a
    laptop it read the trunk flexed 66-77° (REBA 10-11) and seven lifts
    in a session where the person only sat (calibrated on 2026-11-18).
    And it may go on calling them seen: a point out of the picture is
    not seen, whatever the model says (walking up to the camera, the
    knees were "bent" at made-up places, 2026-11-20; see ``INSIDE``).
    """
    if skeleton is None:
        return False

    def seen(*points: Point) -> bool:
        return skeleton.seen(*points) and all(
            _inside(skeleton, point) for point in points)

    return seen(Point.LEFT_HIP, Point.RIGHT_HIP) and (
        seen(Point.LEFT_KNEE, Point.LEFT_ANKLE)
        or seen(Point.RIGHT_KNEE, Point.RIGHT_ANKLE))


def _inside(skeleton: Skeleton, point: Point) -> bool:
    """Whether a point is in the picture (see ``INSIDE``); without the
    picture's positions, assumed so."""
    if skeleton.pixels is None:
        return True
    x, y = (float(v) for v in skeleton.pixels[point])
    below = (ANKLES_BELOW if point in (Point.LEFT_ANKLE, Point.RIGHT_ANKLE)
             else INSIDE)
    return -INSIDE <= x <= 1 + INSIDE and -INSIDE <= y <= 1 + below


def _nose(skeleton: Skeleton) -> NDArray[np.float64] | None:
    """Where the nose is in the image (0-1)."""
    if skeleton.pixels is None:
        return None
    return np.asarray(skeleton.pixels[Point.NOSE], dtype=float)


def match_faces(skeletons: dict[int, Skeleton], faces: list[Face]
                ) -> dict[int, Face]:
    """Each person's face: the nearest one to their nose (one face and
    one person: theirs)."""
    if len(skeletons) == 1 and len(faces) == 1:
        return {next(iter(skeletons)): faces[0]}
    pairs = sorted(
        (float(np.linalg.norm(np.asarray(face.centre) - nose)), number, n)
        for number, skeleton in skeletons.items()
        if (nose := _nose(skeleton)) is not None
        for n, face in enumerate(faces) if face.centre is not None)
    found: dict[int, Face] = {}
    used: set[int] = set()
    for distance, number, n in pairs:
        if distance <= FACE_NEAR and number not in found and n not in used:
            found[number] = faces[n]
            used.add(n)
    return found


@dataclass
class _Person:
    """The analysis of one person, frame by frame."""

    number: int
    timeline: Timeline
    lifting: Lifting | None
    office: Office | None
    pushing: Pushing | None
    smoother: Smoother = field(default_factory=Smoother)
    events: list[Event] = field(default_factory=list)
    # When they were seen whole and how tall (share of the image).
    tall: deque[tuple[float, float]] = field(default_factory=deque)
    heights: BodyHeights = field(default_factory=BodyHeights)
    seconds_seen: float = 0.0
    last_seen: float | None = None

    def saw(self, second: float) -> None:
        """In the picture in this frame."""
        if self.last_seen is not None:
            self.seconds_seen += min(max(0.0, second - self.last_seen),
                                     1.0)
        self.last_seen = second

    def near(self, skeleton: Skeleton | None) -> bool:
        """Whether the whole body is seen, near enough to measure it
        (see ``NEAR_ENOUGH``), with the head no higher than a person's
        (``BodyHeights.add``: the floor made up). A stature too short is
        not checked here: seated, or coming into the picture, a person
        is shorter (the lifts check it, see ``Lifting``)."""
        if not whole_body(skeleton):
            return False
        assert skeleton is not None
        if not self.heights.add(skeleton):
            return False
        place = box(skeleton)
        if place is None:
            return False
        self.tall.append((skeleton.timestamp,
                          min(place[3], 1.0) - max(place[1], 0.0)))
        while skeleton.timestamp - self.tall[0][0] > TALL_SECONDS:
            self.tall.popleft()
        return max(height for _, height in self.tall) >= NEAR_ENOUGH

    def report(self) -> PersonAnalysis:
        if self.lifting is not None and self.pushing is not None:
            self.lifting.leave_out([(p.start, p.end)
                                    for p in self.pushing.pushes
                                    if p.start is not None
                                    and p.end is not None], NEAR_PUSH)
        return PersonAnalysis(
            person=self.number, seconds_seen=round(self.seconds_seen, 1),
            summary=self.timeline.summary,
            moments=len(self.events),
            moments_by_level={
                level: sum(1 for e in self.events if e.level == level)
                for level in sorted({e.level for e in self.events})},
            risks=count_risks(self.events),
            lifting=self.lifting.report() if self.lifting else None,
            office=self.office.report() if self.office else None,
            pushing=self.pushing.report() if self.pushing else None)

    @property
    def seen(self) -> bool:
        """Whether they were in the picture long enough (not something
        taken for a person for an instant) and anything was measured."""
        report = self.report()
        return self.seconds_seen >= MIN_SECONDS and bool(
            report.summary.seconds >= MIN_SECONDS
            or (report.office
                and report.office.seconds_at_screen >= MIN_SECONDS)
            or (report.lifting and report.lifting.lifts)
            or (report.pushing and report.pushing.pushes))


def analyse(frames: Iterable[Frame], estimator: PoseEstimator,
            source: str, task: Task = Task(), alert_level: int = 2,
            seconds: float = 0.0,
            on_event: Callable[[int, Event], None] | None = None,
            on_frame: Callable[[Frame], None] | None = None,
            interruptible: bool = False,
            on_result: OnResult | None = None,
            lifting: Lifting | None = None,
            on_lift: Callable[[int, Lift], None] | None = None,
            office: Office | None = None,
            faces: FaceEstimator | None = None,
            on_office: Callable[[int, OfficeEvent], None] | None = None,
            pushing: Pushing | None = None,
            on_push: Callable[[int, Push], None] | None = None,
            max_people: int = MAX_PEOPLE
            ) -> Analysis:
    """Analyse frames until they end, or for ``seconds`` (0: no limit).

    Each person in the picture is analysed apart (see ``people``): the
    callbacks get the person's number first.

    Args:
        frames: A ``VideoFile`` or a ``Camera``.
        estimator: The pose model, already loaded.
        source: The name of the source, for the report.
        task: The load and the grip.
        alert_level: Action level from which a moment is of risk.
        seconds: Stop after this time (for a camera); 0 never stops.
        on_event: Called with each moment of risk as soon as it ends.
        on_frame: Called with each frame read (a progress bar).
        interruptible: Ctrl+C ends the reading and the analysis so far is
            kept, as a live camera is stopped; otherwise it stops
            everything.
        on_result: Called with each analysed frame and what was found in
            it (to draw it); returning False stops the analysis.
        lifting: Finds and scores the lifts, with the load and the
            person's data: each person gets a copy; ``None`` does not
            look for them.
        on_lift: Called with each lift as soon as it ends.
        office: Tells the camera's view from the skeleton and follows
            the work at a screen (a copy for each person); ``None`` only
            analyses the body. At a screen (the hips out of the picture),
            REBA and the lifts are not scored: without the legs they
            would be wrong.
        faces: The face model, used only at a screen.
        on_office: Called with each warning of the screen work as soon
            as it ends.
        pushing: Finds the pushes and pulls of loads (the whole body
            seen, from the side), a copy for each person; ``None`` does
            not look for them.
        on_push: Called with each push or pull as soon as it ends.
        max_people: The most people analysed at once.

    Returns:
        The summary and every moment of risk of each person.
    """
    tracker = Tracker(max_people)
    people: dict[int, _Person] = {}

    def person(number: int) -> _Person:
        if number not in people:
            people[number] = _Person(
                number, Timeline(task, alert_level),
                copy.deepcopy(lifting), copy.deepcopy(office),
                copy.deepcopy(pushing))
        return people[number]

    def keep(who: _Person, event: Event | None) -> None:
        if event is not None:
            who.events.append(event)
            if on_event is not None:
                on_event(who.number, event)

    def push_ended(who: _Person, push: Push | None) -> None:
        if push is not None and on_push is not None:
            on_push(who.number, push)

    def tell(who: _Person, found: list[OfficeEvent]) -> None:
        if on_office is not None:
            for warning in found:
                on_office(who.number, warning)

    def step(who: _Person, frame: Frame, raw: Skeleton | None,
             face: Face | None, desk: bool) -> Seen | None:
        """One frame of one person (``raw``: ``None`` when gone)."""
        skeleton = angles = None
        if raw is None:
            who.smoother.reset()
        else:
            who.saw(frame.timestamp)
            skeleton = who.smoother.add(raw)
            angles = measure(skeleton)
        state = None
        if desk:
            assert who.office is not None
            tell(who, who.office.add(frame.timestamp, skeleton, face))
            state = who.office.state()
        elif who.office is not None and who.office.started:
            # Up from the screen (the whole body seen): away from it,
            # as if gone, so standing up is a break.
            tell(who, who.office.add(frame.timestamp, None, None))
        # The body is only scored when it is seen whole, near enough, in a
        # posture a person can take.
        body = (not desk and who.near(skeleton) and angles is not None
                and angles.trunk <= TRUNK_MAX)
        result, ended = who.timeline.add(frame.timestamp,
                                         angles if body else None)
        keep(who, ended)
        if who.lifting is not None:
            for lift in who.lifting.add(skeleton if body else None):
                if on_lift is not None:
                    on_lift(who.number, lift)
        if who.pushing is not None:
            height, width = frame.image.shape[:2]
            push_ended(who, who.pushing.add(
                skeleton if body else None, angles if body else None,
                width / height if height else 1.0, frame.timestamp))
        if skeleton is None:
            return None
        recent = None
        if who.lifting is not None and who.lifting.lifts:
            lift_end = who.lifting.lifts[-1].end.second
            if (lift_end is not None
                    and frame.timestamp - lift_end <= LIFT_SHOWN):
                recent = who.lifting.lifts[-1]
        return Seen(who.number, skeleton, angles if body else None,
                    result if body else None, who.timeline.current,
                    recent, state)

    try:
        for frame in frames:
            if seconds and frame.timestamp > seconds:
                break
            if on_frame is not None:
                on_frame(frame)
            found = tracker.assign(frame.timestamp,
                                   estimator.find_all(frame))
            for number in found:
                person(number)
            desks = {number for number, who in people.items()
                     if who.office is not None and who.office.see(
                         frame.timestamp, found.get(number)) == "desk"}
            matched: dict[int, Face] = {}
            if faces is not None and desks & set(found):
                matched = match_faces(
                    {n: s for n, s in found.items() if n in desks},
                    faces.find_all(frame))
            shown = []
            for number, who in sorted(people.items()):
                raw = found.get(number)
                if raw is None and tracker.present(number, frame.timestamp):
                    # Missing for an instant: still there (see people).
                    continue
                seen = step(who, frame, raw, matched.get(number),
                            number in desks)
                if seen is not None:
                    shown.append(seen)
            if on_result is not None and on_result(frame, shown) is False:
                break
    except KeyboardInterrupt:
        if not interruptible:
            raise
    for who in people.values():
        keep(who, who.timeline.finish())
        if who.office is not None:
            tell(who, who.office.finish())
        if who.pushing is not None:
            push_ended(who, who.pushing.finish())
    return Analysis(
        source=source, created=date.today().isoformat(),
        task={"load_kg": task.load_kg, "sudden": task.sudden,
              "coupling": task.coupling},
        max_people=max_people,
        people=numbered([who.report() for _, who in sorted(people.items())
                         if who.seen]))


def free_name(folder: Path, stem: str, suffix: str = ".json",
              number: bool = False) -> str:
    """The first name ``stem`` (or ``stem_2``, ``stem_3``...; always
    numbered from ``stem_1`` with ``number``) with no ``suffix`` file in
    ``folder``. Sessions are named by their day and this number, never
    by their time (see ``PersonAnalysis``)."""
    n = 1
    while True:
        name = f"{stem}_{n}" if number or n > 1 else stem
        if not (folder / f"{name}{suffix}").exists():
            return name
        n += 1


def numbered(people: list[PersonAnalysis]) -> list[PersonAnalysis]:
    """The people reported, numbered again from 1 in order: a number
    taken by something seen for an instant (left out) leaves no gap, so
    one person is always "Persona 1"."""
    return [person.model_copy(update={"person": number})
            for number, person in enumerate(people, start=1)]


def _span(start: float | None, end: float | None) -> str:
    """``[mm:ss–mm:ss] `` while analysing; nothing once it is not kept."""
    if start is None or end is None:
        return ""

    def clock(second: float) -> str:
        return f"{int(second // 60):02d}:{second % 60:04.1f}"

    return f"[{clock(start)}–{clock(end)}] "


def describe_lift(lift: Lift) -> str:
    """One line for a lift: when, how much, its limit and its index."""
    index = (f", IL {lift.index:.2f} ({lift.risk.lower()})"
             if lift.index is not None
             else f" ({lift.risk.lower()})" if lift.beyond_limits else "")
    notes = f". {'; '.join(lift.notes)}" if lift.notes else ""
    return (f"{_span(lift.start.second, lift.end.second)}"
            f"{lift.kind.capitalize()} de {lift.d:.0f} cm: LPR "
            f"{lift.limit:.1f} kg{index}{notes}")


def describe(event: Event) -> str:
    """One line for a moment of risk: when, how much and why."""
    return (f"{_span(event.start, event.end)}Riesgo "
            f"{event.risk.lower()} (REBA {event.score}, nivel "
            f"{event.level}): {', '.join(event.causes) or 'postura'}")


def describe_push(push: Push) -> str:
    """One line for a push or a pull."""
    notes = f". {'; '.join(push.notes)}" if push.notes else ""
    return (f"{_span(push.start, push.end)}"
            f"{push.kind.capitalize()} de {push.metres:.1f} m, manos a "
            f"{push.hands_low_cm:.0f}–{push.hands_high_cm:.0f} cm del "
            f"suelo{notes}")


def describe_office(event: OfficeEvent) -> str:
    """One line for a warning of the screen work."""
    return (f"{_span(event.start, event.end)}{event.title}: "
            f"{event.detail}")
