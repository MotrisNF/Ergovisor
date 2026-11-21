"""Pushing and pulling of loads seen in the skeleton, checked against the
INSST's guide on manual handling of loads (2024 edition).

What the guide asks of a push or a pull, and what a video shows of it:

- The hands between the height of the knuckles and that of the shoulders:
  outside it "el punto de aplicación de las fuerzas será excesivamente
  bajo o alto". Measured, as for the lifts.
- The trunk bent or twisted during the effort. Measured (the REBA
  angles).
- The force: at most 240 N to start and 150 N to keep moving for women,
  360 N and 250 N for men (ISO/TR 12295, table 3 of the guide). A camera
  cannot see a force: it is compared only when the person gives it.
- The start of the movement is the moment of most risk; the episodes keep
  when each one starts.

A push or a pull is the person moving across the image with the arms
reaching out (elbows fairly straight): towards where they face with the
hands in front is a push; backwards with the hands in front, or forwards
with the hands behind, is a pull. The 3D points of the pose model are
centred on the hips and do not move with the person, so the movement is
measured in the image, in statures: the camera must see the person from
the side or at an angle. Every threshold marked "ours" is to be
calibrated.

Like the rest, it only looks back: each episode is given when it ends.
"""
from collections import deque
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from pydantic import BaseModel

from .angles import Angles, body_axes
from .lifting import BodyHeights, Sex, place
from .skeleton import Point, Skeleton

Kind = Literal["empuje", "arrastre"]

# INSST guide 2024, table 3 (from ISO/TR 12295): force not to be passed,
# N, (initial, sustained), by sex.
FORCE_LIMITS: dict[str, tuple[float, float]] = {
    "mujer": (240.0, 150.0), "hombre": (360.0, 250.0)}
# Without the sex, the lower values (as for the reference weight).
FORCE_UNKNOWN = FORCE_LIMITS["mujer"]
# Ours: the hands at least this far (m) in front of the shoulders, or
# behind the hips, along the person's front.
REACH_M = 0.20
# Ours: the hands this far (statures) ahead of or behind the hips across
# the image tell a push from a pull (about 8 cm); closer, where the
# person faces does.
HANDS_LEAD = 0.05
# Ours: arms reaching out, elbows bent at most this much (0 is straight);
# carrying a box held close bends them more.
ELBOW_MAX = 70.0
# Ours: moving at least this many statures a second (about 0.2 m/s).
SPEED_STATURES = 0.12
# Ours: the speed is measured over at least this many seconds (over
# more when the frames come further apart), and over no more than
# ``SPEED_STALE`` (older positions are from before a gap).
SPEED_WINDOW = 0.5
SPEED_STALE = 1.5
# Ours: an episode lasts at least this long (s) and covers at least this
# distance (m); shorter gaps than ``GAP_SECONDS`` do not split it.
MIN_SECONDS = 1.5
MIN_METRES = 0.5
# (0.7: a real push of C1 has 0.6 s without the reach, someone walking
# with the arms ahead in C5 0.8 s between two bursts.)
GAP_SECONDS = 0.7
# A camera analysed live gives fewer frames when the computer is busy
# (seen on 2026-11-21: with two cameras, 0.2-0.3 s apart): a gap is at
# least this many times the usual time between two frames, so that one
# frame missed does not split an episode.
GAP_FRAMES = 2.5
# Ours: a push turning into a pull (or back) for this long (s) ends the
# episode and starts another; shorter, it is taken for noise.
SWITCH_SECONDS = 0.4
# Ours: a push or a pull is one steady effort: at least this share of the
# frames of the episode are of it. Walking with the arms ahead looks
# like it in bursts of 0.3 s every second (C5), which fewer frames would
# otherwise join into one.
DENSE = 0.6
# Times compared in seconds allow for the rounding of the frames' times.
EPSILON = 1e-6
# The usual time between two frames is taken over this many of them.
INTERVALS = 30
# REBA (NTP 601): the trunk scores more bent over 20°.
TRUNK_BENT = 20.0
# The twist REBA counts (as in timeline.py).
TRUNK_TWISTED = 15.0
# From body proportions (as in lifting.py: shoulders at about 0.82 of
# the stature, hips at 0.52): the trunk is about 0.30 of it.
TRUNK_OF_STATURE = 0.30
# Ours: a stature for a person whose height is not known, m (the model's
# own scale does not reach the image).
STATURE_M = 1.70
# The guide's methods for pushing and pulling (ISO 11228-2) apply when
# "se aplique la fuerza con ambas manos" on an object in front: both
# hands on it, at the same height and no further apart than a handle or
# a chair's back. Ours, calibrated on 2026-11-18: in the eleven real
# pushes and pulls of C1-C4 the hands' heights differed at most 9 cm
# (median of the episode) and they were at most 49 cm apart (90th
# percentile); someone walking past with the arms swinging (B6), 12 cm
# and 62 cm. The same height as for lifts (lifting.HANDS_LEVEL_CM).
HANDS_LEVEL_CM = 10.0
HANDS_APART_CM = 55.0


class Push(BaseModel):
    """One push or pull, from its start to its end.

    Attributes:
        kind: ``empuje`` or ``arrastre``.
        start: Second it started (the start is the riskiest moment);
            ``None`` in a report (not kept, see
            ``analysis.PersonAnalysis``).
        end: Second it ended (``None`` in a report).
        seconds: How long it lasted.
        metres: Distance covered, approximately.
        hands_low_cm: Lowest height of the hands above the floor.
        hands_high_cm: Highest height of the hands above the floor.
        trunk: Largest bend of the trunk, degrees.
        twist: Largest twist of the trunk, degrees.
        notes: What the INSST guide warns of in this push or pull.
    """

    kind: Kind
    start: float | None = None
    end: float | None = None
    seconds: float = 0.0
    metres: float
    hands_low_cm: float
    hands_high_cm: float
    trunk: float
    twist: float
    notes: list[str]


class PushingReport(BaseModel):
    """All the pushes and pulls of a recording.

    Attributes:
        method: What was measured and against what.
        pushes: Every push and pull, in order, without when.
        per_minute: How many a minute, over the time seen.
        force_initial_n: The force to start the movement, if given.
        force_sustained_n: The force to keep it moving, if given.
        force_limits_n: The guide's limits used (initial, sustained).
        notes: What the guide warns of for the whole task.
        assumptions: What the analysis assumes and the person should
            know.
    """

    method: str = ("Criterios de empuje y tracción de la Guía técnica de "
                   "manipulación manual de cargas del INSST (2024), con los "
                   "límites de fuerza de la norma ISO/TR 12295")
    pushes: list[Push]
    per_minute: float
    force_initial_n: float | None
    force_sustained_n: float | None
    force_limits_n: tuple[float, float]
    notes: list[str]
    assumptions: list[str]


def reaching(skeleton: Skeleton, angles: Angles | None
             ) -> Literal["front", "back"] | None:
    """Where the arms reach out, if they do: in front of the shoulders or
    behind the hips, with the elbows fairly straight."""
    if angles is None:
        return None
    arms = [arm for arm in (angles.left, angles.right) if arm is not None]
    if not arms or any(arm.elbow > ELBOW_MAX for arm in arms):
        return None
    hands = [skeleton.at(p) for p in (Point.LEFT_WRIST, Point.RIGHT_WRIST)
             if skeleton.seen(p)]
    if not hands:
        return None
    hand = np.mean(hands, axis=0)
    forward = body_axes(skeleton)[2]
    shoulders = skeleton.middle(Point.LEFT_SHOULDER, Point.RIGHT_SHOULDER)
    hips = skeleton.middle(Point.LEFT_HIP, Point.RIGHT_HIP)
    if float((hand - shoulders) @ forward) >= REACH_M:
        return "front"
    if float((hand - hips) @ forward) <= -REACH_M:
        return "back"
    return None


def _stature_px(skeleton: Skeleton) -> float | None:
    """The person's height in the image, in image heights (0 unknown)."""
    if skeleton.pixels is None:
        return None
    top = [skeleton.pixels[p] for p in (Point.NOSE,) if skeleton.seen(p)]
    feet = [skeleton.pixels[p] for p in (Point.LEFT_HEEL, Point.RIGHT_HEEL,
                                         Point.LEFT_ANKLE, Point.RIGHT_ANKLE)
            if skeleton.seen(p)]
    if top and feet:
        # The nose is at about 0.92 of the stature (as in lifting.py).
        height = (max(f[1] for f in feet) - top[0][1]) / 0.92
        return float(height) if height > 0 else None
    # Close to the camera the head or the feet go out of the picture
    # (seen on 2026-11-18: a quarter of the frames of a push): the trunk,
    # almost always seen, is about ``TRUNK_OF_STATURE`` of the stature.
    if not skeleton.seen(Point.LEFT_SHOULDER, Point.RIGHT_SHOULDER,
                         Point.LEFT_HIP, Point.RIGHT_HIP):
        return None
    px = skeleton.pixels
    trunk = ((px[Point.LEFT_HIP][1] + px[Point.RIGHT_HIP][1]) / 2
             - (px[Point.LEFT_SHOULDER][1] + px[Point.RIGHT_SHOULDER][1]) / 2)
    return float(trunk / TRUNK_OF_STATURE) if trunk > 0 else None


@dataclass
class _Step:
    second: float
    kind: Kind
    x: float
    stature: float
    hands: float | None
    trunk: float
    twist: float
    level: float = 0.0
    apart: float = 0.0


@dataclass
class Pushing:
    """Finds the pushes and pulls frame by frame.

    Attributes:
        sex: For the guide's force limits (empty: the lower ones).
        force_initial_n: The force to start the movement, if measured.
        force_sustained_n: The force to keep it moving, if measured.
        height_cm: The person's real height, for the distances.
    """

    sex: Sex = ""
    force_initial_n: float | None = None
    force_sustained_n: float | None = None
    height_cm: float | None = None
    pushes: list[Push] = field(default_factory=list)
    _track: deque[tuple[float, float, float]] = field(default_factory=deque)
    _steps: list[_Step] = field(default_factory=list)
    _body: BodyHeights = field(default_factory=BodyHeights)
    _first: float | None = None
    _last: float = 0.0
    _aspect: float = 1.0
    _steps_between: deque[float] = field(
        default_factory=lambda: deque(maxlen=INTERVALS))

    @property
    def _interval(self) -> float:
        """The usual time between two frames (0: not known yet): a low
        quantile, as a frame missed leaves a longer step."""
        if not self._steps_between:
            return 0.0
        return float(np.percentile(self._steps_between, 25))

    @property
    def stature_m(self) -> float:
        return (self.height_cm or 0) / 100 or STATURE_M

    def add(self, skeleton: Skeleton | None, angles: Angles | None,
            aspect: float, second: float | None = None) -> Push | None:
        """One frame (``aspect``: the image's width over its height;
        ``second``: its time, needed when there is no skeleton, so the
        frames not scored count in the time between frames); the push or
        pull that just ended, if any."""
        if second is None:
            second = skeleton.timestamp if skeleton is not None else None
        if second is not None:
            if self._first is None:
                self._first = second
            elif second > self._last:
                self._steps_between.append(second - self._last)
            self._last = second
        if skeleton is None or skeleton.pixels is None:
            return self._end_if_gap(self._last)
        second = skeleton.timestamp if second is None else second
        self._body.add(skeleton)
        stature = _stature_px(skeleton)
        hips = skeleton.pixels[[Point.LEFT_HIP, Point.RIGHT_HIP]].mean(axis=0)
        if stature is None:
            return self._end_if_gap(second)
        # Horizontal position in image heights, so it compares with the
        # stature measured vertically.
        self._aspect = aspect
        x = float(hips[0] * aspect)
        self._track.append((second, x, stature))
        # Keep the newest position at least ``SPEED_WINDOW`` old, so the
        # speed is measured over that long however far apart the frames.
        while (len(self._track) > 2
               and second - self._track[1][0] >= SPEED_WINDOW):
            self._track.popleft()
        while second - self._track[0][0] > SPEED_STALE:
            self._track.popleft()
        reach = reaching(skeleton, angles)
        kind = self._kind(skeleton, reach)
        if kind is None:
            return self._end_if_gap(second)
        # Frames not scored in between do not check the gap: check it
        # before this one joins the episode.
        ended = self._end_if_gap(second)
        spot = place(skeleton)
        left = skeleton.at(Point.LEFT_WRIST)
        right = skeleton.at(Point.RIGHT_WRIST)
        self._steps.append(_Step(
            second, kind, x, stature,
            spot.v if spot is not None else None,
            max(0.0, angles.trunk) if angles else 0.0,
            angles.trunk_twist if angles else 0.0,
            abs(float(left[1] - right[1])) * 100,
            float(np.linalg.norm(left - right)) * 100))
        return ended or self._split()

    def _split(self) -> Push | None:
        """A push turned into a pull, or the other way round, without a
        stop (pushing a chair away and pulling it back, 2026-11-21): the
        episode so far ends and the new one goes on. Measured as one,
        the way there and back cancelled out."""
        steps = self._steps
        run = len(steps)
        while run > 0 and steps[run - 1].kind != steps[0].kind:
            run -= 1
        if run == len(steps) or run == 0:
            return None
        if steps[-1].second - steps[run].second < SWITCH_SECONDS - EPSILON:
            return None
        if (steps[run - 1].second - steps[0].second + self._interval
                < MIN_SECONDS - EPSILON):
            # A doubtful start, not an episode of its own: kept in it.
            return None
        self._steps = steps[:run]
        ended = self._close()
        self._steps = steps[run:]
        return ended

    def _kind(self, skeleton: Skeleton, reach: str | None) -> Kind | None:
        """Push, pull or neither, from the reach and the movement."""
        if reach is None or len(self._track) < 2:
            return None
        (t0, x0, s0), (t1, x1, s1) = self._track[0], self._track[-1]
        if t1 - t0 < SPEED_WINDOW * 0.6 - EPSILON:
            return None
        speed = (x1 - x0) / ((s0 + s1) / 2) / (t1 - t0)
        if abs(speed) < SPEED_STATURES:
            return None
        lead = self._hands_lead(skeleton, speed, (s0 + s1) / 2)
        if lead is not None:
            # The hands ahead of the movement push; behind it, they pull
            # (reaching back, only that: backing into a thing is not
            # handling it).
            if reach == "back" and lead:
                return None
            return "empuje" if lead else "arrastre"
        # The person's front, seen in the image: x grows to the right in
        # both the 3D points and the image.
        facing = 1.0 if body_axes(skeleton)[2][0] >= 0 else -1.0
        ahead = speed * facing > 0
        if reach == "front":
            return "empuje" if ahead else "arrastre"
        return "arrastre" if ahead else None

    def _hands_lead(self, skeleton: Skeleton, speed: float,
                    stature: float) -> bool | None:
        """Whether the hands go ahead of the movement in the image (a
        push) or behind it (a pull); ``None`` when they are too close to
        the hips across the image to tell (moving towards or away from
        the camera). Unlike where the person faces, it is seen the same
        from any side: facing the camera, the front's sideways part is
        noise, and pushes were taken for pulls (2026-11-21)."""
        if skeleton.pixels is None or not stature:
            return None
        hands = [skeleton.pixels[p] for p in (Point.LEFT_WRIST,
                                              Point.RIGHT_WRIST)
                 if skeleton.seen(p)]
        if not hands:
            return None
        hips = skeleton.pixels[[Point.LEFT_HIP, Point.RIGHT_HIP]].mean(axis=0)
        ahead = float(np.mean([h[0] for h in hands]) - hips[0]) * self._aspect
        if abs(ahead) / stature < HANDS_LEAD:
            return None
        return ahead * speed > 0

    def _end_if_gap(self, second: float) -> Push | None:
        """Close the episode going on when it has stopped long enough."""
        gap = max(GAP_SECONDS, GAP_FRAMES * self._interval)
        if (self._steps
                and second - self._steps[-1].second >= gap - EPSILON):
            return self._close()
        return None

    def _close(self) -> Push | None:
        steps, self._steps = self._steps, []
        if not steps:
            return None
        # Each frame stands for the time until the next one.
        interval = self._interval
        duration = steps[-1].second - steps[0].second + interval
        statures = abs(steps[-1].x - steps[0].x) / (
            sum(s.stature for s in steps) / len(steps))
        metres = statures * self.stature_m
        if duration < MIN_SECONDS - EPSILON or metres < MIN_METRES:
            return None
        if interval and len(steps) < DENSE * duration / interval:
            return None
        if not self._both_hands(steps):
            return None
        kinds = [s.kind for s in steps]
        kind: Kind = max(("empuje", "arrastre"), key=kinds.count)
        heights = [s.hands for s in steps if s.hands is not None]
        k = self._scale()
        low = min(heights) * k if heights else 0.0
        high = max(heights) * k if heights else 0.0
        trunk = max(s.trunk for s in steps)
        twist = max(s.twist for s in steps)
        push = Push(kind=kind, start=round(steps[0].second, 2),
                    end=round(steps[-1].second, 2),
                    seconds=round(steps[-1].second - steps[0].second, 1),
                    metres=round(metres, 1),
                    hands_low_cm=round(low, 1), hands_high_cm=round(high, 1),
                    trunk=round(trunk, 1), twist=round(twist, 1),
                    notes=self._notes(kind, low, high, trunk, twist, k))
        self.pushes.append(push)
        return push

    def _both_hands(self, steps: list[_Step]) -> bool:
        """Whether both hands were on one object (see
        ``HANDS_LEVEL_CM``). The positions of both are used even when the
        far one is hidden by the body, seen from the side."""
        k = self._scale()
        level = float(np.median([s.level for s in steps])) * k
        apart = float(np.percentile([s.apart for s in steps], 90)) * k
        return level <= HANDS_LEVEL_CM and apart <= HANDS_APART_CM

    def _scale(self) -> float:
        if not self.height_cm or not self._body.stature:
            return 1.0
        return self.height_cm / self._body.stature

    def _notes(self, kind: Kind, low: float, high: float, trunk: float,
               twist: float, k: float) -> list[str]:
        """What the guide warns of in one push or pull."""
        notes = []
        body = self._body
        verb = "empujar" if kind == "empuje" else "tirar"
        if body.hip and low and low < body.knuckles * k:
            notes.append(f"manos por debajo de los nudillos al {verb} "
                         f"({low:.0f} cm del suelo): la fuerza se aplica "
                         "demasiado baja")
        if body.shoulder and high > body.shoulder * k:
            notes.append(f"manos por encima de los hombros al {verb} "
                         f"({high:.0f} cm del suelo): la fuerza se aplica "
                         "demasiado alta")
        if trunk > TRUNK_BENT:
            notes.append(f"tronco inclinado {trunk:.0f}° durante el "
                         "esfuerzo")
        if twist > TRUNK_TWISTED:
            notes.append(f"tronco girado {twist:.0f}° durante el esfuerzo")
        return notes

    def finish(self) -> Push | None:
        """End the episode going on (the end of the video)."""
        return self._close()

    def report(self) -> PushingReport | None:
        """The pushes and pulls seen, or ``None`` if there was none."""
        if not self.pushes:
            return None
        limits = FORCE_LIMITS.get(self.sex, FORCE_UNKNOWN)
        minutes = max((self._last - (self._first or self._last)) / 60, 1.0)
        notes = []
        for given, limit, moment in (
                (self.force_initial_n, limits[0], "inicial"),
                (self.force_sustained_n, limits[1], "mantenida")):
            if given is not None and given > limit:
                notes.append(f"fuerza {moment} de {given:.0f} N: más de "
                             f"{limit:.0f} N, el límite de la guía del "
                             "INSST (ISO/TR 12295)")
        assumptions = [
            "Empujar o arrastrar se reconoce por el desplazamiento con los "
            "brazos extendidos hacia la carga; la cámara debe ver a la "
            "persona de lado o en diagonal.",
            "La distancia recorrida es aproximada: sale de la imagen, "
            "escalada con la estatura.",
            "Los criterios de detección (velocidad, alcance de los brazos, "
            "duración mínima) están por calibrar.",
        ]
        if self.force_initial_n is None and self.force_sustained_n is None:
            assumptions.append(
                "La fuerza no se indicó: no se puede comparar con los "
                f"límites de la guía ({limits[0]:.0f} N inicial y "
                f"{limits[1]:.0f} N mantenida"
                f"{'' if self.sex else ', los menores al no indicar el sexo'}"
                "); se mide con un dinamómetro.")
        if not self.height_cm:
            assumptions.append(
                f"Sin la estatura real, las distancias suponen una persona "
                f"de {STATURE_M:.2f} m.")
        return PushingReport(
            pushes=[push.model_copy(update={"start": None, "end": None})
                    for push in self.pushes],
            per_minute=round(len(self.pushes) / minutes, 2),
            force_initial_n=self.force_initial_n,
            force_sustained_n=self.force_sustained_n,
            force_limits_n=limits, notes=notes, assumptions=assumptions)
