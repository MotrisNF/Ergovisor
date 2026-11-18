"""Lifts and lowerings seen in the skeleton, scored with the NIOSH
equation and checked against the INSST's guide on manual handling of
loads (2024 edition).

What the equation needs is measured on the skeleton:

- V: the height of the hands above the floor (the lowest point of the
  feet).
- H: how far the hands are, horizontally, in front of the middle of the
  ankles (along the person's front).
- A: the twist of the trunk (the shoulders against the hips) while
  handling the load.

H and A are taken along the person's front and from the trunk because,
seen from the side, the depth of the points (towards or away from the
camera) is the least certain measure of the pose model: measured with it,
a lift without any twist read 50-85° of A and the hands 15-20 cm further
(calibrated on 2026-11-18).
- D: how much the hands go up or down in a lift.
- F: how many lifts a minute.

A lift (or a lowering) is the hands going up (or down) at least 25 cm
between two turning points: where they turn back, or stop. Natural
handling does not stop at the floor (the load is picked up in one
movement), so a pause is not needed (calibrated on 2026-11-18: waiting
for pauses found 0 to 5 of 6 to 8 real movements).

A pose model does not see the load, so the movements that carry it are
told apart from the empty ones (reaching down for the load, standing up
after leaving it) by their pattern: every time the hands go down to a
low point the load is handled once there, picked up (the way up carries
it) or left (the way down does). Of the two, the one that rises higher
is kept; both are kept when both rise clearly above the hanging hands
(leaving a load on a low shelf and taking it again). A movement that
rises clearly above the hanging hands, or with the hands held together,
is a loaded one. Every movement kept counts as handling the load given
by the person; this is said in the report.

Like the rest, it only looks back: each lift is given when it ends.
"""
from collections import deque
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from pydantic import BaseModel

from .angles import UP, Vector, body_axes
from .niosh import (D_MIN, DURATIONS, F_MIN, ZERO_FACTORS, Duration, Grip,
                    Niosh, evaluate)
from .skeleton import Point, Skeleton

Sex = Literal["hombre", "mujer", ""]

# Ours: the hands are still when they moved less than this (cm) in the
# last ``STILL_SECONDS``: a pause also ends a movement.
STILL_CM = 3.0
STILL_SECONDS = 1.0
# Ours: a movement turns back when the hands come back this much (cm)
# from its furthest point.
REVERSAL_CM = 10.0
# Ours, calibrated on 2026-11-18: a movement clearly carries the load
# when its high end is at least this far (cm) above the hanging hands
# (loaded to the chest: 25-45 cm above; standing up empty: 7-14 cm), or
# the hands are this close together (cm) there (holding it in front:
# 18-20 cm with the "full" model; hanging apart: 37-49). A box carried
# at the waist is neither, so this only decides between two movements
# around the same low point (see the module).
LOADED_ABOVE_CM = 15.0
HANDS_TOGETHER_CM = 25.0
# A lift is made with both hands holding the load: the NIOSH equation
# "no está diseñada para evaluar tareas en las que la carga se levante con
# una sola mano" (NTP 477). Holding one object, the hands move together:
# the same way and as far, at the same height, at a steady distance from
# each other (the object). Moving the hands while talking, crossing the
# arms or swinging them walking does not. Ours, calibrated on 2026-11-18
# on the 16 real lifts of B5, B7 and the sample video (each hand's
# distance differed at most 19 %, their heights 4 cm, their distance from
# each other changed at most 24 cm) and a recording of three people
# moving freely (most of its false lifts broke one of them): with some
# margin, each hand's distance may differ this share of the longer one,
# their heights this much (cm, the median), and the distance between
# them this much (cm, from its 10th to its 90th percentile). The
# positions of both hands are used even when the model does not call
# one of them seen: from the side, the far hand is hidden by the body
# and every real lift would fail.
HANDS_SAME_WAY = 0.25
HANDS_LEVEL_CM = 10.0
HANDS_STEADY_CM = 30.0
# How long the hands are kept, s (longer than any lift).
HANDS_KEPT = 120.0
# Ours, from body proportions: the knuckles hang at about 0.83 times the
# height of the hip joint (about 0.43 and 0.52 of the stature).
KNUCKLES_OF_HIP = 0.83
# Ours, from body proportions: the nose is at about 0.92 of the stature;
# with the real stature given, every distance is scaled by it.
NOSE_OF_STATURE = 0.92

# INSST guide 2024, table 2 (from ISO 11228-1): heaviest load in ideal
# conditions, kg, by sex and age.
REFERENCE_KG = {("hombre", True): 25.0, ("hombre", False): 20.0,
                ("mujer", True): 20.0, ("mujer", False): 15.0}
# Without sex and age, the lowest value for people of 20 to 45.
REFERENCE_UNKNOWN = 20.0
# INSST guide 2024, table 4 (from ISO 11228-1): lifts a minute above
# which the risk is high, by duration of the task.
FREQUENCY_LIMIT = {"corta": 15.0, "moderada": 12.0, "larga": 10.0}
# INSST guide 2024: recommended horizontal distance, cm.
H_RECOMMENDED = 25.0


class Place(BaseModel):
    """Where the hands are at the start or the end of a lift.

    Attributes:
        second: When.
        v: Height of the hands above the floor, cm.
        h: Horizontal distance from the middle of the ankles, cm.
        a: Twist from the front of the feet, degrees.
    """

    second: float
    v: float
    h: float
    a: float


class Lift(BaseModel):
    """One lift or lowering, scored with the NIOSH equation.

    Attributes:
        kind: ``levantamiento`` (up) or ``descenso`` (down).
        start: Where the hands were before moving.
        end: Where they stopped.
        d: Vertical travel, cm.
        limit: Recommended weight limit (LPR), kg: the lower of the start
            and the end, as NIOSH asks when the load is controlled at
            both.
        index: Lifting index (IL), or ``None`` without a load or when
            the limit is 0 kg.
        risk: What NTP 477 says of that index (with a limit of 0 kg, an
            "incremento acusado del riesgo" whatever the load).
        factors: HM, VM, DM, AM, FM and CM where the limit is lowest.
        notes: What the INSST guide warns of in this lift.
        beyond_limits: The limit is 0 kg: some measure is beyond what the
            equation allows (see the notes): the lift is not acceptable
            with any load.
    """

    kind: str
    start: Place
    end: Place
    d: float
    limit: float
    index: float | None
    risk: str
    factors: dict[str, float]
    notes: list[str]
    beyond_limits: bool = False


class LiftingReport(BaseModel):
    """All the lifts of a recording.

    Attributes:
        method: The method and where it is described.
        load_kg: The load given by the person.
        grip: How good the grip is.
        duration: How long the task lasts, as NIOSH counts it.
        lifts_per_minute: The frequency used (counted, or given).
        reference_kg: The INSST guide's heaviest load for the person.
        lifts: Every lift and lowering, in order.
        max_index: The highest lifting index (of the lifts with one).
        beyond_limits: Lifts with a limit of 0 kg (no index: not
            acceptable with any load).
        notes: What the INSST guide warns of for the whole task.
        assumptions: What the analysis assumes and the person should
            know.
        scale: How much the model's distances were stretched to the real
            height (1 without it).
        stature_cm: The person's height used, cm.
    """

    method: str = ("Ecuación de NIOSH revisada (NTP 477 del INSST), con los "
                   "criterios de la Guía técnica de manipulación manual de "
                   "cargas del INSST (2024)")
    load_kg: float | None
    grip: str
    duration: str
    lifts_per_minute: float
    reference_kg: float
    lifts: list[Lift]
    max_index: float | None
    beyond_limits: int = 0
    notes: list[str]
    assumptions: list[str]
    scale: float = 1.0
    stature_cm: float = 0.0


def _seen_mean(skeleton: Skeleton, points: list[Point]) -> Vector | None:
    """The middle of the points seen, or ``None`` if none is."""
    seen = [skeleton.at(p) for p in points if skeleton.seen(p)]
    return np.mean(seen, axis=0) if seen else None


def _height(point: Vector, floor: float) -> float:
    """Height above the floor, cm (y points down)."""
    return float((floor - point[1]) * 100)


def place(skeleton: Skeleton) -> Place | None:
    """Where the hands are, or ``None`` without hands or feet seen."""
    hands = _seen_mean(skeleton, [Point.LEFT_WRIST, Point.RIGHT_WRIST,
                                  Point.LEFT_INDEX, Point.RIGHT_INDEX])
    feet = [p for p in (Point.LEFT_HEEL, Point.RIGHT_HEEL, Point.LEFT_FOOT,
                        Point.RIGHT_FOOT, Point.LEFT_ANKLE,
                        Point.RIGHT_ANKLE) if skeleton.seen(p)]
    ankles = _seen_mean(skeleton, [Point.LEFT_ANKLE, Point.RIGHT_ANKLE])
    if hands is None or not feet or ankles is None:
        return None
    floor = max(float(skeleton.at(p)[1]) for p in feet)
    reach = hands - ankles
    _, lateral, forward = body_axes(skeleton)
    h = abs(float(reach @ forward)) * 100
    twist = 0.0
    if skeleton.seen(Point.LEFT_SHOULDER, Point.RIGHT_SHOULDER):
        shoulders = (skeleton.at(Point.RIGHT_SHOULDER)
                     - skeleton.at(Point.LEFT_SHOULDER))
        shoulders = shoulders - float(shoulders @ UP) * UP
        cosine = float(shoulders @ lateral) / float(
            np.linalg.norm(shoulders) or 1.0)
        twist = float(np.degrees(np.arccos(np.clip(cosine, -1, 1))))
    return Place(second=skeleton.timestamp, v=_height(hands, floor), h=h,
                 a=twist)


@dataclass
class _Body:
    """The person's heights standing, to know where the knuckles and the
    shoulders are."""

    hip: float = 0.0
    shoulder: float = 0.0
    head: float = 0.0

    def add(self, skeleton: Skeleton) -> None:
        """Keep the highest heights seen: standing upright."""
        feet = [p for p in (Point.LEFT_HEEL, Point.RIGHT_HEEL,
                            Point.LEFT_FOOT, Point.RIGHT_FOOT)
                if skeleton.seen(p)]
        if not feet:
            return
        floor = max(float(skeleton.at(p)[1]) for p in feet)
        for name, points in (("hip", [Point.LEFT_HIP, Point.RIGHT_HIP]),
                             ("shoulder", [Point.LEFT_SHOULDER,
                                           Point.RIGHT_SHOULDER]),
                             ("head", [Point.NOSE])):
            middle = _seen_mean(skeleton, points)
            if middle is not None:
                setattr(self, name, max(getattr(self, name),
                                        _height(middle, floor)))

    @property
    def knuckles(self) -> float:
        """Height of the knuckles with the arms hanging, cm."""
        return self.hip * KNUCKLES_OF_HIP

    @property
    def stature(self) -> float:
        """The person's height as the model measures it, cm (0 unknown)."""
        return self.head / NOSE_OF_STATURE


@dataclass
class Lifting:
    """Finds the lifts frame by frame and scores them.

    Attributes:
        load_kg: The load handled, or ``None`` if not given.
        grip: How good the grip on it is.
        duration: How long the task lasts.
        reference_kg: The INSST guide's heaviest load for the person.
        lifts_per_minute: The frequency, or 0 to count it.
        height_cm: The person's real height: every distance is scaled so
            the model's measure of it matches; ``None`` keeps the model's
            scale.
    """

    load_kg: float | None = None
    grip: Grip = "bueno"
    duration: Duration = "larga"
    reference_kg: float = REFERENCE_UNKNOWN
    lifts_per_minute: float = 0.0
    height_cm: float | None = None
    lifts: list[Lift] = field(default_factory=list)
    # The start and end of each lift at the model's scale, to score them
    # again at the end with the scale and frequency of the whole video.
    _raw: list[tuple[Place, Place]] = field(default_factory=list)
    _recent: deque[Place] = field(default_factory=deque)
    # The last turning point, the furthest point since, and the
    # direction of the movement (1 up, -1 down, 0 not moving), each place
    # with whether the hands were together.
    _turn: tuple[Place, bool] | None = None
    _extreme: tuple[Place, bool] | None = None
    _direction: int = 0
    # The last three turning points confirmed, to find the low points.
    _turns: list[tuple[Place, bool]] = field(default_factory=list)
    # Each frame's heights of the left and right hand and the distance
    # between them (cm, the model's scale), to tell two-handed lifts.
    _hands: deque[tuple[float, float, float, float]] = field(
        default_factory=deque)
    # The last low point scored and how many lifts it gave: a pause half
    # way merged afterwards scores it again, replacing them.
    _scored_low: tuple[Place, bool] | None = None
    _scored: int = 0
    # Pushes and pulls whose lifts are left out (see ``leave_out``).
    _spans: list[tuple[float, float]] = field(default_factory=list)
    _margin: float = 0.0
    _first: float | None = None
    _last: float = 0.0
    _body: _Body = field(default_factory=_Body)

    def add(self, skeleton: Skeleton | None) -> list[Lift]:
        """One frame; the lifts that just ended (usually none or one;
        two when a load is left low and taken again)."""
        if skeleton is None:
            return []
        self._body.add(skeleton)
        now = place(skeleton)
        if now is None:
            return []
        self._keep_hands(skeleton)
        if self._first is None:
            self._first = now.second
        self._last = now.second
        self._recent.append(now)
        while now.second - self._recent[0].second > STILL_SECONDS:
            self._recent.popleft()
        held = self._together(skeleton)
        if self._turn is None or self._extreme is None:
            self._turn = self._extreme = (now, held)
            self._turns = [self._turn]
            return []
        k = self.scale()
        turn, extreme = self._turn[0], self._extreme[0]
        if self._direction == 0:
            # Not moving: start a movement once the hands left the turn,
            # or follow them while they drift a little.
            if abs(now.v - turn.v) * k >= REVERSAL_CM:
                self._direction = 1 if now.v > turn.v else -1
                self._extreme = (now, held)
            elif self._still():
                self._turn = self._extreme = (now, held)
                self._turns[-1] = self._turn
            return []
        if (now.v - extreme.v) * self._direction >= 0:
            self._extreme = (now, held)
            return self._confirm(0) if self._still() else []
        if abs(now.v - extreme.v) * k >= REVERSAL_CM:
            return self._confirm(-self._direction, (now, held))
        return self._confirm(0) if self._still() else []

    def _still(self) -> bool:
        """Whether the hands stayed put over the last ``STILL_SECONDS``."""
        if (self._recent[-1].second - self._recent[0].second
                < STILL_SECONDS * 0.8):
            return False
        travel = (max(p.v for p in self._recent)
                  - min(p.v for p in self._recent))
        return travel * self.scale() < STILL_CM

    @staticmethod
    def _together(skeleton: Skeleton) -> bool:
        """Whether both hands are seen close together (holding something
        in front)."""
        if not skeleton.seen(Point.LEFT_WRIST, Point.RIGHT_WRIST):
            return False
        apart = float(np.linalg.norm(skeleton.at(Point.LEFT_WRIST)
                                     - skeleton.at(Point.RIGHT_WRIST)))
        return apart * 100 <= HANDS_TOGETHER_CM

    def _keep_hands(self, skeleton: Skeleton) -> None:
        """Keep where each hand is (see ``HANDS_SAME_WAY``)."""
        feet = [p for p in (Point.LEFT_HEEL, Point.RIGHT_HEEL,
                            Point.LEFT_FOOT, Point.RIGHT_FOOT,
                            Point.LEFT_ANKLE, Point.RIGHT_ANKLE)
                if skeleton.seen(p)]
        if not feet:
            return
        floor = max(float(skeleton.at(p)[1]) for p in feet)
        left = skeleton.at(Point.LEFT_WRIST)
        right = skeleton.at(Point.RIGHT_WRIST)
        self._hands.append((skeleton.timestamp, _height(left, floor),
                            _height(right, floor),
                            float(np.linalg.norm(left - right)) * 100))
        while skeleton.timestamp - self._hands[0][0] > HANDS_KEPT:
            self._hands.popleft()

    def _both_hands(self, start: Place, end: Place) -> bool:
        """Whether a movement was made with both hands holding one thing
        (see ``HANDS_SAME_WAY``)."""
        during = [h for h in self._hands
                  if start.second - 0.05 <= h[0] <= end.second + 0.05]
        if len(during) < 2:
            return False
        k = self.scale()
        left = during[-1][1] - during[0][1]
        right = during[-1][2] - during[0][2]
        longer = max(abs(left), abs(right))
        if left * right <= 0 or abs(left - right) > HANDS_SAME_WAY * longer:
            return False
        level = float(np.median([abs(h[1] - h[2]) for h in during])) * k
        apart = [h[3] for h in during]
        steady = float(np.percentile(apart, 90)
                       - np.percentile(apart, 10)) * k
        return level <= HANDS_LEVEL_CM and steady <= HANDS_STEADY_CM

    def _confirm(self, direction: int,
                 nxt: tuple[Place, bool] | None = None) -> list[Lift]:
        """The movement from the last turn to the furthest point ended:
        a turning point more; the lifts of a low point, if it was one.
        ``direction``: of the next movement."""
        assert self._turn is not None and self._extreme is not None
        end = self._extreme
        self._turn = end
        self._direction = direction
        self._extreme = nxt if nxt is not None else end
        turns = self._turns + [end]
        # A pause half way (the movement then goes on the same way) is
        # not a turning point: it would cut one movement in two.
        if len(turns) >= 3 and ((turns[-3][0].v - turns[-2][0].v)
                                * (turns[-2][0].v - turns[-1][0].v) > 0):
            del turns[-2]
        self._turns = turns[-3:]
        if len(self._turns) == 2:
            # The recording started at a low point: its way up.
            low, after = self._turns
            if not low[0].v < after[0].v:
                return []
            visit: tuple[tuple[Place, bool] | None, tuple[Place, bool],
                         tuple[Place, bool] | None] = (None, low, after)
        else:
            before, low, after = self._turns
            if not (low[0].v < before[0].v and low[0].v < after[0].v):
                return []
            visit = (before, low, after)
        if visit[1] is self._scored_low and self._scored:
            # Its way up went on after a pause: the lift is the whole
            # movement, not one more (it was counted twice, 2026-11-18).
            del self.lifts[-self._scored:], self._raw[-self._scored:]
        found = []
        for start, stop in self._handled(*visit):
            lift = self._score(start, stop)
            if lift is not None:
                self.lifts.append(lift)
                self._raw.append((start, stop))
                found.append(lift)
        self._scored_low, self._scored = visit[1], len(found)
        return found

    def _handled(self, before: tuple[Place, bool] | None,
                 low: tuple[Place, bool],
                 after: tuple[Place, bool] | None
                 ) -> list[tuple[Place, Place]]:
        """The movements that carried the load around a low point: the
        way down (``before`` to ``low``) and/or the way up (``low`` to
        ``after``), either missing at the ends of the recording; see the
        module."""
        k = self.scale()
        down = (before[0], low[0]) if before is not None else None
        up = (low[0], after[0]) if after is not None else None
        candidates = [(m, high) for m, high in ((down, before), (up, after))
                      if m is not None and high is not None
                      and abs(m[1].v - m[0].v) * k >= D_MIN
                      and self._both_hands(*m)]
        if not candidates:
            return []
        clear = [m for m, high in candidates if self._clearly(high)]
        if len(clear) == 2:
            return clear
        if clear:
            return clear
        # Neither rises clearly (a box carried at the waist): the higher.
        best = max(candidates, key=lambda c: (c[1][0].v, c[0] is up))
        return [best[0]]

    def _clearly(self, high: tuple[Place, bool]) -> bool:
        """Whether a movement's high end shows the load: well above the
        hanging hands, or the hands together."""
        if high[1]:
            return True
        if not self._body.hip:
            return False
        above = (high[0].v - self._body.knuckles) * self.scale()
        return above >= LOADED_ABOVE_CM

    def _pending(self) -> list[tuple[Place, Place]]:
        """The lifts of the low point the recording ended at or after
        (its way up may still be going on)."""
        if self._turn is None or self._extreme is None:
            return []
        turns = list(self._turns)
        if self._direction != 0 and self._extreme is not turns[-1]:
            turns.append(self._extreme)
        turns = turns[-3:]
        if len(turns) == 3 and (turns[1][0].v < turns[0][0].v
                                and turns[1][0].v < turns[2][0].v):
            # Already scored when the third turn was confirmed.
            if self._extreme is turns[-1] and self._direction != 0:
                return self._handled(*turns)
            return []
        if len(turns) >= 2 and turns[-1][0].v < turns[-2][0].v:
            # Ended at a low point: only its way down.
            return self._handled(turns[-2], turns[-1], None)
        return []

    def rate(self, count: int | None = None) -> float:
        """Lifts a minute: given, or counted over the time seen (at least
        a minute, so a first lift is not taken for a fast pace).

        Args:
            count: Lifts to count (by default, those found so far).
        """
        if self.lifts_per_minute:
            return self.lifts_per_minute
        if self._first is None:
            return F_MIN
        minutes = max((self._last - self._first) / 60, 1.0)
        found = len(self.lifts) if count is None else count
        return max(found / minutes, F_MIN)

    def _score(self, start: Place, end: Place) -> Lift | None:
        """A lift between two stops, if the hands travelled enough."""
        if abs(end.v - start.v) * self.scale() < D_MIN:
            return None
        rate = self.lifts_per_minute
        if not rate:
            # This lift included: the pace so far, over a minute at least.
            minutes = max((end.second - (self._first or end.second)) / 60,
                          1.0)
            rate = max((len(self.lifts) + 1) / minutes, F_MIN)
        return self._lift(start, end, rate)

    def scale(self) -> float:
        """How much to stretch the model's distances to the real height."""
        if not self.height_cm or not self._body.stature:
            return 1.0
        return self.height_cm / self._body.stature

    def _lift(self, start: Place, end: Place, rate: float) -> Lift:
        """The NIOSH score of a lift at a given frequency."""
        k = self.scale()
        start, end = _scaled(start, k), _scaled(end, k)
        d = abs(end.v - start.v)
        scores = [evaluate(p.h, p.v, d, p.a, rate, self.duration, self.grip,
                           self.load_kg) for p in (start, end)]
        worst: Niosh = min(scores, key=lambda s: s.limit)
        notes = self._notes(start, end, k)
        factors = {"HM": worst.hm, "VM": worst.vm, "DM": worst.dm,
                   "AM": worst.am, "FM": worst.fm}
        zero = [ZERO_FACTORS[name] for name, value in factors.items()
                if value <= 0]
        if worst.limit <= 0:
            notes.insert(0, "límite de peso recomendado 0 kg por "
                         + " y ".join(zero or ["la posición"])
                         + ": no es aceptable con ninguna carga")
        return Lift(
            kind="levantamiento" if end.v > start.v else "descenso",
            start=_rounded(start), end=_rounded(end), d=round(d, 1),
            limit=worst.limit, index=worst.index, risk=worst.risk,
            factors={"HM": round(worst.hm, 2), "VM": round(worst.vm, 2),
                     "DM": round(worst.dm, 2), "AM": round(worst.am, 2),
                     "FM": round(worst.fm, 2), "CM": round(worst.cm, 2)},
            notes=notes, beyond_limits=worst.limit <= 0)

    def _notes(self, start: Place, end: Place, k: float) -> list[str]:
        """What the INSST guide warns of in a lift."""
        notes = []
        far = max(start.h, end.h)
        if far > H_RECOMMENDED:
            notes.append(f"carga separada del cuerpo ({far:.0f} cm; la guía "
                         f"recomienda {H_RECOMMENDED:.0f} cm o menos)")
        body = self._body
        for name, spot in (("origen", start), ("destino", end)):
            if body.head and spot.v > body.head * k:
                notes.append(f"manos por encima de la cabeza en el {name}: "
                             "nunca se deben elevar cargas ahí")
            elif body.shoulder and spot.v > body.shoulder * k:
                notes.append(f"manos por encima de los hombros en el {name}")
            elif body.hip and spot.v < body.knuckles * k:
                notes.append(f"manos por debajo de los nudillos en el "
                             f"{name} ({spot.v:.0f} cm del suelo)")
        if self.load_kg and self.load_kg > self.reference_kg:
            notes.append(f"peso superior al máximo de referencia de "
                         f"{self.reference_kg:.0f} kg (guía del INSST)")
        return notes

    def leave_out(self, spans: list[tuple[float, float]],
                  margin: float) -> None:
        """Forget the lifts that start or end within ``margin`` seconds of
        one of ``spans`` (the person's pushes and pulls: the hands going
        onto or off the thing pushed are not a lift)."""
        def near(start: Place, end: Place) -> bool:
            return any(a - margin <= end.second and start.second <= b + margin
                       for a, b in spans)

        kept = [(n, raw) for n, raw in enumerate(self._raw)
                if not near(*raw)]
        self.lifts = [self.lifts[n] for n, _ in kept if n < len(self.lifts)]
        self._raw = [raw for _, raw in kept]
        self._spans = list(spans)
        self._margin = margin

    def report(self) -> LiftingReport:
        """Every lift, scored again with the frequency of the whole
        recording, and the warnings of the task."""
        raw = self._raw + [
            (start, end) for start, end in self._pending()
            if not any(a - self._margin <= end.second
                       and start.second <= b + self._margin
                       for a, b in self._spans)]
        count = len(raw)
        rate = self.rate(count)
        lifts = [self._lift(start, end, rate) for start, end in raw]
        indexes = [lift.index for lift in lifts if lift.index is not None]
        notes = []
        limit = FREQUENCY_LIMIT[self.duration]
        if rate > limit:
            notes.append(f"{rate:.1f} levantamientos por minuto: más de "
                         f"{limit:.0f}, el límite de la guía del INSST para "
                         f"una tarea {self.duration}")
        assumptions = [
            "El modelo de pose no ve la carga: cuenta como manipulación de "
            "la carga indicada cada subida o bajada de 25 cm o más hecha "
            "con las dos manos juntas, a la misma altura y moviéndose a la "
            "vez (la ecuación de NIOSH no se aplica a levantamientos con "
            "una sola mano, NTP 477).",
            f"Duración de la tarea: {self.duration} "
            f"({DURATIONS[self.duration]}).",
            "Las distancias salen del esqueleto en 3D y tienen un error de "
            "varios centímetros.",
        ]
        if not self.lifts_per_minute:
            assumptions.append("Frecuencia contada en el vídeo, que se "
                               "supone representativo de la tarea.")
        if not self.height_cm:
            assumptions.append(
                "Sin la estatura real (--height_cm), las distancias usan la "
                "escala que estima el modelo, que puede errar un 10 % o más.")
        return LiftingReport(
            load_kg=self.load_kg, grip=self.grip, duration=self.duration,
            lifts_per_minute=round(rate, 2), reference_kg=self.reference_kg,
            lifts=lifts, max_index=max(indexes) if indexes else None,
            beyond_limits=sum(lift.beyond_limits for lift in lifts),
            notes=notes, assumptions=assumptions, scale=round(self.scale(), 3),
            stature_cm=round(self._body.stature * self.scale(), 1))


def _scaled(spot: Place, k: float) -> Place:
    """A place with its distances stretched ``k`` times."""
    return Place(second=spot.second, v=spot.v * k, h=spot.h * k, a=spot.a)


def _rounded(spot: Place) -> Place:
    """A place with its measures rounded to the unit."""
    return Place(second=round(spot.second, 2), v=round(spot.v, 1),
                 h=round(spot.h, 1), a=round(spot.a, 1))


def reference_kg(sex: Sex, age: int | None) -> float:
    """The INSST guide's heaviest load in ideal conditions (table 2)."""
    if not sex or age is None:
        return REFERENCE_UNKNOWN
    return REFERENCE_KG[(sex, 20 <= age <= 45)]
