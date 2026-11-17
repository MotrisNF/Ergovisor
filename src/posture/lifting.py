"""Lifts and lowerings seen in the skeleton, scored with the NIOSH
equation and checked against the INSST's guide on manual handling of
loads (2024 edition).

What the equation needs is measured on the skeleton:

- V: the height of the hands above the floor (the lowest point of the
  feet).
- H: how far the hands are, horizontally, from the middle of the ankles.
- A: the twist between the direction of the hands and the front of the
  feet.
- D: how much the hands go up or down in a lift.
- F: how many lifts a minute.

A lift (or a lowering) is the hands stopping, going up (or down) at least
25 cm and stopping again. A pose model does not see the load, so every
such movement counts as handling the load given by the person; this is
said in the report.

Like the rest, it only looks back: each lift is given when it ends.
"""
from collections import deque
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from pydantic import BaseModel

from .angles import UP, Vector, body_axes
from .niosh import (D_MIN, DURATIONS, F_MIN, Duration, Grip, Niosh,
                    evaluate)
from .skeleton import Point, Skeleton

Sex = Literal["hombre", "mujer", ""]

# Ours: the hands are still when they moved less than this (cm) in the
# last ``STILL_SECONDS``; measuring travel, not speed, also finds the slow
# lifts of a demonstration.
STILL_CM = 3.0
STILL_SECONDS = 1.0
# Ours: with the hands this close to the ankles (cm) the direction of the
# twist is noise, and it counts as no twist.
TWIST_MIN_H = 10.0
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
        index: Lifting index (IL), or ``None`` without a load.
        risk: What NTP 477 says of that index.
        factors: HM, VM, DM, AM, FM and CM where the limit is lowest.
        notes: What the INSST guide warns of in this lift.
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
        max_index: The highest lifting index.
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
    flat = reach - float(reach @ UP) * UP
    h = float(np.linalg.norm(flat) * 100)
    front = _feet_front(skeleton)
    twist = 0.0
    if h >= TWIST_MIN_H:
        cosine = float(flat @ front) / (float(np.linalg.norm(flat))
                                        * float(np.linalg.norm(front)))
        twist = float(np.degrees(np.arccos(np.clip(cosine, -1, 1))))
    return Place(second=skeleton.timestamp, v=_height(hands, floor), h=h,
                 a=twist)


def _feet_front(skeleton: Skeleton) -> Vector:
    """Where the feet point (from heels to toes), or the body's front."""
    toes = _seen_mean(skeleton, [Point.LEFT_FOOT, Point.RIGHT_FOOT])
    heels = _seen_mean(skeleton, [Point.LEFT_HEEL, Point.RIGHT_HEEL])
    if toes is not None and heels is not None:
        front = toes - heels
        front = front - float(front @ UP) * UP
        if float(np.linalg.norm(front)) > 0.03:
            return front
    return body_axes(skeleton)[2]


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
    _rest: Place | None = None
    _moving: bool = False
    _first: float | None = None
    _last: float = 0.0
    _body: _Body = field(default_factory=_Body)

    def add(self, skeleton: Skeleton | None) -> Lift | None:
        """One frame; the lift that just ended, if any."""
        if skeleton is None:
            return None
        self._body.add(skeleton)
        now = place(skeleton)
        if now is None:
            return None
        if self._first is None:
            self._first = now.second
        self._last = now.second
        self._recent.append(now)
        while now.second - self._recent[0].second > STILL_SECONDS:
            self._recent.popleft()
        if now.second - self._recent[0].second < STILL_SECONDS * 0.8:
            return None
        travel = max(p.v for p in self._recent) - min(p.v
                                                      for p in self._recent)
        still = travel < STILL_CM
        if not still:
            self._moving = True
            return None
        ended = None
        rest = self._rest
        if self._moving and rest is not None:
            ended = self._score(rest, now)
            if ended is not None:
                self.lifts.append(ended)
                self._raw.append((rest, now))
        self._moving = False
        self._rest = now
        return ended

    def rate(self) -> float:
        """Lifts a minute: given, or counted over the time seen (at least
        a minute, so a first lift is not taken for a fast pace)."""
        if self.lifts_per_minute:
            return self.lifts_per_minute
        if self._first is None:
            return F_MIN
        minutes = max((self._last - self._first) / 60, 1.0)
        return max(len(self.lifts) / minutes, F_MIN)

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
        return Lift(
            kind="levantamiento" if end.v > start.v else "descenso",
            start=_rounded(start), end=_rounded(end), d=round(d, 1),
            limit=worst.limit, index=worst.index, risk=worst.risk,
            factors={"HM": round(worst.hm, 2), "VM": round(worst.vm, 2),
                     "DM": round(worst.dm, 2), "AM": round(worst.am, 2),
                     "FM": round(worst.fm, 2), "CM": round(worst.cm, 2)},
            notes=self._notes(start, end, k))

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

    def report(self) -> LiftingReport:
        """Every lift, scored again with the frequency of the whole
        recording, and the warnings of the task."""
        rate = self.rate()
        lifts = [self._lift(start, end, rate) for start, end in self._raw]
        indexes = [lift.index for lift in lifts if lift.index is not None]
        notes = []
        limit = FREQUENCY_LIMIT[self.duration]
        if rate > limit:
            notes.append(f"{rate:.1f} levantamientos por minuto: más de "
                         f"{limit:.0f}, el límite de la guía del INSST para "
                         f"una tarea {self.duration}")
        assumptions = [
            "El modelo de pose no ve la carga: cada subida o bajada de las "
            "manos de 25 cm o más cuenta como manipulación de la carga "
            "indicada.",
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
