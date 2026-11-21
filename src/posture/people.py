"""Several people in the picture: who is who from one frame to the next.

The pose model finds up to ``MAX_PEOPLE`` skeletons in each frame, in no
particular order. Each one is given the number of the person it was in
the previous frames by where it is in the image: the nearest one. Nobody
is recognised (no face, no features are kept): a person is a number,
"Persona 1", "Persona 2"...

So as not to lose anybody on the way:
- a person missing for a few frames (the model failing for an instant,
  someone passing in front) keeps their number and their analysis goes
  on as if they had not left (``BRIDGE``);
- where each person is going is followed, so two people crossing are
  told apart by where each one should be now, and by their size;
- someone appearing where nobody was takes the number of a person not
  seen now, the one last seen nearest: a person coming back keeps their
  number (and the screen work counts their absence as a break). A
  number seen less than ``MIN_SECONDS`` in all (most likely something
  taken for a person: the chair being pulled in C3) is only given back
  when there is no other. A new number is only given when every known
  person is in the picture. And someone barely seen (less than
  ``MIN_SECONDS``) takes back the number of a person seen longer that has
  been out of the picture more than ``BRIDGE``: standing up from a
  laptop, the chair taken for a person for an instant kept the person's
  number and the person went on with another, so the report had the
  same person twice (E2, 2026-11-18). So
  with one person in the picture there is always "Persona 1", and there
  are never more numbers than people seen at once; the cost is that
  someone else coming in where a person left takes that person's number.
- two skeletons of the same person (the model sometimes finds one twice,
  the whole body and only its upper half inside it) count once: one box
  inside the other with the head and shoulders in the same place. A
  person behind another is also inside their box, but with the head
  elsewhere: they are two (E2: someone walking behind a person at a
  laptop took their place, 2026-11-18);
- a "person" seen less than ``MIN_SECONDS`` (a chair or a bag taken for
  one for an instant) is left out of the report (see ``analysis``).

The confidence of the pose model is left at its default: lower (0.3) it
finds the person in a few more frames, but also "people" in a bag on the
floor or a chair (calibrated on 2026-11-18 with B5 and C2); at the
default, the frames it misses with three people that it found with one
were a single frame (bridged) or a person too close to the camera, with
no head, that nothing could be measured of.

``MAX_PEOPLE`` is the only limit: raise it for a warehouse with more
people in one camera (each one costs some time of the pose model).
"""
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from .skeleton import MIN_VISIBILITY, Point, Skeleton

# Ours: people analysed at once in one camera. The pose model takes
# 18 ms a frame with one and 28 ms with three ("full", 2026-11-18).
MAX_PEOPLE = 3
# Ours: a person missing this long (s) is still in the picture: the
# frames without them are skipped, not counted as their absence (someone
# passing in front of them). As long as the longest step between two
# frames that the timeline still counts as time seen.
BRIDGE = 1.0
# Ours: in the next frame a person is at most this unlike where they
# were (how far, a share of the image's width or height, plus how
# different in size, see ``SIZE_WEIGHT``) to keep the number in the first
# place; more unlike, they get it only if nobody more alike does.
GATE = 0.25
# Ours: two skeletons are the same person found twice when this much of
# the smaller box is inside the other (C2, entering: the whole body and
# its upper half, 2026-11-18) and their head and shoulders are at most
# this far apart (a share of the image, the mean of the points seen in
# both).
DUPLICATE = 0.7
SAME_PLACE = 0.06
# Ours: a person seen less than this (s) is left out of the report: the
# model took something for a person for an instant (one frame in B7).
MIN_SECONDS = 2.0
# Ours: how much a different size (the log of the ratio of the widths of
# their shoulders, see ``locate``) counts against being the same person,
# in shares of the image: half as wide counts as 0.14 of the image away.
SIZE_WEIGHT = 0.2
# Ours: how much each new step counts in the speed followed (the rest is
# the previous speed), to smooth the jitter of the points.
SPEED_SMOOTHING = 0.5


Box = tuple[float, float, float, float]
# A place in the image (x, y, shares of it).
ImagePoint = NDArray[np.float64]


def box(skeleton: Skeleton) -> Box | None:
    """The box around the points seen in the image (0-1), or ``None``."""
    if skeleton.pixels is None:
        return None
    seen = skeleton.pixels[skeleton.visibility >= MIN_VISIBILITY]
    if not len(seen):
        return None
    x0, y0 = (float(v) for v in seen.min(axis=0))
    x1, y1 = (float(v) for v in seen.max(axis=0))
    return x0, y0, x1, y1


def _overlap(a: Box, b: Box) -> float:
    """How much of the smaller box is inside the other (0 to 1)."""
    width = min(a[2], b[2]) - max(a[0], b[0])
    height = min(a[3], b[3]) - max(a[1], b[1])
    if width <= 0 or height <= 0:
        return 0.0
    smaller = min((a[2] - a[0]) * (a[3] - a[1]),
                  (b[2] - b[0]) * (b[3] - b[1]))
    return width * height / smaller if smaller > 0 else 0.0


def _centre(place: Box) -> ImagePoint:
    return np.array([(place[0] + place[2]) / 2, (place[1] + place[3]) / 2])


HEAD = (Point.NOSE, Point.LEFT_SHOULDER, Point.RIGHT_SHOULDER)
# Ours, from body proportions: the nose is about two thirds of the width
# of the shoulders above their middle, so this many times that height is
# about their width (it gives the size seen side-on, when the shoulders
# are one behind the other).
NECK_TO_SHOULDERS = 1.5


def locate(skeleton: Skeleton) -> tuple[ImagePoint, float] | None:
    """Where a person is in the image and how big: the middle of their
    head and shoulders, and the width of their shoulders (0: unknown);
    ``None`` with nothing seen.

    Not their box: at a laptop the model makes up the legs out of the
    picture, some frames calling them seen and some not, and the box of
    the same person jumped from 0.3 to 1.1 of the image's height from one
    frame to the next, so they were taken for someone else 17 times in 35
    minutes and the screen work was cut in three. The head and shoulders
    are seen in every view (D1, 2026-11-18: none; the calibration
    videos gave the same as with the box).
    """
    head = [p for p in HEAD if skeleton.seen(p)]
    if skeleton.pixels is None or not head:
        place = box(skeleton)
        return (_centre(place), 0.0) if place is not None else None
    centre = np.asarray(skeleton.pixels[head].mean(axis=0), dtype=float)
    if not skeleton.seen(Point.LEFT_SHOULDER, Point.RIGHT_SHOULDER):
        return centre, 0.0
    left = skeleton.pixels[Point.LEFT_SHOULDER]
    right = skeleton.pixels[Point.RIGHT_SHOULDER]
    size = float(np.linalg.norm(left - right))
    if skeleton.seen(Point.NOSE):
        neck = float(np.linalg.norm(skeleton.pixels[Point.NOSE]
                                    - (left + right) / 2))
        size = max(size, neck * NECK_TO_SHOULDERS)
    return centre, size


def _same_head(a: Skeleton, b: Skeleton) -> bool:
    """Whether two skeletons have the head and shoulders in the same place
    in the image (unknown: assumed so)."""
    if a.pixels is None or b.pixels is None:
        return True
    both = [p for p in HEAD if a.seen(p) and b.seen(p)]
    if not both:
        return True
    apart = np.linalg.norm(a.pixels[both] - b.pixels[both], axis=1)
    return float(np.mean(apart)) <= SAME_PLACE


def distinct(skeletons: list[Skeleton]) -> list[Skeleton]:
    """The skeletons without the second finding of the same person (the
    one the model is least sure of is dropped)."""
    ranked = sorted(skeletons, key=lambda s: -float(np.mean(s.visibility)))
    kept: list[tuple[Skeleton, Box | None]] = []
    for skeleton in ranked:
        place = box(skeleton)
        if place is not None and any(
                other is not None and _overlap(place, other) > DUPLICATE
                and _same_head(skeleton, found)
                for found, other in kept):
            continue
        kept.append((skeleton, place))
    return [skeleton for skeleton, _ in kept]


@dataclass
class _Known:
    """A person seen: where, how big, how fast and when, the last time
    (where in shares of the image a second)."""

    number: int
    centre: ImagePoint
    size: float
    second: float
    speed: ImagePoint = field(default_factory=lambda: np.zeros(2))
    seen: float = 0.0

    def expected(self, second: float) -> ImagePoint:
        """Where they should be now, at most ``BRIDGE`` seconds on (a
        person stopping behind someone is not followed further)."""
        ahead = min(max(0.0, second - self.second), BRIDGE)
        return np.asarray(self.centre + self.speed * ahead)

    def distance(self, second: float, centre: ImagePoint) -> float:
        """How far a skeleton is from where they should be."""
        return float(np.linalg.norm(centre - self.expected(second)))

    def cost(self, second: float, centre: ImagePoint,
             size: float) -> float:
        """How unlike them a skeleton is: how far from where they should
        be, and how different in size."""
        unlike = (abs(float(np.log(size / self.size)))
                  if size > 0 and self.size > 0 else 0.0)
        return self.distance(second, centre) + SIZE_WEIGHT * unlike

    def move(self, second: float, centre: ImagePoint,
             size: float) -> None:
        """Seen again here."""
        step = second - self.second
        self.seen += min(max(0.0, step), BRIDGE)
        if 0 < step <= BRIDGE:
            now = (centre - self.centre) / step
            self.speed = (SPEED_SMOOTHING * now
                          + (1 - SPEED_SMOOTHING) * self.speed)
        elif step > BRIDGE:
            self.speed = np.zeros(2)
        if size > 0:
            self.size = size
        self.centre, self.second = centre, second


class Tracker:
    """Gives each skeleton the number of its person (see the module).

    Attributes:
        max_people: The most people followed at once.
    """

    def __init__(self, max_people: int = MAX_PEOPLE) -> None:
        """Nobody seen yet."""
        self.max_people = max_people
        self._known: dict[int, _Known] = {}

    def assign(self, second: float, skeletons: list[Skeleton]
               ) -> dict[int, Skeleton]:
        """The skeletons of one frame, by person number.

        Args:
            second: When the frame was taken.
            skeletons: What the pose model found in it.

        Returns:
            ``{number: skeleton}`` for each person in the frame.
        """
        found = distinct(skeletons)[:self.max_people]
        places: list[tuple[ImagePoint, float] | None] = []
        for skeleton in found:
            places.append(locate(skeleton))
        # (cost, skeleton, number, cost after the others): someone
        # barely seen goes after everybody else (the 1 is more than any
        # distance in the image).
        pairs = sorted(
            (cost, n, known.number,
             cost + (0.0 if known.seen >= MIN_SECONDS else 1.0))
            for n, where in enumerate(places) if where is not None
            for known in self._known.values()
            for cost in (known.cost(second, *where),))
        given: dict[int, int] = {}
        # Near where they were, among those still in the picture, first;
        # then whoever is not in the picture.
        for near_only in (True, False):
            ordered = (pairs if near_only
                       else sorted(pairs, key=lambda pair: pair[3]))
            for cost, n, number, _ in ordered:
                if n in given or number in given.values():
                    continue
                # Near counts the size too: someone far behind a person
                # who sits down is near where the person was, but much
                # smaller (E2, 2026-11-18).
                if near_only and (cost > GATE or second
                                  - self._known[number].second > BRIDGE):
                    continue
                given[n] = number
        for n in range(len(found)):
            if n not in given:
                free = next(number for number in range(1, len(found) + 1
                                                       + len(self._known))
                            if number not in given.values()
                            and number not in self._known)
                given[n] = free
        self._take_back(second, given, places)
        for n, number in given.items():
            where = places[n]
            known = self._known.get(number)
            if where is None:
                if known is not None:
                    known.second = second
                    continue
                where = (np.array([.5, .5]), 0.0)
            if known is None:
                self._known[number] = _Known(number, where[0], where[1],
                                             second)
            else:
                known.move(second, *where)
        return {number: found[n] for n, number in sorted(
            given.items(), key=lambda item: item[1])}

    def _take_back(self, second: float, given: dict[int, int],
                   places: list[tuple[ImagePoint, float] | None]) -> None:
        """Give the number of someone seen long and gone (more than
        ``BRIDGE``) to whoever was barely seen, the nearest first."""
        gone = [known for known in self._known.values()
                if known.seen >= MIN_SECONDS
                and known.number not in given.values()
                and second - known.second > BRIDGE]
        for n, number in sorted(given.items()):
            young = self._known.get(number)
            if not gone or (young is not None
                            and young.seen >= MIN_SECONDS):
                continue
            where = places[n]
            old = min(gone, key=lambda known: (
                known.distance(second, where[0]) if where is not None
                else 0.0))
            gone.remove(old)
            given[n] = old.number

    def present(self, number: int, second: float) -> bool:
        """Whether a person still counts as in the picture: seen in the
        last ``BRIDGE`` seconds."""
        known = self._known.get(number)
        return known is not None and second - known.second <= BRIDGE
