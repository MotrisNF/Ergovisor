"""Tests for pushing and pulling: a synthetic person crossing the image."""
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from src.posture.pushing import (FORCE_LIMITS, FORCE_UNKNOWN, Push,
                                 Pushing)
from src.posture.regulation import TOPICS, push_warnings
from src.posture.skeleton import Point, Skeleton
from src.posture.writer import describe_analysis

ASPECT = 16 / 9


def person(second: float, x: float, hands: str = "front",
           hands_y: float = -0.2, swing: float = 0.0) -> Skeleton:
    """A person facing the right of the image (+x), whose hips are at
    ``x`` in the image (0 to 1). World points in metres, y down, centred
    on the hips; ``hands``: in ``front`` of the shoulders or ``back``
    behind the hips, at height ``hands_y``; ``swing``: each hand that
    much (m) above and below it, in turns, and further apart."""
    points = np.zeros((33, 3))
    points[Point.LEFT_HIP] = (0, 0, -0.1)
    points[Point.RIGHT_HIP] = (0, 0, 0.1)
    points[Point.LEFT_SHOULDER] = (0, -0.5, -0.18)
    points[Point.RIGHT_SHOULDER] = (0, -0.5, 0.18)
    points[Point.NOSE] = (0.1, -0.65, 0)
    reach = 0.45 if hands == "front" else -0.35
    turn = swing if int(second * 2) % 2 else -swing
    points[Point.LEFT_WRIST] = (reach, hands_y + turn, -0.15 - swing)
    points[Point.RIGHT_WRIST] = (reach, hands_y - turn, 0.15 + swing)
    points[Point.LEFT_INDEX] = (reach + 0.08, hands_y, -0.15)
    points[Point.RIGHT_INDEX] = (reach + 0.08, hands_y, 0.15)
    for ankle, heel, toe, z in ((Point.LEFT_ANKLE, Point.LEFT_HEEL,
                                 Point.LEFT_FOOT, -0.1),
                                (Point.RIGHT_ANKLE, Point.RIGHT_HEEL,
                                 Point.RIGHT_FOOT, 0.1)):
        points[ankle] = (0, 0.85, z)
        points[heel] = (-0.05, 0.9, z)
        points[toe] = (0.12, 0.9, z)
    pixels = np.full((33, 2), 0.5)
    pixels[:, 0] = x
    pixels[Point.NOSE] = (x, 0.2)
    for p in (Point.LEFT_HEEL, Point.RIGHT_HEEL, Point.LEFT_ANKLE,
              Point.RIGHT_ANKLE):
        pixels[p] = (x, 0.9)
    pixels[Point.LEFT_HIP] = pixels[Point.RIGHT_HIP] = (x, 0.55)
    visibility = np.ones(33)
    return Skeleton(second, points, visibility, pixels)


def angles(elbow: float = 20.0, trunk: float = 10.0,
           twist: float = 0.0) -> Any:
    arm = SimpleNamespace(elbow=elbow)
    return SimpleNamespace(left=arm, right=arm, trunk=trunk,
                           trunk_twist=twist)


def walk(pushing: Pushing, seconds: float, speed: float, start: float = 0.0,
         x0: float = 0.2, step: float = 0.1, elbow: float = 20.0,
         trunk: float = 10.0, **kwargs: Any) -> list[Push]:
    """Move the person ``speed`` image widths a second; then stand still
    long enough for the episode to end."""
    found = []
    second, x = start, x0
    while second < start + seconds:
        ended = pushing.add(person(second, x, **kwargs),
                            angles(elbow, trunk), ASPECT)
        found += [ended] if ended else []
        second += step
        x += speed * step
    # Standing still long enough for the episode to end.
    for _ in range(15):
        ended = pushing.add(person(second, x, **kwargs), angles(elbow, trunk),
                            ASPECT)
        found += [ended] if ended else []
        second += step
    return found


class TestDetection:
    """Push, pull, or neither."""

    def test_push(self) -> None:
        found = walk(Pushing(height_cm=170), 4.0, 0.13)
        assert [p.kind for p in found] == ["empuje"]
        assert found[0].metres == pytest.approx(2.0, rel=0.15)

    def test_arms_swinging_is_no_push(self) -> None:
        """Walking past with the arms swinging (one hand high, the other
        low, far apart), not both hands on something: no push (the
        guide's methods are for both hands, ISO 11228-2; B6)."""
        assert walk(Pushing(height_cm=170), 4.0, 0.13, swing=0.15) == []

    def test_pull_backwards(self) -> None:
        """Walking backwards with the hands in front."""
        found = walk(Pushing(), 4.0, -0.13, x0=0.8)
        assert [p.kind for p in found] == ["arrastre"]

    def test_tow(self) -> None:
        """Walking forwards with the hands behind."""
        found = walk(Pushing(), 4.0, 0.13, hands="back")
        assert [p.kind for p in found] == ["arrastre"]

    def test_standing_still(self) -> None:
        assert walk(Pushing(), 4.0, 0.0) == []

    def test_carrying_with_bent_elbows(self) -> None:
        """Arms bent (a box held close) is not a push."""
        assert walk(Pushing(), 4.0, 0.13, elbow=95.0) == []

    def test_too_short(self) -> None:
        assert walk(Pushing(), 1.0, 0.13) == []

    def test_finish_closes(self) -> None:
        pushing = Pushing()
        second, x = 0.0, 0.2
        while second < 4:
            pushing.add(person(second, x), angles(), ASPECT)
            second, x = second + 0.1, x + 0.013
        assert pushing.finish() is not None


class TestNotes:
    """What the guide warns of."""

    def test_hands_low(self) -> None:
        found = walk(Pushing(), 4.0, 0.13, hands_y=0.3)
        assert any("por debajo de los nudillos" in n
                   for n in found[0].notes)

    def test_hands_between_knuckles_and_shoulders(self) -> None:
        found = walk(Pushing(), 4.0, 0.13)
        assert not any(n.startswith("manos") for n in found[0].notes)

    def test_trunk_bent(self) -> None:
        found = walk(Pushing(), 4.0, 0.13, trunk=35.0)
        assert any("tronco inclinado 35°" in n for n in found[0].notes)


class TestReport:
    """Forces and the report."""

    def test_force_over_limit(self) -> None:
        pushing = Pushing(sex="hombre", force_initial_n=400,
                          force_sustained_n=200)
        walk(pushing, 4.0, 0.13)
        report = pushing.report()
        assert report is not None
        assert report.force_limits_n == FORCE_LIMITS["hombre"]
        assert [n.split(" de ")[0] for n in report.notes] == [
            "fuerza inicial"]

    def test_without_force(self) -> None:
        pushing = Pushing()
        walk(pushing, 4.0, 0.13)
        report = pushing.report()
        assert report is not None
        assert report.force_limits_n == FORCE_UNKNOWN
        assert any("La fuerza no se indicó" in a
                   for a in report.assumptions)

    def test_none_without_pushes(self) -> None:
        assert Pushing().report() is None


class TestRegulationAndWriting:
    """Pushes find their topics and reach the model's input."""

    def test_topics(self) -> None:
        push = Push(kind="empuje", start=1, end=4, metres=2.0,
                    hands_low_cm=60, hands_high_cm=70, trunk=30, twist=20,
                    notes=["manos por debajo de los nudillos al empujar",
                           "tronco inclinado 30° durante el esfuerzo",
                           "tronco girado 20° durante el esfuerzo"])
        topics = [t for t, _ in push_warnings(push)]
        assert topics == ["push_pull", "push_height", "trunk_flexion",
                          "trunk_twist"]
        assert all(t in TOPICS for t in topics)
        assert "rd_488" in TOPICS["push_pull"].avoid

    def test_described_for_the_model(self) -> None:
        pushing = Pushing()
        walk(pushing, 4.0, 0.13)
        report = pushing.report()
        assert report is not None
        text = describe_analysis({
            "source": "x", "created": "", "people": [{
                "person": 1, "summary": {}, "moments": 0,
                "pushing": report.model_dump()}]})
        assert "Empuje y tracción de cargas" in text
        assert "La fuerza no se midió" in text


def test_stature_from_the_trunk() -> None:
    """With the head out of the picture, the trunk gives the stature."""
    from src.posture.pushing import TRUNK_OF_STATURE, _stature_px

    skeleton = person(0.0, 0.5)
    skeleton.visibility[Point.NOSE] = 0.0
    assert skeleton.pixels is not None
    skeleton.pixels[Point.LEFT_SHOULDER] = (0.5, 0.40)
    skeleton.pixels[Point.RIGHT_SHOULDER] = (0.5, 0.40)
    trunk = 0.55 - 0.40
    assert _stature_px(skeleton) == pytest.approx(trunk / TRUNK_OF_STATURE)
