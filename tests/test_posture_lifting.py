"""Tests for ``src.posture.niosh`` and ``src.posture.lifting``.

NIOSH values are checked against NTP 477; the lifts are built with the
hand-made skeletons of ``test_posture_reba``.
"""
from collections.abc import Sequence

import numpy as np
import pytest

from src.posture.lifting import (Lift, Lifting, place, reference_kg,
                                 REFERENCE_UNKNOWN)
from src.posture.niosh import (LC, asymmetry, coupling, displacement,
                               evaluate, frequency, horizontal, risk_of,
                               vertical)
from src.posture.skeleton import Point
from test_posture_reba import Body, FORWARD, UP

FPS = 10


class TestFactors:
    """Each NIOSH factor, as NTP 477 defines it."""

    @pytest.mark.parametrize("h,value", [(20, 1.0), (25, 1.0), (50, 0.5),
                                         (63, 25 / 63), (64, 0.0)])
    def test_horizontal(self, h: float, value: float) -> None:
        """25/H, 1 close to the body, 0 beyond 63 cm."""
        assert horizontal(h) == pytest.approx(value)

    @pytest.mark.parametrize("v,value", [(75, 1.0), (0, 0.775),
                                         (175, 0.7), (176, 0.0)])
    def test_vertical(self, v: float, value: float) -> None:
        """1 - 0.003 |V - 75|, 0 above 175 cm."""
        assert vertical(v) == pytest.approx(value)

    @pytest.mark.parametrize("d,value", [(10, 1.0), (25, 1.0), (50, 0.91),
                                         (175, 0.82 + 4.5 / 175),
                                         (176, 0.0)])
    def test_displacement(self, d: float, value: float) -> None:
        """0.82 + 4.5/D, 1 below 25 cm, 0 beyond 175."""
        assert displacement(d) == pytest.approx(value)

    @pytest.mark.parametrize("a,value", [(0, 1.0), (90, 0.712),
                                         (135, 0.568), (136, 0.0)])
    def test_asymmetry(self, a: float, value: float) -> None:
        """1 - 0.0032 A, 0 beyond 135 degrees."""
        assert asymmetry(a) == pytest.approx(value)

    @pytest.mark.parametrize("rate,duration,v,value", [
        (0.1, "corta", 50, 1.00), (1, "larga", 50, 0.75),
        (4, "moderada", 80, 0.72), (9, "larga", 50, 0.00),
        (9, "larga", 80, 0.15), (13, "corta", 80, 0.34),
        (1.5, "corta", 50, 0.91), (16, "corta", 80, 0.0)])
    def test_frequency(self, rate: float, duration: str, v: float,
                       value: float) -> None:
        """Table of NTP 477; between rows, the next one up (safer)."""
        assert frequency(rate, duration, v) == value  # type: ignore[arg-type]

    @pytest.mark.parametrize("grip,v,value", [
        ("bueno", 50, 1.0), ("regular", 50, 0.95), ("regular", 80, 1.0),
        ("malo", 80, 0.9)])
    def test_coupling(self, grip: str, v: float, value: float) -> None:
        """Table of NTP 477."""
        assert coupling(grip, v) == value  # type: ignore[arg-type]

    @pytest.mark.parametrize("index,risk", [
        (0.8, "Riesgo limitado"), (2.0, "Incremento moderado del riesgo"),
        (3.5, "Incremento acusado del riesgo")])
    def test_risk(self, index: float, risk: str) -> None:
        """The three zones of the lifting index."""
        assert risk_of(index) == risk


class TestEvaluate:
    """The whole equation."""

    def test_ideal_lift_is_the_constant(self) -> None:
        """Close, at 75 cm, short, straight, rare, good grip: 23 kg."""
        result = evaluate(25, 75, 20, 0, 0.1, "corta", "bueno", 23)
        assert result.limit == LC and result.index == 1.0

    def test_far_low_lift(self) -> None:
        """From the floor and far: a much lower limit."""
        result = evaluate(50, 10, 70, 0, 1, "larga", "bueno", 10)
        expected = (23 * 0.5 * (1 - 0.003 * 65) * (0.82 + 4.5 / 70)
                    * 0.75)
        assert result.limit == pytest.approx(expected, abs=0.01)
        assert result.index is not None and result.index > 1

    def test_without_load(self) -> None:
        """The limit, but no index."""
        result = evaluate(25, 75, 30, 0, 1, "corta", "bueno", None)
        assert result.index is None and result.limit > 0


def reaching(depth: float, height_drop: float) -> Body:
    """Standing with the hands held ``depth`` metres ahead of the ankles
    and ``height_drop`` metres below the hips."""
    body = Body()
    hands = np.array([0.0, height_drop, 0.0]) + FORWARD * depth
    for side in ("LEFT", "RIGHT"):
        offset = 0.05 if side == "LEFT" else -0.05
        body.points[Point[f"{side}_WRIST"]] = hands + [offset, 0, 0]
        body.points[Point[f"{side}_INDEX"]] = hands + [offset, 0.05, 0]
    for side, x in (("LEFT", 0.10), ("RIGHT", -0.10)):
        body.points[Point[f"{side}_HEEL"]] = np.array([x, 0.95, 0.05])
    return body


class TestPlace:
    """Where the hands are, measured on the skeleton."""

    def test_height_and_distance(self) -> None:
        """Hands 30 cm ahead, 20 cm below the hips."""
        found = place(reaching(0.30, 0.20).skeleton())
        assert found is not None
        # The floor is the lowest point of the feet (y = 0.95 m); the hands
        # are the middle of wrists (0.20) and index fingers (0.25).
        assert found.v == pytest.approx(95 - 22.5, abs=1)
        assert found.h == pytest.approx(30, abs=1)
        assert found.a == pytest.approx(0, abs=2)

    def test_twist(self) -> None:
        """The hands as far to one side as ahead: 45 degrees."""
        body = reaching(0.30, 0.20)
        for p in (Point.LEFT_WRIST, Point.RIGHT_WRIST, Point.LEFT_INDEX,
                  Point.RIGHT_INDEX):
            body.points[p] = body.points[p] + np.array([0.30, 0, 0])
        found = place(body.skeleton())
        assert found is not None and found.a == pytest.approx(45, abs=3)

    def test_feet_not_seen(self) -> None:
        """Without the feet there is no floor to measure from."""
        body = reaching(0.30, 0.20)
        body.hidden = {Point.LEFT_ANKLE, Point.RIGHT_ANKLE,
                       Point.LEFT_HEEL, Point.RIGHT_HEEL, Point.LEFT_FOOT,
                       Point.RIGHT_FOOT}
        assert place(body.skeleton()) is None


def play(lifting: Lifting,
         plan: Sequence[tuple[float, float, float]]) -> list[Lift]:
    """Feed hand heights: ``(seconds, from_drop, to_drop)`` segments, the
    hands moving linearly 30 cm ahead; the lifts found."""
    found: list[Lift] = []
    second = 0.0
    for seconds, start, end in plan:
        steps = int(seconds * FPS)
        for n in range(steps):
            drop = start + (end - start) * n / max(steps - 1, 1)
            lift = lifting.add(reaching(0.30, drop).skeleton(second))
            if lift is not None:
                found.append(lift)
            second += 1 / FPS
    return found


class TestLifting:
    """Lifts found over time."""

    def test_one_lift_from_low_to_high(self) -> None:
        """Still low, up 60 cm, still: one lift with D of 60 cm."""
        lifts = play(Lifting(load_kg=10),
                     [(2, 0.70, 0.70), (2, 0.70, 0.10), (2, 0.10, 0.10)])
        [lift] = lifts
        assert lift.kind == "levantamiento"
        assert lift.d == pytest.approx(60, abs=3)
        assert lift.index is not None and lift.limit < 23

    def test_lowering(self) -> None:
        """Down counts too, as a lowering."""
        lifts = play(Lifting(load_kg=10),
                     [(2, 0.10, 0.10), (2, 0.10, 0.70), (2, 0.70, 0.70)])
        assert [lift.kind for lift in lifts] == ["descenso"]

    def test_small_movement_is_not_a_lift(self) -> None:
        """Less than 25 cm: not a lift."""
        assert play(Lifting(), [(2, 0.3, 0.3), (2, 0.3, 0.15),
                                (2, 0.15, 0.15)]) == []

    def test_slow_lift(self) -> None:
        """A lift done slowly (10 s for 60 cm) is still one lift."""
        lifts = play(Lifting(), [(2, 0.7, 0.7), (10, 0.7, 0.1),
                                 (2, 0.1, 0.1)])
        assert len(lifts) == 1

    def test_report_and_notes(self) -> None:
        """Frequency, the INSST's reference weight and its warnings."""
        lifting = Lifting(load_kg=22, duration="corta",
                          reference_kg=reference_kg("mujer", 30))
        play(lifting, [(2, 0.70, 0.70), (2, 0.70, 0.10), (2, 0.10, 0.10)])
        report = lifting.report()
        [lift] = report.lifts
        assert report.reference_kg == 20
        assert any("separada del cuerpo" in n for n in lift.notes)
        assert any("máximo de referencia de 20 kg" in n for n in lift.notes)
        assert report.lifts_per_minute == pytest.approx(1.0)
        assert report.max_index == lift.index

    def test_scaled_to_the_real_height(self) -> None:
        """With the real height, every distance is stretched to it."""
        plan = [(2, 0.70, 0.70), (2, 0.70, 0.10), (2, 0.10, 0.10)]
        plain, tall = Lifting(load_kg=10), Lifting(load_kg=10,
                                                   height_cm=200)
        play(plain, plan)
        play(tall, plan)
        k = tall.report().scale
        assert k > 1
        assert tall.report().lifts[0].d == pytest.approx(
            plain.report().lifts[0].d * k, abs=0.2)

    def test_frequency_limit_of_the_guide(self) -> None:
        """More lifts a minute than the guide allows: warned."""
        report = Lifting(lifts_per_minute=11, duration="larga").report()
        assert any("más de 10" in note for note in report.notes)


@pytest.mark.parametrize("sex,age,kg", [
    ("hombre", 30, 25), ("hombre", 50, 20), ("mujer", 30, 20),
    ("mujer", 19, 15), ("", None, REFERENCE_UNKNOWN)])
def test_reference_kg(sex: str, age: int | None, kg: float) -> None:
    """Table 2 of the INSST guide (2024)."""
    assert reference_kg(sex, age) == kg  # type: ignore[arg-type]


def test_up_is_up() -> None:
    """The test skeletons and the code agree on which way is up."""
    assert UP[1] == -1
