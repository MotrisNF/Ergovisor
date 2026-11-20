"""Tests for ``src.posture.niosh`` and ``src.posture.lifting``.

NIOSH values are checked against NTP 477; the lifts are built with the
hand-made skeletons of ``test_posture_reba``.
"""
from collections.abc import Sequence

import numpy as np
import pytest

from src.posture.lifting import (Lift, Lifting, Place, place,
                                 reference_kg,
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

    def test_beyond_the_equation(self) -> None:
        """H over 63 cm: the limit is 0 kg (NTP 477) and no index. With a
        load, the risk is high whatever it weighs; without one, no risk is
        named (NTP 477 reads it from the index, and moving the empty hands
        is no load: raising the arms in B3 read "acusado")."""
        loaded = evaluate(70, 75, 50, 0, 1, "larga", "bueno", 10)
        assert loaded.limit == 0 and loaded.index is None
        assert loaded.risk == "Incremento acusado del riesgo"
        empty = evaluate(70, 75, 50, 0, 1, "larga", "bueno", None)
        assert empty.limit == 0 and empty.index is None
        assert empty.risk == ""

    def test_lift_beyond_the_equation_says_why(self) -> None:
        """A lift far from the body: marked, with the reason first."""
        lifting = Lifting(load_kg=5)
        lift = lifting._lift(Place(second=0, v=10, h=70, a=0),
                             Place(second=2, v=80, h=70, a=0), 1.0)
        assert lift.beyond_limits and lift.index is None
        assert lift.notes[0].startswith("límite de peso recomendado 0 kg "
                                        "por la carga a más de 63 cm")

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


def reaching(depth: float, height_drop: float,
             apart: float = 0.10) -> Body:
    """Standing with the hands held ``depth`` metres ahead of the ankles,
    ``height_drop`` metres below the hips and ``apart`` metres from each
    other (0.10: holding something; about 0.40: hanging at the sides)."""
    body = Body()
    hands = np.array([0.0, height_drop, 0.0]) + FORWARD * depth
    for side in ("LEFT", "RIGHT"):
        offset = apart / 2 if side == "LEFT" else -apart / 2
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
        """The shoulders turned 45 degrees against the hips: A is 45."""
        body = reaching(0.30, 0.20)
        centre = (body.points[Point.LEFT_SHOULDER]
                  + body.points[Point.RIGHT_SHOULDER]) / 2
        body.turn([Point.LEFT_SHOULDER, Point.RIGHT_SHOULDER], centre,
                  FORWARD, np.array([1.0, 0.0, 0.0]), 45)
        found = place(body.skeleton())
        assert found is not None and found.a == pytest.approx(45, abs=3)

    def test_hands_to_the_side_are_no_twist(self) -> None:
        """Only the trunk counts as twist: hands off to one side with the
        trunk straight read no twist (the depth of a side view is too
        uncertain to tell them from hands in front)."""
        body = reaching(0.30, 0.20)
        for p in (Point.LEFT_WRIST, Point.RIGHT_WRIST, Point.LEFT_INDEX,
                  Point.RIGHT_INDEX):
            body.points[p] = body.points[p] + np.array([0.30, 0, 0])
        found = place(body.skeleton())
        assert found is not None and found.a == pytest.approx(0, abs=3)
        assert found.h == pytest.approx(30, abs=1)

    def test_feet_not_seen(self) -> None:
        """Without the feet there is no floor to measure from."""
        body = reaching(0.30, 0.20)
        body.hidden = {Point.LEFT_ANKLE, Point.RIGHT_ANKLE,
                       Point.LEFT_HEEL, Point.RIGHT_HEEL, Point.LEFT_FOOT,
                       Point.RIGHT_FOOT}
        assert place(body.skeleton()) is None


def play(lifting: Lifting,
         plan: Sequence[tuple[float, float, float]],
         apart: float = 0.10) -> list[Lift]:
    """Feed hand heights: ``(seconds, from_drop, to_drop)`` segments, the
    hands moving linearly 30 cm ahead, ``apart`` metres from each other;
    the lifts found."""
    return play_from(lifting, 0.0, plan, apart)


def play_from(lifting: Lifting, second: float,
              plan: Sequence[tuple[float, float, float]],
              apart: float = 0.10) -> list[Lift]:
    """Like ``play``, from ``second`` on."""
    found: list[Lift] = []
    for seconds, start, end in plan:
        steps = int(seconds * FPS)
        for n in range(steps):
            drop = start + (end - start) * n / max(steps - 1, 1)
            found += lifting.add(reaching(0.30, drop, apart)
                                 .skeleton(second))
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
        """Down counts too, as a lowering. Ending low, it is in the
        report (live it is known when the hands rise again)."""
        lifting = Lifting(load_kg=10)
        play(lifting, [(2, 0.10, 0.10), (2, 0.10, 0.70), (2, 0.70, 0.70)])
        assert [lift.kind for lift in lifting.report().lifts] == [
            "descenso"]

    def test_pick_up_counts_once(self) -> None:
        """Reaching down empty, lifting to the chest, lowering it to the
        floor and standing up empty: two handlings, not four."""
        lifting = Lifting(load_kg=5)
        play(lifting, [(2, 0.0, 0.0), (2, 0.0, 0.75), (1, 0.75, 0.75),
                       (2, 0.75, -0.35), (2, -0.35, -0.35),
                       (2, -0.35, 0.75), (1, 0.75, 0.75), (2, 0.75, 0.0),
                       (2, 0.0, 0.0)], apart=0.40)
        kinds = [lift.kind for lift in lifting.report().lifts]
        assert kinds == ["levantamiento", "descenso"]

    def test_pause_half_way_down(self) -> None:
        """Pausing half way down and going on: one lowering, not two
        short movements."""
        lifting = Lifting(load_kg=5)
        play(lifting, [(2, -0.35, -0.35), (1.5, -0.35, 0.10),
                       (1.5, 0.10, 0.10), (1.5, 0.10, 0.75),
                       (1, 0.75, 0.75), (2, 0.75, 0.0), (2, 0.0, 0.0)],
             apart=0.40)
        [lowering] = [lift for lift in lifting.report().lifts
                      if lift.kind == "descenso"]
        assert lowering.d == pytest.approx(110, abs=8)

    def test_pause_on_the_way_up_counts_once(self) -> None:
        """Up from the floor, a pause half way after the low point was
        scored, and on up: one lift, the whole way."""
        lifting = Lifting(load_kg=5)
        play(lifting, [(2, 0.10, 0.10), (2, 0.10, 0.70), (2, 0.70, 0.70),
                       (1, 0.70, 0.45), (2, 0.45, 0.45), (1, 0.45, 0.10),
                       (2, 0.10, 0.10)])
        starts = [lift.start.second for lift in lifting.lifts]
        assert len(starts) == len(set(starts))
        assert lifting.report().lifts[-1].d == pytest.approx(60, abs=6)

    def test_one_hand_is_no_lift(self) -> None:
        """One hand goes down to the floor and up, the other stays: not a
        lift (NIOSH is for both hands, NTP 477)."""
        lifting = Lifting(load_kg=5)
        second = 0.0
        for seconds, start, end in [(2, -0.30, -0.30), (2, -0.30, 0.70),
                                    (2, 0.70, -0.30), (2, -0.30, -0.30)]:
            steps = int(seconds * FPS)
            for n in range(steps):
                drop = start + (end - start) * n / max(steps - 1, 1)
                body = reaching(0.30, -0.30)
                lowered = reaching(0.30, drop)
                for p in (Point.LEFT_WRIST, Point.LEFT_INDEX):
                    body.points[p] = lowered.points[p]
                lifting.add(body.skeleton(second))
                second += 1 / FPS
        assert lifting.report().lifts == []

    def test_hands_coming_together_is_no_lift(self) -> None:
        """Both hands rise from hanging apart to together at the waist
        (crossing the arms, talking): not holding one thing."""
        lifting = Lifting(load_kg=5)
        second = 0.0
        for seconds, start, end in [(2, 0.0, 0.0), (2, 0.0, 1.0),
                                    (2, 1.0, 1.0), (2, 1.0, 0.0),
                                    (2, 0.0, 0.0)]:
            steps = int(seconds * FPS)
            for n in range(steps):
                share = start + (end - start) * n / max(steps - 1, 1)
                body = reaching(0.30, 0.45 - 0.35 * share,
                                apart=0.60 - 0.50 * share)
                lifting.add(body.skeleton(second))
                second += 1 / FPS
        assert lifting.report().lifts == []

    def test_next_to_a_push_is_no_lift(self) -> None:
        """A lift right after a push (the hands coming off the cart) is
        left out; one well apart from it is kept."""
        lifting = Lifting(load_kg=5)
        play(lifting, [(2, 0.70, 0.70), (2, 0.70, 0.10), (2, 0.10, 0.10),
                       (2, 0.10, 0.70), (2, 0.70, 0.70)])
        assert len(lifting.report().lifts) == 2
        lifting.leave_out([(0.0, 1.5)], margin=1.5)
        [kept] = lifting.report().lifts
        assert kept.kind == "descenso"

    def test_box_at_the_waist(self) -> None:
        """A box lifted only to the waist (not clearly above the hanging
        hands) still counts once per low point."""
        lifting = Lifting(load_kg=5)
        play(lifting, [(2, 0.05, 0.05), (2, 0.05, 0.75), (1, 0.75, 0.75),
                       (2, 0.75, 0.05), (2, 0.05, 0.05)], apart=0.40)
        assert len(lifting.report().lifts) == 1

    def test_small_movement_is_not_a_lift(self) -> None:
        """Less than 25 cm: not a lift."""
        assert play(Lifting(), [(2, 0.3, 0.3), (2, 0.3, 0.15),
                                (2, 0.15, 0.15)]) == []

    def test_slow_lift(self) -> None:
        """A lift done slowly (10 s for 60 cm) is still one lift."""
        lifts = play(Lifting(), [(2, 0.7, 0.7), (10, 0.7, 0.1),
                                 (2, 0.1, 0.1)])
        assert len(lifts) == 1

    def test_gone_between_is_no_lift(self) -> None:
        """Hands low, the person out of the picture for 5 s, back with the
        hands high: no lift (it was not seen; an office session gave
        "lifts" of minutes that way). Leaving a load low and taking it
        again after it still count."""
        lifting = Lifting(load_kg=5)
        low, high = reaching(0.30, 0.70), reaching(0.30, 0.10)
        for n in range(2 * FPS):
            lifting.add(low.skeleton(n / FPS))
        for n in range(5 * FPS):
            lifting.add(None)
        for n in range(2 * FPS):
            lifting.add(high.skeleton(7 + n / FPS))
        assert lifting.report().lifts == []
        found = play_from(lifting, 9.0, [(2, 0.10, 0.70), (2, 0.70, 0.70),
                                         (2, 0.70, 0.10), (2, 0.10, 0.10)])
        assert [lift.kind for lift in found] == ["descenso", "levantamiento"]
        assert (found[0].start.second or 0) >= 7

    def test_impossible_measures_not_scored(self) -> None:
        """Hands beyond what a person reaches, or a stature no person has:
        the model's error, not a lift (a "limit of 0 kg" for hands above
        175 cm in a person of 164, 2026-11-20)."""
        from src.posture.lifting import NOSE_OF_STATURE

        lifting = Lifting()
        lifting._body.head = 170 * NOSE_OF_STATURE

        def spot(v: float, h: float = 30.0) -> Place:
            return Place(second=0.0, v=v, h=h, a=0.0)

        assert lifting._possible(spot(20), spot(200))
        assert not lifting._possible(spot(20), spot(220))
        assert not lifting._possible(spot(-20), spot(100))
        assert not lifting._possible(spot(20, h=110), spot(100))
        lifting._body.head = 95 * NOSE_OF_STATURE
        assert not lifting._possible(spot(20), spot(80))

    def test_duration_assumed_is_said(self) -> None:
        """Without the task's duration, a short one is assumed and the
        report says so; given, it is not."""
        assumed = " ".join(Lifting().report().assumptions)
        assert "corta" in assumed and "supuesta" in assumed
        given = " ".join(Lifting(duration="larga", duration_assumed=False)
                         .report().assumptions)
        assert "larga" in given and "supuesta" not in given

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
