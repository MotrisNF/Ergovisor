"""Tests for ``src.posture.angles`` and ``src.posture.reba``.

The skeletons are built by hand: a person standing upright, then bent by
a known number of degrees. Coordinates follow MediaPipe's world
landmarks: metres, centred on the hips, y down, the person facing the
camera (forward is -z), their left on the +x side.
"""
import numpy as np
import pytest

from src.posture.angles import measure
from src.posture.reba import (LEVELS, TABLE_A, TABLE_B, TABLE_C, Task,
                              level_of, score)
from src.posture.skeleton import Point, Skeleton

UP = np.array([0.0, -1.0, 0.0])
DOWN = -UP
FORWARD = np.array([0.0, 0.0, -1.0])
LEFT = np.array([1.0, 0.0, 0.0])

STANDING = {
    Point.NOSE: (0.0, -0.65, -0.14),
    # The ears 15 degrees ahead of the shoulders, as a straight head is.
    Point.LEFT_EAR: (0.07, -0.65, -0.0402),
    Point.RIGHT_EAR: (-0.07, -0.65, -0.0402),
    Point.LEFT_SHOULDER: (0.18, -0.50, 0.0),
    Point.RIGHT_SHOULDER: (-0.18, -0.50, 0.0),
    Point.LEFT_ELBOW: (0.18, -0.20, 0.0),
    Point.RIGHT_ELBOW: (-0.18, -0.20, 0.0),
    Point.LEFT_WRIST: (0.18, 0.05, 0.0), Point.RIGHT_WRIST: (-0.18, 0.05, 0.0),
    Point.LEFT_INDEX: (0.18, 0.13, 0.0), Point.RIGHT_INDEX: (-0.18, 0.13, 0.0),
    Point.LEFT_HIP: (0.10, 0.0, 0.0), Point.RIGHT_HIP: (-0.10, 0.0, 0.0),
    Point.LEFT_KNEE: (0.10, 0.45, 0.0), Point.RIGHT_KNEE: (-0.10, 0.45, 0.0),
    Point.LEFT_ANKLE: (0.10, 0.90, 0.0), Point.RIGHT_ANKLE: (-0.10, 0.90, 0.0),
    Point.LEFT_FOOT: (0.10, 0.95, -0.12),
    Point.RIGHT_FOOT: (-0.10, 0.95, -0.12),
}
UPPER_BODY = [Point.NOSE, Point.LEFT_EAR, Point.RIGHT_EAR,
              Point.LEFT_SHOULDER, Point.RIGHT_SHOULDER, Point.LEFT_ELBOW,
              Point.RIGHT_ELBOW, Point.LEFT_WRIST, Point.RIGHT_WRIST,
              Point.LEFT_INDEX, Point.RIGHT_INDEX]
HEAD = [Point.NOSE, Point.LEFT_EAR, Point.RIGHT_EAR]


def rotation(axis: np.ndarray, degrees: float) -> np.ndarray:
    """Rodrigues: turns vectors around ``axis`` by ``degrees``."""
    k = axis / np.linalg.norm(axis)
    angle = np.radians(degrees)
    cross = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.asarray(np.eye(3) + np.sin(angle) * cross
                      + (1 - np.cos(angle)) * cross @ cross, dtype=float)


class Body:
    """A skeleton that can be bent joint by joint."""

    def __init__(self) -> None:
        """A person standing upright, every point seen."""
        self.points = {p: np.array(xyz, dtype=float)
                       for p, xyz in STANDING.items()}
        self.hidden: set[Point] = set()

    def turn(self, points: list[Point], centre: Point | np.ndarray,
             towards: np.ndarray, start: np.ndarray,
             degrees: float) -> "Body":
        """Turn some points around ``centre``, from ``start`` towards
        ``towards``."""
        middle = (self.points[centre] if isinstance(centre, Point)
                  else centre)
        turn = rotation(np.cross(start, towards), degrees)
        for p in points:
            self.points[p] = middle + turn @ (self.points[p] - middle)
        return self

    def bend_trunk(self, degrees: float,
                   towards: np.ndarray = FORWARD) -> "Body":
        """Bend everything above the hips."""
        hips = (self.points[Point.LEFT_HIP] + self.points[Point.RIGHT_HIP]) / 2
        return self.turn(UPPER_BODY, hips, towards, UP, degrees)

    def raise_arm(self, degrees: float, towards: np.ndarray = FORWARD,
                  side: str = "LEFT") -> "Body":
        """Raise one arm from hanging down."""
        names = ["ELBOW", "WRIST", "INDEX"]
        return self.turn([Point[f"{side}_{n}"] for n in names],
                         Point[f"{side}_SHOULDER"], towards, DOWN, degrees)

    def bend_elbow(self, degrees: float, side: str = "LEFT") -> "Body":
        """Bend one elbow forwards."""
        return self.turn([Point[f"{side}_WRIST"], Point[f"{side}_INDEX"]],
                         Point[f"{side}_ELBOW"], FORWARD, DOWN, degrees)

    def bend_knees(self, degrees: float) -> "Body":
        """Bend both knees backwards."""
        for side in ("LEFT", "RIGHT"):
            self.turn([Point[f"{side}_ANKLE"], Point[f"{side}_FOOT"]],
                      Point[f"{side}_KNEE"], -FORWARD, DOWN, degrees)
        return self

    def yaw(self, degrees: float) -> "Body":
        """Turn the whole person around the vertical (another camera)."""
        turn = rotation(UP, degrees)
        self.points = {p: turn @ xyz for p, xyz in self.points.items()}
        return self

    def skeleton(self, timestamp: float = 0.0) -> Skeleton:
        """The skeleton, with the hidden points not seen."""
        points = np.zeros((33, 3))
        visibility = np.zeros(33)
        for p, xyz in self.points.items():
            points[p] = xyz
            visibility[p] = 0.0 if p in self.hidden else 1.0
        return Skeleton(timestamp, points, visibility)


def angles_of(body: Body):  # type: ignore[no-untyped-def]
    """The angles of a body; it must be measurable."""
    found = measure(body.skeleton())
    assert found is not None
    return found


class TestAngles:
    """``measure``: the angles of known postures."""

    def test_standing(self) -> None:
        """Upright: nothing bent."""
        a = angles_of(Body())
        assert abs(a.trunk) < 1 and a.trunk_side < 1 and a.trunk_twist < 1
        assert abs(a.neck) < 1 and a.knee < 1 and not a.one_leg
        assert a.left is not None and abs(a.left.upper) < 1
        assert a.left.elbow < 1 and not a.left.abducted

    @pytest.mark.parametrize("degrees", [15, 30, 70])
    def test_trunk_flexion(self, degrees: float) -> None:
        """The trunk bent forwards, measured as flexion."""
        assert angles_of(Body().bend_trunk(degrees)).trunk == pytest.approx(
            degrees, abs=0.5)

    def test_trunk_extension(self) -> None:
        """Bent backwards: negative."""
        assert angles_of(Body().bend_trunk(25, -FORWARD)).trunk == \
            pytest.approx(-25, abs=0.5)

    def test_trunk_sideways(self) -> None:
        """Bent to one side: not flexion, a sideways bend."""
        a = angles_of(Body().bend_trunk(30, LEFT))
        assert abs(a.trunk) < 1 and a.trunk_side == pytest.approx(30, abs=1)

    @pytest.mark.parametrize("camera", [90, 180, 37])
    def test_any_camera(self, camera: float) -> None:
        """The same posture seen from the side, the back or any angle."""
        body = Body().bend_trunk(40).raise_arm(100).yaw(camera)
        if camera == 180:
            body.hidden = {Point.NOSE}
        a = angles_of(body)
        assert a.trunk == pytest.approx(40, abs=1)
        assert a.left is not None and a.left.upper == pytest.approx(100,
                                                                    abs=1)

    def test_head_bent_alone(self) -> None:
        """The neck is measured relative to the trunk."""
        body = Body().bend_trunk(30)
        shoulders = (body.points[Point.LEFT_SHOULDER]
                     + body.points[Point.RIGHT_SHOULDER]) / 2
        trunk_up = rotation(np.cross(UP, FORWARD), 30) @ UP
        body.turn(HEAD, shoulders, rotation(np.cross(UP, FORWARD), 30)
                  @ FORWARD, trunk_up, 25)
        a = angles_of(body)
        assert a.trunk == pytest.approx(30, abs=1)
        assert a.neck == pytest.approx(25, abs=1.5)

    def test_arm_forward_and_sideways(self) -> None:
        """Forwards is flexion; sideways is abduction."""
        a = angles_of(Body().raise_arm(60).raise_arm(90, LEFT, "RIGHT"))
        assert a.left is not None and a.right is not None
        assert a.left.upper == pytest.approx(60, abs=1)
        assert not a.left.abducted
        assert a.right.upper == pytest.approx(90, abs=1)

    def test_arm_backwards(self) -> None:
        """Behind the trunk: extension, negative."""
        a = angles_of(Body().raise_arm(30, -FORWARD))
        assert a.left is not None
        assert a.left.upper == pytest.approx(-30, abs=1)

    def test_elbow_and_knees(self) -> None:
        """Joint flexions."""
        a = angles_of(Body().bend_elbow(80).bend_knees(45))
        assert a.left is not None
        assert a.left.elbow == pytest.approx(80, abs=1)
        assert a.knee == pytest.approx(45, abs=1)

    def test_one_foot_lifted(self) -> None:
        """Standing on one leg."""
        body = Body()
        for p in (Point.LEFT_ANKLE, Point.LEFT_FOOT):
            body.points[p] = body.points[p] + UP * 0.2
        assert angles_of(body).one_leg

    def test_hips_not_seen(self) -> None:
        """Nothing can be measured without the hips."""
        body = Body()
        body.hidden = {Point.LEFT_HIP}
        assert measure(body.skeleton()) is None

    def test_arm_not_seen(self) -> None:
        """An arm out of sight is not guessed."""
        body = Body()
        body.hidden = {Point.RIGHT_WRIST}
        assert angles_of(body).right is None


class TestTables:
    """The tables, as NTP 601 prints them."""

    def test_shapes(self) -> None:
        """5 x 3 x 4, 6 x 2 x 3 and 12 x 12."""
        assert [len(TABLE_A), len(TABLE_A[0]), len(TABLE_A[0][0])] == \
            [5, 3, 4]
        assert [len(TABLE_B), len(TABLE_B[0]), len(TABLE_B[0][0])] == \
            [6, 2, 3]
        assert len(TABLE_C) == 12 and all(len(r) == 12 for r in TABLE_C)

    @pytest.mark.parametrize("trunk,neck,legs,value", [
        (1, 1, 1, 1), (3, 2, 2, 5), (5, 3, 4, 9), (1, 3, 1, 3)])
    def test_table_a(self, trunk: int, neck: int, legs: int,
                     value: int) -> None:
        """Some cells of Tabla A."""
        assert TABLE_A[trunk - 1][neck - 1][legs - 1] == value

    def test_table_c_never_decreases(self) -> None:
        """More strain never gives a lower score."""
        for row in TABLE_C:
            assert row == sorted(row)
        for column in zip(*TABLE_C):
            assert list(column) == sorted(column)

    @pytest.mark.parametrize("final,level,risk", [
        (1, 0, "Inapreciable"), (3, 1, "Bajo"), (4, 2, "Medio"),
        (10, 3, "Alto"), (11, 4, "Muy alto"), (15, 4, "Muy alto")])
    def test_levels(self, final: int, level: int, risk: str) -> None:
        """The action levels of NTP 601."""
        assert level_of(final)[:2] == (level, risk)
        assert LEVELS[-1][0] == 15


class TestScore:
    """``score``: whole postures."""

    def test_standing_is_negligible(self) -> None:
        """Upright, arms down: score 1 but the elbows straight (2)."""
        result = score(angles_of(Body()))
        assert result is not None
        assert (result.trunk, result.neck, result.legs) == (1, 1, 1)
        assert result.lower_arm == 2 and result.level <= 1

    def test_bent_lifting_heavy(self) -> None:
        """Trunk at 70 degrees, knees at 45, arm at 100, 12 kg: high."""
        body = Body().bend_trunk(70).bend_knees(45).raise_arm(100)
        result = score(angles_of(body), Task(load_kg=12))
        assert result is not None
        assert result.trunk == 4 and result.legs == 2
        assert result.upper_arm == 4
        assert result.a == TABLE_A[3][0][1] + 2
        assert result.level >= 3

    def test_worse_arm_is_kept(self) -> None:
        """Group B is the arm under more strain."""
        result = score(angles_of(Body().raise_arm(120, side="RIGHT")))
        assert result is not None
        assert result.side == "derecho" and result.upper_arm == 4

    def test_activity_adds(self) -> None:
        """Held or repeated postures add to the final score."""
        angles = angles_of(Body().bend_trunk(30))
        plain, held = score(angles), score(angles, activity=2)
        assert plain is not None and held is not None
        assert held.score == plain.score + 2

    @pytest.mark.parametrize("kg,sudden,points", [
        (3, False, 0), (5, False, 1), (10, False, 1), (10.5, False, 2),
        (3, True, 1)])
    def test_load(self, kg: float, sudden: bool, points: int) -> None:
        """Less than 5 kg, 5-10 kg, more than 10 kg, +1 if sudden."""
        angles = angles_of(Body())
        base = score(angles)
        loaded = score(angles, Task(load_kg=kg, sudden=sudden))
        assert base is not None and loaded is not None
        assert loaded.a - base.a == points

    def test_no_arm_seen(self) -> None:
        """Without any arm there is no group B."""
        body = Body()
        body.hidden = {Point.LEFT_WRIST, Point.RIGHT_WRIST}
        assert score(angles_of(body)) is None
