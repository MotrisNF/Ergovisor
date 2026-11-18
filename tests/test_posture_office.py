"""Tests for the work at a screen: the view, the face and the warnings."""
from dataclasses import replace
import math

import numpy as np
import pytest

from src.posture.face import Face, eye_distance_cm, head_angles
from src.posture.office import (BREAK_EVERY, REST_EVERY, Office,
                                OfficeEvent, view_of)
from src.posture.regulation import (OFFICE_TOPICS, TOPICS, LiveRegulation,
                                    office_warnings)
from src.posture.skeleton import Point, Skeleton


def skeleton(hips: bool = False, shoulder_width: float = 0.4,
             ears_above: float = 0.12) -> Skeleton:
    """A person facing the camera; ``hips`` says whether the hips are in
    the picture. Pixels: shoulders at y 0.7, ears ``ears_above`` higher."""
    visibility = np.ones(33)
    if not hips:
        visibility[[Point.LEFT_HIP, Point.RIGHT_HIP, Point.LEFT_KNEE,
                    Point.RIGHT_KNEE, Point.LEFT_ANKLE,
                    Point.RIGHT_ANKLE]] = 0.0
    pixels = np.full((33, 2), 0.5)
    half = shoulder_width / 2
    pixels[Point.LEFT_SHOULDER] = (0.5 + half, 0.7)
    pixels[Point.RIGHT_SHOULDER] = (0.5 - half, 0.7)
    pixels[Point.LEFT_EAR] = (0.55, 0.7 - ears_above)
    pixels[Point.RIGHT_EAR] = (0.45, 0.7 - ears_above)
    return Skeleton(0.0, np.zeros((33, 3)), visibility, pixels)


def face(second: float, pitch: float = 5.0, yaw: float = 0.0,
         distance: float | None = 55.0, blink: float = 0.0) -> Face:
    return Face(second, pitch, yaw, 0.0, distance, blink)


def run(office: Office, seconds: float, step: float = 1.0,
        start: float = 0.0, body: Skeleton | None = None,
        present: bool = True, **kwargs: float) -> list[OfficeEvent]:
    """Feed frames at the screen from ``start`` for ``seconds``."""
    found = []
    second = start
    body = body or skeleton()
    while second < start + seconds:
        seen = face(second, **kwargs) if present else None
        found += office.add(second, body if present else None, seen)
        second += step
    return found


class TestView:
    """Which view the camera has, from the skeleton."""

    def test_hips_out_is_desk(self) -> None:
        assert view_of(skeleton(hips=False)) == "desk"

    def test_whole_body(self) -> None:
        assert view_of(skeleton(hips=True, shoulder_width=0.1)) == "body"

    def test_close_shoulders_is_desk(self) -> None:
        assert view_of(skeleton(hips=True, shoulder_width=0.4)) == "desk"

    def test_unknown(self) -> None:
        assert view_of(None) is None

    def test_majority_of_last_seconds(self) -> None:
        """One odd frame does not switch the view."""
        office = Office()
        for i in range(20):
            office.see(i * 0.1, skeleton(hips=False))
        assert office.see(2.0, skeleton(hips=True, shoulder_width=0.1)) \
            == "desk"


class TestWarnings:
    """Each warning needs its condition to last."""

    def test_too_close(self) -> None:
        office = Office()
        found = run(office, 15, distance=25.0)
        found += run(office, 2, start=15)
        assert [e.kind for e in found] == ["too_close"]
        assert "25 cm" in found[0].detail

    def test_glance_too_close_is_no_warning(self) -> None:
        office = Office()
        found = run(office, 5, distance=25.0) + run(office, 2, start=5)
        assert found == []

    def test_neck_flexed_held(self) -> None:
        office = Office()
        found = run(office, 70, pitch=30.0) + run(office, 2, start=70)
        assert [e.kind for e in found] == ["neck_flexed"]
        assert "30°" in found[0].detail

    def test_looking_down_is_not_a_rest(self) -> None:
        """A paper on the desk (turned 28°, down 25°) keeps the eyes near:
        no visual rest; turning to the window (34°, level) is one."""
        office = Office()
        run(office, 10)
        run(office, 30, start=10, yaw=28.0, pitch=25.0)
        report = office.report()
        assert report is not None and report.visual_rests == 0
        office = Office()
        run(office, 10)
        run(office, 30, start=10, yaw=34.0, pitch=3.0)
        report = office.report()
        assert report is not None and report.visual_rests == 1

    def test_neck_flexed_short(self) -> None:
        office = Office()
        assert run(office, 40, pitch=30.0) + run(office, 2, start=40) == []

    def test_finish_ends_warnings(self) -> None:
        office = Office()
        run(office, 70, pitch=30.0)
        assert [e.kind for e in office.finish()] == ["neck_flexed"]

    def test_no_visual_rest(self) -> None:
        office = Office()
        found = run(office, REST_EVERY + 5, step=5.0)
        assert "no_visual_rest" in [e.kind for e in found]

    def test_visual_rest_resets(self) -> None:
        """Looking away 20 s at minute 15 restarts the count."""
        office = Office()
        found = run(office, 900, step=5.0)
        found += run(office, 25, step=5.0, start=900, yaw=60.0)
        found += run(office, 600, step=5.0, start=925)
        assert "no_visual_rest" not in [e.kind for e in found]
        assert office.report() is not None
        assert office.report().visual_rests == 1  # type: ignore[union-attr]

    def test_no_break(self) -> None:
        office = Office()
        found = run(office, BREAK_EVERY + 5, step=10.0)
        assert "no_break" in [e.kind for e in found]

    def test_break_resets(self) -> None:
        office = Office()
        found = run(office, 1500, step=10.0)
        found += run(office, 60, step=10.0, start=1500, present=False)
        found += run(office, 1000, step=10.0, start=1560)
        assert "no_break" not in [e.kind for e in found]
        report = office.report()
        assert report is not None and report.breaks == 1

    def test_shoulders_not_warned(self) -> None:
        """Raised shoulders are not measured (unreliable from the front
        camera): ears close to the shoulders give no warning."""
        office = Office()
        run(office, 120, step=1.0)
        found = run(office, 70, step=1.0, start=120,
                    body=skeleton(ears_above=0.05))
        found += run(office, 2, start=190)
        assert found == []


class TestReport:
    """What the report keeps."""

    def test_blinks(self) -> None:
        office = Office()
        second = 0.0
        for _ in range(30):
            for blink in (0.0, 0.0, 0.9, 0.1):
                office.add(second, skeleton(), face(second, blink=blink))
                second += 0.5
        report = office.report()
        assert report is not None
        assert report.blinks == 30
        assert report.blinks_per_minute == pytest.approx(30, rel=0.05)

    def test_distance(self) -> None:
        office = Office()
        run(office, 60, distance=50.0)
        report = office.report()
        assert report is not None and report.distance_cm == 50.0

    def test_never_at_screen(self) -> None:
        assert Office().report() is None


class TestFace:
    """The measures taken from the face model."""

    def test_distance_from_iris(self) -> None:
        """An iris 11.7 mm wide seen at 50 cm with a 65° camera."""
        width = 1280
        focal = (width / 2) / math.tan(math.radians(65) / 2)
        iris = focal * 11.7 / 500 / width
        points = np.full((478, 3), 0.5)
        points[469] = (0.5, 0.5, 0)
        points[471] = (0.5 + iris, 0.5, 0)
        assert eye_distance_cm(points, width) == pytest.approx(50, rel=0.01)

    def test_no_iris(self) -> None:
        assert eye_distance_cm(np.zeros((468, 3)), 640) is None

    def test_straight_head(self) -> None:
        assert head_angles(np.eye(4)) == pytest.approx((0, 0, 0), abs=1e-6)

    @pytest.mark.parametrize("degrees", [10.0, 25.0])
    def test_turn_alone(self, degrees: float) -> None:
        """A head turned around the vertical axis: only the yaw."""
        a = math.radians(degrees)
        matrix = np.eye(4)
        matrix[:3, :3] = np.array([[math.cos(a), 0, math.sin(a)],
                                   [0, 1, 0],
                                   [-math.sin(a), 0, math.cos(a)]])
        pitch, yaw, roll = head_angles(matrix)
        assert abs(yaw) == pytest.approx(degrees)
        assert pitch == pytest.approx(0, abs=1e-6)
        assert roll == pytest.approx(0, abs=1e-6)

    @pytest.mark.parametrize("degrees", [10.0, 25.0])
    def test_nod_alone(self, degrees: float) -> None:
        a = math.radians(degrees)
        matrix = np.eye(4)
        matrix[:3, :3] = np.array([[1, 0, 0],
                                   [0, math.cos(a), -math.sin(a)],
                                   [0, math.sin(a), math.cos(a)]])
        pitch, yaw, roll = head_angles(matrix)
        assert abs(pitch) == pytest.approx(degrees)
        assert yaw == pytest.approx(0, abs=1e-6)


class TestRegulation:
    """Screen warnings find the screen regulation."""

    def test_every_kind_has_a_topic(self) -> None:
        for kind, topic in OFFICE_TOPICS.items():
            assert topic in TOPICS, kind

    def test_screen_topics_avoid_load_documents(self) -> None:
        assert "rd_487" in TOPICS["screen_distance"].avoid
        assert "rd_488" in TOPICS["trunk_flexion"].avoid

    def test_live(self) -> None:
        event = OfficeEvent(kind="too_close", title="Pantalla demasiado cerca",
                            start=1, end=12, detail="a 25 cm")
        assert office_warnings(event)[0][0] == "screen_distance"

        class Empty:
            def search(self, query: str, k: int) -> list[object]:
                return []

        live = LiveRegulation(Empty())  # type: ignore[arg-type]
        found = [f.topic for f in live.office(event)]
        assert found == ["screen_distance", "screen_employer"]
        assert live.office(event) == []


class TestStandingUp:
    """Seated at the screen, then up and seen whole by the camera."""

    def test_standing_up_is_a_break(self) -> None:
        office = Office()
        found = run(office, 1500, step=10.0)
        # Up for a minute: the analysis passes no skeleton or face.
        found += run(office, 60, step=10.0, start=1500, present=False)
        found += run(office, 1000, step=10.0, start=1560)
        assert "no_break" not in [e.kind for e in found]
        report = office.report()
        assert report is not None
        assert report.breaks == 1
        assert report.seconds_at_screen == pytest.approx(2490, abs=10)

    def test_warning_ends_when_standing_up(self) -> None:
        """A bent neck before standing up ends there: the time up is not
        added to it."""
        office = Office()
        found = run(office, 70, pitch=30.0)
        found += run(office, 120, step=10.0, start=70, present=False)
        assert [e.kind for e in found] == ["neck_flexed"]
        assert found[0].end == pytest.approx(70, abs=1)

    def test_started(self) -> None:
        office = Office()
        assert not office.started
        run(office, 1)
        assert office.started


class TestAnalysisSwitchesView:
    """The whole analysis, seated, then up in front of the camera, then
    seated again."""

    def test_seated_up_seated(self) -> None:
        from src.posture.analysis import analyse
        from src.posture.sources import Frame

        def view(second: float) -> str:
            return "body" if 100 <= second < 160 else "desk"

        class Poses:
            def find_all(self, frame: Frame) -> list[Skeleton]:
                seen = (skeleton(hips=True, shoulder_width=0.1)
                        if view(frame.timestamp) == "body"
                        else skeleton(hips=False))
                return [replace(seen, timestamp=frame.timestamp)]

        class Faces:
            def find_all(self, frame: Frame) -> list[Face]:
                return [face(frame.timestamp)]

        frames = [Frame(float(s), np.zeros((4, 4, 3), dtype=np.uint8))
                  for s in range(200)]
        result = analyse(frames, Poses(), "test",  # type: ignore[arg-type]
                         office=Office(),
                         faces=Faces())  # type: ignore[arg-type]
        [person] = result.people
        assert person.office is not None
        assert person.office.breaks == 1
        # Seated 100 + 40 s (the view needs a moment to switch back).
        assert person.office.seconds_at_screen == pytest.approx(140, abs=4)
        assert person.summary.frames_with_person > 0
