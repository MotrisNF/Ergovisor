"""Tests for several people: ``src.posture.people`` (who is who), the
analysis of each one (``src.posture.analysis``) and the joint report
(regulation, writer, overlay)."""
from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from src.posture.analysis import (Analysis, analyse, match_faces, people_of,
                                  whole_body)
from src.posture.face import Face
from src.posture.lifting import Lifting
from src.posture.overlay import Overlay
from src.posture.people import Tracker, distinct
from src.posture.regulation import warnings
from src.posture.skeleton import Point, Skeleton
from src.posture.sources import Frame
from src.posture.writer import describe_analysis
from test_posture_reba import Body

FPS = 10


def at(x: float, second: float = 0.0, body: Body | None = None,
       scale: float = 1.0) -> Skeleton:
    """A person standing with the middle of their image at ``x`` (0-1),
    ``scale`` times as tall as usual in the image."""
    skeleton = (body or Body()).skeleton(second)
    pixels = np.column_stack((x + skeleton.points[:, 0] * 0.25 * scale,
                              0.45 + skeleton.points[:, 1] * 0.4 * scale))
    return replace(skeleton, pixels=pixels)


def track(tracker: Tracker, frames: list[list[float]],
          start: float = 0.0) -> list[dict[int, float]]:
    """Each frame's people (by where they are), with their numbers."""
    found = []
    for n, places in enumerate(frames):
        second = start + n / FPS
        given = tracker.assign(second, [at(x, second) for x in places])
        found.append({number: float(np.mean(s.pixels[:, 0]))  # type: ignore
                      for number, s in given.items()})
    return found


class TestTracker:
    """Who is who from one frame to the next."""

    def test_one_person_is_always_one(self) -> None:
        """Walking across, leaving and coming back elsewhere: always 1."""
        tracker = Tracker()
        frames = ([[0.2 + 0.01 * n] for n in range(20)] + [[]] * 30
                  + [[0.8]] * 5)
        numbers = {n for frame in track(tracker, frames) for n in frame}
        assert numbers == {1}

    def test_two_people_keep_their_numbers(self) -> None:
        """Each person keeps their number, whatever the model's order."""
        tracker = Tracker()
        frames = [[0.2, 0.7] if n % 2 else [0.7, 0.2] for n in range(10)]
        found = track(tracker, frames)
        for frame in found:
            assert frame == pytest.approx(found[0], abs=0.05)

    def test_crossing(self) -> None:
        """Two people walking towards each other and past: each keeps
        their number (where they are going is followed)."""
        tracker = Tracker()
        frames = [[0.3 + 0.03 * n, 0.7 - 0.03 * n] for n in range(14)]
        found = track(tracker, frames)
        assert found[-1][1] > found[-1][2]

    def test_coming_back_keeps_the_number(self) -> None:
        """Person 2 leaves for a while and comes back where they were:
        number 2 again, not a new one."""
        tracker = Tracker()
        frames = [[0.2, 0.8]] * 5 + [[0.2]] * 30 + [[0.2, 0.75]] * 3
        found = track(tracker, frames)
        assert set(found[-1]) == {1, 2}
        assert found[-1][2] == pytest.approx(0.75, abs=0.05)

    def test_never_more_numbers_than_people_at_once(self) -> None:
        """People coming and going one at a time: one number."""
        tracker = Tracker()
        frames = [[0.2]] * 5 + [[]] * 20 + [[0.8]] * 5 + [[]] * 20 + [[0.5]]
        numbers = {n for frame in track(tracker, frames) for n in frame}
        assert numbers == {1}

    def test_the_most_people(self) -> None:
        """Four found, at most three followed."""
        tracker = Tracker(max_people=3)
        [frame] = track(tracker, [[0.1, 0.35, 0.6, 0.85]])
        assert set(frame) == {1, 2, 3}

    def test_present_through_a_short_gap(self) -> None:
        """Missing for less than a second still counts as there."""
        tracker = Tracker()
        tracker.assign(0.0, [at(0.5)])
        assert tracker.present(1, 0.9)
        assert not tracker.present(1, 1.2)
        assert not tracker.present(2, 0.1)

    def test_something_seen_an_instant_goes_last(self) -> None:
        """A chair taken for a person for an instant beside the person:
        when the person comes back they keep their number, not the
        chair's, even where the chair was."""
        tracker = Tracker()
        track(tracker, [[0.5]] * 30)
        track(tracker, [[0.5, 0.75]] * 3, start=3.0)
        found = track(tracker, [[]] * 20 + [[0.75]], start=3.3)
        assert set(found[-1]) == {1}


class TestTakeBack:
    """Standing up from a laptop (E2): something taken for a person for an
    instant keeps the person's number; the person takes it back."""

    def test_person_gets_their_number_back(self) -> None:
        tracker = Tracker()
        track(tracker, [[0.5]] * 30)
        # The chair where the person was, the person further.
        moved = track(tracker, [[0.5, 0.8]] * 3, start=3.0)
        assert moved[-1][2] == pytest.approx(0.8, abs=0.05)
        found = track(tracker, [[0.8]] * 20, start=3.3)
        assert set(found[-1]) == {1}

    def test_two_people_keep_theirs(self) -> None:
        """Two people seen long: nobody takes anybody's number."""
        tracker = Tracker()
        track(tracker, [[0.2, 0.7]] * 30)
        found = track(tracker, [[0.2, 0.7]] * 20, start=3.0)
        assert found[-1][1] == pytest.approx(0.2, abs=0.05)
        assert found[-1][2] == pytest.approx(0.7, abs=0.05)


class TestDistinct:
    """The same person found twice counts once."""

    def test_upper_half_inside_the_whole(self) -> None:
        whole = at(0.5)
        half = at(0.5)
        assert half.pixels is not None
        upper = half.visibility.copy()
        upper[[Point.LEFT_KNEE, Point.RIGHT_KNEE, Point.LEFT_ANKLE,
               Point.RIGHT_ANKLE, Point.LEFT_HEEL, Point.RIGHT_HEEL,
               Point.LEFT_FOOT, Point.RIGHT_FOOT]] = 0.0
        half = replace(half, visibility=upper * 0.9)
        assert distinct([half, whole]) == [whole]

    def test_two_people_apart(self) -> None:
        assert len(distinct([at(0.2), at(0.8)])) == 2

    def test_person_behind_is_not_a_duplicate(self) -> None:
        """A person far behind another is inside their box, but with the
        head elsewhere: two people, even when the one in front is less
        surely seen (legs hidden at a laptop)."""
        front = at(0.5, scale=2.5)
        front = replace(front, visibility=front.visibility * 0.6)
        behind = at(0.45, scale=0.4)
        assert len(distinct([front, behind])) == 2


class TestWholeBody:
    """The body is only scored seen whole."""

    def test_standing(self) -> None:
        assert whole_body(Body().skeleton())

    def test_legs_out_of_the_picture(self) -> None:
        """Standing up close to a laptop: hips seen, legs not."""
        body = Body()
        body.hidden = {Point.LEFT_KNEE, Point.RIGHT_KNEE, Point.LEFT_ANKLE,
                       Point.RIGHT_ANKLE}
        assert not whole_body(body.skeleton())

    def test_one_leg_is_enough(self) -> None:
        """Seen from the side, the far leg hidden."""
        body = Body()
        body.hidden = {Point.RIGHT_KNEE, Point.RIGHT_ANKLE}
        assert whole_body(body.skeleton())

    def test_nobody(self) -> None:
        assert not whole_body(None)


def face_at(x: float) -> Face:
    return Face(0.0, 5.0, 0.0, 0.0, 55.0, 0.0, (x, 0.3))


class TestMatchFaces:
    """Each person's face is the one at their nose."""

    def test_nearest(self) -> None:
        people = {1: at(0.2), 2: at(0.7)}
        found = match_faces(people, [face_at(0.7), face_at(0.2)])
        assert found[1].centre == (0.2, 0.3)
        assert found[2].centre == (0.7, 0.3)

    def test_one_and_one(self) -> None:
        """One person and one face: theirs, wherever it was found."""
        face = Face(0.0, 5.0, 0.0, 0.0, 55.0, 0.0)
        assert match_faces({3: at(0.5)}, [face]) == {3: face}

    def test_far_face_is_nobody_s(self) -> None:
        assert match_faces({1: at(0.2), 2: at(0.4)}, [face_at(0.9)]) == {}


class Poses:
    """A pose model giving ``people`` (where each one is) every frame;
    the person at ``bending`` bends the trunk 70° with the knees bent
    and an arm raised (REBA high) from 2 s to 6 s."""

    def __init__(self, people: list[float], bending: float | None = None,
                 gone: tuple[float, float] | None = None) -> None:
        self.people, self.bending, self.gone = people, bending, gone

    def find_all(self, frame: Frame) -> list[Skeleton]:
        found = []
        for x in self.people:
            second = frame.timestamp
            if (self.gone is not None and x == self.people[-1]
                    and self.gone[0] <= second < self.gone[1]):
                continue
            body = Body()
            if x == self.bending and 2 <= second < 6:
                body.bend_trunk(70).bend_knees(70).raise_arm(110)
            found.append(at(x, second, body))
        return found


def fake(model: object) -> Any:
    """A fake model, where ``analyse`` wants the real one."""
    return model


def frames(seconds: float) -> list[Frame]:
    return [Frame(n / FPS, np.zeros((40, 60, 3), dtype=np.uint8))
            for n in range(int(seconds * FPS))]


class TestAnalyseSeveral:
    """Each person is analysed apart."""

    def test_two_people(self) -> None:
        """Two people; only the one on the left bends: their moment of
        risk is theirs, told with their number."""
        told = []
        result = analyse(frames(10), fake(Poses([0.2, 0.7], bending=0.2)),
                         "test",
                         on_event=lambda person, event: told.append(person),
                         lifting=Lifting())
        assert [p.person for p in result.people] == [1, 2]
        first, second = result.people
        assert first.events and not second.events
        assert told == [1] * len(first.events)
        assert first.lifting is not None and second.lifting is not None
        assert first.seconds_seen == pytest.approx(9.9, abs=0.2)

    def test_the_drawing_has_everybody(self) -> None:
        drawn = []

        def keep(frame: Frame, seen: list[Any]) -> None:
            drawn.append([s.person for s in seen])

        analyse(frames(1), fake(Poses([0.2, 0.7])), "test",
                on_result=keep)
        assert drawn[-1] == [1, 2]

    def test_a_short_gap_is_not_an_absence(self) -> None:
        """Person 2 missing for half a second: still seen all along."""
        poses = fake(Poses([0.2, 0.7], gone=(4.0, 4.5)))
        result = analyse(frames(10), poses, "test")
        assert result.people[1].summary.seconds == pytest.approx(9.9,
                                                                 abs=0.2)

    def test_nobody(self) -> None:
        result = analyse(frames(2), fake(Poses([])), "test")
        assert result.people == []

    def test_an_instant_is_nobody(self) -> None:
        """Something taken for a person for a second is not reported."""

        class Instant(Poses):
            def find_all(self, frame: Frame) -> list[Skeleton]:
                found = super().find_all(frame)
                if 3 <= frame.timestamp < 4:
                    found.append(at(0.8, frame.timestamp))
                return found

        result = analyse(frames(10), fake(Instant([0.2])), "test")
        assert [p.person for p in result.people] == [1]

    def test_numbers_without_gaps(self) -> None:
        """Something seen first for an instant takes number 1 and is left
        out: the person is still "Persona 1" in the report."""

        class First(Poses):
            def find_all(self, frame: Frame) -> list[Skeleton]:
                # The thing alone first, then beside the person.
                if frame.timestamp < 0.3:
                    return [at(0.8, frame.timestamp)]
                found = super().find_all(frame)
                if frame.timestamp < 0.6:
                    found.append(at(0.8, frame.timestamp))
                return found

        result = analyse(frames(10), fake(First([0.2])), "test")
        assert [p.person for p in result.people] == [1]


class TestSaved:
    """Analyses saved before there were several people."""

    OLD: dict[str, Any] = {
        "source": "a.mp4", "created": "2026-11-17", "task": {},
        "summary": {"seconds": 5.0}, "events": [], "lifting": None,
        "office": None, "pushing": None}

    def test_people_of_an_old_one(self) -> None:
        [person] = people_of(self.OLD)
        assert person["person"] == 1
        assert person["summary"] == {"seconds": 5.0}

    def test_loads_as_one_person(self) -> None:
        analysis = Analysis.model_validate(self.OLD)
        assert len(analysis.people) == 1
        assert analysis.people[0].summary.seconds == 5.0

    def test_people_of_a_new_one(self) -> None:
        assert people_of({"people": []}) == []


def two_people() -> Analysis:
    return analyse(frames(10), fake(Poses([0.2, 0.7], bending=0.2)),
                   "test")


class TestJointReport:
    """The report of several people says whose each thing is."""

    def test_warnings_say_whose(self) -> None:
        found = warnings(two_people())
        assert found and all(text.startswith("Persona 1:")
                             for _, text in found)

    def test_one_person_warnings_unchanged(self) -> None:
        result = analyse(frames(10), fake(Poses([0.2], bending=0.2)),
                         "test")
        assert not any(text.startswith("Persona")
                       for _, text in warnings(result))

    def test_writer_describes_each(self) -> None:
        text = describe_analysis(two_people().model_dump())
        assert "Personas analizadas: 2" in text
        assert "# Persona 1" in text and "# Persona 2" in text

    def test_writer_nobody(self) -> None:
        text = describe_analysis({"source": "x", "people": []})
        assert "ninguna persona" in text


class TestOverlay:
    """Each person drawn with their number."""

    def test_draws_several(self) -> None:
        from src.posture.analysis import Seen

        image = np.zeros((200, 300, 3), dtype=np.uint8)
        seen = [Seen(1, at(0.3)), Seen(2, at(0.7))]
        drawn = Overlay().draw(image, 1.0, seen)
        assert drawn.shape == image.shape and drawn.any()

    def test_draws_nobody(self) -> None:
        image = np.zeros((200, 300, 3), dtype=np.uint8)
        assert Overlay().draw(image, 1.0, []).any()


class FakeLandmark:
    def __init__(self, x: float, y: float) -> None:
        self.x, self.y, self.z, self.visibility = x, y, 0.0, 1.0


class FakeLandmarker:
    """Finds one person filling the middle of whatever image it gets."""

    def __init__(self) -> None:
        self.images: list[tuple[int, int]] = []

    def detect_for_video(self, image, millisecond):  # type: ignore
        self.images.append(image.numpy_view().shape[:2])

        class Found:
            pose_world_landmarks = [[FakeLandmark(0, 0)] * 33]
            pose_landmarks = [[FakeLandmark(0.5, 0.5)] * 33]
        return Found()

    def close(self) -> None:
        pass


class FakeDetector:
    """Finds the boxes it is given for each second."""

    def __init__(self, boxes) -> None:  # type: ignore[no-untyped-def]
        self.boxes, self.calls, self.second = boxes, 0, 0.0

    def find(self, image):  # type: ignore[no-untyped-def]
        self.calls += 1
        return [(box, 0.9) for box in self.boxes(self.second)]


def estimator(boxes, people: int = 3):  # type: ignore[no-untyped-def]
    """A ``PoseEstimator`` with fake models."""
    from src.posture.pose import PoseEstimator

    pose = PoseEstimator.__new__(PoseEstimator)
    pose._path = None  # type: ignore[assignment]
    pose.people = people
    pose._single = FakeLandmarker()
    pose._detector = FakeDetector(boxes)  # type: ignore[assignment]
    pose._slots = []
    pose._several = False
    pose._checked = -1.0
    pose._one_since = None
    pose._last = -1
    pose._load = lambda path: FakeLandmarker()  # type: ignore
    return pose


def run_pose(pose, seconds: float):  # type: ignore[no-untyped-def]
    found = []
    for frame in frames(seconds):
        if pose._detector is not None:
            pose._detector.second = frame.timestamp
        found.append(pose.find_all(Frame(
            frame.timestamp, np.zeros((100, 200, 3), dtype=np.uint8))))
    return found


class TestPoseEstimator:
    """One person: the whole image, as calibrated; several: each in their
    part of the image."""

    def test_one_person_whole_image(self) -> None:
        """With one person, the detector only looks once a second."""
        pose = estimator(lambda second: [(0.4, 0.1, 0.6, 0.9)])
        found = run_pose(pose, 3)
        assert all(len(f) == 1 for f in found)
        assert pose._detector.calls == 3
        assert set(pose._single.images) == {(100, 200)}

    def test_several_in_their_part(self) -> None:
        """Two people: each found in their part, placed back in the whole
        image."""
        pose = estimator(lambda second: [(0.0, 0.0, 0.5, 1.0),
                                         (0.5, 0.0, 1.0, 1.0)])
        found = run_pose(pose, 1)
        assert all(len(f) == 2 for f in found)
        middles = sorted(float(s.pixels[0][0]) for s in found[-1])
        # Each box widened 20 % (and cut at the image's edge).
        assert middles == pytest.approx([0.3, 0.7])
        assert len(pose._slots) == 2

    def test_back_to_one(self) -> None:
        """Two people, then one for more than 2 s: the whole image
        again."""
        pose = estimator(lambda second: (
            [(0.0, 0.0, 0.5, 1.0), (0.5, 0.0, 1.0, 1.0)] if second < 1
            else [(0.0, 0.0, 0.5, 1.0)]))
        run_pose(pose, 4)
        assert not pose._several

    def test_only_one_wanted(self) -> None:
        """Without a detector (one person wanted): the whole image."""
        pose = estimator(lambda second: [])
        pose._detector = None
        assert len(run_pose(pose, 1)[-1]) == 1


class FakeNet:
    """YOLOX's raw output: one person in the cell at the middle of the
    image, 100 x 200 pixels of the model's input."""

    def setInput(self, blob) -> None:  # type: ignore[no-untyped-def]
        self.blob = blob

    def forward(self):  # type: ignore[no-untyped-def]
        from src.posture.detector import STRIDES, SIZE

        cells = sum((SIZE // s) ** 2 for s in STRIDES)
        out = np.zeros((1, cells, 85))
        out[0, :, 4] = -10.0
        # Stride 8 grid: the cell (26, 26) is at 208 pixels: the middle.
        n = 26 * (SIZE // 8) + 26
        out[0, n] = 0.0
        out[0, n, 2:4] = np.log([100 / 8, 200 / 8])
        out[0, n, 4:6] = 1.0
        return out


class TestDetector:
    """YOLOX's output read as boxes of people."""

    def test_one_person(self) -> None:
        from src.posture.detector import PersonDetector, _grid

        detector = PersonDetector.__new__(PersonDetector)
        detector._net = FakeNet()  # type: ignore[assignment]
        detector._cells, detector._strides = _grid()
        # A square image: the model's input is the image scaled.
        [(box, score)] = detector.find(np.zeros((416, 416, 3),
                                                dtype=np.uint8))
        assert score == pytest.approx(1.0)
        assert box == pytest.approx((158 / 416, 108 / 416, 258 / 416,
                                     308 / 416))

    def test_nobody(self) -> None:
        from src.posture.detector import PersonDetector, _grid

        detector = PersonDetector.__new__(PersonDetector)
        net = FakeNet()
        net.forward = lambda: np.zeros((1, 3549, 85))  # type: ignore
        detector._net = net  # type: ignore[assignment]
        detector._cells, detector._strides = _grid()
        assert detector.find(np.zeros((100, 100, 3), dtype=np.uint8)) == []
