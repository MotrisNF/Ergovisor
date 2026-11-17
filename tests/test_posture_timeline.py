"""Tests for ``src.posture.timeline``: postures over time.

The frames are skeletons built by hand (see ``test_posture_reba``), at
10 frames a second.
"""
from collections.abc import Callable

import pytest

from src.posture.angles import Angles, measure
from src.posture.reba import Task
from src.posture.timeline import Event, Timeline
from test_posture_reba import Body

FPS = 10


def stooping(degrees: float = 70) -> Body:
    """Picking something up from the floor: the trunk bent forwards, the
    knees bent 45 degrees and the arms hanging straight down, as gravity
    leaves them (raised ``degrees`` from the trunk).

    REBA scores this a medium risk (5). Bent with straight knees and no
    load it is only a low one (3): that is the method, not a mistake.
    """
    return (Body().bend_trunk(degrees).raise_arm(degrees)
            .raise_arm(degrees, side="RIGHT").bend_knees(45))


def angles(body: Body, second: float) -> Angles:
    """The angles of a body at a given second."""
    found = measure(body.skeleton(second))
    assert found is not None
    return found


def play(timeline: Timeline, seconds: float, start: float,
         posture: Callable[[float], Body]) -> tuple[list[int], list[Event]]:
    """Feed ``seconds`` of frames; the scores and the moments that ended.

    ``posture`` gives the body at each second.
    """
    scores, events = [], []
    for n in range(int(seconds * FPS)):
        second = start + n / FPS
        result, ended = timeline.add(second, angles(posture(second), second))
        if result is not None:
            scores.append(result.score)
        if ended is not None:
            events.append(ended)
    return scores, events


class TestActivity:
    """The activity points of NTP 601, seen over time."""

    def test_held_more_than_a_minute(self) -> None:
        """Bent and still: one more point after 60 seconds."""
        timeline = Timeline()
        scores, _ = play(timeline, 70, 0, lambda s: Body().bend_trunk(30))
        assert scores[10 * FPS] + 1 == scores[-1]

    def test_neutral_still_is_not_held(self) -> None:
        """Standing still is not a forced posture."""
        scores, _ = play(Timeline(), 70, 0, lambda s: Body())
        assert scores[0] == scores[-1]

    def test_repeated(self) -> None:
        """Bending forward 6 times in a minute: one more point."""
        def bending(second: float) -> Body:
            return Body().bend_trunk(40 if int(second) % 10 < 3 else 0)

        timeline = Timeline()
        scores, _ = play(timeline, 59, 0, bending)
        bent_late = scores[-int(9.5 * FPS)]
        bent_first = scores[int(1 * FPS)]
        assert bent_late == bent_first + 1

    def test_sudden_change(self) -> None:
        """From upright to 60 degrees in half a second."""
        timeline = Timeline()
        play(timeline, 3, 0, lambda s: Body())
        quick, _ = play(timeline, 1, 3, lambda s: Body().bend_trunk(60))
        slow_timeline = Timeline()
        play(slow_timeline, 3, 0, lambda s: Body())
        alone, _ = play(slow_timeline, 0.1, 3, lambda s: Body())
        steady = Timeline()
        held, _ = play(steady, 5, 0, lambda s: Body().bend_trunk(60))
        assert quick[-1] == held[-1] + 1
        assert alone


class TestEvents:
    """Moments of risk, given when they end."""

    @staticmethod
    def session(posture: Callable[[float], Body],
                seconds: float = 15) -> tuple[Timeline, list[Event]]:
        """A recording, finished; every moment of risk in it."""
        timeline = Timeline()
        _, events = play(timeline, seconds, 0, posture)
        last = timeline.finish()
        return timeline, events + ([last] if last else [])

    def test_bent_moment(self) -> None:
        """Upright, bent at 70 degrees for 5 s, upright: one moment."""
        _, events = self.session(
            lambda s: stooping() if 5 <= s < 10 else Body())
        [event] = events
        assert event.start == pytest.approx(5) and event.end == \
            pytest.approx(9.9)
        assert event.level >= 2
        assert "tronco flexionado 70°" in event.causes
        assert any(c.startswith("brazo") for c in event.causes)

    def test_given_as_soon_as_it_ends(self) -> None:
        """Live: the moment comes a little after the risk stops, not at
        the end of the recording."""
        timeline = Timeline()
        _, events = play(timeline, 12, 0, lambda s: stooping()
                         if 5 <= s < 10 else Body())
        assert len(events) == 1

    def test_short_gap_does_not_split(self) -> None:
        """Half a second upright inside a bent moment: still one."""
        _, events = self.session(
            lambda s: Body() if s < 2 or 6 <= s < 6.5 or s >= 12
            else stooping())
        assert len(events) == 1

    def test_blip_is_not_reported(self) -> None:
        """Less than a second bent: noise, not a moment."""
        _, events = self.session(
            lambda s: stooping() if 5 <= s < 5.5 else Body())
        assert events == []

    def test_load_makes_it_risky(self) -> None:
        """The same posture with 15 kg in the hands."""
        timeline = Timeline(Task(load_kg=15))
        _, events = play(timeline, 5, 0, lambda s: Body().bend_trunk(25))
        light = Timeline()
        play(light, 5, 0, lambda s: Body().bend_trunk(25))
        assert timeline.summary.max_score > light.summary.max_score


class TestSummary:
    """Time at each level."""

    def test_seconds_by_level(self) -> None:
        """The seconds add up to the time covered."""
        timeline, _ = TestEvents.session(
            lambda s: stooping() if s >= 5 else Body(), 10)
        summary = timeline.summary
        assert summary.frames == summary.frames_with_person == 100
        assert summary.seconds == pytest.approx(9.9)
        assert sum(summary.seconds_by_level.values()) == pytest.approx(9.9)
        assert summary.max_level >= 2

    def test_frames_without_person(self) -> None:
        """A frame without a skeleton counts, but not as time at a
        level."""
        timeline = Timeline()
        timeline.add(0.0, angles(Body(), 0.0))
        timeline.add(0.1, None)
        timeline.add(0.2, angles(Body(), 0.2))
        assert timeline.summary.frames == 3
        assert timeline.summary.frames_with_person == 2


class TestSmoother:
    """``Smoother``: the median of the last half second."""

    def test_removes_a_jump(self) -> None:
        """One frame off by 40 degrees does not move the result."""
        from src.posture.smoothing import Smoother

        smoother = Smoother()
        for n in range(5):
            smoother.add(Body().skeleton(n / FPS))
        jumped = smoother.add(Body().bend_trunk(40).skeleton(0.5))
        found = measure(jumped)
        assert found is not None and abs(found.trunk) < 1

    def test_follows_a_real_movement(self) -> None:
        """A posture held for longer than the window is reached."""
        from src.posture.smoothing import Smoother

        smoother = Smoother()
        last = None
        for n in range(10):
            last = smoother.add(Body().bend_trunk(40).skeleton(n / FPS))
        assert last is not None
        found = measure(last)
        assert found is not None and found.trunk == pytest.approx(40, abs=1)
