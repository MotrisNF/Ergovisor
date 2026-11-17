"""The whole analysis: frames in, moments of risk and a summary out.

It reads any source of frames (a file or a camera) one frame at a time
and calls ``on_event`` as soon as a moment of risk ends: printed for a
recording, it would be a warning for a live camera.
"""
from collections.abc import Callable, Iterable
from datetime import datetime, timezone

from pydantic import BaseModel

from .angles import Angles, measure
from .lifting import Lift, Lifting, LiftingReport
from .pose import PoseEstimator
from .reba import Reba, Task
from .skeleton import Skeleton
from .smoothing import Smoother
from .sources import Frame
from .timeline import Event, Summary, Timeline

# Called with each analysed frame: the frame, its smoothed skeleton, its
# angles, its score and the moment of risk going on. Returning False stops
# the analysis (a window closed by the person).
OnResult = Callable[[Frame, Skeleton | None, Angles | None, Reba | None,
                     Event | None, Lift | None], bool | None]
# A lift stays on the drawing this long after it ends.
LIFT_SHOWN = 5.0

METHOD = ("REBA (Rapid Entire Body Assessment), según la NTP 601 del INSST "
          "(Hignett y McAtamney)")
UNMEASURED = ["hombro elevado", "brazo apoyado", "rotación del brazo",
              "torsión o desviación de muñeca"]


class Analysis(BaseModel):
    """What an analysis found.

    Attributes:
        source: The video file or the camera.
        created: When the analysis was made (UTC).
        method: The method and where it is described.
        task: The load and the grip given for the task.
        summary: Time at each level, the highest score...
        events: The moments of risk, in order.
        unmeasured: What the method asks that a skeleton cannot show:
            those points are never added, so the score may be lower than
            an evaluator's.
        lifting: The lifts and lowerings scored with the NIOSH equation,
            or ``None`` when not analysed.
    """

    source: str
    created: str
    method: str = METHOD
    task: dict[str, float | bool | int]
    summary: Summary
    events: list[Event]
    unmeasured: list[str] = UNMEASURED
    lifting: LiftingReport | None = None


def analyse(frames: Iterable[Frame], estimator: PoseEstimator,
            source: str, task: Task = Task(), alert_level: int = 2,
            seconds: float = 0.0,
            on_event: Callable[[Event], None] | None = None,
            on_frame: Callable[[Frame], None] | None = None,
            interruptible: bool = False,
            on_result: OnResult | None = None,
            lifting: Lifting | None = None,
            on_lift: Callable[[Lift], None] | None = None) -> Analysis:
    """Analyse frames until they end, or for ``seconds`` (0: no limit).

    Args:
        frames: A ``VideoFile`` or a ``Camera``.
        estimator: The pose model, already loaded.
        source: The name of the source, for the report.
        task: The load and the grip.
        alert_level: Action level from which a moment is of risk.
        seconds: Stop after this time (for a camera); 0 never stops.
        on_event: Called with each moment of risk as soon as it ends.
        on_frame: Called with each frame read (a progress bar).
        interruptible: Ctrl+C ends the reading and the analysis so far is
            kept, as a live camera is stopped; otherwise it stops
            everything.
        on_result: Called with each analysed frame and what was found in
            it (to draw it); returning False stops the analysis.
        lifting: Finds and scores the lifts, with the load and the
            person's data; ``None`` does not look for them.
        on_lift: Called with each lift as soon as it ends.

    Returns:
        The summary and every moment of risk.
    """
    timeline = Timeline(task, alert_level)
    smoother = Smoother()
    events: list[Event] = []

    def keep(event: Event | None) -> None:
        if event is not None:
            events.append(event)
            if on_event is not None:
                on_event(event)

    try:
        for frame in frames:
            if seconds and frame.timestamp > seconds:
                break
            if on_frame is not None:
                on_frame(frame)
            skeleton = estimator.find(frame)
            if skeleton is None:
                smoother.reset()
                angles = None
            else:
                skeleton = smoother.add(skeleton)
                angles = measure(skeleton)
            result, ended = timeline.add(frame.timestamp, angles)
            keep(ended)
            if lifting is not None:
                lift = lifting.add(skeleton)
                if lift is not None and on_lift is not None:
                    on_lift(lift)
            recent = None
            if lifting is not None and lifting.lifts and (
                    frame.timestamp - lifting.lifts[-1].end.second
                    <= LIFT_SHOWN):
                recent = lifting.lifts[-1]
            if on_result is not None and on_result(
                    frame, skeleton, angles, result, timeline.current,
                    recent) is False:
                break
    except KeyboardInterrupt:
        if not interruptible:
            raise
    keep(timeline.finish())
    return Analysis(
        source=source, created=datetime.now(timezone.utc).isoformat(
            timespec="seconds"),
        task={"load_kg": task.load_kg, "sudden": task.sudden,
              "coupling": task.coupling},
        summary=timeline.summary, events=events,
        lifting=lifting.report() if lifting is not None else None)


def describe_lift(lift: Lift) -> str:
    """One line for a lift: when, how much, its limit and its index."""
    def clock(second: float) -> str:
        return f"{int(second // 60):02d}:{second % 60:04.1f}"

    index = (f", IL {lift.index:.2f} ({lift.risk.lower()})"
             if lift.index is not None else "")
    notes = f". {'; '.join(lift.notes)}" if lift.notes else ""
    return (f"[{clock(lift.start.second)}–{clock(lift.end.second)}] "
            f"{lift.kind.capitalize()} de {lift.d:.0f} cm: LPR "
            f"{lift.limit:.1f} kg{index}{notes}")


def describe(event: Event) -> str:
    """One line for a moment of risk: when, how much and why."""
    def clock(second: float) -> str:
        return f"{int(second // 60):02d}:{second % 60:04.1f}"

    return (f"[{clock(event.start)}–{clock(event.end)}] Riesgo "
            f"{event.risk.lower()} (REBA {event.score}, nivel "
            f"{event.level}): {', '.join(event.causes) or 'postura'}")
