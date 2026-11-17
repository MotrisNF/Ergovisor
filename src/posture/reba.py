"""REBA (Rapid Entire Body Assessment), as the INSST describes it in its
technical note NTP 601 (Hignett and McAtamney's method).

The tables and ranges below are copied from NTP 601 and checked against
its text. The note does not give a number for some words ("upright",
"twisted", "bent sideways", "abducted"): those thresholds are the
constants marked as ours, so they can be tuned and are never hidden.

REBA scores one moment. Whether a posture is held, repeated or unstable
(the activity score) depends on time, so it comes from the timeline, not
from here.
"""
from dataclasses import dataclass
from typing import Literal

from .angles import Angles, Arm

Coupling = Literal[0, 1, 2, 3]

# Ours, not NTP 601: below these angles there is no flexion, twist or
# sideways bend worth a correction. They are above the error of a single
# camera, measured on real video: a person standing straight measures up
# to 10-15 degrees of flexion seen from the front, and 10-15 of sideways
# bend seen from the side, where depth is least precise.
UPRIGHT = 10.0
TWIST = 15.0
SIDE_BEND = 15.0

# NTP 601, Tabla A: [trunk 1-5][neck 1-3][legs 1-4].
TABLE_A = [
    [[1, 2, 3, 4], [1, 2, 3, 4], [3, 3, 5, 6]],
    [[2, 3, 4, 5], [3, 4, 5, 6], [4, 5, 6, 7]],
    [[2, 4, 5, 6], [4, 5, 6, 7], [5, 6, 7, 8]],
    [[3, 5, 6, 7], [5, 6, 7, 8], [6, 7, 8, 9]],
    [[4, 6, 7, 8], [6, 7, 8, 9], [7, 8, 9, 9]],
]
# NTP 601, Tabla B: [upper arm 1-6][lower arm 1-2][wrist 1-3].
TABLE_B = [
    [[1, 2, 2], [1, 2, 3]],
    [[1, 2, 3], [2, 3, 4]],
    [[3, 4, 5], [4, 5, 5]],
    [[4, 5, 5], [5, 6, 7]],
    [[6, 7, 8], [7, 8, 8]],
    [[7, 8, 8], [8, 9, 9]],
]
# NTP 601, Tabla C: [score A 1-12][score B 1-12].
TABLE_C = [
    [1, 1, 1, 2, 3, 3, 4, 5, 6, 7, 7, 7],
    [1, 2, 2, 3, 4, 4, 5, 6, 6, 7, 7, 8],
    [2, 3, 3, 3, 4, 5, 6, 7, 7, 8, 8, 8],
    [3, 4, 4, 4, 5, 6, 7, 8, 8, 9, 9, 9],
    [4, 4, 4, 5, 6, 7, 8, 8, 9, 9, 9, 9],
    [6, 6, 6, 7, 8, 8, 9, 9, 10, 10, 10, 10],
    [7, 7, 7, 8, 9, 9, 9, 10, 10, 11, 11, 11],
    [8, 8, 8, 9, 10, 10, 10, 10, 10, 11, 11, 11],
    [9, 9, 9, 10, 10, 10, 11, 11, 11, 12, 12, 12],
    [10, 10, 10, 11, 11, 11, 11, 12, 12, 12, 12, 12],
    [11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12],
    [12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12],
]
# NTP 601: final score -> (action level, risk, intervention).
LEVELS = [
    (1, 0, "Inapreciable", "No necesaria"),
    (3, 1, "Bajo", "Puede ser necesaria"),
    (7, 2, "Medio", "Necesaria"),
    (10, 3, "Alto", "Necesaria pronto"),
    (15, 4, "Muy alto", "Actuación inmediata"),
]


@dataclass(frozen=True)
class Task:
    """What the video cannot show about the task, given by the person.

    Attributes:
        load_kg: The load handled or the force applied.
        sudden: Whether the load or force comes suddenly.
        coupling: How good the grip is: 0 good, 1 fair, 2 poor,
            3 unacceptable (NTP 601).
    """

    load_kg: float = 0.0
    sudden: bool = False
    coupling: Coupling = 0


@dataclass(frozen=True)
class Reba:
    """The REBA score of one moment, with every part of it.

    Attributes:
        trunk, neck, legs: Scores of group A.
        upper_arm, lower_arm, wrist: Scores of group B, for the worse arm.
        side: Which arm group B is: ``izquierdo`` or ``derecho``.
        a, b: Scores A (table A plus load) and B (table B plus coupling).
        activity: Points for held, repeated or unstable postures.
        score: Final score, 1 to 15.
        level: Action level, 0 to 4.
        risk: Risk in words (NTP 601).
        action: Intervention needed (NTP 601).
    """

    trunk: int
    neck: int
    legs: int
    upper_arm: int
    lower_arm: int
    wrist: int
    side: str
    a: int
    b: int
    activity: int
    score: int
    level: int
    risk: str
    action: str


def trunk_score(angles: Angles) -> int:
    """Upright 1; 0-20 degrees of flexion or extension 2; 20-60 of
    flexion or more than 20 of extension 3; more than 60 of flexion 4;
    +1 when twisted or bent sideways."""
    angle = angles.trunk
    if abs(angle) < UPRIGHT:
        score = 1
    elif -20 <= angle <= 20:
        score = 2
    elif angle <= 60:
        score = 3
    else:
        score = 4
    bent = angles.trunk_twist > TWIST or angles.trunk_side > SIDE_BEND
    return score + (1 if bent else 0)


def neck_score(angles: Angles) -> int:
    """0-20 degrees of flexion 1; more than 20 of flexion, or extension,
    2; +1 when twisted or bent sideways."""
    score = 2 if angles.neck > 20 or angles.neck < -UPRIGHT else 1
    twisted = angles.neck_twist is not None and angles.neck_twist > TWIST
    return score + (1 if twisted or angles.neck_side > SIDE_BEND else 0)


def legs_score(angles: Angles) -> int:
    """Both feet on the ground 1, one 2; +1 with the knees bent 30-60
    degrees, +2 with more than 60."""
    score = 2 if angles.one_leg else 1
    if angles.knee > 60:
        return score + 2
    return score + (1 if angles.knee >= 30 else 0)


def upper_arm_score(arm: Arm) -> int:
    """20 degrees of extension to 20 of flexion 1; more than 20 of
    extension or 21-45 of flexion 2; 46-90 of flexion 3; more than 90 4;
    +1 when abducted."""
    angle = arm.upper
    if angle < -20:
        score = 2
    elif angle <= 20:
        score = 1
    elif angle <= 45:
        score = 2
    elif angle <= 90:
        score = 3
    else:
        score = 4
    return score + (1 if arm.abducted else 0)


def lower_arm_score(arm: Arm) -> int:
    """60-100 degrees of flexion 1; less than 60 or more than 100 2."""
    return 1 if 60 <= arm.elbow <= 100 else 2


def wrist_score(arm: Arm) -> int:
    """0-15 degrees of flexion or extension 1; more than 15 2. A wrist
    that is not seen counts as 1: nothing is guessed."""
    return 2 if arm.wrist is not None and arm.wrist > 15 else 1


def load_score(task: Task) -> int:
    """Less than 5 kg 0; 5-10 kg 1; more than 10 kg 2; +1 if sudden."""
    score = 0 if task.load_kg < 5 else 1 if task.load_kg <= 10 else 2
    return score + (1 if task.sudden else 0)


def level_of(score: int) -> tuple[int, str, str]:
    """The action level, the risk and the action of a final score."""
    for top, level, risk, action in LEVELS:
        if score <= top:
            return level, risk, action
    return LEVELS[-1][1:]


def score(angles: Angles, task: Task = Task(),
          activity: int = 0) -> Reba | None:
    """The REBA score of one moment, or ``None`` without any arm seen.

    Group B is scored for each arm seen and the worse one is kept, as an
    evaluator would look at the side under more strain.

    Args:
        angles: The angles of the skeleton.
        task: The load and the grip, which the video does not show.
        activity: Points for held, repeated or unstable postures
            (0 to 3), from the timeline.
    """
    arms = [(name, arm) for name, arm in (("izquierdo", angles.left),
                                          ("derecho", angles.right))
            if arm is not None]
    if not arms:
        return None
    trunk, neck, legs = (trunk_score(angles), neck_score(angles),
                         legs_score(angles))
    a = TABLE_A[trunk - 1][neck - 1][legs - 1] + load_score(task)
    groups = []
    for name, arm in arms:
        upper, lower, wrist = (upper_arm_score(arm), lower_arm_score(arm),
                               wrist_score(arm))
        b = TABLE_B[upper - 1][lower - 1][wrist - 1] + task.coupling
        groups.append((b, name, upper, lower, wrist))
    b, side, upper, lower, wrist = max(groups)
    final = TABLE_C[min(a, 12) - 1][min(b, 12) - 1] + activity
    level, risk, action = level_of(final)
    return Reba(trunk=trunk, neck=neck, legs=legs, upper_arm=upper,
                lower_arm=lower, wrist=wrist, side=side, a=a, b=b,
                activity=activity, score=final, level=level, risk=risk,
                action=action)
