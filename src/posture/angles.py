"""The angles of the joints of one skeleton, in degrees.

The angles are measured in the body's own frame, not the camera's: an
upward axis, a lateral axis from the left hip to the right one, and a
forward axis that points where the nose (or else the toes) is. So a
trunk bent forward measures the same whether the camera sees the person
from the front or from the side.

REBA needs some things a skeleton cannot show (a raised shoulder, an arm
resting on something, a twisted or deviated wrist): they are not
measured, never guessed, and ``Angles.unmeasured`` names them.
"""
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from .skeleton import Point, Skeleton

Vector = NDArray[np.float64]
# MediaPipe's y axis points down.
UP: Vector = np.array([0.0, -1.0, 0.0])
# A foot this much higher than the other (metres) stands on one leg.
FOOT_LIFT = 0.10
# Ours: in a neutral posture the ear is ahead of the shoulder, so the line
# from the shoulders to the ears leans forward this much with the head
# straight (measured on real video: 15-25 degrees seen from the side).
NECK_NEUTRAL = 15.0
# Ours: an arm counts as abducted when raised mostly sideways and more
# than this; below it, the sideways part of a hanging arm is mostly the
# depth error of one camera (seen on real video from the side).
ABDUCTION_MIN = 45.0
# Ours: a sideways bend (of the trunk, or of the head on it) is only
# measured when the person's left-right axis lies at least this much
# across the image (its share along the image's x; 1 facing the camera,
# 0 side-on). Side-on, that bend points at the camera, the depth the
# model measures worst: standing straight it read 10-13° (REBA counts
# from 15°), facing the camera 1°; real bends facing it, 13-19° (B1, B2,
# 2026-11-18). Not measurable, it is not scored.
SIDE_SEEN = 0.5


@dataclass(frozen=True)
class Arm:
    """The angles of one arm.

    Attributes:
        upper: Elevation of the upper arm from hanging along the trunk;
            negative when it goes backwards (extension).
        abducted: Whether the arm is raised mostly to the side.
        elbow: Flexion of the elbow (0 is straight).
        wrist: Bend of the wrist from the line of the forearm, or
            ``None`` when the hand is not seen.
    """

    upper: float
    abducted: bool
    elbow: float
    wrist: float | None


@dataclass(frozen=True)
class Angles:
    """The angles REBA needs, from one skeleton.

    Attributes:
        timestamp: Seconds since the start.
        trunk: Flexion of the trunk from vertical; negative is extension.
        trunk_side: Sideways bend of the trunk, in absolute value; 0
            when the person is seen side-on (see ``SIDE_SEEN``).
        trunk_twist: Twist of the shoulders against the hips.
        neck: Flexion of the head relative to the trunk; negative is
            extension.
        neck_side: Sideways bend of the head relative to the trunk; 0
            seen side-on.
        neck_twist: Twist of the head against the shoulders, or ``None``
            when both ears are not seen.
        knee: The larger flexion of both knees.
        one_leg: Whether one foot is lifted.
        left: The left arm, or ``None`` when it is not seen.
        right: The right arm, or ``None`` when it is not seen.
        unmeasured: What REBA asks that the skeleton cannot show.
    """

    timestamp: float
    trunk: float
    trunk_side: float
    trunk_twist: float
    neck: float
    neck_side: float
    neck_twist: float | None
    knee: float
    one_leg: bool
    left: Arm | None
    right: Arm | None
    unmeasured: tuple[str, ...] = field(default=(
        "hombro elevado", "brazo apoyado", "rotación del brazo",
        "torsión o desviación de muñeca"))


def _unit(vector: Vector) -> Vector:
    """The vector with length 1 (a null vector stays null)."""
    length = float(np.linalg.norm(vector))
    return vector / length if length > 1e-9 else vector


def _flat(vector: Vector, normal: Vector) -> Vector:
    """The vector without its part along ``normal``."""
    return vector - float(vector @ normal) * normal


def _between(a: Vector, b: Vector) -> float:
    """The angle between two vectors, from 0 to 180 degrees."""
    cosine = float(_unit(a) @ _unit(b))
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def _joint(a: Vector, joint: Vector, b: Vector) -> float:
    """The flexion of a joint: 0 when ``a``, ``joint`` and ``b`` are in a
    straight line."""
    return 180.0 - _between(a - joint, b - joint)


def _signed(vector: Vector, along: Vector, towards: Vector) -> float:
    """The angle of ``vector`` from ``along``, positive towards
    ``towards``, in the plane of both axes."""
    return float(np.degrees(np.arctan2(vector @ towards, vector @ along)))


def body_axes(skeleton: Skeleton) -> tuple[Vector, Vector, Vector]:
    """The up, lateral (to the person's right) and forward axes.

    Forward is the side of the nose, or of the toes when the nose is not
    seen (a person with their back to the camera).
    """
    p = skeleton.at
    lateral = _unit(_flat(p(Point.RIGHT_HIP) - p(Point.LEFT_HIP), UP))
    forward = _unit(np.asarray(np.cross(UP, lateral), dtype=np.float64))
    if skeleton.seen(Point.NOSE):
        ahead = p(Point.NOSE) - skeleton.middle(Point.LEFT_SHOULDER,
                                                Point.RIGHT_SHOULDER)
    else:
        ahead = (skeleton.middle(Point.LEFT_FOOT, Point.RIGHT_FOOT)
                 - skeleton.middle(Point.LEFT_ANKLE, Point.RIGHT_ANKLE))
    if float(_flat(ahead, UP) @ forward) < 0:
        forward = -forward
    return UP, lateral, forward


def _arm(skeleton: Skeleton, shoulder: Point, elbow: Point, wrist: Point,
         index: Point, trunk: Vector, lateral: Vector,
         forward: Vector) -> Arm | None:
    """The angles of one arm, relative to the trunk."""
    if not skeleton.seen(shoulder, elbow, wrist):
        return None
    p = skeleton.at
    upper = p(elbow) - p(shoulder)
    down = -trunk
    elevation = _between(upper, down)
    backwards = float(upper @ forward) < 0
    sideways = abs(float(upper @ lateral)) > abs(float(upper @ forward))
    hand = (_joint(p(elbow), p(wrist), p(index))
            if skeleton.seen(index) else None)
    return Arm(upper=-elevation if backwards and not sideways else elevation,
               abducted=sideways and elevation > ABDUCTION_MIN,
               elbow=_joint(p(shoulder), p(elbow), p(wrist)), wrist=hand)


def measure(skeleton: Skeleton) -> Angles | None:
    """The angles of a skeleton, or ``None`` when the hips and shoulders,
    which everything is measured from, are not seen."""
    if not skeleton.seen(Point.LEFT_HIP, Point.RIGHT_HIP,
                         Point.LEFT_SHOULDER, Point.RIGHT_SHOULDER):
        return None
    p = skeleton.at
    up, lateral, forward = body_axes(skeleton)
    hips = skeleton.middle(Point.LEFT_HIP, Point.RIGHT_HIP)
    shoulders = skeleton.middle(Point.LEFT_SHOULDER, Point.RIGHT_SHOULDER)
    trunk = shoulders - hips
    trunk_flexion = _signed(trunk, up, forward)
    # MediaPipe's x axis is the image's.
    side_seen = abs(float(lateral[0])) >= SIDE_SEEN
    trunk_side = abs(_signed(trunk, up, lateral)) if side_seen else 0.0
    shoulder_line = _flat(p(Point.RIGHT_SHOULDER) - p(Point.LEFT_SHOULDER),
                          up)
    twist = _between(shoulder_line, lateral)

    ears = (Point.LEFT_EAR, Point.RIGHT_EAR)
    seen_ears = [e for e in ears if skeleton.seen(e)]
    head_base = (np.mean([p(e) for e in seen_ears], axis=0) if seen_ears
                 else p(Point.NOSE))
    head = head_base - shoulders
    neck = _signed(head, up, forward) - trunk_flexion - NECK_NEUTRAL
    neck_side = (abs(_signed(head, up, lateral)
                     - _signed(trunk, up, lateral)) if side_seen else 0.0)
    neck_twist = (_between(_flat(p(Point.RIGHT_EAR) - p(Point.LEFT_EAR), up),
                           shoulder_line)
                  if len(seen_ears) == 2 else None)

    knees = [_joint(p(hip), p(knee), p(ankle))
             for hip, knee, ankle in ((Point.LEFT_HIP, Point.LEFT_KNEE,
                                       Point.LEFT_ANKLE),
                                      (Point.RIGHT_HIP, Point.RIGHT_KNEE,
                                       Point.RIGHT_ANKLE))
             if skeleton.seen(hip, knee, ankle)]
    one_leg = (skeleton.seen(Point.LEFT_ANKLE, Point.RIGHT_ANKLE)
               and abs(float((p(Point.LEFT_ANKLE) - p(Point.RIGHT_ANKLE))
                             @ up)) > FOOT_LIFT)

    trunk_axis = _unit(trunk)
    trunk_lateral = _unit(_flat(lateral, trunk_axis))
    trunk_forward = _unit(_flat(forward, trunk_axis))
    return Angles(
        timestamp=skeleton.timestamp, trunk=trunk_flexion,
        trunk_side=trunk_side, trunk_twist=twist, neck=neck,
        neck_side=neck_side, neck_twist=neck_twist,
        knee=max(knees, default=0.0), one_leg=one_leg,
        left=_arm(skeleton, Point.LEFT_SHOULDER, Point.LEFT_ELBOW,
                  Point.LEFT_WRIST, Point.LEFT_INDEX, trunk_axis,
                  trunk_lateral, trunk_forward),
        right=_arm(skeleton, Point.RIGHT_SHOULDER, Point.RIGHT_ELBOW,
                   Point.RIGHT_WRIST, Point.RIGHT_INDEX, trunk_axis,
                   trunk_lateral, trunk_forward))
