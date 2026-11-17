"""The skeleton of one person in one frame.

The points follow the 33 landmarks of MediaPipe Pose (BlazePose). Their
coordinates are in metres, centred between the hips, with ``x`` to the
right of the image, ``y`` down and ``z`` away from the camera, as
MediaPipe gives its "world" landmarks.
"""
from dataclasses import dataclass
from enum import IntEnum

import numpy as np
from numpy.typing import NDArray

# A point is used only when the model is at least this sure it is seen.
MIN_VISIBILITY = 0.5


class Point(IntEnum):
    """The landmarks the analysis uses, by their MediaPipe index."""

    NOSE = 0
    LEFT_EAR = 7
    RIGHT_EAR = 8
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_INDEX = 19
    RIGHT_INDEX = 20
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_FOOT = 31
    RIGHT_FOOT = 32


@dataclass(frozen=True)
class Skeleton:
    """One person in one frame.

    Attributes:
        timestamp: Seconds since the start of the video or the camera.
        points: 33 x 3 coordinates in metres (see the module).
        visibility: 33 values from 0 to 1: how sure the model is that
            each point is seen.
        pixels: 33 x 2 places in the image, from 0 to 1 (x to the right,
            y down), to draw the skeleton on it; ``None`` when unknown.
    """

    timestamp: float
    points: NDArray[np.float64]
    visibility: NDArray[np.float64]
    pixels: NDArray[np.float64] | None = None

    def at(self, point: Point) -> NDArray[np.float64]:
        """The coordinates of a point."""
        return np.asarray(self.points[point], dtype=np.float64)

    def seen(self, *points: Point) -> bool:
        """Whether every one of these points is seen well enough."""
        return all(self.visibility[p] >= MIN_VISIBILITY for p in points)

    def middle(self, left: Point, right: Point) -> NDArray[np.float64]:
        """The point halfway between two others (hips, shoulders...)."""
        return np.asarray((self.points[left] + self.points[right]) / 2,
                          dtype=np.float64)
