"""Steadier skeletons: the median of the last instants.

A pose model moves its points a little from one frame to the next even
when the person is still, and a REBA score near a limit then jumps up and
down. The median of each point over the last half second removes those
jumps. It only uses frames already seen, so it works the same live; the
cost is a delay of about half the window.
"""
from collections import deque

import numpy as np

from .skeleton import Skeleton

# Ours: long enough to remove jitter, short enough not to hide a
# movement.
WINDOW = 0.5


class Smoother:
    """The median skeleton of the last ``window`` seconds."""

    def __init__(self, window: float = WINDOW) -> None:
        """Nothing seen yet."""
        self.window = window
        self._recent: deque[Skeleton] = deque()

    def add(self, skeleton: Skeleton) -> Skeleton:
        """This skeleton, with each point the median of the recent ones.

        The time is the newest one: the smoothed skeleton stands for now.
        """
        self._recent.append(skeleton)
        while skeleton.timestamp - self._recent[0].timestamp > self.window:
            self._recent.popleft()
        points = np.median([s.points for s in self._recent], axis=0)
        visibility = np.median([s.visibility for s in self._recent], axis=0)
        return Skeleton(skeleton.timestamp, np.asarray(points, dtype=float),
                        np.asarray(visibility, dtype=float))

    def reset(self) -> None:
        """Forget the recent skeletons (the person was lost)."""
        self._recent.clear()
