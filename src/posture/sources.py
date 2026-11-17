"""Where the frames come from: a recorded video or a live camera.

Both give the same thing, frames with the second they were taken, so the
analysis never knows which one it reads. Moving to real time is choosing
``Camera`` instead of ``VideoFile``.
"""
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from ..exceptions import RagError


@dataclass(frozen=True)
class Frame:
    """One image and when it was taken.

    Attributes:
        timestamp: Seconds since the start of the video or the camera.
        image: The picture, in RGB (as the pose model expects).
    """

    timestamp: float
    image: NDArray[np.uint8]


class VideoFile:
    """The frames of a recorded video, with the time of each one.

    Attributes:
        path: The video file.
        every: Seconds between two analysed frames; frames in between are
            skipped. 0 keeps them all.
        step: One frame of every ``step`` is analysed: ``every`` times the
            frames a second of the video, rounded, so the frames kept are
            evenly spaced.
        fps: Frames a second analysed: the video's divided by ``step``.
    """

    def __init__(self, path: str | Path, every: float = 0.0) -> None:
        """Check the file can be opened; nothing is read yet.

        Raises:
            RagError: The file does not exist or is not a video.
        """
        self.path = Path(path)
        self.every = every
        if not self.path.is_file():
            raise RagError(f"Video not found: {self.path}")
        capture = cv2.VideoCapture(str(self.path))
        opened = capture.isOpened()
        native = capture.get(cv2.CAP_PROP_FPS) if opened else 0.0
        capture.release()
        if not opened:
            raise RagError(f"Cannot open the video: {self.path}")
        native = native if native > 0 else 30.0
        self.step = max(1, round(every * native))
        self.fps = native / self.step

    def __iter__(self) -> Iterator[Frame]:
        """The frames, in order, as fast as they can be read."""
        capture = cv2.VideoCapture(str(self.path))
        number = 0
        try:
            while True:
                # The skipped frames are only decoded far enough to move
                # on (grab), which is faster than reading them.
                if not capture.grab():
                    return
                number += 1
                if (number - 1) % self.step:
                    continue
                read, image = capture.retrieve()
                if not read:
                    return
                # The time of the frame inside the video, not the time it
                # took to read it.
                second = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000
                yield Frame(second, _rgb(image))
        finally:
            capture.release()


def _rgb(image: Any) -> NDArray[np.uint8]:
    """OpenCV reads BGR; the pose model expects RGB."""
    return np.asarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), dtype=np.uint8)


class Camera:
    """The frames of a live camera, with the time they were taken.

    Only the latest frame matters in real time: when the analysis is
    slower than the camera, the frames in between are dropped, never
    queued, so the analysis is never behind.

    Attributes:
        device: A camera number (0 is the first) or a stream address
            (``rtsp://...``).
        every: Seconds between two analysed frames, at least.
    """

    def __init__(self, device: int | str = 0, every: float = 0.0) -> None:
        """Remember the camera; it is opened when the frames are read."""
        self.device = device
        self.every = every

    def __iter__(self) -> Iterator[Frame]:
        """Frames until the camera stops or the reader stops asking.

        Raises:
            RagError: The camera cannot be opened.
        """
        capture = cv2.VideoCapture(self.device)
        if not capture.isOpened():
            raise RagError(f"Cannot open the camera: {self.device}")
        # Keep only the newest frame in the camera's buffer.
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        start = time.monotonic()
        last = -float("inf")
        try:
            while True:
                read, image = capture.read()
                if not read:
                    return
                second = time.monotonic() - start
                if second - last < self.every:
                    continue
                last = second
                yield Frame(second, _rgb(image))
        finally:
            capture.release()
