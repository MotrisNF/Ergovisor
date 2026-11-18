"""Where the frames come from: a recorded video or a live camera.

Both give the same thing, frames with the second they were taken, so the
analysis never knows which one it reads. Moving to real time is choosing
``Camera`` instead of ``VideoFile``.
"""
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
import sys
import threading
import time
from typing import Any
from urllib.parse import urlsplit, urlunsplit

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


# Query parameters some cameras take their credentials in
# (``?user=admin&password=...``): their values are hidden too.
SECRET_KEYS = ("user", "pass", "pwd", "token", "key", "auth")


def safe_name(device: int | str) -> str:
    """How to name a camera in messages and reports, without its password.

    ``rtsp://admin:secret@192.168.1.50:554/s1`` becomes
    ``camera rtsp://***@192.168.1.50:554/s1``: the analysis JSON and the
    terminal keep where the camera is, never how to log in to it.

    Args:
        device: A camera number or a stream address.

    Returns:
        ``camera <number>`` or ``camera <address without credentials>``.
    """
    if isinstance(device, int):
        return f"camera {device}"
    parts = urlsplit(device)
    netloc = parts.netloc
    if "@" in netloc:
        netloc = "***@" + netloc.rpartition("@")[2]
    query = "&".join(
        f"{name}=***" if any(k in name.lower() for k in SECRET_KEYS)
        else f"{name}={value}" if value or "=" in pair else name
        for pair in parts.query.split("&") if pair
        for name, _, value in [pair.partition("=")])
    return "camera " + urlunsplit(
        (parts.scheme, netloc, parts.path, query, parts.fragment))


class _Latest:
    """The newest frame read, shared by the reading thread and the
    analysis: a new frame replaces the one not yet taken."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._frame: tuple[float, Any] | None = None
        self.finished = False
        self.stopped = threading.Event()

    def put(self, second: float, image: Any) -> None:
        """Keep a frame, dropping the previous one if not taken."""
        with self._condition:
            self._frame = (second, image)
            self._condition.notify()

    def finish(self) -> None:
        """No frame will come any more."""
        with self._condition:
            self.finished = True
            self._condition.notify()

    def take(self) -> tuple[float, Any] | None:
        """Wait for a frame not taken yet; ``None`` when they ended.

        The wait wakes up now and then, so Ctrl+C stops it at once.
        """
        with self._condition:
            while self._frame is None and not self.finished:
                self._condition.wait(timeout=0.5)
            frame, self._frame = self._frame, None
            return frame


class Recorder:
    """Keeps every frame a camera gives as a video file, at the camera's
    real rate (the original picture, never the drawn one).

    Many cameras, IP ones above all, do not say their frame rate, or say
    a wrong one: it is measured over the first ``MEASURE`` seconds, and
    those frames are written once it is known.

    Attributes:
        path: The video file.
    """

    MEASURE = 2.0
    FALLBACK_FPS = 30.0

    def __init__(self, path: Path) -> None:
        self.path = path
        self._writer: Any = None
        self._waiting: list[tuple[float, Any]] = []

    def add(self, second: float, image: Any) -> None:
        """One frame read from the camera (BGR)."""
        if self._writer is not None:
            self._writer.write(image)
            return
        self._waiting.append((second, image))
        if second - self._waiting[0][0] >= self.MEASURE:
            self._open()

    def _open(self) -> None:
        span = self._waiting[-1][0] - self._waiting[0][0]
        fps = ((len(self._waiting) - 1) / span if span > 0
               else self.FALLBACK_FPS)
        height, width = self._waiting[0][1].shape[:2]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = cv2.VideoWriter(
            str(self.path), cv2.VideoWriter.fourcc(*"mp4v"),
            max(1.0, min(fps, 120.0)), (width, height))
        for _, image in self._waiting:
            self._writer.write(image)
        self._waiting = []

    def close(self) -> None:
        """Finish the file (a recording shorter than the measure too)."""
        if self._writer is None and self._waiting:
            self._open()
        if self._writer is not None:
            self._writer.release()
            self._writer = None


class Camera:
    """The frames of a live camera, with the time they were taken.

    A thread reads the camera as fast as it gives frames and keeps only
    the newest one: when the analysis is slower than the camera, the
    frames in between are dropped, never queued, so the analysis is
    never behind (OpenCV's own buffer setting is ignored by most IP
    cameras). When the camera stops giving frames (a network cut), it is
    opened again, up to ``reconnects`` times in a row.

    Attributes:
        device: A camera number (0 is the first) or a stream address
            (``rtsp://...``, ``http://...``).
        every: Seconds between two analysed frames, at least.
        reconnects: Attempts to open the camera again after it stops.
        delay: Seconds between two attempts.
        name: How to name the camera, without its password.
        record: A video file where every frame read is kept, or
            ``None`` (the default: nothing is recorded).
    """

    RECONNECTS = 5
    DELAY = 2.0
    # Size asked of a camera of this computer (width, height).
    SIZE = (1280, 720)

    def __init__(self, device: int | str = 0, every: float = 0.0,
                 reconnects: int = RECONNECTS, delay: float = DELAY,
                 warn: Callable[[str], None] | None = None,
                 record: Path | None = None) -> None:
        """Remember the camera; it is opened when the frames are read.

        Args:
            device: A camera number or a stream address.
            every: Seconds between two analysed frames, at least.
            reconnects: Attempts to open it again after it stops.
            delay: Seconds between two attempts.
            warn: Told of each cut and reconnection (stderr by default).
            record: Also keep the camera's video in this file.
        """
        self.device = device
        self.record = record
        self.every = every
        self.reconnects = reconnects
        self.delay = delay
        self.name = safe_name(device)
        self._warn = warn or (lambda text: print(text, file=sys.stderr))

    def __iter__(self) -> Iterator[Frame]:
        """Frames until the camera stops for good or the reader stops
        asking.

        Raises:
            RagError: The camera cannot be opened at the start.
        """
        capture = self._open()
        if not capture.isOpened():
            capture.release()
            raise RagError(f"Cannot open the {self.name}")
        latest = _Latest()
        reader = threading.Thread(target=self._read, args=(capture, latest),
                                  daemon=True)
        reader.start()
        last = -float("inf")
        try:
            while True:
                taken = latest.take()
                if taken is None:
                    return
                second, image = taken
                if second - last < self.every:
                    continue
                last = second
                yield Frame(second, _rgb(image))
        finally:
            latest.stopped.set()
            reader.join(timeout=self.delay + 1)

    def _open(self) -> Any:
        """Open the camera; a camera of this computer is asked for
        ``SIZE`` in MJPG.

        OpenCV opens them at 640x480 unless told otherwise, and many
        webcams then crop the sides of the sensor: the image covers less
        than the camera's field of view, and the distances measured from
        the face come out short (seen on 2026-11-18: 40-45 cm read for
        about 70). MJPG keeps 30 frames a second at that size on most USB
        cameras. A camera that cannot gives its nearest size.
        """
        capture = cv2.VideoCapture(self.device)
        if isinstance(self.device, int) and capture.isOpened():
            capture.set(cv2.CAP_PROP_FOURCC,
                        cv2.VideoWriter.fourcc(*"MJPG"))
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.SIZE[0])
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.SIZE[1])
        return capture

    def _read(self, capture: Any, latest: _Latest) -> None:
        """Read frames into ``latest`` until stopped or given up.

        The time of a frame is when it was read, from the start: it goes
        on during a cut, so the time lost is not hidden.
        """
        start = time.monotonic()
        attempts = 0
        recorder = Recorder(self.record) if self.record else None
        try:
            while not latest.stopped.is_set():
                read, image = capture.read()
                if read:
                    attempts = 0
                    second = time.monotonic() - start
                    if recorder is not None:
                        recorder.add(second, image)
                    latest.put(second, image)
                    continue
                capture.release()
                if attempts == self.reconnects:
                    self._warn(f"The {self.name} stopped; giving up after "
                               f"{attempts} attempts.")
                    return
                attempts += 1
                self._warn(f"The {self.name} stopped; reconnecting "
                           f"({attempts}/{self.reconnects})...")
                if latest.stopped.wait(self.delay):
                    return
                capture = self._open()
        finally:
            capture.release()
            if recorder is not None:
                recorder.close()
            latest.finish()
