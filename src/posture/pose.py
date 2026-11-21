"""The skeleton of the person in each frame, with MediaPipe Pose.

MediaPipe's "VIDEO" mode reads one frame at a time with its time and
uses the previous frames to follow the person, which is exactly what a
live camera gives: the same estimator works for a file and a camera.

The model (a ``.task`` file of 6 to 31 MB) is downloaded from Google the
first time and kept in a cache folder; after that everything runs
offline, on the CPU, and no image leaves the computer.
"""
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
import os
from pathlib import Path
import sys
from typing import Any, Literal
import urllib.request

import numpy as np

from ..exceptions import RagError
from .skeleton import Skeleton
from .sources import Frame

Model = Literal["lite", "full", "heavy"]
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/"
             "pose_landmarker/pose_landmarker_{0}/float16/latest/"
             "pose_landmarker_{0}.task")
CACHE = Path.home() / ".cache" / "rag-posture"
# Set to see MediaPipe's own log lines (written by its C++ part, which no
# Python setting silences).
VERBOSE_ENV = "RAG_POSE_VERBOSE"


@contextmanager
def _quiet() -> Iterator[None]:
    """Hide what MediaPipe's C++ part writes to stderr meanwhile.

    It prints warnings about its model ("Sparsity tensors...") that say
    nothing to the person. Errors still arrive as Python exceptions.
    """
    if os.environ.get(VERBOSE_ENV):
        yield
        return
    sys.stderr.flush()
    saved = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(saved, 2)
        os.close(devnull)
        os.close(saved)


def download(url: str, path: Path, name: str) -> Path:
    """A model file, downloaded the first time and then kept.

    Args:
        url: Where to download it from.
        path: Where it is kept.
        name: What it is, for the error.

    Raises:
        RagError: It is not there and cannot be downloaded.
    """
    if path.is_file():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".part")
    try:
        urllib.request.urlretrieve(url, partial)
    except OSError as e:
        partial.unlink(missing_ok=True)
        raise RagError(f"Cannot download the {name} model: "
                       f"{type(e).__name__}") from e
    partial.rename(path)
    return path


def model_path(model: Model = "full", cache: Path = CACHE) -> Path:
    """The pose model file, downloaded the first time.

    Args:
        model: ``lite`` is fastest, ``heavy`` most precise.
        cache: Folder where it is kept.

    Raises:
        RagError: It is not there and cannot be downloaded.
    """
    return download(MODEL_URL.format(model),
                    cache / f"pose_landmarker_{model}.task",
                    f"pose '{model}'")


# Ours (calibrated on 2026-11-18): MediaPipe's model for several people
# hardly sees a second one, and makes the points of each a little less
# steady (C1: 1 of 2 pushes found; C4: none of the pull; C5: a box
# carried taken for a push). So while there is one person the model for
# one is used on the whole image, as calibrated, and the people detector
# (``detector``) looks every ``CHECK_SECONDS`` whether somebody else came
# in. While there are several, the detector finds them in every frame
# and the model measures each one in their own part of the image; back
# to one when there has been only one for ``SINGLE_AFTER`` seconds.
CHECK_SECONDS = 1.0
SINGLE_AFTER = 2.0
# Ours: with one person, when the model finds nobody on the whole image
# the detector is asked, and the person it sees is measured in their part
# of the image. Dark and against a bright window, at one side of a 1080p
# image, the model missed a person standing for 40 s of 2 minutes while
# the detector found them (0.8-0.9); the frames it already found are not
# touched (2026-11-20).
CROP_WHEN_MISSED = True
# Ours: not for someone cut by a side of the picture (coming in or
# leaving): half a person read a trunk bent 129° in C5 (2026-11-20).
EDGE = 0.01
# Ours: each person's part of the image is their box widened this much
# on each side (a share of its size), so hands and feet reaching out
# are inside.
MARGIN = 0.2
# Ours: a person's box is followed to the next frame when its centre is
# at most this far (a share of the image).
FOLLOW = 0.3


@dataclass
class _Slot:
    """A model following one of several people in their part of the
    image."""

    landmarker: Any
    centre: tuple[float, float] | None = None
    second: float = -1.0


class PoseEstimator:
    """Finds the skeletons of the people in each frame, in order.

    The frames must come with increasing times (as a video or a camera
    gives them): the model follows the people from one to the next.
    """

    def __init__(self, model: Model = "full", cache: Path = CACHE,
                 people: int = 1) -> None:
        """Load the model (about a second).

        Args:
            model: ``lite``, ``full`` or ``heavy``.
            cache: Folder where it is kept.
            people: The most people found in a frame. With more than one,
                the people detector looks for them (see
                ``CHECK_SECONDS``): 18 ms a frame with one person, about
                20 ms with one looking for others, and with several
                21 ms plus 18 ms for each.

        Raises:
            RagError: A model cannot be downloaded or loaded.
        """
        self._path = model_path(model, cache)
        self.people = people
        self._single = self._load(self._path)
        self._detector = None
        if people > 1:
            from .detector import PersonDetector
            self._detector = PersonDetector(cache)
        self._slots: list[_Slot] = []
        self._several = False
        self._checked = -CHECK_SECONDS
        self._one_since: float | None = None
        self._last = -1

    @staticmethod
    def _load(path: Path) -> Any:
        """A MediaPipe landmarker for one person."""
        from mediapipe.tasks.python import BaseOptions, vision

        options = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(path)),
            running_mode=vision.RunningMode.VIDEO, num_poses=1)
        try:
            with _quiet():
                return vision.PoseLandmarker.create_from_options(options)
        except (RuntimeError, ValueError) as e:
            raise RagError(f"Cannot load the pose model: {e}") from e

    def find(self, frame: Frame) -> Skeleton | None:
        """The (first) skeleton in a frame, or ``None`` without a
        person."""
        found = self.find_all(frame)
        return found[0] if found else None

    def find_all(self, frame: Frame) -> list[Skeleton]:
        """The skeletons in a frame, in no particular order.

        A frame whose time is not after the previous one is skipped:
        MediaPipe needs strictly increasing times.
        """
        millisecond = int(frame.timestamp * 1000)
        if millisecond <= self._last:
            return []
        self._last = millisecond
        second = frame.timestamp
        if self._detector is None:
            return self._whole(frame, millisecond)
        boxes: list[Any] | None = None
        if not self._several and second - self._checked >= CHECK_SECONDS:
            self._checked = second
            boxes = self._detector.find(frame.image)
            if len(boxes) > 1:
                self._several, self._one_since = True, None
        if not self._several:
            found = self._whole(frame, millisecond)
            if found or not CROP_WHEN_MISSED:
                return found
            # Nobody on the whole image, but the detector may see them
            # (see ``CROP_WHEN_MISSED``): measured in their part of it.
            if boxes is None:
                boxes = self._detector.find(frame.image)
            whole = [box for box, _ in boxes
                     if box[0] > EDGE and box[2] < 1 - EDGE]
            return (self._each(frame, millisecond, whole[:1])
                    if whole else [])
        if boxes is None:
            boxes = self._detector.find(frame.image)
        if len(boxes) > 1:
            self._one_since = None
        elif self._one_since is None:
            self._one_since = second
        elif second - self._one_since >= SINGLE_AFTER:
            self._several = False
            self._checked = second
        return self._each(frame, millisecond,
                          [box for box, _ in boxes[:self.people]])

    def _whole(self, frame: Frame, millisecond: int) -> list[Skeleton]:
        """The one person in the whole image."""
        return self._detect(self._single, frame.image, millisecond,
                            frame.timestamp, (0.0, 0.0, 1.0, 1.0))

    def _each(self, frame: Frame, millisecond: int,
              boxes: list[tuple[float, float, float, float]]
              ) -> list[Skeleton]:
        """Each person in their part of the image, by a model that
        follows them (the one that had the nearest box)."""
        centres = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in boxes]
        while len(self._slots) < len(boxes):
            self._slots.append(_Slot(self._load(self._path)))
        pairs = sorted(
            (float(np.hypot(c[0] - slot.centre[0], c[1] - slot.centre[1])),
             n, k)
            for n, c in enumerate(centres)
            for k, slot in enumerate(self._slots)
            if slot.centre is not None)
        given: dict[int, int] = {}
        for distance, n, k in pairs:
            if (distance <= FOLLOW and n not in given
                    and k not in given.values()):
                given[n] = k
        for n in range(len(boxes)):
            if n not in given:
                # The model idle longest.
                free = min((k for k in range(len(self._slots))
                            if k not in given.values()),
                           key=lambda k: self._slots[k].second)
                given[n] = free
        found = []
        height, width = frame.image.shape[:2]
        for n, k in given.items():
            x0, y0, x1, y1 = boxes[n]
            wide, tall = (x1 - x0) * MARGIN, (y1 - y0) * MARGIN
            left, top = max(0.0, x0 - wide), max(0.0, y0 - tall)
            right, bottom = min(1.0, x1 + wide), min(1.0, y1 + tall)
            crop = frame.image[int(top * height):int(bottom * height) or 1,
                               int(left * width):int(right * width) or 1]
            if crop.shape[0] < 8 or crop.shape[1] < 8:
                continue
            slot = self._slots[k]
            slot.centre, slot.second = centres[n], frame.timestamp
            found += self._detect(slot.landmarker, crop, millisecond,
                                  frame.timestamp,
                                  (left, top, right, bottom))
        return found

    @staticmethod
    def _detect(landmarker: Any, image: Any, millisecond: int,
                second: float, part: tuple[float, float, float, float]
                ) -> list[Skeleton]:
        """The skeleton a landmarker finds in a part of the image
        (``part``: its box in the whole image, from 0 to 1); its places
        in the image are given in the whole image."""
        import mediapipe as mp

        picture = mp.Image(image_format=mp.ImageFormat.SRGB,
                           data=np.ascontiguousarray(image))
        with _quiet():
            found = landmarker.detect_for_video(picture, millisecond)
        left, top, right, bottom = part
        skeletons = []
        for n, world in enumerate(found.pose_world_landmarks):
            points = np.array([[p.x, p.y, p.z] for p in world], dtype=float)
            visibility = np.array([p.visibility or 0.0 for p in world],
                                  dtype=float)
            pixels = None
            if n < len(found.pose_landmarks):
                pixels = np.array([[left + p.x * (right - left),
                                    top + p.y * (bottom - top)]
                                   for p in found.pose_landmarks[n]],
                                  dtype=float)
            skeletons.append(Skeleton(second, points, visibility, pixels))
        return skeletons

    def close(self) -> None:
        """Free the models."""
        with _quiet():
            self._single.close()
            for slot in self._slots:
                slot.landmarker.close()
